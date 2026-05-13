import math

import torch
from torch import nn


def _aggregate_value_logits(scores, claim_value_ids, mask, num_values):
    """Module-local copy of aggregate_value_logits — kept here (and in
    mbs/model.py) so this module has no dependency on train.py."""
    scores_fp32 = scores.float()
    bsz = scores_fp32.size(0)
    device = scores_fp32.device
    mask_per_sample = mask.sum(dim=1)
    empty = (mask_per_sample == 0)
    if bool(empty.any().item()):
        bad = torch.nonzero(empty, as_tuple=False).flatten().tolist()
        raise RuntimeError(
            f"_aggregate_value_logits: empty agg_mask for sample(s) {bad}."
        )
    very_negative = torch.full_like(scores_fp32, -1e9)
    masked_scores = torch.where(mask, scores_fp32, very_negative)
    batch_max, _ = masked_scores.max(dim=1, keepdim=True)
    batch_max = batch_max.clamp(min=-1e9)
    shifted = scores_fp32 - batch_max
    exp_shifted = torch.exp(shifted) * mask.float()
    safe_value_ids = claim_value_ids.clamp(min=0)
    accum = torch.zeros(bsz, num_values, device=device, dtype=scores_fp32.dtype)
    accum.scatter_add_(1, safe_value_ids, exp_shifted)
    value_logits = torch.log(accum.clamp(min=1e-30)) + batch_max
    return value_logits


def compute_halt_aux_features(
    claim_scores_t,
    agg_mask,
    claim_value_ids,
    num_values,
    step_number,
    max_steps,
    prev_value_margin,
):
    """Compute the 5 anytime aux features for the enriched halting controller.

    Order (matches ENRICHED_HALT_AUX_DIM order below):
      0: normalized_step      = t / max_steps
      1: selector_entropy_t   = entropy of softmax over query-CLAIM scores
      2: selector_max_prob_t  = max softmax prob over query-CLAIM scores
      3: value_margin_t       = top1 - top2 of value_logits at step t
      4: delta_value_margin_t = value_margin_t - value_margin_{t-1}

    No gold-label dependency. agg_mask is built from cell_type_ids,
    is_query_claim_node, claim_value_ids only — all structural fields.
    Returns (aux: (B, 5), new_prev_value_margin: (B,)).

    This helper is shared by MBSModel and RelationalGCNHaltingClassifier so
    both substrates feed exactly the same aux features to the enriched
    halting controller.
    """
    bsz = claim_scores_t.size(0)
    device = claim_scores_t.device
    dtype = claim_scores_t.dtype

    scores_fp32 = claim_scores_t.float()
    very_negative = torch.full_like(scores_fp32, -1e9)
    masked_scores = torch.where(agg_mask, scores_fp32, very_negative)
    batch_max, _ = masked_scores.max(dim=1, keepdim=True)
    batch_max = batch_max.clamp(min=-1e9)
    shifted = scores_fp32 - batch_max
    exp_shifted = torch.exp(shifted) * agg_mask.float()
    Z = exp_shifted.sum(dim=1, keepdim=True).clamp_min(1e-30)
    probs = exp_shifted / Z
    log_probs = torch.log(probs.clamp_min(1e-30))
    ent = -(probs * log_probs).sum(dim=1)
    max_prob = probs.max(dim=1).values

    value_logits_t = _aggregate_value_logits(
        claim_scores_t, claim_value_ids, agg_mask, num_values
    )
    topk = torch.topk(value_logits_t, k=min(2, num_values), dim=1).values
    if topk.size(1) >= 2:
        value_margin_t = topk[:, 0] - topk[:, 1]
    else:
        value_margin_t = topk[:, 0]
    delta_margin = value_margin_t - prev_value_margin

    normalized_step = torch.full(
        (bsz,), float(step_number) / float(max_steps), device=device, dtype=dtype
    )

    aux = torch.stack(
        [
            normalized_step.to(dtype),
            ent.to(dtype),
            max_prob.to(dtype),
            value_margin_t.to(dtype),
            delta_margin.to(dtype),
        ],
        dim=1,
    )
    return aux, value_margin_t.detach()


class AdaptiveHaltingController(nn.Module):
    def __init__(self, d_state, init_halt_prob=0.05):
        super().__init__()
        init_halt_prob = min(max(float(init_halt_prob), 1e-4), 1.0 - 1e-4)
        self.halt_head = nn.Linear(d_state, 1)
        nn.init.zeros_(self.halt_head.weight)
        nn.init.constant_(self.halt_head.bias, math.log(init_halt_prob / (1.0 - init_halt_prob)))

    def forward(self, h, query_node_idx, node_mask=None):
        batch_idx = torch.arange(h.size(0), device=h.device)
        query_state = h[batch_idx, query_node_idx]
        return torch.sigmoid(self.halt_head(query_state)).squeeze(-1)


# Number of auxiliary anytime features fed to EnrichedAdaptiveHaltingController.
# Order:
#   0: normalized_step      = t / max_steps
#   1: selector_entropy_t   = entropy of softmax over query-CLAIM scores
#   2: selector_max_prob_t  = max softmax prob over query-CLAIM scores
#   3: value_margin_t       = top1 - top2 of value_logits at step t
#   4: delta_value_margin_t = value_margin_t - value_margin_{t-1} (0 at t=1)
ENRICHED_HALT_AUX_DIM = 5


class EnrichedAdaptiveHaltingController(nn.Module):
    """MLP-based halting controller that takes the query-node state plus
    a small set of *anytime* features at the current step:

      input  = concat(query_state (d_state), aux_features (5))
      output = halt_prob in [0, 1]

    Aux features are derived from the per-step latent_claim_selector
    aggregator and never reference the gold label. See ENRICHED_HALT_AUX_DIM
    above for the feature order.

    The final layer is initialised so that, with zeroed weights, the bias
    corresponds to `init_halt_prob` after sigmoid — same logic as
    AdaptiveHaltingController.
    """

    def __init__(self, d_state, init_halt_prob=0.05, hidden=(128, 64)):
        super().__init__()
        self.d_state = int(d_state)
        self.aux_dim = ENRICHED_HALT_AUX_DIM
        h1, h2 = hidden
        self.mlp = nn.Sequential(
            nn.Linear(self.d_state + self.aux_dim, h1),
            nn.GELU(),
            nn.Linear(h1, h2),
            nn.GELU(),
            nn.Linear(h2, 1),
        )
        # Init last layer: weight zero, bias = logit(init_halt_prob).
        # The hidden layers' default init is fine; only the final weights
        # need to vanish so that the *initial* halt_prob is determined by
        # the bias alone.
        init_halt_prob = min(max(float(init_halt_prob), 1e-4), 1.0 - 1e-4)
        with torch.no_grad():
            self.mlp[-1].weight.zero_()
            self.mlp[-1].bias.fill_(math.log(init_halt_prob / (1.0 - init_halt_prob)))

    def forward(self, query_state, aux_features):
        # query_state: (B, d_state); aux_features: (B, ENRICHED_HALT_AUX_DIM)
        x = torch.cat([query_state, aux_features], dim=-1)
        return torch.sigmoid(self.mlp(x).squeeze(-1))
