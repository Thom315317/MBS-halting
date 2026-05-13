import argparse
import csv
import os
import time

import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader

from .baselines import RelationalGCNClassifier, RelationalGCNHaltingClassifier
from .datasets import VALUES, build_belief_repair_datasets
from .graph import CELL_TYPES, collate_graph_samples
from .model import MBSModel
from .tokenizer import SimpleTokenizer
from .utils import ensure_dir, load_config, resolve_device, save_json, set_seed


# v0.3 release: only the variants used in the published 3-seed campaign are kept.
# - mbs_adaptive_halting          : reference, MBS substrate + ACT-lite halting
# - rgcn_repair_stability         : strong fixed-step baseline (T=8)
# - rgcn_repair_stability_act_forced_t8 : diagnostic baseline (forced T=8 every epoch)
# - rgcn_repair_stability_act_warmup_t8 : diagnostic baseline (3-epoch warmup at T=8 then free ACT)
MBS_VARIANTS = {"mbs_adaptive_halting"}
RGCN_VARIANTS = {
    "rgcn_repair_stability",
    "rgcn_repair_stability_act_forced_t8",
    "rgcn_repair_stability_act_warmup_t8",
    # H6_detached_aux protocol on the RGCN backbone (two-stage partial-freeze
    # with enriched MLP halting controller and anytime aux features). No
    # internal warmup / no forced step — the two stages are configured via
    # controller_only=true / trainable_modules at the YAML level, mirroring
    # the MBS H6_detached_aux configs.
    "rgcn_h6_two_stage",
}
GRAPH_VARIANTS = MBS_VARIANTS | RGCN_VARIANTS
ALL_VARIANTS = GRAPH_VARIANTS


def build_model(variant, config, tokenizer):
    if variant in RGCN_VARIANTS:
        if variant in {"rgcn_repair_stability_act", "rgcn_repair_stability_act_forced_t8", "rgcn_repair_stability_act_warmup_t8", "rgcn_h6_two_stage"}:
            halting = config.get("halting", {}) or {}
            force_terminal_step = 8 if variant == "rgcn_repair_stability_act_forced_t8" else None
            warmup_terminal_step = 8 if variant == "rgcn_repair_stability_act_warmup_t8" else None
            return RelationalGCNHaltingClassifier(
                vocab_size=len(tokenizer.tokens),
                num_values=len(VALUES),
                d_model=config.get("d_state", 96),
                num_cell_types=config.get("num_cell_types", 8),
                num_edge_types=config.get("num_edge_types", 12),
                message_steps=int(halting.get("max_message_steps", config.get("message_steps", 16))),
                dropout=config.get("dropout", 0.1),
                halting_config=halting,
                force_terminal_step=force_terminal_step,
                warmup_terminal_step=warmup_terminal_step,
            )
        return RelationalGCNClassifier(
            vocab_size=len(tokenizer.tokens),
            num_values=len(VALUES),
            d_model=config.get("d_state", 96),
            num_cell_types=config.get("num_cell_types", 8),
            num_edge_types=config.get("num_edge_types", 12),
            message_steps=config.get("message_steps", 8),
            dropout=config.get("dropout", 0.1),
        )
    if variant in MBS_VARIANTS:
        halting = config.get("halting", {}) or {}
        message_steps = int(halting.get("max_message_steps", 16))
        return MBSModel(
            vocab_size=len(tokenizer.tokens),
            num_values=len(VALUES),
            d_state=config.get("d_state", 96),
            num_cell_types=config.get("num_cell_types", 8),
            num_edge_types=config.get("num_edge_types", 12),
            num_operation_modes=config.get("num_operation_modes", 4),
            message_steps=message_steps,
            dropout=config.get("dropout", 0.1),
            use_modes=True,
            use_gate=True,
            adaptive_halting=True,
            halting_config=halting,
        )
    raise ValueError(f"unknown variant {variant}")


def make_loaders(config, tokenizer):
    datasets = build_belief_repair_datasets(config, tokenizer)
    batch_size = int(config.get("batch_size", 16))
    loaders = {}
    for split, dataset in datasets.items():
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            collate_fn=lambda samples: collate_graph_samples(samples, tokenizer),
        )
    return loaders


CLAIM_TYPE_ID = CELL_TYPES["CLAIM"]


def build_coarsened_binary_target(batch, mode):
    """Per-node coarsened binary target with values in {-100, 0, 1}.

    Reads ONLY structural batch fields (cell_type_ids, is_query_claim_node,
    repair_labels mask via `>= 0`, conflict_labels, node_mask). Never the
    answer label or the value-bearing repair targets.

    Returns a long tensor of shape (B, max_nodes) with -100 for ignored nodes.
    """
    device = batch["cell_type_ids"].device
    cell_type_ids = batch["cell_type_ids"]
    node_mask = batch["node_mask"].bool()
    bsz, max_nodes = cell_type_ids.shape
    target = torch.full((bsz, max_nodes), -100, dtype=torch.long, device=device)
    is_claim = (cell_type_ids == CLAIM_TYPE_ID) & node_mask

    if mode == "claim_is_query_relevant":
        is_qc = batch.get("is_query_claim_node")
        if is_qc is None:
            raise RuntimeError(
                "compute_loss: coarsened_target_mode='claim_is_query_relevant' requires "
                "batch['is_query_claim_node']; rebuild the dataset with the patched "
                "mbs/datasets.py + mbs/graph.py."
            )
        is_qc = is_qc.bool().to(device)
        target = torch.where(is_claim & is_qc, torch.ones_like(target), target)
        target = torch.where(is_claim & ~is_qc, torch.zeros_like(target), target)
    elif mode == "binary_repair_supervised_node":
        repair_labels = batch["repair_labels"]
        supervised = repair_labels >= 0
        target = torch.where(node_mask & supervised, torch.ones_like(target), target)
        target = torch.where(node_mask & ~supervised, torch.zeros_like(target), target)
    elif mode == "claim_node_binary":
        target = torch.where(node_mask & is_claim, torch.ones_like(target), target)
        target = torch.where(node_mask & ~is_claim, torch.zeros_like(target), target)
    elif mode == "conflict_binary":
        conflict_labels = batch.get("conflict_labels")
        if conflict_labels is None:
            raise RuntimeError(
                "compute_loss: coarsened_target_mode='conflict_binary' requires "
                "batch['conflict_labels']."
            )
        is_conflict = (conflict_labels.to(device) > 0.5)
        target = torch.where(is_claim & is_conflict, torch.ones_like(target), target)
        target = torch.where(is_claim & ~is_conflict, torch.zeros_like(target), target)
    else:
        raise ValueError(
            f"compute_loss: unknown coarsened_target_mode={mode!r}; valid: "
            "claim_is_query_relevant, binary_repair_supervised_node, claim_node_binary, conflict_binary"
        )
    return target


def build_structural_target(batch, mode):
    """Per-node structural target for repair_loss_mode='structural'.

    Returns (target, num_classes) where:
      - target shape (B, max_nodes), values in {-100, 0, 1, ...}
      - num_classes is 2 for binary modes, 3 for claim_conflict_role.

    Reads ONLY structural batch fields (cell_type_ids, node_mask, is_query_claim_node,
    is_winner_query_claim_node, claim_source_is_trusted, claim_is_rolled_back).
    Never reads the answer label, repair_labels values, or claim text.

    STRICT-FAIL: if a required structural field is missing, raises RuntimeError
    pointing at the missing field. No silent fallback.
    """
    device = batch["cell_type_ids"].device
    cell_type_ids = batch["cell_type_ids"]
    node_mask = batch["node_mask"].bool()
    bsz, max_nodes = cell_type_ids.shape
    is_claim = (cell_type_ids == CLAIM_TYPE_ID) & node_mask

    def _need(field):
        v = batch.get(field)
        if v is None:
            raise RuntimeError(
                f"compute_loss: structural_target_mode={mode!r} requires batch[{field!r}]; "
                "rebuild the dataset with the patched mbs/datasets.py + mbs/graph.py."
            )
        return v.to(dtype=torch.bool, device=device)

    target = torch.full((bsz, max_nodes), -100, dtype=torch.long, device=device)

    if mode == "claim_is_query_relevant":
        is_q = _need("is_query_claim_node")
        target = torch.where(is_claim & is_q, torch.ones_like(target), target)
        target = torch.where(is_claim & ~is_q, torch.zeros_like(target), target)
        return target, 2
    if mode == "claim_should_be_kept":
        is_winner = _need("is_winner_query_claim_node")
        target = torch.where(is_claim & is_winner, torch.ones_like(target), target)
        target = torch.where(is_claim & ~is_winner, torch.zeros_like(target), target)
        return target, 2
    if mode == "claim_conflict_role":
        is_q = _need("is_query_claim_node")
        is_winner = _need("is_winner_query_claim_node")
        # 0 = irrelevant CLAIM, 1 = loser-query CLAIM, 2 = winner-query CLAIM
        cls_irrelevant = is_claim & ~is_q
        cls_loser = is_claim & is_q & ~is_winner
        cls_winner = is_claim & is_winner
        target = torch.where(cls_irrelevant, torch.zeros_like(target), target)
        target = torch.where(cls_loser, torch.ones_like(target), target)
        target = torch.where(cls_winner, torch.full_like(target, 2), target)
        return target, 3
    if mode == "source_is_trusted":
        st = _need("claim_source_is_trusted")
        target = torch.where(is_claim & st, torch.ones_like(target), target)
        target = torch.where(is_claim & ~st, torch.zeros_like(target), target)
        return target, 2
    if mode == "claim_is_rolled_back":
        rb = _need("claim_is_rolled_back")
        target = torch.where(is_claim & rb, torch.ones_like(target), target)
        target = torch.where(is_claim & ~rb, torch.zeros_like(target), target)
        return target, 2
    raise ValueError(
        f"compute_loss: unknown structural_target_mode={mode!r}; valid: "
        "claim_is_query_relevant, claim_should_be_kept, claim_conflict_role, "
        "source_is_trusted, claim_is_rolled_back"
    )


def aggregate_value_logits(scores, claim_value_ids, mask, num_values):
    """Differentiable aggregation: value_logits[v] = logsumexp(scores_i for valid i with value_id_i == v).

    scores: (B, max_nodes) real-valued. Promoted to fp32 internally for numerical stability
            (the surrounding autocast context typically casts to fp16 which overflows on
            sentinel values like -1e9).
    claim_value_ids: (B, max_nodes) long, with -1 on positions not to use.
    mask: (B, max_nodes) bool, True where the position should contribute.
    Returns:
      value_logits: (B, num_values) fp32 (very negative for empty value buckets).
    """
    scores_fp32 = scores.float()
    bsz = scores_fp32.size(0)
    device = scores_fp32.device
    # Guard: any sample with an empty agg_mask would silently produce NaN/-inf
    # logits (no valid scores to log-sum-exp). Surface the bug explicitly.
    mask_per_sample = mask.sum(dim=1)
    empty = (mask_per_sample == 0)
    if bool(empty.any().item()):
        bad = torch.nonzero(empty, as_tuple=False).flatten().tolist()
        raise RuntimeError(
            f"aggregate_value_logits: empty agg_mask for sample(s) {bad}. "
            "All scores would be masked out → NaN/-inf logits. Check the "
            "dataset (claim_value_ids / is_query_claim_node) or the aggregator's "
            "agg_mask construction."
        )
    very_negative = torch.full_like(scores_fp32, -1e9)
    masked_scores = torch.where(mask, scores_fp32, very_negative)
    batch_max, _ = masked_scores.max(dim=1, keepdim=True)
    batch_max = batch_max.clamp(min=-1e9)  # floor in case all positions are invalid
    shifted = scores_fp32 - batch_max
    exp_shifted = torch.exp(shifted) * mask.float()  # zero out invalid contributions
    safe_value_ids = claim_value_ids.clamp(min=0)  # invalid (-1) clamped to 0; their contribution is 0 due to mask
    accum = torch.zeros(bsz, num_values, device=device, dtype=scores_fp32.dtype)
    accum.scatter_add_(1, safe_value_ids, exp_shifted)
    value_logits = torch.log(accum.clamp(min=1e-30)) + batch_max
    return value_logits


def compute_loss(outputs, batch, config, variant):
    diagnostics = outputs.get("diagnostics", {})

    answer_readout_mode = str(config.get("answer_readout_mode", "standard")).lower()
    latent_query_only = bool(config.get("latent_selector_query_only", True))

    if answer_readout_mode == "latent_claim_selector":
        if "repair_logits" not in outputs:
            raise RuntimeError(
                "compute_loss: answer_readout_mode='latent_claim_selector' requires "
                "outputs['repair_logits'] (per-node scores). Variant must be a GRAPH_VARIANTS member."
            )
        if "claim_value_ids" not in batch:
            raise RuntimeError(
                "compute_loss: answer_readout_mode='latent_claim_selector' requires "
                "batch['claim_value_ids']; rebuild the dataset with the patched mbs/datasets.py + mbs/graph.py."
            )
        # build per-node CLAIM mask
        cell_type_ids = batch["cell_type_ids"]
        node_mask = batch["node_mask"].bool()
        is_claim = (cell_type_ids == CLAIM_TYPE_ID) & node_mask
        if latent_query_only:
            is_qc = batch.get("is_query_claim_node")
            if is_qc is None:
                raise RuntimeError(
                    "compute_loss: latent_selector_query_only=True requires batch['is_query_claim_node']."
                )
            agg_mask = is_claim & is_qc.bool()
        else:
            agg_mask = is_claim
        # value-class id per CLAIM node
        claim_value_ids = batch["claim_value_ids"].long()
        agg_mask = agg_mask & (claim_value_ids >= 0)
        target = batch["target_value_id"]
        # Step-aware path: when MBSModel exposes claim_scores_per_step
        # (adaptive halting) compute Σ_t halt_w_t · CE_t so that the halting
        # controller actually receives a gradient on the answer signal.
        # Backward-compatible fallback for any model variant that does not
        # expose per-step scores: use repair_logits[..., 0] on the final state.
        scores_per_step = outputs.get("claim_scores_per_step")
        halt_weights = outputs.get("halt_weights")
        if scores_per_step is not None and halt_weights is not None and len(scores_per_step) > 0:
            T = len(scores_per_step)
            bsz = halt_weights.size(0)
            # In controller_only mode, detach CE_t before the weighted sum so
            # the only gradient flowing back is on halt_weights → halting_controller.
            controller_only = bool(config.get("controller_only", False))
            # Streaming accumulators to keep memory low at large T.
            weighted_ce = torch.zeros(bsz, device=halt_weights.device, dtype=torch.float32)
            weighted_value_logits = torch.zeros(
                bsz, len(VALUES), device=halt_weights.device, dtype=torch.float32,
            )
            per_step_ce_means = []
            per_step_acc_means = []
            per_step_correct = []  # (B,) bool per step — for expected_halted_acc
            for t in range(T):
                value_logits_t = aggregate_value_logits(
                    scores_per_step[t], claim_value_ids, agg_mask, len(VALUES),
                )
                ce_t = F.cross_entropy(value_logits_t, target, reduction="none")
                ce_t_for_loss = ce_t.detach() if controller_only else ce_t
                w_t = halt_weights[:, t].float()
                weighted_ce = weighted_ce + w_t * ce_t_for_loss
                weighted_value_logits = weighted_value_logits + w_t.unsqueeze(-1) * value_logits_t
                with torch.no_grad():
                    per_step_ce_means.append(ce_t.mean())
                    correct_t = (value_logits_t.argmax(dim=-1) == target).float()  # (B,)
                    per_step_acc_means.append(correct_t.mean())
                    per_step_correct.append(correct_t)
            answer_loss = weighted_ce.mean()
            outputs["logits"] = weighted_value_logits
            # Distinct halted accuracy metrics (diagnostic only — do not enter
            # the loss). See Task C / CODE_AUDIT_FINAL_REPORT for the
            # distinction between mixture-logits, expected-halted, and
            # chosen-step accuracies.
            with torch.no_grad():
                mixture_pred = weighted_value_logits.argmax(dim=-1)
                mixture_logits_acc = (mixture_pred == target).float().mean()
                # (B, T) correct flags  →  Σ_t w_t · correct_t  →  mean over B
                correct_stack = torch.stack(per_step_correct, dim=1).float()  # (B, T)
                expected_halted_acc = (halt_weights.float() * correct_stack).sum(dim=1).mean()
                # per-sample chosen step = argmax_t halt_weight
                chosen_step_idx = halt_weights.float().argmax(dim=1)  # (B,) 0-indexed
                chosen_step_mean = (chosen_step_idx.float() + 1.0).mean()  # 1-indexed
                chosen_correct = correct_stack.gather(1, chosen_step_idx.unsqueeze(1)).squeeze(1)
                chosen_step_acc = chosen_correct.mean()
                expected_halted_nll = weighted_ce.detach().mean()
                outputs.setdefault("diagnostics", {}).update({
                    "mixture_logits_acc_mean": mixture_logits_acc,
                    "expected_halted_acc_mean": expected_halted_acc,
                    "chosen_step_acc_mean": chosen_step_acc,
                    "chosen_step_mean": chosen_step_mean,
                    "expected_halted_nll_mean": expected_halted_nll,
                })
            scores = scores_per_step[-1]  # diagnostics use the final-step scores
            outputs.setdefault("diagnostics", {}).update({
                "per_step_ce_mean": torch.stack(per_step_ce_means),
                "per_step_accuracy_mean": torch.stack(per_step_acc_means),
            })
        else:
            # Legacy single-step path (final state of repair_head, channel 0)
            scores = outputs["repair_logits"][..., 0]
            value_logits = aggregate_value_logits(scores, claim_value_ids, agg_mask, len(VALUES))
            outputs["logits"] = value_logits
            answer_loss = F.cross_entropy(value_logits, target)
        # diagnostics: selector entropy / max prob / counts (fp32 to avoid Half overflow)
        with torch.no_grad():
            scores_fp32_diag = scores.float()
            probs = torch.softmax(scores_fp32_diag.masked_fill(~agg_mask, -1e9), dim=1)
            ent = -(probs * (probs.clamp(min=1e-30)).log()).sum(dim=1)
            outputs.setdefault("diagnostics", {}).update({
                "latent_selector_entropy_mean": ent.mean(),
                "latent_selector_max_prob_mean": probs.max(dim=1).values.mean(),
                "latent_num_claims_mean": agg_mask.sum(dim=1).float().mean(),
                "latent_num_claim_values_present_mean": (
                    torch.tensor(
                        [(claim_value_ids[b][agg_mask[b]].unique().numel() if agg_mask[b].any() else 0) for b in range(scores.size(0))],
                        dtype=torch.float, device=scores.device,
                    ).mean()
                ),
            })
    else:
        answer_loss = F.cross_entropy(outputs["logits"], batch["target_value_id"])

    loss = config.get("lambda_answer", 1.0) * answer_loss
    parts = {
        "answer_loss": answer_loss.detach(),
        "conflict_loss": torch.tensor(0.0, device=answer_loss.device),
        "repair_loss": torch.tensor(0.0, device=answer_loss.device),
        "stability_loss": diagnostics.get("stability_loss", torch.tensor(0.0, device=answer_loss.device)),
        "ponder_loss": torch.tensor(0.0, device=answer_loss.device),
        "margin_loss": torch.tensor(0.0, device=answer_loss.device),
    }

    # Optional margin loss on the final answer-bucket logits.
    # Off by default (lambda_margin=0). When on, penalises samples whose
    # gold-vs-max-other margin is below `margin_target`. Computed on
    # outputs["logits"] which holds value_logits in latent_claim_selector mode
    # and the standard answer-head logits otherwise. Does NOT modify the
    # aggregator.
    lambda_margin = float(config.get("lambda_margin", 0.0))
    if lambda_margin > 0.0:
        margin_target = float(config.get("margin_target", 0.1))
        final_logits = outputs["logits"].float()
        target = batch["target_value_id"]
        bsz = final_logits.size(0)
        gold_logit = final_logits.gather(1, target.unsqueeze(1)).squeeze(1)
        other_mask = torch.ones_like(final_logits, dtype=torch.bool)
        other_mask[torch.arange(bsz, device=final_logits.device), target] = False
        masked_others = final_logits.masked_fill(~other_mask, float("-inf"))
        max_other = masked_others.max(dim=1).values
        gold_margin = gold_logit - max_other  # (B,)
        margin_violation = (margin_target - gold_margin).clamp(min=0).mean()
        loss = loss + lambda_margin * margin_violation
        parts["margin_loss"] = margin_violation.detach()
        outputs.setdefault("diagnostics", {}).update({
            "margin_violation_mean": margin_violation.detach(),
            "gold_margin_mean": gold_margin.detach().mean(),
            "gold_margin_median": gold_margin.detach().median(),
        })

    if uses_conflict_loss(variant, config) and outputs.get("conflict_logits") is not None and batch.get("conflict_mask") is not None:
        mask = batch["conflict_mask"]
        if mask.any():
            conflict_loss = F.binary_cross_entropy_with_logits(outputs["conflict_logits"][mask], batch["conflict_labels"][mask])
            loss = loss + config.get("lambda_conflict", 0.2) * conflict_loss
            parts["conflict_loss"] = conflict_loss.detach()

    repair_loss_mode = str(config.get("repair_loss_mode", "value_original")).lower()
    if repair_loss_mode in {"value_original", "masked_value"} and uses_repair_loss(variant, config) and outputs.get("repair_logits") is not None:
        repair_labels = batch["repair_labels"]
        base_valid = repair_labels >= 0
        mode = str(config.get("repair_mask_mode", "original")).lower()
        allow_missing = bool(config.get("allow_missing_query_claim_mask", False))

        # build per-graph QUERY-node flag and query-claim flag for diagnostics + masking
        device = repair_labels.device
        bsz, max_nodes = repair_labels.shape
        batch_arange = torch.arange(bsz, device=device)
        query_node_idx = batch.get("query_node_idx")
        query_node_flag = torch.zeros_like(base_valid)
        if query_node_idx is not None:
            query_node_flag[batch_arange, query_node_idx] = True
        query_claim_flag = batch.get("is_query_claim_node")
        if query_claim_flag is not None:
            query_claim_flag = query_claim_flag.to(dtype=torch.bool, device=device)

        # apply mask per mode
        mask = base_valid.clone()
        if mode == "original":
            pass
        elif mode == "no_query":
            mask = mask & ~query_node_flag
        elif mode == "no_query_and_query_claims":
            if query_claim_flag is None:
                if not allow_missing:
                    raise RuntimeError(
                        "compute_loss: repair_mask_mode='no_query_and_query_claims' requires "
                        "batch['is_query_claim_node']; this batch does not provide it. "
                        "Either regenerate the dataset (mbs/datasets.py + mbs/graph.py now emit "
                        "this field) or set allow_missing_query_claim_mask=true to fall back to "
                        "no_query (this fallback is opt-in and explicit)."
                    )
                mask = mask & ~query_node_flag
            else:
                mask = mask & ~query_node_flag & ~query_claim_flag
        else:
            raise ValueError(
                f"compute_loss: unknown repair_mask_mode={mode!r} "
                "(valid: 'original', 'no_query', 'no_query_and_query_claims')"
            )

        # diagnostics (always emit, regardless of mode)
        valid_before = base_valid.sum().float()
        valid_after = mask.sum().float()
        query_masked = (base_valid & query_node_flag).sum().float() if mode != "original" else torch.tensor(0.0, device=device)
        if mode == "no_query_and_query_claims" and query_claim_flag is not None:
            query_claims_masked = (base_valid & query_claim_flag).sum().float()
        else:
            query_claims_masked = torch.tensor(0.0, device=device)
        other_kept = mask.sum().float()  # post-mask total = "other" supervised left
        fraction_kept = valid_after / valid_before.clamp(min=1.0)
        repair_diag = {
            "repair_labels_valid_before_mask": valid_before,
            "repair_labels_valid_after_mask": valid_after,
            "repair_labels_fraction_kept": fraction_kept,
            "repair_query_nodes_masked": query_masked,
            "repair_query_claim_nodes_masked": query_claims_masked,
            "repair_other_nodes_kept": other_kept,
        }
        outputs.setdefault("diagnostics", {}).update(repair_diag)

        if mask.any():
            repair_loss = F.cross_entropy(outputs["repair_logits"][mask], repair_labels[mask])
            loss = loss + config.get("lambda_repair", 0.5) * repair_loss
            parts["repair_loss"] = repair_loss.detach()
            # repair_loss_before_mask: cheap, train-only diagnostic (recomputed without mask)
            with torch.no_grad():
                if base_valid.any():
                    repair_loss_before = F.cross_entropy(
                        outputs["repair_logits"][base_valid], repair_labels[base_valid]
                    )
                    outputs["diagnostics"]["repair_loss_before_mask"] = repair_loss_before.detach()
                outputs["diagnostics"]["repair_loss_after_mask"] = repair_loss.detach()

    if repair_loss_mode == "coarsened_binary" and uses_repair_loss(variant, config) and outputs.get("repair_logits") is not None:
        coarsened_target_mode = str(config.get("coarsened_target_mode", "claim_is_query_relevant")).lower()
        target = build_coarsened_binary_target(batch, coarsened_target_mode)
        valid = target >= 0
        # diagnostics
        n_pos = (target == 1).sum().float()
        n_neg = (target == 0).sum().float()
        n_supervised = valid.sum().float()
        outputs.setdefault("diagnostics", {}).update({
            "coarsened_positive_count": n_pos,
            "coarsened_negative_count": n_neg,
            "coarsened_supervised_count": n_supervised,
            "coarsened_positive_fraction": n_pos / n_supervised.clamp(min=1.0),
        })
        if valid.any():
            # reuse channel 0 of repair_logits as the per-node binary logit (no new params)
            binary_logit = outputs["repair_logits"][..., 0]
            repair_loss_coarsened = F.binary_cross_entropy_with_logits(
                binary_logit[valid], target[valid].float()
            )
            loss = loss + config.get("lambda_repair", 0.5) * repair_loss_coarsened
            parts["repair_loss"] = repair_loss_coarsened.detach()
            outputs["diagnostics"]["repair_loss_coarsened"] = repair_loss_coarsened.detach()
    elif repair_loss_mode == "structural" and uses_repair_loss(variant, config) and outputs.get("repair_logits") is not None:
        structural_target_mode = str(config.get("structural_target_mode", "claim_should_be_kept")).lower()
        target, num_classes = build_structural_target(batch, structural_target_mode)
        valid = target >= 0
        # diagnostics
        class_counts = {f"structural_class_{c}_count": (target == c).sum().float() for c in range(num_classes)}
        n_supervised = valid.sum().float()
        n_pos = (target >= 1).sum().float()  # any non-zero class as "positive"
        n_neg = (target == 0).sum().float()
        outputs.setdefault("diagnostics", {}).update({
            "structural_supervised_count": n_supervised,
            "structural_positive_count": n_pos,
            "structural_negative_count": n_neg,
            "structural_positive_fraction": n_pos / n_supervised.clamp(min=1.0),
            **class_counts,
            "structural_num_classes": torch.tensor(float(num_classes), device=target.device),
        })
        if valid.any():
            if num_classes == 2:
                # reuse channel 0 of repair_logits as the binary logit
                logit = outputs["repair_logits"][..., 0]
                structural_loss = F.binary_cross_entropy_with_logits(
                    logit[valid], target[valid].float()
                )
            else:
                # reuse channels [0..num_classes) for multi-class CE
                logits = outputs["repair_logits"][..., :num_classes]
                structural_loss = F.cross_entropy(
                    logits[valid], target[valid]
                )
            loss = loss + config.get("lambda_repair", 0.5) * structural_loss
            parts["repair_loss"] = structural_loss.detach()
            outputs["diagnostics"]["repair_loss_structural"] = structural_loss.detach()
    elif repair_loss_mode == "none":
        pass  # explicit no-op

    if uses_stability_loss(variant, config) and "stability_loss" in diagnostics:
        loss = loss + config.get("lambda_stability", 0.01) * diagnostics["stability_loss"]

    if variant in {"mbs_adaptive_halting", "rgcn_repair_stability_act", "rgcn_repair_stability_act_warmup_t8", "rgcn_h6_two_stage"} and "expected_steps_mean" in diagnostics:
        ponder_signal = diagnostics.get("ponder_active_signal")
        ponder_active = True if ponder_signal is None else float(ponder_signal.item() if torch.is_tensor(ponder_signal) else ponder_signal) >= 0.5
        if ponder_active:
            lambda_ponder = float((config.get("halting", {}) or {}).get("lambda_ponder", config.get("lambda_ponder", 0.001)))
            ponder_loss = lambda_ponder * diagnostics["expected_steps_mean"]
            loss = loss + ponder_loss
            parts["ponder_loss"] = ponder_loss.detach()

    if "mode_entropy" in diagnostics:
        loss = loss - config.get("lambda_mode_entropy", 0.001) * diagnostics["mode_entropy"]

    return loss, parts


def uses_conflict_loss(variant, config):
    return float(config.get("lambda_conflict", 0.2)) != 0.0 and variant in MBS_VARIANTS


def uses_repair_loss(variant, config):
    if float(config.get("lambda_repair", 0.5)) == 0.0:
        return False
    return variant in GRAPH_VARIANTS


def uses_stability_loss(variant, config):
    if float(config.get("lambda_stability", 0.01)) == 0.0:
        return False
    return variant in GRAPH_VARIANTS


@torch.no_grad()
def evaluate(model, loader, device, config, variant, message_steps=None):
    model.eval()
    total = 0
    correct = 0
    total_loss = 0.0
    metric_sums = {}
    metric_counts = {}
    for batch in loader:
        batch = move_batch(batch, device)
        outputs = model(batch, message_steps=message_steps)
        loss, parts = compute_loss(outputs, batch, config, variant)
        preds = outputs["logits"].argmax(dim=-1)
        correct += (preds == batch["target_value_id"]).sum().item()
        total += batch["target_value_id"].numel()
        total_loss += loss.item() * batch["target_value_id"].numel()
        for key, value in {**parts, **outputs.get("diagnostics", {})}.items():
            for metric_key, metric_value in flatten_metric(key, value):
                metric_sums[metric_key] = metric_sums.get(metric_key, 0.0) + metric_value
                metric_counts[metric_key] = metric_counts.get(metric_key, 0) + 1
    metrics = {key: metric_sums[key] / metric_counts[key] for key in metric_sums}
    metrics["acc"] = correct / max(total, 1)
    metrics["loss"] = total_loss / max(total, 1)
    return metrics


def _validate_config(config, variant):
    """Static guard-rails for the v2 / latent_claim_selector regime.

    Raises ValueError on any inconsistent combination so we never burn GPU
    time on a config that would silently mix incompatible signals.

    Rules:
      1. answer_readout_mode='latent_claim_selector' is incompatible with any
         repair_loss_mode != 'none' or any non-zero lambda_repair (the latent
         path takes over the answer signal, mixing with repair_label leakage
         is exactly what audit_v1 was designed to prevent).
      2. tasks 'depth_controlled_latent_halting_probe_v2' and
         'depth_controlled_latent_halting_probe_v2_small' need
         num_edge_types >= 15 because they emit edge_type_id 14
         (MORE_RELIABLE_THAN_BACKWARD).
      3. configs for the same v2/v2_small tasks must set lambda_conflict=0.0
         and lambda_mode_entropy=0.0 (these losses act on auxiliary signals
         that v2 does not provide and would inject noise / regularisation
         that we audited as harmful in earlier rounds).
    """
    answer_readout = str(config.get("answer_readout_mode", "standard")).lower()
    repair_mode = str(config.get("repair_loss_mode", "value_original")).lower()
    lambda_repair = float(config.get("lambda_repair", 0.5))
    if answer_readout == "latent_claim_selector":
        if repair_mode != "none":
            raise ValueError(
                f"latent_claim_selector requires repair_loss_mode='none' "
                f"(got {repair_mode!r}). The latent path replaces the answer "
                "supervision; mixing with repair_label-bearing losses is leakage."
            )
        if abs(lambda_repair) > 1e-12:
            raise ValueError(
                f"latent_claim_selector requires lambda_repair == 0.0 "
                f"(got {lambda_repair}). See audit_v1 / repair_loss_leakage_audit."
            )
    task = str(config.get("task", "")).lower()
    if task in ("depth_controlled_latent_halting_probe_v2",
                "depth_controlled_latent_halting_probe_v2_small",
                "depth_controlled_latent_halting_probe_v3",
                "depth_controlled_latent_halting_probe_v3_1"):
        n_et = int(config.get("num_edge_types", 0))
        if n_et < 15:
            raise ValueError(
                f"task={task!r} requires num_edge_types >= 15 "
                f"(got {n_et}). The generator emits edge_type ids 13/14 "
                "(MORE_RELIABLE_THAN_FORWARD/_BACKWARD)."
            )
        for key, allowed in (("lambda_conflict", 0.0),
                             ("lambda_mode_entropy", 0.0)):
            val = float(config.get(key, allowed))
            if abs(val - allowed) > 1e-12:
                raise ValueError(
                    f"task={task!r} requires {key}={allowed} (got {val}). "
                    "These auxiliary losses act on signals v2 does not provide "
                    "and were observed to harm calibration in earlier rounds."
                )


def train_one(config, variant, output_dir, checkpoint_dir):
    _validate_config(config, variant)
    set_seed(int(config.get("seed", 1)))
    tokenizer = SimpleTokenizer()
    loaders = make_loaders(config, tokenizer)
    device = resolve_device(config.get("device", "cuda_if_available"))
    model = build_model(variant, config, tokenizer).to(device)

    # Optional warm-start from a previous checkpoint. strict=False because we
    # may load into a slightly extended architecture (e.g. claim_selector_head
    # added in the step-aware patch); missing keys init from the new model's
    # default init.
    init_ckpt_path = config.get("init_from_checkpoint")
    if init_ckpt_path:
        payload = torch.load(init_ckpt_path, map_location=device, weights_only=False)
        state_dict = payload.get("model_state", payload.get("state_dict"))
        if state_dict is None:
            raise ValueError(f"checkpoint {init_ckpt_path} has no model_state")
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"[init_from_checkpoint] loaded from {init_ckpt_path}; "
              f"missing_keys={len(missing)} unexpected_keys={len(unexpected)}")

    # Optional: partial-freeze training. Two equivalent ways to specify it:
    #   - controller_only=true        → trainable = ["halting_controller"]
    #   - trainable_modules: [a, b]   → trainable = the listed prefixes
    # If both are set, trainable_modules wins. A parameter whose qualified name
    # does NOT start with any listed prefix is frozen (requires_grad=False).
    trainable_prefixes = list(config.get("trainable_modules") or [])
    if not trainable_prefixes and bool(config.get("controller_only", False)):
        trainable_prefixes = ["halting_controller"]
    if trainable_prefixes:
        for name, p in model.named_parameters():
            if not any(name.startswith(prefix) for prefix in trainable_prefixes):
                p.requires_grad_(False)

    parameter_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if trainable_prefixes:
        total = sum(p.numel() for p in model.parameters())
        print(f"[partial_freeze] frozen {total - parameter_count}/{total} params; "
              f"training {parameter_count} params on modules: {trainable_prefixes}.")

    # Optional: per-module learning rates via param groups. If
    # lr_halting_controller or lr_claim_selector_head are set, build groups
    # by name prefix; otherwise fall back to single-group lr.
    base_lr = float(config.get("lr", 1e-3))
    lr_hc = float(config.get("lr_halting_controller", base_lr))
    lr_csh = float(config.get("lr_claim_selector_head", base_lr))
    weight_decay = float(config.get("weight_decay", 0.01))
    if ("lr_halting_controller" in config) or ("lr_claim_selector_head" in config):
        hc_params, csh_params, other_params = [], [], []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if name.startswith("halting_controller"):
                hc_params.append(p)
            elif name.startswith("claim_selector_head"):
                csh_params.append(p)
            else:
                other_params.append(p)
        groups = []
        if hc_params:
            groups.append({"params": hc_params, "lr": lr_hc, "name": "halting_controller"})
        if csh_params:
            groups.append({"params": csh_params, "lr": lr_csh, "name": "claim_selector_head"})
        if other_params:
            groups.append({"params": other_params, "lr": base_lr, "name": "other"})
        optimizer = AdamW(groups, weight_decay=weight_decay)
        print(f"[lr_groups] halting_controller lr={lr_hc} (n={len(hc_params)}), "
              f"claim_selector_head lr={lr_csh} (n={len(csh_params)}), "
              f"other lr={base_lr} (n={len(other_params)}).")
    else:
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = AdamW(trainable, lr=base_lr, weight_decay=weight_decay)

    # Optional: policy distillation teacher. The teacher is a frozen model
    # loaded from policy_teacher_checkpoint, used to compute a per-batch
    # policy distillation loss against the student. The teacher is NOT given
    # the gold label — distillation operates on halting policy outputs only.
    teacher_model = None
    policy_distill_type = config.get("policy_distill_type", None)
    beta_policy = float(config.get("beta_policy", 0.0))
    teacher_ckpt_path = config.get("policy_teacher_checkpoint")
    if policy_distill_type and beta_policy > 0.0 and teacher_ckpt_path:
        teacher_model = build_model(variant, config, tokenizer).to(device)
        t_payload = torch.load(teacher_ckpt_path, map_location=device, weights_only=False)
        t_state = t_payload.get("model_state", t_payload.get("state_dict"))
        if t_state is None:
            raise ValueError(f"teacher checkpoint {teacher_ckpt_path} has no model_state")
        t_missing, t_unexpected = teacher_model.load_state_dict(t_state, strict=False)
        for p in teacher_model.parameters():
            p.requires_grad_(False)
        teacher_model.eval()
        print(f"[policy_teacher] loaded from {teacher_ckpt_path}; "
              f"missing_keys={len(t_missing)} unexpected_keys={len(t_unexpected)} "
              f"distill_type={policy_distill_type} beta_policy={beta_policy}")
    scaler = GradScaler("cuda", enabled=(device.type == "cuda"))
    ensure_dir(output_dir)
    ensure_dir(checkpoint_dir)
    selected_row = None
    selected_checkpoint_path = os.path.join(checkpoint_dir, f"{variant}_best.pt")
    final_checkpoint_path = os.path.join(checkpoint_dir, f"{variant}_final.pt")
    best_ood_diagnostic_checkpoint_path = os.path.join(checkpoint_dir, f"{variant}_best_ood_diagnostic.pt")
    best_ood_value = float("-inf")
    best_ood_epoch_seen = None
    history = []

    warmup_epochs = int((config.get("halting", {}) or {}).get("warmup_epochs", 3))
    max_epochs = int(config.get("max_epochs", 5))
    epoch_metrics_csv = os.path.join(output_dir, f"{variant}_epoch_metrics.csv")
    csv_header_written = False
    run_start = time.time()
    repair_mask_mode_str = str(config.get("repair_mask_mode", "original"))
    repair_loss_mode_str = str(config.get("repair_loss_mode", "value_original"))
    coarsened_mode_str = str(config.get("coarsened_target_mode", "claim_is_query_relevant"))
    structural_mode_str = str(config.get("structural_target_mode", "claim_should_be_kept"))
    answer_readout_str = str(config.get("answer_readout_mode", "standard"))
    latent_qonly = bool(config.get("latent_selector_query_only", True))
    print(
        f"[run] variant={variant} seed={config.get('seed')} device={device.type}"
        f"{':' + str(device.index) if device.index is not None else ''} "
        f"batch_size={int(config.get('batch_size', 16))} parameter_count={parameter_count} "
        f"max_epochs={max_epochs} repair_loss_mode={repair_loss_mode_str} "
        f"repair_mask_mode={repair_mask_mode_str} coarsened_target_mode={coarsened_mode_str} "
        f"structural_target_mode={structural_mode_str} "
        f"answer_readout_mode={answer_readout_str} latent_selector_query_only={latent_qonly}",
        flush=True,
    )
    for epoch in range(1, max_epochs + 1):
        epoch_start = time.time()
        if variant == "rgcn_repair_stability_act_warmup_t8" and hasattr(model, "set_warmup_active"):
            is_warmup = epoch <= warmup_epochs
            model.set_warmup_active(is_warmup)
            print(
                f"[warmup_t8] epoch {epoch}: warmup="
                f"{'ACTIVE (force step 8, ponder=0)' if is_warmup else 'INACTIVE (free ACT)'}",
                flush=True,
            )
        model.train()
        train_metrics = []
        for batch in loaders["train"]:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=device.type, enabled=(device.type == "cuda")):
                outputs = model(batch)
                loss, parts = compute_loss(outputs, batch, config, variant)
                # Policy distillation: pull halting policy back toward the
                # frozen teacher (stage 1 H4 checkpoint, typically). No gold
                # label dependency — operates on halt_weights / expected_steps
                # only. Implements expected_step_mse (default) or halt_weight_kl.
                policy_distill_loss_val = 0.0
                if teacher_model is not None:
                    with torch.no_grad():
                        teacher_outputs = teacher_model(batch)
                    if policy_distill_type == "expected_step_mse":
                        s = outputs.get("expected_steps")
                        t = teacher_outputs.get("expected_steps")
                        if s is None or t is None:
                            raise RuntimeError(
                                "expected_step_mse distill requires 'expected_steps' in outputs."
                            )
                        distill_loss = F.mse_loss(s.float(), t.float())
                    elif policy_distill_type == "halt_weight_kl":
                        s = outputs.get("halt_weights")
                        t = teacher_outputs.get("halt_weights")
                        if s is None or t is None:
                            raise RuntimeError(
                                "halt_weight_kl distill requires 'halt_weights' in outputs."
                            )
                        s = s.float().clamp_min(1e-9)
                        t = t.float().clamp_min(1e-9)
                        # KL(s || t) per sample, mean over batch. halt_weights
                        # already sum to ~1 over steps by construction.
                        distill_loss = (s * (torch.log(s) - torch.log(t))).sum(dim=-1).mean()
                    else:
                        raise ValueError(
                            f"unknown policy_distill_type: {policy_distill_type!r}"
                        )
                    loss = loss + float(beta_policy) * distill_loss
                    policy_distill_loss_val = float(distill_loss.detach().item())
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            # Per-module grad norms BEFORE clipping (diagnostic only). Useful
            # for spotting silent stoppers (e.g. halting_controller getting no
            # gradient in latent_claim_selector mode pre-step-aware-fix).
            def _module_grad_norm(module):
                if module is None:
                    return 0.0
                grads = [p.grad for p in module.parameters() if p.grad is not None]
                if not grads:
                    return 0.0
                return float(torch.cat([g.flatten() for g in grads]).norm().item())
            csh_grad_norm = _module_grad_norm(getattr(model, "claim_selector_head", None))
            hc_grad_norm = _module_grad_norm(getattr(model, "halting_controller", None))
            ah_grad_norm = _module_grad_norm(getattr(model, "answer_head", None))
            rh_grad_norm = _module_grad_norm(getattr(model, "repair_head", None))
            # capture pre-clipping grad norm for diagnostic logging.
            total_grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(config.get("grad_clip", 1.0))
            )
            scaler.step(optimizer)
            scaler.update()
            with torch.no_grad():
                preds = outputs["logits"].argmax(dim=-1)
                train_metrics.append({
                    "acc": (preds == batch["target_value_id"]).float().mean().item(),
                    "loss": loss.item(),
                    "grad_norm": float(total_grad_norm.item() if torch.is_tensor(total_grad_norm) else total_grad_norm),
                    "claim_selector_head_grad_norm": csh_grad_norm,
                    "halting_controller_grad_norm": hc_grad_norm,
                    "answer_head_grad_norm": ah_grad_norm,
                    "repair_head_grad_norm": rh_grad_norm,
                    "policy_distill_loss": policy_distill_loss_val,
                })

        val = evaluate(model, loaders["val"], device, config, variant)
        ood = {split: evaluate(model, loaders[split], device, config, variant) for split in ["ood_entity", "ood_conflict", "ood_rule", "ood_mixed"]}
        train_acc = sum(item["acc"] for item in train_metrics) / max(len(train_metrics), 1)
        train_loss = sum(item["loss"] for item in train_metrics) / max(len(train_metrics), 1)
        train_grad_norm_mean = (
            sum(item.get("grad_norm", 0.0) for item in train_metrics) / max(len(train_metrics), 1)
            if train_metrics else 0.0
        )
        train_csh_grad_norm_mean = (
            sum(item.get("claim_selector_head_grad_norm", 0.0) for item in train_metrics) / max(len(train_metrics), 1)
            if train_metrics else 0.0
        )
        train_hc_grad_norm_mean = (
            sum(item.get("halting_controller_grad_norm", 0.0) for item in train_metrics) / max(len(train_metrics), 1)
            if train_metrics else 0.0
        )
        train_ah_grad_norm_mean = (
            sum(item.get("answer_head_grad_norm", 0.0) for item in train_metrics) / max(len(train_metrics), 1)
            if train_metrics else 0.0
        )
        train_rh_grad_norm_mean = (
            sum(item.get("repair_head_grad_norm", 0.0) for item in train_metrics) / max(len(train_metrics), 1)
            if train_metrics else 0.0
        )
        train_policy_distill_loss_mean = (
            sum(item.get("policy_distill_loss", 0.0) for item in train_metrics) / max(len(train_metrics), 1)
            if train_metrics else 0.0
        )
        row = {
            "epoch": epoch,
            "train_acc": train_acc,
            "train_loss": train_loss,
            "train_grad_norm_mean": train_grad_norm_mean,
            "train_claim_selector_head_grad_norm_mean": train_csh_grad_norm_mean,
            "train_halting_controller_grad_norm_mean": train_hc_grad_norm_mean,
            "train_answer_head_grad_norm_mean": train_ah_grad_norm_mean,
            "train_repair_head_grad_norm_mean": train_rh_grad_norm_mean,
            "train_policy_distill_loss_mean": train_policy_distill_loss_mean,
            "beta_policy": float(beta_policy),
            "policy_distill_type": policy_distill_type or "",
            "lr_halting_controller": lr_hc,
            "lr_claim_selector_head": lr_csh,
            "val_acc": val["acc"],
            "val_loss": val["loss"],
            "ood_entity_acc": ood["ood_entity"]["acc"],
            "ood_conflict_acc": ood["ood_conflict"]["acc"],
            "ood_rule_acc": ood["ood_rule"]["acc"],
            "ood_mixed_acc": ood["ood_mixed"]["acc"],
        }
        for key in [
            "answer_loss",
            "margin_loss",
            "margin_violation_mean",
            "gold_margin_mean",
            "gold_margin_median",
            "conflict_loss",
            "repair_loss",
            "stability_loss",
            "energy_mean",
            "mode_entropy",
            "update_scale_mean",
            "update_scale_std",
            "update_norm_mean",
            "state_norm_mean",
            "mode_PROPAGATE_mean",
            "mode_STABILIZE_mean",
            "mode_REPAIR_mean",
            "mode_RESOLVE_CONFLICT_mean",
            "expected_steps_mean",
            "final_step_mass_mean",
            "halt_weight_sum_mean",
            "halt_weight_sum_std",
            "ponder_loss",
            "repair_labels_valid_before_mask",
            "repair_labels_valid_after_mask",
            "repair_labels_fraction_kept",
            "repair_query_nodes_masked",
            "repair_query_claim_nodes_masked",
            "repair_other_nodes_kept",
            "repair_loss_before_mask",
            "repair_loss_after_mask",
            "coarsened_positive_count",
            "coarsened_negative_count",
            "coarsened_supervised_count",
            "coarsened_positive_fraction",
            "repair_loss_coarsened",
            "structural_supervised_count",
            "structural_positive_count",
            "structural_negative_count",
            "structural_positive_fraction",
            "structural_num_classes",
            "structural_class_0_count",
            "structural_class_1_count",
            "structural_class_2_count",
            "repair_loss_structural",
            "latent_selector_entropy_mean",
            "latent_selector_max_prob_mean",
            "latent_num_claims_mean",
            "latent_num_claim_values_present_mean",
        ]:
            if key in val:
                row[f"val_{key}"] = val[key]
        row["repair_mask_mode"] = str(config.get("repair_mask_mode", "original"))
        row["repair_loss_mode"] = str(config.get("repair_loss_mode", "value_original"))
        row["coarsened_target_mode"] = str(config.get("coarsened_target_mode", "claim_is_query_relevant"))
        row["structural_target_mode"] = str(config.get("structural_target_mode", "claim_should_be_kept"))
        row["answer_readout_mode"] = str(config.get("answer_readout_mode", "standard"))
        row["latent_selector_query_only"] = bool(config.get("latent_selector_query_only", True))
        for key in [
            "expected_steps_mean",
            "final_step_mass_mean",
            "halt_weight_sum_mean",
            "halt_weight_sum_std",
            "ponder_loss",
        ]:
            if key in ood["ood_mixed"]:
                row[f"ood_mixed_{key}"] = ood["ood_mixed"][key]
        for key, value in val.items():
            if key.startswith("halt_prob_mean_by_step_") or key.startswith("halt_weight_mean_by_step_"):
                row[f"val_{key}"] = value
        epoch_duration = time.time() - epoch_start
        elapsed = time.time() - run_start
        eta = epoch_duration * (max_epochs - epoch)
        row["epoch_duration_seconds"] = epoch_duration
        row["elapsed_seconds"] = elapsed
        row["eta_remaining_seconds"] = eta
        if device.type == "cuda":
            row["gpu_mem_allocated_mb"] = torch.cuda.memory_allocated(device) / (1024 * 1024)
            row["gpu_mem_reserved_mb"] = torch.cuda.memory_reserved(device) / (1024 * 1024)
        history.append(row)
        write_epoch_metrics_csv(epoch_metrics_csv, row, write_header=not csv_header_written)
        csv_header_written = True
        print_epoch_block(variant, config, epoch, max_epochs, row, val, ood, device, parameter_count, history)
        if variant == "mbs_adaptive_halting":
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "variant": variant,
                    "config": config,
                    "tokenizer": tokenizer.state_dict(),
                    "epoch": epoch,
                },
                os.path.join(checkpoint_dir, f"{variant}_epoch_{epoch}.pt"),
            )
        if checkpoint_row_is_better(row, selected_row):
            selected_row = row
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "variant": variant,
                    "config": config,
                    "tokenizer": tokenizer.state_dict(),
                    "epoch": epoch,
                },
                selected_checkpoint_path,
            )
        # Always overwrite the "final" checkpoint at the end of every epoch so
        # that it ultimately points to the last epoch trained. Keeps best.pt
        # untouched (selected by the official policy) and exposes a separate
        # final.pt for runs where val_acc saturates and the tie-break is the
        # de facto OOD selector.
        torch.save(
            {
                "model_state": model.state_dict(),
                "variant": variant,
                "config": config,
                "tokenizer": tokenizer.state_dict(),
                "epoch": epoch,
            },
            final_checkpoint_path,
        )
        # Diagnostic-only: track the best OOD-mixed checkpoint seen so far.
        # NEVER use this for model selection; it is leakage to choose by OOD.
        # We expose the path so external scripts can audit it explicitly.
        current_ood = float(row.get("ood_mixed_acc", float("-inf")))
        if current_ood > best_ood_value:
            best_ood_value = current_ood
            best_ood_epoch_seen = epoch
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "variant": variant,
                    "config": config,
                    "tokenizer": tokenizer.state_dict(),
                    "epoch": epoch,
                    "diagnostic_only": True,
                    "diagnostic_note": (
                        "best OOD-mixed checkpoint seen during training. "
                        "DIAGNOSTIC ONLY — do not use for model selection."
                    ),
                },
                best_ood_diagnostic_checkpoint_path,
            )
        metadata = build_training_metadata(history, selected_row, selected_checkpoint_path, variant)
        # Expose the additional checkpoint paths in the train metadata so
        # downstream scripts (audit, eval, aggregate) can find them without
        # guessing.
        metadata["checkpoint_paths"] = {
            "selected": str(selected_checkpoint_path),
            "final": str(final_checkpoint_path),
            "best_ood_diagnostic": str(best_ood_diagnostic_checkpoint_path),
            "best_ood_diagnostic_epoch": best_ood_epoch_seen,
            "best_ood_diagnostic_value": best_ood_value if best_ood_value > float("-inf") else None,
            "note": (
                "best.pt = official model selection (val_acc max, tie val_loss min, "
                "tie earlier_epoch). final.pt = last epoch trained. "
                "best_ood_diagnostic.pt = diagnostic only, NOT for selection."
            ),
        }
        for warning in metadata["warnings"]:
            print(warning)
        save_json(
            os.path.join(output_dir, f"{variant}_train_results.json"),
            {
                "variant": variant,
                "config": config,
                "history": history,
                "parameter_count": parameter_count,
                **metadata,
            },
        )
    return history


def move_batch(batch, device):
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def flatten_metric(key, value):
    if torch.is_tensor(value):
        value = value.detach().float()
        if value.numel() == 1:
            yield key, float(value.item())
            return
        flat = value.reshape(-1)
        for idx, item in enumerate(flat, start=1):
            yield f"{key}_{idx:02d}", float(item.item())
        return
    if isinstance(value, (list, tuple)):
        for idx, item in enumerate(value, start=1):
            yield f"{key}_{idx:02d}", float(item)
        return
    yield key, float(value)


def checkpoint_row_is_better(candidate, selected, eps=1e-12):
    if selected is None:
        return True
    candidate_val = float(candidate["val_acc"])
    selected_val = float(selected["val_acc"])
    if candidate_val > selected_val + eps:
        return True
    if abs(candidate_val - selected_val) <= eps:
        candidate_loss = float(candidate["val_loss"])
        selected_loss = float(selected["val_loss"])
        if candidate_loss < selected_loss - eps:
            return True
    return False


def build_training_metadata(history, selected_row, selected_checkpoint_path, variant):
    selected_row = selected_row or history[-1]
    best_ood_row = max(history, key=lambda row: float(row.get("ood_mixed_acc", float("-inf"))))
    selected_ood = float(selected_row.get("ood_mixed_acc", 0.0))
    best_ood = float(best_ood_row.get("ood_mixed_acc", 0.0))
    regression = best_ood - selected_ood
    checkpoint_policy = {
        "primary_metric": "val_acc",
        "mode": "max",
        "tie_breaker": ["val_loss:min", "earlier_epoch"],
        "selected_epoch": int(selected_row["epoch"]),
        "selected_metric_value": float(selected_row["val_acc"]),
        "selected_checkpoint_path": str(selected_checkpoint_path),
        "selected_epoch_val_acc": float(selected_row["val_acc"]),
        "selected_epoch_val_loss": float(selected_row["val_loss"]),
        "selected_epoch_ood_mixed_acc": selected_ood,
    }
    if variant == "mbs_adaptive_halting":
        checkpoint_policy.update(
            {
                "selected_epoch_expected_steps_val": selected_row.get("val_expected_steps_mean"),
                "selected_epoch_final_step_mass_val": selected_row.get("val_final_step_mass_mean"),
                "selected_epoch_expected_steps_ood_mixed": selected_row.get("ood_mixed_expected_steps_mean"),
                "selected_epoch_final_step_mass_ood_mixed": selected_row.get("ood_mixed_final_step_mass_mean"),
            }
        )
    best_ood_mixed_epoch = {
        "epoch": int(best_ood_row["epoch"]),
        "ood_mixed_acc": best_ood,
        "val_acc": float(best_ood_row["val_acc"]),
        "val_loss": float(best_ood_row["val_loss"]),
    }
    expected_steps = best_ood_row.get("ood_mixed_expected_steps_mean", best_ood_row.get("val_expected_steps_mean"))
    final_step_mass = best_ood_row.get("ood_mixed_final_step_mass_mean", best_ood_row.get("val_final_step_mass_mean"))
    if expected_steps is not None:
        best_ood_mixed_epoch["expected_steps_mean"] = expected_steps
    if final_step_mass is not None:
        best_ood_mixed_epoch["final_step_mass_mean"] = final_step_mass
    warning = None
    if regression > 0.03:
        warning = "WARNING: selected checkpoint is substantially worse than best OOD epoch."
    return {
        "checkpoint_policy": checkpoint_policy,
        "best_ood_mixed_epoch": best_ood_mixed_epoch,
        "ood_regression_from_best_epoch": regression,
        "warnings": [warning] if warning else [],
    }


def write_epoch_metrics_csv(path, row, write_header=False):
    flat = {key: value for key, value in row.items() if not isinstance(value, (list, tuple, dict))}
    fieldnames = sorted(flat.keys())
    mode = "w" if write_header else "a"
    with open(path, mode, encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow({key: flat.get(key) for key in fieldnames})


def _format_seconds(seconds):
    seconds = int(round(float(seconds)))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def _format_step_table(row, prefix, max_steps=16):
    keys = [f"{prefix}{idx:02d}" for idx in range(1, max_steps + 1) if f"{prefix}{idx:02d}" in row]
    if not keys:
        return None
    items = [f"{key.split('_')[-1]}:{float(row[key]):.3f}" for key in keys]
    return "  ".join(items)


def print_epoch_block(variant, config, epoch, max_epochs, row, val, ood_dict, device, parameter_count, history):
    head_lines = [
        "=" * 72,
        f"Variant: {variant} | seed={config.get('seed')} | epoch {epoch}/{max_epochs}"
        + (
            f" | mode={'WARMUP_T8' if epoch <= int((config.get('halting', {}) or {}).get('warmup_epochs', 3)) else 'FREE_ACT'}"
            if variant == "rgcn_repair_stability_act_warmup_t8"
            else ""
        ),
        f"device={device.type}{':' + str(device.index) if device.index is not None else ''} "
        f"batch_size={int(config.get('batch_size', 16))} parameter_count={parameter_count}"
        + (
            f" | gpu_alloc={row.get('gpu_mem_allocated_mb', 0):.0f}MB "
            f"gpu_reserved={row.get('gpu_mem_reserved_mb', 0):.0f}MB"
            if device.type == "cuda"
            else ""
        ),
        f"duration={_format_seconds(row['epoch_duration_seconds'])}  "
        f"elapsed={_format_seconds(row['elapsed_seconds'])}  "
        f"ETA_remaining={_format_seconds(row['eta_remaining_seconds'])}",
    ]
    perf = (
        f"Performance:\n"
        f"  train_acc={row['train_acc']:.4f}  val_acc={row['val_acc']:.4f}\n"
        f"  ood_entity={row['ood_entity_acc']:.4f}  ood_conflict={row['ood_conflict_acc']:.4f}  "
        f"ood_rule={row['ood_rule_acc']:.4f}  ood_mixed={row['ood_mixed_acc']:.4f}"
    )
    losses = (
        f"Losses (val):\n"
        f"  total={row['val_loss']:.4f}  "
        f"answer={row.get('val_answer_loss', float('nan')):.4f}  "
        f"conflict={row.get('val_conflict_loss', float('nan')):.4f}  "
        f"repair={row.get('val_repair_loss', float('nan')):.4f}  "
        f"stability={row.get('val_stability_loss', float('nan')):.4f}  "
        f"ponder={row.get('val_ponder_loss', float('nan')):.6f}"
    )
    halting_lines = []
    if "val_expected_steps_mean" in row:
        halting_lines.append(
            f"Halting:\n"
            f"  E[steps] val={row['val_expected_steps_mean']:.4f}  "
            f"ood_mixed={row.get('ood_mixed_expected_steps_mean', float('nan')):.4f}\n"
            f"  final_mass val={row.get('val_final_step_mass_mean', float('nan')):.4f}  "
            f"ood_mixed={row.get('ood_mixed_final_step_mass_mean', float('nan')):.4f}\n"
            f"  halt_weight_sum: mean={row.get('val_halt_weight_sum_mean', float('nan')):.4f} "
            f"std={row.get('val_halt_weight_sum_std', float('nan')):.4f}"
        )
        prob_table = _format_step_table(row, "val_halt_prob_mean_by_step_")
        weight_table = _format_step_table(row, "val_halt_weight_mean_by_step_")
        if prob_table:
            halting_lines.append("  halt_prob_mean_by_step (val):  " + prob_table)
        if weight_table:
            halting_lines.append("  halt_weight_mean_by_step (val):  " + weight_table)
    dynamics = (
        f"Dynamics:\n"
        f"  update_norm_mean={row.get('val_update_norm_mean', float('nan')):.4f}  "
        f"state_norm_mean={row.get('val_state_norm_mean', float('nan')):.4f}  "
        f"update_scale_mean={row.get('val_update_scale_mean', float('nan')):.4f} "
        f"(std={row.get('val_update_scale_std', float('nan')):.4f})"
    )
    repair_lines = []
    if "val_repair_labels_valid_before_mask" in row:
        repair_lines.append(
            f"Repair (value mode) masking:\n"
            f"  repair_loss_mode={config.get('repair_loss_mode', 'value_original')}  "
            f"repair_mask_mode={config.get('repair_mask_mode', 'original')}\n"
            f"  valid_before={int(row.get('val_repair_labels_valid_before_mask', 0))}  "
            f"valid_after={int(row.get('val_repair_labels_valid_after_mask', 0))}  "
            f"fraction_kept={row.get('val_repair_labels_fraction_kept', float('nan')):.4f}\n"
            f"  query_masked={int(row.get('val_repair_query_nodes_masked', 0))}  "
            f"query_claims_masked={int(row.get('val_repair_query_claim_nodes_masked', 0))}  "
            f"other_kept={int(row.get('val_repair_other_nodes_kept', 0))}\n"
            f"  repair_loss_before={row.get('val_repair_loss_before_mask', float('nan')):.4f}  "
            f"repair_loss_after={row.get('val_repair_loss_after_mask', float('nan')):.4f}"
        )
    if "val_coarsened_supervised_count" in row:
        repair_lines.append(
            f"Repair (coarsened) :\n"
            f"  repair_loss_mode={config.get('repair_loss_mode', 'value_original')}  "
            f"coarsened_target_mode={config.get('coarsened_target_mode', 'claim_is_query_relevant')}\n"
            f"  pos={int(row.get('val_coarsened_positive_count', 0))}  "
            f"neg={int(row.get('val_coarsened_negative_count', 0))}  "
            f"supervised={int(row.get('val_coarsened_supervised_count', 0))}  "
            f"pos_fraction={row.get('val_coarsened_positive_fraction', float('nan')):.4f}\n"
            f"  repair_loss_coarsened={row.get('val_repair_loss_coarsened', float('nan')):.4f}"
        )
    if "val_latent_selector_entropy_mean" in row:
        repair_lines.append(
            f"Latent claim selector :\n"
            f"  answer_readout_mode={config.get('answer_readout_mode', 'standard')}  "
            f"latent_selector_query_only={config.get('latent_selector_query_only', True)}\n"
            f"  selector_entropy={row.get('val_latent_selector_entropy_mean', float('nan')):.4f}  "
            f"selector_max_prob={row.get('val_latent_selector_max_prob_mean', float('nan')):.4f}\n"
            f"  num_claims_per_sample={row.get('val_latent_num_claims_mean', float('nan')):.4f}  "
            f"num_values_present={row.get('val_latent_num_claim_values_present_mean', float('nan')):.4f}"
        )
    if "val_structural_supervised_count" in row:
        ncls = int(row.get("val_structural_num_classes", 2))
        class_counts = "  ".join(
            f"class{c}={int(row.get(f'val_structural_class_{c}_count', 0))}"
            for c in range(ncls)
        )
        repair_lines.append(
            f"Repair (structural) :\n"
            f"  repair_loss_mode={config.get('repair_loss_mode', 'value_original')}  "
            f"structural_target_mode={config.get('structural_target_mode', 'claim_should_be_kept')}  "
            f"num_classes={ncls}\n"
            f"  {class_counts}  "
            f"supervised={int(row.get('val_structural_supervised_count', 0))}  "
            f"pos_fraction={row.get('val_structural_positive_fraction', float('nan')):.4f}\n"
            f"  repair_loss_structural={row.get('val_repair_loss_structural', float('nan')):.4f}"
        )
    best_ood_row = max(history, key=lambda r: float(r.get("ood_mixed_acc", float("-inf"))))
    selected = max(history, key=lambda r: float(r.get("val_acc", float("-inf"))))
    regression = float(best_ood_row.get("ood_mixed_acc", 0.0)) - float(selected.get("ood_mixed_acc", 0.0))
    ckpt_lines = [
        f"Checkpoint policy (running):",
        f"  selected_epoch={selected['epoch']} (val_acc={selected['val_acc']:.4f}, "
        f"ood_mixed={selected['ood_mixed_acc']:.4f})",
        f"  best_ood_epoch={best_ood_row['epoch']} (ood_mixed={best_ood_row['ood_mixed_acc']:.4f})",
        f"  ood_regression_from_best={regression:.4f}"
        + ("  WARNING: selected ckpt worse than best OOD epoch" if regression > 0.03 else ""),
    ]
    block = "\n".join(head_lines + [perf, losses] + halting_lines + [dynamics] + repair_lines + ckpt_lines + ["=" * 72])
    print(block, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--variant",
        required=True,
        choices=sorted(ALL_VARIANTS),
    )
    parser.add_argument("--output-dir", default="results/belief_repair_debug")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    args = parser.parse_args()
    config = load_config(args.config)
    train_one(config, args.variant, args.output_dir, args.checkpoint_dir)


if __name__ == "__main__":
    main()
