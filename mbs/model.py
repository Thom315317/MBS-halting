import torch
from torch import nn

from .graph import CELL_TYPES
from .halting import (
    AdaptiveHaltingController,
    EnrichedAdaptiveHaltingController,
    ENRICHED_HALT_AUX_DIM,
    compute_halt_aux_features,
    _aggregate_value_logits,
)
from .substrate import MorphogeneticBeliefSubstrate


class MBSModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        num_values,
        d_state=96,
        num_cell_types=8,
        num_edge_types=12,
        num_operation_modes=4,
        message_steps=8,
        dropout=0.1,
        use_modes=True,
        use_gate=True,
        adaptive_halting=False,
        halting_config=None,
    ):
        super().__init__()
        self.message_steps = message_steps
        self.adaptive_halting = adaptive_halting
        halting_config = halting_config or {}
        self.halting_min_steps = int(halting_config.get("min_message_steps", 4))
        self.halting_max_steps = int(halting_config.get("max_message_steps", message_steps))
        self.enriched_halting = bool(halting_config.get("enriched", False))
        # When True (default), the aux features fed to the enriched halting
        # controller are detached before entering it: the halting loss CANNOT
        # back-propagate into the claim_selector_head through the aux features
        # (it still flows via the step-aware answer CE on claim_selector_head).
        # See Task B / CODE_AUDIT_FINAL_REPORT — closes a subtle gradient leak
        # that may have driven H6/H7/H8 co-adaptation drift.
        self.detach_aux_features_from_selector = bool(
            halting_config.get("detach_aux_features_from_selector", True)
        )
        self.token_embedding = nn.Embedding(vocab_size, d_state, padding_idx=0)
        self.node_type_embedding = nn.Embedding(num_cell_types, d_state)
        self.input_norm = nn.LayerNorm(d_state)
        self.substrate = MorphogeneticBeliefSubstrate(
            d_state=d_state,
            num_cell_types=num_cell_types,
            num_edge_types=num_edge_types,
            num_operation_modes=num_operation_modes,
            dropout=dropout,
            use_modes=use_modes,
            use_gate=use_gate,
        )
        self.answer_head = nn.Sequential(nn.LayerNorm(d_state), nn.Linear(d_state, num_values))
        self.conflict_head = nn.Linear(d_state, 1)
        self.repair_head = nn.Linear(d_state, num_values)
        # Dedicated head for the latent claim selector. Used by step-aware
        # latent_claim_selector mode (compute_loss reads
        # outputs["claim_scores_per_step"]). The legacy single-step path in
        # compute_loss falls back to repair_head channel 0 when this head's
        # outputs are not available — preserves backward compatibility for
        # checkpoints saved before this head existed.
        self.claim_selector_head = nn.Linear(d_state, 1)
        if adaptive_halting:
            if self.enriched_halting:
                self.halting_controller = EnrichedAdaptiveHaltingController(
                    d_state=d_state,
                    init_halt_prob=halting_config.get("init_halt_prob", 0.05),
                    hidden=tuple(halting_config.get("enriched_hidden", (128, 64))),
                )
            else:
                self.halting_controller = AdaptiveHaltingController(
                    d_state=d_state,
                    init_halt_prob=halting_config.get("init_halt_prob", 0.05),
                )
        else:
            self.halting_controller = None

    def forward(self, batch, message_steps=None):
        steps = self.message_steps if message_steps is None else int(message_steps)
        cell_type_ids = batch["cell_type_ids"]
        token_ids = batch["cell_token_ids"]
        h = self.token_embedding(token_ids) + self.node_type_embedding(cell_type_ids)
        h = self.input_norm(h)
        batch_idx = torch.arange(h.size(0), device=h.device)
        query_embedding = h[batch_idx, batch["query_node_idx"]]
        if self.adaptive_halting:
            return self._forward_adaptive_halting(batch, h, query_embedding, steps)
        h, diagnostics, mode_logits = self.substrate(
            h,
            cell_type_ids,
            batch["edge_index"],
            batch["edge_type_ids"],
            batch["node_mask"],
            batch["edge_mask"],
            query_embedding,
            message_steps=steps,
        )
        query_state = h[batch_idx, batch["query_node_idx"]]
        logits = self.answer_head(query_state)
        conflict_logits = self.conflict_head(h).squeeze(-1)
        repair_logits = self.repair_head(h)
        return {
            "logits": logits,
            "diagnostics": diagnostics,
            "conflict_logits": conflict_logits,
            "repair_logits": repair_logits,
            "mode_logits": mode_logits,
        }

    def _forward_adaptive_halting(self, batch, h, query_embedding, max_steps):
        cell_type_ids = batch["cell_type_ids"]
        batch_size = h.size(0)
        batch_idx = torch.arange(batch_size, device=h.device)
        query_node_idx = batch["query_node_idx"]
        num_values = self.answer_head[-1].out_features
        min_steps = max(1, min(int(self.halting_min_steps), int(max_steps)))

        weighted_logits = h.new_zeros(batch_size, num_values)
        remaining_mass = h.new_ones(batch_size)
        expected_steps = h.new_zeros(batch_size)
        halt_probs = []
        halt_weights = []
        step_diagnostics = []
        # Step-aware claim selector scores (B, max_nodes) per step. Consumed
        # by compute_loss in latent_claim_selector mode to enable adaptive
        # halting on the latent answer (Σ_t halt_w_t · CE_t).
        claim_scores_per_step = []
        mode_logits = None

        # Aux feature plumbing (only used when enriched_halting=True; we
        # compute the agg_mask once since it is constant per batch).
        if self.enriched_halting:
            node_mask_b = batch["node_mask"].bool()
            is_claim = (cell_type_ids == CELL_TYPES["CLAIM"]) & node_mask_b
            is_qc = batch["is_query_claim_node"].bool()
            cvi = batch["claim_value_ids"].long()
            agg_mask = is_claim & is_qc & (cvi >= 0)
            prev_value_margin = h.new_zeros(batch_size)
            T_max_for_norm = max(int(max_steps), 1)
        else:
            agg_mask = None
            cvi = None
            prev_value_margin = None
            T_max_for_norm = None

        for step_idx in range(int(max_steps)):
            step_number = step_idx + 1
            h, diagnostics, mode_logits = self.substrate(
                h,
                cell_type_ids,
                batch["edge_index"],
                batch["edge_type_ids"],
                batch["node_mask"],
                batch["edge_mask"],
                query_embedding,
                message_steps=1,
            )
            step_diagnostics.append(diagnostics)
            query_state = h[batch_idx, query_node_idx]
            step_logits = self.answer_head(query_state)
            # Step-aware claim selector score on every node — used by latent mode
            # AND by the enriched halting controller's aux features.
            claim_scores_t = self.claim_selector_head(h).squeeze(-1)
            if self.enriched_halting:
                aux, prev_value_margin = compute_halt_aux_features(
                    claim_scores_t, agg_mask, cvi, num_values,
                    step_number, T_max_for_norm, prev_value_margin,
                )
                # Optionally detach aux features so the halting loss CANNOT
                # back-propagate into claim_selector_head via the aux signal.
                # Default (True) blocks the leak that may have driven H6/H7/H8
                # co-adaptation drift; False = legacy behavior (for ablations).
                aux_in = aux.detach() if self.detach_aux_features_from_selector else aux
                halt_prob = self.halting_controller(query_state, aux_in)
            else:
                halt_prob = self.halting_controller(h, query_node_idx, batch["node_mask"])
            if step_number < min_steps:
                weight = torch.zeros_like(remaining_mass)
            elif step_number == int(max_steps):
                weight = remaining_mass
            else:
                weight = halt_prob * remaining_mass
            weighted_logits = weighted_logits + weight.unsqueeze(-1) * step_logits
            expected_steps = expected_steps + weight * float(step_number)
            remaining_mass = (remaining_mass - weight).clamp_min(0.0)
            halt_probs.append(halt_prob)
            halt_weights.append(weight)
            claim_scores_per_step.append(claim_scores_t)

        halt_probs_tensor = torch.stack(halt_probs, dim=1)
        halt_weights_tensor = torch.stack(halt_weights, dim=1)
        diagnostics = aggregate_step_diagnostics(step_diagnostics)
        diagnostics.update(
            {
                "expected_steps_mean": expected_steps.mean(),
                "final_step_mass_mean": halt_weights_tensor[:, -1].mean(),
                "halt_weight_sum_mean": halt_weights_tensor.sum(dim=1).mean(),
                "halt_weight_sum_std": halt_weights_tensor.sum(dim=1).std(unbiased=False),
                "halt_prob_mean_by_step": halt_probs_tensor.mean(dim=0),
                "halt_weight_mean_by_step": halt_weights_tensor.mean(dim=0),
            }
        )
        conflict_logits = self.conflict_head(h).squeeze(-1)
        repair_logits = self.repair_head(h)
        return {
            "logits": weighted_logits,
            "diagnostics": diagnostics,
            "conflict_logits": conflict_logits,
            "repair_logits": repair_logits,
            "mode_logits": mode_logits,
            "halt_probs": halt_probs_tensor,
            "halt_weights": halt_weights_tensor,
            "expected_steps": expected_steps,
            "claim_scores_per_step": claim_scores_per_step,
            # Audit hook: final substrate state at step max_steps. Pure
            # instrumentation, never read by the loss / training path.
            "final_h": h,
        }


def aggregate_step_diagnostics(step_diagnostics):
    keys = step_diagnostics[0].keys()
    aggregated = {}
    for key in keys:
        values = [diagnostics[key] for diagnostics in step_diagnostics]
        if torch.is_tensor(values[0]):
            aggregated[key] = torch.stack(values).mean(dim=0)
    return aggregated
