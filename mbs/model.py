import torch
from torch import nn

from .graph import CELL_TYPES
from .halting import AdaptiveHaltingController
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
        if adaptive_halting:
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
        mode_logits = None

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
        }


def aggregate_step_diagnostics(step_diagnostics):
    keys = step_diagnostics[0].keys()
    aggregated = {}
    for key in keys:
        values = [diagnostics[key] for diagnostics in step_diagnostics]
        if torch.is_tensor(values[0]):
            aggregated[key] = torch.stack(values).mean(dim=0)
    return aggregated
