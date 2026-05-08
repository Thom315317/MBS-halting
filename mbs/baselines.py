import torch
from torch import nn

from .halting import AdaptiveHaltingController


class RelationalGCNClassifier(nn.Module):
    def __init__(
        self,
        vocab_size,
        num_values,
        d_model=96,
        num_cell_types=8,
        num_edge_types=12,
        message_steps=8,
        dropout=0.1,
    ):
        super().__init__()
        self.message_steps = message_steps
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.node_type_embedding = nn.Embedding(num_cell_types, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        self.rel_linears = nn.ModuleList([nn.Linear(d_model, d_model, bias=False) for _ in range(num_edge_types)])
        self.update = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.state_norm = nn.LayerNorm(d_model)
        self.answer_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, num_values))
        self.conflict_head = nn.Linear(d_model, 1)
        self.repair_head = nn.Linear(d_model, num_values)

    def forward(self, batch, message_steps=None):
        steps = self.message_steps if message_steps is None else int(message_steps)
        cell_type_ids = batch["cell_type_ids"]
        token_ids = batch["cell_token_ids"]
        node_mask = batch["node_mask"]
        edge_index = batch["edge_index"]
        edge_type_ids = batch["edge_type_ids"]
        edge_mask = batch["edge_mask"]

        h = self.token_embedding(token_ids) + self.node_type_embedding(cell_type_ids)
        h = self.input_norm(h)
        batch_size, num_nodes, d_model = h.shape
        num_edges = edge_index.size(1)
        batch_idx = torch.arange(batch_size, device=h.device).unsqueeze(1).expand(batch_size, num_edges)

        update_norms = []
        state_norms = []
        stability_losses = []

        for _ in range(steps):
            src = edge_index[..., 0].clamp(min=0, max=max(num_nodes - 1, 0))
            dst = edge_index[..., 1].clamp(min=0, max=max(num_nodes - 1, 0))
            h_src = h[batch_idx, src]
            messages = h.new_zeros(batch_size, num_edges, d_model)
            for edge_type, rel_linear in enumerate(self.rel_linears):
                mask = (edge_type_ids == edge_type) & edge_mask
                if mask.any():
                    messages[mask] = rel_linear(h_src[mask]).to(messages.dtype)
            messages = messages * edge_mask.unsqueeze(-1).float()

            flat_dst = (batch_idx * num_nodes + dst).reshape(-1)
            flat_messages = messages.reshape(-1, d_model)
            agg = h.new_zeros(batch_size * num_nodes, d_model)
            agg.index_add_(0, flat_dst, flat_messages)
            degree = h.new_zeros(batch_size * num_nodes, 1)
            degree.index_add_(0, flat_dst, edge_mask.reshape(-1, 1).float())
            agg = agg / degree.clamp(min=1.0)
            agg = agg.view(batch_size, num_nodes, d_model)

            delta = self.update(torch.cat([h, agg], dim=-1))
            update = delta * node_mask.unsqueeze(-1).float()
            next_h = self.state_norm(h + update)
            h = torch.where(node_mask.unsqueeze(-1), next_h, h)
            update_norms.append(masked_mean(update.pow(2).sum(dim=-1).sqrt(), node_mask))
            state_norms.append(masked_mean(h.pow(2).sum(dim=-1).sqrt(), node_mask))
            stability_losses.append(masked_mean(update.pow(2).sum(dim=-1), node_mask))

        batch_ids = torch.arange(batch_size, device=h.device)
        query_state = h[batch_ids, batch["query_node_idx"]]
        diagnostics = {
            "update_scale_mean": h.new_tensor(1.0),
            "update_scale_std": h.new_tensor(0.0),
            "update_norm_mean": torch.stack(update_norms).mean(),
            "state_norm_mean": torch.stack(state_norms).mean(),
            "stability_loss": torch.stack(stability_losses).mean(),
        }
        return {
            "logits": self.answer_head(query_state),
            "diagnostics": diagnostics,
            "conflict_logits": self.conflict_head(h).squeeze(-1),
            "repair_logits": self.repair_head(h),
        }


class RelationalGCNHaltingClassifier(nn.Module):
    """RGCN backbone identical to RelationalGCNClassifier, with the same
    ACT-lite halting scheme as MBSModel._forward_adaptive_halting. Used by
    the rgcn_repair_stability_act baseline."""

    def __init__(
        self,
        vocab_size,
        num_values,
        d_model=96,
        num_cell_types=8,
        num_edge_types=12,
        message_steps=16,
        dropout=0.1,
        halting_config=None,
        force_terminal_step=None,
        warmup_terminal_step=None,
    ):
        super().__init__()
        halting_config = halting_config or {}
        self.force_terminal_step = int(force_terminal_step) if force_terminal_step else None
        self.warmup_terminal_step = int(warmup_terminal_step) if warmup_terminal_step else None
        self._warmup_active = False
        if self.force_terminal_step is not None:
            self.message_steps = self.force_terminal_step
            self.halting_min_steps = self.force_terminal_step
        else:
            self.message_steps = int(halting_config.get("max_message_steps", message_steps))
            self.halting_min_steps = int(halting_config.get("min_message_steps", 4))
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.node_type_embedding = nn.Embedding(num_cell_types, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        self.rel_linears = nn.ModuleList([nn.Linear(d_model, d_model, bias=False) for _ in range(num_edge_types)])
        self.update = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.state_norm = nn.LayerNorm(d_model)
        self.answer_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, num_values))
        self.conflict_head = nn.Linear(d_model, 1)
        self.repair_head = nn.Linear(d_model, num_values)
        self.halting_controller = AdaptiveHaltingController(
            d_state=d_model,
            init_halt_prob=halting_config.get("init_halt_prob", 0.05),
        )

    def set_warmup_active(self, active):
        self._warmup_active = bool(active) and self.warmup_terminal_step is not None

    def forward(self, batch, message_steps=None):
        if message_steps is not None:
            max_steps = int(message_steps)
        elif self._warmup_active and self.warmup_terminal_step is not None:
            max_steps = int(self.warmup_terminal_step)
        else:
            max_steps = int(self.message_steps)
        cell_type_ids = batch["cell_type_ids"]
        token_ids = batch["cell_token_ids"]
        node_mask = batch["node_mask"]
        edge_index = batch["edge_index"]
        edge_type_ids = batch["edge_type_ids"]
        edge_mask = batch["edge_mask"]
        query_node_idx = batch["query_node_idx"]

        h = self.token_embedding(token_ids) + self.node_type_embedding(cell_type_ids)
        h = self.input_norm(h)
        batch_size, num_nodes, d_model = h.shape
        num_edges = edge_index.size(1)
        batch_idx = torch.arange(batch_size, device=h.device).unsqueeze(1).expand(batch_size, num_edges)
        seq_idx = torch.arange(batch_size, device=h.device)
        num_values = self.answer_head[-1].out_features
        if self._warmup_active and self.warmup_terminal_step is not None and message_steps is None:
            min_steps = int(self.warmup_terminal_step)
        else:
            min_steps = max(1, min(int(self.halting_min_steps), int(max_steps)))

        weighted_logits = h.new_zeros(batch_size, num_values)
        remaining_mass = h.new_ones(batch_size)
        expected_steps = h.new_zeros(batch_size)
        halt_probs = []
        halt_weights = []
        update_norms = []
        state_norms = []
        stability_losses = []

        for step_idx in range(int(max_steps)):
            step_number = step_idx + 1
            src = edge_index[..., 0].clamp(min=0, max=max(num_nodes - 1, 0))
            dst = edge_index[..., 1].clamp(min=0, max=max(num_nodes - 1, 0))
            h_src = h[batch_idx, src]
            messages = h.new_zeros(batch_size, num_edges, d_model)
            for edge_type, rel_linear in enumerate(self.rel_linears):
                mask = (edge_type_ids == edge_type) & edge_mask
                if mask.any():
                    messages[mask] = rel_linear(h_src[mask]).to(messages.dtype)
            messages = messages * edge_mask.unsqueeze(-1).float()

            flat_dst = (batch_idx * num_nodes + dst).reshape(-1)
            flat_messages = messages.reshape(-1, d_model)
            agg = h.new_zeros(batch_size * num_nodes, d_model)
            agg.index_add_(0, flat_dst, flat_messages)
            degree = h.new_zeros(batch_size * num_nodes, 1)
            degree.index_add_(0, flat_dst, edge_mask.reshape(-1, 1).float())
            agg = agg / degree.clamp(min=1.0)
            agg = agg.view(batch_size, num_nodes, d_model)

            delta = self.update(torch.cat([h, agg], dim=-1))
            update = delta * node_mask.unsqueeze(-1).float()
            next_h = self.state_norm(h + update)
            h = torch.where(node_mask.unsqueeze(-1), next_h, h)
            update_norms.append(masked_mean(update.pow(2).sum(dim=-1).sqrt(), node_mask))
            state_norms.append(masked_mean(h.pow(2).sum(dim=-1).sqrt(), node_mask))
            stability_losses.append(masked_mean(update.pow(2).sum(dim=-1), node_mask))

            query_state = h[seq_idx, query_node_idx]
            step_logits = self.answer_head(query_state)
            halt_prob = self.halting_controller(h, query_node_idx, node_mask)
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
        diagnostics = {
            "update_scale_mean": h.new_tensor(1.0),
            "update_scale_std": h.new_tensor(0.0),
            "update_norm_mean": torch.stack(update_norms).mean(),
            "state_norm_mean": torch.stack(state_norms).mean(),
            "stability_loss": torch.stack(stability_losses).mean(),
            "expected_steps_mean": expected_steps.mean(),
            "final_step_mass_mean": halt_weights_tensor[:, -1].mean(),
            "halt_weight_sum_mean": halt_weights_tensor.sum(dim=1).mean(),
            "halt_weight_sum_std": halt_weights_tensor.sum(dim=1).std(unbiased=False),
            "halt_prob_mean_by_step": halt_probs_tensor.mean(dim=0),
            "halt_weight_mean_by_step": halt_weights_tensor.mean(dim=0),
            "ponder_active_signal": h.new_tensor(0.0 if (self.force_terminal_step is not None or self._warmup_active) else 1.0),
        }
        return {
            "logits": weighted_logits,
            "diagnostics": diagnostics,
            "conflict_logits": self.conflict_head(h).squeeze(-1),
            "repair_logits": self.repair_head(h),
            "halt_probs": halt_probs_tensor,
            "halt_weights": halt_weights_tensor,
            "expected_steps": expected_steps,
        }


def masked_mean(values, mask):
    mask_f = mask.float()
    return (values * mask_f).sum() / mask_f.sum().clamp(min=1.0)
