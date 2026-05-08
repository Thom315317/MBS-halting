import torch
from torch import nn

from .graph import OPERATION_MODES


class MorphogeneticBeliefSubstrate(nn.Module):
    def __init__(
        self,
        d_state=96,
        num_cell_types=8,
        num_edge_types=12,
        num_operation_modes=4,
        dropout=0.1,
        use_modes=True,
        use_gate=True,
    ):
        super().__init__()
        self.d_state = d_state
        self.num_operation_modes = num_operation_modes
        self.use_modes = use_modes
        self.use_gate = use_gate
        self.node_type_embedding = nn.Embedding(num_cell_types, d_state)
        self.edge_type_embedding = nn.Embedding(num_edge_types, d_state)
        msg_in = d_state * 4
        upd_in = d_state * 4
        self.message_mlp = nn.Sequential(
            nn.Linear(msg_in, d_state * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_state * 2, d_state),
        )
        self.mode_head = nn.Sequential(
            nn.Linear(msg_in, d_state),
            nn.GELU(),
            nn.Linear(d_state, num_operation_modes),
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(upd_in, d_state * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_state * 2, d_state),
        )
        self.state_norm = nn.LayerNorm(d_state)
        self.scale_head = nn.Sequential(
            nn.Linear(upd_in, d_state),
            nn.GELU(),
            nn.Linear(d_state, 1),
        )
        nn.init.zeros_(self.scale_head[-1].weight)
        nn.init.zeros_(self.scale_head[-1].bias)
        self.mode_scale_bias = nn.Parameter(torch.zeros(num_operation_modes))
        self.energy_head = nn.Sequential(
            nn.Linear(msg_in, d_state),
            nn.GELU(),
            nn.Linear(d_state, 1),
        )

    def forward(
        self,
        h,
        node_type_ids,
        edge_index,
        edge_type_ids,
        node_mask,
        edge_mask,
        query_embedding,
        message_steps=8,
    ):
        batch_size, num_nodes, _ = h.shape
        num_edges = edge_index.size(1)
        batch_idx = torch.arange(batch_size, device=h.device).unsqueeze(1).expand(batch_size, num_edges)
        query_nodes = query_embedding.unsqueeze(1).expand(batch_size, num_nodes, self.d_state)
        node_type_emb = self.node_type_embedding(node_type_ids)

        scale_means = []
        scale_stds = []
        update_norms = []
        state_norms = []
        stability_losses = []
        energy_means = []
        entropy_terms = []
        mode_totals = h.new_zeros(self.num_operation_modes)
        mode_count = h.new_tensor(0.0)
        last_mode_logits = None

        for _ in range(int(message_steps)):
            src = edge_index[..., 0].clamp(min=0, max=max(num_nodes - 1, 0))
            dst = edge_index[..., 1].clamp(min=0, max=max(num_nodes - 1, 0))
            h_src = h[batch_idx, src]
            h_dst = h[batch_idx, dst]
            edge_emb = self.edge_type_embedding(edge_type_ids)
            query_edges = query_embedding.unsqueeze(1).expand(batch_size, num_edges, self.d_state)
            edge_features = torch.cat([h_src, h_dst, edge_emb, query_edges], dim=-1)
            messages = self.message_mlp(edge_features)
            mode_logits = self.mode_head(edge_features)
            last_mode_logits = mode_logits
            if self.use_modes:
                mode_probs = torch.softmax(mode_logits, dim=-1)
                mode_scale = 1.0 + 0.1 * torch.tanh((mode_probs * self.mode_scale_bias).sum(dim=-1, keepdim=True))
                messages = messages * mode_scale
                masked_probs = mode_probs * edge_mask.unsqueeze(-1).float()
                mode_totals = mode_totals + masked_probs.sum(dim=(0, 1))
                mode_count = mode_count + edge_mask.float().sum()
                entropy = -(mode_probs.clamp_min(1e-6) * mode_probs.clamp_min(1e-6).log()).sum(dim=-1)
                entropy_terms.append(masked_mean(entropy, edge_mask))
            messages = messages * edge_mask.unsqueeze(-1).float()

            agg = h.new_zeros(batch_size * num_nodes, self.d_state)
            flat_dst = (batch_idx * num_nodes + dst).reshape(-1)
            agg.index_add_(0, flat_dst, messages.reshape(-1, self.d_state))
            agg = agg.view(batch_size, num_nodes, self.d_state)

            update_input = torch.cat([h, agg, node_type_emb, query_nodes], dim=-1)
            delta = self.update_mlp(update_input)
            if self.use_gate:
                raw_scale = self.scale_head(update_input)
                scale = 1.0 + 0.5 * torch.tanh(raw_scale)
            else:
                scale = torch.ones(batch_size, num_nodes, 1, device=h.device, dtype=h.dtype)
            update = scale * delta
            update = update * node_mask.unsqueeze(-1).float()
            next_h = h + update
            next_h = self.state_norm(next_h)
            h = torch.where(node_mask.unsqueeze(-1), next_h, h)

            scale_means.append(masked_mean(scale.squeeze(-1), node_mask))
            scale_stds.append(masked_std(scale.squeeze(-1), node_mask))
            update_norms.append(masked_mean(update.pow(2).sum(dim=-1).sqrt(), node_mask))
            state_norms.append(masked_mean(h.pow(2).sum(dim=-1).sqrt(), node_mask))
            stability_losses.append(masked_mean(update.pow(2).sum(dim=-1), node_mask))
            energy = self.energy_head(edge_features).squeeze(-1).abs()
            structural_edge = (edge_type_ids == 7) | (edge_type_ids == 8)
            energy_means.append(masked_mean(energy, edge_mask & structural_edge))

        mode_means = mode_totals / mode_count.clamp(min=1.0)
        diagnostics = {
            "update_scale_mean": torch.stack(scale_means).mean(),
            "update_scale_std": torch.stack(scale_stds).mean(),
            "update_norm_mean": torch.stack(update_norms).mean(),
            "state_norm_mean": torch.stack(state_norms).mean(),
            "stability_loss": torch.stack(stability_losses).mean(),
            "energy_mean": torch.stack(energy_means).mean(),
            "mode_entropy": torch.stack(entropy_terms).mean() if entropy_terms else h.new_tensor(0.0),
            "mode_PROPAGATE_mean": mode_means[OPERATION_MODES["PROPAGATE"]],
            "mode_STABILIZE_mean": mode_means[OPERATION_MODES["STABILIZE"]],
            "mode_REPAIR_mean": mode_means[OPERATION_MODES["REPAIR"]],
            "mode_RESOLVE_CONFLICT_mean": mode_means[OPERATION_MODES["RESOLVE_CONFLICT"]],
        }
        return h, diagnostics, last_mode_logits


def masked_mean(values, mask):
    mask_f = mask.float()
    denom = mask_f.sum().clamp(min=1.0)
    return (values * mask_f).sum() / denom


def masked_std(values, mask):
    mean = masked_mean(values, mask)
    mask_f = mask.float()
    denom = mask_f.sum().clamp(min=1.0)
    return (((values - mean).pow(2) * mask_f).sum() / denom).sqrt()
