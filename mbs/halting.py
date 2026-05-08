import math

import torch
from torch import nn


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
