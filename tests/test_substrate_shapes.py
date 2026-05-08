import torch

from mbs.datasets import BeliefRepairDataset
from mbs.graph import collate_graph_samples
from mbs.model import MBSModel
from mbs.tokenizer import SimpleTokenizer


def test_mbs_forward_variable_nodes_cpu():
    tokenizer = SimpleTokenizer()
    samples = [
        BeliefRepairDataset(1, split="train", seed=4)[0],
        BeliefRepairDataset(1, split="ood_entity", seed=5)[0],
    ]
    batch = collate_graph_samples(samples, tokenizer)
    model = MBSModel(vocab_size=len(tokenizer.tokens), num_values=8, d_state=32, message_steps=2, dropout=0.0)
    outputs = model(batch)
    assert outputs["logits"].shape == (2, 8)
    assert outputs["conflict_logits"].shape == batch["cell_type_ids"].shape
    for key, value in outputs["diagnostics"].items():
        assert torch.isfinite(value), key
    assert torch.isclose(outputs["diagnostics"]["update_scale_mean"], torch.tensor(1.0), atol=1e-4)
