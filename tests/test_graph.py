from mbs.datasets import BeliefRepairDataset
from mbs.graph import collate_graph_samples
from mbs.tokenizer import SimpleTokenizer


def test_graph_batch_shapes_and_indices():
    tokenizer = SimpleTokenizer()
    samples = [BeliefRepairDataset(2, split="train", seed=3)[idx] for idx in range(2)]
    batch = collate_graph_samples(samples, tokenizer)
    assert batch["cell_type_ids"].ndim == 2
    assert batch["edge_index"].ndim == 3
    assert batch["edge_type_ids"].shape == batch["edge_mask"].shape
    for batch_idx in range(2):
        valid_nodes = int(batch["node_mask"][batch_idx].sum().item())
        valid_edges = int(batch["edge_mask"][batch_idx].sum().item())
        query_idx = int(batch["query_node_idx"][batch_idx])
        assert 0 <= query_idx < valid_nodes
        assert batch["edge_index"][batch_idx, :valid_edges].max().item() < valid_nodes
        assert (batch["edge_index"][batch_idx, :valid_edges, 1] == query_idx).any()
