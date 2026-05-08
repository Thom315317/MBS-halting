from mbs.datasets import BeliefRepairDataset
from mbs.graph import EDGE_TYPES


def test_dataset_sample_contains_required_fields():
    sample = BeliefRepairDataset(1, split="train", seed=1)[0]
    assert sample["text"]
    assert isinstance(sample["belief_cells"], list) and sample["belief_cells"]
    assert isinstance(sample["edges"], list) and sample["edges"]
    assert isinstance(sample["answer_class"], int)
    assert sample["target_value"]


def test_ood_entity_has_more_entities_than_train():
    train = BeliefRepairDataset(1, split="train", seed=1)[0]
    ood = BeliefRepairDataset(1, split="ood_entity", seed=1)[0]
    assert ood["metadata"]["num_entities"] > train["metadata"]["num_entities"]


def test_conflict_examples_contain_contradicts_edges():
    sample = BeliefRepairDataset(8, split="ood_conflict", seed=2)[0]
    assert any(edge["type"] == "CONTRADICTS" for edge in sample["edges"])
