import torch

from mbs.datasets import BeliefRepairDataset
from mbs.graph import collate_graph_samples
from mbs.tokenizer import SimpleTokenizer
from mbs.train import build_model, compute_loss


def tiny_config():
    return {
        "d_state": 32,
        "dropout": 0.0,
        "message_steps": 2,
        "lambda_answer": 1.0,
        "lambda_conflict": 0.2,
        "lambda_repair": 0.5,
        "lambda_stability": 0.01,
        "lambda_mode_entropy": 0.001,
        "halting": {
            "max_message_steps": 4,
            "min_message_steps": 2,
            "init_halt_prob": 0.05,
            "lambda_ponder": 0.001,
        },
    }


def test_one_training_step_v0_3_variants_cpu():
    tokenizer = SimpleTokenizer()
    batch = collate_graph_samples([BeliefRepairDataset(2, split="train", seed=6)[idx] for idx in range(2)], tokenizer)
    for variant in ["mbs_adaptive_halting", "rgcn_repair_stability"]:
        model = build_model(variant, tiny_config(), tokenizer)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        outputs = model(batch)
        loss, _ = compute_loss(outputs, batch, tiny_config(), variant)
        loss.backward()
        optimizer.step()
        assert torch.isfinite(loss)
