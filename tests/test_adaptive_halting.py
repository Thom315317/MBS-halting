import torch

from mbs.datasets import BeliefRepairDataset
from mbs.graph import collate_graph_samples
from mbs.tokenizer import SimpleTokenizer
from mbs.train import build_model, compute_loss


def halting_config():
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
            "max_message_steps": 6,
            "min_message_steps": 3,
            "lambda_ponder": 0.001,
            "init_halt_prob": 0.05,
        },
    }


def make_batch(batch_size=2):
    tokenizer = SimpleTokenizer()
    samples = [BeliefRepairDataset(batch_size, split="train", seed=11)[idx] for idx in range(batch_size)]
    return tokenizer, collate_graph_samples(samples, tokenizer)


def test_adaptive_halting_forward_cpu_shapes_and_diagnostics():
    tokenizer, batch = make_batch()
    model = build_model("mbs_adaptive_halting", halting_config(), tokenizer)
    outputs = model(batch)
    diagnostics = outputs["diagnostics"]

    assert outputs["logits"].shape == (2, 8)
    assert outputs["halt_weights"].shape == (2, 6)
    assert outputs["halt_probs"].shape == (2, 6)
    assert "expected_steps_mean" in diagnostics
    assert "final_step_mass_mean" in diagnostics
    assert "halt_prob_mean_by_step" in diagnostics
    assert "halt_weight_mean_by_step" in diagnostics
    for value in diagnostics.values():
        if torch.is_tensor(value):
            assert torch.isfinite(value).all()


def test_halt_weights_sum_to_one_and_min_steps_are_respected():
    tokenizer, batch = make_batch()
    model = build_model("mbs_adaptive_halting", halting_config(), tokenizer)
    outputs = model(batch)
    halt_weights = outputs["halt_weights"]

    assert torch.allclose(halt_weights.sum(dim=1), torch.ones(halt_weights.size(0)), atol=1e-5)
    assert torch.allclose(halt_weights[:, :2], torch.zeros_like(halt_weights[:, :2]), atol=1e-7)
    expected_steps = outputs["expected_steps"]
    assert (expected_steps >= 3.0).all()
    assert (expected_steps <= 6.0).all()
    previous_mass = halt_weights[:, :-1].sum(dim=1)
    assert torch.allclose(halt_weights[:, -1], 1.0 - previous_mass, atol=1e-5)


def test_adaptive_halting_training_step_cpu():
    tokenizer, batch = make_batch()
    config = halting_config()
    model = build_model("mbs_adaptive_halting", config, tokenizer)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    outputs = model(batch)
    loss, parts = compute_loss(outputs, batch, config, "mbs_adaptive_halting")
    loss.backward()
    optimizer.step()

    assert torch.isfinite(loss)
    assert parts["ponder_loss"].item() > 0.0


