import json

from mbs.benchmark import make_summary_row
from mbs.train import checkpoint_row_is_better, train_one


def tiny_halting_train_config():
    return {
        "seed": 13,
        "device": "cpu",
        "task": "belief_repair_hard",
        "hard_dataset": True,
        "train_size": 4,
        "val_size": 2,
        "ood_size": 2,
        "batch_size": 2,
        "max_epochs": 2,
        "d_state": 16,
        "dropout": 0.0,
        "message_steps": 2,
        "lambda_answer": 1.0,
        "lambda_conflict": 0.2,
        "lambda_repair": 0.5,
        "lambda_stability": 0.01,
        "lambda_mode_entropy": 0.001,
        "grad_clip": 1.0,
        "halting": {
            "max_message_steps": 4,
            "min_message_steps": 2,
            "lambda_ponder": 0.001,
            "init_halt_prob": 0.05,
        },
    }


def test_checkpoint_tie_breaker_prefers_lower_val_loss_then_earlier_epoch():
    selected = {"epoch": 1, "val_acc": 1.0, "val_loss": 0.10}
    worse_loss = {"epoch": 2, "val_acc": 1.0, "val_loss": 0.20}
    better_loss = {"epoch": 3, "val_acc": 1.0, "val_loss": 0.05}
    exact_tie = {"epoch": 4, "val_acc": 1.0, "val_loss": 0.10}
    better_acc = {"epoch": 5, "val_acc": 1.01, "val_loss": 1.00}

    assert not checkpoint_row_is_better(worse_loss, selected)
    assert checkpoint_row_is_better(better_loss, selected)
    assert not checkpoint_row_is_better(exact_tie, selected)
    assert checkpoint_row_is_better(better_acc, selected)


def test_halting_train_writes_checkpoint_policy_and_epoch_checkpoints(tmp_path):
    output_dir = tmp_path / "out"
    checkpoint_dir = tmp_path / "checkpoints"
    train_one(tiny_halting_train_config(), "mbs_adaptive_halting", str(output_dir), str(checkpoint_dir))

    result_path = output_dir / "mbs_adaptive_halting_train_results.json"
    data = json.loads(result_path.read_text(encoding="utf-8"))

    assert "checkpoint_policy" in data
    assert data["checkpoint_policy"]["primary_metric"] == "val_acc"
    assert data["checkpoint_policy"]["tie_breaker"] == ["val_loss:min", "earlier_epoch"]
    assert "selected_epoch" in data["checkpoint_policy"]
    assert "best_ood_mixed_epoch" in data
    assert "ood_regression_from_best_epoch" in data
    assert (checkpoint_dir / "mbs_adaptive_halting_epoch_1.pt").exists()
    assert (checkpoint_dir / "mbs_adaptive_halting_epoch_2.pt").exists()
    assert (checkpoint_dir / "mbs_adaptive_halting_best.pt").exists()


def test_benchmark_summary_row_includes_checkpoint_diagnostics():
    train = {
        "parameter_count": 123,
        "history": [
            {"epoch": 1, "val_acc": 0.9, "val_loss": 0.2, "ood_entity_acc": 0.1, "ood_conflict_acc": 0.2, "ood_rule_acc": 0.3, "ood_mixed_acc": 0.4},
            {"epoch": 2, "val_acc": 1.0, "val_loss": 0.1, "ood_entity_acc": 0.5, "ood_conflict_acc": 0.6, "ood_rule_acc": 0.7, "ood_mixed_acc": 0.8},
        ],
        "checkpoint_policy": {
            "selected_epoch": 2,
            "selected_epoch_val_acc": 1.0,
            "selected_epoch_val_loss": 0.1,
            "selected_epoch_ood_mixed_acc": 0.8,
        },
        "best_ood_mixed_epoch": {"epoch": 2, "ood_mixed_acc": 0.8, "val_acc": 1.0, "val_loss": 0.1},
        "ood_regression_from_best_epoch": 0.0,
        "warnings": [],
    }
    eval_results = {"best_ood_mixed_sweep_acc": 0.82, "ood_mixed_loss_degradation": 0.12}

    row = make_summary_row("mbs_adaptive_halting", train, eval_results)

    assert row["selected_checkpoint_epoch"] == 2
    assert row["selected_checkpoint_ood_mixed_acc"] == 0.8
    assert row["best_ood_mixed_epoch"]["epoch"] == 2
    assert row["ood_regression_from_best_epoch"] == 0.0
