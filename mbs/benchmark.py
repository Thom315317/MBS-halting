import argparse
import json
import os
import subprocess
import sys

from .utils import load_config, save_json


def run(cmd):
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def make_markdown(summary):
    lines = [
        "# MBS Benchmark Summary",
        "",
        f"- config: `{summary['config_path']}`",
        f"- output_dir: `{summary['output_dir']}`",
        "",
        "| Variant | Val Acc | OOD Entity | OOD Conflict | OOD Rule | OOD Mixed | Best Sweep OOD Mixed | Loss Deg. | Selected Epoch | Best OOD Epoch | OOD Regr. |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["variants"]:
        lines.append(
            f"| `{row['variant']}` | {row['val_acc']:.4f} | {row['ood_entity_acc']:.4f} | {row['ood_conflict_acc']:.4f} | "
            f"{row['ood_rule_acc']:.4f} | {row['ood_mixed_acc']:.4f} | {row.get('best_ood_mixed_sweep_acc', 0.0):.4f} | "
            f"{row.get('ood_mixed_loss_degradation', 0.0):.4f} | {row.get('selected_checkpoint_epoch', 0)} | "
            f"{row.get('best_ood_mixed_epoch', {}).get('epoch', 0)} | {row.get('ood_regression_from_best_epoch', 0.0):.4f} |"
        )
    return "\n".join(lines)


def make_summary_row(variant, train, eval_results):
    last = train["history"][-1]
    policy = train.get("checkpoint_policy", {})
    selected_epoch = policy.get("selected_epoch")
    best_ood = train.get("best_ood_mixed_epoch", {})
    return {
        "variant": variant,
        "parameter_count": train.get("parameter_count"),
        "val_acc": last["val_acc"],
        "ood_entity_acc": last["ood_entity_acc"],
        "ood_conflict_acc": last["ood_conflict_acc"],
        "ood_rule_acc": last["ood_rule_acc"],
        "ood_mixed_acc": last["ood_mixed_acc"],
        "final_epoch": last["epoch"],
        "final_epoch_val_acc": last["val_acc"],
        "final_epoch_val_loss": last["val_loss"],
        "final_epoch_ood_mixed_acc": last["ood_mixed_acc"],
        "selected_checkpoint_epoch": selected_epoch,
        "selected_checkpoint_val_acc": policy.get("selected_epoch_val_acc"),
        "selected_checkpoint_val_loss": policy.get("selected_epoch_val_loss"),
        "selected_checkpoint_ood_mixed_acc": policy.get("selected_epoch_ood_mixed_acc"),
        "selected_checkpoint_expected_steps_val": policy.get("selected_epoch_expected_steps_val"),
        "selected_checkpoint_final_step_mass_val": policy.get("selected_epoch_final_step_mass_val"),
        "best_ood_mixed_epoch": best_ood,
        "ood_regression_from_best_epoch": train.get("ood_regression_from_best_epoch"),
        "checkpoint_policy": policy,
        "checkpoint_warnings": train.get("warnings", []),
        "best_ood_mixed_sweep_acc": eval_results.get("best_ood_mixed_sweep_acc", last["ood_mixed_acc"]),
        "ood_mixed_loss_degradation": eval_results.get("ood_mixed_loss_degradation", 0.0),
        "expected_steps_mean": last.get("val_expected_steps_mean"),
        "train_results": train,
        "eval_results": eval_results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--variants",
        default="mbs_adaptive_halting,rgcn_repair_stability,rgcn_repair_stability_act_warmup_t8",
    )
    parser.add_argument("--output-dir", default="results/belief_repair_debug")
    parser.add_argument("--checkpoint-dir", default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    checkpoint_dir = args.checkpoint_dir or os.path.join(args.output_dir, "checkpoints")
    variants = [variant.strip() for variant in args.variants.split(",") if variant.strip()]
    summary = {"config_path": args.config, "output_dir": args.output_dir, "variants": []}
    for variant in variants:
        run([sys.executable, "-m", "mbs.train", "--config", args.config, "--variant", variant, "--output-dir", args.output_dir, "--checkpoint-dir", checkpoint_dir])
        checkpoint = os.path.join(checkpoint_dir, f"{variant}_best.pt")
        run([sys.executable, "-m", "mbs.eval", "--checkpoint", checkpoint, "--config", args.config, "--output-dir", args.output_dir])
        train = load_json(os.path.join(args.output_dir, f"{variant}_train_results.json"))
        eval_results = load_json(os.path.join(args.output_dir, f"{variant}_eval_results.json"))
        summary["variants"].append(make_summary_row(variant, train, eval_results))
    save_json(os.path.join(args.output_dir, "benchmark_summary.json"), summary)
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "benchmark_summary.md"), "w", encoding="utf-8") as handle:
        handle.write(make_markdown(summary))
    print(f"Saved {os.path.join(args.output_dir, 'benchmark_summary.json')}")


if __name__ == "__main__":
    main()
