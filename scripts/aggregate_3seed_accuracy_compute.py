#!/usr/bin/env python
"""
Aggregator for the 3-seed accuracy/compute campaign.

Walks <root>/seed{N}/<variant>/, reads each benchmark_summary.json and
the matching <variant>_train_results.json, then writes <root>/aggregate_summary.{json,md}.

Verdict rules (indicative, not a scientific claim):
  A. mbs_compute_efficient_win:
     - mbs_adaptive_halting ood_mixed_mean >= rgcn_fixed ood_mixed_mean - 0.02
     - mbs_adaptive_halting expected_steps_mean <= 0.75 * 8.0  (RGCN fixed runs at T=8)
     - mbs_adaptive_halting ood_mixed_mean >= rgcn_act_warmup ood_mixed_mean
  B. rgcn_act_tradeoff:
     - rgcn_act_warmup expected_steps_mean < 8.0
     - rgcn_act_warmup ood_mixed_mean <= rgcn_fixed ood_mixed_mean - 0.03
  C. no_clear_winner: otherwise.
"""
from pathlib import Path
import argparse
import json
import math
import sys


VARIANTS = (
    "mbs_adaptive_halting",
    "rgcn_repair_stability",
    "rgcn_repair_stability_act_warmup_t8",
)


def _load(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _mean_std(values):
    values = [float(v) for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not values:
        return {"mean": None, "std": None, "n": 0}
    if len(values) == 1:
        return {"mean": values[0], "std": 0.0, "n": 1}
    m = sum(values) / len(values)
    s = math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))
    return {"mean": m, "std": s, "n": len(values)}


def _collect_variant(root, variant, seeds):
    rows = []
    summaries_paths = []
    for seed in seeds:
        run_dir = root / f"seed{seed}" / variant
        bench_path = run_dir / "benchmark_summary.json"
        train_path = run_dir / f"{variant}_train_results.json"
        if not bench_path.exists():
            continue
        bench = _load(bench_path)
        train = _load(train_path) if train_path.exists() else None
        bench_row = next((r for r in bench["variants"] if r["variant"] == variant), None)
        if bench_row is None:
            continue
        last = train["history"][-1] if train else {}
        best_ood = train.get("best_ood_mixed_epoch", {}) if train else {}
        rows.append({
            "seed": seed,
            "summary_path": str(bench_path),
            "train_results_path": str(train_path),
            "parameter_count": bench_row.get("parameter_count"),
            "selected_checkpoint_epoch": bench_row.get("selected_checkpoint_epoch"),
            "selected_checkpoint_ood_mixed_acc": bench_row.get("selected_checkpoint_ood_mixed_acc"),
            "best_ood_mixed_epoch": best_ood.get("epoch"),
            "best_ood_mixed_acc": best_ood.get("ood_mixed_acc"),
            "best_ood_expected_steps": best_ood.get("expected_steps_mean"),
            "best_ood_final_step_mass": best_ood.get("final_step_mass_mean"),
            "ood_regression_from_best_epoch": bench_row.get("ood_regression_from_best_epoch"),
            "final_epoch_ood_mixed_acc": bench_row.get("final_epoch_ood_mixed_acc"),
            "ood_entity_acc": bench_row.get("ood_entity_acc"),
            "ood_conflict_acc": bench_row.get("ood_conflict_acc"),
            "ood_rule_acc": bench_row.get("ood_rule_acc"),
            "ood_mixed_acc_final": bench_row.get("ood_mixed_acc"),
            "expected_steps_mean": last.get("ood_mixed_expected_steps_mean") or last.get("val_expected_steps_mean"),
            "final_step_mass_mean": last.get("ood_mixed_final_step_mass_mean") or last.get("val_final_step_mass_mean"),
            "ponder_loss": last.get("val_ponder_loss"),
        })
        summaries_paths.append(str(bench_path))
    if not rows:
        return None
    aggregate = {
        "variant": variant,
        "seeds": [r["seed"] for r in rows],
        "n_seeds_valid": len(rows),
        "parameter_count": rows[0]["parameter_count"],
        "summary_paths": summaries_paths,
        "metrics": {
            "ood_mixed_selected": _mean_std(r["selected_checkpoint_ood_mixed_acc"] for r in rows),
            "ood_mixed_best": _mean_std(r["best_ood_mixed_acc"] for r in rows),
            "ood_mixed_final_epoch": _mean_std(r["final_epoch_ood_mixed_acc"] for r in rows),
            "ood_entity": _mean_std(r["ood_entity_acc"] for r in rows),
            "ood_conflict": _mean_std(r["ood_conflict_acc"] for r in rows),
            "ood_rule": _mean_std(r["ood_rule_acc"] for r in rows),
            "expected_steps": _mean_std(r["expected_steps_mean"] for r in rows),
            "final_step_mass": _mean_std(r["final_step_mass_mean"] for r in rows),
            "ponder_loss": _mean_std(r["ponder_loss"] for r in rows),
            "ood_regression_from_best_epoch": _mean_std(r["ood_regression_from_best_epoch"] for r in rows),
            "best_ood_expected_steps": _mean_std(r["best_ood_expected_steps"] for r in rows),
        },
        "per_seed": rows,
    }
    return aggregate


def _compact_table(per_variant):
    rows = []
    for variant, agg in per_variant.items():
        if agg is None:
            continue
        ood_sel = agg["metrics"]["ood_mixed_selected"]["mean"]
        ood_sel_std = agg["metrics"]["ood_mixed_selected"]["std"] or 0.0
        es_mean = agg["metrics"]["expected_steps"]["mean"]
        es_std = agg["metrics"]["expected_steps"]["std"] or 0.0
        es_used = es_mean if es_mean is not None else 8.0  # RGCN fixed: T=8
        per_step = (ood_sel / es_used) if (ood_sel is not None and es_used > 0) else None
        compute_adj = (ood_sel - 0.01 * es_used) if ood_sel is not None else None
        rows.append({
            "variant": variant,
            "ood_mixed_mean": ood_sel,
            "ood_mixed_std": ood_sel_std,
            "expected_steps_mean": es_used,
            "expected_steps_std": es_std,
            "accuracy_per_step": per_step,
            "compute_adjusted_score": compute_adj,
            "n_seeds_valid": agg["n_seeds_valid"],
        })
    return rows


def _verdict(per_variant):
    mbs = per_variant.get("mbs_adaptive_halting")
    rgcn_fixed = per_variant.get("rgcn_repair_stability")
    rgcn_act = per_variant.get("rgcn_repair_stability_act_warmup_t8")
    if not (mbs and rgcn_fixed and rgcn_act):
        return {"verdict": "incomplete", "details": "missing one or more variants"}
    mbs_ood = mbs["metrics"]["ood_mixed_selected"]["mean"]
    rgcn_fixed_ood = rgcn_fixed["metrics"]["ood_mixed_selected"]["mean"]
    rgcn_act_ood = rgcn_act["metrics"]["ood_mixed_selected"]["mean"]
    mbs_es = mbs["metrics"]["expected_steps"]["mean"]
    rgcn_act_es = rgcn_act["metrics"]["expected_steps"]["mean"]
    rgcn_fixed_steps = 8.0
    if any(v is None for v in (mbs_ood, rgcn_fixed_ood, rgcn_act_ood, mbs_es)):
        return {"verdict": "incomplete", "details": "missing aggregated metric"}
    a_cond = (
        mbs_ood >= rgcn_fixed_ood - 0.02
        and mbs_es <= 0.75 * rgcn_fixed_steps
        and mbs_ood >= rgcn_act_ood
    )
    if a_cond:
        return {
            "verdict": "mbs_compute_efficient_win",
            "details": (
                f"mbs_ood={mbs_ood:.4f} >= rgcn_fixed_ood-0.02={rgcn_fixed_ood - 0.02:.4f}, "
                f"mbs_steps={mbs_es:.2f} <= 0.75*8={0.75 * rgcn_fixed_steps:.2f}, "
                f"mbs_ood >= rgcn_act_warmup_ood={rgcn_act_ood:.4f}"
            ),
        }
    b_cond = rgcn_act_es is not None and rgcn_act_es < rgcn_fixed_steps and rgcn_act_ood <= rgcn_fixed_ood - 0.03
    if b_cond:
        return {
            "verdict": "rgcn_act_tradeoff",
            "details": (
                f"rgcn_act_steps={rgcn_act_es:.2f} < 8.0, "
                f"rgcn_act_ood={rgcn_act_ood:.4f} <= rgcn_fixed_ood-0.03={rgcn_fixed_ood - 0.03:.4f}"
            ),
        }
    return {
        "verdict": "no_clear_winner",
        "details": (
            f"mbs_ood={mbs_ood:.4f} steps={mbs_es:.2f} | "
            f"rgcn_fixed_ood={rgcn_fixed_ood:.4f} steps=8.0 | "
            f"rgcn_act_ood={rgcn_act_ood:.4f} steps={rgcn_act_es if rgcn_act_es is not None else 'n/a'}"
        ),
    }


def _fmt_pair(entry):
    if entry["mean"] is None:
        return "—"
    return f"{entry['mean']:.4f} ± {(entry['std'] or 0.0):.4f}"


def _fmt_pair_n(entry, decimals=2):
    if entry["mean"] is None:
        return "—"
    fmt = f"{{:.{decimals}f}} ± {{:.{decimals}f}}"
    return fmt.format(entry["mean"], entry["std"] or 0.0)


def _markdown(root, per_variant, compact, verdict, seeds):
    lines = [
        "# 3-seed accuracy/compute campaign — belief_repair_hard",
        "",
        f"- root: `{root}`",
        f"- seeds: `{seeds}`",
        f"- verdict: `{verdict['verdict']}`",
        f"- verdict_details: {verdict['details']}",
        "",
        "## Per-variant aggregates",
        "",
        "| Variant | n_seeds | params | OOD mixed (selected) | OOD mixed (best) | OOD entity | OOD conflict | OOD rule | E[steps] | final_step_mass | ponder_loss | OOD regression from best |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in VARIANTS:
        agg = per_variant.get(variant)
        if agg is None:
            lines.append(f"| `{variant}` | 0 | — | — | — | — | — | — | — | — | — | — |")
            continue
        m = agg["metrics"]
        lines.append(
            f"| `{variant}` | {agg['n_seeds_valid']} | {agg['parameter_count']} | "
            f"{_fmt_pair(m['ood_mixed_selected'])} | {_fmt_pair(m['ood_mixed_best'])} | "
            f"{_fmt_pair(m['ood_entity'])} | {_fmt_pair(m['ood_conflict'])} | {_fmt_pair(m['ood_rule'])} | "
            f"{_fmt_pair_n(m['expected_steps'], 2)} | {_fmt_pair(m['final_step_mass'])} | "
            f"{_fmt_pair(m['ponder_loss'])} | {_fmt_pair(m['ood_regression_from_best_epoch'])} |"
        )
    lines.extend([
        "",
        "## Compact diagnostic table",
        "",
        "| variant | ood_mixed_mean | ood_mixed_std | expected_steps_mean | expected_steps_std | accuracy_per_step | compute_adjusted_score |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in compact:
        lines.append(
            f"| `{row['variant']}` | "
            f"{(row['ood_mixed_mean'] if row['ood_mixed_mean'] is not None else 0.0):.4f} | "
            f"{row['ood_mixed_std']:.4f} | "
            f"{(row['expected_steps_mean'] if row['expected_steps_mean'] is not None else 0.0):.2f} | "
            f"{(row['expected_steps_std'] if row['expected_steps_std'] is not None else 0.0):.2f} | "
            f"{(row['accuracy_per_step'] if row['accuracy_per_step'] is not None else 0.0):.4f} | "
            f"{(row['compute_adjusted_score'] if row['compute_adjusted_score'] is not None else 0.0):.4f} |"
        )
    lines.append("")
    lines.append("## Per-seed file paths")
    lines.append("")
    for variant in VARIANTS:
        agg = per_variant.get(variant)
        if agg is None:
            continue
        lines.append(f"### `{variant}`")
        for r in agg["per_seed"]:
            lines.append(f"- seed{r['seed']}: `{r['summary_path']}`  /  `{r['train_results_path']}`")
        lines.append("")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default="results/belief_repair_hard_3seed_accuracy_compute_v1",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    args = parser.parse_args()
    root = Path(args.root)
    if not root.is_absolute():
        root = (Path(__file__).resolve().parents[1] / root).resolve()
    if not root.exists():
        sys.exit(f"missing root {root}")
    per_variant = {variant: _collect_variant(root, variant, args.seeds) for variant in VARIANTS}
    compact = _compact_table(per_variant)
    verdict = _verdict(per_variant)
    payload = {
        "root": str(root),
        "seeds": args.seeds,
        "per_variant": per_variant,
        "compact": compact,
        "verdict": verdict,
    }
    with (root / "aggregate_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    (root / "aggregate_summary.md").write_text(_markdown(root, per_variant, compact, verdict, args.seeds), encoding="utf-8")
    print(f"wrote {root / 'aggregate_summary.json'}")
    print(f"wrote {root / 'aggregate_summary.md'}")
    print(f"verdict: {verdict['verdict']}")
    print(f"  details: {verdict['details']}")


if __name__ == "__main__":
    main()
