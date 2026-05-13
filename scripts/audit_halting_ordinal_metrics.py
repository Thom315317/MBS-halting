#!/usr/bin/env python
"""Ordinal-metric audit for adaptive halting on v1.

Computes the H7 metric set (S_all, S_easy, AUC_h>=θ for θ ∈ {6,7,8,9},
MACRO_AUC, adjacent-bucket margins, chosen-step entropy, dominant-bin
mass, collapse taxonomy) on any per-seed audit CSV that follows the
shape produced by
`scripts/audit_rgcn_h6_two_stage_controller_vs_required_hops.py`.

Usage :
    python scripts/audit_halting_ordinal_metrics.py \
        --input  results/claim_strengthening/rgcn_h6_two_stage/rgcn_h6_two_stage_per_seed.csv \
        --label  rgcn_h6_two_stage \
        --output-dir results/claim_strengthening/h7_ordinal_halting/audits

Inputs the script understands :
    *_per_seed.csv with columns
        seed, split, sample_idx, controller_step_expected, chosen_step,
        oracle_step, required_hops, oracle_depth

Optional second input (for accuracy / floor / final masses, taken
from the same campaign's summary.json) :
    --summary results/claim_strengthening/rgcn_h6_two_stage/rgcn_h6_two_stage_summary.json

Outputs three files in --output-dir, all keyed by --label :
    {label}_ordinal_metrics_per_seed_split.csv
    {label}_ordinal_metrics_summary.json
    {label}_ORDINAL_AUDIT_REPORT.md
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path


# ---------------- statistics helpers ----------------

def spearman_rank(seq):
    """Average-rank tie handling, 1-based ranks."""
    pairs = sorted(enumerate(seq), key=lambda p: p[1])
    ranks = [0.0] * len(seq)
    i = 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][1] == pairs[i][1]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[pairs[k][0]] = avg
        i = j + 1
    return ranks


def spearman(xs, ys):
    if len(xs) < 2 or len(set(xs)) < 2 or len(set(ys)) < 2:
        return None
    rx = spearman_rank(xs)
    ry = spearman_rank(ys)
    mx = statistics.fmean(rx)
    my = statistics.fmean(ry)
    sx = math.sqrt(sum((x - mx) ** 2 for x in rx))
    sy = math.sqrt(sum((y - my) ** 2 for y in ry))
    if sx == 0 or sy == 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(rx, ry)) / (sx * sy)


def auc_binary(scores, labels):
    """ROC AUC with label 1 positive, no sklearn dependency.

    Implements the Mann–Whitney U formulation with tie correction.
    Returns None if either class is empty.
    """
    n_pos = sum(1 for l in labels if l == 1)
    n_neg = sum(1 for l in labels if l == 0)
    if n_pos == 0 or n_neg == 0:
        return None
    # Average-rank of all scores
    ranks_of_scores = spearman_rank(scores)
    pos_rank_sum = sum(r for r, l in zip(ranks_of_scores, labels) if l == 1)
    return (pos_rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def shannon_entropy_bits(values):
    c = Counter(values)
    total = sum(c.values())
    if total == 0:
        return 0.0
    h = 0.0
    for v in c.values():
        p = v / total
        if p > 0:
            h -= p * math.log2(p)
    return h


def dominant_mass(values):
    c = Counter(values)
    total = sum(c.values())
    if total == 0:
        return 0.0
    return max(c.values()) / total


# ---------------- per-cell computation ----------------

ADJACENT_PAIRS = [(5, 6), (6, 7), (7, 8), (8, 9)]
AUC_THRESHOLDS = [6, 7, 8, 9]   # AUC(E -> 1[h >= t]) averaged into MACRO_AUC


def cell_metrics(rows, baseline_val_acc=None, summary_extra=None):
    """Compute the H7 metric set on a list of per-sample rows.

    rows = list of dicts with at least controller_step_expected, chosen_step,
    required_hops, oracle_step. All rows are assumed to be the same (seed,
    split) bucket.

    baseline_val_acc and summary_extra carry split-level fields
    (mixture_logits_acc, floor_mass_mean, final_mass_mean, floor_mass_max,
    final_mass_max) supplied by the optional summary.json, since the
    per-sample CSV does not record them.
    """
    if not rows:
        return None

    E = [float(r["controller_step_expected"]) for r in rows]
    chosen = [float(r["chosen_step"]) for r in rows]
    hops = [int(r["required_hops"]) for r in rows
            if r.get("required_hops") not in (None, "", "None")]
    if len(hops) != len(rows):
        return {"error": "missing required_hops on some rows", "n": len(rows)}
    oracle = [float(r["oracle_step"]) for r in rows]

    n = len(rows)

    # Core correlation metrics
    s_all = spearman(E, hops)
    mask_easy = [i for i, h in enumerate(hops) if h <= 8]
    E_easy = [E[i] for i in mask_easy]
    hops_easy = [hops[i] for i in mask_easy]
    s_easy = spearman(E_easy, hops_easy) if len(set(hops_easy)) > 1 else None
    s_oracle = spearman(oracle, hops)
    s_chosen = spearman(chosen, hops)

    # AUCs : AUC(E -> 1[h >= t]) for t in {6,7,8,9}
    auc_by_t = {}
    for t in AUC_THRESHOLDS:
        labels = [1 if h >= t else 0 for h in hops]
        auc_by_t[t] = auc_binary(E, labels)
    auc9 = auc_by_t[9]
    valid_aucs = [v for v in auc_by_t.values() if v is not None]
    macro_auc = statistics.fmean(valid_aucs) if valid_aucs else None

    # Bucket means + adjacent margins + spread
    bucket_means = {}
    for h in sorted(set(hops)):
        vs = [E[i] for i, hh in enumerate(hops) if hh == h]
        bucket_means[h] = statistics.fmean(vs)
    spread = (max(bucket_means.values()) - min(bucket_means.values())
              if bucket_means else None)
    adjacent_margins = {}
    for a, b in ADJACENT_PAIRS:
        if a in bucket_means and b in bucket_means:
            adjacent_margins[f"m_{a}{b}"] = bucket_means[b] - bucket_means[a]
        else:
            adjacent_margins[f"m_{a}{b}"] = None
    valid_margins = [m for m in adjacent_margins.values() if m is not None]
    adj_mean = statistics.fmean(valid_margins) if valid_margins else None
    adj_min = min(valid_margins) if valid_margins else None

    # Δ E[step] (h=9 − h≤8)
    E_h9 = [E[i] for i, h in enumerate(hops) if h == 9]
    E_h_le_8 = [E[i] for i, h in enumerate(hops) if h <= 8]
    delta_h9 = ((statistics.fmean(E_h9) - statistics.fmean(E_h_le_8))
                if E_h9 and E_h_le_8 else None)

    # chosen_step entropy + dominant bin
    chosen_entropy = shannon_entropy_bits(chosen)
    dom_mass = dominant_mass(chosen)
    chosen_distinct = len(set(chosen))

    # E[step] basic stats
    E_mean = statistics.fmean(E)
    E_std = statistics.pstdev(E) if len(E) > 1 else 0.0

    # Pull split-level numbers from summary_extra (mixture_logits_acc,
    # floor/final masses, etc.). These are NOT in the per-sample CSV.
    accuracy = (summary_extra or {}).get("mixture_logits_acc")
    floor_mean = (summary_extra or {}).get("floor_mass_mean")
    final_mean = (summary_extra or {}).get("final_mass_mean")
    # The per-sample CSV does not carry floor / final masses, so the "max"
    # variants are not computable here from the per-sample data. We mark
    # them None unless the summary supplies them.
    floor_max = (summary_extra or {}).get("floor_mass_max")
    final_max = (summary_extra or {}).get("final_mass_max")

    # ---- Collapse taxonomy ----
    flags = []
    notes = []

    # hard_floor / hard_final require the summary numbers.
    if floor_mean is not None and (floor_mean >= 0.5
                                   or (floor_max is not None and floor_max >= 0.8)):
        flags.append("hard_floor")
    if final_mean is not None and (final_mean >= 0.5
                                   or (final_max is not None and final_max >= 0.8)):
        flags.append("hard_final")

    # soft_middle_step : bucket_spread <= 0.2 AND chosen_entropy < 1.0
    # OR dominant_chosen_step_mass >= 0.8
    soft = False
    if (spread is not None and spread <= 0.20 and chosen_entropy < 1.0):
        soft = True
        notes.append(f"soft_middle_step:spread<=0.20+entropy<1.0 (spread={spread:.3f}, "
                     f"entropy={chosen_entropy:.3f})")
    if dom_mass >= 0.80:
        soft = True
        notes.append(f"soft_middle_step:dominant_mass>=0.80 (mass={dom_mass:.3f})")
    if soft:
        flags.append("soft_middle_step")

    # binary_h9_shortcut : AUC9 >= 0.95 AND |S_easy| <= 0.10
    if (auc9 is not None and auc9 >= 0.95
            and s_easy is not None and abs(s_easy) <= 0.10):
        flags.append("binary_h9_shortcut")

    # ordinal_healthy : none of the above + S_easy >= 0.15 + MACRO_AUC >= 0.70
    # + adjacent_margin_mean > 0 + adjacent_margin_min >= -0.10
    # + val_acc >= baseline_val_acc - 0.02 (when baseline supplied)
    healthy_reasons_failed = []
    if "hard_floor" in flags or "hard_final" in flags or "soft_middle_step" in flags:
        healthy_reasons_failed.append("collapse flag present")
    if s_easy is None or s_easy < 0.15:
        healthy_reasons_failed.append(f"S_easy<0.15 (got {s_easy})")
    if macro_auc is None or macro_auc < 0.70:
        healthy_reasons_failed.append(f"MACRO_AUC<0.70 (got {macro_auc})")
    if adj_mean is None or adj_mean <= 0:
        healthy_reasons_failed.append(f"adjacent_margin_mean<=0 (got {adj_mean})")
    if adj_min is None or adj_min < -0.10:
        healthy_reasons_failed.append(f"adjacent_margin_min<-0.10 (got {adj_min})")
    if (baseline_val_acc is not None and accuracy is not None
            and accuracy < baseline_val_acc - 0.02):
        healthy_reasons_failed.append(
            f"val_acc<baseline-0.02 (acc={accuracy}, baseline={baseline_val_acc})")
    if not healthy_reasons_failed:
        flags.append("ordinal_healthy")

    return {
        "n": n,
        "mixture_logits_acc": accuracy,
        # core correlations
        "S_all": s_all,
        "S_easy": s_easy,
        "S_chosen_vs_hops": s_chosen,
        "S_oracle_vs_hops": s_oracle,
        # AUCs
        "AUC9": auc9,
        "AUC_h_ge_6": auc_by_t[6],
        "AUC_h_ge_7": auc_by_t[7],
        "AUC_h_ge_8": auc_by_t[8],
        "AUC_h_ge_9": auc_by_t[9],
        "MACRO_AUC": macro_auc,
        # bucket geometry
        "bucket_means": bucket_means,
        "bucket_spread": spread,
        "delta_h9": delta_h9,
        **adjacent_margins,
        "adjacent_margin_mean": adj_mean,
        "adjacent_margin_min": adj_min,
        # chosen_step shape
        "chosen_step_entropy_bits": chosen_entropy,
        "dominant_chosen_step_mass": dom_mass,
        "chosen_step_distinct": chosen_distinct,
        # E[step] stats
        "E_step_mean": E_mean,
        "E_step_std": E_std,
        # masses (from summary, if available)
        "floor_mass_mean": floor_mean,
        "final_mass_mean": final_mean,
        "floor_mass_max": floor_max,
        "final_mass_max": final_max,
        # taxonomy
        "flags": flags,
        "notes": notes,
    }


# ---------------- driver ----------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True,
                   help="Per-sample audit CSV with columns "
                        "seed,split,sample_idx,controller_step_expected,"
                        "chosen_step,oracle_step,required_hops,oracle_depth")
    p.add_argument("--summary", default=None,
                   help="Optional summary.json (same campaign) to pull "
                        "mixture_logits_acc and floor/final masses from.")
    p.add_argument("--label", required=True,
                   help="Identifier prefix for the output files.")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--baseline-val-acc", type=float, default=None,
                   help="Reference val_acc for the ordinal_healthy gate. "
                        "Default : None (the val_acc threshold is skipped).")
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ---- load per-sample CSV ----
    rows = []
    with open(args.input) as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    seeds = sorted({int(r["seed"]) for r in rows})
    splits = sorted({r["split"] for r in rows})

    # ---- load summary.json if supplied ----
    summary_by_cell = {}
    if args.summary:
        s = json.loads(Path(args.summary).read_text())
        for r in s.get("per_seed_split", []):
            summary_by_cell[(int(r["seed"]), r["split"])] = {
                "mixture_logits_acc": r.get("mixture_logits_acc"),
                "floor_mass_mean": r.get("floor_mass_mean"),
                "final_mass_mean": r.get("final_mass_mean"),
                # floor_max / final_max are not in the existing schema ;
                # left None unless explicitly provided.
                "floor_mass_max": r.get("floor_mass_max"),
                "final_mass_max": r.get("final_mass_max"),
            }

    # ---- compute per (seed, split) cell ----
    cells = {}
    for seed in seeds:
        for split in splits:
            grp = [r for r in rows
                   if int(r["seed"]) == seed and r["split"] == split]
            extra = summary_by_cell.get((seed, split))
            cells[(seed, split)] = cell_metrics(grp, args.baseline_val_acc, extra)

    # ---- emit CSV (one row per cell, scalar columns only) ----
    csv_columns = [
        "label", "seed", "split", "n",
        "mixture_logits_acc",
        "S_all", "S_easy", "S_chosen_vs_hops", "S_oracle_vs_hops",
        "AUC_h_ge_6", "AUC_h_ge_7", "AUC_h_ge_8", "AUC_h_ge_9", "MACRO_AUC",
        "bucket_spread", "delta_h9",
        "m_56", "m_67", "m_78", "m_89",
        "adjacent_margin_mean", "adjacent_margin_min",
        "chosen_step_entropy_bits", "dominant_chosen_step_mass",
        "chosen_step_distinct",
        "E_step_mean", "E_step_std",
        "floor_mass_mean", "final_mass_mean",
        "floor_mass_max", "final_mass_max",
        "flag_hard_floor", "flag_hard_final",
        "flag_soft_middle_step", "flag_binary_h9_shortcut",
        "flag_ordinal_healthy",
        "flags_concat",
    ]
    csv_path = out / f"{args.label}_ordinal_metrics_per_seed_split.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=csv_columns)
        w.writeheader()
        for (seed, split), m in sorted(cells.items()):
            flags = m.get("flags", [])
            row = {
                "label": args.label, "seed": seed, "split": split,
                "n": m.get("n"),
                "mixture_logits_acc": m.get("mixture_logits_acc"),
                "S_all": m.get("S_all"),
                "S_easy": m.get("S_easy"),
                "S_chosen_vs_hops": m.get("S_chosen_vs_hops"),
                "S_oracle_vs_hops": m.get("S_oracle_vs_hops"),
                "AUC_h_ge_6": m.get("AUC_h_ge_6"),
                "AUC_h_ge_7": m.get("AUC_h_ge_7"),
                "AUC_h_ge_8": m.get("AUC_h_ge_8"),
                "AUC_h_ge_9": m.get("AUC_h_ge_9"),
                "MACRO_AUC": m.get("MACRO_AUC"),
                "bucket_spread": m.get("bucket_spread"),
                "delta_h9": m.get("delta_h9"),
                "m_56": m.get("m_56"), "m_67": m.get("m_67"),
                "m_78": m.get("m_78"), "m_89": m.get("m_89"),
                "adjacent_margin_mean": m.get("adjacent_margin_mean"),
                "adjacent_margin_min": m.get("adjacent_margin_min"),
                "chosen_step_entropy_bits": m.get("chosen_step_entropy_bits"),
                "dominant_chosen_step_mass": m.get("dominant_chosen_step_mass"),
                "chosen_step_distinct": m.get("chosen_step_distinct"),
                "E_step_mean": m.get("E_step_mean"),
                "E_step_std": m.get("E_step_std"),
                "floor_mass_mean": m.get("floor_mass_mean"),
                "final_mass_mean": m.get("final_mass_mean"),
                "floor_mass_max": m.get("floor_mass_max"),
                "final_mass_max": m.get("final_mass_max"),
                "flag_hard_floor": "hard_floor" in flags,
                "flag_hard_final": "hard_final" in flags,
                "flag_soft_middle_step": "soft_middle_step" in flags,
                "flag_binary_h9_shortcut": "binary_h9_shortcut" in flags,
                "flag_ordinal_healthy": "ordinal_healthy" in flags,
                "flags_concat": "|".join(flags),
            }
            w.writerow(row)

    # ---- emit JSON (full data, including bucket_means dicts) ----
    json_payload = {
        "label": args.label,
        "input_csv": args.input,
        "summary_json": args.summary,
        "baseline_val_acc": args.baseline_val_acc,
        "per_cell": [
            {
                "seed": seed, "split": split,
                **m,
            }
            for (seed, split), m in sorted(cells.items())
        ],
    }
    json_path = out / f"{args.label}_ordinal_metrics_summary.json"
    json_path.write_text(json.dumps(json_payload, indent=2, default=str))

    # ---- emit Markdown summary ----
    md = []
    md.append(f"# Ordinal audit — `{args.label}`")
    md.append("")
    md.append(f"- input per-sample CSV : `{args.input}`")
    md.append(f"- input summary.json    : `{args.summary or '(none)'}`")
    md.append(f"- baseline_val_acc      : `{args.baseline_val_acc}`")
    md.append("")
    md.append("## Per (seed × split) cells")
    md.append("")
    md.append("| seed | split | n | acc | S_all | S_easy | AUC9 | MACRO_AUC "
              "| spread | adj_mean | entropy | dom_mass | flags |")
    md.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for (seed, split), m in sorted(cells.items()):
        def fmt(v, prec=3):
            if v is None:
                return "—"
            return f"{v:+.{prec}f}" if isinstance(v, float) else str(v)
        md.append(
            f"| {seed} | {split} | {m['n']} | "
            f"{m.get('mixture_logits_acc'):.4f} "
            if m.get("mixture_logits_acc") is not None
            else f"| {seed} | {split} | {m['n']} | — "
        )
    # Re-do the table cleanly (the lambda above broke on None acc)
    md = md[:-len(cells)]
    for (seed, split), m in sorted(cells.items()):
        def f3(v):
            return "—" if v is None else f"{v:+.3f}"
        def f4(v):
            return "—" if v is None else f"{v:.4f}"
        acc = m.get("mixture_logits_acc")
        md.append(
            f"| {seed} | {split} | {m['n']} | "
            f"{f4(acc)} | "
            f"{f3(m.get('S_all'))} | {f3(m.get('S_easy'))} | "
            f"{f3(m.get('AUC9'))} | {f3(m.get('MACRO_AUC'))} | "
            f"{f3(m.get('bucket_spread'))} | "
            f"{f3(m.get('adjacent_margin_mean'))} | "
            f"{f3(m.get('chosen_step_entropy_bits'))} | "
            f"{f3(m.get('dominant_chosen_step_mass'))} | "
            f"{'|'.join(m.get('flags', [])) or '—'} |"
        )
    md.append("")
    md.append("## Per-cell bucket means (E[step] by required_hops)")
    md.append("")
    md.append("| seed | split | h=5 | h=6 | h=7 | h=8 | h=9 |")
    md.append("|---:|---|---:|---:|---:|---:|---:|")
    for (seed, split), m in sorted(cells.items()):
        bm = m.get("bucket_means", {})
        def fb(h):
            return "—" if h not in bm else f"{bm[h]:.2f}"
        md.append(f"| {seed} | {split} | {fb(5)} | {fb(6)} | {fb(7)} | {fb(8)} | {fb(9)} |")
    md.append("")
    md.append("## Per-cell notes")
    md.append("")
    for (seed, split), m in sorted(cells.items()):
        if m.get("notes"):
            md.append(f"- seed{seed} {split}: {'; '.join(m['notes'])}")
    md_path = out / f"{args.label}_ORDINAL_AUDIT_REPORT.md"
    md_path.write_text("\n".join(md) + "\n")

    print(f"wrote {csv_path}")
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
