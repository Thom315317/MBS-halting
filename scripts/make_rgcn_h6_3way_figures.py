#!/usr/bin/env python
"""Three-way comparison figures (MBS H6dau, RGCN ACT, RGCN+H6) for the
paper integration.

Inputs :
  - results/final_scientific_package/aggregates/h6_detached_aux_summary.json
    (MBS H6dau per-seed acc + val floor/final)
  - results/claim_strengthening/controller_required_hops_summary.json
    (MBS H6dau per-seed Spearman vs required_hops + bucket data)
  - results/claim_strengthening/rgcn_act_postpatch_summary.json
    (RGCN ACT post-patch per-seed all metrics)
  - results/claim_strengthening/rgcn_h6_two_stage/rgcn_h6_two_stage_summary.json
    (RGCN+H6 per-seed all metrics, plus per-bucket per-seed)
  - results/claim_strengthening/data_audit/h6_required_hops_bucket_summary.csv
    (MBS H6dau per-bucket per-seed)

Outputs (PNG + PDF each) in results/claim_strengthening/paper_update/ :
  - fig_acc_vs_policy_3way
  - fig_collapse_modes_3way
  - fig_bucket_alignment_mbs_vs_rgcn_h6
"""
import csv
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO = Path("/home/thom315/MBS-halting")
OUT = REPO / "results" / "claim_strengthening" / "paper_update"
OUT.mkdir(parents=True, exist_ok=True)

# ---------------- Load all data ----------------

# MBS H6dau per-seed acc and val floor/final (from final_scientific_package).
mbs_aggr = json.loads(
    (REPO / "results/final_scientific_package/aggregates/h6_detached_aux_summary.json").read_text()
)
mbs_per_seed_acc = {r["seed"]: r for r in mbs_aggr["per_seed"]}

# MBS H6dau per-seed Spearman vs required_hops.
mbs_sp = json.loads(
    (REPO / "results/claim_strengthening/controller_required_hops_summary.json").read_text()
)
mbs_per_split = {(r["seed"], r["split"]): r for r in mbs_sp["per_seed_split"]}

# RGCN ACT post-patch per-seed all metrics.
rgcn_act = json.loads(
    (REPO / "results/claim_strengthening/rgcn_act_postpatch_summary.json").read_text()
)
rgcn_act_per_split = {(r["seed"], r["split"]): r for r in rgcn_act["per_seed_split"]}

# RGCN+H6 per-seed all metrics + bucket data.
rgcn_h6 = json.loads(
    (REPO / "results/claim_strengthening/rgcn_h6_two_stage/rgcn_h6_two_stage_summary.json").read_text()
)
rgcn_h6_per_split = {(r["seed"], r["split"]): r for r in rgcn_h6["per_seed_split"]}
rgcn_h6_buckets_xseed = rgcn_h6["cross_seed_buckets"]  # list of dicts

# MBS H6dau per-bucket per-seed (for figure 3).
mbs_bucket_rows = []
with (REPO / "results/claim_strengthening/data_audit/h6_required_hops_bucket_summary.csv").open() as f:
    reader = csv.DictReader(f)
    for row in reader:
        mbs_bucket_rows.append({
            "seed": int(row["seed"]),
            "split": row["split"],
            "required_hops": int(row["required_hops"]),
            "n": int(row["n"]),
            "exp_mean": float(row["exp_mean"]),
        })


SEEDS = [1, 2, 3, 4, 5]


# ---------------- Figure 1 : acc vs policy alignment, OOD, 3-way ----------------

def fig_acc_vs_policy():
    fig, ax = plt.subplots(figsize=(7.0, 5.0))

    mbs_acc, mbs_sp_v, mbs_seed = [], [], []
    rgcn_act_acc, rgcn_act_sp_v, rgcn_act_seed = [], [], []
    rgcn_h6_acc, rgcn_h6_sp_v, rgcn_h6_seed = [], [], []

    for s in SEEDS:
        # MBS H6dau ood acc from aggr (ood_acc field), ood Spearman vs hops from controller_required_hops.
        mbs_acc.append(mbs_per_seed_acc[s]["ood_acc"])
        mbs_sp_v.append(mbs_per_split[(s, "ood_mixed")]["spearman_expected_vs_required_hops"])
        mbs_seed.append(s)
        # RGCN ACT
        r = rgcn_act_per_split[(s, "ood_mixed")]
        rgcn_act_acc.append(r["mixture_logits_acc"])
        rgcn_act_sp_v.append(r["spearman_expected_vs_required_hops"])
        rgcn_act_seed.append(s)
        # RGCN+H6
        r = rgcn_h6_per_split[(s, "ood_mixed")]
        rgcn_h6_acc.append(r["mixture_logits_acc"])
        rgcn_h6_sp_v.append(r["spearman_expected_vs_required_hops"])
        rgcn_h6_seed.append(s)

    ax.scatter(mbs_acc, mbs_sp_v, marker="o", s=80, c="#1f77b4",
               edgecolors="black", linewidths=0.6,
               label="MBS H6_detached_aux (5 seeds)", zorder=3)
    ax.scatter(rgcn_act_acc, rgcn_act_sp_v, marker="s", s=80, c="#d62728",
               edgecolors="black", linewidths=0.6,
               label="RGCN ACT post-patch (5 seeds)", zorder=3)
    ax.scatter(rgcn_h6_acc, rgcn_h6_sp_v, marker="^", s=90, c="#2ca02c",
               edgecolors="black", linewidths=0.6,
               label="RGCN + H6 two-stage (5 seeds, this work)", zorder=3)

    # Annotate seed numbers as small labels next to each RGCN+H6 point.
    for x, y, s in zip(rgcn_h6_acc, rgcn_h6_sp_v, rgcn_h6_seed):
        ax.annotate(f"s{s}", (x, y), xytext=(6, -3), textcoords="offset points",
                    fontsize=8, color="#1f5510")

    # Highlight the RGCN+H6 seed-3 outlier with a dashed circle.
    s3_x = rgcn_h6_acc[rgcn_h6_seed.index(3)]
    s3_y = rgcn_h6_sp_v[rgcn_h6_seed.index(3)]
    ax.scatter([s3_x], [s3_y], marker="o", s=350, facecolors="none",
               edgecolors="#d6a200", linewidths=2.0, linestyle="--",
               label="RGCN+H6 seed 3 (alignment outlier)", zorder=2)

    ax.axhline(0.0, color="grey", linestyle=":", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("OOD mixed accuracy")
    ax.set_ylabel("Spearman(expected_step, required_hops) — OOD")
    ax.set_xlim(0.78, 0.93)
    ax.set_ylim(-0.08, 0.85)
    ax.set_title("Accuracy vs halting policy alignment (5 seeds × 3 configurations)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower left", fontsize=8.5, framealpha=0.92)

    fig.text(0.5, 0.005,
             ("RGCN ACT reaches high accuracy but collapses to zero policy alignment ; "
              "RGCN+H6 preserves similar accuracy and recovers bucket-aligned halting in 4/5 seeds."),
             ha="center", fontsize=8.0, style="italic", wrap=True)

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(OUT / "fig_acc_vs_policy_3way.png", dpi=180, bbox_inches="tight")
    fig.savefig(OUT / "fig_acc_vs_policy_3way.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT / 'fig_acc_vs_policy_3way.png'}")
    print(f"wrote {OUT / 'fig_acc_vs_policy_3way.pdf'}")


# ---------------- Figure 2 : collapse modes, 3-way ----------------

def fig_collapse_modes():
    """Per-seed floor / final mass for the three configurations, on val.
    Threshold 0.5 marked in red. MBS H6dau has only val masses available
    in its aggregate ; we show val for all three for consistency."""
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.2), sharey=True)

    configs = [
        ("MBS H6_detached_aux",
         [(s, mbs_per_seed_acc[s]["val_floor_mass"], mbs_per_seed_acc[s]["val_final_mass"]) for s in SEEDS],
         "#1f77b4"),
        ("RGCN ACT post-patch",
         [(s, rgcn_act_per_split[(s, "val")]["floor_mass_mean"],
              rgcn_act_per_split[(s, "val")]["final_mass_mean"]) for s in SEEDS],
         "#d62728"),
        ("RGCN + H6 two-stage",
         [(s, rgcn_h6_per_split[(s, "val")]["floor_mass_mean"],
              rgcn_h6_per_split[(s, "val")]["final_mass_mean"]) for s in SEEDS],
         "#2ca02c"),
    ]

    x_pos = np.arange(len(SEEDS))
    width = 0.38

    for ax, (name, rows, color) in zip(axes, configs):
        seeds = [r[0] for r in rows]
        floor = [r[1] for r in rows]
        final = [r[2] for r in rows]
        ax.bar(x_pos - width / 2, floor, width, label="floor_mass (val)",
               color=color, alpha=0.85, edgecolor="black", linewidth=0.6)
        ax.bar(x_pos + width / 2, final, width, label="final_mass (val)",
               color=color, alpha=0.45, edgecolor="black", linewidth=0.6, hatch="//")
        ax.axhline(0.5, color="red", linestyle="--", linewidth=1.0, alpha=0.8,
                   label=("collapse threshold (0.5)" if ax is axes[0] else None))
        n_collapse = sum(1 for f, ff in zip(floor, final) if f >= 0.5 or ff >= 0.5)
        ax.set_title(f"{name}\n{n_collapse}/5 collapse", fontsize=10)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f"s{s}" for s in seeds], fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.grid(True, axis="y", alpha=0.25)
        if ax is axes[0]:
            ax.set_ylabel("halt weight mass (val)")
            ax.legend(loc="upper right", fontsize=8)

    fig.suptitle("Collapse counts per configuration (5 seeds each, val split)",
                 fontsize=11.5)
    fig.text(0.5, 0.005,
             ("MBS+H6 : 0/5 collapse. RGCN ACT : 5/5 collapse (mostly floor, some final). "
              "RGCN+H6 : 0/5 collapse — the protocol fixes the collapse mode on the same backbone."),
             ha="center", fontsize=8.5, style="italic", wrap=True)

    plt.tight_layout(rect=[0, 0.045, 1, 0.94])
    fig.savefig(OUT / "fig_collapse_modes_3way.png", dpi=180, bbox_inches="tight")
    fig.savefig(OUT / "fig_collapse_modes_3way.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT / 'fig_collapse_modes_3way.png'}")
    print(f"wrote {OUT / 'fig_collapse_modes_3way.pdf'}")


# ---------------- Figure 3 : bucket alignment MBS vs RGCN+H6 ----------------

def fig_bucket_alignment():
    """Cross-seed mean of per-seed bucket means, for each required_hops in
    {5, 6, 7, 8, 9}. Both val and OOD shown side-by-side."""
    buckets = [5, 6, 7, 8, 9]

    # MBS H6dau cross-seed per-bucket mean of `exp_mean` from the bucket CSV.
    mbs_means = {}  # (split, hops) -> (mean_of_seed_means, stdev_of_seed_means)
    for split in ("val", "ood_mixed"):
        for h in buckets:
            vals = [r["exp_mean"] for r in mbs_bucket_rows
                    if r["split"] == split and r["required_hops"] == h]
            mbs_means[(split, h)] = (
                float(np.mean(vals)) if vals else float("nan"),
                float(np.std(vals)) if len(vals) > 1 else 0.0,
            )

    # RGCN+H6 cross-seed means already aggregated in the summary.
    rgcn_h6_means = {}
    for r in rgcn_h6_buckets_xseed:
        rgcn_h6_means[(r["split"], r["required_hops"])] = (
            r["expected_step_mean_of_seed_means"],
            r["expected_step_stdev_of_seed_means"],
        )

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3), sharey=True)
    split_titles = {"val": "val split", "ood_mixed": "ood_mixed split"}

    for ax, split in zip(axes, ("val", "ood_mixed")):
        mbs_y = [mbs_means[(split, h)][0] for h in buckets]
        mbs_e = [mbs_means[(split, h)][1] for h in buckets]
        h6_y = [rgcn_h6_means[(split, h)][0] for h in buckets]
        h6_e = [rgcn_h6_means[(split, h)][1] for h in buckets]

        ax.errorbar(buckets, mbs_y, yerr=mbs_e, marker="o", color="#1f77b4",
                    label="MBS H6_detached_aux (5 seeds)",
                    capsize=3, linewidth=1.4, markersize=7)
        ax.errorbar(buckets, h6_y, yerr=h6_e, marker="^", color="#2ca02c",
                    label="RGCN + H6 two-stage (5 seeds)",
                    capsize=3, linewidth=1.4, markersize=8)

        ax.axvline(8.5, color="grey", linestyle=":", linewidth=0.8, alpha=0.6)
        ax.set_xticks(buckets)
        ax.set_xlabel("required_hops bucket")
        ax.set_title(split_titles[split], fontsize=10.5)
        ax.grid(True, alpha=0.25)
        if ax is axes[0]:
            ax.set_ylabel("controller expected halting step")
            ax.legend(loc="upper left", fontsize=8.5)

    fig.suptitle("Bucket alignment of the controller expected step "
                 "(cross-seed mean ± stdev of seed means)", fontsize=11.5)
    fig.text(0.5, 0.005,
             ("Both MBS+H6 and RGCN+H6 learn hardest-bucket detection, "
              "not fine-grained continuous depth tracking. The jump at h=9 is smaller on RGCN+H6."),
             ha="center", fontsize=8.5, style="italic", wrap=True)

    plt.tight_layout(rect=[0, 0.04, 1, 0.94])
    fig.savefig(OUT / "fig_bucket_alignment_mbs_vs_rgcn_h6.png", dpi=180, bbox_inches="tight")
    fig.savefig(OUT / "fig_bucket_alignment_mbs_vs_rgcn_h6.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT / 'fig_bucket_alignment_mbs_vs_rgcn_h6.png'}")
    print(f"wrote {OUT / 'fig_bucket_alignment_mbs_vs_rgcn_h6.pdf'}")


if __name__ == "__main__":
    fig_acc_vs_policy()
    fig_collapse_modes()
    fig_bucket_alignment()
    print("done.")
