#!/usr/bin/env python
"""Audit for RGCN + H6_detached_aux two-stage runs.

For each seed under
`results/claim_strengthening/rgcn_h6_two_stage/seed{N}/stage2/`, evaluate
the selected (val_acc) Stage 2 checkpoint on val + ood_mixed and report :

  - Spearman(expected_controller_step, oracle_step)
  - Spearman(expected_controller_step, required_hops)
  - Spearman(chosen_controller_step, required_hops)
  - Spearman(oracle_step, required_hops)
  - val/ood mixture_logits_acc, expected_halted_acc, chosen_step_acc,
    chosen_step_mean, floor_mass, final_mass.

required_hops is reconstructed the same way as in the H6_detached_aux audit.

Outputs in --output-dir :
  rgcn_h6_two_stage_per_seed.csv
  rgcn_h6_two_stage_summary.json
"""
from pathlib import Path
import argparse, csv, datetime as dt, json, math, statistics, sys

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mbs.datasets import build_belief_repair_datasets, VALUES
from mbs.graph import collate_graph_samples, CELL_TYPES
from mbs.tokenizer import SimpleTokenizer
from mbs.train import build_model, aggregate_value_logits
from mbs.utils import load_config, set_seed


REPO = Path("/home/thom315/MBS-halting")


def _spearman(xs, ys):
    if len(xs) < 2:
        return 0.0

    def _rank(seq):
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

    rx, ry = _rank(xs), _rank(ys)
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    sx = math.sqrt(sum((x - mx) ** 2 for x in rx))
    sy = math.sqrt(sum((y - my) ** 2 for y in ry))
    if sx == 0 or sy == 0:
        return 0.0
    return sum((x - mx) * (y - my) for x, y in zip(rx, ry)) / (sx * sy)


def _derive_required_hops_v1(meta):
    crs = meta.get("candidate_ranks_used")
    wr = meta.get("winner_rank", 1)
    if crs is None or wr is None:
        return None
    return (max(crs) - int(wr)) + 2


@torch.no_grad()
def _evaluate_seed(checkpoint_path, config_path, seed_id, device, oracle_lambda):
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    ckpt_config = dict(payload.get("config") or {})
    tok = (SimpleTokenizer.from_state_dict(payload["tokenizer"])
           if "tokenizer" in payload else SimpleTokenizer())
    model = build_model(payload["variant"], ckpt_config, tok)
    model.load_state_dict(payload["model_state"], strict=False)
    model = model.to(device).eval()

    eval_cfg = load_config(config_path); eval_cfg["seed"] = seed_id
    set_seed(seed_id)
    ds = build_belief_repair_datasets(eval_cfg, tok)
    run_cfg = dict(ckpt_config)

    rows = []
    splits_metrics = {}
    for split in ("val", "ood_mixed"):
        samples = list(ds[split])
        B = int(run_cfg.get("batch_size", 16))
        n_total = 0; mix_corr = 0
        exp_halted_sum = 0.0; chosen_acc_sum = 0.0
        floor_sum = 0.0; final_sum = 0.0
        for start in range(0, len(samples), B):
            chunk = samples[start:start + B]
            batch = collate_graph_samples(chunk, tok)
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            out = model(batch)
            target = batch["target_value_id"]
            scores_per_step = out["claim_scores_per_step"]
            halt_weights = out["halt_weights"]
            is_claim = (batch["cell_type_ids"] == CELL_TYPES["CLAIM"]) & batch["node_mask"].bool()
            is_qc = batch["is_query_claim_node"].bool()
            cvi = batch["claim_value_ids"].long()
            agg_mask = is_claim & is_qc & (cvi >= 0)
            T = halt_weights.size(1)
            ce_per_step = []; per_step_correct = []
            weighted_logits = torch.zeros(halt_weights.size(0), len(VALUES), device=device, dtype=torch.float32)
            for t in range(T):
                vl = aggregate_value_logits(scores_per_step[t], cvi, agg_mask, len(VALUES))
                ce_t = F.cross_entropy(vl, target, reduction="none").cpu()
                ce_per_step.append(ce_t)
                weighted_logits = weighted_logits + halt_weights[:, t].float().unsqueeze(-1) * vl
                per_step_correct.append((vl.argmax(dim=-1) == target).float().cpu())
            ce = torch.stack(ce_per_step, dim=1)
            correct_stack = torch.stack(per_step_correct, dim=1)
            hw = halt_weights.cpu()
            t_idx = torch.arange(1, T + 1, dtype=torch.float32).unsqueeze(0)
            controller_step_expected = (hw * t_idx).sum(dim=1)
            chosen_idx = hw.argmax(dim=1)
            chosen_step = chosen_idx.float() + 1.0
            chosen_correct = correct_stack.gather(1, chosen_idx.unsqueeze(1)).squeeze(1)
            exp_halted = (hw * correct_stack).sum(dim=1)
            oracle_cost = ce + oracle_lambda * t_idx
            oracle_step = (oracle_cost.argmin(dim=1) + 1).float()
            mix_pred = weighted_logits.argmax(dim=-1)
            mix_corr += (mix_pred == target).sum().item()
            n_total += target.size(0)
            exp_halted_sum += float(exp_halted.sum().item())
            chosen_acc_sum += float(chosen_correct.sum().item())
            min_s = int((run_cfg.get("halting") or {}).get("min_message_steps", 4))
            floor_sum += float(hw[:, max(min_s - 1, 0)].sum().item())
            final_sum += float(hw[:, T - 1].sum().item())
            for j, s in enumerate(chunk):
                hops = _derive_required_hops_v1(s["metadata"])
                rows.append({
                    "seed": seed_id, "split": split,
                    "sample_idx": start + j,
                    "controller_step_expected": float(controller_step_expected[j].item()),
                    "chosen_step": float(chosen_step[j].item()),
                    "oracle_step": float(oracle_step[j].item()),
                    "required_hops": (int(hops) if hops is not None else None),
                    "oracle_depth": int(s["metadata"].get("oracle_depth", -1)),
                })
        splits_metrics[split] = {
            "n": n_total,
            "mixture_logits_acc": mix_corr / max(n_total, 1),
            "expected_halted_acc": exp_halted_sum / max(n_total, 1),
            "chosen_step_acc": chosen_acc_sum / max(n_total, 1),
            "floor_mass_mean": floor_sum / max(n_total, 1),
            "final_mass_mean": final_sum / max(n_total, 1),
        }
    return rows, splits_metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", required=True)
    p.add_argument("--oracle-lambda", type=float, default=0.01)
    p.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    p.add_argument("--device", default=None)
    args = p.parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    all_rows = []
    splits_metrics_per_seed = {}
    for seed in args.seeds:
        ckpt = REPO / "results" / "claim_strengthening" / "rgcn_h6_two_stage" / f"seed{seed}" / "stage2" / "checkpoints" / "rgcn_h6_two_stage_best.pt"
        cfg = REPO / "configs" / f"rgcn_h6_stage2_seed{seed}.yaml"
        if not ckpt.exists() or not cfg.exists():
            print(f"SKIP seed{seed}: missing artefacts (ckpt={ckpt.exists()}, cfg={cfg.exists()})")
            continue
        rows, met = _evaluate_seed(str(ckpt), str(cfg), seed, device, args.oracle_lambda)
        all_rows += rows
        splits_metrics_per_seed[seed] = met
        print(f"  seed{seed} done: val_mix={met['val']['mixture_logits_acc']:.4f} "
              f"ood_mix={met['ood_mixed']['mixture_logits_acc']:.4f} "
              f"val_floor={met['val']['floor_mass_mean']:.3f} "
              f"val_final={met['val']['final_mass_mean']:.3f}")

    cols = ["seed", "split", "sample_idx", "controller_step_expected",
            "chosen_step", "oracle_step", "required_hops", "oracle_depth"]
    with (out / "rgcn_h6_two_stage_per_seed.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in all_rows: w.writerow(r)

    by_keys = {}
    for r in all_rows:
        by_keys.setdefault((r["seed"], r["split"]), []).append(r)
    summary_per = []
    bucket_rows = []
    for (seed, split), grp in sorted(by_keys.items()):
        cs = [g["controller_step_expected"] for g in grp]
        ch = [g["chosen_step"] for g in grp]
        os_ = [g["oracle_step"] for g in grp]
        rh = [g["required_hops"] for g in grp if g["required_hops"] is not None]
        if len(rh) != len(grp):
            continue
        rh_f = [float(h) for h in rh]
        chosen_distinct = len(set(ch))
        # Variance of chosen_step (pop variance since we evaluate on the
        # whole eval set, not a sample of it).
        chosen_var = statistics.pvariance(ch) if len(ch) > 1 else 0.0
        # Per-required_hops bucket means within this (seed, split).
        by_bucket = {}
        for g in grp:
            by_bucket.setdefault(g["required_hops"], []).append(g)
        for hops in sorted(by_bucket):
            bg = by_bucket[hops]
            bucket_rows.append({
                "seed": seed, "split": split, "required_hops": hops,
                "n": len(bg),
                "chosen_step_mean": statistics.fmean(g["chosen_step"] for g in bg),
                "chosen_step_std": (statistics.pstdev(g["chosen_step"] for g in bg)
                                    if len(bg) > 1 else 0.0),
                "controller_step_expected_mean":
                    statistics.fmean(g["controller_step_expected"] for g in bg),
                "controller_step_expected_std":
                    (statistics.pstdev(g["controller_step_expected"] for g in bg)
                     if len(bg) > 1 else 0.0),
            })
        # Collapse mode verdict (matches the convention used in the
        # data_audit phase 2 H6 vs RGCN audit: threshold 0.5 on either
        # floor_mass or final_mass).
        floor_m = splits_metrics_per_seed[seed][split]["floor_mass_mean"]
        final_m = splits_metrics_per_seed[seed][split]["final_mass_mean"]
        if floor_m >= 0.5:
            collapse_mode = "floor"
        elif final_m >= 0.5:
            collapse_mode = "final"
        else:
            collapse_mode = "none"
        summary_per.append({
            "seed": seed, "split": split, "n": len(grp),
            "spearman_expected_vs_oracle": _spearman(cs, os_),
            "spearman_chosen_vs_oracle": _spearman(ch, os_),
            "spearman_expected_vs_required_hops": _spearman(cs, rh_f),
            "spearman_chosen_vs_required_hops": _spearman(ch, rh_f),
            "spearman_oracle_vs_required_hops": _spearman(os_, rh_f),
            "mixture_logits_acc": splits_metrics_per_seed[seed][split]["mixture_logits_acc"],
            "expected_halted_acc": splits_metrics_per_seed[seed][split]["expected_halted_acc"],
            "chosen_step_acc": splits_metrics_per_seed[seed][split]["chosen_step_acc"],
            "floor_mass_mean": floor_m,
            "final_mass_mean": final_m,
            "collapse_mode": collapse_mode,
            "controller_step_expected_mean": statistics.fmean(cs),
            "chosen_step_mean": statistics.fmean(ch),
            "chosen_step_var": chosen_var,
            "chosen_step_distinct": chosen_distinct,
        })
        print(f"  {seed} {split:>9s}  sρ(exp,hops)={summary_per[-1]['spearman_expected_vs_required_hops']:+.3f}  "
              f"sρ(chosen,hops)={summary_per[-1]['spearman_chosen_vs_required_hops']:+.3f}  "
              f"E[s]={summary_per[-1]['controller_step_expected_mean']:.2f}  "
              f"chosen_distinct={chosen_distinct}  var={chosen_var:.3f}  "
              f"floor={summary_per[-1]['floor_mass_mean']:.3f}  "
              f"final={summary_per[-1]['final_mass_mean']:.3f}  "
              f"collapse={collapse_mode}")

    def stats_block(values):
        values = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
        if not values:
            return {"n": 0, "mean": None}
        return {"n": len(values),
                "mean": statistics.fmean(values),
                "stdev": statistics.pstdev(values) if len(values) > 1 else 0.0,
                "min": min(values), "max": max(values)}

    metric_set = ("spearman_expected_vs_oracle",
                  "spearman_chosen_vs_oracle",
                  "spearman_expected_vs_required_hops",
                  "spearman_chosen_vs_required_hops",
                  "spearman_oracle_vs_required_hops",
                  "mixture_logits_acc", "expected_halted_acc", "chosen_step_acc",
                  "floor_mass_mean", "final_mass_mean",
                  "controller_step_expected_mean", "chosen_step_mean",
                  "chosen_step_var", "chosen_step_distinct")
    cross_seed = {}
    for split in ("val", "ood_mixed"):
        block = [s for s in summary_per if s["split"] == split]
        cross_seed[split] = {m: stats_block([s[m] for s in block]) for m in metric_set}

    # Per-bucket cross-seed summary (chosen_step and expected_step means by
    # required_hops, aggregated over seeds for each split).
    bucket_aggr = {}
    for r in bucket_rows:
        key = (r["split"], r["required_hops"])
        bucket_aggr.setdefault(key, []).append(r)
    cross_seed_buckets = []
    for (split, hops), grp in sorted(bucket_aggr.items()):
        cross_seed_buckets.append({
            "split": split, "required_hops": hops,
            "n_seeds": len(grp),
            "n_samples_total": sum(g["n"] for g in grp),
            "chosen_step_mean_of_seed_means":
                statistics.fmean(g["chosen_step_mean"] for g in grp),
            "chosen_step_stdev_of_seed_means":
                (statistics.pstdev(g["chosen_step_mean"] for g in grp)
                 if len(grp) > 1 else 0.0),
            "expected_step_mean_of_seed_means":
                statistics.fmean(g["controller_step_expected_mean"] for g in grp),
            "expected_step_stdev_of_seed_means":
                (statistics.pstdev(g["controller_step_expected_mean"] for g in grp)
                 if len(grp) > 1 else 0.0),
        })

    # Bucket-row CSV
    bcols = ["seed", "split", "required_hops", "n",
             "chosen_step_mean", "chosen_step_std",
             "controller_step_expected_mean", "controller_step_expected_std"]
    with (out / "rgcn_h6_two_stage_bucket_rows.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=bcols); w.writeheader()
        for r in bucket_rows: w.writerow(r)

    final = {
        "date": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "n_seeds": len(args.seeds),
        "per_seed_split": summary_per,
        "cross_seed_means": cross_seed,
        "per_bucket_per_seed": bucket_rows,
        "cross_seed_buckets": cross_seed_buckets,
        "oracle_lambda": args.oracle_lambda,
    }
    (out / "rgcn_h6_two_stage_summary.json").write_text(json.dumps(final, indent=2, default=str))
    print(f"wrote {out / 'rgcn_h6_two_stage_per_seed.csv'}")
    print(f"wrote {out / 'rgcn_h6_two_stage_bucket_rows.csv'}")
    print(f"wrote {out / 'rgcn_h6_two_stage_summary.json'}")


if __name__ == "__main__":
    main()
