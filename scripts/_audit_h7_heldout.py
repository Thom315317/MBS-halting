"""Audit each held-out H7-fixed seed's selected checkpoint.

For each seed in {1, 2, 4, 5}:
  1. Load `seedN_w001/checkpoints/rgcn_h7_two_stage_best.pt`.
  2. Run a per-sample forward on val + ood_mixed splits.
  3. Write `seedN_w001/v3_seedN_w001_per_seed.csv` and `..._summary.json`.

Then invoke `scripts/audit_halting_ordinal_metrics.py` per seed to produce
the ordinal-metric flag classification.
"""
import csv
import json
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path("/home/thom315/MBS-halting-h7")
sys.path.insert(0, str(REPO))

from mbs.datasets import build_belief_repair_datasets, VALUES
from mbs.graph import collate_graph_samples, CELL_TYPES
from mbs.tokenizer import SimpleTokenizer
from mbs.train import build_model, aggregate_value_logits
from mbs.utils import load_config, set_seed


def _derive_required_hops_v1(meta):
    crs = meta.get("candidate_ranks_used")
    wr = meta.get("winner_rank", 1)
    if crs is None or wr is None:
        return None
    return (max(crs) - int(wr)) + 2


@torch.no_grad()
def audit_seed(seed):
    ckpt_path = REPO / f"results/claim_strengthening/h7_ordinal_halting/seed{seed}_w001/checkpoints/rgcn_h7_two_stage_best.pt"
    cfg_path = REPO / f"configs/h7_ordinal_halting/rgcn_h7_seed{seed}_w001.yaml"
    out_dir = REPO / f"results/claim_strengthening/h7_ordinal_halting/seed{seed}_w001"

    print(f"=== audit seed {seed} ===")
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ckpt_config = dict(payload.get("config") or {})
    tok = (SimpleTokenizer.from_state_dict(payload["tokenizer"])
           if "tokenizer" in payload else SimpleTokenizer())
    model = build_model(payload["variant"], ckpt_config, tok)
    model.load_state_dict(payload["model_state"], strict=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    eval_cfg = load_config(str(cfg_path))
    eval_cfg["seed"] = seed
    set_seed(seed)
    ds = build_belief_repair_datasets(eval_cfg, tok)

    rows = []
    splits_metrics = {}
    for split in ("val", "ood_mixed"):
        samples = list(ds[split])
        B = int(eval_cfg.get("batch_size", 16))
        n_total = 0; mix_corr = 0
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
            per_step_correct = []
            weighted_logits = torch.zeros(halt_weights.size(0), len(VALUES),
                                          device=device, dtype=torch.float32)
            ce_per_step = []
            for t in range(T):
                vl = aggregate_value_logits(scores_per_step[t], cvi, agg_mask, len(VALUES))
                ce_t = F.cross_entropy(vl, target, reduction="none").cpu()
                ce_per_step.append(ce_t)
                weighted_logits = weighted_logits + halt_weights[:, t].float().unsqueeze(-1) * vl
                per_step_correct.append((vl.argmax(dim=-1) == target).float().cpu())
            ce = torch.stack(ce_per_step, dim=1)
            hw = halt_weights.cpu()
            t_idx = torch.arange(1, T + 1, dtype=torch.float32).unsqueeze(0)
            controller_step_expected = (hw * t_idx).sum(dim=1)
            chosen_idx = hw.argmax(dim=1)
            chosen_step = chosen_idx.float() + 1.0
            oracle_cost = ce + 0.01 * t_idx
            oracle_step = (oracle_cost.argmin(dim=1) + 1).float()
            mix_pred = weighted_logits.argmax(dim=-1)
            mix_corr += (mix_pred == target).sum().item()
            n_total += target.size(0)
            min_s = int((eval_cfg.get("halting") or {}).get("min_message_steps", 4))
            floor_sum += float(hw[:, max(min_s - 1, 0)].sum().item())
            final_sum += float(hw[:, T - 1].sum().item())
            for j, s in enumerate(chunk):
                hops = _derive_required_hops_v1(s["metadata"])
                rows.append({
                    "seed": seed, "split": split, "sample_idx": start + j,
                    "controller_step_expected": float(controller_step_expected[j].item()),
                    "chosen_step": float(chosen_step[j].item()),
                    "oracle_step": float(oracle_step[j].item()),
                    "required_hops": (int(hops) if hops is not None else None),
                    "oracle_depth": int(s["metadata"].get("oracle_depth", -1)),
                })
        splits_metrics[split] = {
            "mixture_logits_acc": mix_corr / max(n_total, 1),
            "floor_mass_mean": floor_sum / max(n_total, 1),
            "final_mass_mean": final_sum / max(n_total, 1),
            "n": n_total,
        }

    cols = ["seed", "split", "sample_idx", "controller_step_expected",
            "chosen_step", "oracle_step", "required_hops", "oracle_depth"]
    csv_path = out_dir / f"v3_seed{seed}_w001_per_seed.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in rows: w.writerow(r)

    summ = {"per_seed_split": [{"seed": seed, "split": split, **m}
                                for split, m in splits_metrics.items()]}
    sum_path = out_dir / f"v3_seed{seed}_w001_summary.json"
    sum_path.write_text(json.dumps(summ, indent=2))

    print(f"  val: acc={splits_metrics['val']['mixture_logits_acc']:.4f}  "
          f"ood: acc={splits_metrics['ood_mixed']['mixture_logits_acc']:.4f}")
    # Invoke the ordinal-metric audit per seed
    subprocess.check_call([
        "python", "scripts/audit_halting_ordinal_metrics.py",
        "--input", str(csv_path),
        "--summary", str(sum_path),
        "--label", f"rgcn_h7_seed{seed}_w001",
        "--output-dir", str(out_dir),
        "--baseline-val-acc", "0.867",
    ], cwd=str(REPO))


def main():
    for seed in (1, 2, 4, 5):
        audit_seed(seed)


if __name__ == "__main__":
    main()
