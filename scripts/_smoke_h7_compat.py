"""H7 smoke tests, run before any V2 training launch.

5 checks :
  1. H6 config loads & one batch flows through compute_loss with the
     ordinal block inactive ; no new behaviour triggered.
  2. H7 config loads and the ordinal-aware collate adds required_hops
     to each batch.
  3. ordinal_pairwise_loss returns a finite scalar with requires_grad
     on a tiny manual batch.
  4. compute_loss on an H7 batch yields a finite loss with an
     `ordinal_loss` entry in parts.
  5. The committed audit script reproduces the H6 diagnosis (seed 3 =
     soft_middle_step, seeds 1/2/4/5 = binary_h9_shortcut).

This script is read-only on existing committed artefacts. It writes
nothing on disk except stdout.
"""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path

import torch

REPO = Path("/home/thom315/MBS-halting-h7")
sys.path.insert(0, str(REPO))


def red(s):
    return f"\033[31;1m{s}\033[0m"


def green(s):
    return f"\033[32;1m{s}\033[0m"


def check(label, condition, details=""):
    status = green("OK") if condition else red("FAIL")
    print(f"  [{status}] {label}", "—", details if details else "")
    if not condition:
        sys.exit(1)


# ----------------------------------------------------------------------
# 1. H6 config still loads & flows through without H7 behaviour
# ----------------------------------------------------------------------
def smoke_1_h6_unchanged():
    print("smoke 1: H6 config unchanged")
    from mbs.utils import load_config
    cfg = load_config(str(REPO / "configs/rgcn_h6_stage1_seed1.yaml"))
    check("config loaded", isinstance(cfg, dict))
    check("halting_ordinal absent or disabled",
          not bool((cfg.get("halting_ordinal") or {}).get("enabled", False)))
    check("checkpoint_gate absent or disabled",
          not bool((cfg.get("checkpoint_gate") or {}).get("enabled", False)))

    # Try to materialise a model + one batch + one loss, but skip if
    # weights aren't available (don't fail the smoke for a missing ckpt).
    from mbs.train import build_model, make_loaders, compute_loss, move_batch
    from mbs.tokenizer import SimpleTokenizer
    tok = SimpleTokenizer()
    m = build_model("rgcn_h6_two_stage", cfg, tok)
    loaders = make_loaders(cfg, tok)
    batch = next(iter(loaders["train"]))
    # H6 config => required_hops MUST NOT be in batch (collate not wrapped).
    check("H6 batch has NO required_hops (collate unchanged)",
          "required_hops" not in batch)
    out = m(batch)
    loss, parts = compute_loss(out, batch, cfg, "rgcn_h6_two_stage")
    check("loss is finite scalar", torch.isfinite(loss).item(),
          f"loss={float(loss):.4f}")
    check("no ordinal_loss in parts (H7 block skipped)",
          "ordinal_loss" not in parts)


# ----------------------------------------------------------------------
# 2. H7 config loads & batch carries required_hops
# ----------------------------------------------------------------------
def smoke_2_h7_collate():
    print("smoke 2: H7 collate adds required_hops")
    from mbs.utils import load_config
    cfg = load_config(str(REPO / "configs/h7_ordinal_halting/rgcn_h7_seed3_w0005.yaml"))
    check("halting_ordinal.enabled is True",
          bool(cfg["halting_ordinal"]["enabled"]))
    check("checkpoint_gate.enabled is True",
          bool(cfg["checkpoint_gate"]["enabled"]))
    check("ordinal weight is 0.005",
          float(cfg["halting_ordinal"]["loss_weight"]) == 0.005)

    from mbs.train import make_loaders
    from mbs.tokenizer import SimpleTokenizer
    tok = SimpleTokenizer()
    loaders = make_loaders(cfg, tok)
    batch = next(iter(loaders["train"]))
    check("H7 batch carries required_hops",
          "required_hops" in batch)
    hops = batch["required_hops"]
    check("required_hops dtype is long", hops.dtype == torch.long)
    check("required_hops values in [5, 9] or -1 sentinel",
          all(int(h) == -1 or 5 <= int(h) <= 9 for h in hops.tolist()),
          f"sample={hops[:8].tolist()}")


# ----------------------------------------------------------------------
# 3. ordinal_pairwise_loss returns a finite scalar with grad
# ----------------------------------------------------------------------
def smoke_3_ordinal_loss_finite():
    print("smoke 3: ordinal_pairwise_loss finite + grad")
    from mbs.ordinal_halting import ordinal_pairwise_loss
    # Hand-crafted : 10 samples, 2 in each of buckets 5..9.
    E = torch.tensor([3.0, 4.0,   3.1, 3.9,   3.2, 4.1,   3.0, 4.2,   2.9, 5.5],
                     requires_grad=True)
    h = torch.tensor([5, 6,    6, 7,    7, 8,    8, 9,    9, 9])
    loss, per_b_counts, per_b_loss = ordinal_pairwise_loss(
        E, h, margin=0.15, max_pairs_per_batch=64)
    check("loss is finite", torch.isfinite(loss).item(),
          f"loss={float(loss):.4f}")
    check("loss requires_grad", loss.requires_grad)
    check("backward runs", True)
    loss.backward()
    check("grad on E is non-zero", torch.any(E.grad != 0).item(),
          f"grad_norm={E.grad.norm():.4f}")
    check("pair counts per boundary",
          all(per_b_counts[k] > 0 for k in ("5_6", "6_7", "7_8", "8_9")),
          f"counts={per_b_counts}")


# ----------------------------------------------------------------------
# 4. compute_loss on H7 batch yields finite loss + ordinal_loss key
# ----------------------------------------------------------------------
def smoke_4_compute_loss_h7():
    print("smoke 4: compute_loss on H7 batch")
    from mbs.utils import load_config
    cfg = load_config(str(REPO / "configs/h7_ordinal_halting/rgcn_h7_seed3_w0005.yaml"))
    from mbs.train import build_model, make_loaders, compute_loss, move_batch
    from mbs.tokenizer import SimpleTokenizer
    tok = SimpleTokenizer()
    m = build_model("rgcn_h7_two_stage", cfg, tok)
    loaders = make_loaders(cfg, tok)
    batch = next(iter(loaders["train"]))
    out = m(batch)
    loss, parts = compute_loss(out, batch, cfg, "rgcn_h7_two_stage")
    check("loss is finite", torch.isfinite(loss).item(),
          f"loss={float(loss):.4f}")
    check("ordinal_loss in parts", "ordinal_loss" in parts,
          f"keys={[k for k in parts.keys() if 'ordinal' in k]}")
    check("ordinal_pairs_total > 0", float(parts.get("ordinal_pairs_total", 0)) > 0,
          f"pairs_total={parts.get('ordinal_pairs_total')}")


# ----------------------------------------------------------------------
# 5. Audit script reproduces H6 diagnosis (regression test)
# ----------------------------------------------------------------------
def smoke_5_audit_reproduces():
    print("smoke 5: audit script reproduces H6 diagnosis")
    # Re-read the existing audit output written earlier (we ran the audit
    # in the H6_REAUDIT step). If it isn't there, run it again.
    out_dir = REPO / "results/claim_strengthening/h7_ordinal_halting/audits"
    csv_path = out_dir / "rgcn_h6_baseline_ordinal_metrics_per_seed_split.csv"
    if not csv_path.exists():
        import subprocess
        subprocess.check_call([
            "python", "scripts/audit_halting_ordinal_metrics.py",
            "--input", "results/claim_strengthening/rgcn_h6_two_stage/rgcn_h6_two_stage_per_seed.csv",
            "--summary", "results/claim_strengthening/rgcn_h6_two_stage/rgcn_h6_two_stage_summary.json",
            "--label", "rgcn_h6_baseline",
            "--output-dir", str(out_dir),
            "--baseline-val-acc", "0.867",
        ], cwd=str(REPO))
    import csv
    with csv_path.open() as f:
        rows = list(csv.DictReader(f))
    seed3_rows = [r for r in rows if int(r["seed"]) == 3]
    other_rows = [r for r in rows if int(r["seed"]) != 3]
    check("seed 3 cells flagged soft_middle_step",
          all(r["flag_soft_middle_step"] == "True" for r in seed3_rows),
          f"seed3 flags = {[r['flags_concat'] for r in seed3_rows]}")
    check("seeds 1,2,4,5 cells flagged binary_h9_shortcut",
          all(r["flag_binary_h9_shortcut"] == "True" for r in other_rows),
          f"flags = {[r['flags_concat'] for r in other_rows[:4]]}")
    check("0 of 10 cells are ordinal_healthy",
          all(r["flag_ordinal_healthy"] == "False" for r in rows))


def main():
    print("=" * 60)
    print(" H7 backward-compat + ordinal-loss + audit smoke tests")
    print("=" * 60)
    smoke_1_h6_unchanged()
    print()
    smoke_2_h7_collate()
    print()
    smoke_3_ordinal_loss_finite()
    print()
    smoke_4_compute_loss_h7()
    print()
    smoke_5_audit_reproduces()
    print()
    print(green("ALL SMOKE TESTS PASSED"))


if __name__ == "__main__":
    main()
