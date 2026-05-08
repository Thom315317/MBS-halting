#!/usr/bin/env python
"""
Orchestrator for the 3-seed accuracy/compute campaign on belief_repair_hard.

Iterates seeds x variants, calls `python -u -m mbs.benchmark` for each pair,
writes outputs under <root>/seed{N}/<variant>/. Refuses to overwrite an
existing run by default; pass --overwrite to force.

This script does NOT modify training behaviour; it only sequences runs and
streams stdout so progress is visible live.
"""
from pathlib import Path
import argparse
import datetime as dt
import os
import shutil
import subprocess
import sys


VARIANTS_DEFAULT = (
    "mbs_adaptive_halting",
    "rgcn_repair_stability",
    "rgcn_repair_stability_act_warmup_t8",
)


def _now():
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _run_dir_has_results(run_dir):
    if not run_dir.exists():
        return False
    return (run_dir / "benchmark_summary.json").exists()


def run_one(repo_root, root, seed, variant, overwrite, dry_run):
    seed_dir = root / f"seed{seed}"
    config_path = seed_dir / "config.yaml"
    run_dir = seed_dir / variant
    if not config_path.exists():
        raise SystemExit(
            f"missing per-seed config: {config_path}\n"
            "Generate the configs first (see header of this script)."
        )
    if _run_dir_has_results(run_dir) and not overwrite:
        print(f"[{_now()}] SKIP seed={seed} variant={variant} (existing benchmark_summary.json) — pass --overwrite to force")
        return "skipped"
    if overwrite and run_dir.exists():
        print(f"[{_now()}] CLEAN {run_dir}")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-u", "-m", "mbs.benchmark",
        "--config", str(config_path),
        "--variants", variant,
        "--output-dir", str(run_dir),
    ]
    log_path = run_dir / "run.log"
    print(f"[{_now()}] START seed={seed} variant={variant}")
    print("  cmd:", " ".join(cmd))
    print(f"  log: {log_path}")
    if dry_run:
        return "dry"
    with log_path.open("w", encoding="utf-8") as log_handle:
        env = dict(os.environ)
        env.setdefault("PYTHONUNBUFFERED", "1")
        proc = subprocess.Popen(
            cmd,
            cwd=str(repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log_handle.write(line)
        proc.wait()
    if proc.returncode != 0:
        print(f"[{_now()}] FAIL seed={seed} variant={variant} returncode={proc.returncode}")
        return "failed"
    print(f"[{_now()}] DONE seed={seed} variant={variant}")
    return "ok"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default="results/belief_repair_hard_3seed_accuracy_compute_v1",
        help="output root (one subfolder per seed)",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS_DEFAULT))
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="force re-run even if benchmark_summary.json exists",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the commands but do not execute",
    )
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    root = Path(args.root)
    if not root.is_absolute():
        root = repo_root / root
    print(f"[{_now()}] CAMPAIGN root={root}")
    print(f"  seeds={args.seeds}  variants={args.variants}  overwrite={args.overwrite}  dry_run={args.dry_run}")
    statuses = []
    for seed in args.seeds:
        for variant in args.variants:
            status = run_one(repo_root, root, seed, variant, overwrite=args.overwrite, dry_run=args.dry_run)
            statuses.append({"seed": seed, "variant": variant, "status": status})
    print()
    print(f"[{_now()}] CAMPAIGN END")
    for entry in statuses:
        print(f"  seed={entry['seed']} variant={entry['variant']} status={entry['status']}")


if __name__ == "__main__":
    main()
