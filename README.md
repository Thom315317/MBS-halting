# MBS-Halting: Compute-Efficient Belief-Graph Repair

> **Status:** research prototype. The result reported here is a workshop / preprint-class result, not a production system. Public release v0.3.

## What this is

A small graph neural model (≈ 308k parameters, `d_state = 96`) that performs **belief-graph repair** on a synthetic reasoning task `belief_repair_hard`. The model uses:

- **MBS substrate** — a relational message-passing layer with operation modes and a learned gate, applied to graphs of typed cells (entities, claims, conflicts, rules).
- **ACT-lite halting** — a per-step halting head that learns to halt the message-passing loop early when the current state already supports a confident answer, with a ponder-cost regulariser on the expected number of message-passing steps.

The repository ships the minimal code to:

1. generate the `belief_repair_hard` dataset deterministically from a config,
2. train and evaluate **MBS-Halting** plus the two RGCN baselines used in the v0.3 comparison,
3. aggregate a 3-seed campaign and produce the result card.

The result card claims a **better accuracy/compute tradeoff** vs the tested RGCN baselines on this single synthetic task. It does **not** claim adaptive halting, transfer, or general reasoning — see §"Non-claims" below.

## Main result (v0.3, 3 seeds, `belief_repair_hard`)

| Method | Params | OOD mixed | Expected steps |
|---|---:|---:|---:|
| **MBS-Halting (v0.3)** | 308,860 | **0.9948 ± 0.0060** | **4.52 ± 0.33** |
| RGCN + repair + stability (fixed T = 8) | 176,657 | 0.9440 ± 0.0030 | 8 (fixed) |
| RGCN + repair + stability + ACT-warmup | 176,754 | 0.9154 ± 0.0203 | 7.90 ± 0.37 |

Source: [`results/belief_repair_hard_3seed_accuracy_compute_v1/aggregate_summary.json`](results/belief_repair_hard_3seed_accuracy_compute_v1/aggregate_summary.json).

Verdict (encoded in [`scripts/aggregate_3seed_accuracy_compute.py`](scripts/aggregate_3seed_accuracy_compute.py)): `mbs_compute_efficient_win`.

The full result card with per-seed numbers, per-split accuracies, and the verdict rule is in [`docs/RESULT_CARD_MBS_HALTING_V0_3.md`](docs/RESULT_CARD_MBS_HALTING_V0_3.md).

## Supported claim

> MBS-Halting reaches a better accuracy/compute tradeoff than the tested RGCN baselines on `belief_repair_hard` (3 seeds, single task, fixed protocol).

## Non-claims (explicitly NOT supported by this release)

- **Difficulty-adaptive halting.** This release does not include a per-instance difficulty probe; the result is about *average* expected steps, not about per-instance calibration.
- **Transfer beyond `belief_repair_hard`.** Single-task release.
- **General reasoning.** The benchmark is small and synthetic.
- **Stable attractor dynamics.** No attractor analysis here.
- **Absence of `repair_loss` leakage.** `repair_loss` is central to the training signal and may carry structural supervision; a leakage audit is an explicit follow-up.

The full claims / anti-claims discussion (including reviewer-2 anticipations) is in [`docs/MBS_HALTING_V0_3_CLAIMS.md`](docs/MBS_HALTING_V0_3_CLAIMS.md).

## Installation

Linux / WSL Ubuntu, Python 3.10+:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Smoke check (instant, just imports the package and constructs the three v0.3 variants on CPU):

```bash
python -c "import mbs, mbs.train, mbs.benchmark, mbs.eval; print('mbs OK')"
python -m pytest tests/test_substrate_shapes.py tests/test_graph.py tests/test_dataset.py -q
```

A GPU is recommended for any actual training run (an RTX 3070 8 GB is enough for the v0.3 protocol).

## Reproducing / inspecting v0.3

The campaign that produced the v0.3 result lives under `results/belief_repair_hard_3seed_accuracy_compute_v1/`. The aggregate `aggregate_summary.json` plus the artifact manifest are checked into the repo; per-seed checkpoints and per-seed train/eval JSONs are *not* (size + reproducibility policy — see [`docs/RESULT_CARD_MBS_HALTING_V0_3.md`](docs/RESULT_CARD_MBS_HALTING_V0_3.md) §11).

### Inspect the published result (no compute)

```bash
# pretty-print the verdict and per-variant aggregate
jq '{verdict: .verdict, compact: .compact}' \
  results/belief_repair_hard_3seed_accuracy_compute_v1/aggregate_summary.json

# read the result card
less docs/RESULT_CARD_MBS_HALTING_V0_3.md
```

### Re-run the 3-seed campaign (expensive — see ETA)

```bash
# single seed, single variant — the cheapest sanity check (~15-20 min on RTX 3070)
python scripts/run_3seed_accuracy_compute.py \
  --root results/belief_repair_hard_3seed_accuracy_compute_v1 \
  --seeds 1 --variants rgcn_repair_stability

# all variants for one seed (~80-90 min on RTX 3070)
python scripts/run_3seed_accuracy_compute.py \
  --root results/belief_repair_hard_3seed_accuracy_compute_v1 \
  --seeds 1

# full 3-seed campaign (~3h45-4h25 on RTX 3070, sequential)
python scripts/run_3seed_accuracy_compute.py \
  --root results/belief_repair_hard_3seed_accuracy_compute_v1 \
  --seeds 1 2 3
```

`run_3seed_accuracy_compute.py` refuses to overwrite an existing `benchmark_summary.json` unless `--overwrite` is passed. The configs at `results/belief_repair_hard_3seed_accuracy_compute_v1/seed{1,2,3}/config.yaml` are the exact configs used.

### Re-aggregate (cheap, no training)

```bash
python scripts/aggregate_3seed_accuracy_compute.py \
  --root results/belief_repair_hard_3seed_accuracy_compute_v1 \
  --seeds 1 2 3
```

## Repository layout

```
MBS-halting/
├── README.md                          ← this file
├── LICENSE                            ← MIT
├── requirements.txt                   ← torch, pyyaml, pytest
├── pytest.ini
├── .gitignore
├── configs/
│   └── tiny_hard_halting.yaml         ← belief_repair_hard config used in v0.3
├── mbs/                               ← the package
│   ├── __init__.py
│   ├── baselines.py                   ← RGCN baselines (fixed + halting variants)
│   ├── benchmark.py                   ← train + eval orchestrator for one config
│   ├── datasets.py                    ← belief_repair_hard generator
│   ├── eval.py                        ← evaluator + post-hoc message-step sweep
│   ├── graph.py                       ← graph collator + edge/cell types
│   ├── halting.py                     ← AdaptiveHaltingController
│   ├── model.py                       ← MBSModel (substrate + halting)
│   ├── substrate.py                   ← MBS message-passing layer
│   ├── tokenizer.py
│   ├── train.py                       ← training loop, loss assembly, build_model
│   └── utils.py
├── scripts/
│   ├── run_3seed_accuracy_compute.py  ← campaign orchestrator (no auto-launch)
│   └── aggregate_3seed_accuracy_compute.py  ← aggregator + verdict
├── tests/                             ← short pytest sanity tests
├── docs/
│   ├── RESULT_CARD_MBS_HALTING_V0_3.md
│   ├── RESULT_CARD_MBS_HALTING_V0_3.json
│   ├── MBS_HALTING_V0_3_CLAIMS.md
│   └── tables/
│       ├── mbs_halting_v0_3_main_table.md
│       └── mbs_halting_v0_3_main_table.csv
└── results/
    └── belief_repair_hard_3seed_accuracy_compute_v1/
        ├── aggregate_summary.json
        ├── aggregate_summary.md
        ├── ARTIFACT_MANIFEST.json
        └── ARTIFACT_MANIFEST.md
```

## Variants implemented in this release

| Variant | Class | Notes |
|---|---|---|
| `mbs_adaptive_halting` | `MBSModel(adaptive_halting=True)` | The main v0.3 model. |
| `rgcn_repair_stability` | `RelationalGCNClassifier` | Fixed T = 8 RGCN with `repair_loss + stability_loss`. |
| `rgcn_repair_stability_act_forced_t8` | `RelationalGCNHaltingClassifier(force_terminal_step=8)` | Diagnostic baseline: same wrapper as warmup but halting head is never used. Used to verify that the wrapper is not the source of any failure. |
| `rgcn_repair_stability_act_warmup_t8` | `RelationalGCNHaltingClassifier(warmup_terminal_step=8)` | Diagnostic baseline: 3 epochs at forced T = 8, then 2 epochs of free ACT with `lambda_ponder = 0.01`. |

Older ablation variants (`bow`, `gru`, `adaptive_latent`, `mbs_no_modes`, `mbs_no_gate`, `mbs_no_repair_loss`, `mbs_no_stability_loss`) belong to v0.1 / v0.2 ablation studies and are intentionally not shipped in this release.

## Citation / status

This is a **research prototype**. The v0.3 result is a 3-seed result on a single synthetic task family; please cite it as such if you reference it.

```
MBS-Halting v0.3
Compute-efficient belief-graph repair on belief_repair_hard.
3-seed result, single synthetic task. Workshop / preprint-class.
2026.
```

The full result card with all caveats and limitations is the load-bearing document: [`docs/RESULT_CARD_MBS_HALTING_V0_3.md`](docs/RESULT_CARD_MBS_HALTING_V0_3.md).

## License

MIT — see [LICENSE](LICENSE).
