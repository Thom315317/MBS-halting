# MBS-Halting v0.3 — main table (paper-ready)

Source: [`results/belief_repair_hard_3seed_accuracy_compute_v1/aggregate_summary.json`](../../results/belief_repair_hard_3seed_accuracy_compute_v1/aggregate_summary.json) (3 seeds).

All metrics are the mean ± std over seeds {1, 2, 3} of the **selected checkpoint** OOD mixed accuracy.
The `accuracy/step` and `compute-adjusted score` columns are diagnostic, not scientific claims.

| Method | Params | OOD mixed | Expected steps | Accuracy / step | Compute-adjusted score | Verdict note |
|---|---:|---:|---:|---:|---:|---|
| MBS-Halting (v0.3) | 308860 | 0.9948 ± 0.0060 | 4.52 ± 0.33 | 0.2199 | 0.9495 | compute-efficient short-horizon halting |
| RGCN + repair + stability (fixed T=8) | 176657 | 0.9440 ± 0.0030 | 8.00 ± 0.00 | 0.1180 | 0.8640 | strong fixed-step baseline |
| RGCN + repair + stability + ACT-warmup | 176754 | 0.9154 ± 0.0203 | 7.90 ± 0.37 | 0.1158 | 0.8363 | diagnostic baseline (3-epoch forced-T8 warmup) |

## Definitions

- `OOD mixed` is the OOD mixed accuracy of the **selected** checkpoint (selection by val_acc, see [`mbs/train.py`](../../mbs/train.py)).
- `Expected steps` is the mean expected number of message-passing iterations on the OOD mixed split (RGCN-fixed runs at exactly T = 8).
- `Accuracy / step` = OOD mixed mean / expected steps mean. Diagnostic only.
- `Compute-adjusted score` = OOD mixed mean − 0.01 × expected steps mean. Diagnostic only.
