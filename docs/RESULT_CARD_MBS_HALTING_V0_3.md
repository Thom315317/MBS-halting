# MBS-Halting v0.3: 3-seed accuracy/compute result on belief_repair_hard

## 1. Identity

- Version: **MBS-Halting v0.3**
- Result class: 3-seed accuracy/compute comparison on `belief_repair_hard`
- Date archived: **2026-05-08**
- Repository: this result card ships in the public release `MBS-halting`. Only the aggregate-level artifacts are checked in (no checkpoints, no per-seed train/eval JSONs); see [`results/belief_repair_hard_3seed_accuracy_compute_v1/ARTIFACT_MANIFEST.md`](../results/belief_repair_hard_3seed_accuracy_compute_v1/ARTIFACT_MANIFEST.md) for the canonical file list with sha256.

## 2. Experimental objective

Test whether **MBS-Halting** retains a better accuracy/compute tradeoff than two strong RGCN baselines on `belief_repair_hard`:

- `rgcn_repair_stability` — same losses (`repair_loss`, `stability_loss`) as MBS, fixed `T = 8`.
- `rgcn_repair_stability_act_warmup_t8` — RGCN backbone with the same ACT-lite halting head as MBS, plus a 3-epoch warmup at forced `T = 8` to avoid early ACT collapse, then 2 epochs of free ACT with `lambda_ponder = 0.01`.

The objective is **not** to show difficulty-adaptive halting. It is to show that MBS reaches comparable or better OOD performance with substantially fewer expected message-passing steps than the RGCN baselines under the same ponder budget.

## 3. Common configuration

| Field | Value |
|---|---|
| `task` | `belief_repair_hard` |
| `hard_dataset` | `true` |
| `train_size` | 5000 |
| `val_size` | 512 |
| `ood_size` | 512 |
| `seeds` | `[1, 2, 3]` |
| `d_state` | 96 |
| `batch_size` | 16 |
| `max_epochs` | 5 |
| `lr` | 1e-3 |
| `weight_decay` | 0.01 |
| `lambda_repair` | 0.5 |
| `lambda_stability` | 0.01 |
| `grad_clip` | 1.0 |
| `halting.lambda_ponder` | 0.01 |
| `halting.min_message_steps` | 4 |
| `halting.max_message_steps` | 16 |
| `halting.init_halt_prob` | 0.05 |
| `message_steps` (RGCN fixed) | 8 |

Per-seed config files live under [`results/belief_repair_hard_3seed_accuracy_compute_v1/seed{1,2,3}/config.yaml`](../results/belief_repair_hard_3seed_accuracy_compute_v1/seed1/config.yaml).

## 4. Variants compared

| Variant | Backbone | Halting | Notes |
|---|---|---|---|
| `mbs_adaptive_halting` | MBS substrate (modes + gate) | ACT-lite, λ_ponder = 0.01 | Reference |
| `rgcn_repair_stability` | RGCN with repair + stability | none, fixed T = 8 | Strong fixed-step baseline |
| `rgcn_repair_stability_act_warmup_t8` | RGCN with repair + stability | ACT-lite + 3-epoch warmup at forced T = 8 | Diagnostic baseline. Not pure off-the-shelf ACT. |

## 5. Main result table

| Variant | Params | OOD mixed (selected) | OOD mixed (best) | OOD entity | OOD conflict | OOD rule | E[steps] | final_step_mass | ponder_loss | accuracy / step | compute-adj. score |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `mbs_adaptive_halting` | 308,860 | **0.9948 ± 0.0060** | **0.9948 ± 0.0060** | 0.9993 ± 0.0011 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | **4.52 ± 0.33** | 0.0011 ± 0.0010 | 0.0423 ± 0.0013 | **0.2199** | **0.9495** |
| `rgcn_repair_stability` | 176,657 | 0.9440 ± 0.0030 | 0.9466 ± 0.0011 | 0.9473 ± 0.0078 | 0.9818 ± 0.0049 | 0.9811 ± 0.0081 | 8.00 (fixed) | — | 0.0000 | 0.1180 | 0.8640 |
| `rgcn_repair_stability_act_warmup_t8` | 176,754 | 0.9154 ± 0.0203 | 0.9258 ± 0.0052 | 0.9160 ± 0.0090 | 0.9577 ± 0.0137 | 0.9577 ± 0.0096 | 7.90 ± 0.37 | 0.0643 ± 0.0231 | 0.0766 ± 0.0023 | 0.1158 | 0.8363 |

Definitions (diagnostic, not scientific claims):
- `accuracy / step` = `OOD mixed mean / expected_steps_mean` (RGCN fixed uses `8.0`).
- `compute-adjusted score` = `OOD mixed mean − 0.01 × expected_steps_mean`.

Source: [`results/belief_repair_hard_3seed_accuracy_compute_v1/aggregate_summary.json`](../results/belief_repair_hard_3seed_accuracy_compute_v1/aggregate_summary.json).

## 6. Headline result

- **`mbs_adaptive_halting`** : OOD mixed = **0.9948 ± 0.0060**, expected steps = **4.52 ± 0.33** on `belief_repair_hard` (3 seeds).
- **`rgcn_repair_stability`** : OOD mixed = **0.9440 ± 0.0030**, fixed T = **8** (3 seeds).
- **`rgcn_repair_stability_act_warmup_t8`** : OOD mixed = **0.9154 ± 0.0203**, expected steps = **7.90 ± 0.37** (3 seeds).

## 7. Verdict

`verdict = mbs_compute_efficient_win`

Decision rule (encoded in [`scripts/aggregate_3seed_accuracy_compute.py`](../scripts/aggregate_3seed_accuracy_compute.py)):

- `mbs_adaptive_halting` OOD mixed (mean) ≥ `rgcn_repair_stability` OOD mixed (mean) − 0.02 → **0.9948 ≥ 0.9240** ✓
- `mbs_adaptive_halting` expected steps ≤ 0.75 × 8.0 → **4.52 ≤ 6.00** ✓
- `mbs_adaptive_halting` OOD mixed (mean) ≥ `rgcn_repair_stability_act_warmup_t8` OOD mixed (mean) → **0.9948 ≥ 0.9154** ✓

All three conditions hold simultaneously across all 3 seeds.

## 8. Claims supported by this version

- **Supported.** MBS-Halting outperforms the tested RGCN baselines on the accuracy/compute tradeoff on `belief_repair_hard` (3 seeds, single dataset, single random-seed sample).
- **Supported.** MBS-Halting uses substantially fewer expected message-passing steps than fixed-step RGCN (4.52 vs 8.00) while reaching higher OOD mixed accuracy.
- **Supported.** RGCN + ACT-lite with 3-epoch warmup and λ_ponder = 0.01 reduces compute pressure on RGCN (compared to the no-warmup ACT collapse documented in earlier diagnostics) but underperforms MBS-Halting on OOD mixed by ≈ 8 percentage points.

## 9. Claims explicitly NOT supported by this version

- **Not supported.** Difficulty-adaptive halting. This version does not run the difficulty-controlled probe on `belief_repair_hard`, and the v0.3 claim is purely about average compute, not about per-instance difficulty calibration.
- **Not supported.** Transfer beyond `belief_repair_hard`. No second task in this card.
- **Not supported.** General reasoning. Single synthetic task family.
- **Not supported.** Stable attractor dynamics. No attractor analysis here.
- **Not resolved.** Possible `repair_loss` leakage / teacher-forcing concern. `repair_loss` provides per-node supervision and may be carrying structural information beyond what is fair for the comparison; an explicit leakage audit is needed before stronger claims.

## 10. Known limitations

1. **Single task family.** All 3 variants are tested on the same synthetic `belief_repair_hard` dataset.
2. **Synthetic benchmark.** The dataset is generated by the project's own pipeline ([`mbs/datasets.py`](../mbs/datasets.py)). Co-design risk between dataset and model has not been ruled out.
3. **`repair_loss` is central.** It supervises per-node repair predictions and may carry leakage that the bare answer-loss baselines would not have. This is suspicious and must be audited.
4. **No depth-controlled causal probe yet.** The result speaks to *average* expected steps, not to per-instance difficulty calibration.
5. **No transfer result in v0.3.**
6. **RGCN + ACT-warmup is not pure off-the-shelf ACT.** It uses a 3-epoch warmup at forced `T = 8`, then 2 epochs of free ACT. Without warmup, ACT collapses to step 4 (documented in earlier diagnostic runs). This makes the baseline a controlled diagnostic, not a vanilla reference.
7. **Parameter count asymmetry.** MBS has more parameters (308,860) than the RGCN baselines (176,657 / 176,754). This is reported explicitly in §5 and §11.

## 11. Reproducibility

### Output root

`results/belief_repair_hard_3seed_accuracy_compute_v1/`

### Aggregate

- [`aggregate_summary.json`](../results/belief_repair_hard_3seed_accuracy_compute_v1/aggregate_summary.json)
- [`aggregate_summary.md`](../results/belief_repair_hard_3seed_accuracy_compute_v1/aggregate_summary.md)
- [`ARTIFACT_MANIFEST.json`](../results/belief_repair_hard_3seed_accuracy_compute_v1/ARTIFACT_MANIFEST.json)
- [`ARTIFACT_MANIFEST.md`](../results/belief_repair_hard_3seed_accuracy_compute_v1/ARTIFACT_MANIFEST.md)

### Per-seed × per-variant outputs

For each `seed ∈ {1, 2, 3}` and each `variant ∈ {mbs_adaptive_halting, rgcn_repair_stability, rgcn_repair_stability_act_warmup_t8}`:

```
results/belief_repair_hard_3seed_accuracy_compute_v1/seed{N}/{variant}/
  benchmark_summary.json
  benchmark_summary.md
  {variant}_train_results.json
  {variant}_eval_results.json
  {variant}_epoch_metrics.csv
  run.log
  checkpoints/{variant}_best.pt
```

### Configs used

- [`results/belief_repair_hard_3seed_accuracy_compute_v1/seed1/config.yaml`](../results/belief_repair_hard_3seed_accuracy_compute_v1/seed1/config.yaml)
- [`results/belief_repair_hard_3seed_accuracy_compute_v1/seed2/config.yaml`](../results/belief_repair_hard_3seed_accuracy_compute_v1/seed2/config.yaml)
- [`results/belief_repair_hard_3seed_accuracy_compute_v1/seed3/config.yaml`](../results/belief_repair_hard_3seed_accuracy_compute_v1/seed3/config.yaml)

### Commands to reproduce (do not auto-run)

Reproduce the campaign (warning: ~3h45-4h25 sequential on RTX 3070 8 GB):

```bash
cd MBS-halting
source .venv/bin/activate
python scripts/run_3seed_accuracy_compute.py --seeds 1 2 3
python scripts/aggregate_3seed_accuracy_compute.py --seeds 1 2 3
```

Regenerate aggregate summary only (no training, reads existing per-run JSONs):

```bash
python scripts/aggregate_3seed_accuracy_compute.py \
  --root results/belief_repair_hard_3seed_accuracy_compute_v1 \
  --seeds 1 2 3
```

## 12. Recommended paper framing

Use:
- "compute-efficient belief-graph repair"
- "short-horizon learned halting"
- "accuracy/compute tradeoff"

Avoid:
- "adaptive halting"
- "general reasoning"
- "emergent cognitive control"
- "stable attractor dynamics"
- "transferable reasoning"

## 13. Cross-references

- Claims note: [`docs/MBS_HALTING_V0_3_CLAIMS.md`](MBS_HALTING_V0_3_CLAIMS.md)
- Compact main table: [`docs/tables/mbs_halting_v0_3_main_table.md`](tables/mbs_halting_v0_3_main_table.md) (also `.csv`)
- Result card JSON twin: [`docs/RESULT_CARD_MBS_HALTING_V0_3.json`](RESULT_CARD_MBS_HALTING_V0_3.json)
