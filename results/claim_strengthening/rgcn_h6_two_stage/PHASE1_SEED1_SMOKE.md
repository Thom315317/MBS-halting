# Phase 1 — Seed 1 smoke test (RGCN + H6_detached_aux protocol)

- date: 2026-05-13
- output dir: `results/claim_strengthening/rgcn_h6_two_stage/seed1/`
- patches applied : P1 (extract `compute_halt_aux_features` to `mbs/halting.py`),
  P2 (RGCN `__init__` honors `enriched_halting`), P3 (RGCN forward computes aux
  features per step and feeds them to the enriched controller), P4 (RGCN
  outputs include `final_h` audit hook). Patches add a new training-pipeline
  variant `rgcn_h6_two_stage` to `mbs/train.py`. Existing tests (14/14) and
  smoke scripts pass unchanged.

## 1. Protocol

| stage | init | trainable | epochs | E[steps] target | controller |
|---|---|---|---|---|---|
| **1** (H4-style) | `rgcn_act_postpatch/seed1/.../rgcn_repair_stability_act_warmup_t8_best.pt` (frozen backbone, 187k params) | `halting_controller` only (21,377 params) | 5 | min_steps=2, max_steps=16 | EnrichedAdaptiveHaltingController (MLP d_state+5 → 128 → 64 → 1) with 5 anytime aux features (`normalized_step`, `selector_entropy_t`, `selector_max_prob_t`, `value_margin_t`, `Δvalue_margin_t`) ; `detach_aux_features_from_selector=true` |
| **2** (H5b-style) | Stage 1 best.pt (this seed) | `halting_controller` + `claim_selector_head` (21,474 params) | 5 | min_steps=2, max_steps=16 | same enriched controller, now co-trained with selector |

Both stages use the same step-aware latent loss (`Σ_t halt_w_t · CE_t` via
`answer_readout_mode=latent_claim_selector`), `λ_ponder=0.01`,
`λ_stability=0.01`, `lr=1e-3`, `batch_size=16`, `train_size=5000`,
`val_size=ood_size=512`. Selection : official `val_acc` (no OOD selection).

## 2. Training results

### Stage 1 (controller-only warmup)

- Selected epoch : 3 (val_acc=0.8652, ood_mixed=0.8906)
- Wall clock : 14m03s for 5 epochs
- Halt-policy at selected epoch : E[steps] ≈ 3.4, halt_prob_mean_by_step
  shows mass concentrated at steps 3–5 (no floor / no final collapse).

### Stage 2 (co-train halting_controller + claim_selector_head)

- Selected epoch : 1 (val_acc=0.8496, ood_mixed=0.8926)
- Wall clock : 14m11s for 5 epochs
- The post-Stage-2 audit on the selected ckpt is the official policy
  measurement reported below.

## 3. Audit on the official selected Stage 2 checkpoint

Computed with `scripts/audit_rgcn_h6_two_stage_controller_vs_required_hops.py`
on the v1 `depth_controlled_latent_halting_probe` task ; 512 val + 512
ood_mixed samples ; same `_aggregate_value_logits` path used by the loss.
Output : `rgcn_h6_two_stage_per_seed.csv`, `rgcn_h6_two_stage_summary.json`.

| metric | val | ood_mixed |
|---|---:|---:|
| mixture_logits_acc | 0.8496 | 0.8926 |
| expected_halted_acc | 0.8494 | 0.8876 |
| chosen_step_acc | 0.8496 | 0.8867 |
| **floor_mass_mean** | **0.0001** | **0.0001** |
| **final_mass_mean** | **0.0000** | **0.0000** |
| controller_step_expected_mean | 3.42 | 3.42 |
| chosen_step_mean | 3.38 | 3.38 |
| **chosen_step_distinct** | **3** | **3** |
| Spearman(expected_step, oracle_step) | +0.243 | +0.261 |
| **Spearman(expected_step, required_hops)** | **+0.729** | **+0.689** |
| Spearman(chosen_step, required_hops) | +0.861 | +0.855 |
| Spearman(oracle_step, required_hops) | +0.264 | +0.248 |

## 4. Gate check (criteria from the brief)

| criterion | threshold | seed 1 result | verdict |
|---|---|---|---|
| no floor/final collapse | both < 0.5 | floor=0.0001, final=0.0000 | ✓ PASS |
| Spearman(expected_step, required_hops) | > 0.30 on val OR ood | +0.729 val / +0.689 ood | ✓ PASS (both) |
| chosen_step distinct values per seed | > 1 | 3 | ✓ PASS |
| OOD acc | > 0.75 | 0.8926 | ✓ PASS |

→ **All 4 criteria PASS with margin to spare. GO for Phase 2 (seeds 2..5).**

## 5. First read of the result

Three observations, presented as observations not yet aggregated facts —
they need the 5-seed run to be claimed at scale.

1. **Spearman ≈ 0.69 on RGCN+H6 (seed 1)** is approximately on par with the
   MBS H6_detached_aux 5-seed mean (val 0.697, ood 0.678). Within seed-noise
   bounds, this is consistent with the H6 protocol's expected behavior.

2. **No collapse** : floor_mass ≈ 1e-4, final_mass = 0 on RGCN+H6 seed 1.
   The same naive RGCN ACT backbone had collapsed in seed 1 of
   `rgcn_act_postpatch` (floor_mass ≥ 0.98). The protocol therefore appears
   to fix the collapse failure mode that the naive ACT controller exhibited
   on the same backbone weights.

3. **Spearman(oracle_step, required_hops) ≈ 0.26 vs Spearman(expected_step,
   required_hops) ≈ 0.69** : the 0.26→0.69 jump is reproduced on RGCN
   (it was 0.0→0.69 on MBS), consistent with the controller responding to
   structural difficulty independently of the CE oracle trajectory.

These are seed-1 observations. The 5-seed campaign (Phase 2) is required
before any claim is made.

## 6. Files produced

| produit | path |
|---|---|
| Stage 1 ckpt | `seed1/stage1/checkpoints/rgcn_h6_two_stage_best.pt` |
| Stage 1 log + metrics | `seed1/stage1/run.log`, `..._epoch_metrics.csv`, `..._train_results.json` |
| Stage 2 ckpt | `seed1/stage2/checkpoints/rgcn_h6_two_stage_best.pt` |
| Stage 2 log + metrics | `seed1/stage2/run.log`, `..._epoch_metrics.csv`, `..._train_results.json` |
| Audit summary (seed 1) | `rgcn_h6_two_stage_summary.json` |
| Audit per-sample CSV (seed 1) | `rgcn_h6_two_stage_per_seed.csv` |
| this report | `PHASE1_SEED1_SMOKE.md` |

## 7. Code patches summary (P1..P4)

- `mbs/halting.py` : added module-level `_aggregate_value_logits` and
  `compute_halt_aux_features` (shared by MBS and RGCN paths) ; existing
  `ENRICHED_HALT_AUX_DIM = 5` constant retained.
- `mbs/model.py` : removed the now-redundant `_aggregate_value_logits` and
  `MBSModel._compute_halt_aux_features` ; re-exports `_aggregate_value_logits`
  from `mbs/halting.py` for backward compatibility (used by
  `scripts/smoke_aggregate_empty_mask.py`).
- `mbs/baselines.py`:`RelationalGCNHaltingClassifier.__init__` now reads
  `halting_config["enriched"]` and instantiates either
  `EnrichedAdaptiveHaltingController` or `AdaptiveHaltingController` ; honors
  `halting_config["detach_aux_features_from_selector"]` (default True).
- `mbs/baselines.py`:`RelationalGCNHaltingClassifier.forward` adds the
  enriched halting path (aux features computed once per step, optionally
  detached, fed to the enriched MLP controller). Outputs now include
  `"final_h": h` for parity with `MBSModel`.
- `mbs/train.py`: registered new variant `rgcn_h6_two_stage` in
  `RGCN_VARIANTS`, routed to `RelationalGCNHaltingClassifier` with no
  warmup / no forced step, included in the ponder-loss + step-aware loss
  variant sets.

All patches preserve the existing tests (`pytest tests/ -q` : 14 passed),
the existing `smoke_aggregate_empty_mask.py` smoke, and the existing
`rgcn_repair_stability*` variants' behavior.
