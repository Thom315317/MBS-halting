# RGCN + H6_detached_aux protocol — 5-seed report (Phase 2)

- date: 2026-05-13
- output dir: `results/claim_strengthening/rgcn_h6_two_stage/`
- input ckpts : the 5 `rgcn_act_postpatch/seed{1..5}/...best.pt` (the
  RGCN ACT post-patch backbone is used as the frozen Stage 1 init for
  each seed, mirroring the H6 protocol's "init from H1b MBS ckpt").
- patches in code : P1+P2+P3+P4 (see PHASE0_FEASIBILITY.md and
  PHASE1_SEED1_SMOKE.md §7) plus a new training-pipeline variant
  `rgcn_h6_two_stage` registered in `mbs/train.py`.

## 1. Headline numbers (cross-seed mean over 5 seeds)

| metric | val | ood_mixed |
|---|---:|---:|
| mixture_logits_acc | **0.867 ± 0.016** | **0.869 ± 0.019** |
| expected_halted_acc | 0.866 ± 0.014 | 0.867 ± 0.018 |
| chosen_step_acc | 0.867 ± 0.014 | 0.868 ± 0.017 |
| floor_mass_mean | 0.003 ± 0.005 | 0.003 ± 0.005 |
| final_mass_mean | 0.000 ± 0.000 | 0.000 ± 0.000 |
| collapse_mode count | **0/5 collapse** | 0/5 collapse |
| Spearman(expected_step, required_hops) | +0.605 ± 0.232 | +0.600 ± 0.192 |
| Spearman(chosen_step, required_hops) | +0.674 ± 0.306 | +0.702 ± 0.229 |
| Spearman(oracle_step, required_hops) | +0.274 ± 0.079 | +0.241 ± 0.074 |
| Spearman(expected_step, oracle_step) | +0.345 ± 0.149 | +0.344 ± 0.103 |
| controller_step_expected_mean | 4.13 ± 0.83 | 4.11 ± 0.83 |
| chosen_step_mean | 4.12 ± 0.82 | 4.11 ± 0.83 |
| chosen_step_var | 0.69 ± 0.67 | 0.68 ± 0.66 |
| chosen_step_distinct | 3.8 ± 0.7 | 3.4 ± 1.0 |

## 2. Per-seed breakdown (val + ood_mixed)

| seed | val acc | ood acc | sρ(E[s],hops) val | sρ(E[s],hops) ood | chosen_distinct | E[s] | floor / final | collapse |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 0.850 | 0.893 | +0.729 | +0.689 | 3 / 3 | 3.42 | 0.000 / 0.000 | none |
| 2 | 0.883 | 0.891 | +0.695 | +0.690 | 4 / 4 | 3.65 | 0.000 / 0.000 | none |
| 3 | 0.856 | 0.861 | **+0.141** | **+0.217** | 4 / 3 | 5.12 | 0.000 / 0.000 | none |
| 4 | 0.859 | 0.854 | +0.750 | +0.697 | 5 / 5 | 5.15 | 0.000 / 0.000 | none |
| 5 | 0.889 | 0.846 | +0.709 | +0.708 | 3 / 2 | 3.29 | 0.014 / 0.000 | none |

→ **4 of 5 seeds** replicate the H6_detached_aux bucket-alignment signal
(Spearman in the [0.69, 0.75] range, indistinguishable from MBS H6dau's
[0.66, 0.74] range across its 5 seeds). **Seed 3 is a clear outlier** with
Spearman ≈ 0.14–0.22 ; its E[step] is the highest of the 5 (5.12) and its
chosen_step variance the lowest (0.14) — the controller halts at a single
late step for almost every sample, regardless of required_hops. The
remaining four seeds replicate the protocol cleanly.

→ **0 of 5 seeds collapse** : floor_mass max = 0.014 (seed 5), final_mass
max = 0.000. The RGCN ACT post-patch baseline had collapsed 5/5 seeds on
the same backbone weights. The protocol therefore fixes the collapse failure
mode that the naive single-stage ACT controller exhibited.

## 3. Cross-seed bucket means (the H6dau hardest-bucket effect)

Mean over 5 seeds of the per-seed bucket means of the controller's expected
step, grouped by `required_hops`. The same qualitative pattern as MBS
H6_detached_aux on v1 emerges : easy buckets indistinguishable, hardest
bucket clearly later.

| split | h=5 | h=6 | h=7 | h=8 | h=9 | Δ(h=9 − h=8) |
|---|---:|---:|---:|---:|---:|---:|
| val | 3.64 ± 0.84 | 3.67 ± 0.85 | 3.64 ± 0.84 | 3.64 ± 0.83 | **5.01 ± 1.04** | **+1.37 step** |
| ood | 3.65 ± 0.84 | 3.66 ± 0.84 | 3.65 ± 0.85 | 3.65 ± 0.85 | **5.03 ± 1.06** | **+1.38 step** |

(stdev across the 5 seed-means.)

For reference, MBS H6_detached_aux on the same task had buckets at
val {5:4.69, 6:4.60, 7:4.68, 8:4.61, 9:7.37} : the qualitative shape is
identical — easy buckets ~constant, hardest bucket clearly elevated — but
the absolute step values are lower on RGCN (E[s]_easy ≈ 3.64 vs 4.65 on
MBS) and the jump is smaller (+1.37 vs +2.75 step). RGCN+H6 halts earlier
overall, and the hardest-bucket detection effect is dampened relative to
MBS but is still present and consistent.

## 4. Comparison vs the prior baselines (paper Figure B framing)

Numbers from the existing 5-seed campaigns ; data sources :
`controller_required_hops_summary.json` (MBS H6dau),
`rgcn_act_postpatch_summary.json` (RGCN ACT post-patch),
`rgcn_h6_two_stage_summary.json` (this work).

| configuration | n seeds | val acc | ood_mixed acc | sρ(E[s],hops) val | sρ(E[s],hops) ood | collapse (≥0.5) | chosen_distinct |
|---|---:|---:|---:|---:|---:|---:|---:|
| MBS H6_detached_aux | 5 | n/a (in source) | n/a (in source) | +0.697 ± 0.024 | +0.678 ± 0.032 | 0/5 | 3.8 |
| RGCN ACT post-patch | 5 | 0.836 | **0.872** | NaN (var=0) | NaN (var=0) | **5/5** (3 floor + 2 final) | **1.0** |
| **RGCN + H6 (this)** | 5 | **0.867** | 0.869 | +0.605 ± 0.232 | +0.600 ± 0.192 | **0/5** | 3.8 / 3.4 |

Or, dropping the seed-3 outlier from the RGCN+H6 row to mirror the per-seed
homogeneity check :

| configuration | n seeds | sρ(E[s],hops) val | sρ(E[s],hops) ood |
|---|---:|---:|---:|
| MBS H6_detached_aux (all 5) | 5 | +0.697 ± 0.024 | +0.678 ± 0.032 |
| RGCN + H6 (4/5, seed 3 excluded) | 4 | +0.721 ± 0.025 | +0.696 ± 0.009 |
| RGCN + H6 (all 5) | 5 | +0.605 ± 0.232 | +0.600 ± 0.192 |

On 4 of 5 seeds, the RGCN+H6 alignment is **statistically indistinguishable**
from MBS H6dau (both ≈ 0.69–0.72, narrow stdev).

## 5. What the 5-seed result demonstrates

**Demonstrated (data-supported):**

1. **The H6_detached_aux protocol transfers to the RGCN backbone**:
   on 4 of 5 seeds, the protocol produces the same Spearman(expected_step,
   required_hops) ≈ 0.69–0.75 alignment that the MBS H6dau 5-seed campaign
   produced, with comparable cross-seed stdev (0.025 on the 4-seed subset
   vs 0.024 on the MBS 5-seed). The same hardest-bucket-boundary pattern
   emerges in the cross-seed bucket means (easy buckets ≈ 3.64 step,
   hardest bucket ≈ 5.01 step ; jump +1.37 step).

2. **The protocol fixes the collapse failure mode** that the RGCN ACT
   post-patch baseline exhibited on the same backbone : 5/5 collapse →
   0/5 collapse, chosen_step_distinct 1.0 → 3.4–3.8, Spearman NaN → 0.60.

3. **The substrate-vs-protocol confound has been removed** : we now have
   the same H6 protocol on two different substrates (MBS, RGCN), enabling
   the comparison "RGCN ACT collapses, RGCN+H6 does not" without conflating
   architecture and protocol. The 0.872 OOD acc of the collapsed RGCN ACT
   baseline and the 0.869 OOD acc of the non-degenerate RGCN+H6 are within
   0.003 — accuracy is not the discriminator, halting policy is.

**Not demonstrated (and not claimed):**

1. **Full robustness** : 1 of 5 seeds (seed 3) does not replicate the
   alignment cleanly. The 5-seed mean is therefore Spearman ≈ 0.60 ± 0.19,
   dragged down by the outlier ; the 4-seed mean (excluding seed 3) is
   0.72 ± 0.03. We did not investigate the seed-3 failure mode in this
   campaign.

2. **Quantitative equivalence with MBS H6dau** : the hardest-bucket jump
   on RGCN+H6 is +1.37 step, vs +2.75 step on MBS H6dau ; the protocol
   transfers qualitatively but the bucket-separation magnitude is smaller.

3. **The protocol is causally isolated**: we have not ablated the
   individual components (step-aware loss, enriched controller, partial
   freeze, aux features, composite selection) on RGCN — only the full
   bundle has been tested. The MBS H6dau audit identified the same
   limitation.

## 6. Verdict — protocol transfer to RGCN

→ **A_protocol_transfers_with_one_outlier** :
the H6_detached_aux protocol transfers to the RGCN backbone on 4 of 5
seeds, with bucket-alignment strength comparable to MBS H6dau (within
stdev). The protocol fixes the naive-ACT collapse failure mode on the same
backbone (0/5 vs 5/5 collapse). One seed (seed 3) does not replicate the
alignment ; the cross-seed mean Spearman is 0.60 ± 0.19 with the outlier,
0.72 ± 0.03 without.

This closes the future-work item flagged in
`data_audit/DATA_AUDIT_FINAL_REPORT_PATCHED.md` :
> *"The H6 protocol transfers to RGCN."* → **partially demonstrated** :
> the protocol transfers on 4 of 5 seeds and avoids collapse on all 5.
> A single-seed outlier remains.

## 7. Files produced (this Phase 2)

| produit | path |
|---|---|
| training logs / ckpts | `seed{1..5}/stage{1,2}/{run.log,checkpoints,...}` |
| seed configs | `configs/rgcn_h6_stage{1,2}_seed{1..5}.yaml` |
| launcher | `scripts/_run_rgcn_h6_phase2.sh` |
| audit script | `scripts/audit_rgcn_h6_two_stage_controller_vs_required_hops.py` |
| 5-seed per-sample audit CSV | `rgcn_h6_two_stage_per_seed.csv` |
| 5-seed bucket-rows CSV | `rgcn_h6_two_stage_bucket_rows.csv` |
| 5-seed summary JSON | `rgcn_h6_two_stage_summary.json` |
| Phase 1 (seed-1 smoke) report | `PHASE1_SEED1_SMOKE.md` |
| this report | `RGCN_H6_TWO_STAGE_5SEED_REPORT.md` |

## 8. Constraints respected by this campaign

- Did NOT modify any existing H6_detached_aux artefacts.
- Did NOT modify any existing RGCN ACT post-patch artefacts.
- All new artefacts went in `results/claim_strengthening/rgcn_h6_two_stage/`.
- No OOD selection (selection is `val_acc` only, both stages).
- No v3.x architecture work (this is the v1 task only).
- No composite selection redefinition (the baseline `_best.pt` is the
  official `val_acc` checkpoint).
- All 4 code patches P1..P4 are non-destructive on the existing pipeline
  (existing tests 14/14 pass ; existing variants behave unchanged).
