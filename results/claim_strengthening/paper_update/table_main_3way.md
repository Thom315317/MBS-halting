# Main results — three-way comparison (5 seeds × 3 configurations, v1)

Values are cross-seed means (and stdev where applicable). All numbers
are pulled directly from existing artefacts ; see "data sources" at the
bottom of this table.

| configuration | backbone | protocol | OOD acc | Spearman(E[step], required_hops) OOD | collapse count | chosen_step_distinct | interpretation |
|---|---|---|---:|---:|---:|---:|---|
| MBS H6_detached_aux | MBS (d_state=96, MLP-depth-2, scatter agg, no attention) | step-aware latent loss + EnrichedAdaptiveHaltingController (MLP `d_state+5→128→64→1`) + 2-stage partial freeze (H4 warmup → H5b co-train of `halting_controller` + `claim_selector_head`) + composite val-only selection ; 5 seeds | **0.843 ± 0.019** | **+0.678 ± 0.032** | **0 / 5** | 3.8 (mean) | non-degenerate bucket-aligned halting policy ; hardest-bucket detection effect (h=9 vs h=8 +2.75 step) ; reference for the methodological claim |
| RGCN ACT post-patch | RGCN (d_model=96, 13 rel_linears, residual via state_norm) | step-aware latent loss + naive linear `AdaptiveHaltingController(d→1)` + single-stage end-to-end + 3-epoch warmup at T=8 then free ACT ; 5 seeds | **0.872 ± 0.013** | **+0.014 ± 0.025** (≈ NaN, var ≈ 0) | **5 / 5** (3 floor + 2 final) | **1.0** (constant) | strong negative control : high task accuracy under naive ACT does NOT imply a non-degenerate halting policy |
| **RGCN + H6 two-stage** (this work) | RGCN (same as above, init from `rgcn_act_postpatch/seed{N}/...best.pt`) | same protocol as MBS H6dau, applied to the RGCN backbone (P1+P2+P3+P4 patches in `mbs/{halting,model,baselines,train}.py`) ; backbone + selector frozen at Stage 1, controller + selector co-trained at Stage 2 ; 5 seeds | **0.869 ± 0.019** | **+0.600 ± 0.192** (5 seeds) / **+0.696 ± 0.009** (4 seeds excl. outlier) | **0 / 5** | 3.4 (mean) | **partial protocol transfer** : 4/5 seeds replicate MBS H6dau alignment range (0.69–0.75) ; 1 outlier (seed 3) without collapse ; protocol fixes collapse mode on the same backbone ; cross-seed robustness weaker than on MBS |

## Per-seed table (RGCN+H6 two-stage)

| seed | val acc | ood acc | sρ(E[s],hops) val | sρ(E[s],hops) ood | chosen_distinct (val) | E[s] (val) | floor / final (val) | collapse |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 0.850 | 0.893 | +0.729 | +0.689 | 3 | 3.42 | 0.000 / 0.000 | none |
| 2 | 0.883 | 0.891 | +0.695 | +0.690 | 4 | 3.65 | 0.000 / 0.000 | none |
| **3** | 0.856 | 0.861 | **+0.141** | **+0.217** | 4 | 5.12 | 0.000 / 0.000 | none (alignment outlier) |
| 4 | 0.859 | 0.854 | +0.750 | +0.697 | 5 | 5.15 | 0.000 / 0.000 | none |
| 5 | 0.889 | 0.846 | +0.709 | +0.708 | 3 | 3.29 | 0.014 / 0.000 | none |

## Bucket means (cross-seed mean of seed means, val split)

| h | MBS H6_detached_aux | RGCN + H6 two-stage |
|---:|---:|---:|
| 5 | 4.69 | 3.64 |
| 6 | 4.60 | 3.67 |
| 7 | 4.68 | 3.64 |
| 8 | 4.61 | 3.64 |
| **9** | **7.37 (+2.76)** | **5.01 (+1.37)** |

Both substrates exhibit the same qualitative pattern (4 easy buckets
indistinguishable, hardest bucket clearly elevated) ; the absolute step
values and the magnitude of the h=9 jump are substrate-dependent.

## Data sources

| value | path |
|---|---|
| MBS H6dau per-seed val/ood acc + val floor/final | `results/final_scientific_package/aggregates/h6_detached_aux_summary.json` (`per_seed`) |
| MBS H6dau per-seed Spearman(E[s], required_hops) | `results/claim_strengthening/controller_required_hops_summary.json` (`per_seed_split`) |
| MBS H6dau per-bucket per-seed | `results/claim_strengthening/data_audit/h6_required_hops_bucket_summary.csv` |
| RGCN ACT post-patch per-seed all metrics | `results/claim_strengthening/rgcn_act_postpatch_summary.json` (`per_seed_split`) |
| RGCN+H6 per-seed all metrics + buckets | `results/claim_strengthening/rgcn_h6_two_stage/rgcn_h6_two_stage_summary.json` |
| 5-seed RGCN+H6 narrative report | `results/claim_strengthening/rgcn_h6_two_stage/RGCN_H6_TWO_STAGE_5SEED_REPORT.md` |
| Paper integration narrative | `results/claim_strengthening/paper_update/RGCN_H6_INTEGRATION_SUMMARY.md` |
| Figures | `paper_update/fig_acc_vs_policy_3way.{png,pdf}`, `fig_collapse_modes_3way.{png,pdf}`, `fig_bucket_alignment_mbs_vs_rgcn_h6.{png,pdf}` |
