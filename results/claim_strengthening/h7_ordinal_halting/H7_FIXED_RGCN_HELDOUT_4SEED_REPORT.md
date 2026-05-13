# H7_FIXED_RGCN_HELDOUT_4SEED_REPORT

- date : 2026-05-13
- worktree : `/home/thom315/MBS-halting-h7`, branch `h7-ordinal-halting`,
  HEAD = `b534d41 Prepare H7-fixed held-out RGCN evaluation configs`.
- protocol : `H7_HELDOUT_EVALUATION_PROTOCOL.md` (binding).
- H7-fixed config : `ordinal_loss_weight = 0.01`, val-only gate, full
  thresholds in `H7_FIXED_CANDIDATE.md` §4.
- this report : evaluation of H7-fixed on **4 held-out seeds (1, 2, 4, 5)**,
  with **seed 3 as separately-reported development seed**.

## 1. Held-out seeds table (1, 2, 4, 5) — H7-fixed

### val + ood_mixed per seed

| seed | split | acc | S_all | **S_easy** | AUC9 | **MACRO_AUC** | spread | adj_mean | entropy | dom_mass | **flags** |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | val | 0.8457 | +0.730 | +0.033 | 1.000 | 0.863 | 1.861 | +0.458 | 1.160 | 0.63 | binary_h9_shortcut |
| 1 | ood | 0.8867 | +0.689 | −0.080 | 1.000 | 0.834 | 1.856 | +0.459 | 1.166 | 0.63 | binary_h9_shortcut |
| 2 | val | 0.8809 | +0.697 | +0.041 | 1.000 | 0.850 | 1.364 | +0.341 | 1.204 | 0.68 | binary_h9_shortcut |
| 2 | ood | 0.8945 | +0.695 | +0.081 | 1.000 | 0.850 | 1.341 | +0.335 | 1.152 | 0.69 | binary_h9_shortcut |
| 4 | val | 0.8750 | +0.748 | +0.039 | 1.000 | 0.863 | 3.686 | +0.913 | 1.533 | 0.59 | binary_h9_shortcut |
| 4 | ood | 0.8594 | +0.696 | −0.028 | 1.000 | 0.847 | 3.690 | +0.915 | 1.606 | 0.59 | binary_h9_shortcut |
| 5 | val | 0.8926 | +0.708 | −0.009 | 1.000 | 0.843 | 1.309 | +0.324 | 1.278 | 0.65 | binary_h9_shortcut |
| 5 | ood | 0.8457 | +0.712 | −0.023 | 1.000 | 0.845 | 1.325 | +0.330 | 1.310 | 0.64 | binary_h9_shortcut |

### Held-out cross-seed means (4 seeds × 2 splits)

| metric | val mean | val stdev | ood mean | ood stdev |
|---|---:|---:|---:|---:|
| acc | **0.8736** | 0.020 | **0.8716** | 0.022 |
| S_all | +0.721 | 0.022 | +0.698 | 0.010 |
| **S_easy** | **+0.026** | 0.022 | **−0.013** | 0.064 |
| AUC9 | 1.000 | 0.000 | 1.000 | 0.000 |
| MACRO_AUC | 0.855 | 0.010 | 0.844 | 0.008 |
| bucket_spread | 2.055 | 1.069 | 2.053 | 1.082 |
| adjacent_margin_mean | +0.509 | 0.270 | +0.510 | 0.272 |
| chosen_step_entropy_bits | 1.294 | 0.157 | 1.309 | 0.211 |
| dominant_chosen_step_mass | 0.638 | 0.038 | 0.638 | 0.043 |

### Held-out flag counts

| flag | val count | ood count |
|---|---:|---:|
| hard_floor | 0 / 4 | 0 / 4 |
| hard_final | 0 / 4 | 0 / 4 |
| soft_middle_step | 0 / 4 | 0 / 4 |
| **binary_h9_shortcut** | **4 / 4** | **4 / 4** |
| **ordinal_healthy** | **0 / 4** | **0 / 4** |
| gate-eligible epochs (sum across 4 seeds × 5 epochs = 20) | **0 / 20** | n/a (gate is val-only) |

## 2. Development seed 3 — reported separately (NOT pooled with held-out)

| seed | split | acc | S_easy | AUC9 | MACRO_AUC | flags |
|---:|---|---:|---:|---:|---:|---|
| 3 | val | 0.8555 | **+0.153** | 1.000 | 0.888 | **ordinal_healthy** |
| 3 | ood | 0.8828 | +0.009 | 1.000 | 0.859 | binary_h9_shortcut |

Seed 3 was the development seed where soft_middle_step was the H6
failure mode. H7-fixed (w=0.01) was chosen precisely to repair that
failure on this seed. The seed-3 val cell is the only `ordinal_healthy`
cell across all the H6 / H7 cells audited in this session ; this
result is **train-on-seed-3** evidence, not generalisation.

## 3. H6 vs H7 per-seed comparison (the substantive question)

### val

| seed | H6 acc | H7 acc | Δ | H6 S_easy | H7 S_easy | Δ | H6 flag | H7 flag |
|---:|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 0.8496 | 0.8457 | −0.004 | +0.027 | +0.033 | +0.006 | binary_h9_shortcut | binary_h9_shortcut |
| 2 | 0.8828 | 0.8809 | −0.002 | +0.036 | +0.041 | +0.005 | binary_h9_shortcut | binary_h9_shortcut |
| 4 | 0.8594 | 0.8750 | +0.016 | +0.046 | +0.039 | −0.007 | binary_h9_shortcut | binary_h9_shortcut |
| 5 | 0.8887 | 0.8926 | +0.004 | −0.006 | −0.009 | −0.003 | binary_h9_shortcut | binary_h9_shortcut |
| **(dev) 3** | **0.8555** | **0.8555** | **0.000** | **+0.130** | **+0.153** | **+0.023** | **soft_middle_step** | **ordinal_healthy** |

### ood_mixed

| seed | H6 acc | H7 acc | Δ | H6 S_easy | H7 S_easy | Δ | H6 flag | H7 flag |
|---:|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 0.8926 | 0.8867 | −0.006 | −0.080 | −0.080 | 0.000 | binary_h9_shortcut | binary_h9_shortcut |
| 2 | 0.8906 | 0.8945 | +0.004 | +0.065 | +0.081 | +0.016 | binary_h9_shortcut | binary_h9_shortcut |
| 4 | 0.8535 | 0.8594 | +0.006 | −0.025 | −0.028 | −0.003 | binary_h9_shortcut | binary_h9_shortcut |
| 5 | 0.8457 | 0.8457 | 0.000 | −0.037 | −0.023 | +0.014 | binary_h9_shortcut | binary_h9_shortcut |
| **(dev) 3** | **0.8613** | **0.8828** | **+0.022** | **+0.051** | **+0.009** | **−0.042** | **soft_middle_step** | **binary_h9_shortcut** |

## 4. Gate eligibility per epoch (held-out)

| seed | epoch 1 | epoch 2 | epoch 3 | epoch 4 | epoch 5 | gate-eligible count |
|---:|---|---|---|---|---|---:|
| 1 | binary_h9_shortcut | binary_h9_shortcut | binary_h9_shortcut | binary_h9_shortcut | **soft_middle_step\|binary_h9_shortcut** | 0 / 5 |
| 2 | binary_h9_shortcut | binary_h9_shortcut | binary_h9_shortcut | binary_h9_shortcut | binary_h9_shortcut | 0 / 5 |
| 4 | binary_h9_shortcut | binary_h9_shortcut | binary_h9_shortcut | binary_h9_shortcut | binary_h9_shortcut | 0 / 5 |
| 5 | binary_h9_shortcut | binary_h9_shortcut | binary_h9_shortcut | binary_h9_shortcut | binary_h9_shortcut | 0 / 5 |

Per-epoch S_easy on val (max across epochs per seed) :

| seed | max S_easy across epochs | epoch achieved |
|---:|---:|---:|
| 1 | +0.045 | 3 |
| 2 | +0.049 | 5 |
| 4 | +0.088 | 1 |
| 5 | +0.000 | 4 |

All max S_easy values across all 5 epochs × 4 seeds = 20 cells are
**below the 0.15 threshold** for `ordinal_healthy`.

Notable : seed 1 epoch 5 is the only held-out epoch to trigger the
`soft_middle_step` flag (transient regression to the seed-3 attractor
that does NOT occur at the selected checkpoint epoch 1).

Seed-5 selected ckpt is epoch 1 (val_acc = 0.8887), seed 4 is epoch 5
(val_acc = 0.8750 ; the highest of seed 4's epochs).

## 5. Accuracy preserved (held-out 4-seed mean)

| | H6 baseline (4 held-out seeds mean) | H7 (4 held-out seeds mean) | Δ |
|---|---:|---:|---:|
| val acc | 0.8701 | 0.8736 | **+0.003** |
| ood acc | 0.8706 | 0.8716 | +0.001 |

H7-fixed **does not break the 4 already-healthy seeds**. Accuracy is
preserved (within 0.003 on average across val + ood) and no flag
regression at the selected checkpoint. But also : H7-fixed **does not
move S_easy** on the 4 healthy seeds — the binary_h9_shortcut pattern
that H6 already exhibited remains.

## 6. What does change on held-out seeds ?

Even though no held-out seed becomes `ordinal_healthy`, three structural
changes are visible vs H6 :

1. **bucket_spread grows** on seeds 1, 2, 4 :
   - seed 1 val : H6 1.18 → H7 1.86 (+0.68 step)
   - seed 2 val : H6 1.90 → H7 1.36 (−0.54 step — DECREASES, unique)
   - seed 4 val : H6 2.80 → H7 3.69 (+0.89 step)
   - seed 5 val : H6 0.91 → H7 1.31 (+0.40 step)
2. **chosen_step entropy rises** on every held-out seed :
   - mean across 4 held-out seeds val : H6 1.151 bits → H7 1.294 bits.
3. **adjacent_margin_mean rises** on seeds 1, 4, 5 :
   - mean across 4 held-out seeds val : H6 0.422 → H7 0.509.

But the gate threshold of `S_easy ≥ 0.15` is **not met by any
held-out cell**. The ordinal loss does shape the policy slightly more
multi-modal — but the within-easy-bucket ordering remains random
noise (S_easy ≈ 0 on all 4 held-out seeds, just as in H6).

## 7. Verdict (per H7_HELDOUT_EVALUATION_PROTOCOL.md §7)

### Decision tree

| label | criterion | applies ? |
|---|---|:-:|
| `A_h7_generalises_cleanly` | ≥ 3 / 4 held-out reach `ordinal_healthy` | ✗ (0/4) |
| `B_h7_partial_generalisation` | 1–2 / 4 held-out reach `ordinal_healthy`, no collapse | ✗ (0/4) |
| **`C_h7_is_seed3_specific`** | **0 / 4 held-out reach `ordinal_healthy` AND majority retain H6 failure mode (`binary_h9_shortcut`)** | **✓** |
| `D_h7_breaks_baseline` | cross-seed val_acc drops > 0.02 AND no flag improvement | ✗ (acc within ±0.02 ; held-out acc Δ = +0.003) |
| `E_mixed_inconclusive` | does not fit A–D | ✗ |

### Verdict label

```
C_h7_is_seed3_specific
```

### Plain-language interpretation

**H7-fixed repairs the soft middle-step attractor that H6 produced on
seed 3 (the development seed). It does NOT generalise to the 4
held-out seeds 1, 2, 4, 5.** All 4 held-out seeds remain
`binary_h9_shortcut` after H7-fixed — exactly as they were under H6.
H7 does **not break** the 4 healthy seeds (accuracy preserved within
±0.02, no flag regression), but it does **not improve** them either.

The development-seed → held-out generalisation is **negative** for the
ordinal_healthy criterion. The protocol explicitly required this
distinction by treating seed 3 as development, and the result confirms
the value of that discipline : a naïve 5-seed mean would have shown
"1/5 ordinal_healthy" or, with the soft-middle-step elimination
mistaken for ordinal-healthy by a less strict audit, "robust repair".
Neither claim is supported by the held-out data.

## 8. Why this happened (hypothesis)

The H6 audit established that 4/5 H6 seeds were `binary_h9_shortcut`
and 1/5 was `soft_middle_step`. The H7 ordinal loss pushes E[step]
into a more ordinal arrangement, but :

- on a soft_middle_step seed (seed 3), the policy was sitting at a
  middle-step attractor with ~no compute allocation difference
  across buckets. The ordinal loss has room to move the policy
  toward a 5-bucket structure → V3 reached ordinal_healthy.
- on a binary_h9_shortcut seed (seeds 1, 2, 4, 5), the policy is
  already at a sharply-bimodal "step 3-5 for easy / step 6-8 for
  h=9" attractor. The ordinal loss tries to spread within h≤8 too,
  but the controller's bias for h=9-only detection is already so
  strong (AUC9 = 1.000) that the within-easy gradient cannot
  overcome it.

In short : the H7 ordinal loss can **rescue middle-step collapse**
but cannot **shatter binary detection** that the H6 controller
already settled into.

## 9. Reviewer-safe claim post-held-out

> "H7 (ordinal_loss_weight = 0.01) repairs the soft middle-step
> attractor that H6 produced on seed 3 of the RGCN+H6 5-seed
> campaign. It does not improve the 4 other RGCN+H6 seeds, which
> remained `binary_h9_shortcut` both under H6 and under H7, with
> accuracy preserved within ±0.02. We therefore characterise H7 as
> **a stabilisation / repair patch for the seed-3 attractor**, not a
> protocol that achieves OOD fine-grained ordinal halting nor that
> generalises across seeds."

Forbidden claims (Reviewer 2 would catch) :

- "H7 produces 1/5 ordinal_healthy on RGCN+H6 5-seed" — this
  conflates the development seed with held-out seeds.
- "H7 generalises across seeds" — falsified by the 0/4 held-out
  ordinal_healthy count.
- "H7 fixes the binary_h9_shortcut failure mode" — H7 leaves the 4
  healthy H6 seeds as binary_h9_shortcut.

## 10. GO / NO-GO for next experimental moves

| move | recommendation | reason |
|---|---|---|
| **Launch V4 (weight 0.05) on seed 3** | NO-GO | V3 already saturates seed 3 ; V4 is more likely to hurt than help. |
| **5-seed rerun of V3 config (already done implicitly)** | DONE | held-out 4 seeds + dev seed 3 = full 5-seed picture. No additional rerun needed. |
| **MBS H7 evaluation** | conditional GO | the question shifts from "does H7 fix the failure modes ?" to "does the same ordinal_loss + gate apply MBS+H6's residual soft-collapse-like patterns ?" MBS+H6 has 0/5 collapse already, so the gain is unclear ; the audit might still find that MBS+H6 cells are binary_h9_shortcut (untested under H7 metrics — needs per-sample MBS CSV which is NOT in this worktree at HEAD). |
| **3rd substrate (GraphSAGE-relational)** | NO-GO yet | the H7 protocol is now characterised as a seed-3 repair patch on RGCN. Adding a 3rd substrate would not strengthen the case for H7 ordinal calibration ; it would expand the H6 audit scope (which is a different experimental line). |
| **Component-level H7 ablation on seed 3** | OPTIONAL | which H7 piece (ordinal loss vs gate-based selection) is necessary ? With a passing seed-3 only and 0/4 held-out passes, the case for component ablation is weaker than before. |
| **Stop the H7 experimental line** | DEFENSIBLE | given the C verdict, H7 is honestly characterised as "fixes seed-3 soft-collapse, does not generalise." The paper can use H7 as a **methodological case study** : even with the audit infrastructure right, a single development seed cannot prove generalisation. This is itself a publishable finding. |

My personal recommendation : **stop the H7 experimental line here**.
The held-out data is decisive : H7-fixed is a seed-3 repair, not a
protocol. Further H7 work (V4, V6, MBS-H7) without a redesign of the
ordinal loss would be confirming the negative.

If the user wants to continue the H7 family, the productive next step
is **NOT more variants of the current ordinal loss**. It is a redesign
that explicitly targets the binary_h9_shortcut → multi-bucket
transition (e.g. : a per-bucket-pair contrastive loss with higher
weight on within-h≤8 pairs ; or an explicit MACRO_AUC-based loss).
That is a methodological branch, not a hyperparameter sweep.

## 11. Files produced

| file | path |
|---|---|
| this report | `H7_FIXED_RGCN_HELDOUT_4SEED_REPORT.md` |
| per-seed per-sample CSV (× 4) | `seed{N}_w001/v3_seed{N}_w001_per_seed.csv` |
| per-seed summary JSON (× 4) | `seed{N}_w001/v3_seed{N}_w001_summary.json` |
| per-seed ordinal-metric audit MD (× 4) | `seed{N}_w001/rgcn_h7_seed{N}_w001_ORDINAL_AUDIT_REPORT.md` |
| per-seed ordinal-metric CSV (× 4) | `seed{N}_w001/rgcn_h7_seed{N}_w001_ordinal_metrics_per_seed_split.csv` |
| per-seed gate journal (× 4) | `seed{N}_w001/rgcn_h7_two_stage_gate_eligibility.json` |
| audit script | `scripts/_audit_h7_heldout.py` |
| training launcher | `scripts/_run_h7_fixed_heldout_rgcn.sh` |

Heavy artefacts (`*.pt`, `run.log`, `*_train_results.json`,
`*_epoch_metrics.csv`) are gitignored or excluded as training
journal ; not staged for commit.

## 12. Recommended commit message

```
H7-fixed held-out 4-seed evaluation : seed-3-specific repair
```

Commit body :

```
4-seed held-out evaluation of H7-fixed (ordinal_loss_weight = 0.01)
on RGCN+H6 seeds 1, 2, 4, 5. Seed 3 (development) reported
separately.

Verdict : C_h7_is_seed3_specific.

Held-out results :
  - 0/4 seeds reach ordinal_healthy on val.
  - 4/4 seeds retain binary_h9_shortcut classification (same as
    H6 baseline).
  - val_acc Δ vs H6 = +0.003 (preserved), ood_acc Δ = +0.001.
  - bucket_spread, chosen_step_entropy, adjacent_margin_mean
    slightly increased but not enough to push S_easy above 0.15.

Dev seed 3 (for context, not pooled in held-out mean) :
  - reaches ordinal_healthy on val.
  - OOD remains binary_h9_shortcut.
  - val_acc 0.8555 (= H6), ood_acc 0.8828 (+0.022 vs H6).

H7 honestly characterised as a soft-middle-step REPAIR patch on the
specific RGCN+H6 failure seed, not a protocol that achieves
fine-grained ordinal halting across seeds. See
results/.../H7_FIXED_RGCN_HELDOUT_4SEED_REPORT.md §7.
```
