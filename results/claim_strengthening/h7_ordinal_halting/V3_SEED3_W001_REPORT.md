# V3_SEED3_W001_REPORT — H7 ordinal-calibration on RGCN+H6 seed 3 (weight 0.01)

- date: 2026-05-13
- worktree: `/home/thom315/MBS-halting-h7`, branch `h7-ordinal-halting`,
  HEAD = `465adc4 Fix tokenizer vocab parity required by committed H6 checkpoint`.
- variant V3 : seed 3, ordinal loss weight = **0.01** (2× V2).
- config: `configs/h7_ordinal_halting/rgcn_h7_seed3_w001.yaml`.
- init: same as V2 — `results/.../rgcn_h6_two_stage/seed3/stage1/checkpoints/rgcn_h6_two_stage_best.pt`
  (loaded 0 missing / 0 unexpected).
- training : 5 epochs, partial-freeze on
  `[halting_controller, claim_selector_head]`, wall-clock 14m11s.

## 1. Selected checkpoint

| field | value |
|---|---|
| selected epoch | **5** (out of 5) |
| selection rule | gate-eligible epoch with highest val_acc (epoch 3 and epoch 5 tied at val_acc 0.8555 ; gate prefers eligible epoch → epoch 5) |
| was eligible under val-only gate ? | **YES** |
| `any_epoch_eligible` | **True** (epoch 5 only) |
| `selected_epoch_eligible` | **True** |
| `no_eligible_checkpoint` | **False** |

→ **V3 epoch 5 is the first H6 / H7 cell, across all the campaigns this
session has audited, to PASS the strict val-only gate.**

## 2. Val + OOD metrics (V3 selected checkpoint, epoch 5)

| metric | val | ood_mixed | H6 seed 3 val | H6 seed 3 ood | V2 val | V2 ood |
|---|---:|---:|---:|---:|---:|---:|
| mixture_logits_acc | **0.8555** | **0.8828** | 0.8555 | 0.8613 | 0.8535 | 0.8750 |
| floor_mass_mean | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| final_mass_mean | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| S_all | +0.761 | +0.705 | +0.141 | +0.217 | +0.760 | +0.712 |
| **S_easy** | **+0.153** | +0.009 | +0.130 | +0.051 | +0.150 | +0.032 |
| AUC9 | **1.000** | **1.000** | 0.571 | 0.641 | 1.000 | 1.000 |
| MACRO_AUC | **0.888** | 0.859 | 0.575 | 0.611 | 0.888 | 0.863 |
| bucket_spread | **3.140** | 3.221 | 0.091 | 0.167 | 2.501 | 2.635 |
| adjacent_margin_mean | **+0.785** | +0.805 | +0.023 | +0.042 | +0.622 | +0.650 |
| chosen_step entropy (bits) | **1.856** | 1.860 | 0.737 | 0.693 | 1.905 | 1.862 |
| dominant_chosen_step_mass | 0.559 | 0.568 | 0.852 | 0.846 | 0.494 | 0.518 |
| controller_step_expected_mean | 5.84 | 5.92 | 5.12 | 5.14 | 5.55 | 5.60 |

## 3. Collapse / health flags

| flag | V3 val | V3 ood | V2 val | V2 ood | H6 seed 3 val | H6 seed 3 ood |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| hard_floor | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| hard_final | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| soft_middle_step | ✗ | ✗ | ✗ | ✗ | **✓** | **✓** |
| binary_h9_shortcut | ✗ | ✓ | ✗ | ✓ | ✗ | ✗ |
| **ordinal_healthy** | **✓** | ✗ | ✗ (ε miss) | ✗ | ✗ | ✗ |

→ **V3 val achieves `ordinal_healthy` flag on its own**.

→ V3 ood retains the `binary_h9_shortcut` flag with S_easy = 0.009 (vs
V2 ood S_easy = 0.032). The OOD ordinal alignment did not transfer
strongly ; it is the structural-bucket-detection pattern that the 4
healthy H6 seeds also exhibit on OOD.

## 4. Per-bucket E[step] means

### val

| h | V3 | V2 | H6 seed 3 |
|---:|---:|---:|---:|
| 5 | 5.06 (n=148) | 5.20 (n=148) | 5.08 (n=148) |
| 6 | 5.09 (n=15)  | 5.19 (n=15)  | 5.11 (n=15) |
| 7 | 5.14 (n=148) | 5.28 (n=148) | 5.09 (n=148) |
| 8 | 5.07 (n=19)  | 5.21 (n=19)  | 5.09 (n=19) |
| **9** | **8.20** (n=182) | 7.69 (n=182) | 5.17 (n=182) |

### ood_mixed

| h | V3 | V2 | H6 seed 3 |
|---:|---:|---:|---:|
| 5 | 5.07 | 5.21 | 5.07 |
| 6 | 5.09 | 5.26 | 5.09 |
| 7 | 5.13 | 5.27 | 5.08 |
| 8 | 5.12 | 5.17 | 5.13 |
| **9** | **8.29** | 7.81 | 5.24 |

→ V3 pushes the h=9 allocation further than V2 (8.20 vs 7.69 step on
val) while keeping the easy buckets near-constant. The Δ_h9 has gone
from H6's +0.08 step → V2's +2.52 step → **V3's +3.10 step**. The
ordinal compute allocation at the hardest bucket grew with the larger
ordinal weight, exactly as the dose-response hypothesis predicted.

## 5. chosen_step histogram

### val

| step | V3 | V2 | H6 seed 3 |
|---:|---:|---:|---:|
| 4 | 9 (1.8%) | 1 (0.2%) | 18 (3.5%) |
| **5** | **286 (55.9%)** | 253 (49.4%) | 436 (85.2%) |
| 6 | 35 (6.8%) | 79 (15.4%) | 57 (11.1%) |
| 7 | 19 (3.7%) | 65 (12.7%) | 1 (0.2%) |
| 8 | 108 (21.1%) | 102 (19.9%) | 0 |
| 9 | 53 (10.4%) | 12 (2.3%) | 0 |
| 10 | 2 (0.4%) | 0 | 0 |
| distinct | **7** | 6 | 4 |

### ood_mixed

| step | V3 | V2 | H6 seed 3 |
|---:|---:|---:|---:|
| 4 | 8 | 0 | 8 |
| 5 | 291 | 265 | 433 |
| 6 | 41 | 76 | 71 |
| 7 | 16 | 52 | 0 |
| 8 | 97 | 102 | 0 |
| 9 | 55 | 17 | 0 |
| 10 | 4 | 0 | 0 |
| distinct | 7 | 5 | 3 |

→ V3 has 7 distinct chosen-step values (V2 had 6, H6 had 4) and
shifts more mass to higher steps (especially step 9) while reducing
the step-6 and step-7 mass relative to V2. The "halt at step 5"
default is still the dominant single bin (55.9 % val / 56.8 % ood)
but the controller now explicitly allocates >30 % of its mass to
steps 8–9 on val.

## 6. Per-epoch gate eligibility (val-only)

| epoch | val_acc | S_easy | AUC9 | MACRO_AUC | spread | adj_mean | entropy | dom_mass | eligible | flags |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|:-:|---|
| 1 | 0.8516 | 0.133 | 0.856 | 0.783 | 0.464 | +0.116 | 0.880 | 0.822 | ✗ | **soft_middle_step** |
| 2 | 0.8457 | 0.137 | 0.999 | 0.885 | 1.243 | +0.311 | 1.504 | 0.574 | ✗ | — |
| 3 | 0.8555 | 0.148 | 1.000 | 0.888 | 2.499 | +0.621 | 1.835 | 0.527 | ✗ (S_easy 0.148<0.15) | — |
| 4 | 0.8398 | 0.126 | 0.722 | 0.683 | 0.119 | +0.028 | 0.508 | 0.912 | ✗ | **soft_middle_step** |
| **5** | **0.8555** | **0.153** | **1.000** | **0.888** | **3.140** | **+0.785** | **1.856** | **0.559** | **✓** | **ordinal_healthy** |

V3 is **less stable across epochs** than V2 :

- Epochs 2, 3, 5 are clean (no collapse flag).
- Epochs 1 and 4 fall into `soft_middle_step` (the seed-3 attractor
  comes back when the controller dips below the ordinal pull).
- Epoch 5 is the only fully gate-eligible epoch.

This is the dose-response signature : a stronger ordinal weight
(0.01 vs 0.005) makes the controller move faster, but at the cost
of more epoch-to-epoch variance. The gate correctly picks the
strongest epoch.

V2 (weight 0.005) had 0/5 collapses but 0/5 gate-eligible epochs.
V3 (weight 0.01) has 2/5 collapses but 1/5 gate-eligible epoch.

## 7. V2 vs V3 head-to-head

| dimension | H6 seed 3 | V2 (w=0.005) | V3 (w=0.01) | winner |
|---|---:|---:|---:|---|
| val_acc | 0.8555 | 0.8535 | 0.8555 | V3 ties H6 baseline |
| ood_acc | 0.8613 | 0.8750 | **0.8828** | **V3** (+0.022 vs H6) |
| soft_middle_step (val) | ✓ | ✗ | ✗ | tie |
| binary_h9_shortcut (val) | ✗ | ✗ | ✗ | tie |
| **ordinal_healthy (val)** | ✗ | ✗ (ε) | **✓** | **V3** |
| S_easy val | 0.130 | 0.150 | **0.153** | V3 |
| S_easy ood | 0.051 | 0.032 | 0.009 | V2 (slightly less binary on ood) |
| MACRO_AUC val | 0.575 | 0.888 | 0.888 | tie |
| MACRO_AUC ood | 0.611 | 0.863 | 0.859 | ≈ tie |
| AUC9 val | 0.571 | 1.000 | 1.000 | tie |
| adjacent_margin_mean val | +0.023 | +0.622 | +0.785 | V3 |
| bucket_spread val | 0.091 | 2.501 | **3.140** | V3 |
| chosen_step entropy val | 0.737 | 1.905 | 1.856 | V2 |
| dominant_chosen_mass val | 0.852 | 0.494 | 0.559 | V2 (slightly more multi-modal) |
| checkpoint_gate_eligibility | n/a | 0/5 epochs | 1/5 epoch | **V3** |
| epoch-level stability | n/a | 0 soft_middle_step epochs | 2 soft_middle_step epochs | **V2** (more stable) |

## 8. Decision per user-specified criteria

The user's decision rule (this turn's prompt) :

1. **"If V3 passes the val gate and does not lose >0.02 accuracy:
   V3 becomes the H7-fixed candidate."**
   - V3 passes val gate ✓
   - V3 does not lose >0.02 accuracy : val 0.8555 = H6 baseline 0.8555 (Δ=0), ood 0.8828 > H6 baseline 0.8613 (Δ=+0.022) ✓
   → **V3 becomes the H7-fixed candidate.**

2. "If V3 improves S_easy/MACRO_AUC but still misses the gate: recommend V4."
   - Not applicable (V3 passes the gate).

3. "If V3 worsens accuracy or collapses back to h=9-only behavior: keep V2 as stabilization candidate, but do not claim ordinal health."
   - Not applicable.

4. **"If V3 increases S_easy on val but not OOD, report that as
   validation-only ordinal calibration with weak OOD transfer."**
   - V3 val S_easy = 0.153 (above threshold). V3 ood S_easy = 0.009
     (close to zero, still classified `binary_h9_shortcut`).
   - **This criterion applies.** The honest framing :

> **V3 achieves ordinal-healthy halting on the v1 validation split.
> The same training does not produce ordinal-healthy halting on the
> v1 ood_mixed split : the OOD controller still behaves as a binary
> hardest-bucket detector (AUC9 = 1.000, S_easy ≈ 0.009).
> Validation-only ordinal calibration with weak OOD transfer.**

## 9. Reviewer-2 framing

> "Your `ordinal_healthy` flag is val-only ; OOD remains a binary
> shortcut. You haven't shown ordinal transfer."

Response : **we report it as such**. V3 is val-side-only ordinal-
healthy. OOD ordinal transfer is weak. This is consistent with the
H6 audit's already-published finding that ordinal alignment on v1
is dominated by the hardest-bucket-boundary detection — adding the
ordinal loss tightens this on val but not on the OOD distribution
of bucket sizes (which differs slightly from val).

> "Your V3 epoch 5 is one lucky epoch out of 5."

Response : 1/5 fully eligible, 2/5 with no flag at all (epochs 2, 3,
5). The selected checkpoint is the strict-gate-passing one. We
report all 5 epochs in the gate journal. The decision rule
(pre-registered) is "select highest val_acc among eligible
epochs" — epoch 5 wins by construction, not by hindsight.

> "Why didn't you preregister the 5-seed rerun ?"

The 5-seed rerun is GATED on seed-3 success per H7_PREREGISTRATION
§9. Seed-3 has now passed the gate on V3. The 5-seed rerun is now
launchable as the next step, with **V3 (weight 0.01) as the
H7-fixed config**.

## 10. GO / NO-GO for V4 (weight 0.05)

User explicit instruction : "Do not launch V4 automatically."

Decision per the pre-registration §4 (V5 conditional clause :
"run ONLY if V2..V4 do not degrade accuracy by > 0.02") and the
user's decision rule :

→ **NO-GO for V4 automatic launch.** V3 satisfies the criteria for
H7-fixed candidate. V4 is **OPTIONAL** for dose-response
characterisation but NOT REQUIRED to defend the V3 result.

**Recommendation : freeze V3 as H7-fixed, do not run V4, decide
next step (5-seed rerun, or seed-3 stability replications).**

V4 risk : weight 0.05 = 5× V3. Possible outcomes :

- Improvement : OOD S_easy crosses 0.15, OOD becomes ordinal_healthy
  too. Best case.
- Degradation : the controller over-allocates step 9 to easy
  buckets (a kind of reverse soft-collapse at high steps). Could
  hurt accuracy.
- No change : V3 is already at the limit of this seed.

If the user wants OOD ordinal transfer confirmed before any 5-seed
launch, V4 is informative. If the user wants to freeze and rerun,
V3 is sufficient.

## 11. What this turn produced

| file | purpose |
|---|---|
| `seed3_w001/checkpoints/rgcn_h7_two_stage_best.pt` | V3 best ckpt (gitignored) |
| `seed3_w001/rgcn_h7_two_stage_gate_eligibility.json` | per-epoch gate journal (1/5 eligible, epoch 5) |
| `seed3_w001/rgcn_h7_two_stage_train_results.json` | full history with `checkpoint_gate.any_epoch_eligible=True` |
| `seed3_w001/v3_seed3_w001_per_seed.csv` | 1024 per-sample rows (val + ood) |
| `seed3_w001/v3_seed3_w001_summary.json` | acc / floor / final per split |
| `seed3_w001/rgcn_h7_seed3_w001_ordinal_metrics_*` | audit-script outputs |
| `seed3_w001/rgcn_h7_seed3_w001_ORDINAL_AUDIT_REPORT.md` | audit one-pager |
| this `V3_SEED3_W001_REPORT.md` | top-level narrative |

## 12. Open questions / next steps (not done in this turn)

1. **5-seed rerun of V3 config**. Freeze
   `configs/h7_ordinal_halting/rgcn_h7_seed3_w001.yaml` as the
   template, derive `_seed{1,2,4,5}.yaml`, run them. Expected
   wall-clock: ~70 min on a single GPU (5 × 14 min, sequential).
2. **V4 (weight 0.05) dose-response** on seed 3 only — optional ;
   only if OOD ordinal transfer is required as a paper claim.
3. **Stability replication of V3 on seed 3 with different Python
   seeds** — the V3 epoch-5 win is the only eligible epoch ; an
   `n=3` replicate would tell us whether epoch 5 is reproducibly
   the best or whether it's an epoch-level fluke.
4. **OOD ordinal transfer audit** — why does S_easy drop from 0.153
   (val) to 0.009 (ood) ? Is the OOD bucket distribution different
   from val, or is the controller's calibration genuinely val-bound ?

All four are deferred to user-approval.
