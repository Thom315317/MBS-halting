# H7 — Pre-registration

- date: 2026-05-13
- written: **before any H7 training**, in worktree
  `/home/thom315/MBS-halting-h7`, branch `h7-ordinal-halting`,
  starting from commit `dfb99b0 Update paper claims and figures
  after RGCN H6 transfer`.
- safety tag : `safety/pre-h7-ordinal-halting-2026-05-13` applied
  on this worktree.
- this document is **pre-registered**. Any deviation between this
  plan and what is actually run must be reported as a deviation,
  not silently retrofitted.

## 0. Clean-base guarantee

All H7 experiments are run from a separate clean git worktree
created at commit `dfb99b0`, the committed H6 baseline. The dirty
files (`mbs/benchmark.py`, `mbs/datasets.py`, `mbs/graph.py`,
`mbs/tokenizer.py`) and the 251 historical untracked artefacts
present in the main working tree are deliberately excluded from
the H7 worktree. This prevents local generator / tokenizer /
dataset modifications from contaminating the H7 comparison.

`git status --short` in this worktree is empty at the moment the
H7 work begins.

## 1. Known H6 failure facts (pre-registered scope)

These facts are taken from
`results/claim_strengthening/conference_audit/SEED3_DIAGNOSTIC.md`
and
`results/claim_strengthening/conference_audit/SPEARMAN_INFLATION_AUDIT.md`
(both committed in `dfb99b0`). They are the targets H7 attempts to
fix.

### 1.a Seed 3 — soft middle-step collapse

- OOD Spearman global = +0.217
- OOD AUC(E[step] → 1[h=9]) = 0.641
- OOD Δ E[step] (h=9 − h≤8) = +0.159
- chosen_step entropy = 0.693 bits (ood) / 0.737 bits (val)
- ≈ 84.6 % chosen_step mass on step 5 (ood) / 85.2 % (val)
- bucket spread of E[step] ≤ 0.17 step across h ∈ {5..9}
- floor_mass = 0.0000, final_mass = 0.0000 → escapes the current
  hard-collapse threshold (≥ 0.5) but is operationally a constant
  policy.

### 1.b Seeds 1, 2, 4, 5 — binary h=9 detectors

- global Spearman ≈ 0.69 each
- Spearman on h ≤ 8 (within-easy-buckets) ≈ 0.03 (range
  [−0.08, +0.07])
- AUC(E[step] → 1[h=9]) = **1.000** exactly on all 8 of these 4
  seeds × 2 splits = 8 cells
- mean Δ E[step] (h=9 − h≤8) = +1.18 to +2.77 step depending on
  seed.

→ The 0.69 Spearman is mathematically a binary detector's induced
rank correlation, not bucket-level depth tracking.

## 2. H7 hypothesis

**Adding a small ordinal pairwise ranking loss on E[step] with
adjacent-balanced pair sampling, plus an anti-soft-collapse
validation gate, will :**

- **H7-H1 (necessary)** : produce a non-zero Spearman on the
  within-easy-bucket subset (S_easy ≥ 0.15) without destroying
  task accuracy (val_acc within 0.02 of H6 baseline).
- **H7-H2 (necessary)** : eliminate the soft middle-step attractor
  on RGCN+H6 seed 3, OR provably reject the soft-collapse run via
  the new validation gate.
- **H7-H3 (preferred)** : raise MACRO_AUC (mean AUC over thresholds
  h ≥ 6, 7, 8, 9) above 0.70 on val, indicating ordinal allocation
  not just binary detection.

H7 is **NOT expected** to discover semantic reasoning depth. The
loss uses `required_hops` as **supervised generator metadata** for
calibration. H7 must be reported as **ordinal-calibrated**, not
**emergent**.

## 3. Two regimes, kept separate in code + reports + paper text

### Regime A — Audit-pure (no ordinal loss in training)

- Train H6-baseline-equivalent (no use of `required_hops` in loss,
  no use of `required_hops` in checkpoint selection).
- Use `required_hops` ONLY for offline audit and checkpoint
  diagnosis.
- This is the cleanest evidence that "accuracy is not halting."

### Regime B — Ordinal-calibrated (uses required_hops in training)

- The ordinal pairwise ranking loss on the **val split's**
  `required_hops` is added to the training objective at a small
  weight.
- The validation-only checkpoint gate computes ordinal-health
  metrics on the **val split's** `required_hops`.
- **OOD `required_hops` is never used** for training, checkpoint
  selection, hyperparameter tuning, or early stopping. OOD is
  final evaluation only.
- All paper text must say "ordinal-calibrated using generator
  metadata", never "emergent alignment".

## 4. Exact H7 variants (pre-registered)

The seed-3 micro-battery runs the following 6 variants. They will
all be reported, including failures. **Seed 3 only** at this stage.

| variant | description | new training ? | ordinal loss weight | init |
|---|---|:-:|---:|---|
| V0 | H6 existing baseline, re-audited only with the new ordinal metrics | no | 0 | (existing seed-3 H6 ckpt) |
| V1 | H6 with NEW checkpoint gate only (no new loss) — re-select among H6 epoch checkpoints if per-epoch ckpts exist ; otherwise report that re-selection is impossible | no (gate is post-hoc) | 0 | (existing seed-3 H6 ckpts) |
| V2 | H7 ordinal loss weight 0.005 | yes | 0.005 | same init as RGCN+H6 seed 3 (`rgcn_act_postpatch/seed3/.../best.pt`) |
| V3 | H7 ordinal loss weight 0.01 | yes | 0.01 | same |
| V4 | H7 ordinal loss weight 0.05 | yes | 0.05 | same |
| V5 | H7 ordinal loss weight 0.10 — **conditional** : run ONLY if V2..V4 do not degrade val_acc by > 0.02 vs H6 baseline | conditional | 0.10 | same |
| V6 | H7 from scratch (no RGCN ACT init), ordinal loss weight = best of V2..V4 — **conditional** : run ONLY if V2..V4 produce at least one passing variant | conditional | best of V2..V4 | RGCN random init, full H7 protocol |

Stage-1 + Stage-2 schedule mirrors H6 exactly (5 + 5 epochs,
partial freeze).

## 5. Validation-only checkpoint gate (pre-registered thresholds)

A checkpoint is **eligible** at epoch e iff ALL of :

```
val_acc(e) >= max_val_acc_so_far(e) - 0.02
floor_mass_mean(e)  < 0.5     (no hard_floor)
final_mass_mean(e)  < 0.5     (no hard_final)
NOT soft_middle_step(e)       # bucket_spread > 0.20 OR chosen_step_entropy_bits >= 1.0
S_easy(e)             >= 0.15
MACRO_AUC(e)          >= 0.70
adjacent_margin_mean(e)  > 0
adjacent_margin_min(e)   >= -0.10
```

Among eligible epochs, pick the one with the highest `val_acc`.

If **no** epoch is eligible :

- pick the epoch with the highest `val_acc` (so the run still
  produces a final checkpoint),
- **mark the run as `no_eligible_checkpoint`** in the audit JSON,
- write the failing reason(s) in the audit JSON per epoch,
- do **not** silently relax the thresholds.

All of the above operates on `val` data only. `ood_mixed` is
**never** read during selection.

## 6. Metrics to report per variant × seed × split

### Core metrics (paste-ready columns for the per-seed CSV)

- `mixture_logits_acc` (task accuracy)
- `S_all` = Spearman(E[step], required_hops)
- `S_easy` = Spearman(E[step], required_hops | h ≤ 8)
- `AUC9` = AUC(E[step] → 1[h = 9])
- `delta_h9` = mean(E[step] | h = 9) − mean(E[step] | h ≤ 8)
- `MACRO_AUC` = mean of AUC(E[step] → 1[h ≥ θ]) for θ ∈ {6, 7, 8, 9}
- adjacent margins `m_56, m_67, m_78, m_89`
- `adjacent_margin_mean` and `adjacent_margin_min`
- `bucket_spread` = max bucket mean − min bucket mean
- `chosen_step_entropy_bits`
- `dominant_chosen_step_mass`
- `floor_mass_mean`, `floor_mass_max`
- `final_mass_mean`, `final_mass_max`

### Collapse taxonomy (one label per cell)

- `hard_floor`        iff floor_mass_mean ≥ 0.5 or floor_mass_max ≥ 0.8
- `hard_final`        iff final_mass_mean ≥ 0.5 or final_mass_max ≥ 0.8
- `soft_middle_step`  iff `(bucket_spread ≤ 0.20 AND chosen_step_entropy_bits < 1.0)` OR `dominant_chosen_step_mass ≥ 0.80`
- `binary_h9_shortcut` iff `AUC9 ≥ 0.95 AND |S_easy| ≤ 0.10`
- `ordinal_healthy`   iff none of the above AND `S_easy ≥ 0.15` AND `MACRO_AUC ≥ 0.70` AND `adjacent_margin_mean > 0` AND `adjacent_margin_min ≥ -0.10` AND `val_acc ≥ baseline_val_acc − 0.02`

A cell can match multiple labels (e.g. an h=9 shortcut **and**
soft_middle_step is logically possible). All matched labels are
reported.

## 7. Reviewer-proof rules (binding)

1. **No OOD in selection.** The validation gate uses val only. OOD
   metrics are final-evaluation only.
2. **All variants reported.** V0..V6 each get a row in the
   micro-battery results, including failures. No variant is
   silently dropped.
3. **No post-hoc threshold relaxation.** If no checkpoint meets the
   thresholds, the run is `no_eligible_checkpoint`. The thresholds
   are not relaxed to make a checkpoint eligible.
4. **No metric leakage.** The training-time ordinal loss is computed
   on the **training split** `required_hops`. The validation gate
   uses **validation split** `required_hops`. No OOD reads anywhere
   except final evaluation.
5. **Seed 3 only at this stage.** No full 5-seed rerun is launched
   until seed 3 passes the success criteria in §8.
6. **No retroactive tuning.** Variant weights {0.005, 0.01, 0.05,
   0.10} are pre-registered. They are not adjusted after seeing the
   seed-3 outcome.
7. **No H6 modification.** H6 artefacts, configs, scripts remain
   exactly as committed in `dfb99b0`. H7 adds new files only.

## 8. Seed-3 success criteria (pre-registered ; binding)

A seed-3 H7 variant **passes** iff all of :

- `hard_floor` = false
- `hard_final` = false
- `soft_middle_step` = false
- `S_easy(val) ≥ 0.15` (preferably also on OOD, reported but not
  required)
- `MACRO_AUC(val) ≥ 0.70` (preferably also on OOD)
- `AUC9 ≥ 0.85`
- `adjacent_margin_mean(val) > 0`
- `val_acc` is within −0.02 of H6 seed-3 val_acc (= 0.8555).
  Equivalently, `val_acc ≥ 0.8355`.
- `ood_acc` within −0.02 of H6 seed-3 ood_acc (= 0.8613).
  Equivalently, `ood_acc ≥ 0.8413`.

A variant that meets fewer than all of these is reported but
**does not pass**.

## 9. Decision rule for full 5-seed rerun

- If **no** seed-3 variant passes : **NO-GO** for 5-seed rerun.
  Write `H7_DECISION.md` with the failure analysis and propose the
  next minimal patch (e.g. different aux-feature subset, different
  pair-sampling strategy, different init).
- If **one or more** seed-3 variants pass : **freeze H7-fixed
  config** = the one passing variant with the smallest
  `ordinal_loss_weight`. Then write `H7_DECISION.md` with the
  frozen config and the GO recommendation. **Do not run the 5-seed
  rerun in this session** — record GO, hand off to user.

## 10. Implementation plan (pre-registered file list)

### New files (no overwrites)

- `scripts/audit_halting_ordinal_metrics.py` (new)
- `scripts/run_h7_seed3_microbattery.sh` (new)
- `scripts/make_h7_ordinal_figures.py` (new)
- `configs/h7_ordinal_halting/rgcn_h7_seed3_w0005.yaml` (new)
- `configs/h7_ordinal_halting/rgcn_h7_seed3_w001.yaml` (new)
- `configs/h7_ordinal_halting/rgcn_h7_seed3_w005.yaml` (new)
- `configs/h7_ordinal_halting/rgcn_h7_seed3_w010.yaml` (new)
- `results/claim_strengthening/h7_ordinal_halting/H7_PREREGISTRATION.md` (this file)
- `results/claim_strengthening/h7_ordinal_halting/H6_REAUDIT_WITH_ORDINAL_METRICS.md` (next step)
- `results/claim_strengthening/h7_ordinal_halting/SEED3_MICROBATTERY_RESULTS.md` (after training)
- `results/claim_strengthening/h7_ordinal_halting/H7_DECISION.md` (after training)
- `results/claim_strengthening/h7_ordinal_halting/audits/ordinal_metrics_per_seed_split.csv` (output of audit)
- `results/claim_strengthening/h7_ordinal_halting/audits/ordinal_metrics_summary.json` (output of audit)
- `results/claim_strengthening/h7_ordinal_halting/audits/ORDINAL_AUDIT_REPORT.md` (output of audit)

### Minimal patches (config-gated, backward-compatible)

The following 4 files **may** be patched. All new behaviour is gated
by config keys ; existing H6 configs must continue to work
unchanged.

- `mbs/halting.py` (no necessary patch — the existing
  `EnrichedAdaptiveHaltingController` already returns `halt_prob` ;
  the ordinal loss does not change the controller forward.)
- `mbs/model.py` (the existing `_forward_adaptive_halting` already
  returns `halt_weights`, `expected_steps`. No model-side change
  needed.)
- `mbs/baselines.py` (same — the RGCN forward already returns
  `expected_steps`. No model-side change needed.)
- `mbs/train.py` (the ONLY required code patch) :
  - read `config["halting_ordinal"]` keys.
  - if `enabled = true`, compute the ordinal pairwise ranking loss
    on `outputs["expected_steps"]` and `batch["required_hops"]`
    (training split) ; add to total loss with weight
    `loss_weight`.
  - if `enabled = false` or key absent → existing behaviour
    unchanged.
  - read `config["checkpoint_gate"]` keys ; compute the ordinal
    validation metrics at each epoch and apply the gate during
    selection. If `enabled = false` or key absent → existing
    behaviour unchanged.

### Backward-compatibility test

Before running any H7 variant, run a smoke test that loads any one
of the existing committed H6 configs
(`configs/rgcn_h6_stage1_seed1.yaml` or
`configs/rgcn_h6_stage2_seed1.yaml`) and asserts that
`build_model(...)` + the `train_one` loop run end-to-end **without**
the ordinal-halting or checkpoint-gate code path being activated
(checks `config.get("halting_ordinal", {}).get("enabled", False)` is
False, and same for `checkpoint_gate`).

## 11. Compute and time budget (pre-registered)

- V0 + V1 : 0 GPU time (audit / re-selection only).
- V2 + V3 + V4 : 3 × (Stage 1 + Stage 2) ≈ 3 × 28 min ≈ 1.5 h GPU.
- V5 : conditional, ≈ 28 min if run.
- V6 : conditional, ≈ 28 min (Stage 1 from scratch) + 14 min
  (Stage 2) ≈ 42 min if run.

**Maximum seed-3 micro-battery compute** : V2 + V3 + V4 + V5 + V6
≈ 2.5 h GPU + audits.

## 12. What this pre-registration does NOT cover

- The full 5-seed rerun of H7-fixed. That is gated on §9.
- Any third substrate (GraphSAGE-relational, RGAT). That is a
  separate plan, in `conference_audit/THIRD_SUBSTRATE_PLAN.md`.
- Any change to the v1 task generator. H7 stays on v1 with the
  same generator state that produced the H6 committed artefacts.
- Any change to `λ_ponder`, `min_message_steps`, `max_message_steps`.
  Those are inherited from H6 unchanged.

## 13. Reporting standard (binding)

Every H7 report file (this one, `H6_REAUDIT_*`, `SEED3_MICROBATTERY_*`,
`H7_DECISION`) follows this structure :

1. Facts observed (numbers, no interpretation).
2. Hypotheses (clearly tagged).
3. Reviewer-2 risk (what a hostile reviewer would say).
4. Decision (what we do next).
5. Next minimal patch (if relevant).

No section may claim more than the data supports. Specifically :

- If H7 uses `required_hops` in training :
  say "ordinal-calibrated using generator metadata."
- If H7 improves S_easy :
  say "improves ordinal calibration on the v1 structural audit
  variable."
- If H7 only fixes seed 3 :
  say "mitigates the middle-step attractor in the previously
  failing seed."
- If H7 hurts accuracy : report it directly, do not hide.

## 14. End of pre-registration

This document was written before :

- The ordinal audit script was implemented.
- The H6 re-audit was run.
- Any H7 code patch was made.
- Any H7 training was launched.

No edit to this file is permitted after the seed-3 micro-battery
begins. Any change to the protocol after that point must be
recorded in a separate `H7_DEVIATION.md` file.
