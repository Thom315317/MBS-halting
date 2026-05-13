# H7_FIXED_CANDIDATE

- date : 2026-05-13
- worktree : `/home/thom315/MBS-halting-h7`, branch `h7-ordinal-halting`,
  HEAD = `465adc4 Fix tokenizer vocab parity required by committed H6 checkpoint`.
- decision : **freeze V3 (ordinal_loss_weight = 0.01) as H7-fixed**, with an
  explicit OOD caveat. **NO training launched in this turn.**

## 1. V2 vs V3 vs H6 seed 3 — head-to-head

### val split

| metric | H6 seed 3 | V2 (w=0.005) | V3 (w=0.01) |
|---|---:|---:|---:|
| mixture_logits_acc | 0.8555 | 0.8535 | **0.8555** |
| floor_mass_mean | 0.0000 | 0.0000 | 0.0000 |
| final_mass_mean | 0.0000 | 0.0000 | 0.0000 |
| S_all | +0.141 | +0.760 | **+0.761** |
| **S_easy** | +0.130 | +0.150 (ε miss) | **+0.153** (passes 0.15) |
| AUC9 | 0.571 | 1.000 | **1.000** |
| MACRO_AUC | 0.575 | 0.888 | **0.888** |
| bucket_spread | 0.091 | 2.501 | **3.140** |
| adjacent_margin_mean | +0.023 | +0.622 | **+0.785** |
| Δ E[step] (h=9 − h≤8) | +0.08 step | +2.52 step | **+3.10 step** |
| chosen_step entropy (bits) | 0.737 | 1.905 | 1.856 |
| dominant_chosen_step_mass | 0.852 | 0.494 | 0.559 |
| chosen_step distinct | 4 | 6 | **7** |
| **soft_middle_step flag** | **✓** | ✗ | ✗ |
| binary_h9_shortcut flag | ✗ | ✗ | ✗ |
| **ordinal_healthy flag** | ✗ | ✗ | **✓** |
| gate-eligible epochs | n/a | 0 / 5 | **1 / 5** |

### ood_mixed split

| metric | H6 seed 3 | V2 (w=0.005) | V3 (w=0.01) |
|---|---:|---:|---:|
| mixture_logits_acc | 0.8613 | 0.8750 | **0.8828** |
| S_all | +0.217 | +0.712 | +0.705 |
| **S_easy** | +0.051 | +0.032 | **+0.009** |
| AUC9 | 0.641 | 1.000 | 1.000 |
| MACRO_AUC | 0.611 | 0.863 | 0.859 |
| bucket_spread | 0.167 | 2.635 | 3.221 |
| adjacent_margin_mean | +0.042 | +0.650 | +0.805 |
| Δ E[step] (h=9 − h≤8) | +0.16 step | +2.61 step | **+3.18 step** |
| chosen_step entropy (bits) | 0.693 | 1.862 | 1.860 |
| **soft_middle_step flag** | **✓** | ✗ | ✗ |
| **binary_h9_shortcut flag** | ✗ | ✓ | ✓ |
| ordinal_healthy flag | ✗ | ✗ | ✗ |

## 2. Why V3 is selected as H7-fixed

1. **Passes the validation ordinal-health gate.** V3 epoch 5 is the
   first H6 / H7 cell across this session's audits to satisfy all 8
   gate criteria simultaneously :
   - no `hard_floor`, no `hard_final`, no `soft_middle_step` ;
   - `S_easy ≥ 0.15` (V3 = 0.153) ;
   - `MACRO_AUC ≥ 0.70` (V3 = 0.888) ;
   - `adjacent_margin_mean > 0` (V3 = +0.785) ;
   - `adjacent_margin_min ≥ −0.10` ;
   - `val_acc ≥ baseline − 0.02` (V3 = 0.8555 = H6 baseline).
2. **No accuracy loss.** val Δ vs H6 seed 3 = 0.000 ; ood Δ vs H6
   seed 3 = **+0.022** (V3 ood acc 0.8828 vs H6 ood 0.8613).
3. **`soft_middle_step` eliminated** on both val and ood. The
   seed-3 attractor that the audit script flagged on every H6 /
   RGCN+H6 audit is gone. Bucket spread of E[step] across the 5
   required_hops levels jumps from 0.09 step (H6) to 3.14 step (V3
   val).
4. **OOD accuracy improved** from 0.8613 to 0.8828 — V3 is not just
   a halting fix, it preserves the task signal cleanly.

## 3. Why V4 is NOT launched

V4 would be `ordinal_loss_weight = 0.05`, i.e. 5× V3. Three reasons
not to launch :

1. **V3 already strengthens the h=9 jump beyond what is needed.**
   Δ_h9 = +0.08 step (H6) → +2.52 (V2) → +3.10 (V3, val). The
   per-bucket E[step] at h=9 sits at 8.20 step on val (versus 5.84
   for the V3-mean E[step]). A 5× weight would amplify this further,
   pushing more compute to the hardest bucket without addressing
   the issue identified at item 2 below.
2. **OOD S_easy is already near zero (0.009) and OOD already shows
   `binary_h9_shortcut`.** V4 would test stronger ordinal pull at
   train time, but the OOD distribution shifts the bucket population
   in ways the ordinal loss does not directly control. Stronger
   weight is therefore more likely to amplify h=9 detection than to
   transfer ordinal calibration to OOD.
3. **V3 is already at the stability edge.** V3 has 2/5 epochs that
   fall into `soft_middle_step` (epochs 1 and 4), and only 1/5 fully
   eligible (epoch 5). V4's larger pull is likely to increase that
   instability, not reduce it. The risk profile of V4 is worse than
   the marginal expected gain.

V4 remains a documented option in `H7_PREREGISTRATION.md` §4 and can
be launched on explicit user request, with the caveat that the
expected outcome is "stronger h=9 amplification, possibly at the
cost of accuracy or epoch-level stability".

## 4. Exact H7-fixed config

```yaml
halting_ordinal:
  enabled: true
  use_required_hops: true
  loss_weight: 0.01                  # <-- the H7-fixed value
  margin: 0.15
  pair_sampling: adjacent_balanced
  max_pairs_per_batch: 512
  stop_gradient_expected_step: false

checkpoint_gate:
  enabled: true
  val_only: true
  min_acc_within_best: 0.02
  reject_hard_collapse: true
  reject_soft_middle_step: true
  min_s_easy: 0.15
  min_macro_auc: 0.70
  min_adjacent_margin_mean: 0.0
  min_adjacent_margin_min: -0.10
```

The full config is in
`configs/h7_ordinal_halting/rgcn_h7_seed3_w001.yaml`. For the
5-seed rerun, this template is replicated with `seed: N` and a
matching `init_from_checkpoint` pointing to seed N's H6 Stage-1
best.pt. The non-seed-3 RGCN+H6 Stage-1 ckpts are at
`results/claim_strengthening/rgcn_h6_two_stage/seed{1,2,4,5}/stage1/checkpoints/rgcn_h6_two_stage_best.pt`
in the **main** repo (they are gitignored ; symlinks from the H7
worktree are required at runtime, same pattern as seed 3).

## 5. Reviewer-safe claim

> **"H7 with ordinal_loss_weight = 0.01 repairs the seed-3
> middle-step attractor and passes the validation ordinal-health
> gate on RGCN+H6 seed 3. It does not yet demonstrate OOD
> fine-grained ordinal halting : the OOD controller still behaves
> as a binary hardest-bucket detector (AUC9 ≈ 1.000, S_easy ≈ 0)."**

What this claim **does** assert (and what data supports it) :

- on val : `ordinal_healthy` flag triggered, S_easy = 0.153,
  MACRO_AUC = 0.888, bucket spread = 3.14 step, 7 distinct
  chosen-step values, dom_mass = 0.56, val_acc = 0.8555 (= H6
  baseline within 1e-4).
- on ood : val_acc improved (+0.022) ; OOD ordinal calibration
  fails (S_easy ≈ 0) ; `binary_h9_shortcut` flag retained.
- across epochs : 1/5 fully eligible (epoch 5), 2/5 clean (epochs
  2, 3, 5), 2/5 fall into soft_middle_step (epochs 1, 4). The gate
  correctly selects the only eligible epoch.
- training-time : the ordinal loss is computed on the **training
  split** `required_hops`. **OOD `required_hops` is never used**
  for training or selection.

What this claim **does NOT** assert :

- H7 is robust across seeds. We have run **seed 3 only** at this
  stage. 5-seed rerun is the next experiment.
- H7 produces OOD ordinal calibration. It does not — OOD remains
  binary hardest-bucket detection.
- H7 reduces compute or wall-clock time. Halting weights are still
  computed at every step ; no early-exit is implemented.
- The protocol is "ordinal-emergent". H7 uses generator
  `required_hops` as a supervised calibration signal. Per
  pre-registration §3 (Regime B), the framing must be
  **"ordinal-calibrated using generator metadata"**, not
  **"emergent depth alignment"**.

## 6. Next experiment

**RGCN + H7-fixed on the full 5-seed campaign.**

- variant : `rgcn_h7_two_stage` (already registered in `mbs/train.py`).
- config : the H7-fixed template (`rgcn_h7_seed3_w001.yaml` with
  per-seed substitutions for `seed` and `init_from_checkpoint`).
- init : each seed's RGCN+H6 Stage-1 best.pt (same family of
  symlinks as seed 3).
- compute estimate : 5 seeds × ~14 min ≈ **70 min** of GPU
  (sequential on a single GPU), audit and report inclus.
- expected outputs (per seed) :
  - `results/claim_strengthening/h7_ordinal_halting/seed{N}_w001/checkpoints/rgcn_h7_two_stage_best.pt` (gitignored)
  - `seed{N}_w001/rgcn_h7_two_stage_gate_eligibility.json`
  - `seed{N}_w001/v3_seed{N}_w001_per_seed.csv`
  - `seed{N}_w001/v3_seed{N}_w001_summary.json`
  - `seed{N}_w001/rgcn_h7_seed{N}_w001_ORDINAL_AUDIT_REPORT.md`
- aggregated outputs after all 5 seeds :
  - `H7_FIXED_5SEED_REPORT.md` (cross-seed table, 5-seed Spearman
    distribution, per-seed flag classification, paper-grade
    conclusion).

This 5-seed rerun is the **gated next step** per the
H7 pre-registration §9. It is **not launched in this turn**.

## 7. Metrics required for the 5-seed rerun

Per the user's spec, the 5-seed report must record, for every seed
× split cell :

| metric | source |
|---|---|
| mixture_logits_acc | per-seed audit script |
| `hard_floor` flag | `classify_collapse_flags` |
| `hard_final` flag | same |
| `soft_middle_step` flag | same |
| `binary_h9_shortcut` flag | same |
| **`ordinal_healthy` flag** | same |
| S_all | `compute_gate_metrics` |
| S_easy | same |
| AUC9 | same |
| MACRO_AUC | same |
| adjacent margins (m_56, m_67, m_78, m_89) | same |
| adjacent_margin_mean, adjacent_margin_min | same |
| bucket means (E[step] by required_hops) | same |
| chosen_step entropy (bits) | same |
| (recommended) bucket_spread, dominant_chosen_step_mass | same |
| (recommended) val_acc per epoch + gate eligibility per epoch | training journal `*_gate_eligibility.json` |

These are exactly the metrics already produced by
`scripts/audit_halting_ordinal_metrics.py`. The 5-seed cross-seed
aggregation is a one-line wrapper that calls the script per seed
and unions the per-cell CSV outputs.

## 8. State of the worktree at this freeze

- HEAD : `465adc4`. Three local commits since `dfb99b0` baseline :
  - `1197bd7 Fix committed v1 graph fields required by enriched halting`
  - `0d081b7 Add H7 ordinal-halting audit and seed-3 configs`
  - `465adc4 Fix tokenizer vocab parity required by committed H6 checkpoint`
- V2 and V3 training : both completed. No `*.pt`, `*.log`,
  `*_train_results.json`, `*_epoch_metrics.csv` staged or committed.
- V3 outputs : in `seed3_w001/` (per-sample CSV, audit MD, gate
  journal).
- V2 outputs : in `seed3_w0005/` (same shape).
- H6 historical artefacts : unchanged.
- No `git push` executed. Reflog intact. All choices reversible.

## 9. Files NOT yet staged for commit (decision left to user)

After V2 and V3 we have on disk :

```
?? results/claim_strengthening/h7_ordinal_halting/H7_BLOCKED_GRAPH_PY_DEPENDENCY.md
?? results/claim_strengthening/h7_ordinal_halting/V2_BLOCKED_TOKENIZER_VOCAB_MISMATCH.md
?? results/claim_strengthening/h7_ordinal_halting/V2_SEED3_W0005_REPORT.md
?? results/claim_strengthening/h7_ordinal_halting/V3_SEED3_W001_REPORT.md
?? results/claim_strengthening/h7_ordinal_halting/H7_FIXED_CANDIDATE.md      # this file
?? results/claim_strengthening/h7_ordinal_halting/seed3_w0005/{rgcn_h7_seed3_w0005_*.csv,*.json,*.md, v2_seed3_w0005_*.csv,*.json}
?? results/claim_strengthening/h7_ordinal_halting/seed3_w001/{rgcn_h7_seed3_w001_*.csv,*.json,*.md, v3_seed3_w001_*.csv,*.json, rgcn_h7_two_stage_gate_eligibility.json}
?? scripts/_audit_v2_selected_ckpt.py
?? scripts/_run_v2_seed3_w0005.sh
?? scripts/_run_v3_seed3_w001.sh
```

All small (.md / .csv / .json), all reviewer-relevant. A single
"H7 V2+V3 seed-3 micro-battery" commit would include all of the
above except the two BLOCKED reports (which are session artefacts ;
keep them OR drop them per user preference).

The heavy artefacts under `seed3_w0005/checkpoints/` and
`seed3_w001/checkpoints/` (3 × 853K each per seed) are gitignored
and will not be staged accidentally.

`scripts/_smoke_h7_compat.py` is also modified (the robustness fix
from the Group 0.b context) ; it was already committed in
`465adc4` so the modification visible in `git status` is a stale
artefact — re-check before any future commit.

## 10. What this turn did NOT do

- No training launched (V4 / 5-seed rerun / V6 from-scratch all
  pending user approval).
- No commits created in this turn.
- No `git push` executed.
- No H7 method change.
- No claim about OOD ordinal calibration. The honest framing is
  "validation-only ordinal calibration on seed 3".

## 11. User decisions queued

The user can choose, as the next session step :

| option | action | wall-clock |
|---|---|---:|
| (a) 5-seed rerun of V3-config | launch seeds 1, 2, 4, 5 (seed 3 already done) → ~56 min GPU + ~5 min audit | ~1 h |
| (b) V4 dose-response on seed 3 | weight 0.05, single run, audit | ~14 min |
| (c) V3 stability replication on seed 3 | re-train V3 config with a different Python seed for the loader | ~14 min |
| (d) Commit V2 + V3 artefacts as a single "H7 V2/V3 seed-3 micro-battery" commit | (writing only, no training) | < 5 min |
| (e) Stop ; leave the H7 branch in its current state, no further work | — | 0 |

This file does not pre-commit to any of these — it just freezes V3
as the H7-fixed candidate so the next experiment knows what config
to inherit.
