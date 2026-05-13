# H6_REAUDIT_WITH_ORDINAL_METRICS — H6 / RGCN+H6 reproduced under H7 metrics

- date: 2026-05-13
- worktree : `/home/thom315/MBS-halting-h7` (commit `dfb99b0`,
  branch `h7-ordinal-halting`, tag `safety/pre-h7-ordinal-halting-2026-05-13`).
- audit script : `scripts/audit_halting_ordinal_metrics.py` (this
  branch).
- input data : `results/claim_strengthening/rgcn_h6_two_stage/rgcn_h6_two_stage_per_seed.csv`
  (5,120 rows = 5 seeds × 2 splits × 512 samples) +
  `rgcn_h6_two_stage_summary.json` (for `mixture_logits_acc`,
  `floor_mass_mean`, `final_mass_mean`).
- baseline_val_acc supplied to the audit : 0.867 (cross-seed mean
  val_acc of H6 RGCN+H6).

## 1. Facts observed

### 1.a Reproduction of seed-3 soft middle-step collapse

| seed3, split | bucket spread | chosen entropy (bits) | dominant chosen mass | flag |
|---|---:|---:|---:|---|
| val | **0.091** | **0.737** | **0.852** | **soft_middle_step** ✓ |
| ood_mixed | **0.167** | **0.693** | **0.846** | **soft_middle_step** ✓ |

Both criteria of the new `soft_middle_step` definition trigger
(bucket_spread ≤ 0.20 AND entropy < 1.0 ; AND dominant_chosen_mass
≥ 0.80). This reproduces the seed-3 diagnosis from
`results/claim_strengthening/conference_audit/SEED3_DIAGNOSTIC.md`.

### 1.b Reproduction of binary h=9 shortcut on the 4 other seeds

| seed | split | AUC9 | S_easy | flag |
|---:|---|---:|---:|---|
| 1 | val | 1.000 | +0.027 | binary_h9_shortcut ✓ |
| 1 | ood | 1.000 | −0.080 | binary_h9_shortcut ✓ |
| 2 | val | 1.000 | +0.036 | binary_h9_shortcut ✓ |
| 2 | ood | 1.000 | +0.065 | binary_h9_shortcut ✓ |
| 4 | val | 1.000 | +0.046 | binary_h9_shortcut ✓ |
| 4 | ood | 1.000 | −0.025 | binary_h9_shortcut ✓ |
| 5 | val | 1.000 | −0.006 | binary_h9_shortcut ✓ |
| 5 | ood | 1.000 | −0.037 | binary_h9_shortcut ✓ |

8 / 8 cells trigger the `binary_h9_shortcut` flag (`AUC9 ≥ 0.95
AND |S_easy| ≤ 0.10`). This reproduces the diagnosis from
`results/claim_strengthening/conference_audit/SPEARMAN_INFLATION_AUDIT.md`.

### 1.c S_easy is approximately zero on every cell where Spearman ≈ 0.69

| seed | split | S_all | S_easy | gap |
|---:|---|---:|---:|---:|
| 1 | val | +0.729 | +0.027 | 0.702 |
| 1 | ood | +0.689 | −0.080 | 0.769 |
| 2 | val | +0.695 | +0.036 | 0.659 |
| 2 | ood | +0.690 | +0.065 | 0.625 |
| 4 | val | +0.750 | +0.046 | 0.704 |
| 4 | ood | +0.697 | −0.025 | 0.722 |
| 5 | val | +0.709 | −0.006 | 0.715 |
| 5 | ood | +0.708 | −0.037 | 0.745 |
| 3 | val | +0.141 | +0.130 | 0.011 |
| 3 | ood | +0.217 | +0.051 | 0.166 |

On the 4 healthy seeds, the **gap between global Spearman and
within-easy Spearman is ≈ 0.7**, i.e. essentially the entire global
Spearman is the h=9-vs-easy detection. Seed 3 is the only cell
where S_all ≈ S_easy ; that is consistent with seed-3 being a
constant-policy run (the within-easy correlation is small because
the within-easy E[step] values themselves are tiny noise around
≈ 5.1).

### 1.d No cell triggers `ordinal_healthy`

0 / 10 cells of the H6 RGCN+H6 5-seed × 2-split block meet all of :
- no hard collapse,
- no soft middle-step,
- S_easy ≥ 0.15,
- MACRO_AUC ≥ 0.70,
- adjacent_margin_mean > 0,
- adjacent_margin_min ≥ −0.10,
- val_acc within 0.02 of baseline.

The most often-violated criterion is `S_easy ≥ 0.15` (failed in
10 / 10 cells). This is the H7 target metric.

### 1.e MACRO_AUC numbers (new metric)

| seed | val MACRO_AUC | ood MACRO_AUC |
|---:|---:|---:|
| 1 | 0.862 | 0.834 |
| 2 | 0.849 | 0.847 |
| 3 | **0.575** | **0.611** |
| 4 | 0.864 | 0.847 |
| 5 | 0.843 | 0.843 |

Healthy seeds : MACRO_AUC ≈ 0.83–0.86. Seed 3 : 0.58–0.61. The
MACRO_AUC ≥ 0.70 threshold in the H7 gate is met by all healthy
seeds and failed by seed 3, consistent with the qualitative
diagnosis.

### 1.f Bucket means (Δ E[step] at the h=8→h=9 boundary)

| seed | h=5 | h=6 | h=7 | h=8 | h=9 | Δ_89 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 (val) | 3.00 | 3.00 | 3.00 | 3.02 | 4.18 | **+1.16** |
| 2 (val) | 3.03 | 3.04 | 3.04 | 3.04 | 4.93 | **+1.89** |
| **3 (val)** | **5.08** | **5.11** | **5.09** | **5.09** | **5.17** | **+0.08** |
| 4 (val) | 4.10 | 4.19 | 4.11 | 4.07 | 6.86 | **+2.79** |
| 5 (val) | 2.98 | 2.99 | 2.97 | 2.99 | 3.88 | **+0.89** |

The 4 easy buckets agree to within ≤ 0.12 step across all 5 seeds,
val and ood (data above is val ; ood is structurally identical).
Seed 3's Δ_89 = +0.08 step is below the noise of the healthy seeds'
within-easy spread.

## 2. Hypotheses (clearly tagged)

- **H-1 (likely)** : the seed-3 soft middle-step attractor is driven
  by the **specific RGCN ACT post-patch checkpoint used as init** for
  seed 3's RGCN+H6 Stage 1. The seed-3 RGCN ACT post-patch run is the
  one that collapsed to floor or final (we have not yet verified
  which mode), and the H6 enriched controller, trained on top of
  it, settles at a middle step that the backbone state can match
  without effort.
- **H-2 (likely)** : the binary h=9 shortcut on the 4 healthy seeds
  is the path of least resistance under the current H6 loss
  (step-aware latent CE + light ponder). With 5 ordinal buckets but
  only one large compute-payoff (longer chain → harder
  classification), a single threshold suffices.
- **H-3 (plausible)** : an ordinal pairwise ranking loss with
  adjacent-pair sampling will push the controller off the binary
  attractor and into a partial ordinal allocation, **if the loss
  weight is large enough to overcome the easy detector and small
  enough to not dominate the task CE**.
- **H-4 (possible)** : the soft middle-step attractor on seed 3 is
  **independent of init** and is a local minimum of the H6
  controller objective at this Python seed. H7 ordinal loss would
  push out of the local minimum if H-3 holds.

These four hypotheses are not yet tested by the H7 micro-battery
in this turn.

## 3. Reviewer-2 risk under the current H6 framing

| reviewer move | data response |
|---|---|
| "Your Spearman 0.69 is hardest-bucket detection in disguise." | confirmed by the new metrics : S_easy ≈ 0, AUC9 = 1.000. |
| "You count seed 3 as 'no collapse'." | flagged as `soft_middle_step` by the new audit, NOT as `none`. |
| "You report 0/5 collapse but seed 3 is a constant policy." | the new taxonomy gives 0/5 hard collapse, 1/5 soft middle-step. |
| "You haven't decomposed the AUC into multiple thresholds." | MACRO_AUC is now reported per-cell. |
| "You give one bucket-spread number ; show me adjacent margins." | `m_56, m_67, m_78, m_89` + `adjacent_margin_mean / _min` are in the new CSV. |
| "Your selection used OOD." | (still false ; selection is val-only and that does not change here.) |
| "H6 is ordinally healthy in the sense the paper requires." | falsified : 0 / 10 cells meet `ordinal_healthy`. |

Every move that the prior conference_audit raised verbally is now
**machine-flagged in `audits/rgcn_h6_baseline_*` files**.

## 4. Decision

- **Audit reproduces all known H6 failure-mode diagnoses.** The
  ordinal audit script is therefore validated against the existing
  data. **Proceed to H7 design.**
- The `ordinal_healthy` definition is **operational** and rejects
  100 % of current H6 cells. H7 has a non-trivial bar to clear.
- The next step is to **write the H7 implementation** (pairwise
  ranking loss on `expected_steps` + validation-only checkpoint
  gate, both config-gated) and **smoke-test it on a single H6
  config** to confirm backward compatibility.

## 5. Next minimal patch (queued, not done in this turn)

- `mbs/train.py` : add 2 config-gated blocks (`halting_ordinal:` +
  `checkpoint_gate:`). All new behaviour gated by `enabled: true`.
  Backward compatibility test : load
  `configs/rgcn_h6_stage1_seed1.yaml`, assert no new code path
  triggers.
- `scripts/audit_halting_ordinal_metrics.py` already exists and is
  used by both the H6 re-audit (this report) and the upcoming H7
  seed-3 micro-battery.

## 6. Files produced by this re-audit

| file | size |
|---|---:|
| `audits/rgcn_h6_baseline_ordinal_metrics_per_seed_split.csv` | 10 rows, ~5 KB |
| `audits/rgcn_h6_baseline_ordinal_metrics_summary.json` | full per-cell incl. bucket_means dicts |
| `audits/rgcn_h6_baseline_ORDINAL_AUDIT_REPORT.md` | per-cell scoreboard + bucket-means table + notes |

## 7. Coverage gap (declared, not patched here)

The current `audit_halting_ordinal_metrics.py` works on per-sample
data. The two upstream comparators are :

- **MBS H6_detached_aux** : only `data_audit/h6_required_hops_bucket_summary.csv`
  (per-seed × per-bucket aggregates) is available in the committed
  artefacts. The per-sample CSV is not in the repo at HEAD. Therefore
  S_easy, AUC9, MACRO_AUC, chosen_step entropy, and dominant_mass
  are **NOT** computable for MBS H6dau from committed data alone. The
  bucket-level adjacent margins are computable.
- **RGCN ACT post-patch** : `rgcn_act_postpatch_per_seed.csv` is
  available in the main working tree but **not committed** in
  `dfb99b0`. Inside the H7 worktree, this file is therefore
  absent. The naïve baseline cannot be re-audited under the new
  metrics from this worktree alone.

These two gaps are noted ; they do not block H7 implementation.
They suggest a follow-up to either (a) re-export per-sample CSVs
for the MBS H6dau and RGCN_ACT campaigns in a future commit, or
(b) adapt the audit script to consume per-bucket aggregates for
upstream comparisons (with reduced metric set).
