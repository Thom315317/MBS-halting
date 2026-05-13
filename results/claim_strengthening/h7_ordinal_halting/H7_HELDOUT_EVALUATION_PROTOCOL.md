# H7_HELDOUT_EVALUATION_PROTOCOL

- date : 2026-05-13
- worktree : `/home/thom315/MBS-halting-h7`, branch `h7-ordinal-halting`.
- purpose : define the rules for evaluating H7-fixed (variant V3,
  `ordinal_loss_weight = 0.01`) on seeds not used for its
  selection. Binding for any 5-seed-style table that the paper or
  audit produces.

## 1. Seed-3 is the development seed

Seed 3 was used to :

- diagnose the soft middle-step attractor (in the H6 conference audit,
  seed 3 was the only RGCN+H6 cell flagged `soft_middle_step`).
- run the V2 / V3 / V4 / V5 micro-battery (V4 / V5 not launched).
- pick `ordinal_loss_weight = 0.01` as the H7-fixed weight (per the
  decision rule in `H7_FIXED_CANDIDATE.md`).

Seed 3 is therefore **NOT independent evidence that H7-fixed
generalises.** Reporting H7-fixed results including seed 3 as if it
were a held-out seed would be a classic train-on-test leak.

## 2. Held-out evaluation = seeds 1, 2, 4, 5

The held-out evaluation of H7-fixed is on RGCN+H6 seeds **1, 2, 4, 5**.
These are the 4 RGCN+H6 seeds NOT used to choose the H7 weight or to
diagnose the failure mode. They are the genuine generalisation test.

Each held-out seed runs the H7-fixed config :

| | held-out seed N (N ∈ {1, 2, 4, 5}) |
|---|---|
| variant | `rgcn_h7_two_stage` |
| seed | N |
| ordinal_loss_weight | 0.01 |
| margin | 0.15 |
| pair_sampling | adjacent_balanced |
| checkpoint_gate.enabled | true |
| init_from_checkpoint | `results/.../rgcn_h6_two_stage/seedN/stage1/checkpoints/rgcn_h6_two_stage_best.pt` (per-seed) |
| trainable_modules | `[halting_controller, claim_selector_head]` |
| max_epochs | 5 |

## 3. Reporting rules for any 5-seed-style table

When a table aggregates over all 5 seeds, it must either :

(a) **clearly mark seed 3 as the development seed** in the row label
(e.g. `seed=3 (dev)`), or

(b) **split the report into two blocks** :
- 4 held-out seeds {1, 2, 4, 5} as the **primary evidence**, with
  cross-seed mean ± stdev.
- seed 3 as a **separate development-seed row**, with its own
  metrics, NOT averaged in.

Cross-seed means that pool seed 3 with the 4 held-out seeds **without
flagging** are **forbidden** under this protocol.

## 4. OOD usage rule

**OOD `required_hops` is never used for checkpoint selection or
training.** The held-out evaluation uses :

- training : the seed-N v1 train split (with its own
  `required_hops` per sample, fed into the ordinal loss).
- validation : the seed-N v1 val split — the checkpoint gate's
  ordinal metrics are computed here, on val only.
- OOD evaluation : the 4 OOD splits {ood_entity, ood_conflict,
  ood_rule, ood_mixed} are computed at every epoch and at the
  selected checkpoint, but are **never read** by the
  selection logic. OOD is final-evaluation only.

This rule is enforced at the code level by
`mbs/train.py:_collect_val_ordinal_per_sample` (reads only
`loaders["val"]`) and by `mbs/train.py:checkpoint_row_is_better`
(which reads only `gate_eligible` and `val_acc` / `val_loss`).

## 5. Why V4 is not launched as part of the held-out evaluation

Per `H7_FIXED_CANDIDATE.md` §3 :

- V3 already amplifies the h=9 jump beyond what is needed
  (Δ_h9 ≈ +3.10 step on val).
- OOD `S_easy` is already near zero on V3 (0.009) ; the issue is
  OOD bucket distribution shift, not insufficient ordinal pull.
- V4 (weight 0.05) would likely amplify h=9 detection at the cost
  of accuracy or epoch-level stability, without addressing OOD
  ordinal transfer.

V4 remains an optional follow-up under explicit user request.

## 6. Main metrics required per seed × split

The held-out audit must produce, for each of the 4 held-out seeds and
each of the 2 splits {val, ood_mixed}, the following columns :

### Performance

- `mixture_logits_acc` (a.k.a. acc) — final task accuracy.

### Collapse flags (binary)

- `hard_floor`
- `hard_final`
- `soft_middle_step`
- `binary_h9_shortcut`
- `ordinal_healthy`

### Ordinal-calibration metrics

- `S_all` = Spearman(E[step], required_hops) — global rank corr.
- `S_easy` = Spearman(E[step], required_hops | h ≤ 8) — within-easy.
- `AUC9` = AUC(E[step] → 1[h = 9]).
- `MACRO_AUC` = mean of AUC(E[step] → 1[h ≥ θ]) for θ ∈ {6, 7, 8, 9}.

### Bucket structure

- adjacent margins `m_56, m_67, m_78, m_89`.
- `adjacent_margin_mean`, `adjacent_margin_min`.
- per-bucket mean E[step] for h ∈ {5, 6, 7, 8, 9} (the "bucket
  means" table).
- `bucket_spread` = max bucket mean − min bucket mean.

### Halting policy shape

- `chosen_step_entropy_bits` — Shannon entropy of the chosen-step
  distribution (in bits).
- `dominant_chosen_step_mass` — fraction of chosen_step values in
  the dominant bin.

### Per-epoch gate journal

- For every epoch and every seed : `val_acc`, `S_easy`,
  `MACRO_AUC`, `bucket_spread`, `adjacent_margin_mean`,
  `chosen_step_entropy_bits`, `dominant_chosen_step_mass`,
  `gate_eligible`, `gate_reasons`. Source : the per-run
  `*_gate_eligibility.json` written by `_append_gate_journal` in
  `mbs/train.py`.

### Aggregation

Cross-seed means over the **4 held-out seeds only** :

- mean ± stdev of each scalar metric.
- count of seeds in each collapse-flag category (e.g.
  "1 / 4 seeds flagged `binary_h9_shortcut` on val").

A separate development-seed row records seed 3's metrics for
context but is NOT averaged in.

## 7. Verdict templates (binding)

The held-out report `H7_FIXED_RGCN_HELDOUT_4SEED_REPORT.md` must end
with exactly one of :

| verdict label | criterion |
|---|---|
| `A_h7_generalises_cleanly` | ≥ 3 / 4 held-out seeds reach `ordinal_healthy` on val, with no collapse flag triggered. |
| `B_h7_partial_generalisation` | 1–2 / 4 held-out seeds reach `ordinal_healthy` on val, others miss but don't collapse. |
| `C_h7_is_seed3_specific` | 0 / 4 held-out seeds reach `ordinal_healthy` AND a majority retain the H6 failure mode (`binary_h9_shortcut` or, worse, regression to `soft_middle_step`). |
| `D_h7_breaks_baseline` | Cross-seed mean val_acc on held-out seeds drops > 0.02 vs H6 baseline AND ordinal flags do not improve. |
| `E_mixed_inconclusive` | The pattern does not fit A–D ; explicit explanation required. |

The verdict label must be quoted in `H7_FIXED_RGCN_HELDOUT_4SEED_REPORT.md`
§final. No silent extrapolation.

## 8. What the protocol does NOT promise

- It does NOT promise OOD ordinal transfer (per V3 audit, OOD
  retains `binary_h9_shortcut` on seed 3 ; we expect similar on
  most held-out seeds, with possible exceptions).
- It does NOT promise that all 4 held-out seeds will benefit. The
  seed-3 attractor was a specific failure mode ; H7-fixed might
  not change the 4 already-healthy seeds at all, or could push
  them slightly in either direction.
- It does NOT promise compute reduction or wall-clock saving.

## 9. Held-out launch order

The launcher script
`scripts/_run_h7_fixed_heldout_rgcn.sh` will run seeds **1, 2, 4, 5**
in that order, sequentially. Estimated wall-clock : 4 × ~14 min ≈
**56 min** on the existing GPU. The script stops on the first
failure.

Outputs go under :

```
results/claim_strengthening/h7_ordinal_halting/seed1_w001/
results/claim_strengthening/h7_ordinal_halting/seed2_w001/
results/claim_strengthening/h7_ordinal_halting/seed4_w001/
results/claim_strengthening/h7_ordinal_halting/seed5_w001/
```

Each contains `checkpoints/` (gitignored), `run.log` (gitignored),
`rgcn_h7_two_stage_gate_eligibility.json` (small, can be
committed), `*_train_results.json` (gitignored), per-sample audit
CSV (will be produced post-run via `audit_v2_selected_ckpt.py` and
`audit_halting_ordinal_metrics.py`).

## 10. End

This protocol is binding for the held-out evaluation. Any
deviation must be recorded in a separate `H7_DEVIATION.md` file
and explained.
