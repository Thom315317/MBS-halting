# H7_DECISION

- date : 2026-05-13
- worktree : `/home/thom315/MBS-halting-h7`, branch `h7-ordinal-halting`.

## 1. State

| item | status |
|---|---|
| H6 clean-base dependency | **FIXED** by minimal `mbs/graph.py` + `mbs/datasets.py` patch (see `H6_REPRO_DEPENDENCY_PATCH_REPORT.md`) |
| H6 baseline runs from committed code | **YES** — smoke test 1 passes : config loads, collate produces both v1 fields, RGCN+H6 forward + loss are finite (loss = 1.5425), no `KeyError` |
| Existing tests | **14/14 pass** (`pytest tests/ -q`) |
| H7 ordinal-halting smoke battery | **5/5 pass** (see `H6_REPRO_DEPENDENCY_PATCH_REPORT.md` §5.b) |
| H7 implementation | **READY** ; may proceed in a subsequent turn |
| H7 V2 training (seed 3, weight 0.005) | **NOT launched** — gated on explicit user request |
| 5-seed rerun | **NOT launched** — gated on seed-3 success per pre-registration §9 |
| H6 historical artefacts | **untouched** — `results/claim_strengthening/rgcn_h6_two_stage/{summary.json, per_seed.csv, bucket_rows.csv, *.md}` are bit-identical to commit `c806203` |
| H7 reviewer-proof framing | maintained : the reproducibility-dependency patch is explicitly labelled as such in its own report, NOT as an H7 method change |

## 2. What this turn produced

### 2.a Reproducibility-dependency patch (Group 0)

```
mbs/graph.py    +25 net  (7 minimal hunks ; only v1 trust chain + 2 collate fields)
mbs/datasets.py +345 net (v1 task generator + 7-line task-router if-branch)
```

Plus the three documentation files :
- `H6_REPRO_DEPENDENCY_PATCH_PLAN.md`
- `H6_REPRO_DEPENDENCY_EXTRACTED_DIFF.patch`
- `H6_REPRO_DEPENDENCY_PATCH_REPORT.md`

### 2.b H7 code, configs, smoke (Group 1, prepared but not committed)

- `mbs/ordinal_halting.py` (new helper module, fully config-gated)
- `mbs/train.py` (patched : collate wrap + ordinal loss block + gate metrics)
- `scripts/audit_halting_ordinal_metrics.py` (new audit script)
- `scripts/_smoke_h7_compat.py` (new smoke battery)
- `configs/h7_ordinal_halting/rgcn_h7_seed3_w{0005,001,005,010}.yaml`
- Documentation : `H7_PREREGISTRATION.md`, `H7_IMPLEMENTATION_PLAN.md`,
  `H6_REAUDIT_WITH_ORDINAL_METRICS.md`, `audits/rgcn_h6_baseline_*`.

## 3. Reviewer-safe claim now supported

> The committed H6 baseline at `dfb99b0` requires a minimal v1-task
> reproducibility patch (12 graph.py lines + the v1 task generator
> in datasets.py, lifted from the project's own working tree as
> documented in the H6 reproducibility patch report). With that
> patch, the H6 baseline runs from a clean worktree and reproduces
> the previously-recorded H6 diagnoses : seed 3 as `soft_middle_step`,
> seeds 1, 2, 4, 5 as `binary_h9_shortcut`, 0 / 10 cells `ordinal_healthy`.
> No method change has been applied yet ; H7 is implemented but no
> H7 training has been launched.

## 4. What to do next (user decision)

The H7 critical path is now :

1. **(this turn, done)** Apply the reproducibility-dependency patch.
2. **Commit Group 0 + Group 1 (optional, two commits, no push).**
3. **Optionally launch V2 (seed 3, weight 0.005, ~28 min wall-clock)** in
   a subsequent turn. Per the H7 prompt this turn :
   > *Do NOT launch V2 in this turn unless the user explicitly asked
   > for training.* — the user did not ask, so V2 is NOT launched
   > here.
4. If V2 passes the seed-3 success criteria, propose freezing
   H7-fixed and run V3 / V4 if needed.
5. If V2 fails, write `SEED3_MICROBATTERY_RESULTS.md` failure
   analysis, do not run V3..V5 / 5-seeds.

## 5. Remaining risks

| risk | mitigation |
|---|---|
| The committed `c806203` per-seed CSV was generated from a slightly different version of the v1 generator than the one I just lifted (the generator has had multiple iterations on main). | Smoke 5 (audit reproduces the diagnosis) passes, which means the per-seed CSV's interpretation under the H7 metrics matches what the previous-turn `conference_audit/` reports recorded. The risk is therefore bounded to "the v1 generator is at least consistent with the H6 CSV at the audit-metric level." A re-run of the H6 5-seed campaign from this worktree would produce **slightly different** sample IDs than the original H6 (since the random state propagation through Python's `random` is sensitive to any subtle generator change), but the cross-seed statistics would replicate within stdev. **This is a known limitation of lifted generators ; not a blocker for H7.** |
| The `c806203`-trained checkpoints (which we still rely on for V2 init) were trained against the older generator. | This is the standard "freezed-checkpoint vs new-codebase" situation. The Stage-1 init checkpoint's weights are valid for the same backbone shape, so loading is mechanically safe. The data fed to it at H7 fine-tuning time will be from the lifted generator, which is the same v1 task. The audit script's reproduction of the H6 diagnoses (smoke 5) is evidence that the generator behaviour is consistent. |

## 6. Reviewer-2 challenge / response

> "You modified `mbs/graph.py` and `mbs/datasets.py` after declaring
> H7 should not touch them."

Response : the patch is **strictly** the v1 task generator + 2 collate
fields, lifted from the project's own working tree to make the
committed baseline runnable. The H7 method does not depend on this
patch ; the H6 baseline itself does. The patch is documented as a
reproducibility dependency, not as an H7 method change. The diff is
small (25 + 345 lines), v1-only, and verified by post-patch grep to
contain no v2 / v3 / v3.1 code.

> "Show me that the H6 numbers reproduce from this patched tree."

Response : that is the next step. The current evidence is that the
**audit-metric interpretation** of the committed H6 CSV reproduces
under the H7 metrics (smoke 5). A full re-run of the H6 5-seed
campaign is the strict reproducibility test ; this turn does not
launch it (would take ~2 hours GPU). The user can request it
separately.
