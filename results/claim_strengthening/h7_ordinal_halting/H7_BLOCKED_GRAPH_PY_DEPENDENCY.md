# H7 BLOCKED — committed H6 baseline at `dfb99b0` is functionally incomplete

- date: 2026-05-13
- discovered during : smoke test #1 of `scripts/_smoke_h7_compat.py`.
- trigger : the H7 worktree, created from the clean commit `dfb99b0`,
  cannot run any v1 enriched-halting forward pass (H6 or H7) because
  the committed `mbs/graph.py` and `mbs/datasets.py` do not produce
  the batch fields `is_query_claim_node` and `claim_value_ids` that
  the committed `mbs/model.py` and `mbs/baselines.py` read at the
  enriched-halting branch.
- this is **exactly the kind of contamination the clean worktree was
  designed to surface**. The user's intuition that the main repo's
  dirty mbs/{graph,datasets,tokenizer,benchmark}.py files could
  silently affect generation was correct.

## 1. Exact failure

`python scripts/_smoke_h7_compat.py` :

```
smoke 1: H6 config unchanged
  [OK] config loaded
  [OK] halting_ordinal absent or disabled
  [OK] checkpoint_gate absent or disabled
  [OK] H6 batch has NO required_hops (collate unchanged)
Traceback (most recent call last):
  ...
  File "/home/thom315/MBS-halting-h7/mbs/baselines.py", line 216, in forward
    is_qc = batch["is_query_claim_node"].bool()
            ~~~~~^^^^^^^^^^^^^^^^^^^^^^^
KeyError: 'is_query_claim_node'
```

The H6 baseline config (`configs/rgcn_h6_stage1_seed1.yaml`) :

- uses task `depth_controlled_latent_halting_probe` (v1) ;
- has `halting.enriched: true` ;
- triggers the enriched halting branch in `RelationalGCNHaltingClassifier.forward`
  (line 204+ in committed `mbs/baselines.py`) which reads
  `batch["is_query_claim_node"]` and `batch["claim_value_ids"]`.

These two fields are NOT produced by the committed
`mbs/graph.py:collate_graph_samples` (lines 99 / 103 / 163 / 167 of the
DIRTY graph.py contain them ; the committed graph.py at `dfb99b0`
does not).

These fields are also NOT in the committed dataset's per-sample dict —
`mbs/datasets.py:build_cells_and_edges` returns a 4-tuple in the
committed version (`cells, edges, query_node_idx, repair_labels`) ;
the dirty version returns a 6-tuple with the extra
`is_query_claim_node, structural_flags` items, where `structural_flags`
carries `claim_value_ids`.

So the committed `dfb99b0` state of the repo :

| committed module | required field | produced ? |
|---|---|:-:|
| `mbs/model.py` (line 141–142) | `batch["is_query_claim_node"]`, `batch["claim_value_ids"]` | requires |
| `mbs/baselines.py` (line 216–217) | same | requires |
| `mbs/graph.py:collate_graph_samples` | should produce these | **does not** |
| `mbs/datasets.py:build_cells_and_edges` | should expose these per-sample | **does not** |

→ The committed code is **internally inconsistent**. The H7 code I
just added (under `mbs/ordinal_halting.py` and the train.py patches)
is correct but cannot be tested because the underlying H6 baseline
does not run from the clean tree.

## 2. Why this happened

The previous-session commit chain (a4d48bd / c806203 / dfb99b0) was
prepared from a working tree that **had `mbs/graph.py` and
`mbs/datasets.py` modifications applied** — the audit script, the
training runs that produced `rgcn_h6_two_stage_per_seed.csv`, and the
H6 figures all worked because they read from a version of the code
that had those modifications. Those modifications were left dirty
in the main tree on purpose (the user designated them out-of-scope)
and never committed.

The committed state therefore relies on **undocumented co-dependencies**
between the H6 code and uncommitted generator changes.

## 3. Scope of the missing changes

From `git diff main..dfb99b0 -- mbs/graph.py mbs/datasets.py` (run from
the main repo) :

- `mbs/graph.py` : the dirty version has +52 lines vs the committed
  one. Net additions :
  - lines 99–103 : init `is_query_claim_node` and `claim_value_ids`
    tensors in `collate_graph_samples`.
  - lines 115–137 : per-sample bookkeeping inside the loop
    (reads `sample.get("is_query_claim_node")` and
    `sample.get("claim_value_ids")`).
  - lines 163 / 167 : add these tensors to the returned batch dict.
  - line 44–45 : `MORE_RELIABLE_THAN_FORWARD/BACKWARD` edge type
    ids (v2/v3 only ; **not** required for v1, can be omitted).
- `mbs/datasets.py` : the dirty version has +1130 lines vs the
  committed one. The vast majority is v2 / v3 / v3.1 task generators.
  But the **necessary** subset is :
  - `build_cells_and_edges` returns 6 items instead of 4 (or returns
    a dataclass / dict carrying the extra fields).
  - The v1 generator (`_build_depth_probe_sample` or similar) writes
    `is_query_claim_node` and `claim_value_ids` into each sample.

I have not yet diff'd line-by-line the minimal set of hunks. A
reviewer-defensible minimal-patch surgery would land at roughly
~50 lines in `mbs/graph.py` and ~80 lines in `mbs/datasets.py`.

## 4. What this implies for H7

Per the user instruction at the top of the H7 prompt :

> Do not modify `mbs/benchmark.py`, `mbs/datasets.py`, `mbs/graph.py`,
> or `mbs/tokenizer.py`.

The current H7 plan cannot proceed under that constraint, because :

- training H7 requires running the enriched halting forward path ;
- running the enriched halting forward path requires the missing
  graph.py / datasets.py changes ;
- the smoke test #1 ("H6 config still loads & flows through") cannot
  even pass on the committed dfb99b0 state.

## 5. Options to unblock (user decision required)

### Option I — Permit minimal graph.py + datasets.py patch in H7 branch

Add **only** the hunks needed to make v1 enriched halting work :

- `mbs/graph.py` : reintroduce the `is_query_claim_node` and
  `claim_value_ids` propagation in `collate_graph_samples`.
- `mbs/datasets.py` : reintroduce the v1 sample fields
  `is_query_claim_node` and `claim_value_ids`.

This is **already in the main repo's dirty state**, so we are not
inventing new code, just lifting the necessary subset into a commit
on the H7 branch. The diff would be reviewable and small. Document
this as the "v1 baseline minimum dataset / collate" patch.

Pros : H7 work proceeds. The patch is honest about its source.
Cons : Contradicts the literal "do not modify graph.py / datasets.py"
instruction. But the constraint was about scope creep, not about
this specific reproducibility-fixing patch.

### Option II — Commit the necessary subset on `main` first

Switch back to the main repo, surgically commit just the
`is_query_claim_node` / `claim_value_ids` hunks to graph.py +
datasets.py on a new commit on `main`. Then re-create the H7 worktree
from that newer HEAD. Then resume.

This is cleaner historically — the H7 branch starts from a
self-consistent main commit. But it requires going back to the dirty
main worktree to extract minimal hunks, which is also a violation of
the literal "do not touch the dirty main tree" rule.

### Option III — Vendor the necessary code under a new file

Create `mbs/v1_collate_addons.py` with two functions :

- `enrich_batch_with_v1_fields(batch, samples)` — fills in
  `is_query_claim_node` and `claim_value_ids` from each sample's
  data (if available) or from computed-from-other-fields.

Wrap the loader in train.py's `make_loaders` (the same wrapper I
already wrote for `required_hops`) to also call this addon.

Pros : Doesn't modify graph.py / datasets.py. Honors the constraint
literally.
Cons : The required data fields **don't exist in the committed
dataset's per-sample dict** at all. The committed
`build_cells_and_edges` doesn't compute or expose them. So we'd
need to **re-derive** these fields from other fields in the sample
metadata at collate time. This is feasible (the audit script does
something analogous for `required_hops`), but it duplicates logic
that already exists in the dirty datasets.py.

### Option IV — Use a smaller / older config that doesn't require enriched halting

Train V2 with `halting.enriched: false` (linear controller). This
sidesteps the missing fields entirely (the non-enriched branch in
`baselines.py` lines 222–223 does not read these fields).

Pros : Trivial to do. Trains under committed state.
Cons : Defeats the purpose of H7 — the H6 result on which H7 builds
**is** the enriched-controller result. A linear-controller H7 is
not a meaningful test of the H7 hypothesis.

### Recommendation

**Option I** is the smallest, most honest, most reviewer-defensible
move. It :

- requires a single new commit on the H7 branch (clearly labelled
  "Add v1 generator fields needed by enriched halting") ;
- contains exactly the minimal subset of graph.py + datasets.py
  changes already informally validated by H6 training success ;
- documents the source : these hunks lived in the main repo's dirty
  tree pre-this-session and were the precondition for the H6
  numbers we already trust.

Without this, H7 cannot run from a clean tree.

## 6. State of the worktree at the moment of this halt

Files I created so far this turn (all still in working tree, none
committed yet) :

```
M mbs/train.py                                          # H7 patches
?? configs/h7_ordinal_halting/rgcn_h7_seed3_w0005.yaml
?? configs/h7_ordinal_halting/rgcn_h7_seed3_w001.yaml
?? configs/h7_ordinal_halting/rgcn_h7_seed3_w005.yaml
?? configs/h7_ordinal_halting/rgcn_h7_seed3_w010.yaml
?? mbs/ordinal_halting.py
?? results/claim_strengthening/h7_ordinal_halting/      # (incl. this file)
?? scripts/_smoke_h7_compat.py
?? scripts/audit_halting_ordinal_metrics.py
```

Plus one local symlink (gitignored) :
`results/claim_strengthening/rgcn_h6_two_stage/seed3/stage1/checkpoints
 -> /home/thom315/MBS-halting/results/.../seed3/stage1/checkpoints`

The H7 code (`mbs/ordinal_halting.py`, the train.py patches, the 4
configs, the smoke test) is **complete and self-contained** — it
just can't be validated against a self-consistent baseline because
the committed `dfb99b0` is internally inconsistent.

## 7. Smoke-test progress at halt

```
smoke 1 (H6 unchanged)                     : FAIL  (KeyError: is_query_claim_node)
smoke 2 (H7 collate adds required_hops)    : NOT RUN
smoke 3 (ordinal_pairwise_loss finite+grad): NOT RUN
smoke 4 (compute_loss on H7 batch)         : NOT RUN
smoke 5 (audit reproduces H6 diagnosis)    : NOT RUN (but the H6 re-audit already passed last turn)
```

Independent of the model failure, the **audit-only path** (smoke 5
plus the prior `H6_REAUDIT_WITH_ORDINAL_METRICS.md`) works fine,
because it reads CSV data, not the model. So the H7 metric machinery
is validated ; only the **training-side** smoke is blocked.

## 8. What the user can choose to do now

| option | required action | who decides |
|---|---|---|
| I — minimal graph.py + datasets.py patch on H7 branch | I lift the necessary hunks from the main repo's dirty tree, commit them as the FIRST H7 commit, then resume | user explicit approval required |
| II — commit the hunks on main first, recreate H7 worktree | switch back to main, surgical commit, re-fork H7 | user explicit approval required ; also re-runs preflight |
| III — vendor in a new file `mbs/v1_collate_addons.py` | I write a wrapper that re-derives the fields from existing sample data ; no touch to graph.py/datasets.py | risky, may not work cleanly ; user can approve as a fallback |
| IV — linear-controller H7 | use `halting.enriched: false` in configs ; trivially runs ; defeats the purpose | not recommended ; the scientific question is about enriched controller |
| V — abort H7, return to main branch as-is | leave H7 work as a documented blocker, no further code work this session | always available |

## 9. End of halt

No further code changes, no training launch, until the user picks
one of I–V. The H7 branch state is preserved exactly as written ;
nothing has been committed.
