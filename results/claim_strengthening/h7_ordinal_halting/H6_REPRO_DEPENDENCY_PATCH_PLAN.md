# H6_REPRO_DEPENDENCY_PATCH_PLAN

- date : 2026-05-13
- worktree : `/home/thom315/MBS-halting-h7`, branch `h7-ordinal-halting`.
- scope of this patch : **NOT an H7 method change**. This is a
  reproducibility-dependency patch to make the committed H6/RGCN+H6
  baseline runnable from the clean worktree. The patch lifts ONLY the
  hunks required to produce `is_query_claim_node` and `claim_value_ids`
  for v1 samples (the fields read by the enriched halting branches in
  `mbs/model.py` and `mbs/baselines.py`).

## 1. Exact missing fields, where consumed, and where they should be produced

### 1.a `batch["is_query_claim_node"]` (bool tensor, shape `(B, max_nodes)`)

- Consumed at :
  - `mbs/model.py:141` (in `MBSModel._forward_adaptive_halting`, enriched
    halting path) : `is_qc = batch["is_query_claim_node"].bool()`
  - `mbs/baselines.py:216` (in
    `RelationalGCNHaltingClassifier.forward`, enriched halting path) :
    same.
- Produced where : in `mbs/graph.py:collate_graph_samples` from each
  sample's `sample["is_query_claim_node"]` list (one bool per node).
- Source of the per-sample list : the v1 generator
  `_build_depth_probe_sample` (`mbs/datasets.py`) — which **does not
  exist in the committed `dfb99b0` tree**, so a corresponding code
  block must be lifted from the main repo's dirty `mbs/datasets.py`.

### 1.b `batch["claim_value_ids"]` (long tensor, shape `(B, max_nodes)`)

- Consumed at :
  - `mbs/model.py:142` : `cvi = batch["claim_value_ids"].long()`
  - `mbs/baselines.py:217` : same.
- Produced where : in `collate_graph_samples` from each sample's
  `sample["claim_value_ids"]` list (one int per node, `-1` for
  non-CLAIM nodes ; for CLAIM nodes, the `VALUES` index of the asserted
  value).
- Source : same v1 generator.

## 2. Scope discovery — the patch surface is larger than the user's "~50/80 lines" estimate

The user's option I assumed the missing hunks were small. Inspection of
`git -C /home/thom315/MBS-halting diff -- mbs/graph.py mbs/datasets.py`
shows :

| file | dirty diff size | minimal subset for v1 enriched halting |
|---|---:|---:|
| `mbs/graph.py` | +52 lines (5 hunks) | **~12 lines** : add `MORE_RELIABLE_THAN: 12` edge type, its MODE_FOR_EDGE_TYPE mapping, and propagate 2 new fields through `collate_graph_samples`. The 3 other fields (`is_winner_query_claim_node`, `claim_source_is_trusted`, `claim_is_rolled_back`) are NOT consumed by the committed model code and are deliberately EXCLUDED. The v2/v3 edge types (id 13 / 14) are also excluded. |
| `mbs/datasets.py` | +1130 lines (huge ; covers v1, v2, v3, v3.1, audit helpers, etc.) | **~360 lines** : add the v1-only generator (`_build_depth_probe_sample`, `_depth_probe_split_profile`, `DepthControlledLatentHaltingProbeDataset`, `build_depth_controlled_latent_halting_datasets`, `depth_probe_gold_oracle`, the 3 v1 constants) and a 2-line if-branch in `build_belief_repair_datasets` to route the task. v2 / v3 / v3.1 generators are EXCLUDED. |

The reason the patch is ~360 lines and not "small" is that the
**entire v1 task generator** is missing from committed `dfb99b0` —
not just the new fields. The committed `build_belief_repair_datasets`
silently falls through to `BeliefRepairDataset` for the v1 task name,
which is a different generator and would not produce
`candidate_ranks_used` / `winner_rank` metadata (i.e. it cannot
reproduce the H6 numbers).

This is a **reproducibility crisis** of the committed H6 baseline,
not an H7 question. The clean worktree successfully surfaced it.

## 3. Exact intended semantics (matching the source in dirty main)

### 3.a `is_query_claim_node`

- A list of bools of length `n_nodes` in each sample.
- Element `i` is `True` iff node `i` is one of the **4 CLAIM cells that
  represent the candidate values for the current query**. False for
  everything else (entity, attribute, value, source, rule, query,
  non-query-claim nodes).
- In v1, every sample has exactly 4 CLAIM cells, one per candidate
  rank in `metadata["candidate_ranks_used"]`. All 4 of them are
  query CLAIMs ; there are no non-query CLAIMs in v1. So
  `is_query_claim_node` is equivalent to `cell_type == CLAIM` on v1.
  (We carry it as an explicit field because v2 / v3 introduce
  non-query CLAIMs.)
- Collated to `(B, max_nodes)` bool tensor, defaulting to False outside
  the per-sample node range.

### 3.b `claim_value_ids`

- A list of ints of length `n_nodes` in each sample.
- For each query CLAIM node : the `VALUES.index(value)` of the
  asserted value (an int in `[0, len(VALUES))` ; for v1
  `len(VALUES) = 8`, so values in `[0, 7]`).
- For non-CLAIM nodes : `-1` sentinel.
- Collated to `(B, max_nodes)` long tensor, defaulting to `-1`.

These match the existing assumptions in the committed model code :

- `mbs/model.py:_forward_adaptive_halting` (lines 144–145) builds an
  `agg_mask = is_claim & is_qc & (cvi >= 0)` mask of nodes that
  contribute to the per-step value-logit aggregation. `is_qc` masks
  out non-query CLAIMs (none in v1, but the model is written generically).
  `cvi >= 0` filters out the `-1` sentinel.
- `mbs/baselines.py:216` does the same for the RGCN forward.

## 4. Why this is a reproducibility dependency, not an H7 method change

- The committed H6 result CSVs (`rgcn_h6_two_stage_per_seed.csv`,
  etc.) contain `required_hops` fields populated for every sample.
  Those numbers were produced by `_derive_required_hops_v1(meta)`
  reading `metadata["candidate_ranks_used"]` and `winner_rank`,
  which only the v1 generator writes.
- The H6 model checkpoints have `rel_linears.0..12.weight` (13
  relational linears) — meaning the model was trained with
  `num_edge_types: 13` AND with edges of type
  `MORE_RELIABLE_THAN` (id 12) actually present in the data.
- The H6 trainings were therefore run on a code base that had this
  same v1 generator + `is_query_claim_node` / `claim_value_ids`
  collate fields + `MORE_RELIABLE_THAN` edge type. That code base
  was not committed. The committed `dfb99b0` state cannot regenerate
  the H6 numbers.
- This patch **does not change** any v1 semantics, edge meanings,
  token meanings, repair / conflict labels, or train / val / OOD
  split logic. It restores the missing code so the committed model
  and committed configs work together.
- H7 does not invent any new field. The H7 ordinal loss reads
  `required_hops` (derived from `metadata`, never collated by
  graph.py) and `expected_steps` (already produced by the model).
  The two new collated fields (`is_query_claim_node`,
  `claim_value_ids`) are required by H6, not by H7.

## 5. Risk assessment

| risk | likelihood | mitigation |
|---|---|---|
| Patch introduces v2 / v3 logic by accident → scope creep | low | Hunks are scoped : only the 5 v1 names (`_build_depth_probe_sample`, `_depth_probe_split_profile`, `DepthControlledLatentHaltingProbeDataset`, `build_depth_controlled_latent_halting_datasets`, `depth_probe_gold_oracle`) + the 3 v1 constants + the 2-line if-branch. Verified by post-patch grep for `_v2`, `_v3`, `_v3_1`. |
| Patched semantics differ subtly from H6 training | low | We are lifting the SAME generator that produced the H6 numbers. The committed model code already assumes this semantics. |
| Patch breaks other committed tests | low | `pytest tests/ -q` re-run after patch (smoke test #1 verifies a forward pass on H6 config). |
| Patched `collate_graph_samples` changes returned dict shape for non-v1 tasks | low | Two new tensor keys are added unconditionally (all-zero / all-`-1` for non-v1 samples where the per-sample fields are absent). Downstream consumers that don't read these keys are unaffected. |
| Patch adds the 3 unused fields (`is_winner_query_claim_node`, `claim_source_is_trusted`, `claim_is_rolled_back`) by accident | low | Explicitly excluded ; verified by post-patch diff. |
| The committed `pytest tests/ -q` passes vacuously because the tests don't cover the enriched halting forward | medium | I will add a separate smoke check that exercises the enriched halting forward path on a small batch and asserts no `KeyError`. |

## 6. Smoke tests (binding)

After applying the patch, `scripts/_smoke_h7_compat.py` must pass all
5 checks :

1. H6 config loads, H6 collate **now produces** `is_query_claim_node`
   and `claim_value_ids`, forward + loss are finite. **This is the
   primary check this patch is designed to satisfy**.
2. H7 config loads and the ordinal-aware collate adds `required_hops`.
3. `ordinal_pairwise_loss` finite + grad.
4. compute_loss on H7 batch yields finite loss + ordinal_loss in parts.
5. Audit script reproduces H6 diagnosis on the committed CSV.

If any of these fails after the patch, I will NOT train. I will
write `H6_REPRO_DEPENDENCY_STILL_BLOCKED.md` with full traceback +
next minimal patch.

## 7. What this patch does NOT do

- Does not add v2 / v3 / v3.1 task generators.
- Does not add `is_winner_query_claim_node`, `claim_source_is_trusted`,
  `claim_is_rolled_back` to the collate (unused by committed model).
- Does not add `MORE_RELIABLE_THAN_FORWARD` / `MORE_RELIABLE_THAN_BACKWARD`
  edge types (v2 only).
- Does not modify `mbs/tokenizer.py`, `mbs/benchmark.py`,
  `mbs/model.py`, `mbs/baselines.py`, `mbs/halting.py`,
  `mbs/train.py`.
- Does not change any output schema of existing H6 artefacts.
- Does not delete or move any existing file.
- Does not launch training.
- Does not commit automatically.

## 8. Sequence

1. (this plan, ✓)
2. Extract minimal hunks → `H6_REPRO_DEPENDENCY_EXTRACTED_DIFF.patch`.
3. Apply patch to `mbs/graph.py` + `mbs/datasets.py` (manual surgical
   edits, NOT `git apply` of the full dirty diff).
4. Verify : `git diff --stat` shows exactly graph.py + datasets.py
   touched, and the line counts roughly match (~12 + ~360).
5. Run `pytest tests/ -q` (must stay 14/14).
6. Run `scripts/_smoke_h7_compat.py` (must reach 5/5).
7. Write either `H7_DECISION.md` (success) or
   `H6_REPRO_DEPENDENCY_STILL_BLOCKED.md` (failure).
