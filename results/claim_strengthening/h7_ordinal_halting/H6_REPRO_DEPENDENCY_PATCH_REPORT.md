# H6_REPRO_DEPENDENCY_PATCH_REPORT

- date : 2026-05-13
- worktree : `/home/thom315/MBS-halting-h7`, branch `h7-ordinal-halting`.
- scope : reproducibility-dependency patch only ; **NOT an H7
  methodological change**. See `H6_REPRO_DEPENDENCY_PATCH_PLAN.md`.

## 1. Diff summary

```
mbs/graph.py    +26 / -1   (net +25 lines, 7 hunks)
mbs/datasets.py +345 / -0  (net +345 lines, 2 hunks : 7-line if-branch + ~340-line v1 generator block)
```

(The 246-line `mbs/train.py` diff visible in `git diff --stat` belongs
to the H7 patches from the previous turn ; it is NOT part of this
reproducibility-dependency patch.)

## 2. Line-level description

### 2.a `mbs/graph.py`

- Hunk g1 — `EDGE_TYPES["MORE_RELIABLE_THAN"] = 12` (the v1 trust
  chain edge type). The H6 RGCN configs declare `num_edge_types: 13`
  and the trained checkpoints have `rel_linears.12.weight`, both of
  which require this edge type to exist.
- Hunk g2 — `MODE_FOR_EDGE_TYPE[MORE_RELIABLE_THAN] = RESOLVE_CONFLICT`
  so the operation-mode mapping is complete.
- Hunk g3 — allocate `is_query_claim_node` and `claim_value_ids`
  tensors in `collate_graph_samples` (default all-False / `-1`).
- Hunk g4 — read `sample.get("is_query_claim_node")` and
  `sample.get("claim_value_ids")` once per sample.
- Hunk g5 — copy the per-node values into the batch tensors inside
  the existing for-loop.
- Hunk g6 — add the two keys to the returned batch dict.
- Hunk g7 — list the two field names in `permute_sample_nodes` so
  node permutation reorders them along with `repair_labels` and
  `conflict_labels` (defensive ; H6 configs do not use permutation
  but the consistency is cheap).

### 2.b `mbs/datasets.py`

- Hunk d1 — at the top of `build_belief_repair_datasets`, route
  `task == "depth_controlled_latent_halting_probe"` to the new
  `build_depth_controlled_latent_halting_datasets`. 7 lines including
  a 5-line comment block.
- Hunk d2 — append `~340 lines` at the end of the file :
  - section banner explaining the lift,
  - 3 constants (`DEPTH_PROBE_RULE_NAME`, `DEFAULT_DEPTH_BUCKETS`,
    `DEFAULT_K_MAX`),
  - function `_build_depth_probe_sample` (the v1 single-sample
    generator),
  - function `_depth_probe_split_profile`,
  - class `DepthControlledLatentHaltingProbeDataset`,
  - function `build_depth_controlled_latent_halting_datasets`,
  - function `depth_probe_gold_oracle`.

The v1 generator produces samples with all the per-node and metadata
fields the H6 audit script and the committed model expect (`metadata.candidate_ranks_used`,
`metadata.winner_rank`, `metadata.rank_to_source`, per-node
`is_query_claim_node` and `claim_value_ids`, etc.).

## 3. Why no tokenizer / benchmark changes were needed

- `mbs/tokenizer.py` already covers the alphabet used by the v1
  generator. The v1 `SOURCES` pool is `["A", "B", "C", "D", "E", "F",
  "G", "H"]` (already in committed `mbs/datasets.py:36`) and
  `k_max=8`, so the existing tokenizer's source token coverage is
  sufficient.
- `mbs/benchmark.py` is not on the H6 train/eval critical path ; it
  provides aggregation utilities only. No call from `train_one` or
  the audit script triggers it.

## 4. Unrelated changes in dirty main that were DELIBERATELY EXCLUDED

| dirty-main hunk | excluded ? | why |
|---|:-:|---|
| `MORE_RELIABLE_THAN_FORWARD` (id 13) + `MORE_RELIABLE_THAN_BACKWARD` (id 14) in `EDGE_TYPES` | ✓ | v2 / v3 only ; H6 configs use v1's bidirectional `MORE_RELIABLE_THAN` (id 12). |
| `MODE_FOR_EDGE_TYPE` entries for 13 and 14 | ✓ | same |
| `is_winner_query_claim_node` collated tensor | ✓ | not read by committed `mbs/model.py` or `mbs/baselines.py`. |
| `claim_source_is_trusted` collated tensor | ✓ | same |
| `claim_is_rolled_back` collated tensor | ✓ | same |
| `_build_depth_probe_v2_sample`, `_v2_small`, `_v3`, `_v3_1` generators | ✓ | not in scope ; H6 is v1 only. |
| `DepthControlledLatentHaltingProbeV2*` / `V3*` / `V3_1*` dataset classes | ✓ | same |
| `build_depth_controlled_latent_halting_v2_datasets` etc. | ✓ | same |
| Modifications to `build_cells_and_edges` return signature (`+is_query_claim_node, structural_flags`) | ✓ | not needed by v1 generator (which builds its own cells/edges) and not called by it. Would touch belief_repair / adaptive_halting generators unnecessarily. |
| Hard-coded entity / attribute / rule list expansions | ✓ | v2 / v3 hardness ramp ; v1 uses the committed lists unchanged. |
| Extra `randomize_query_claim_order` kwarg propagation through `BeliefRepairDataset` and `build_belief_repair_datasets` arg-list (visible at lines 791–795 of dirty file) | ✓ | only relevant for belief_repair, not v1 (the v1 dataset takes its own `randomize_query_claim_order=True` default). |
| Comments / docstrings unrelated to v1 | ✓ | excluded |

Post-patch grep verification :

```
$ grep -nE '_v2|_v3|_v3_1|depth_probe_v2|MORE_RELIABLE_THAN_FORWARD|MORE_RELIABLE_THAN_BACKWARD' mbs/graph.py mbs/datasets.py
(empty)
```

→ No v2 / v3 / v3.1 code present in either file.

## 5. Post-patch verification (passed)

### 5.a Regression tests

```
$ pytest tests/ -q
14 passed in 4.23s
```

### 5.b Smoke battery (`scripts/_smoke_h7_compat.py`)

```
smoke 1: H6 config unchanged                                  : PASS (6/6 checks)
smoke 2: H7 collate adds required_hops                        : PASS (6/6 checks)
smoke 3: ordinal_pairwise_loss finite + grad                  : PASS (5/5 checks)
smoke 4: compute_loss on H7 batch                             : PASS (3/3 checks)
smoke 5: audit script reproduces H6 diagnosis on committed CSV: PASS (3/3 checks)
ALL SMOKE TESTS PASSED
```

Concretely :

- Smoke 1, the primary test this patch is meant to satisfy : H6
  Stage-1 config loads, builds the RGCN+H6 model, batches flow
  through `collate_graph_samples` carrying the new fields, the
  enriched halting forward reads them without `KeyError`, loss is
  finite (1.5425), and the H7 ordinal block is not triggered.
- Smoke 2 : H7 config triggers the ordinal-aware collate ; batches
  carry both `required_hops` (per H7 patch) AND the v1 fields per
  this reproducibility patch. `required_hops` values observed in
  range [5, 9] as expected for v1.
- Smoke 3 : ordinal pairwise loss returns a finite scalar with
  gradient, all 4 boundaries (5↔6, 6↔7, 7↔8, 8↔9) sample at least
  2 pairs on the test mini-batch.
- Smoke 4 : end-to-end `compute_loss` on an H7 batch yields
  `ordinal_loss`, `ordinal_loss_weighted`, `ordinal_pairs_total`,
  and per-boundary entries in `parts`.
- Smoke 5 : the H6 ordinal-metric audit on the committed RGCN+H6
  per-seed CSV correctly classifies seed 3 as `soft_middle_step`
  on both val and ood, and seeds 1 / 2 / 4 / 5 as
  `binary_h9_shortcut`. 0 / 10 cells reach `ordinal_healthy`.

## 6. State of the H7 worktree after this patch

```
$ git status --short
 M mbs/datasets.py
 M mbs/graph.py
 M mbs/train.py                                       # H7 patches (separate)
?? configs/h7_ordinal_halting/
?? mbs/ordinal_halting.py
?? results/claim_strengthening/h7_ordinal_halting/
?? results/claim_strengthening/rgcn_h6_two_stage/seed3/  # gitignored symlink
?? scripts/_smoke_h7_compat.py
?? scripts/audit_halting_ordinal_metrics.py
```

→ H7 implementation is now unblocked. V2 training is NOT launched
in this turn (per user instruction).
