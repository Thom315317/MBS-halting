# H6_REPRO_DEPENDENCY_DIFF_AUDIT

- date : 2026-05-13
- worktree : `/home/thom315/MBS-halting-h7`, branch `h7-ordinal-halting`.
- purpose : pre-commit reviewer-proof audit of the
  `mbs/datasets.py` (+345 lines) and `mbs/graph.py` (+25 nets) patch.
  Verifies that the diff contains **only** the minimal hunks required
  by the H6 reproducibility dependency, and **no** v2 / v3 / v3.1 /
  tokenizer / benchmark / target-label / split / required_hops
  formula changes.

## 1. Functions changed in `mbs/datasets.py`

Two hunks total ; one function modified, one new code block appended.

### 1.a Modified : `build_belief_repair_datasets`

- **Diff hunk** : `@@ -781,6 +781,12 @@` inserts 6 new lines just after
  the function header.
- **Old behaviour** : the function dispatched
  - `task == "adaptive_halting_probe" or "belief_repair_difficulty_ladder"` → `build_adaptive_halting_probe_datasets`
  - all other tasks → fall through to `BeliefRepairDataset` (incl.
    `depth_controlled_latent_halting_probe`, which is **wrong** —
    BeliefRepair is a different task with different sample shape and
    no `candidate_ranks_used`, `winner_rank` metadata).
- **New behaviour** : adds a single new `if` at the very top :
  `if task == "depth_controlled_latent_halting_probe": return build_depth_controlled_latent_halting_datasets(config, tokenizer)`.
  All other code paths in this function are unchanged.
- **Why required** : without this routing, `task =
  "depth_controlled_latent_halting_probe"` (used by every H6 RGCN+H6
  config) silently falls through to `BeliefRepair`, which cannot
  produce `is_query_claim_node` / `claim_value_ids` and cannot
  produce the `metadata.candidate_ranks_used` / `winner_rank` fields
  the audit script reads.
- **v1 semantics affected** : no — the new branch returns the v1
  dataset, which IS the one the H6 numbers were generated from.
- **v2 / v3 / v3.1 logic touched** : no.

### 1.b New (appended) : v1 task generator block

- **Diff hunk** : `@@ -838,3 +844,342 @@` appends 339 new lines after
  the last existing function. (Plus 3 blank lines / banner = 342 net
  lines in the second hunk ; 345 total including the +6 from hunk 1.a.)
- **What is added** :
  - 3 constants : `DEPTH_PROBE_RULE_NAME = "trusted_chain_top_wins"`,
    `DEFAULT_DEPTH_BUCKETS = [2, 4, 6, 8]`, `DEFAULT_K_MAX = 8`.
  - Function `_build_depth_probe_sample(rng, depths, k_max,
    value_pool, source_pool, entity_pool, attribute_pool,
    randomize_query_claim_order=True)` — the v1 single-sample
    generator. Builds a graph with ENTITY+ATTRIBUTE+VALUE×4+SOURCE×k_max
    +RULE+QUERY+CLAIM×4 nodes ; emits `MORE_RELIABLE_THAN` edges along
    the trust chain ; writes per-node fields `is_query_claim_node`,
    `claim_value_ids`, `is_winner_query_claim_node`,
    `claim_source_is_trusted`, `claim_is_rolled_back`, `repair_labels`,
    `conflict_labels` ; writes metadata with `winner_rank`,
    `runner_up_rank`, `candidate_ranks_used`, `rank_to_source`,
    `oracle_depth`, etc.
  - Function `_depth_probe_split_profile(split)` — returns
    `DEFAULT_DEPTH_BUCKETS` for every split (uniform across train /
    val / OOD).
  - Class `DepthControlledLatentHaltingProbeDataset(Dataset)` —
    eagerly materialises self.size samples via
    `_build_depth_probe_sample`.
  - Function `build_depth_controlled_latent_halting_datasets(config,
    tokenizer=None)` — returns the train / val / ood_{entity,
    conflict, rule, mixed} dict, each backed by the v1 Dataset.
  - Function `depth_probe_gold_oracle(sample)` — non-neural oracle
    used by audits only ; returns the rank-1 source's value.
- **Why required** : these symbols are needed by hunk 1.a's routing.
  Without them, `task = "depth_controlled_latent_halting_probe"`
  would still fail at runtime even with hunk 1.a's `if` branch.
- **v1 semantics affected** : yes (intentional — this IS the v1
  task definition). Matches the version that generated the
  committed H6 per-seed CSV (verified by smoke 5 audit
  reproduction).
- **v2 / v3 / v3.1 logic touched** : no. All v2 / v3 / v3.1
  generators / dataset classes / builder functions / constants
  from the dirty main are EXCLUDED.

## 2. Functions changed in `mbs/graph.py`

7 minimal hunks across 2 module-level dicts and 2 functions.

### 2.a Modified : `EDGE_TYPES` dict (module-level)

- **Diff hunk** : `@@ -29,6 +29,11 @@` adds 5 lines after
  `"TEMPORAL_NEXT": 11`.
- **Old behaviour** : 12 edge type names mapped to ids 0..11.
- **New behaviour** : adds `"MORE_RELIABLE_THAN": 12` (the v1
  bidirectional trust-chain edge type).
- **Why required** : v1 emits edges of this type along the rank-to-rank
  trust chain. Without it, `_build_depth_probe_sample` raises
  `KeyError` on `EDGE_TYPES[edge["type"]]` at line 1306.
- **v1 semantics affected** : no — adds a new edge type ID, does not
  reinterpret existing ids 0..11.
- **v2 / v3 / v3.1 logic touched** : no — does NOT add
  `MORE_RELIABLE_THAN_FORWARD` (id 13) or `_BACKWARD` (id 14).

### 2.b Modified : `MODE_FOR_EDGE_TYPE` dict (module-level)

- **Diff hunk** : `@@ -51,6 +56,7 @@` adds 1 line.
- **Old behaviour** : 12 edge ids mapped to operation modes.
- **New behaviour** : adds
  `EDGE_TYPES["MORE_RELIABLE_THAN"]: OPERATION_MODES["RESOLVE_CONFLICT"]`
  so the v1 trust-chain edges are assigned the same mode the model
  expects (RESOLVE_CONFLICT — consistent with their semantic role).
- **Why required** : `collate_graph_samples` line 124 does
  `mode_labels[batch_idx, edge_idx] = MODE_FOR_EDGE_TYPE[edge_type]`.
  Missing the v1 entry would raise `KeyError`.
- **v1 semantics affected** : no — adds a new id-to-mode entry, does
  not change existing entries.
- **v2 / v3 / v3.1 logic touched** : no.

### 2.c Modified : `collate_graph_samples` function

Four contiguous hunks inside the function body :

#### 2.c.i Tensor allocation block — `@@ -79,6 +85,11 @@`

- **Old** : 4 allocations (cell_type_ids, cell_token_ids, node_mask,
  ...).
- **New** : also allocates `is_query_claim_node = torch.zeros(B,
  max_nodes, dtype=torch.bool)` and `claim_value_ids =
  torch.full((B, max_nodes), -1, dtype=torch.long)`. **No tokenizer
  call, no benchmark call**.
- **Why required** : the model's enriched halting branch reads
  these two tensors from the batch.
- **v1 semantics affected** : no — defaults (all-False / -1) are
  no-ops for non-v1 samples that don't carry the per-sample lists.

#### 2.c.ii Per-sample field reads — `@@ -90,6 +101,8 @@`

- **Old** : reads `sample_repair_labels = sample.get("repair_labels")`.
- **New** : also reads `sample_is_query_claim_node = sample.get("is_query_claim_node")`
  and `sample_claim_value_ids = sample.get("claim_value_ids")`. Both
  use `.get(...)` defaulting to `None` so missing-field samples
  (non-v1) are handled cleanly.
- **Why required** : feeds the per-node loop below.
- **v1 semantics affected** : no.

#### 2.c.iii Per-node propagation — `@@ -98,6 +111,10 @@`

- **Old** : copied `repair_labels` per node.
- **New** : also propagates `is_query_claim_node[batch_idx, node_idx]`
  and `claim_value_ids[batch_idx, node_idx]` for each node. Each is
  guarded by `is not None and node_idx < len(...)` for back-compat
  with samples that don't carry these lists.
- **Why required** : moves per-sample lists into batch tensors.
- **v1 semantics affected** : no — adds 2 conditional writes to 2
  new tensors. Does not modify `repair_labels` / `cell_type_ids` /
  `cell_token_ids` writes.

#### 2.c.iv Returned dict — `@@ -123,6 +140,8 @@`

- **Old** : returned 14 keys.
- **New** : returns 16 keys ; the 2 new ones are `is_query_claim_node`
  and `claim_value_ids`.
- **Why required** : downstream consumers (`mbs/model.py:141`,
  `mbs/baselines.py:216`) need these keys in `batch`.
- **v1 semantics affected** : no — adds 2 keys, removes nothing.

### 2.d Modified : `permute_sample_nodes` function

- **Diff hunk** : `@@ -177,7 +196,12 @@` extends the tuple of
  field-name strings from `("repair_labels", "conflict_labels")` to
  `("repair_labels", "conflict_labels", "is_query_claim_node",
  "claim_value_ids")`.
- **Old behaviour** : when node order was permuted, only
  `repair_labels` and `conflict_labels` were also reordered.
- **New behaviour** : the 2 new v1 per-node fields are reordered
  alongside them. Permutation correctness for v1 samples.
- **Why required** : defensive ; not actually exercised by H6 RGCN
  configs (which don't enable node permutation). Kept for
  consistency with the v1 generator's semantics.
- **v1 semantics affected** : only when permutation is enabled,
  which is off in every committed H6 config.

## 3. Explicit confirmations

| confirmation | result | how verified |
|---|:-:|---|
| no tokenizer behaviour changed | ✓ | the only `tokenizer.` references in the patched diff are pre-existing `tokenizer.encode(...)` calls at lines 75 / 108 / 127 of `mbs/graph.py`, NOT modified by this patch ; `mbs/tokenizer.py` is not in `git status --short` |
| no benchmark behaviour changed | ✓ | `mbs/benchmark.py` is not in `git diff --name-only` of this patch |
| no train / val / OOD split changed | ✓ | `_depth_probe_split_profile` returns `DEFAULT_DEPTH_BUCKETS` uniformly for every split (matches the version that generated the H6 numbers, where OOD differences came from seed only) |
| no target-label semantics changed | ✓ | `answer_class = VALUES.index(answer_value)` is the same formula as in every other generator in this file ; `target_value_id` collation is unchanged |
| no required_hops formula changed | ✓ | the v1 generator writes `metadata.candidate_ranks_used` and `metadata.winner_rank` ; the audit's `_derive_required_hops_v1(meta) = max(crs) - wr + 2` formula (`scripts/audit_rgcn_h6_two_stage_controller_vs_required_hops.py:60`) is unchanged in this patch |
| no edge schema changed except `MORE_RELIABLE_THAN: 12` | ✓ | grep against the patched files shows no `MORE_RELIABLE_THAN_FORWARD` or `_BACKWARD` (v2 only) ; existing ids 0..11 are untouched |
| no non-v1 generator logic copied in | ✓ | grep for `_v2`, `_v3`, `_v3_1`, `depth_probe_v2`, `depth_probe_v3` against the patched files returns empty |

`grep` verification :

```bash
$ grep -nE '_v2|_v3|_v3_1|MORE_RELIABLE_THAN_FORWARD|MORE_RELIABLE_THAN_BACKWARD' \
       mbs/graph.py mbs/datasets.py
(empty)
$ grep -nE 'tokenizer\.|benchmark\.' mbs/datasets.py
(empty — no tokenizer/benchmark CALLS in datasets.py)
$ grep -nE 'tokenizer\.|benchmark\.' mbs/graph.py
mbs/graph.py:75:    max_text = max(min(max_text_len, len(tokenizer.encode(...))) ...)
mbs/graph.py:108:            cell_token_ids[batch_idx, node_idx] = tokenizer.encode_cell(cell["text"])
mbs/graph.py:127:        encoded = tokenizer.encode(sample["text"], max_text_len)
# All three are pre-existing calls in collate_graph_samples — NOT
# in the diff hunks. Verified by viewing each line against the
# committed copy.
```

## 4. Unrelated hunks detected ?

**None.** Every line in the diff is justified by §1–§2 above.

→ No removal needed. Proceeding to Step 2 (smoke tests).

## 5. Diff stat

```
mbs/datasets.py +345 / -0   (2 hunks, both v1-only)
mbs/graph.py    +26  / -1   (7 hunks, all v1 minimal)
              -----------
total           +371 / -1
```

The `-1` in graph.py is one `}` line whose immediate predecessor
got a new `,` — purely punctuation, not a behaviour change.

## 6. Decision

→ Diff is **scoped correctly**. **Proceed with Step 2 (smoke
re-run) and then commits Group 0 and Group 1.**
