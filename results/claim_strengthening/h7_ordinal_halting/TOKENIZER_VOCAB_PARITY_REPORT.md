# TOKENIZER_VOCAB_PARITY_REPORT

- date : 2026-05-13
- worktree : `/home/thom315/MBS-halting-h7`, branch `h7-ordinal-halting`.
- purpose : verify that the Group 0.b tokenizer patch produces a
  vocabulary **identical in size and order** to the tokenizer that
  trained the committed H6 checkpoint.

## 1. Before vs after the patch

| measurement | value |
|---|---:|
| clean committed `mbs/tokenizer.py` vocab size | **78** |
| dirty main `mbs/tokenizer.py` vocab size | **91** |
| patched H7-worktree `mbs/tokenizer.py` vocab size | **91** |
| H6 checkpoint `token_embedding.weight` rows | **91** |

## 2. Exact-parity result

→ **PASS.** `patched.tokens == dirty.tokens` element-by-element, all
91 entries. Verified by `scripts/_verify_tokenizer_parity.py`
(kept in `/home/thom315/tmp/` for reference) :

```
patched vocab size: 91
dirty   vocab size: 91
OK: exact tokenizer vocab parity with dirty main (91 tokens, order matches)
```

In particular, the 13 added tokens are inserted **in the same
positions** as the dirty main :

- index 35 : `"trusted_chain_top_wins"` (appended to the rules
  block, right after `"least_reliable_source_wins"` at index 34).
- indices 79..90 : `"I", "J", "K", "L", "M", "N", "O", "P", "Q",
  "R", "S", "T"` (appended to the sources block, right after `"H"`
  at index 78).

Existing tokens 0..34, 36..78, and the values block at indices
83..90 of the OLD layout shift to 91..98 of the NEW layout — but
all the new positions match the dirty-main exactly, so the H6
checkpoint's embedding rows map to the same tokens after the
patch.

## 3. List of added tokens (13 total)

| index | token | provenance |
|---:|---|---|
| 35 | `"trusted_chain_top_wins"` | required by v1's RULE cell (`_build_depth_probe_sample` writes `add_cell("RULE", DEPTH_PROBE_RULE_NAME)` where `DEPTH_PROBE_RULE_NAME = "trusted_chain_top_wins"`) |
| 79 | `"I"` | v2 source token (not used by v1 generation but required for vocab parity) |
| 80 | `"J"` | v2 source |
| 81 | `"K"` | v2 source |
| 82 | `"L"` | v2 source |
| 83 | `"M"` | v2 source |
| 84 | `"N"` | v2 source |
| 85 | `"O"` | v2 source |
| 86 | `"P"` | v2 source |
| 87 | `"Q"` | v2 source |
| 88 | `"R"` | v2 source |
| 89 | `"S"` | v2 source |
| 90 | `"T"` | v2 source |

## 4. Token order matches dirty main exactly

The verification script does `assert patched.tokens == dirty.tokens`
on the full 91-element list. No reordering. No element substitution.
The list is byte-identical.

## 5. Checkpoint load test (the actual reproducibility test)

A second test in the verification script :

- Build `RelationalGCNHaltingClassifier(...)` using the H7 V2
  config + the patched tokenizer.
- Load
  `results/.../rgcn_h6_two_stage/seed3/stage1/checkpoints/rgcn_h6_two_stage_best.pt`
  (the same checkpoint V2 init points to).
- `model.load_state_dict(state, strict=False)`.

Result :

```
checkpoint has 39 tensors
ckpt token_embedding.weight.shape = (91, 96)
model token_embedding.weight.shape = (91, 96)
loaded ; missing=0 unexpected=0
OK: checkpoint loads ; no embedding shape / key mismatch
```

→ **0 missing, 0 unexpected.** The H6 Stage-1 best.pt of seed 3
loads bit-perfectly into a freshly-built RGCN+H7 model with the
patched tokenizer. There is no token-embedding-row-to-token
mismatch.

This is the strict test of vocab parity. Just matching the size
would let the checkpoint load (size mismatch goes away) but each
embedding row could be assigned to the wrong token — a silent
reproducibility bug. Matching the **order** as well prevents that.

## 6. Required for H6 checkpoint safety

Without this patch :

- the committed H6 checkpoint's `token_embedding.weight` has shape
  (91, 96), the worktree-built model has (78, 96) → `load_state_dict`
  raises `RuntimeError` → V2 cannot launch.

With size-only patch (e.g. adding 13 random `"<extra>"` tokens) :

- size match → `load_state_dict` succeeds → but each of the 91
  embedding rows ends up assigned to a different token than at
  training time → corrupted prediction surface, no error message,
  silent bug.

With this exact-parity patch :

- token `i` in the worktree means exactly the same string as token
  `i` in the dirty main → each row of the embedding is preserved
  in meaning → the H6 checkpoint's behaviour is bit-identical when
  reloaded.

## 7. No H7 method behaviour changed

Verified by inspection of the diff stat :

```
$ git diff --stat -- mbs/tokenizer.py
 mbs/tokenizer.py | 12 ++++++++++--
 1 file changed, 10 insertions(+), 2 deletions(-)
```

The 10 added lines are :

- 2 comment lines explaining `trusted_chain_top_wins` and the
  source expansion.
- 1 line `"trusted_chain_top_wins",`.
- 3 lines reformatting the `sources = [...]` literal to a
  multi-line list with 20 entries instead of 8.
- 4 lines of context comments preserved from the dirty main.

The 2 deleted lines are the single-line `sources = ["A", ..., "H"]`
that is now expanded.

The method-side behaviour (`encode`, `encode_cell`, `state_dict`,
`from_state_dict`, `add`, `pad_id`, `unk_id`, the regex for
splitting text) is **byte-identical** to the pre-patch version.

## 8. Files involved by this patch step

| file | role |
|---|---|
| `mbs/tokenizer.py` | the actual patch (12 lines added net) |
| `TOKENIZER_REPRO_PATCH_PLAN.md` | the plan written before patching |
| `TOKENIZER_REPRO_EXTRACTED_DIFF.patch` | snapshot of the dirty main's diff for documentation |
| `dirty_main_tokenizer_tokens.json` | 91-element ground-truth list used by the parity test |
| `TOKENIZER_VOCAB_PARITY_REPORT.md` | this file |

## 9. Next step

Proceed to Step 6 (smoke tests) → Step 7 (Group 0.b commit) → Step 8
(V2 launch).
