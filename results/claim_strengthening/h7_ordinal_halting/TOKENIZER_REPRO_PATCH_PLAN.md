# TOKENIZER_REPRO_PATCH_PLAN

- date : 2026-05-13
- worktree : `/home/thom315/MBS-halting-h7`, branch `h7-ordinal-halting`.
- scope : Group 0.b reproducibility-dependency commit on
  `mbs/tokenizer.py`. **NOT an H7 method change.**

## 1. Exact failure (recap)

V2 launch fails at `model.load_state_dict(...)` :

```
RuntimeError: size mismatch for token_embedding.weight:
    copying a param with shape torch.Size([91, 96])
    from checkpoint, the shape in current model is torch.Size([78, 96]).
```

- committed `mbs/tokenizer.py` vocab : **78 tokens**.
- committed H6 checkpoint trained on : **91 tokens**.
- Δ = 13 tokens missing.

## 2. Why vocab **size** AND vocab **order** both matter

`token_embedding` is a `(vocab_size, d_state)` lookup. The H6
checkpoint contains row `i` for token `i` of the **training-time
tokenizer's order**. If we patch the tokenizer to vocab size 91 but
insert the 13 new tokens at the wrong indices, then :

- size mismatch goes away → `load_state_dict` succeeds.
- but each row is now silently assigned to the WRONG token.
- token "Alice" might map to the embedding originally trained for
  "Bob" ; the model would still produce a finite, plausible-looking
  loss ; but the prediction surface is corrupted in undetectable
  ways.

This is the classic **silent reproducibility bug**. The fix is
**exact token-order parity**, not just size parity.

The user's framing is explicit :
> *"It is not enough that the tokenizer vocab size becomes 91. The
> exact token order must match the dirty main tokenizer that was used
> to train the committed H6 checkpoint."*

We will therefore :

1. Build the dirty main's tokenizer (without modifying it).
2. Dump its `tokens` list to a JSON file under the H7 reports tree.
3. Patch the committed tokenizer.
4. Build the patched tokenizer.
5. Verify `patched.tokens == dirty.tokens` element-by-element.
6. Only if exact equality holds, proceed.

## 3. Exact dirty tokenizer hunks to lift

From `git -C /home/thom315/MBS-halting diff -- mbs/tokenizer.py` :

### 3.a Rule addition

```diff
@@ -41,6 +41,8 @@ class SimpleTokenizer:
             "earliest_wins",
             "second_latest_wins",
             "least_reliable_source_wins",
+            # Used by the depth_controlled_latent_halting_probe task only.
+            "trusted_chain_top_wins",
         ]
```

→ Add `"trusted_chain_top_wins"` to the rules list, **after**
`"least_reliable_source_wins"`. Position matters : the rules list
contributes a contiguous block to the final `tokens` list, and any
re-ordering would invalidate downstream IDs.

### 3.b Sources expansion

```diff
@@ -69,7 +71,13 @@ class SimpleTokenizer:
             "Quinn",
             "Ruth",
         ]
-        sources = ["A", "B", "C", "D", "E", "F", "G", "H"]
+        # SOURCES extended from 8 (A..H) to 20 (A..T) for the
+        # depth_controlled_latent_halting_probe_v2 task, which needs k_max up
+        # to 20. The first 8 IDs are unchanged so v1 / v0.4 / belief_repair_*
+        # checkpoints reload bit-identically (they only ever saw A..H).
+        sources = ["A", "B", "C", "D", "E", "F", "G", "H",
+                   "I", "J", "K", "L", "M", "N", "O", "P",
+                   "Q", "R", "S", "T"]
```

→ Extend `sources` from `A..H` to `A..T` (12 additional tokens
`I..T` in alphabetical order, in that exact position in the list).
The first 8 entries are unchanged, so existing checkpoint IDs
0..7-relative-to-this-block are preserved.

These are the **only two** changes I will make. No comment
rewording beyond the minimum that keeps the dirty-main intent
faithful.

## 4. Expected vocab size after patch

Should be **91 tokens**. Verified by :

```python
tok = SimpleTokenizer()
assert len(tok.tokens) == 91
```

If `len(tok.tokens) != 91`, abort and write
`TOKENIZER_REPRO_STILL_BLOCKED.md`.

## 5. Expected scientific status

**Reproducibility-only patch.** Same family as Group 0
(`1197bd7 Fix committed v1 graph fields required by enriched
halting`). Does not :

- change H7 method (no ordinal loss / gate / aux feature touched).
- change the model architecture (mbs/model.py, mbs/baselines.py
  untouched).
- change the training pipeline (mbs/train.py untouched).
- change the existing tokens' IDs (the patch is purely additive in
  the right slots).
- change the encode / decode / unknown-token behaviour.
- modify any other module.

If any future H7 paper claims about embeddings / vocabulary, this
patch must be cited as a Methods footnote :
> *"During clean-worktree reproduction, we found that the
> committed H6 checkpoints had been trained with a tokenizer
> vocabulary of 91 tokens, while the committed tokenizer exposed
> 78. We patched the tokenizer to restore exact vocabulary parity
> (size + order) with the training-time tokenizer. This does not
> change the H7 method ; it is required to load the committed
> checkpoint without reinitialising token embeddings."*

## 6. Risk

- **Risk** : silent embedding-to-token misalignment if order
  differs even by one position.
- **Mitigation** : Step 4 (parity verification) is binding. If
  `patched.tokens != dirty.tokens` element-by-element, abort, do
  not commit, do not train.

## 7. Smoke tests

After parity check passes :

1. `pytest tests/ -q` (must stay 14/14).
2. `scripts/_smoke_h7_compat.py` 5/5 :
   - **smoke 1 now must additionally** : load
     `init_from_checkpoint` from H7 V2 config without
     `token_embedding.weight` size mismatch. (The current smoke 1
     only does a forward, not a checkpoint load ; I will rely on
     the V2 launch itself as the load test.)

Actually — looking at the smoke 1 spec : it builds the model + a
batch + a forward pass, but **does not load the H6 checkpoint**.
The size-mismatch error only fires when init_from_checkpoint is
used. So smoke 1 was passing on the broken tokenizer because it
never triggered the load.

For this patch I will add an extra check : **manually load** the
H6 seed-3 Stage-1 best.pt into a freshly-built RGCN+H7 model and
assert no shape mismatch. This is the actual test of vocab parity.

If smoke + parity + the extra checkpoint-load test all pass, the
patch is reviewer-defensible.

## 8. Files written by this plan step

- `TOKENIZER_REPRO_PATCH_PLAN.md` (this file).
