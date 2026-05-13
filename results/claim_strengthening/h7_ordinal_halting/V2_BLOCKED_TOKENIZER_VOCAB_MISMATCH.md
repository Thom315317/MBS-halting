# V2_BLOCKED_TOKENIZER_VOCAB_MISMATCH

- date : 2026-05-13
- worktree : `/home/thom315/MBS-halting-h7`, branch `h7-ordinal-halting`,
  HEAD = `0d081b7 Add H7 ordinal-halting audit and seed-3 configs`.
- trigger : V2 training launch fails at `init_from_checkpoint` step.

## 1. Exact failure

V2 launcher : `scripts/_run_v2_seed3_w0005.sh` invokes
`python -m mbs.train --config configs/h7_ordinal_halting/rgcn_h7_seed3_w0005.yaml
--variant rgcn_h7_two_stage ...`.

Traceback :

```
File "/home/thom315/MBS-halting-h7/mbs/train.py", line 770, in train_one
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
RuntimeError: Error(s) in loading state_dict for RelationalGCNHaltingClassifier:
    size mismatch for token_embedding.weight: copying a param with shape
    torch.Size([91, 96]) from checkpoint, the shape in current model is
    torch.Size([78, 96]).
```

- Checkpoint : `results/.../rgcn_h6_two_stage/seed3/stage1/checkpoints/rgcn_h6_two_stage_best.pt`
  (the H6 seed-3 Stage-1 best.pt, symlinked into the H7 worktree from
  the main repo).
- Checkpoint `token_embedding.weight.shape = (91, 96)` → trained
  against a tokenizer with vocab size 91.
- Worktree-built `RelationalGCNHaltingClassifier(num_cell_types=8,
  vocab_size=len(tokenizer.tokens)) → token_embedding.weight.shape =
  (78, 96)`.
- The committed `mbs/tokenizer.py` produces vocab size 78. The
  worktree state of `mbs/tokenizer.py` (untouched, since the user
  instruction explicitly forbids modifying it) is therefore also 78.

## 2. Where the 13-token gap comes from

`git -C /home/thom315/MBS-halting diff -- mbs/tokenizer.py` shows two
additions in the dirty main `mbs/tokenizer.py` (NOT in the committed
worktree) :

```diff
@@ -41,6 +41,8 @@ class SimpleTokenizer:
             "earliest_wins",
             "second_latest_wins",
             "least_reliable_source_wins",
+            # Used by the depth_controlled_latent_halting_probe task only.
+            "trusted_chain_top_wins",
         ]
@@ -69,7 +71,13 @@ class SimpleTokenizer:
             ...
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

→ 1 new rule + 12 new sources = +13 tokens, taking the vocab from 78
to 91 in the dirty main.

The committed H6 checkpoint was trained with this dirty tokenizer.
The committed worktree tokenizer cannot load it as-is.

## 3. Scope of what V1 actually NEEDS from the tokenizer

- The new rule `"trusted_chain_top_wins"` is **directly used** by
  `_build_depth_probe_sample` in `mbs/datasets.py`
  (`add_cell("RULE", DEPTH_PROBE_RULE_NAME)`). Without it tokenized,
  the v1 generator emits a token id collision or an `<unk>` for the
  RULE cell. Required for v1.
- The 12 new sources (`I` through `T`) are **only required for
  v2 / v3 / v3.1** (k_max up to 20). v1 has `k_max=8` and uses only
  `A..H`. **Not required for v1 sample generation**, but **required
  for checkpoint vocab parity** : the H6 checkpoint was trained on
  vocab size 91, so loading into a vocab-78 model fails on shape
  mismatch.

So the gap can be split :

| addition | required by v1 logic ? | required by H6 ckpt loading ? |
|---|:-:|:-:|
| `"trusted_chain_top_wins"` rule | ✓ | ✓ |
| 12 new sources `I..T` | ✗ | ✓ |

## 4. The user constraint

> Do NOT modify : `mbs/benchmark.py`, **`mbs/tokenizer.py`**, v2/v3/v3.1
> generator logic, any result files outside results/claim_strengthening/h7_ordinal_halting/

The user's instruction explicitly forbids touching `mbs/tokenizer.py`.
The constraint was anticipating possible tokenizer issues but
disallowing modifications without explicit approval. **I have not
modified mbs/tokenizer.py.**

## 5. State at the moment of this halt

- Group 0 commit (`1197bd7`) : reproducibility-dependency patch on
  graph.py + datasets.py. **Done, committed locally.**
- Group 1 commit (`0d081b7`) : H7 code + configs + audits. **Done,
  committed locally.**
- V2 training : **NOT started** (fails at init load before any
  forward / backward pass). No checkpoint produced, no run.log
  beyond the traceback, no GPU time consumed.
- No `git push` executed.
- H6 historical artefacts unchanged.

## 6. Options to unblock V2 (user decision required)

### Option A — Permit a minimal tokenizer patch (RECOMMENDED)

Lift exactly the two hunks from the dirty `mbs/tokenizer.py` :
- Add `"trusted_chain_top_wins"` to the rules list.
- Extend `sources = [..., "I", ..., "T"]`.

This is a 14-line addition (1 rule + 12 sources + 1 comment block).
After this patch :
- `len(tokenizer.tokens)` becomes 91, matching the committed H6
  checkpoint.
- V2 loads cleanly.
- `pytest tests/ -q` still passes.
- The smoke battery still passes (test 1 verifies H6 forward, which
  needs the right vocab).

Suggested commit message :
`Fix tokenizer vocab parity required by committed H6 checkpoint`

This is **another reproducibility-dependency patch**, same family as
the graph.py + datasets.py Group 0. Same justification : the
committed H6 baseline relies on a tokenizer state that was never
committed.

### Option B — Strip the token_embedding from the init load

Modify the `init_from_checkpoint` path in `mbs/train.py` to discard
`token_embedding.weight` before loading (it would be re-initialised
from the smaller vocab-78 default init).

Pros : no tokenizer change.
Cons : the model loses the trained token embedding from H6 — this is
exactly the embedding that gives meaning to the per-cell text. V2
would effectively start with a randomised text-encoder on top of a
trained graph backbone. This **invalidates the V2 init-controlled
comparison** with H6.

### Option C — Use a different init source

The H6 RGCN+H6 Stage-1 best.pt is the only available init that
matches the V2 protocol. There is no clean alternative.

### Option D — Retrain the v1 baseline from scratch with vocab 78

Train RGCN ACT post-patch + RGCN+H6 from scratch in the H7 worktree
with vocab 78. Then use those checkpoints as init for V2.

Pros : fully self-contained worktree.
Cons : **3–4 hours of GPU time** before V2 can even start ; also
voids the comparison with the previously-committed H6 numbers
(different vocab → different generalisation surface). Defeats the
"H7 builds on the committed H6 baseline" framing.

### Option E — Abort V2 in this turn

Leave the commits as-is, do not launch V2, hand off the choice
between A, B, C, D to the user.

## 7. Recommendation

**Option A**, framed identically to the graph.py + datasets.py
reproducibility patch :

> "The committed H6 checkpoint was trained against a tokenizer with
> +13 tokens that were never committed. The new rule
> `trusted_chain_top_wins` is required by v1's RULE cell ; the 12
> new sources `I..T` are not used by v1 logic but are required for
> checkpoint vocab parity. We lift exactly these 14 tokens, no v2 /
> v3 / v3.1 logic. Diff stat : +14 lines tokenizer.py."

This is the smallest patch that lets V2 load the committed H6
checkpoint.

## 8. Reviewer-proof framing

If Option A is chosen, the commit must be a **separate, third
"Group 0.b" commit** :

```
Fix tokenizer vocab parity required by committed H6 checkpoint

The committed dfb99b0 H6 checkpoint's token_embedding.weight has
shape (91, 96), trained against a tokenizer with vocab size 91
(committed mbs/tokenizer.py only produces vocab 78). This patch
adds the 13 missing tokens : 1 new rule "trusted_chain_top_wins"
(used by v1's RULE cell ; see _build_depth_probe_sample in
mbs/datasets.py) and 12 new sources "I" through "T" (used by v2
generators only ; included here for vocab parity with the H6
checkpoint).

Same family as 1197bd7 (graph.py / datasets.py reproducibility
patch). NOT an H7 method change.
```

After this commit, V2 launches cleanly. Group 0 + Group 0.b + Group 1
are the three commits on `h7-ordinal-halting`.

## 9. What I will NOT do without user approval

- I will NOT modify `mbs/tokenizer.py`.
- I will NOT launch V2.
- I will NOT `git reset` the existing commits.
- I will NOT push.
- I will NOT downgrade the smoke battery or relax the gate
  thresholds.

## 10. Current git state

```
$ git log --oneline -3
0d081b7 Add H7 ordinal-halting audit and seed-3 configs
1197bd7 Fix committed v1 graph fields required by enriched halting
dfb99b0 Update paper claims and figures after RGCN H6 transfer
```

```
$ git status --short
?? results/claim_strengthening/h7_ordinal_halting/V2_BLOCKED_TOKENIZER_VOCAB_MISMATCH.md
?? results/claim_strengthening/h7_ordinal_halting/seed3_w0005/run.log
?? results/claim_strengthening/rgcn_h6_two_stage/seed3/  # symlink, gitignored content
?? scripts/_run_v2_seed3_w0005.sh
```

The `run.log` contains the 25-line traceback above. The `seed3_w0005/`
directory is otherwise empty (no checkpoint produced).

## 11. Next step

Tell me which of A / B / C / D / E you choose. If A, I will :

1. Apply the tokenizer minimal patch (14 lines).
2. Re-run smoke (all 5 should still pass ; smoke 1 now has the
   right vocab to load).
3. Re-attempt V2 training.
4. If V2 fails on a new mismatch, write another `STILL_BLOCKED`
   report and stop.
5. If V2 succeeds, audit it and write `V2_SEED3_W0005_REPORT.md`.

Total wall-clock for path A : ~5 min (patch + smoke) + 14 min
(V2 training) + 1 min (audit) ≈ 20 min.
