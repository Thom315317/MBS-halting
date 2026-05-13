# H7_IMPLEMENTATION_PLAN

- date: 2026-05-13
- worktree : `/home/thom315/MBS-halting-h7`, branch `h7-ordinal-halting`,
  starting from `dfb99b0`.
- scope of this plan : minimum code needed to run **V2 (seed 3,
  ordinal_loss_weight = 0.005)**. V3 / V4 / V5 reuse the same code,
  toggled by config only.

## 1. Files modified / created

### 1.a New files

| path | role |
|---|---|
| `mbs/ordinal_halting.py` | new helper module. Contains the ordinal pairwise ranking loss, the validation-only metric computation, and the checkpoint-gate eligibility check. **No import from this module is added to `mbs/__init__.py`** ; only `mbs/train.py` imports it, and only inside the code path gated by `halting_ordinal.enabled` / `checkpoint_gate.enabled`. |
| `configs/h7_ordinal_halting/rgcn_h7_seed3_w0005.yaml` | V2 config |
| `configs/h7_ordinal_halting/rgcn_h7_seed3_w001.yaml` | V3 config (this turn, written but not run) |
| `configs/h7_ordinal_halting/rgcn_h7_seed3_w005.yaml` | V4 config (this turn, written but not run) |
| `configs/h7_ordinal_halting/rgcn_h7_seed3_w010.yaml` | V5 config (this turn, written but not run) |

### 1.b Modified files

| path | minimal patches |
|---|---|
| `mbs/train.py` | (A) wrap the `collate_fn` in `make_loaders` to add `required_hops` to the batch dict iff the config requests it (config-gated, no behaviour change on H6). (B) in `compute_loss`, add an ordinal-loss block that triggers only when `halting_ordinal.enabled` is True. (C) in `train_one`, after each epoch's validation, compute the gate-metrics on a per-sample pass and apply the gate to checkpoint selection. New behaviour gated by `checkpoint_gate.enabled`. (D) record per-epoch gate-eligibility reasons. |
| `mbs/halting.py` | **not modified.** The existing `EnrichedAdaptiveHaltingController.forward(query_state, aux_features) → halt_prob` is sufficient. No model-side change. |
| `mbs/model.py` | **not modified.** `_forward_adaptive_halting` already returns `expected_steps`, `halt_weights`, `claim_scores_per_step` ; that's all the ordinal loss needs. |
| `mbs/baselines.py` | **not modified.** Same reasoning ; `RelationalGCNHaltingClassifier.forward` already returns `expected_steps`. |

→ **Only one code file is patched** : `mbs/train.py`. Everything else
is a new file. This preserves the H6 contract intact.

## 2. New config keys

```yaml
halting_ordinal:
  enabled: false                       # default false ; H6 configs unchanged
  use_required_hops: true              # always true when enabled ; placeholder for future variants
  loss_weight: 0.01
  pair_sampling: adjacent_balanced     # the only supported value at V2
  margin: 0.15
  max_pairs_per_batch: 512
  stop_gradient_expected_step: false   # passthrough flag, default off

checkpoint_gate:
  enabled: false                       # default false ; H6 configs unchanged
  val_only: true                       # binding constraint, never overridden
  min_acc_within_best: 0.02
  reject_hard_collapse: true
  reject_soft_middle_step: true
  min_s_easy: 0.15
  min_macro_auc: 0.70
  min_adjacent_margin_mean: 0.00
  min_adjacent_margin_min: -0.10
```

Anywhere `config.get(...)` reads these keys, the default value must be
the legacy behaviour (no ordinal loss, no gate). H6 configs that do
not declare `halting_ordinal:` and `checkpoint_gate:` continue to use
the exact same code path as before.

## 3. Backward-compatibility guarantee for existing H6 configs

| invariant | how it is preserved |
|---|---|
| H6 configs load without modification | `config.get("halting_ordinal", {}).get("enabled", False)` is `False` for any H6 config → ordinal block is skipped. Same for `checkpoint_gate`. |
| H6 outputs file structure unchanged | The H7 block writes to `gate_eligibility.json` (new file) and does NOT touch `_train_results.json` / `_eval_results.json` / `_epoch_metrics.csv` schema except for optional additional columns (`val_s_easy`, `val_macro_auc`, `val_collapse_flags`) that are only populated when `checkpoint_gate.enabled` is True. |
| H6 numeric outputs unchanged | The collate-wrapping is also gated : if `halting_ordinal.enabled` is False, `required_hops` is **not** added to the batch dict ; the collate output is byte-identical to before. |
| The H7 ordinal loss does not touch parameters when the loss weight is 0 or the flag is off | The ordinal-loss code path is wrapped in `if cfg.get("halting_ordinal", {}).get("enabled", False) and weight > 0: ...`. |
| Existing tests (`pytest tests/ -q` → 14 passed) continue to pass | The patches are additive ; no existing public function changes signature. |

A backward-compatibility smoke test is part of §7 below.

## 4. How `required_hops` enters the ordinal loss

### 4.a Source

Each v1 sample carries `metadata["candidate_ranks_used"]` and
`metadata["winner_rank"]`. The formula (used by the existing audit
script `audit_rgcn_h6_two_stage_controller_vs_required_hops.py`,
line 60–67) is :

```python
required_hops = (max(metadata["candidate_ranks_used"])
                 - int(metadata["winner_rank"]) + 2)
```

`metadata` is part of each sample dict that the dataset yields. It
is NOT propagated through `collate_graph_samples(...)` by default
(verified : `mbs/graph.py:collate_graph_samples` ignores `metadata`).

### 4.b Pipeline

A new wrapper `collate_with_required_hops(samples, tokenizer)` is
defined inside `mbs/train.py` (not `mbs/graph.py`, since we are not
allowed to modify graph.py).

```python
def collate_with_required_hops(samples, tokenizer):
    batch = collate_graph_samples(samples, tokenizer)
    hops = []
    for s in samples:
        meta = s.get("metadata") or {}
        crs = meta.get("candidate_ranks_used")
        wr = meta.get("winner_rank")
        if crs is None or wr is None:
            hops.append(-1)  # sentinel for "unknown"
        else:
            hops.append((max(crs) - int(wr)) + 2)
    batch["required_hops"] = torch.tensor(hops, dtype=torch.long)
    return batch
```

`make_loaders` uses this wrapper iff
`halting_ordinal.enabled` OR `checkpoint_gate.enabled` is True.
Otherwise it uses the existing collate, byte-identical to before.

### 4.c Ordinal loss

In `mbs/ordinal_halting.py` :

```python
def ordinal_pairwise_loss(expected_steps, required_hops, *,
                          margin=0.15,
                          pair_sampling="adjacent_balanced",
                          max_pairs_per_batch=512,
                          stop_gradient_expected_step=False):
    """For pairs (i, j) with required_hops_i < required_hops_j on
    *adjacent* boundaries (h → h+1), penalty = max(0, margin + E_i - E_j).

    Sampling : balanced across the 4 adjacent boundaries (5→6, 6→7,
    7→8, 8→9), up to max_pairs_per_batch total. If a bucket is
    missing in a batch, skip that boundary cleanly.

    Returns (loss_scalar, per_boundary_pair_counts, per_boundary_loss).
    """
```

The loss is added to `compute_loss` (in `mbs/train.py`) :

```python
ord_cfg = (config.get("halting_ordinal") or {})
if ord_cfg.get("enabled", False) and "expected_steps" in outputs:
    if "required_hops" not in batch:
        raise RuntimeError(
            "halting_ordinal.enabled=True but batch has no 'required_hops'. "
            "Either disable halting_ordinal or use the ordinal-aware collate "
            "(automatic when halting_ordinal.enabled=True via make_loaders)."
        )
    loss_o, per_b_pairs, per_b_loss = ordinal_pairwise_loss(
        outputs["expected_steps"], batch["required_hops"],
        margin=float(ord_cfg.get("margin", 0.15)),
        pair_sampling=str(ord_cfg.get("pair_sampling", "adjacent_balanced")),
        max_pairs_per_batch=int(ord_cfg.get("max_pairs_per_batch", 512)),
        stop_gradient_expected_step=bool(ord_cfg.get("stop_gradient_expected_step", False)),
    )
    w = float(ord_cfg.get("loss_weight", 0.0))
    if w > 0:
        loss = loss + w * loss_o
        parts["ordinal_loss"] = loss_o.detach()
        parts["ordinal_loss_weighted"] = (w * loss_o).detach()
        parts["ordinal_pairs_total"] = sum(per_b_pairs.values())
        for k, v in per_b_pairs.items():
            parts[f"ordinal_pairs_{k}"] = v
```

The error in §4.c above is explicit per design constraint §3 of the
H7 prompt : *"If `required_hops` is missing from a batch and ordinal
loss is enabled, fail clearly during training with an explicit error
message."*

The training-time loss is computed on the **training split's**
`required_hops`. This is supervised calibration using generator
metadata. The pre-registration §3 already records the framing :
**Regime B — Ordinal-calibrated. Do NOT claim it is emergent.**

## 5. How OOD is excluded from checkpoint selection

| layer | how OOD is excluded |
|---|---|
| Ordinal loss | only sees the training-split batch (the data loader is `loaders["train"]`). OOD batches never enter `compute_loss` for training. |
| Validation gate metrics | the gate metrics are computed by a **dedicated** per-sample pass over `loaders["val"]` only, inside `train_one`. The function `compute_gate_metrics_on_val(model, loaders["val"], device, config, variant)` reads only `loaders["val"]`. |
| Checkpoint eligibility check | reads only the val-side metrics in `gate_eligibility.json[epoch]`. Never reads any `ood_*_acc` or `ood_*_*` field. |
| Final OOD evaluation | done after training, on the selected checkpoint, exactly as in H6 — no change. |

The audit constant `val_only: true` in the `checkpoint_gate:` block
of every H7 config is a **declarative** affirmation of the above. The
code reads this key but the OOD-exclusion is a hard invariant : OOD
metrics are computed but never feed into the gate.

## 6. Smoke tests (must pass before any V2 training launch)

Implemented as a new script `scripts/_smoke_h7_compat.py`. Five checks :

1. **H6 config still loads** : load
   `configs/rgcn_h6_stage1_seed1.yaml` and assert
   `config.get("halting_ordinal", {}).get("enabled", False)` is
   `False` AND
   `config.get("checkpoint_gate", {}).get("enabled", False)` is
   `False`. Build the model. Build the loaders. Run **one** training
   batch through `compute_loss` ; assert finite loss. Compare the
   loss scalar against a deterministic baseline (this test passes
   iff the H6 code path is byte-identical to before this patch — we
   verify via `set_seed(seed=1)` + a single forward).
2. **H7 config loads** : load
   `configs/h7_ordinal_halting/rgcn_h7_seed3_w0005.yaml` ; build
   model, build loaders, assert `halting_ordinal.enabled` is True.
3. **Ordinal-aware collate works** : pull one training batch ;
   assert `"required_hops"` is in the batch ; assert the per-sample
   values are in [5, 9] for v1.
4. **Ordinal loss returns a finite scalar** : run
   `ordinal_pairwise_loss(outputs["expected_steps"],
   batch["required_hops"], margin=0.15, ...)` on one batch ; assert
   loss is a finite tensor with `requires_grad=True` and that
   `loss.backward()` runs without error.
5. **Audit script reproduction** : re-run
   `audit_halting_ordinal_metrics.py` on the existing H6 RGCN+H6
   per-seed CSV ; assert seed 3 is `soft_middle_step` and seeds
   1/2/4/5 are `binary_h9_shortcut` (the H6 re-audit established
   this baseline ; this is a regression test).

The 5 smoke tests must all pass before V2 is launched.

## 7. Sequence of execution (this turn)

1. **Plan** (this file). ✓
2. Write `mbs/ordinal_halting.py`.
3. Patch `mbs/train.py` minimally (collate wrap, compute_loss block,
   gate hook in train_one).
4. Write the 4 configs.
5. Write the smoke script `scripts/_smoke_h7_compat.py`.
6. Run smoke tests.
7. Launch V2 in background (~28 min wall clock).
8. While V2 runs, write the V2 launcher / audit scaffolding.
9. When V2 finishes, audit V2 → `audits/rgcn_h7_seed3_w0005_*` files.
10. Write `V2_SEED3_W0005_REPORT.md`.
11. **STOP**. Do not launch V3/V4. Report back to user.

## 8. Compute budget (this turn)

| step | wall clock |
|---|---:|
| code + configs + smoke | ~10 min (engineering) |
| smoke tests | ~30 s |
| V2 Stage 1 (5 epochs) | ~14 min |
| V2 Stage 2 (5 epochs) | ~14 min |
| V2 audit (CPU) | < 1 min |
| V2 report | ~5 min |
| **total** | **~45–50 min wall clock** |

If V2 fails the success criteria, the **immediate next step**
recommended in the V2 report is V3 (weight 0.01). Decision is
deferred to the user.

## 9. What this plan does NOT cover

- V3 / V4 / V5 launches. Configs are written ; runs are not started.
- V6 (RGCN+H7 from scratch). Not in scope for this turn.
- The 5-seed rerun. Gated on seed-3 success per pre-registration §9.
- Any modification of `mbs/datasets.py`, `mbs/graph.py`,
  `mbs/tokenizer.py`, `mbs/benchmark.py`, `mbs/halting.py`,
  `mbs/model.py`, `mbs/baselines.py`.
- Any commit. The user explicitly instructed `do not commit
  automatically`.
