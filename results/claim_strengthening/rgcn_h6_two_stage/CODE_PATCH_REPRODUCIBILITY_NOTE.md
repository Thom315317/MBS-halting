# Code patch reproducibility note — RGCN+H6 two-stage protocol

- date: 2026-05-13
- target commit : "Add adaptive halting protocol stack + RGCN H6 two-stage support"
- purpose : document **which** files / symbols are required to run the
  RGCN+H6 campaign (configs, audit script, launcher), so that a future
  reviewer reading the repo can locate the code support behind the
  RGCN+H6 artefacts.

## 1. Why these patches are necessary

The v0.3 public release (`7eabe38`) shipped the MBS substrate, the v1
`depth_controlled_latent_halting_probe` task, and the legacy linear
`AdaptiveHaltingController`. Running the `rgcn_h6_two_stage` campaign
(the artefacts in `rgcn_h6_two_stage/*.{md,json,csv}` and the configs
in `configs/rgcn_h6_stage{1,2}_seed*.yaml`) needs the following
additions on top of v0.3 :

1. **The enriched halting controller class** (`EnrichedAdaptiveHaltingController`)
   and its auxiliary feature dimension constant (`ENRICHED_HALT_AUX_DIM = 5`)
   — required so that any model (MBS or RGCN) can instantiate the
   MLP halting controller used by H4 / H6 / H6_detached_aux.

2. **The step-aware aux-feature helper** (`compute_halt_aux_features`)
   and the module-local value-logit aggregator (`_aggregate_value_logits`)
   — required so that both `MBSModel` and
   `RelationalGCNHaltingClassifier` produce the 5 anytime features
   (`normalized_step`, `selector_entropy_t`, `selector_max_prob_t`,
   `value_margin_t`, `Δvalue_margin_t`) at every step.

3. **The step-aware latent loss path in `compute_loss`** (in
   `mbs/train.py`) — required so that `Σ_t halt_w_t · CE_t` is the
   loss the controller sees. Without this, halting gradient is zero
   and the controller never trains.

4. **The `claim_selector_head` + `claim_scores_per_step` outputs** on
   both `MBSModel` and `RelationalGCNHaltingClassifier` — required by
   the step-aware loss path.

5. **The `final_h` audit hook** on both models — required by the
   `audit_rgcn_h6_two_stage_controller_vs_required_hops.py` script
   to inspect the final substrate state.

6. **The new training-pipeline variant `rgcn_h6_two_stage`** — required
   so that `python -m mbs.train --variant rgcn_h6_two_stage --config
   configs/rgcn_h6_stage{1,2}_seed{N}.yaml ...` resolves to
   `RelationalGCNHaltingClassifier(halting_config={"enriched": True, ...})`.

## 2. Files required (and what each one carries)

### `mbs/halting.py`

| addition | provenance | needed by |
|---|---|---|
| `_aggregate_value_logits` (module-level helper) | RGCN+H6 P1 refactor | both MBSModel and RGCN forward |
| `compute_halt_aux_features` (module-level helper) | RGCN+H6 P1 refactor | both MBSModel and RGCN forward |
| `ENRICHED_HALT_AUX_DIM = 5` | H4 enriched controller campaign | both MBSModel and RGCN __init__ |
| `EnrichedAdaptiveHaltingController` class | H4 enriched controller campaign | both MBSModel and RGCN __init__ |

### `mbs/model.py`

| addition | provenance | needed by |
|---|---|---|
| import of `compute_halt_aux_features` + `_aggregate_value_logits` from `mbs.halting` | RGCN+H6 P1 refactor | the `_forward_adaptive_halting` path |
| `enriched_halting` + `detach_aux_features_from_selector` `__init__` branches | H4 / H6_detached_aux campaign | enriched MBS forward |
| `claim_selector_head` linear head | CODE_AUDIT Task F (step-aware loss) | step-aware loss + RGCN/MBS unified path |
| step-aware enriched halting path in `_forward_adaptive_halting` | H4 / H6_detached_aux campaign | enriched MBS forward (returns `halt_probs`, `halt_weights`, `expected_steps`, `claim_scores_per_step`, `final_h`) |
| `final_h` audit hook in outputs dict | RGCN+H6 P4 (parity with RGCN) | post-hoc audits |

### `mbs/baselines.py`

| addition | provenance | needed by |
|---|---|---|
| `claim_selector_head` linear head | CODE_AUDIT Task F | step-aware loss + RGCN/MBS unified path |
| `claim_scores_per_step` returned in outputs | CODE_AUDIT Task F | step-aware loss latent path |
| `force_terminal_step` / `warmup_terminal_step` / `set_warmup_active` machinery | RGCN ACT post-patch campaign | the `rgcn_repair_stability_act_*` variants |
| `enriched_halting` + `detach_aux_features_from_selector` `__init__` branch (P2) | RGCN+H6 P2 | the `rgcn_h6_two_stage` variant |
| enriched halting forward path (P3) | RGCN+H6 P3 | the `rgcn_h6_two_stage` variant — computes the 5 aux features per step and feeds them to the enriched controller |
| `final_h` audit hook in outputs dict (P4) | RGCN+H6 P4 | the audit script |

### `mbs/train.py`

| addition | provenance | needed by |
|---|---|---|
| `RGCN_VARIANTS` set extended with `rgcn_repair_stability_act_forced_t8`, `rgcn_repair_stability_act_warmup_t8`, and `rgcn_h6_two_stage` | RGCN ACT post-patch + RGCN+H6 | variant routing |
| `build_model` enriched RGCN branch routing | RGCN ACT + RGCN+H6 P2 | builds the right backbone per variant |
| step-aware latent loss in `compute_loss` | CODE_AUDIT Task F | the step-aware Σ_t halt_w_t · CE_t loss |
| ponder-loss variant set extended with `rgcn_h6_two_stage` | RGCN+H6 | ponder-loss applied to the new variant |
| composite val-only checkpoint selection | H6_detached_aux / H6/H7/H8 campaigns | reproducible selection |
| policy distillation teacher path | H7 negative ablation | optional teacher feature |
| RGCN ACT warmup activation per epoch | RGCN ACT post-patch | warmup behaviour |
| `print_epoch_block` extension for warmup labels | RGCN ACT post-patch | logging |
| `train_one` extensions for `init_from_checkpoint`, partial-freeze (`controller_only` / `trainable_modules`), per-module learning rates | H4 / H5b / H6_detached_aux campaigns | the two-stage protocol |

## 3. Variant added

`rgcn_h6_two_stage` (registered in `RGCN_VARIANTS`) — routes to
`RelationalGCNHaltingClassifier(halting_config=cfg["halting"])` with
`enriched=True`. No `force_terminal_step` / `warmup_terminal_step` —
the protocol's stage control is done via the YAML-level
`controller_only=true` (Stage 1) or `trainable_modules:
[halting_controller, claim_selector_head]` (Stage 2) fields, both
honored by `train_one` line 709-721.

## 4. Artefacts that depend on these patches

Listed by required code symbol :

- `configs/rgcn_h6_stage{1,2}_seed{1..5}.yaml` →
  require `RGCN_VARIANTS` to contain `rgcn_h6_two_stage`, require
  `enriched_halting` branch in `RelationalGCNHaltingClassifier`,
  require `init_from_checkpoint` + `controller_only` /
  `trainable_modules` honored by `train_one`.
- `scripts/audit_rgcn_h6_two_stage_controller_vs_required_hops.py` →
  imports `build_model` (must route `rgcn_h6_two_stage`), reads
  `claim_scores_per_step`, `halt_weights`, `expected_steps` from
  model outputs, optionally `final_h`.
- `scripts/_run_rgcn_h6_phase2.sh` → calls `python -m mbs.train
  --variant rgcn_h6_two_stage`.
- `results/claim_strengthening/rgcn_h6_two_stage/rgcn_h6_two_stage_*.{json,csv}`
  → produced by the audit script ; no direct code dependency.
- `results/claim_strengthening/paper_update/fig_*.{png,pdf}` and
  `table_main_3way.md` — produced by `scripts/make_rgcn_h6_3way_figures.py`
  which reads the audit JSON.

## 5. Reproduction command (after this commit)

```
cd $REPO

# Smoke (instantiate enriched RGCN + tiny forward) :
PYTHONPATH=. python scripts/_smoke_rgcn_h6_p1234.py    # optional, not committed

# Stage 1 (controller-only warmup) :
python -m mbs.train \
  --config configs/rgcn_h6_stage1_seed1.yaml \
  --variant rgcn_h6_two_stage \
  --output-dir results/claim_strengthening/rgcn_h6_two_stage/seed1/stage1 \
  --checkpoint-dir results/claim_strengthening/rgcn_h6_two_stage/seed1/stage1/checkpoints

# Stage 2 (co-train controller + selector) :
python -m mbs.train \
  --config configs/rgcn_h6_stage2_seed1.yaml \
  --variant rgcn_h6_two_stage \
  --output-dir results/claim_strengthening/rgcn_h6_two_stage/seed1/stage2 \
  --checkpoint-dir results/claim_strengthening/rgcn_h6_two_stage/seed1/stage2/checkpoints

# Audit (per seed) :
python scripts/audit_rgcn_h6_two_stage_controller_vs_required_hops.py \
  --output-dir results/claim_strengthening/rgcn_h6_two_stage --seeds 1

# Or full 5-seed campaign :
bash scripts/_run_rgcn_h6_phase2.sh   # seeds 2..5 (seed 1 already done)
```

The training pipeline requires GPU. The audit script can run on CPU
but is faster on GPU. Per-seed wall-clock on RTX 3060-class GPU is
~28 min (Stage 1 14m + Stage 2 14m).

## 6. Scope statement

This commit ships the **full enriched-halting infrastructure** that
sits between the v0.3 public release and the `rgcn_h6_two_stage`
campaign. The infrastructure was accumulated across several
intermediate campaigns (H1d/H1a → H2 → H4 → H5b → H6_detached_aux →
CODE_AUDIT → RGCN_ACT post-patch → RGCN+H6) without intermediate
commits. Splitting it into per-campaign commits would create
non-buildable intermediate states (e.g. an `enriched_halting`
flag with no `EnrichedAdaptiveHaltingController` class to back it).
The single bundled commit preserves repository buildability at every
SHA and matches the actual evolution of the working tree.

The `mbs/datasets.py`, `mbs/graph.py`, `mbs/benchmark.py`,
`mbs/tokenizer.py` modifications (also in the working tree) are NOT
included in this commit because they belong to earlier campaigns
(v2 / v3 / v3.1 stress tests, structural targets, etc.) and are not
required by the RGCN+H6 artefacts. They are listed under
`UNTRACKED_OUT_OF_SCOPE.md` for future cleanup.
