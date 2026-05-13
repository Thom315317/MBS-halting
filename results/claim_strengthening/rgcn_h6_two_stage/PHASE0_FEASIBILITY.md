# Phase 0 — Feasibility for porting the H6_detached_aux protocol to RGCN

- date: 2026-05-13
- output dir: `results/claim_strengthening/rgcn_h6_two_stage/`
- read-only diagnostic ; no training launched yet.

## 1. What "H6_detached_aux protocol" means concretely

From the existing H6_detached_aux runs (5 seeds on v1) and the
CODE_AUDIT report, the protocol is the following sequence :

1. **Stage 1 (H4-style warmup)** — `controller_only=true` :
   - backbone, embeddings, answer_head, claim_selector_head all
     frozen ;
   - only `halting_controller` receives gradient ;
   - same step-aware latent loss ; λ_ponder = 0.01.
2. **Stage 2 (H5b-style co-training)** —
   `trainable_modules=[halting_controller, claim_selector_head]` :
   - backbone + embeddings still frozen ;
   - the two halting/selector modules are jointly trained ;
   - aux features detached (`detach_aux_features_from_selector=true`) ;
   - same step-aware latent loss ; λ_ponder = 0.01.
3. **Controller architecture**: `EnrichedAdaptiveHaltingController`
   (MLP d_state+5 → 128 → 64 → 1) with 5 anytime aux features:
   - normalized_step (t / max_steps),
   - selector_entropy_t (softmax entropy over query CLAIMs),
   - selector_max_prob_t,
   - value_margin_t (top1 − top2 of value_logits),
   - delta_value_margin_t.
4. **Selection**: official val_acc + tie-break val_loss + earliest
   epoch (no OOD selection).

## 2. What RGCN ACT post-patch has today

From `mbs/baselines.py:RelationalGCNHaltingClassifier` :

- ✓ `claim_selector_head` (added by CODE_AUDIT Task F)
- ✓ `claim_scores_per_step` returned in outputs (CODE_AUDIT)
- ✓ step-aware latent loss is compatible (the patch makes
  `compute_loss` step-aware path consume RGCN ACT outputs uniformly
  with MBS)
- ✗ `halting_controller = AdaptiveHaltingController` (linear,
  hardcoded) — no `EnrichedAdaptiveHaltingController` branch
- ✗ no aux features computation per step
- ✗ no `enriched_halting` config flag honored by the RGCN model
- ✗ no `final_h` audit hook (already added on MBSModel ; missing on
  RGCN)

## 3. What the partial-freeze logic needs from RGCN

The 2-stage protocol uses `train.py:train_one`'s `trainable_modules`
flag, which freezes every parameter whose name does not start with
one of the listed prefixes. This works on **any** torch module —
no RGCN-specific change required. Confirmed by reading
`train.py:669-686`.

For the same reason, `controller_only=true` works on RGCN too — it
falls back to `trainable_modules=["halting_controller"]`.

## 4. What needs a code patch

To run "RGCN + H6_detached_aux protocol" with no training-pipeline
hack, the minimal patches are :

| patch | scope | risk |
|---|---|---|
| **P1** — extract `_compute_halt_aux_features` from `MBSModel` to a top-level helper in `mbs/halting.py` so it can be imported by both `MBSModel` and `RelationalGCNHaltingClassifier` | ~30 lines : new function, MBS method becomes a one-line wrapper | very low — pure refactor, equivalent semantics |
| **P2** — `RelationalGCNHaltingClassifier.__init__` reads `halting_config["enriched"]` and instantiates either `EnrichedAdaptiveHaltingController` or `AdaptiveHaltingController` | ~10 lines | very low |
| **P3** — `RelationalGCNHaltingClassifier.forward` adds the enriched halting path (compute aux features per step, optionally detach, pass to controller) | ~30 lines : a copy of the MBS enriched-halting structure adapted to the RGCN forward | low — the structure is identical to MBSModel ; well-typed |
| **P4** — add `"final_h": h` to RGCN outputs (instrumentation hook for diagnostic audits) | 1 line | trivial |

All four patches are **read-only on the training pipeline** (no
changes to `compute_loss`, `train_one`, or the freeze logic).
RGCN_ACT post-patch existing runs and configs remain identical.

## 5. Configs and protocols

Two new config files needed :

- `configs/rgcn_h6_stage1_seed{N}.yaml` (controller-only warmup with
  RGCN backbone, init from H1b-equivalent OR randomly init)
- `configs/rgcn_h6_stage2_seed{N}.yaml` (co-train ctrl + selector,
  init from the stage 1 best ckpt)

**Important difference vs MBS H6_detached_aux**: the MBS runs
init Stage 1 from the **H1b** MBS checkpoint, an MBS-specific
artifact. For RGCN we don't have an "H1b RGCN" pretrained
checkpoint. Options :

(a) **No init**: train RGCN backbone end-to-end at Stage 1 with
    `controller_only=false`, then freeze for Stage 2.
(b) **Use the existing RGCN_ACT post-patch best.pt** as the "frozen
    backbone" : the model already learned the task at acc ≈ 0.87.
    Take its weights, freeze everything except halting_controller,
    swap in the EnrichedAdaptiveHaltingController, run a Stage 1
    warmup, then Stage 2 co-train.

→ **(b) is the cleanest analog of H6_detached_aux** because it
mirrors "use the trained backbone's representations and add an
enriched halting policy on top". It avoids a 3rd training stage and
is comparable to the H6 pipeline that builds on an already-trained
backbone (H1b).

## 6. Verdict

→ **Feasible with a small code patch (P1 + P2 + P3 + P4).**

Estimated patch size : ~70 lines total, none touching the training
loop, loss, or freeze logic. Existing tests / artefacts unaffected.

Plan :
1. Apply P1..P4.
2. Smoke the new config on seed 1 (Stage 1 + Stage 2 + audit).
3. Gate seed 1 against the criteria from the brief :
   - no floor/final collapse (< 0.5 each) ;
   - Spearman(expected_step, required_hops) > 0.30 on val or ood ;
   - chosen_step distinct > 1 ;
   - OOD acc > 0.75.
4. If GO, run seeds 2..5.
5. Otherwise, document as negative transfer and stop.

## 7. Constraints respected by this Phase 0

- Read-only inspection ; no code modified yet.
- No H6_detached_aux file modified.
- No RGCN ACT post-patch artefact modified.
- All new artefacts will go in `results/claim_strengthening/rgcn_h6_two_stage/`.
- No OOD selection planned.
