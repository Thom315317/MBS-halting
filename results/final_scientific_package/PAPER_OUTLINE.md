# Paper outline (draft) — workshop methodological submission

**Working title :**
"Enabling Conditional Halting in Latent Belief-Graph Reasoning:
Failure Modes, Audits, and a Two-Stage Controller"

**Target venue :** workshop on graph reasoning, neural halting, or
ML methodology (e.g. NeurIPS / ICLR workshops on early exit /
adaptive computation / interpretable graph models).

---

## Abstract (draft, honest)

> Adaptive halting in graph neural networks is often presented as a
> compute-saving mechanism, but in practice it is plagued by silent
> failure modes — final-only attractors, floor collapse, gradient
> leaks, unobservable controllers, and checkpoint-selection
> mismatches — that turn the "adaptive" policy into a constant. We
> document six such failure modes encountered on a clean
> belief-graph reasoning bench (`depth_controlled_latent_halting_probe`),
> derive an audit infrastructure that exposes them, and propose a
> **two-stage partial-freeze training protocol** with an enriched
> MLP halting controller fed by five anytime aux features. On the
> clean v1 bench, our protocol achieves mean ood Spearman = 0.231
> between the controller's expected step and an offline oracle,
> across 5 seeds, with 0/5 floor/final collapse — confirmed under a
> patched code base where a subtle aux-features gradient leak is
> closed. On stress-test variants (v3, v3.1) the generator audits
> pass but our current MBS substrate (d_state=96, MLP-depth-2,
> Linear(96→1) selector) cannot learn fixed-step capacity, revealing
> an architectural rather than a methodological bottleneck. We
> position this as **a failure-modes-and-method study, not a
> benchmark-clearing result.**

---

## 1. Introduction

- Adaptive halting / ACT / PonderNet : promise of compute saving on
  variable-difficulty inputs.
- Practical reality : halting controllers often degenerate to a
  constant. Why is that so hard to spot ?
- Contributions :
  1. Six failure modes documented on a clean bench.
  2. Audit infrastructure (smoke, halted accuracy metrics, composite
     checkpoint, policy drift) that exposes each mode.
  3. Two-stage partial-freeze protocol (H6_detached_aux) that
     survives all the failure modes on the clean v1 bench.
  4. Stress tests (v3 / v3.1) revealing a substrate-capacity bottleneck
     beyond which the protocol cannot apply with the current backbone.

## 2. Problem setup

- Belief-graph reasoning : 4 candidate CLAIM cells, 1 winner, depth
  oracle = number of trust-chain hops needed.
- Latent answer mode (`latent_claim_selector`) : the answer is
  produced by aggregating a per-claim score over the value buckets,
  not by a separate decoder.
- Step-aware loss : `Σ_t halt_w_t · CE_t` where `CE_t` is computed
  at every step using the per-step claim scores.
- Goal of the halting controller : produce per-sample
  `halt_w_t` such that `Σ_t halt_w_t · t` (the expected step) is
  positively correlated with the offline oracle step.

## 3. Belief-graph task (v1)

- Generator (`mbs.datasets._build_depth_probe_sample`) :
  - 4 candidates strictly internal or at endpoints (v1)
  - 4 distinct values
  - source IDs permuted per sample
  - trust chain k_max=8, depth_buckets [2,4,6,8]
- Smoke audits : graph constant, gold oracle=1.0, baselines ≈ chance.
- Disclaimer : v1 is a mechanistic validation bench, NOT a benchmark.

## 4. Failure modes discovered

### 4.1 Step-aware necessity (without it, the controller never trains)

Pre-step-aware code : `compute_loss` only consumed final-state value
logits → halting controller received no gradient. v1 sanity acc =
0.875 ood (pre-fix) vs 0.965 ood (post-fix).

### 4.2 Final-only attractor (H1a / H1b)

With λ_ponder = 0 or 1e-4 : all halt weight at T_max=16, mean E[steps]
= 16, halting policy uninformative.

### 4.3 Floor-collapse attractor (H1d)

With λ_ponder = 0.01 : all halt weight at min_steps=2 → degenerate.

### 4.4 Linear controller unobservability (H2)

Linear(d_state → 1) controller-only on frozen backbone : mean
E[steps] correct but Spearman ≈ 0. Architecture cannot encode
conditional policy.

### 4.5 Checkpoint selection mismatch

`val_acc` alone undersamples Spearman ; a composite val-only rule
(val_acc-bounded window + max val_spearman + tie-breaks) recovers
+0.07 ood Spearman on H6 without OOD selection.

### 4.6 Clean benchmark capacity failure (v3 / v3.1)

Generator audits pass, but the substrate cannot extract the signal :
selector entropy frozen at log(4), train_acc at chance.

## 5. Method

### 5.1 Step-aware latent loss

`compute_loss` consumes `claim_scores_per_step` × `halt_weights` to
compute `Σ_t halt_w_t · CE_t`. The halting controller gradient flows
through `halt_w_t`.

### 5.2 Enriched controller with anytime aux features

`EnrichedAdaptiveHaltingController` = MLP(d_state + 5 → 128 → 64 → 1)
where the 5 aux features are :
1. `normalized_step = t / T_max`
2. `selector_entropy_t` (over the 4 query CLAIM scores)
3. `selector_max_prob_t`
4. `value_margin_t = top1_value_logit − top2_value_logit`
5. `delta_value_margin_t = value_margin_t − value_margin_{t-1}`

Aux features are **detached** from the selector by default
(`detach_aux_features_from_selector = True`) — closes the gradient
leak that the audit identified.

### 5.3 Two-stage partial-freeze

- **Stage 1 (H4-style)** : controller-only warm-up. Backbone frozen,
  only `halting_controller` trains. 5 epochs, λ_ponder = 0.01.
- **Stage 2 (H5b-style)** : co-train `halting_controller` +
  `claim_selector_head`. Backbone/embeddings still frozen. 5 epochs.

### 5.4 Composite val-only checkpoint policy

- Eligibility : `val_acc ≥ best_val_acc − 0.01`.
- Among eligible : argmax `val_spearman`, then min `val_regret`, then
  min `val_E[steps]`, then earliest epoch.
- **OOD signals never used for selection.**

## 6. Experiments

### 6.1 H1 attractors (single seed)

Show H1a / H1b / H1d failure modes by sweeping λ_ponder.

### 6.2 H2 / H4 controller ablation (single seed)

Compare Linear (H2) vs MLP+aux (H4) controllers, controller-only on
frozen H1b backbone.

### 6.3 H6_detached_aux main result (5 seeds)

Two-stage protocol with patched code. Report mean ood acc, ood
Spearman, regret, E[steps], 0/5 collapse. Headline numbers : ood acc
= 0.843, ood Spearman = 0.231 (official), 0.263 (composite).

### 6.4 H7 / H8 negative ablations (5 seeds each)

- H7 (`expected_step_mse` teacher distillation) : preserves acc,
  destroys policy alignment (−0.143 ood Spearman vs H6_detached_aux).
- H8 (selector LR ×0.03) : preserves acc, intermediate sρ.

### 6.5 v3 / v3.1 stress tests (single seed)

Cleaner harder generators. Smoke audits pass. Fixed-step capacity
fails. Architectural bottleneck.

### 6.6 RGCN+H6 protocol transfer (5 seeds)

To disentangle the H6 protocol from the MBS substrate, we port the
same two-stage partial-freeze recipe and the same enriched MLP
halting controller to the RGCN backbone, initialised from the
trained RGCN ACT post-patch checkpoints. On 5 seeds, `rgcn_h6_two_stage`
reaches OOD acc 0.869 ± 0.019 (vs 0.872 ± 0.013 naive RGCN ACT) with
**0 of 5 seeds collapsing** (vs 5 of 5 for the naive baseline) and
**Spearman(expected_step, required_hops) = 0.60 ± 0.19 on val**.
4 of 5 seeds replicate the MBS H6dau alignment range (0.69–0.75) ;
1 seed is an alignment outlier without collapse. The same
hardest-bucket-boundary effect is reproduced on RGCN with smaller
magnitude (+1.37 step at h=9 vs +2.75 on MBS). Conclusion : the H6
protocol is **not MBS-specific in this v1 setting**, with weaker
cross-seed robustness on RGCN than on MBS.

### 6.7 Three-way comparison

The triple comparison (MBS H6dau, RGCN ACT post-patch, RGCN+H6
two-stage) on the same v1 task isolates the accuracy / halting-alignment
dichotomy : naive ACT achieves comparable accuracy with zero alignment
and 5/5 collapse ; H6 achieves comparable accuracy with bucket-aligned
halting and 0/5 collapse, irrespective of the substrate (within the
4-seed RGCN subset that replicates). Figures in
`results/claim_strengthening/paper_update/fig_*_3way.{png,pdf}`.

## 7. Discussion

- The methodology generalizes : the audit infrastructure (smoke,
  halted accuracy, composite, drift) catches subtle failure modes
  that would otherwise pass silently.
- The two-stage protocol works on the v1 bench but is bounded by
  substrate capacity. Future work must address backbone expressivity.

## 8. Limitations

- v1 is synthetic and mechanistic.
- Hardest-bucket alignment, not fine-grained continuous depth tracking.
- v3 / v3.1 fail capacity (substrate bottleneck, not yet solved).
- 5 epochs / 5 seeds — modest budget.
- Composite selection is retrospectively defined.
- RGCN+H6 protocol transfer has 1/5 outlier seed → cross-seed
  robustness weaker than on MBS H6dau.
- Component-level causal isolation of the protocol's pieces
  (step-aware loss / enriched controller / partial freeze / aux
  features / composite selection) has not been done on RGCN.

## 9. Future work

Four paths, ordered by ambition / cost :

1. **Architecture upgrade** : d_state=128, MLP-depth-3 substrate,
   attention pooling at the selector head. Test capacity on v3.1.
2. **Pre-declared composite selection** on a fresh run set.
3. **Cross-bench validation** : graph reasoning on a non-synthetic
   bench under the same protocol.
4. **Component-level ablation of the H6 protocol on RGCN** :
   ablate step-aware loss / enriched controller / partial freeze /
   aux features individually on RGCN to isolate which components
   are necessary and which are sufficient.
5. **Investigate the RGCN+H6 seed-3 outlier** : determine whether
   the alignment failure is reproducible across Python seeds, and
   whether it is already present at end of Stage 1 or emerges only
   during Stage 2 co-training.

---

## Sections that need data / figures

- Figure 1 : The 6 failure modes summary (panel of E[steps] vs
  λ_ponder showing H1a/H1b/H1d, plus Spearman vs controller architecture
  for H2/H4, plus checkpoint selection effect for H6).
  Data : `figures_data/halting_attractors_H1.csv`,
  `controller_ablation_H2_H4_H6.csv`,
  `checkpoint_selection_effect.csv`.
- Table 1 : Main results 5-seed means.
  Data : `tables/main_results_table.csv`.
- Table 2 : Halted accuracy semantics (mixture / expected / chosen).
  Data : `figures_data/code_audit_halted_accuracy.csv`.
- Table 3 : v3 / v3.1 stress tests.
  Data : `figures_data/v3_stress_tests.csv`.
- Figure 2 (optional) : composite vs official ood Spearman bars
  across H6 / H6_detached_aux / H7 / H8.

All artifacts ready in `results/final_scientific_package/`.
