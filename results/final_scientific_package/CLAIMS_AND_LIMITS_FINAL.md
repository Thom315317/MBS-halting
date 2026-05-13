# CLAIMS_AND_LIMITS_FINAL — final scientific package

- date: 2026-05-12
- scope: the v1 belief-graph reasoning bench
  (`depth_controlled_latent_halting_probe`) + v3 / v3.1 stress tests on
  the same MBS substrate (d_state=96, MLP-depth-2 message+update,
  Linear(96→1) claim_selector_head).

This document fixes what we can and cannot defend in writing. It
supersedes earlier "claims" sections in per-experiment reports.

---

## 1. Claims défendables

These claims are supported by the evidence in
`results/final_scientific_package/tables/main_results_table.md` and
the cross-seed aggregates linked from the manifest.

1. **Step-aware latent halting was necessary.** Before the step-aware
   patch (see `CODE_AUDIT_FINAL_REPORT`), the latent_claim_selector
   answer head only consumed the final-state value_logits ; the halting
   controller received no answer-related gradient. The pre-step-aware
   v1 sanity probe ceiling was 0.875 ood vs 0.965 ood after the patch,
   and the halting policy was uninformed.

2. **Without step-aware loss, the controller cannot receive the answer
   gradient.** Confirmed by smoke (`scripts/smoke_step_aware_latent.py`)
   and by the inability of pre-fix runs to learn any conditional
   halting policy.

3. **A linear controller is insufficient.** H2 (Linear(d_state → 1)
   controller-only on frozen H1b backbone, λ_ponder=0.01) yields the
   correct mean E[steps] but Spearman ≈ 0 — the architecture cannot
   encode a per-sample conditional policy.

4. **An enriched controller + anytime aux features learns a
   conditional policy under a frozen backbone.** H4 (MLP +
   {normalized_step, selector_entropy, selector_max_prob, value_margin,
   Δvalue_margin}, controller-only on frozen H1b) reaches val Spearman
   = +0.339, val regret = 0.0146, with no floor/final collapse.

5. **Two-stage partial-freeze training avoids the floor/final
   attractors that plague single-stage adaptive halting.** Across H6
   (legacy), H6_detached_aux, H7, H8 — 4 versions × 5 seeds = 20 runs
   — **0/20 had floor_mass>0.5 or final_mass>0.5**.

6. **H6_detached_aux confirms the result in clean code.** With the
   patched code (notably `detach_aux_features_from_selector: true`
   default), the 5-seed mean ood acc=0.843, ood Spearman=0.231 — Δ vs
   H6 legacy ≤ 0.004 on all metrics. The gradient leak via aux
   features was NOT the dominant mechanism.

7. **Composite checkpoint selection on val-only metrics improves the
   policy alignment** when applied to H6_detached_aux : +0.031 ood
   Spearman gain at acc cost ≤ 0.001. The selection rule never uses
   OOD signals (val_acc-bounded window then max val_spearman, tie-break
   val_regret then val_E[steps] then earliest epoch).

8. **v3 / v3.1 generators are clean.** Both pass every smoke audit :
   graph size constant, gold oracle = 1.0, fixed-position baselines ≈
   chance (0.25 ± 0.005), source-id leakage ≈ chance, candidate
   degree uniform, no endpoint candidates, source labels permuted per
   sample, reachability staged (T=4 → 0%, T=8 → 22-37%, T=12 → 88-100%,
   T=16 → 100%), Spearman(required_hops, oracle_depth) > 0.72.

9. **v3 / v3.1 reveal an architectural bottleneck of the MBS substrate
   at d_state=96.** Fixed-step capacity training fails on both
   versions (best val_acc=0.297 v3, 0.260 v3.1) ; train_acc stays at
   chance. Selector entropy stays at log(4). Since v3.1 is strictly
   easier than v3 (shorter chain, lower max depth) and still fails the
   same way, the bottleneck is **not** generator difficulty alone but
   the GNN substrate's inability to propagate rank info over the
   required hops with the current capacity.

## 2. Claims interdits

These are NOT supported by the evidence. Any draft that uses them must
remove or carefully qualify them.

1. **"Adaptive halting solved" or "non-biased final adaptive halting".**
   v1 is a mechanistic validation bench, not a benchmark cleared
   end-to-end ; mean ood Spearman = 0.231 is positive but modest.
2. **"v2 / v3 solved"** — not measured / not measurable on the current
   backbone. v3 and v3.1 explicitly fail capacity.
3. **"SOTA"** — no comparison to other adaptive-halting methods was
   run on this bench, nor was it cross-tested on a community benchmark.
4. **"Real wall-clock compute saving"** — although mean E[steps] is
   ≈5.8 / max 16, every forward still iterates T_max steps to produce
   `halt_weights` ; no actual early-exit was implemented. The compute
   gain is an oracle abstraction, not measured wall-clock.
5. **"No shortcuts anywhere"** — smoke audits cover the obvious
   shortcuts (position, source-id, value frequency, degree, endpoint
   candidates), but not all conceivable shortcuts. In particular,
   `train_val_inconsistency` in the margin audit (train_acc=0.000 on the
   audit eval pass) remains undiagnosed.
6. **Transfer outside v1** — never tested. v3 / v3.1 don't count
   because the backbone can't learn them.
7. **General architectural superiority of MBS** — MBS was used because
   it was the available substrate. No comparison to GAT / transformer /
   GIN with comparable capacity was run. RGCN was added in two
   configurations after the CODE_AUDIT : naive ACT post-patch (5 seeds,
   5/5 collapse, OOD 0.872) and the H6 protocol on RGCN
   (`rgcn_h6_two_stage`, 5 seeds, 0/5 collapse, OOD 0.869, Spearman
   0.60 ± 0.19 with 4/5 alignment replication). RGCN+H6 results show
   that **MBS is not strictly superior on this v1 task** ; the H6
   protocol is not MBS-specific in this v1 setting.

## 3. Caveats

- **v1 is a mechanistic validation bench**, designed to expose
  failure modes (leakage, attractors, unobservability) and to test
  controller architectures, NOT a community benchmark with public
  ranking.
- **v3 / v3.1 fail capacity** on the current backbone. We have not
  yet tested whether a larger backbone or attention pooling would fix
  this.
- **The MBS substrate (d_state=96, MLP-depth-2 message/update, scatter
  aggregation, no attention) cannot propagate rank info over long
  directional chains.** This is observable from the v3 / v3.1 fixed
  capacity failures.
- **H6_detached_aux has seed-by-seed variance** (val Spearman range
  [0.149, 0.416] across 5 seeds). The mean ood Spearman = 0.231 sits at
  the boundary of the conventional "positive policy alignment"
  threshold (0.25). The composite selection lifts the mean to 0.263 ;
  not all seeds benefit equally.
- **Composite checkpoint selection was retrospectively defined** on
  the H6 / H7 / H8 runs. For any future run intended to support
  publication-grade claims, the selection rule must be **pre-declared**
  in the experiment plan and frozen before evaluation. Doing this
  posthoc on the same runs that motivated it is borderline ; the
  results should be read as exploratory rather than confirmatory.
- **The `train_val_inconsistency` issue** in `audit_margin_robustness_v2.py`
  (train_acc=0.000 on the audit eval pass while train_acc=0.85 during
  actual training) is still undiagnosed. It does not propagate to the
  composite verdicts but is a technical debt to close before
  publication.
- **All experiments use 5 epochs per stage** (8 epochs for the v3 /
  v3.1 stress tests). Some seeds plateau by epoch 5, others would
  arguably benefit from more epochs. The conclusions are stable under
  the current budget but may be revised with longer training.

## 4. What would strengthen each defendable claim

(For internal planning ; the brief forbids new experiments now.)

- For **claim 1** (step-aware necessity) : a 2-seed ablation with the
  pre-fix code on v1 sanity to re-confirm the 0.875 → 0.965 jump as a
  controlled A/B.
- For **claim 3** (linear insufficient) : extend H2 to 3 seeds for
  variance.
- For **claim 4** (H4 learns oracle) : 3 seeds with seeds ∈ {2, 3, 4}
  to confirm the conditional policy is reproducible.
- For **claim 7** (composite improves alignment) : pre-declare the
  composite rule on a fresh run set (e.g. a future H6_detached_aux
  multi-seed with k_max=12 v3.x) so the result is confirmatory.
- For **claim 9** (v3 reveals backbone bottleneck) : run a single
  d_state=128 + MLP-depth-3 ablation on v3.1 to verify capacity can
  emerge with reasonable compute uplift.
