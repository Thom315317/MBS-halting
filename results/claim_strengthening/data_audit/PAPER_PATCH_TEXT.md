# Paper patch text — paste-ready paragraphs after the bucket audit

- date: 2026-05-13
- target: replace any over-claim of "fine-grained continuous depth
  alignment" with the **bucket-level** characterization established
  by the Phase 1 audit.

---

## 1. Paragraph: why Spearman jumps from 0.23 (vs oracle_step) to 0.69 (vs required_hops)

> Our initial measurement of the controller's halting policy used
> the CE-derived oracle step `oracle_step = argmin_t (CE_t + λ · t)`,
> a model-dependent quantity that aggregates the entire CE
> trajectory. Under this proxy, we observed
> Spearman(controller_expected_step, oracle_step) ≈ 0.23 — a modest
> alignment. The CE oracle is, however, itself only weakly
> correlated with the task's structural difficulty
> (Spearman(oracle_step, required_hops) ≈ 0 across our 5 seeds × 2
> splits). When we compare the controller's expected step directly
> to `required_hops` — a model-independent BFS distance derived from
> the generator's metadata — the alignment is Spearman ≈ **0.69**
> (val 0.697 ± 0.024, ood_mixed 0.678 ± 0.033 over 5 seeds). The gap
> between the two figures is therefore an indication that the
> controller is **not merely tracking the model's own CE trajectory**
> ; it is responding to the structural difficulty of the sample.
> The anytime aux features fed to the controller
> (`selector_entropy_t`, `selector_max_prob_t`, `value_margin_t`,
> `Δvalue_margin_t`, `normalized_step`) appear to co-vary with the
> structural difficulty enough for the controller to exploit them.

## 2. Paragraph: required_hops is an ordinal bucket variable

> `required_hops` is the BFS distance along the trust chain from
> the winner's source to the farthest competing candidate (plus 2
> hops for the `FROM_SOURCE` endpoints). For v1 with
> `depth_buckets=[2,4,6,8]` and a fixed `winner_rank=1`, the
> reconstructed `required_hops` takes **five discrete ordinal
> values** {5, 6, 7, 8, 9}. We refer to these as "**structural
> difficulty buckets**" rather than as a continuous depth. We do
> not claim a continuous depth alignment, nor do we equate
> `required_hops` to a semantic notion of reasoning depth. The
> Spearman ≈ 0.69 figure is a rank-correlation between the
> controller's continuous-valued expected step and a 5-bucket
> ordinal scale.

> Within the 4 easy buckets (h=5,6,7,8) the controller halts at
> nearly the same expected step (cross-seed mean ≈ 4.65 ± 0.1,
> Cliff's δ ≈ 0 for adjacent pairs, 0/10 (seed × split) cells with
> Holm-corrected Mann–Whitney significance). At the hardest bucket
> (h=9) the controller halts substantially later (cross-seed mean
> ≈ 7.4 ± 1.6, Cliff's δ = 0.98, 10/10 cells significant under
> Holm). The Spearman ≈ 0.69 figure is therefore **almost entirely
> a hardest-bucket detection effect**, not a fine-grained
> step-by-step tracking of structural difficulty.

## 3. Paragraph: RGCN ACT post-patch disclaimer (updated after RGCN+H6 transfer)

> We compare H6_detached_aux to an RGCN ACT post-patch baseline on
> the same v1 task with the same step-aware latent loss and the
> same `λ_ponder = 0.01` (5 seeds, 10 epochs, 3 warmup epochs at
> T=8 then free ACT). The RGCN baseline reaches a **higher mean ood
> accuracy** (0.872 ± 0.013) than our protocol on MBS (0.843 ± 0.019),
> but the halting controller collapses in 5/5 seeds — 3 to the floor
> (halt_mass at min_steps = 4 above 0.98) and 2 to the final step
> (halt_mass at T_max = 16 near 0.54). Spearman(controller_expected_step,
> required_hops) is 0.02 ± 0.03 for RGCN ACT vs 0.69 ± 0.03 for MBS
> H6_detached_aux ; the chosen_step is a single constant integer
> within each seed (chosen_distinct = 1.0). To disentangle architecture
> from protocol, we then ported the same H6_detached_aux protocol
> (H4-warmup + EnrichedAdaptiveHaltingController + anytime aux features
> + partial freeze + val-only selection) to the RGCN backbone,
> initialised from the trained RGCN ACT post-patch checkpoints. On
> 5 seeds, `rgcn_h6_two_stage` reaches OOD accuracy 0.869 ± 0.019
> with 0/5 collapse and Spearman(E[step], required_hops) = 0.60 ± 0.19
> on val ; 4 of 5 seeds match the MBS H6dau alignment range (0.69–0.75),
> one seed is an alignment outlier (0.14–0.22) without collapse. On
> the **same** RGCN backbone weights, the naive single-stage ACT
> controller collapses 5/5 ; the H6 protocol produces 0/5 collapse at
> near-identical OOD accuracy. We therefore characterise the protocol
> as **not MBS-specific in this v1 setting**, with the caveat that
> cross-seed robustness is weaker on RGCN than on MBS and that we have
> not ablated the protocol's individual components.

## 3.b Updated paragraph: RGCN+H6 protocol transfer

> **Protocol transfer to a non-MBS substrate.** To disentangle the H6
> protocol from the MBS substrate, we ported the same two-stage
> partial-freeze recipe (H4-style controller-only warmup → H5b-style
> co-train of `halting_controller` + `claim_selector_head`) and the
> same EnrichedAdaptiveHaltingController (MLP `d_state+5 → 128 → 64
> → 1`) with the same five anytime aux features to the RGCN backbone,
> initialised from the trained RGCN ACT post-patch checkpoints. On
> 5 seeds, the resulting `rgcn_h6_two_stage` runs achieve OOD
> accuracy 0.869 ± 0.019 (vs 0.872 ± 0.013 for the naive RGCN ACT
> baseline on the same weights), with **0 of 5 seeds collapsing** to
> the floor or final step (vs 5 of 5 for the naive baseline) and
> **Spearman(expected_step, required_hops) = 0.60 ± 0.19 on val**.
> Per-seed, four of the five seeds match the MBS H6dau alignment
> range (0.69–0.75 ; the MBS 5-seed range was 0.66–0.74) ; the fifth
> seed has weak alignment (0.14–0.22) without collapse. The
> hardest-bucket-boundary effect (h=9 vs h=8, +1.37 step on RGCN+H6
> val ; +2.75 step on MBS H6dau val) is reproduced qualitatively
> with smaller magnitude. We therefore conclude that **the H6
> protocol is not MBS-specific in this v1 setting** : the same recipe
> induces a bucket-aligned, non-degenerate halting policy on a
> different message-passing substrate. We do not claim that the
> protocol is fully substrate-agnostic (only two substrates have
> been tested, both on a synthetic v1 bench), nor that its components
> are causally isolated (no per-component ablation was run on RGCN).
> Cross-seed robustness on RGCN is weaker than on MBS (1 outlier
> seed out of 5).

## 4. Paragraph: v3.1 bottleneck diagnostic

> The v3 / v3.1 stress tests are clean by smoke audit (graph size
> constant, gold oracle accuracy = 1, position/source baselines at
> chance, no endpoint candidates, reachability staged from T=4 to
> T=16). However, the fixed-step capacity training plateaus at
> chance (best val_acc = 0.260 on v3.1 fixed T=12 after 8 epochs).
> Probing the v3.1 selected checkpoint reveals that the 4 candidate
> CLAIM final states are quasi-colinear (mean cosine distance
> ≈ 10⁻⁴ across 1024 audited samples), that linear, MLP, and even
> a frozen-backbone pairwise MLP scorer fed
> [h_query, h_claim, h_query⊙h_claim, |h_query−h_claim|] all reach
> only chance accuracy (≈ 0.25), and that a linear probe on the
> QUERY state cannot predict required_hops or oracle_depth (Spearman
> ≈ 0). The bottleneck is therefore **upstream of the readout** :
> the substrate's final-state representation is **collapsed** in a
> way reminiscent of GNN oversmoothing. We do not claim to have
> identified the precise mechanism. Plausible causes include
> under-propagation along the longer trust chain, aggregation
> bottleneck (scatter-sum without attention), missing residual
> pathways at the message level, and the absence of explicit
> hop / depth encodings. We position this as a substrate-side
> limitation to address in future work, not as a protocol-side
> failure of H6_detached_aux (which the v1 result still validates).

## 5. Figure captions (paste-ready)

### Figure A — H6 bucket monotonicity (val & ood_mixed)

> **Figure A.** Cross-seed bucket means of the controller's expected
> halting step versus structural difficulty `required_hops` ∈
> {5, 6, 7, 8, 9}, for H6_detached_aux. Solid line: cross-seed
> mean over 5 seeds. Shaded band: bootstrap 95% confidence interval
> of the seed means. Grey dots: individual seed bucket means. The
> alignment between expected step and required_hops is
> **dominated by the h=9 boundary** : the four easy buckets are
> indistinguishable (mean differences within ±0.1 step, Cliff's δ
> ≈ 0, no significance under Holm correction), while the hardest
> bucket shows a large and consistent jump (mean Δ ≈ +2.75 steps,
> Cliff's δ = 0.98, 10/10 cells significant under Holm).

### Figure B — Accuracy vs halting policy alignment

> **Figure B.** Per-seed OOD accuracy versus
> Spearman(expected_step, required_hops) on the v1 task, for
> H6_detached_aux (circles) and RGCN ACT post-patch (squares).
> Both selected by val_acc (no OOD selection). RGCN reaches higher
> accuracy on most seeds but has Spearman ≈ 0 — its halting policy
> is degenerate per seed. H6_detached_aux trades a small accuracy
> margin for a strong policy alignment (Spearman ≈ 0.6–0.7).

### Figure C — Floor / final collapse counts

> **Figure C.** Per-seed halt-weight mass at the floor (min_steps)
> and the final step (T_max), measured on val. For H6_detached_aux,
> both masses stay near 0 across all 5 seeds — no collapse. For
> RGCN ACT post-patch, every seed shows either floor_mass ≥ 0.98
> (3 seeds) or final_mass ≈ 0.54 (2 seeds) ; the conventional
> collapse threshold (0.5) is crossed in 5/5 seeds. Red dashed
> line : collapse threshold.

### Figure D — v3.1 bottleneck diagnostic

> **Figure D.** Audits on the v3.1 fixed T=12 selected checkpoint
> (val split, 512 samples ; train-half / eval-half cuts for the
> probes). Top-left and top-right : distribution of the L2 and
> cosine distances between the 4 CLAIM final states, per sample.
> The cosine distance is concentrated near 0 (mean ≈ 10⁻⁴),
> indicating that the 4 candidate representations are quasi-
> colinear. Bottom-left : winner-slot accuracy of linear / MLP /
> frozen-backbone pairwise probes on the CLAIM states ; all probes
> remain at chance (red dashed line). Bottom-right : linear probe
> on the QUERY state cannot recover required_hops or oracle_depth
> (Spearman ≈ 0). The bottleneck is upstream of the selector head,
> in the substrate's representation, consistent with an
> oversmoothing-like collapse.
