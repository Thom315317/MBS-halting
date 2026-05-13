# RGCN + H6_detached_aux — paper decision (Phase 3)

- date: 2026-05-13
- input : `RGCN_H6_TWO_STAGE_5SEED_REPORT.md` (5-seed run, 0/5 collapse,
  4/5 alignment replication, mean Spearman 0.60 ± 0.19).
- target : decide what changes in the paper given the protocol-transfer
  result. **No paper file modified** : this report only enumerates the
  paste-ready text patches and identifies which sections of the existing
  `data_audit/DATA_AUDIT_FINAL_REPORT_PATCHED.md` future-work block become
  obsolete.

## 1. Verdict

→ **A_partial_transfer_with_one_outlier.**

Per the verdict alphabet from the brief :

| label | meaning | this campaign |
|---|---|---|
| A | protocol transfers cleanly to RGCN | ✗ (4/5 only) |
| **A′** | **protocol transfers on 4/5 seeds, 1 outlier** | **✓ this** |
| B | protocol transfers partially (some criteria met, others not) | n/a |
| C | protocol does not transfer (RGCN+H6 collapses or fails to align) | ✗ |
| D | inconclusive (e.g. one seed only) | n/a |

The 5-seed data supports A′ : the protocol does transfer (the same
hardest-bucket-boundary alignment emerges on RGCN, the same collapse
failure mode is fixed) but the cross-seed robustness is weaker than on MBS
(1 outlier vs 0 outliers on the existing MBS H6dau 5-seed campaign).

## 2. What the result lets the paper now say

**Newly allowed (5-seed data behind each):**

- "The H6_detached_aux protocol transfers to the RGCN backbone : on 4 of
  5 seeds the same Spearman(expected_step, required_hops) ≈ 0.69–0.75
  alignment as on MBS is produced ; 0 of 5 seeds collapse." (5-seed
  Spearman cross-seed mean 0.60 ± 0.19 with the seed-3 outlier, 0.72 ±
  0.03 on the 4-seed subset.)
- "The same hardest-bucket-boundary effect (easy buckets indistinguishable,
  hardest bucket clearly elevated) is reproduced on RGCN+H6 with a smaller
  but non-zero jump (Δh=9 vs h=8 : +1.37 step on RGCN+H6 vs +2.75 step on
  MBS H6dau)."
- "On the **same** RGCN backbone weights, the naive single-stage ACT
  controller collapsed 5/5 seeds at near-identical OOD accuracy (0.872) ;
  the H6_detached_aux protocol produced 0/5 collapse at 0.869 OOD accuracy
  on those same weights. Accuracy is not the discriminator ; the halting
  policy is."

**Still NOT allowed (unchanged) :**

- "fine-grained continuous depth tracking" — still hardest-bucket-only on
  RGCN+H6, same as on MBS.
- "the protocol is causally isolated" — we still did not ablate
  step-aware loss / enriched controller / partial freeze / aux features /
  composite selection individually on RGCN.
- "MBS superior to RGCN" — accuracy comparable, halting alignment
  qualitatively similar but quantitatively smaller on RGCN.
- "SOTA" — no community-benchmark comparison.

## 3. Patches to existing paper text

### 3.a — `data_audit/DATA_AUDIT_FINAL_REPORT_PATCHED.md` §4 ("Verdict H6 vs RGCN ACT")

The current "NOT demonstrated by current experiments" item
> *"The H6 protocol transfers to RGCN."*

is now **partially demonstrated** and should be moved.

Replacement text :

> **Demonstrated** (after Phase 2 of `rgcn_h6_two_stage`) :
>
> - The H6_detached_aux protocol transfers to the RGCN backbone on 4 of 5
>   seeds, producing Spearman(expected_step, required_hops) ≈ 0.69–0.75
>   (the H6dau 5-seed range was 0.66–0.74). 0 of 5 RGCN+H6 seeds collapse
>   (vs 5 of 5 for naive RGCN ACT post-patch on the same backbone weights).
>
> **Partially demonstrated** :
>
> - Cross-seed robustness is weaker on RGCN+H6 than on MBS H6dau : 1 of 5
>   seeds (seed 3) does not replicate the alignment (Spearman ≈ 0.14–0.22)
>   despite no collapse. The 5-seed mean is therefore 0.60 ± 0.19, vs
>   0.69 ± 0.03 on MBS.
>
> **Still not demonstrated** (unchanged) :
>
> - MBS is superior to RGCN (accuracy is comparable on both substrates).
> - The protocol is itself the isolated causal factor — only the bundled
>   protocol was tested, not its components individually.

### 3.b — `data_audit/PAPER_PATCH_TEXT.md` §3 (RGCN ACT post-patch disclaimer)

The current disclaimer
> *"This is a strong negative control for a naive ACT halting controller,
> but it does NOT disentangle 'architecture' from 'protocol' : we have
> not ported the enriched two-stage protocol (H4-warmup +
> EnrichedAdaptiveHaltingController + anytime aux features + composite
> val-only selection) to the RGCN backbone. Doing so is left as future
> work."*

should be replaced with the new RGCN+H6 result :

> *Updated.* We did port the H6_detached_aux protocol to the RGCN
> backbone (Phase 2 of `rgcn_h6_two_stage`, 5 seeds) : 4 of 5 seeds
> reproduce the bucket-alignment signal of MBS H6_detached_aux
> (Spearman ≈ 0.69–0.75, indistinguishable from the MBS 5-seed range
> within stdev), and 0 of 5 seeds collapse — vs 5/5 collapse for the
> naive single-stage RGCN ACT post-patch baseline at near-identical
> OOD accuracy (0.869 vs 0.872). The substrate-vs-protocol confound is
> therefore reduced : "RGCN ACT collapses, RGCN+H6 does not" is now a
> direct comparison on the same backbone. Cross-seed robustness on RGCN
> is weaker than on MBS (1/5 outlier vs 0/5), which we report as a
> remaining limitation.

### 3.c — Paste-ready paragraph for §"Discussion — protocol transfer"

> **Protocol transfer to a non-MBS substrate.** To disentangle the H6
> protocol from the MBS substrate, we ported the same two-stage
> partial-freeze recipe (H4-style controller-only warmup → H5b-style
> co-train of `halting_controller` + `claim_selector_head`) and the same
> EnrichedAdaptiveHaltingController (MLP `d_state+5 → 128 → 64 → 1`) with
> the same five anytime aux features to the RGCN backbone, initialised
> from the trained RGCN ACT post-patch checkpoints. On 5 seeds, the
> resulting `rgcn_h6_two_stage` runs achieve OOD accuracy 0.869 ± 0.019
> (vs 0.872 ± 0.013 for the naive RGCN ACT baseline on the same weights),
> with **0 of 5 seeds collapsing** to the floor / final-step (vs 5 of 5
> for the naive baseline) and **Spearman(expected_step, required_hops) =
> 0.60 ± 0.19 on val**. Per-seed, four of the five seeds match the MBS
> H6dau alignment range (0.69–0.75 ; the MBS 5-seed range was 0.66–0.74) ;
> the fifth seed has weak alignment (0.14–0.22) without collapse. The
> hardest-bucket-boundary effect (h=9 vs h=8, +1.37 step) is reproduced
> on RGCN+H6, qualitatively identical to the +2.75-step jump observed on
> MBS H6dau, with smaller magnitude. **The H6 protocol therefore is not
> MBS-specific** : the same recipe induces a bucket-aligned, non-degenerate
> halting policy on a different message-passing substrate.

### 3.d — Figure update recommendation

**Figure B** ("Acc vs halting policy alignment") should add a third marker
type for the new `rgcn_h6_two_stage` runs (e.g. triangle), per-seed scatter
of OOD acc vs Spearman(E[s], required_hops). The expected scatter :
- circles (MBS H6dau, 5 seeds) : cluster around (0.84, 0.68)
- squares (RGCN ACT post-patch, 5 seeds) : cluster around (0.87, 0.0)
- triangles (RGCN+H6 two-stage, 5 seeds) : 4 around (0.87, 0.70) and 1
  outlier around (0.86, 0.20)

The figure would then show the **two new pieces of evidence**:

1. RGCN+H6 occupies the same upper-right quadrant as MBS H6dau on 4 of 5
   seeds (the protocol transfers).
2. The RGCN ACT baseline at the bottom shows that without the protocol,
   the same backbone produces zero halting alignment despite slightly
   higher accuracy.

(Producing the updated figure is out of scope for this report ; it would
extend `scripts/data_audit_phase3_figures.py:make_fig_B(...)` with the
new RGCN+H6 scatter set.)

## 4. Net effect on the paper

The contribution claim now decomposes as :

1. **Methodologically distinctive** — H6_detached_aux combines step-aware
   loss + enriched MLP controller + 2-stage partial-freeze + composite
   val-only selection. This is unchanged.

2. **Negative-control validated** — RGCN + naive single-stage ACT collapses
   5/5 with comparable accuracy. This is unchanged.

3. **Substrate transfer validated (newly added)** — the same protocol on
   the RGCN backbone produces non-degenerate halting (0/5 collapse) and
   replicates the H6dau bucket-alignment signal on 4 of 5 seeds. The
   protocol is therefore not MBS-specific.

4. **Honest limitations** —
   - bucket-level rather than fine-grained alignment (unchanged) ;
   - v3.1 capacity failure is an open architectural problem upstream of
     the readout (unchanged) ;
   - 1 outlier seed on RGCN+H6 → cross-seed robustness slightly weaker
     on RGCN than on MBS (newly added) ;
   - per-component ablation of the protocol on RGCN not done (newly
     added) ;
   - protocol-vs-substrate causal isolation is partial : we now have the
     same protocol on two substrates, but we have not ablated the protocol's
     components (newly added).

The paper is **strictly strengthened** by this Phase 2 result : the
previous "future work : port H6 to RGCN" item is now a 5-seed empirical
result rather than a promise, and the substrate-vs-protocol confound that
weakened the original H6dau-vs-RGCN-ACT comparison is reduced.

## 5. Files produced by this Phase 3

| produit | path |
|---|---|
| this report | `RGCN_H6_PAPER_DECISION.md` |

No paper file modified, no figure regenerated. The patches listed in §3
are paste-ready and can be applied to the existing PATCHED.md / PAPER_PATCH
text files when the paper draft is opened for revision.
