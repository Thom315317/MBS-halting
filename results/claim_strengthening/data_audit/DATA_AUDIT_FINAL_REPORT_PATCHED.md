# DATA_AUDIT_FINAL_REPORT — post-hoc statistical audit

- date: 2026-05-13
- mode: read-only on existing artefacts (no new training)
- output dir: `results/claim_strengthening/data_audit/`

## 1. Résumé exécutif (≤10 lignes)

L'alignement Spearman(controller_expected_step, required_hops) ≈ 0.69
sur H6_detached_aux (5 seeds × 2 splits) est **réel et reproductible**
mais correspond à une **discrimination grossière du bucket le plus
difficile** (h=9), pas à un fine-grained tracking continu. Les 4
buckets faciles {5, 6, 7, 8} sont **indiscernables** (Cliff δ ≈ 0,
0/10 paires adjacentes significatives sous Holm), tandis que le
boundary (h=8 → h=9) est **universellement détecté** (Cliff δ = 0.98,
10/10 paires significatives, mean Δ ≈ +2.75 step). RGCN ACT post-patch
n'a **aucun signal conditionnel** (chosen_step à variance zéro par
seed, 5/5 seeds collapse au floor ou au final). v3.1 a un bottleneck
**en amont du readout** (cosine distance entre candidates ≈ 10⁻⁴,
probes à chance). Le claim H6 est défendable **comme "structural
difficulty bucket alignment"**, pas comme "fine-grained continuous
depth tracking".

## 2. Verdict bucket-classification vs fine-grained tracking

→ **Bucket classification (binary-ish): YES.** Le contrôleur sépare
proprement le bucket le plus difficile du reste.
→ **Fine-grained tracking: NO.** Aucune séparation significative entre
les 4 buckets faciles.

Verdict Phase 1 : **`B_moderate_bucket_alignment`**.
- Spearman global > 0.5 ✓
- Strict monotonic across all 5 buckets ✗ (0/10 cells)
- R² between buckets > 0.5 ✓ (mean 0.83)
- Effect concentrated on one boundary (h=8 → h=9) ✓

## 3. Verdict solidité du claim required_hops

**Solide pour la formulation "structural difficulty bucket alignment".**

| dimension | verdict |
|---|---|
| reproductibilité cross-seed | ✓ Spearman range [0.63, 0.74] sur 5 seeds, ≤ 0.03 stdev |
| reproductibilité cross-split | ✓ val 0.697 ≈ ood 0.678 |
| indépendance vs CE oracle | ✓ Spearman(oracle, hops) ≈ 0, donc l'effet n'est pas un proxy CE |
| fine-grained continuous | ✗ uniquement la frontière h=8↔9 est significative |
| sémantique "reasoning depth" | ✗ required_hops est une BFS distance, pas une profondeur sémantique |

**Recommandation paper** : reformuler "0.69 Spearman alignment" en
"the controller separates the hardest structural-difficulty bucket
from the rest with Cliff δ = 0.98 across all 5 seeds × 2 splits ;
within the 4 easier buckets the policy is approximately constant".

## 4. Verdict H6 vs RGCN ACT

| dimension | H6_detached_aux | RGCN ACT post-patch | conclusion |
|---|---:|---:|---|
| mean ood acc | 0.843 | **0.872** | RGCN gagne (+0.029) |
| mean Spearman(exp, hops) val | **+0.697** | NaN (var ≈ 0) | H6 gagne |
| chosen_step distinct values per seed | 3.8 | **1.0** | H6 a une policy multi-modale |
| floor/final collapse count | **0/5** | 5/5 (3+2) | H6 robuste, RGCN dégénère |

**Conclusion** : accuracy comparable à RGCN, halting policy
strictement supérieure pour H6dau. H6_detached_aux is the only tested
configuration that produces a non-degenerate bucket-aligned halting
policy ; the RGCN ACT baseline shows that high task accuracy under
naive ACT is insufficient. **Reviewer 2 C6 fermé.**

### What this comparison demonstrates / does not demonstrate

**Demonstrated by data:**

- MBS + H6_detached_aux (two-stage partial-freeze, enriched MLP
  controller, anytime aux features, step-aware loss) avoids
  floor/final collapse and separates the hardest structural-difficulty
  bucket from the rest (Cliff δ = 0.98).
- RGCN + naive single-stage ACT post-patch reaches higher accuracy
  but collapses in 5/5 seeds — high task accuracy is therefore not
  sufficient for a non-degenerate halting policy.
- The H6_detached_aux protocol transfers to the RGCN backbone on
  4 of 5 seeds, producing the same hardest-bucket alignment pattern
  observed on MBS H6_detached_aux. Across all 5 seeds, RGCN+H6
  eliminates floor/final collapse (0/5 collapse versus 5/5 for naive
  RGCN ACT) at comparable OOD accuracy. One seed is an alignment
  outlier, so we report this as partial protocol transfer with
  weaker cross-seed robustness than MBS. See
  `rgcn_h6_two_stage/RGCN_H6_TWO_STAGE_5SEED_REPORT.md`.

**Partially demonstrated (cross-seed robustness):**

- Cross-seed robustness is weaker on RGCN+H6 than on MBS H6dau :
  1 of 5 seeds (seed 3) does not replicate the alignment
  (Spearman ≈ 0.14–0.22) despite no collapse. Five-seed mean
  Spearman = 0.60 ± 0.19 on val (vs 0.697 ± 0.024 for MBS H6dau) ;
  on the 4-seed subset excluding seed 3, the mean is 0.72 ± 0.03,
  indistinguishable from MBS within stdev.

**NOT demonstrated by current experiments:**

- MBS is superior to RGCN (RGCN beats MBS on accuracy by +0.029
  on the naive ACT baseline ; accuracy is comparable on RGCN+H6).
- The protocol is fully substrate-agnostic (we have only two
  substrates tested ; the v1 setting is mechanistic).
- Component-level causal isolation of the protocol (step-aware
  loss / enriched controller / partial freeze / aux features /
  composite selection) — only the bundled protocol was tested.

**What we can say:** H6_detached_aux is a non-degenerate bucket-aligned
halting policy that transfers from MBS to RGCN with partial cross-seed
robustness, while naive ACT collapses on the same RGCN backbone weights.
Accuracy and halting alignment are decoupled : on RGCN, the naive ACT
baseline reaches 0.872 OOD acc with 0 Spearman and 5/5 collapse, while
RGCN+H6 reaches 0.869 OOD acc with mean Spearman 0.60 and 0/5 collapse.

## 5. Patches recommandés dans le papier

1. **Section "Results — halting policy alignment"** : remplacer
   l'unique chiffre "Spearman ≈ 0.69" par la décomposition
   Phase 1 (boundary h=8↔9 versus buckets faciles indiscernables).
   Cf. `PAPER_PATCH_TEXT.md` paragraphe 2.

2. **Section "Method — required_hops definition"** : insérer la
   définition BFS + l'énumération des 5 buckets observés. Reformuler
   "depth alignment" en "structural difficulty bucket alignment".

3. **Section "Experiments — RGCN baseline + RGCN+H6 transfer"** :
   intégrer la table 3-way (MBS H6dau, RGCN ACT, RGCN+H6) avec la
   dichotomie acc-vs-policy. Le disclaimer "future work" est remplacé
   par le résultat 5-seed RGCN+H6 (4/5 alignment replication, 0/5
   collapse, partial cross-seed robustness). Voir
   `paper_update/RGCN_H6_INTEGRATION_SUMMARY.md` et les figures 3-way
   sous `paper_update/fig_*_3way.{png,pdf}`.

4. **Section "Limitations"** : insérer 3 lignes sur v3.1 collapse
   diagnostic, encadrées par "oversmoothing-like representation
   collapse, plausible causes include under-propagation /
   aggregation bottleneck / missing residuals / missing hop
   encodings".

5. **Section "Discussion — accuracy vs policy"** : ajouter le
   paragraphe sur le saut 0.23 → 0.69 (interprétation
   model-dependence de oracle_step).

## 6. Figures générées

| figure | path | usage |
|---|---|---|
| Figure A — H6 bucket monotonicity | `fig_h6_bucket_monotonicity.{png,pdf}` | shows hardest-bucket separation, easy buckets indistinguishable |
| Figure B — Acc vs Spearman tradeoff | `fig_acc_vs_policy.{png,pdf}` | shows H6 vs RGCN per-seed scatter |
| Figure C — Collapse modes | `fig_collapse_modes.{png,pdf}` | shows 0/5 vs 5/5 collapse |
| Figure D — v3.1 bottleneck | `fig_v31_bottleneck.{png,pdf}` | shows cosine ≈ 0, probes at chance, QUERY no signal |

## 7. Claims autorisés après audit

- **"H6_detached_aux aligns expected halting step with structural
  difficulty buckets."**
- **"The alignment is concentrated on the hardest-bucket boundary
  (Cliff's δ = 0.98) rather than on a fine-grained continuous
  depth signal."**
- **"High task accuracy under RGCN ACT does not imply conditional
  halting"** (chosen_step variance is exactly 0 per seed for RGCN).
- **"v3.1 capacity failure occurs upstream of the readout, with
  candidate-state oversmoothing-like collapse"** (cosine ≈ 0,
  probes at chance).
- **"Required_hops is a structural ordinal variable with 5 observed
  levels on v1, derived from the trust-chain BFS distance, and is
  independent of the model's CE trajectory."**

## 8. Claims interdits

- **"fine-grained continuous depth tracking"** — only the
  hardest-bucket boundary is significant.
- **"semantic reasoning depth alignment"** — required_hops is a
  graph BFS distance, not a semantic depth.
- **"protocol proven fully substrate-agnostic"** — H6 protocol has
  been ported to RGCN (4/5 alignment replication, 0/5 collapse) but
  with weaker cross-seed robustness than on MBS ; "not MBS-specific
  in this v1 setting" is allowed, "fully substrate-agnostic" is not.
- **"MBS superior to RGCN"** — RGCN+H6 reaches comparable OOD acc
  (0.869) and bucket-aligned halting ; MBS H6dau is not strictly
  superior.
- **"v3.1 solved"** — capacity failure stands.
- **"real wall-clock compute saving"** — no early-exit implementation.
- **"the controller learns 5 distinct depth levels"** — only 2
  levels are statistically separable (easy vs hardest).
- **"SOTA"** — no comparison to other halting methods on a
  community benchmark.

## 9. Files index

| produit | path |
|---|---|
| Phase 0 inventory | `PHASE0_DATA_INVENTORY.md` |
| Phase 1 audit | `PHASE1_H6_BUCKET_AUDIT.md` |
| Phase 1 bucket CSV | `h6_required_hops_bucket_summary.csv` |
| Phase 1 tests JSON | `h6_required_hops_bucket_tests.json` |
| Phase 2 audit | `PHASE2_H6_VS_RGCN_AUDIT.md` |
| Phase 2 summary CSV | `h6_vs_rgcn_bucket_or_seed_summary.csv` |
| Phase 2 summary JSON | `h6_vs_rgcn_bucket_summary.json` |
| Phase 3 figures (4 × PNG + PDF) | `fig_h6_bucket_monotonicity.{png,pdf}`, `fig_acc_vs_policy.{png,pdf}`, `fig_collapse_modes.{png,pdf}`, `fig_v31_bottleneck.{png,pdf}` |
| Phase 4 paper patch text | `PAPER_PATCH_TEXT.md` |
| this report | `DATA_AUDIT_FINAL_REPORT.md` |

## 10. Net effect on the paper

The audit **does not weaken** the H6_detached_aux contribution — it
**reframes** it from "fine-grained continuous depth alignment" (which
would have been over-claimed) to "**structural difficulty bucket
alignment**, dominated by the hardest-bucket boundary detection". The
contribution remains :

1. methodologically distinctive (step-aware loss + enriched controller
   + 2-stage protocol + composite val-only selection are the components
   that together make up the tested configuration),
2. negative-control validated (RGCN ACT collapses 5/5 with comparable
   accuracy ; high task accuracy under naive ACT is therefore not
   sufficient to produce a non-degenerate halting policy),
3. substrate-transfer validated (the H6 protocol on the same RGCN
   backbone produces 0/5 collapse and replicates the bucket-alignment
   signal on 4/5 seeds ; see `rgcn_h6_two_stage/`),
4. honestly limited (v3.1 capacity fail = open architectural problem ;
   bucket-level rather than fine-grained alignment ; v1 is mechanistic ;
   1/5 RGCN+H6 outlier seed → cross-seed robustness weaker on RGCN
   than on MBS ; component-level causal isolation of the protocol's
   pieces has not been done).

The paper is **stronger** after this audit because the claim is now
**tightly aligned with what the data actually supports**. Specifically,
the framing "H6_detached_aux is the only tested configuration that
produces a non-degenerate bucket-aligned halting policy" replaces any
earlier wording that attributed the result causally to the protocol
in isolation.
