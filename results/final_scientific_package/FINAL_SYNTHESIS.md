# FINAL SYNTHESIS — adaptive halting on belief-graph reasoning

- date: 2026-05-12
- package: `results/final_scientific_package/`

---

## 1. Résumé exécutif (≤10 lignes)

H6_detached_aux est le résultat propre principal : pipeline two-stage
partial-freeze (warm-up controller-only puis co-train controller +
selector_head, backbone gelé), MLP halting controller enrichi avec 5
features anytime détachées du selector, code patché par le CODE_AUDIT.
Sur 5 seeds : 0/5 floor/final collapse, mean ood acc = 0.843, mean
ood Spearman = 0.231 (official val_acc) / 0.263 (composite val-only),
Δ vs H6 legacy ≤ 0.004 — le gradient leak via aux features n'était pas
le mécanisme dominant. H7 (teacher MSE distillation) et H8 (slow
selector LR) sont des négatifs honnêtes. v3 / v3.1 sont des stress
tests propres mais échouent en fixed-step capacity, révélant un goulot
expressif du backbone MBS (d_state=96, MLP-depth-2, Linear(96→1)
selector), pas un échec du protocole.

## 2. Résultat principal propre

**H6_detached_aux (5 seeds, code patché)**

| metric | mean ± std | range |
|---|---:|---|
| ood acc | 0.843 ± 0.019 | [0.816, 0.881] |
| ood Spearman (official) | 0.231 ± 0.064 | [0.131, 0.359] |
| ood Spearman (composite val-only) | **0.263** ± 0.085 | [0.131, 0.359] |
| val regret | 0.038 ± 0.005 | [0.030, 0.050] |
| val E[steps] | 5.78 ± 0.43 | [5.10, 6.45] |
| floor / final collapse | 0 / 0 | — |

Verdict aggregate : **`h6_detached_aux_stable`**. Composite selection
verdict : **`h6dau_composite_inconclusive`** (gain +0.031 réel mais
sub-threshold du `policy_improved`).

## 3. Résultats négatifs importants

- **H7 (teacher MSE distillation)** : mean ood acc préservée (0.840)
  mais mean ood Spearman tombe à 0.085 (Δ = −0.143 vs H6_detached_aux).
  La distillation `expected_step_mse` ramène la policy vers le teacher
  H4 stage 1 qui était lui-même médiocre — propagation des défauts.
- **H8 (slow selector LR ×0.03)** : acc préservée (0.843), ood
  Spearman intermédiaire 0.142. Le LR ralenti freine la dérive mais ne
  restaure pas l'alignment. Composite selection remonte H8 à
  ood Spearman = 0.231, presque à parité avec H6_detached_aux.
- **v3 fixed T=16** : best val_acc = 0.297 — selector_entropy figé à
  log(4), max_prob = 0.25. Capacity FAIL.
- **v3.1 fixed T=12** : best val_acc = 0.260, ood = 0.242 — même
  pattern. Confirme que le bottleneck est architectural, pas
  generator-difficulty.

## 4. Ce qui est publiable maintenant

Un papier méthodologique / workshop sur :
1. Six failure modes (step-aware nécessité, final-only attractor,
   floor-collapse attractor, linear controller unobservable,
   checkpoint selection mismatch, clean-bench capacity bottleneck).
2. Une infrastructure d'audit qui les détecte (smoke, halted accuracy
   metrics, composite checkpoint, policy drift, code audit).
3. Un protocole two-stage partial-freeze qui survit aux 5 premiers
   modes sur la v1 mécanistique (`H6_detached_aux`).
4. Un échec instructif honnête : la v3 / v3.1 expose un goulot du
   backbone qui n'est pas résolu — c'est une **limitation positionnée**,
   pas un échec caché.

Tout est dans `results/final_scientific_package/` :
- `tables/main_results_table.md` — table principale prête.
- `figures_data/` — 5 CSV prêtes pour figures.
- `CLAIMS_AND_LIMITS_FINAL.md` — liste autorisée / interdite.
- `REVIEWER2_HOSTILE_MEMO.md` — 10 critiques triées.
- `PAPER_OUTLINE.md` — abstract + 9 sections détaillées.

## 5. Ce qui manque pour un claim plus fort

- **Bench non synthétique** (graph reasoning réel) pour répondre à
  Reviewer 2 C1.
- ~~**Baseline RGCN ACT post-patch** sur v1 (5 seeds, ~2h compute) pour
  répondre à C6.~~ **DONE** : `rgcn_act_postpatch/` (5 seeds, OOD 0.872,
  5/5 collapse) + `rgcn_h6_two_stage/` (5 seeds, OOD 0.869, 0/5 collapse,
  4/5 alignment replication). Voir `paper_update/RGCN_H6_INTEGRATION_SUMMARY.md`.
- **Pre-declared composite selection** sur un run set indépendant
  pour répondre à C5.
- **Architectural upgrade** (d_state ≥ 128, attention pooling, ou MLP
  plus profonde) pour faire passer v3 / v3.1 capacity et étendre le
  protocole au-delà de la v1 mécanistique (C3, C10).
- **10-15 seeds** au lieu de 5 pour réduire la variance et stabiliser
  les bornes (C8).
- **Diagnostic du `train_val_inconsistency`** dans le margin audit
  (technical debt).

## 6. Décision recommandée

**GO write-up méthodologique / workshop sur v1 + audits.**
Tout est en place pour rédiger dans l'état actuel ;
`PAPER_OUTLINE.md` est exécutable comme template.

**NO-GO H6_v3.x sur le backbone actuel.**
v3 / v3.1 capacity fail démontré ; lancer H6_v3.x serait du temps
perdu sans changement architectural.

**TEST architecture upgrade SI compute budget accepté.**
Le plan d'attaque est dans `FUTURE_STRONG_CLAIM_CHECKLIST.md`, Path 2 :
4 ablations × ~1h = ~4h compute, premier qui fait passer v3.1 fixed
T=12 capacity gagne. Si un passe, faire le multiseed complet H6_v3.1.
Si aucun passe, ne PAS aller en Path 3 (v3.2 plus facile) — accepter
Path 1 comme résultat final.

**Documenter v3 / v3.1 comme stress tests propres échoués.**
Inclure les deux comme une section "Limitations / open problems" dans
le papier, avec les chiffres exacts. Pas comme un échec caché ; comme
une découverte honnête sur la capacité du backbone.

---

## Annexe — chemins clés du package

| artefact | chemin |
|---|---|
| ce document | `results/final_scientific_package/FINAL_SYNTHESIS.md` |
| outline papier | `results/final_scientific_package/PAPER_OUTLINE.md` |
| claims/limits | `results/final_scientific_package/CLAIMS_AND_LIMITS_FINAL.md` |
| reviewer memo | `results/final_scientific_package/REVIEWER2_HOSTILE_MEMO.md` |
| future paths | `results/final_scientific_package/FUTURE_STRONG_CLAIM_CHECKLIST.md` |
| table principale | `results/final_scientific_package/tables/main_results_table.{md,csv}` |
| figure data | `results/final_scientific_package/figures_data/*.csv` |
| manifest | `results/final_scientific_package/manifest.json` |
| H6_detached_aux report (copie) | `results/final_scientific_package/reports/H6_DETACHED_AUX_FINAL_REPORT.md` |
| code audit report (copie) | `results/final_scientific_package/reports/CODE_AUDIT_FINAL_REPORT.md` |
| v3 / v3.1 reports (copies) | `results/final_scientific_package/reports/V3_FINAL_REPORT.md`, `V3_1_FINAL_REPORT.md` |
| writeup pack H6dau | `results/final_scientific_package/writeup_pack/` |
| composite selection audit | `results/final_scientific_package/composite_selection_h6_detached_aux/` |
| stress test artefacts v3/v3.1 | `results/final_scientific_package/stress_tests/` |
| design docs v3 / v3.1 | `results/final_scientific_package/design_docs/` |
| configs clés | `results/final_scientific_package/configs/` |
| aggregates JSON cross-seed | `results/final_scientific_package/aggregates/` |
