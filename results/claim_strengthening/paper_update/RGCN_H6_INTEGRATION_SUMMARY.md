# RGCN+H6 — paper integration summary

- date: 2026-05-13
- target: integrate the `rgcn_h6_two_stage` 5-seed campaign into the
  paper's main result + discussion + limitations, and ship updated
  3-way figures + table.

## 1. Executive summary (10 lines)

We ported the H6_detached_aux protocol (two-stage partial-freeze, enriched
MLP halting controller with 5 anytime aux features, step-aware latent
loss, val-only selection) to the RGCN backbone, initialised from the
existing RGCN ACT post-patch checkpoints. On 5 seeds, `rgcn_h6_two_stage`
reaches OOD accuracy 0.869 ± 0.019 with 0/5 floor or final collapse and
Spearman(expected_step, required_hops) = 0.60 ± 0.19 on val. 4 of 5 seeds
match the MBS H6dau alignment range (0.69–0.75) ; 1 seed is an alignment
outlier without collapse. Compared to the naive single-stage RGCN ACT
baseline on the same backbone weights (0.872 OOD acc, 5/5 collapse,
Spearman ≈ 0), the protocol fixes the collapse failure mode at
near-identical accuracy. The H6 protocol is therefore **not MBS-specific
in this v1 setting**, with weaker cross-seed robustness on RGCN than
on MBS.

## 2. Figure list (paths)

| figure | png | pdf |
|---|---|---|
| Acc vs policy alignment, OOD, 3-way per-seed scatter | `fig_acc_vs_policy_3way.png` | `fig_acc_vs_policy_3way.pdf` |
| Collapse mode counts (floor / final per seed, 3-way) | `fig_collapse_modes_3way.png` | `fig_collapse_modes_3way.pdf` |
| Bucket alignment (MBS vs RGCN+H6, val + ood) | `fig_bucket_alignment_mbs_vs_rgcn_h6.png` | `fig_bucket_alignment_mbs_vs_rgcn_h6.pdf` |
| (unchanged) v3.1 bottleneck diagnostic | `../data_audit/fig_v31_bottleneck.png` | `../data_audit/fig_v31_bottleneck.pdf` |

Source script : `scripts/make_rgcn_h6_3way_figures.py`.

## 3. Updated main claim

> A two-stage partial-freeze training protocol with an enriched MLP
> halting controller and five anytime aux features produces a
> non-degenerate bucket-aligned halting policy on the
> `depth_controlled_latent_halting_probe` v1 task. The protocol works
> on both the MBS substrate (5/5 seeds, OOD Spearman 0.678 ± 0.032)
> and the RGCN substrate (4/5 seeds, OOD Spearman in [0.69, 0.71]),
> with 0/5 collapse on each. On the same RGCN backbone, the naive
> single-stage ACT controller collapses 5/5 at near-identical accuracy ;
> the protocol therefore fixes a substrate-independent failure mode of
> naive ACT, and is **not MBS-specific in this v1 setting**.

## 4. Updated limitations

- The alignment is **bucket-level**, not fine-grained continuous depth
  tracking (the effect is dominated by the h=9 hardest-bucket boundary
  on both substrates ; the 4 easy buckets are indistinguishable).
- The RGCN+H6 protocol transfer has **1/5 outlier seed** : cross-seed
  robustness on RGCN is weaker than on MBS H6dau (stdev 0.19 vs 0.024
  for the 5-seed mean ; 0.025 vs 0.024 on the 4-seed subset).
- **No component-level ablation** of the protocol (step-aware loss,
  enriched controller, partial freeze, aux features, composite
  selection) has been run on RGCN. Only the bundled protocol is tested.
- The **v3.1 stress test** capacity failure stands and is upstream of
  the readout. Plausible causes (under-propagation, aggregation
  bottleneck, missing residuals, missing hop encodings) have not been
  individually validated.
- v1 is a **mechanistic / synthetic validation bench**, not a community
  benchmark.

## 5. Paste-ready paragraph — Results

> **Results — protocol transfer to a non-MBS substrate.** To disentangle
> the H6_detached_aux protocol from the MBS substrate, we ported the
> same two-stage partial-freeze recipe and the same enriched MLP
> halting controller to the RGCN backbone (`mbs/baselines.py:
> RelationalGCNHaltingClassifier` extended with the same enriched
> halting path as `MBSModel`, initialised from the trained RGCN ACT
> post-patch checkpoints ; see `mbs/{halting,model,baselines,train}.py`
> patches P1..P4). On 5 seeds, the resulting `rgcn_h6_two_stage` runs
> achieve **OOD accuracy 0.869 ± 0.019** (vs 0.872 ± 0.013 for the
> naive RGCN ACT baseline on the same weights), with **0 of 5 seeds
> collapsing** to the floor / final step (vs 5 of 5 for the naive
> baseline) and **Spearman(expected_step, required_hops) = 0.60 ± 0.19
> on val**. Per-seed, four of five seeds match the MBS H6dau alignment
> range (0.69–0.75 ; MBS 5-seed range was 0.66–0.74). The fifth seed
> has weak alignment (0.14–0.22) without collapse. The
> hardest-bucket-boundary effect is reproduced qualitatively (+1.37
> step at h=9 on RGCN+H6 val ; +2.75 step on MBS H6dau val). See
> Figure (acc-vs-policy 3-way), Figure (collapse 3-way), Figure
> (bucket alignment MBS vs RGCN+H6), and the main 3-way results
> table.

## 6. Paste-ready paragraph — Discussion

> **Discussion — accuracy and halting alignment are decoupled.** The
> three-way comparison isolates the accuracy / policy dichotomy : the
> RGCN ACT post-patch baseline reaches the highest OOD accuracy
> (0.872) of the three configurations, yet its halting policy is
> degenerate (Spearman ≈ 0, chosen_step constant per seed, 5/5
> collapse). The RGCN+H6 two-stage protocol on the same backbone
> achieves comparable accuracy (0.869) and recovers a bucket-aligned,
> non-degenerate halting policy in 4/5 seeds, with 0/5 collapse. The
> MBS H6dau protocol reaches a slightly lower accuracy (0.843) on a
> weaker substrate but the same qualitative halting behaviour
> (5/5 alignment replication, 0/5 collapse, hardest-bucket detection).
> Two consequences : (i) high task accuracy is **not a sufficient
> signal** of halting policy health — the audit infrastructure
> (`halt_probs`, `halt_weights`, `chosen_step_distinct`, floor / final
> masses, Spearman vs `required_hops`) is necessary to catch silent
> collapse. (ii) The H6 protocol is **not MBS-specific in this v1
> setting** : the bucket-aligned halting policy emerges on a different
> message-passing substrate with the same protocol, even though the
> magnitude of the hardest-bucket jump is smaller on RGCN.

## 7. Paste-ready paragraph — Limitations

> **Limitations — what this work does not establish.** We characterise
> the H6 protocol as transferring to a non-MBS substrate **in this v1
> setting**, but several caveats hold. **Cross-seed robustness on
> RGCN+H6 is weaker than on MBS** : 1/5 RGCN+H6 seeds (seed 3) has
> weak alignment (Spearman ≈ 0.14–0.22) without collapse, dragging
> the 5-seed mean down to 0.60 ± 0.19 ; the 4-seed subset that
> replicates is 0.72 ± 0.03, indistinguishable from the MBS 5-seed
> mean within stdev. We have not investigated whether the outlier is
> reproducible across Python seeds or whether it emerges at Stage 1
> or Stage 2. **No component-level ablation** of the protocol was run
> on RGCN — only the bundled recipe (step-aware loss + enriched
> controller + partial freeze + aux features + val-only selection) is
> tested. The alignment is **bucket-level**, dominated by the h=9
> boundary on both substrates ; we do not claim fine-grained
> continuous depth tracking. The **v3.1 capacity failure** of the
> substrate stands as an open architectural problem upstream of the
> readout. **v1 is a mechanistic / synthetic bench**, not a community
> benchmark ; transfer outside v1 is untested.

## 8. Claims allowed (5-seed data-supported)

- "RGCN+H6 eliminates collapse in 5/5 seeds."
  (floor_mass max = 0.014, final_mass = 0.000 ; collapse threshold 0.5
  not crossed on any seed.)
- "RGCN+H6 recovers bucket-aligned halting in 4/5 seeds."
  (Spearman(E[s], required_hops) in [0.69, 0.71] for seeds 1, 2, 4, 5.)
- "The H6 protocol is not MBS-specific in this v1 setting."
  (Two substrates tested ; alignment + collapse-free behaviour on both.)
- "Accuracy and halting alignment are decoupled."
  (RGCN ACT 0.872 acc / 0 Spearman / 5/5 collapse vs RGCN+H6 0.869 /
  0.60 / 0/5 collapse on the same backbone.)

## 9. Claims forbidden (NOT data-supported / over-claim)

- "Fully substrate-agnostic protocol."
  (Two substrates only ; one synthetic v1 task only.)
- "Component-level causal isolation."
  (No per-component ablation on RGCN ; only the bundled protocol.)
- "Fine-grained continuous depth tracking."
  (Hardest-bucket-boundary effect dominates ; the 4 easy buckets are
  indistinguishable on both substrates.)
- "MBS superior to RGCN."
  (Accuracy comparable ; halting alignment qualitatively similar ;
  RGCN+H6 even has higher per-seed Spearman on 4/5 seeds.)
- "v3.1 solved."
  (Capacity failure stands.)
- "SOTA."
  (No comparison to other halting methods on a community benchmark.)
- "Real wall-clock compute saving."
  (No early-exit implementation ; E[steps] is an oracle abstraction.)

## 10. Files index

| produit | path |
|---|---|
| this summary | `RGCN_H6_INTEGRATION_SUMMARY.md` |
| 3-way main table | `table_main_3way.md` |
| Figure 1 (acc vs policy, 3-way) | `fig_acc_vs_policy_3way.{png,pdf}` |
| Figure 2 (collapse modes, 3-way) | `fig_collapse_modes_3way.{png,pdf}` |
| Figure 3 (bucket alignment, MBS vs RGCN+H6) | `fig_bucket_alignment_mbs_vs_rgcn_h6.{png,pdf}` |
| Source script | `scripts/make_rgcn_h6_3way_figures.py` |
| 5-seed full report (input) | `../rgcn_h6_two_stage/RGCN_H6_TWO_STAGE_5SEED_REPORT.md` |
| paper decision (input) | `../rgcn_h6_two_stage/RGCN_H6_PAPER_DECISION.md` |
| upstream patched audit | `../data_audit/DATA_AUDIT_FINAL_REPORT_PATCHED.md` |
| upstream paper patch text | `../data_audit/PAPER_PATCH_TEXT.md` |
