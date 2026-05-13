# Ordinal audit — `rgcn_h6_baseline_recheck`

- input per-sample CSV : `results/claim_strengthening/rgcn_h6_two_stage/rgcn_h6_two_stage_per_seed.csv`
- input summary.json    : `results/claim_strengthening/rgcn_h6_two_stage/rgcn_h6_two_stage_summary.json`
- baseline_val_acc      : `0.867`

## Per (seed × split) cells

| seed | split | n | acc | S_all | S_easy | AUC9 | MACRO_AUC | spread | adj_mean | entropy | dom_mass | flags |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | ood_mixed | 512 | 0.8926 | +0.689 | -0.080 | +1.000 | +0.834 | +1.195 | +0.299 | +1.084 | +0.652 | binary_h9_shortcut |
| 1 | val | 512 | 0.8496 | +0.729 | +0.027 | +1.000 | +0.862 | +1.181 | +0.295 | +1.076 | +0.646 | binary_h9_shortcut |
| 2 | ood_mixed | 512 | 0.8906 | +0.690 | +0.065 | +1.000 | +0.847 | +1.882 | +0.468 | +1.127 | +0.672 | binary_h9_shortcut |
| 2 | val | 512 | 0.8828 | +0.695 | +0.036 | +1.000 | +0.849 | +1.902 | +0.475 | +1.081 | +0.658 | binary_h9_shortcut |
| 3 | ood_mixed | 512 | 0.8613 | +0.217 | +0.051 | +0.641 | +0.611 | +0.167 | +0.042 | +0.693 | +0.846 | soft_middle_step |
| 3 | val | 512 | 0.8555 | +0.141 | +0.130 | +0.571 | +0.575 | +0.091 | +0.023 | +0.737 | +0.852 | soft_middle_step |
| 4 | ood_mixed | 512 | 0.8535 | +0.697 | -0.025 | +1.000 | +0.847 | +2.820 | +0.692 | +1.567 | +0.562 | binary_h9_shortcut |
| 4 | val | 512 | 0.8594 | +0.750 | +0.046 | +1.000 | +0.864 | +2.795 | +0.691 | +1.531 | +0.562 | binary_h9_shortcut |
| 5 | ood_mixed | 512 | 0.8457 | +0.708 | -0.037 | +1.000 | +0.843 | +0.904 | +0.226 | +0.909 | +0.676 | binary_h9_shortcut |
| 5 | val | 512 | 0.8887 | +0.709 | -0.006 | +1.000 | +0.843 | +0.908 | +0.225 | +0.915 | +0.686 | binary_h9_shortcut |

## Per-cell bucket means (E[step] by required_hops)

| seed | split | h=5 | h=6 | h=7 | h=8 | h=9 |
|---:|---|---:|---:|---:|---:|---:|
| 1 | ood_mixed | 3.00 | 3.00 | 3.00 | 3.00 | 4.20 |
| 1 | val | 3.00 | 3.00 | 3.00 | 3.02 | 4.18 |
| 2 | ood_mixed | 3.04 | 3.08 | 3.03 | 3.06 | 4.91 |
| 2 | val | 3.03 | 3.04 | 3.04 | 3.04 | 4.93 |
| 3 | ood_mixed | 5.07 | 5.09 | 5.08 | 5.13 | 5.24 |
| 3 | val | 5.08 | 5.11 | 5.09 | 5.09 | 5.17 |
| 4 | ood_mixed | 4.15 | 4.15 | 4.17 | 4.10 | 6.92 |
| 4 | val | 4.10 | 4.19 | 4.11 | 4.07 | 6.86 |
| 5 | ood_mixed | 2.98 | 2.98 | 2.98 | 2.98 | 3.88 |
| 5 | val | 2.98 | 2.99 | 2.97 | 2.99 | 3.88 |

## Per-cell notes

- seed3 ood_mixed: soft_middle_step:spread<=0.20+entropy<1.0 (spread=0.167, entropy=0.693); soft_middle_step:dominant_mass>=0.80 (mass=0.846)
- seed3 val: soft_middle_step:spread<=0.20+entropy<1.0 (spread=0.091, entropy=0.737); soft_middle_step:dominant_mass>=0.80 (mass=0.852)
