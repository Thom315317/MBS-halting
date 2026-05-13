# Ordinal audit — `rgcn_h7_seed1_w001`

- input per-sample CSV : `/home/thom315/MBS-halting-h7/results/claim_strengthening/h7_ordinal_halting/seed1_w001/v3_seed1_w001_per_seed.csv`
- input summary.json    : `/home/thom315/MBS-halting-h7/results/claim_strengthening/h7_ordinal_halting/seed1_w001/v3_seed1_w001_summary.json`
- baseline_val_acc      : `0.867`

## Per (seed × split) cells

| seed | split | n | acc | S_all | S_easy | AUC9 | MACRO_AUC | spread | adj_mean | entropy | dom_mass | flags |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | ood_mixed | 512 | 0.8867 | +0.689 | -0.080 | +1.000 | +0.834 | +1.856 | +0.459 | +1.166 | +0.633 | binary_h9_shortcut |
| 1 | val | 512 | 0.8457 | +0.730 | +0.033 | +1.000 | +0.863 | +1.861 | +0.458 | +1.160 | +0.629 | binary_h9_shortcut |

## Per-cell bucket means (E[step] by required_hops)

| seed | split | h=5 | h=6 | h=7 | h=8 | h=9 |
|---:|---|---:|---:|---:|---:|---:|
| 1 | ood_mixed | 3.03 | 3.01 | 3.06 | 3.02 | 4.87 |
| 1 | val | 3.03 | 3.00 | 3.05 | 3.07 | 4.86 |

## Per-cell notes

