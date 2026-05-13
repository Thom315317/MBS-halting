# Ordinal audit — `rgcn_h7_seed2_w001`

- input per-sample CSV : `/home/thom315/MBS-halting-h7/results/claim_strengthening/h7_ordinal_halting/seed2_w001/v3_seed2_w001_per_seed.csv`
- input summary.json    : `/home/thom315/MBS-halting-h7/results/claim_strengthening/h7_ordinal_halting/seed2_w001/v3_seed2_w001_summary.json`
- baseline_val_acc      : `0.867`

## Per (seed × split) cells

| seed | split | n | acc | S_all | S_easy | AUC9 | MACRO_AUC | spread | adj_mean | entropy | dom_mass | flags |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 2 | ood_mixed | 512 | 0.8945 | +0.695 | +0.081 | +1.000 | +0.850 | +1.341 | +0.335 | +1.152 | +0.689 | binary_h9_shortcut |
| 2 | val | 512 | 0.8809 | +0.697 | +0.041 | +1.000 | +0.850 | +1.364 | +0.341 | +1.204 | +0.676 | binary_h9_shortcut |

## Per-cell bucket means (E[step] by required_hops)

| seed | split | h=5 | h=6 | h=7 | h=8 | h=9 |
|---:|---|---:|---:|---:|---:|---:|
| 2 | ood_mixed | 3.00 | 3.00 | 3.00 | 3.00 | 4.34 |
| 2 | val | 3.00 | 3.00 | 3.00 | 3.00 | 4.36 |

## Per-cell notes

