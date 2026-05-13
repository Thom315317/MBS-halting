# Ordinal audit — `rgcn_h7_seed4_w001`

- input per-sample CSV : `/home/thom315/MBS-halting-h7/results/claim_strengthening/h7_ordinal_halting/seed4_w001/v3_seed4_w001_per_seed.csv`
- input summary.json    : `/home/thom315/MBS-halting-h7/results/claim_strengthening/h7_ordinal_halting/seed4_w001/v3_seed4_w001_summary.json`
- baseline_val_acc      : `0.867`

## Per (seed × split) cells

| seed | split | n | acc | S_all | S_easy | AUC9 | MACRO_AUC | spread | adj_mean | entropy | dom_mass | flags |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 4 | ood_mixed | 512 | 0.8594 | +0.696 | -0.028 | +1.000 | +0.847 | +3.690 | +0.915 | +1.606 | +0.586 | binary_h9_shortcut |
| 4 | val | 512 | 0.8750 | +0.748 | +0.039 | +1.000 | +0.863 | +3.686 | +0.913 | +1.533 | +0.586 | binary_h9_shortcut |

## Per-cell bucket means (E[step] by required_hops)

| seed | split | h=5 | h=6 | h=7 | h=8 | h=9 |
|---:|---|---:|---:|---:|---:|---:|
| 4 | ood_mixed | 4.11 | 4.14 | 4.14 | 4.08 | 7.77 |
| 4 | val | 4.06 | 4.08 | 4.08 | 4.03 | 7.71 |

## Per-cell notes

