# Ordinal audit — `rgcn_h7_seed5_w001`

- input per-sample CSV : `/home/thom315/MBS-halting-h7/results/claim_strengthening/h7_ordinal_halting/seed5_w001/v3_seed5_w001_per_seed.csv`
- input summary.json    : `/home/thom315/MBS-halting-h7/results/claim_strengthening/h7_ordinal_halting/seed5_w001/v3_seed5_w001_summary.json`
- baseline_val_acc      : `0.867`

## Per (seed × split) cells

| seed | split | n | acc | S_all | S_easy | AUC9 | MACRO_AUC | spread | adj_mean | entropy | dom_mass | flags |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 5 | ood_mixed | 512 | 0.8457 | +0.712 | -0.023 | +1.000 | +0.845 | +1.325 | +0.330 | +1.310 | +0.643 | binary_h9_shortcut |
| 5 | val | 512 | 0.8926 | +0.708 | -0.009 | +1.000 | +0.843 | +1.309 | +0.324 | +1.278 | +0.648 | binary_h9_shortcut |

## Per-cell bucket means (E[step] by required_hops)

| seed | split | h=5 | h=6 | h=7 | h=8 | h=9 |
|---:|---|---:|---:|---:|---:|---:|
| 5 | ood_mixed | 2.98 | 3.00 | 2.98 | 3.00 | 4.30 |
| 5 | val | 2.98 | 3.00 | 2.97 | 3.00 | 4.28 |

## Per-cell notes

