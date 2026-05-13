# Ordinal audit — `rgcn_h7_seed3_w001`

- input per-sample CSV : `results/claim_strengthening/h7_ordinal_halting/seed3_w001/v3_seed3_w001_per_seed.csv`
- input summary.json    : `results/claim_strengthening/h7_ordinal_halting/seed3_w001/v3_seed3_w001_summary.json`
- baseline_val_acc      : `0.8555`

## Per (seed × split) cells

| seed | split | n | acc | S_all | S_easy | AUC9 | MACRO_AUC | spread | adj_mean | entropy | dom_mass | flags |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 3 | ood_mixed | 512 | 0.8828 | +0.705 | +0.009 | +1.000 | +0.859 | +3.221 | +0.805 | +1.860 | +0.568 | binary_h9_shortcut |
| 3 | val | 512 | 0.8555 | +0.761 | +0.153 | +1.000 | +0.888 | +3.140 | +0.785 | +1.856 | +0.559 | ordinal_healthy |

## Per-cell bucket means (E[step] by required_hops)

| seed | split | h=5 | h=6 | h=7 | h=8 | h=9 |
|---:|---|---:|---:|---:|---:|---:|
| 3 | ood_mixed | 5.07 | 5.09 | 5.13 | 5.12 | 8.29 |
| 3 | val | 5.06 | 5.09 | 5.14 | 5.07 | 8.20 |

## Per-cell notes

