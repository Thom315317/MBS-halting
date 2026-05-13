# Ordinal audit — `rgcn_h7_seed3_w0005`

- input per-sample CSV : `results/claim_strengthening/h7_ordinal_halting/seed3_w0005/v2_seed3_w0005_per_seed.csv`
- input summary.json    : `results/claim_strengthening/h7_ordinal_halting/seed3_w0005/v2_seed3_w0005_summary.json`
- baseline_val_acc      : `0.8555`

## Per (seed × split) cells

| seed | split | n | acc | S_all | S_easy | AUC9 | MACRO_AUC | spread | adj_mean | entropy | dom_mass | flags |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 3 | ood_mixed | 512 | 0.8750 | +0.712 | +0.032 | +1.000 | +0.863 | +2.635 | +0.650 | +1.862 | +0.518 | binary_h9_shortcut |
| 3 | val | 512 | 0.8535 | +0.760 | +0.150 | +1.000 | +0.888 | +2.501 | +0.622 | +1.905 | +0.494 | — |

## Per-cell bucket means (E[step] by required_hops)

| seed | split | h=5 | h=6 | h=7 | h=8 | h=9 |
|---:|---|---:|---:|---:|---:|---:|
| 3 | ood_mixed | 5.21 | 5.26 | 5.27 | 5.17 | 7.81 |
| 3 | val | 5.20 | 5.19 | 5.28 | 5.21 | 7.69 |

## Per-cell notes

