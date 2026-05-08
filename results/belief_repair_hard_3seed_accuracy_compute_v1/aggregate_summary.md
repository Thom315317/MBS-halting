# 3-seed accuracy/compute campaign — belief_repair_hard

- root: `results/belief_repair_hard_3seed_accuracy_compute_v1`
- seeds: `[1, 2, 3]`
- verdict: `mbs_compute_efficient_win`
- verdict_details: mbs_ood=0.9948 >= rgcn_fixed_ood-0.02=0.9240, mbs_steps=4.52 <= 0.75*8=6.00, mbs_ood >= rgcn_act_warmup_ood=0.9154

## Per-variant aggregates

| Variant | n_seeds | params | OOD mixed (selected) | OOD mixed (best) | OOD entity | OOD conflict | OOD rule | E[steps] | final_step_mass | ponder_loss | OOD regression from best |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `mbs_adaptive_halting` | 3 | 308860 | 0.9948 ± 0.0060 | 0.9948 ± 0.0060 | 0.9993 ± 0.0011 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 4.52 ± 0.33 | 0.0011 ± 0.0010 | 0.0423 ± 0.0013 | 0.0000 ± 0.0000 |
| `rgcn_repair_stability` | 3 | 176657 | 0.9440 ± 0.0030 | 0.9466 ± 0.0011 | 0.9473 ± 0.0078 | 0.9818 ± 0.0049 | 0.9811 ± 0.0081 | — | — | 0.0000 ± 0.0000 | 0.0026 ± 0.0023 |
| `rgcn_repair_stability_act_warmup_t8` | 3 | 176754 | 0.9154 ± 0.0203 | 0.9258 ± 0.0052 | 0.9160 ± 0.0090 | 0.9577 ± 0.0137 | 0.9577 ± 0.0096 | 7.90 ± 0.37 | 0.0643 ± 0.0231 | 0.0766 ± 0.0023 | 0.0104 ± 0.0180 |

## Compact diagnostic table

| variant | ood_mixed_mean | ood_mixed_std | expected_steps_mean | expected_steps_std | accuracy_per_step | compute_adjusted_score |
|---|---:|---:|---:|---:|---:|---:|
| `mbs_adaptive_halting` | 0.9948 | 0.0060 | 4.52 | 0.33 | 0.2199 | 0.9495 |
| `rgcn_repair_stability` | 0.9440 | 0.0030 | 8.00 | 0.00 | 0.1180 | 0.8640 |
| `rgcn_repair_stability_act_warmup_t8` | 0.9154 | 0.0203 | 7.90 | 0.37 | 0.1158 | 0.8363 |

## Per-seed file paths

### `mbs_adaptive_halting`
- seed1: `results/belief_repair_hard_3seed_accuracy_compute_v1/seed1/mbs_adaptive_halting/benchmark_summary.json`  /  `results/belief_repair_hard_3seed_accuracy_compute_v1/seed1/mbs_adaptive_halting/mbs_adaptive_halting_train_results.json`
- seed2: `results/belief_repair_hard_3seed_accuracy_compute_v1/seed2/mbs_adaptive_halting/benchmark_summary.json`  /  `results/belief_repair_hard_3seed_accuracy_compute_v1/seed2/mbs_adaptive_halting/mbs_adaptive_halting_train_results.json`
- seed3: `results/belief_repair_hard_3seed_accuracy_compute_v1/seed3/mbs_adaptive_halting/benchmark_summary.json`  /  `results/belief_repair_hard_3seed_accuracy_compute_v1/seed3/mbs_adaptive_halting/mbs_adaptive_halting_train_results.json`

### `rgcn_repair_stability`
- seed1: `results/belief_repair_hard_3seed_accuracy_compute_v1/seed1/rgcn_repair_stability/benchmark_summary.json`  /  `results/belief_repair_hard_3seed_accuracy_compute_v1/seed1/rgcn_repair_stability/rgcn_repair_stability_train_results.json`
- seed2: `results/belief_repair_hard_3seed_accuracy_compute_v1/seed2/rgcn_repair_stability/benchmark_summary.json`  /  `results/belief_repair_hard_3seed_accuracy_compute_v1/seed2/rgcn_repair_stability/rgcn_repair_stability_train_results.json`
- seed3: `results/belief_repair_hard_3seed_accuracy_compute_v1/seed3/rgcn_repair_stability/benchmark_summary.json`  /  `results/belief_repair_hard_3seed_accuracy_compute_v1/seed3/rgcn_repair_stability/rgcn_repair_stability_train_results.json`

### `rgcn_repair_stability_act_warmup_t8`
- seed1: `results/belief_repair_hard_3seed_accuracy_compute_v1/seed1/rgcn_repair_stability_act_warmup_t8/benchmark_summary.json`  /  `results/belief_repair_hard_3seed_accuracy_compute_v1/seed1/rgcn_repair_stability_act_warmup_t8/rgcn_repair_stability_act_warmup_t8_train_results.json`
- seed2: `results/belief_repair_hard_3seed_accuracy_compute_v1/seed2/rgcn_repair_stability_act_warmup_t8/benchmark_summary.json`  /  `results/belief_repair_hard_3seed_accuracy_compute_v1/seed2/rgcn_repair_stability_act_warmup_t8/rgcn_repair_stability_act_warmup_t8_train_results.json`
- seed3: `results/belief_repair_hard_3seed_accuracy_compute_v1/seed3/rgcn_repair_stability_act_warmup_t8/benchmark_summary.json`  /  `results/belief_repair_hard_3seed_accuracy_compute_v1/seed3/rgcn_repair_stability_act_warmup_t8/rgcn_repair_stability_act_warmup_t8_train_results.json`

