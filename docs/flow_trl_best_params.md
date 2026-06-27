# Flow TRL Best Params

Source: `docs/flow_trl_feval_results_choi.csv` (local generated docs use the `_choi` suffix).

Selection rule: per-env best by final-eval IDM success rate. Ties are broken by ACTOR success rate.

Fixed training params:

- `subgoal_distribution: flow`
- `critic_type: trl`
- `subgoal_num_samples: 1`
- `subgoal_value_weight_max: 5`
- `train_epochs: 600`
- `horizon: 25` in the source sweep

## Best By Env

| env | config | gap | eval_N | IDM | ACTOR | gamma |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| puzzle-3x3-play-v0 | `p3_g1_w5_n1` | 1.0 | 32 | 0.672 | 0.392 | 0.99 |
| puzzle-4x4-play-v0 | `p4_g5_w5_n1` | 5.0 | 16 | 0.800 | 0.688 | 0.99 |
| cube-single-play-v0 | `cs_g10_w5_n1` | 10.0 | 2 | 0.768 | 0.656 | 0.99 |
| cube-double-play-v0 | `cd_g10_w5_n1` | 10.0 | 8 | 0.680 | 0.648 | 0.99 |
| cube-triple-play-v0 | `ct_g10_w5_n1` | 10.0 | 8 | 0.328 | 0.208 | 0.995 |
| antmaze-medium-navigate-v0 | `amm_g3_w5_n1` | 3.0 | 8 | 0.960 | 0.928 | 0.99 |
| antmaze-large-navigate-v0 | `aml_g10_w5_n1` | 10.0 | 16 | 0.768 | 0.728 | 0.995 |
| antmaze-giant-navigate-v0 | `amg_g3_w5_n1` | 3.0 | 8 | 0.232 | 0.216 | 0.999 |

## K=40 Follow-Up

Use the table above with `horizon: 40`.

- Keep `subgoal_num_samples: 1` for training.
- Keep `subgoal_value_weight_max: 5`.
- Keep env-specific TRL critic settings and tuned gamma from `utils/trl_critic_config.py`.
- Run final eval only at the selected `eval_N` for each env.

Generated configs:

- `config/sweep_flow_trl_k40_best/`

Runner:

- `scripts/run_flow_trl_k40_best.sh`
