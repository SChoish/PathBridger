# 환경별 Best Run 파라미터

생성: 2026-06-27 21:00 · `scripts/export_env_best_runs.py`

**선정 규칙:** feval JSON 전체(`runs/` + `checkpoints/flow_trl_best_epoch600/`)에서
**`fs10` suffix 제외** 후 환경별 **ACTOR** 최대 → 동률 시 **IDM** → **timestamp**.

총 **10** environments.

## 요약表

| env | ACTOR | IDM | gap | wmax | train_N | λ | eval_n | temp | suffix | run_group |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| antmaze-giant-navigate-v0 | 37.6% | 32.0% | 3.0 | 5.0 | 1 | 0.0 | 8 | 0.5 | cap4000_t0p5 | flow_trl_feval_amg_g3_w5_n1 |
| antmaze-large-navigate-v0 | 72.8% | 76.8% | 10.0 | 5.0 | 1 | 0.0 | 16 | 1.0 | - | flow_trl_feval_aml_g10_w5_n1 |
| antmaze-medium-navigate-v0 | 96.8% | 96.0% | 3.0 | 5.0 | 1 | 0.0 | 8 | 0.5 | t0p5 | flow_trl_feval_amm_g3_w5_n1 |
| cube-double-play-v0 | 80.0% | 78.4% | 10.0 | 5.0 | 1 | 1.0 | 8 | 0.25 | t0p25 | flow_trl_feval_cd_g10_w5_n1 |
| cube-single-play-v0 | 81.6% | 90.4% | 10.0 | 5.0 | 1 | 0.7 | 2 | 0.5 | t0p5 | flow_trl_feval_cs_g10_w5_n1 |
| cube-triple-play-v0 | 56.0% | 66.4% | 20.0 | 5.0 | 1 | 1.0 | 16 | 0.25 | t0p25 | flow_trl_gap_tune_ct_g20_w5_n1 |
| humanoidmaze-large-navigate-v0 | 1.6% | 2.4% | 10.0 | 5.0 | 1 | 0.1 | 32 | 1.0 | cap2000 | flow_trl_humanoidmaze_dw_hml_g10_w5_n1_d |
| humanoidmaze-medium-navigate-v0 | 9.6% | 12.8% | 10.0 | 5.0 | 1 | 0.1 | 32 | 1.0 | cap2000 | flow_trl_humanoidmaze_dw_hmm_g10_w5_n1_d |
| puzzle-3x3-play-v0 | 65.6% | 65.6% | 0.5 | 5.0 | 1 | 0.5 | 32 | 0.5 | t0p5 | flow_trl_gap_tune_p3_g0p5_w5_n1 |
| puzzle-4x4-play-v0 | 70.4% | 78.4% | 5.0 | 5.0 | 1 | 2.0 | 16 | 0.5 | t0p5 | flow_trl_feval_p4_g5_w5_n1 |

## 환경별 상세

### antmaze-giant-navigate-v0

- run: `/home/choi/Pathbridger_flow/checkpoints/flow_trl_best_epoch600/antmaze_giant_amg_g3_w5_n1_evalN8`
- run_group: `flow_trl_feval_amg_g3_w5_n1`

**학습 파라미터**

- `seed`: 0
- `subgoal`: flow
- `critic_type`: trl
- `gap`: 3.0
- `wmax`: 5.0
- `horizon_K`: 25
- `h_a`: 5
- `train_N`: 1
- `train_eval_N`: 1
- `lambda`: 0.0
- `kappa_b`: 0.9
- `kappa_d`: 0.9
- `gamma`: 0.999
- `batch_size`: 1024
- `train_epochs`: 600
- `final_eval_episodes`: 25
- `subgoal_flow_steps`: 8
- `subgoal_temperature`: 1.0
- `subgoal_goal_representation`: phi
- `subgoal_target_mode`: displacement
- `planner_type`: forward_bridge_residual
- `value_hidden_dims`: [512, 512, 512]

**Eval 파라미터 (best row)**

- `eval_epoch`: 600
- `eval_n`: 8
- `eval_episodes`: 25
- `eval_temperature`: 0.5
- `eval_suffix`: cap4000_t0p5
- `eval_score_type`: transitive_ratio
- `eval_max_chunks`: 200
- `eval_flow_steps`: 8
- `IDM`: 0.32
- `ACTOR`: 0.376
- `idm_tasks`: 1:0.0000,2:0.3600,3:0.2400,4:0.2000,5:0.8000
- `actor_tasks`: 1:0.0800,2:0.2400,3:0.2800,4:0.3600,5:0.9200
- `eval_timestamp`: 2026-06-24T16:15:46Z

### antmaze-large-navigate-v0

- run: `/home/choi/Pathbridger_flow/checkpoints/flow_trl_best_epoch600/antmaze_large_aml_g10_w5_n1_evalN16`
- run_group: `flow_trl_feval_aml_g10_w5_n1`

**학습 파라미터**

- `seed`: 0
- `subgoal`: flow
- `critic_type`: trl
- `gap`: 10.0
- `wmax`: 5.0
- `horizon_K`: 25
- `h_a`: 5
- `train_N`: 1
- `train_eval_N`: 1
- `lambda`: 0.0
- `kappa_b`: 0.9
- `kappa_d`: 0.9
- `gamma`: 0.995
- `batch_size`: 1024
- `train_epochs`: 600
- `final_eval_episodes`: 25
- `subgoal_flow_steps`: 8
- `subgoal_temperature`: 1.0
- `subgoal_goal_representation`: phi
- `subgoal_target_mode`: displacement
- `planner_type`: forward_bridge_residual
- `value_hidden_dims`: [512, 512, 512]

**Eval 파라미터 (best row)**

- `eval_epoch`: 600
- `eval_n`: 16
- `eval_episodes`: 25
- `eval_temperature`: 1.0
- `eval_max_chunks`: 200
- `IDM`: 0.768
- `ACTOR`: 0.728
- `idm_tasks`: 1:0.6800,2:0.7200,3:0.9600,4:0.7600,5:0.7200
- `actor_tasks`: 1:0.4800,2:0.4800,3:0.9200,4:0.9200,5:0.8400
- `eval_timestamp`: 2026-06-21T12:07:50Z

### antmaze-medium-navigate-v0

- run: `/home/choi/Pathbridger_flow/checkpoints/flow_trl_best_epoch600/antmaze_medium_amm_g3_w5_n1_evalN8`
- run_group: `flow_trl_feval_amm_g3_w5_n1`

**학습 파라미터**

- `seed`: 0
- `subgoal`: flow
- `critic_type`: trl
- `gap`: 3.0
- `wmax`: 5.0
- `horizon_K`: 25
- `h_a`: 5
- `train_N`: 1
- `train_eval_N`: 1
- `lambda`: 0.0
- `kappa_b`: 0.9
- `kappa_d`: 0.9
- `gamma`: 0.99
- `batch_size`: 1024
- `train_epochs`: 600
- `final_eval_episodes`: 25
- `subgoal_flow_steps`: 8
- `subgoal_temperature`: 1.0
- `subgoal_goal_representation`: phi
- `subgoal_target_mode`: displacement
- `planner_type`: forward_bridge_residual
- `value_hidden_dims`: [512, 512, 512]

**Eval 파라미터 (best row)**

- `eval_epoch`: 600
- `eval_n`: 8
- `eval_episodes`: 25
- `eval_temperature`: 0.5
- `eval_suffix`: t0p5
- `eval_score_type`: transitive_ratio
- `eval_max_chunks`: 200
- `eval_flow_steps`: 8
- `IDM`: 0.96
- `ACTOR`: 0.968
- `idm_tasks`: 1:0.9600,2:1.0000,3:0.9600,4:0.9600,5:0.9200
- `actor_tasks`: 1:0.9600,2:1.0000,3:1.0000,4:0.8800,5:1.0000
- `eval_timestamp`: 2026-06-22T03:28:40Z

### cube-double-play-v0

- run: `/home/choi/Pathbridger_flow/checkpoints/flow_trl_best_epoch600/cube_double_cd_g10_w5_n1_evalN8`
- run_group: `flow_trl_feval_cd_g10_w5_n1`

**학습 파라미터**

- `seed`: 0
- `subgoal`: flow
- `critic_type`: trl
- `gap`: 10.0
- `wmax`: 5.0
- `horizon_K`: 25
- `h_a`: 5
- `train_N`: 1
- `train_eval_N`: 1
- `lambda`: 1.0
- `kappa_b`: 0.9
- `kappa_d`: 0.9
- `gamma`: 0.99
- `batch_size`: 1024
- `train_epochs`: 600
- `final_eval_episodes`: 25
- `subgoal_flow_steps`: 8
- `subgoal_temperature`: 1.0
- `subgoal_goal_representation`: phi
- `subgoal_target_mode`: displacement
- `planner_type`: forward_bridge_residual
- `value_hidden_dims`: [512, 512, 512]

**Eval 파라미터 (best row)**

- `eval_epoch`: 600
- `eval_n`: 8
- `eval_episodes`: 25
- `eval_temperature`: 0.25
- `eval_suffix`: t0p25
- `eval_score_type`: transitive_ratio
- `eval_max_chunks`: 200
- `eval_flow_steps`: 8
- `IDM`: 0.784
- `ACTOR`: 0.8
- `idm_tasks`: 1:1.0000,2:1.0000,3:0.9600,4:0.0800,5:0.8800
- `actor_tasks`: 1:1.0000,2:1.0000,3:1.0000,4:0.1600,5:0.8400
- `eval_timestamp`: 2026-06-24T13:54:59Z

### cube-single-play-v0

- run: `/home/choi/Pathbridger_flow/checkpoints/flow_trl_best_epoch600/cube_single_cs_g10_w5_n1_evalN2`
- run_group: `flow_trl_feval_cs_g10_w5_n1`

**학습 파라미터**

- `seed`: 0
- `subgoal`: flow
- `critic_type`: trl
- `gap`: 10.0
- `wmax`: 5.0
- `horizon_K`: 25
- `h_a`: 5
- `train_N`: 1
- `train_eval_N`: 1
- `lambda`: 0.7
- `kappa_b`: 0.9
- `kappa_d`: 0.9
- `gamma`: 0.99
- `batch_size`: 1024
- `train_epochs`: 600
- `final_eval_episodes`: 25
- `subgoal_flow_steps`: 8
- `subgoal_temperature`: 1.0
- `subgoal_goal_representation`: phi
- `subgoal_target_mode`: displacement
- `planner_type`: forward_bridge_residual
- `value_hidden_dims`: [512, 512, 512]

**Eval 파라미터 (best row)**

- `eval_epoch`: 600
- `eval_n`: 2
- `eval_episodes`: 25
- `eval_temperature`: 0.5
- `eval_suffix`: t0p5
- `eval_score_type`: transitive_ratio
- `eval_max_chunks`: 200
- `eval_flow_steps`: 8
- `IDM`: 0.9039999999999999
- `ACTOR`: 0.8160000000000001
- `idm_tasks`: 1:0.9200,2:0.9200,3:0.9600,4:0.9200,5:0.8000
- `actor_tasks`: 1:0.8400,2:0.6800,3:1.0000,4:0.7600,5:0.8000
- `eval_timestamp`: 2026-06-22T03:12:19Z

### cube-triple-play-v0

- run: `/home/choi/Pathbridger_flow/runs/20260624_082614_seed0_cube-triple-play-v0`
- run_group: `flow_trl_gap_tune_ct_g20_w5_n1`

**학습 파라미터**

- `seed`: 0
- `subgoal`: flow
- `critic_type`: trl
- `gap`: 20.0
- `wmax`: 5.0
- `horizon_K`: 25
- `h_a`: 5
- `train_N`: 1
- `train_eval_N`: 1
- `lambda`: 1.0
- `kappa_b`: 0.9
- `kappa_d`: 0.9
- `gamma`: 0.995
- `batch_size`: 4096
- `train_epochs`: 600
- `final_eval_episodes`: 25
- `subgoal_flow_steps`: 8
- `subgoal_temperature`: 1.0
- `subgoal_goal_representation`: phi
- `subgoal_target_mode`: displacement
- `planner_type`: forward_bridge_residual
- `value_hidden_dims`: [512, 512, 512]

**Eval 파라미터 (best row)**

- `eval_epoch`: 600
- `eval_n`: 16
- `eval_episodes`: 25
- `eval_temperature`: 0.25
- `eval_suffix`: t0p25
- `eval_score_type`: transitive_ratio
- `eval_max_chunks`: 200
- `eval_flow_steps`: 8
- `IDM`: 0.664
- `ACTOR`: 0.56
- `idm_tasks`: 1:0.9600,2:1.0000,3:0.9600,4:0.2400,5:0.1600
- `actor_tasks`: 1:1.0000,2:0.7600,3:0.8800,4:0.1200,5:0.0400
- `eval_timestamp`: 2026-06-24T14:39:56Z

### humanoidmaze-large-navigate-v0

- run: `/home/choi/Pathbridger_flow/runs/20260625_192606_seed0_humanoidmaze-large-navigate-v0`
- run_group: `flow_trl_humanoidmaze_dw_hml_g10_w5_n1_dwp0p1`

**학습 파라미터**

- `seed`: 0
- `subgoal`: flow
- `critic_type`: trl
- `gap`: 10.0
- `wmax`: 5.0
- `horizon_K`: 25
- `h_a`: 5
- `train_N`: 1
- `train_eval_N`: 1
- `lambda`: 0.1
- `kappa_b`: 0.9
- `kappa_d`: 0.9
- `gamma`: 0.999
- `batch_size`: 1024
- `train_epochs`: 600
- `final_eval_episodes`: 25
- `subgoal_flow_steps`: 8
- `subgoal_temperature`: 1.0
- `subgoal_goal_representation`: phi
- `subgoal_target_mode`: displacement
- `planner_type`: forward_bridge_residual
- `value_hidden_dims`: [512, 512, 512]

**Eval 파라미터 (best row)**

- `eval_epoch`: 600
- `eval_n`: 32
- `eval_episodes`: 25
- `eval_temperature`: 1.0
- `eval_suffix`: cap2000
- `eval_score_type`: transitive_ratio
- `eval_max_chunks`: 200
- `eval_flow_steps`: 8
- `IDM`: 0.024
- `ACTOR`: 0.016
- `idm_tasks`: 1:0.0000,2:0.0000,3:0.0800,4:0.0000,5:0.0400
- `actor_tasks`: 1:0.0000,2:0.0000,3:0.0400,4:0.0400,5:0.0000
- `eval_timestamp`: 2026-06-26T16:22:48Z

### humanoidmaze-medium-navigate-v0

- run: `/home/choi/Pathbridger_flow/runs/20260627_065306_seed0_humanoidmaze-medium-navigate-v0`
- run_group: `flow_trl_humanoidmaze_dw_hmm_g10_w5_n1_dwp0p1`

**학습 파라미터**

- `seed`: 0
- `subgoal`: flow
- `critic_type`: trl
- `gap`: 10.0
- `wmax`: 5.0
- `horizon_K`: 25
- `h_a`: 5
- `train_N`: 1
- `train_eval_N`: 1
- `lambda`: 0.1
- `kappa_b`: 0.9
- `kappa_d`: 0.9
- `gamma`: 0.999
- `batch_size`: 1024
- `train_epochs`: 600
- `final_eval_episodes`: 25
- `subgoal_flow_steps`: 8
- `subgoal_temperature`: 1.0
- `subgoal_goal_representation`: phi
- `subgoal_target_mode`: displacement
- `planner_type`: forward_bridge_residual
- `value_hidden_dims`: [512, 512, 512]

**Eval 파라미터 (best row)**

- `eval_epoch`: 600
- `eval_n`: 32
- `eval_episodes`: 25
- `eval_temperature`: 1.0
- `eval_suffix`: cap2000
- `eval_score_type`: transitive_ratio
- `eval_max_chunks`: 400
- `eval_flow_steps`: 8
- `IDM`: 0.128
- `ACTOR`: 0.096
- `idm_tasks`: 1:0.0400,2:0.2000,3:0.2400,4:0.0000,5:0.1600
- `actor_tasks`: 1:0.0000,2:0.1200,3:0.2000,4:0.0000,5:0.1600
- `eval_timestamp`: 2026-06-27T06:58:19Z

### puzzle-3x3-play-v0

- run: `/home/choi/Pathbridger_flow/runs/20260623_225006_seed0_puzzle-3x3-play-v0`
- run_group: `flow_trl_gap_tune_p3_g0p5_w5_n1`

**학습 파라미터**

- `seed`: 0
- `subgoal`: flow
- `critic_type`: trl
- `gap`: 0.5
- `wmax`: 5.0
- `horizon_K`: 25
- `h_a`: 5
- `train_N`: 1
- `train_eval_N`: 1
- `lambda`: 0.5
- `kappa_b`: 0.9
- `kappa_d`: 0.9
- `gamma`: 0.99
- `batch_size`: 1024
- `train_epochs`: 600
- `final_eval_episodes`: 25
- `subgoal_flow_steps`: 8
- `subgoal_temperature`: 1.0
- `subgoal_goal_representation`: phi
- `subgoal_target_mode`: displacement
- `planner_type`: forward_bridge_residual
- `value_hidden_dims`: [512, 512, 512]

**Eval 파라미터 (best row)**

- `eval_epoch`: 600
- `eval_n`: 32
- `eval_episodes`: 25
- `eval_temperature`: 0.5
- `eval_suffix`: t0p5
- `eval_score_type`: transitive_ratio
- `eval_max_chunks`: 200
- `eval_flow_steps`: 8
- `IDM`: 0.656
- `ACTOR`: 0.656
- `idm_tasks`: 1:1.0000,2:0.5200,3:0.6400,4:0.5200,5:0.6000
- `actor_tasks`: 1:0.8400,2:0.6800,3:0.6800,4:0.5200,5:0.5600
- `eval_timestamp`: 2026-06-24T14:09:18Z

### puzzle-4x4-play-v0

- run: `/home/choi/Pathbridger_flow/checkpoints/flow_trl_best_epoch600/puzzle_4x4_p4_g5_w5_n1_evalN16`
- run_group: `flow_trl_feval_p4_g5_w5_n1`

**학습 파라미터**

- `seed`: 0
- `subgoal`: flow
- `critic_type`: trl
- `gap`: 5.0
- `wmax`: 5.0
- `horizon_K`: 25
- `h_a`: 5
- `train_N`: 1
- `train_eval_N`: 1
- `lambda`: 2.0
- `kappa_b`: 0.9
- `kappa_d`: 0.9
- `gamma`: 0.99
- `batch_size`: 1024
- `train_epochs`: 600
- `final_eval_episodes`: 25
- `subgoal_flow_steps`: 8
- `subgoal_temperature`: 1.0
- `subgoal_goal_representation`: phi
- `subgoal_target_mode`: displacement
- `planner_type`: forward_bridge_residual
- `value_hidden_dims`: [512, 512, 512]

**Eval 파라미터 (best row)**

- `eval_epoch`: 600
- `eval_n`: 16
- `eval_episodes`: 25
- `eval_temperature`: 0.5
- `eval_suffix`: t0p5
- `eval_score_type`: transitive_ratio
- `eval_max_chunks`: 200
- `eval_flow_steps`: 8
- `IDM`: 0.784
- `ACTOR`: 0.7040000000000001
- `idm_tasks`: 1:0.9200,2:0.6800,3:0.8400,4:0.8000,5:0.6800
- `actor_tasks`: 1:0.8800,2:0.2800,3:0.8000,4:0.8000,5:0.7600
- `eval_timestamp`: 2026-06-22T03:10:50Z

