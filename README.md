# PathBridger Experiment Guide

OGBench 기반 offline long-horizon goal-conditioned control 실험 코드입니다. 현재 코드스페이스의 주 경로는 **Flow subgoal + forward-bridge residual dynamics + TRL critic + SPI actor**입니다.

자세한 구현 설명은 `docs/project_implementation_overview.md`, script 운영 정리는 `scripts/README.md`를 참고하세요.

## 현재 메인 경로

학습 루프는 다음 구성요소를 함께 업데이트합니다.

1. `DynamicsAgent`가 현재 상태와 final goal에서 flow subgoal endpoint를 샘플링합니다.
2. `forward_bridge_residual` planner가 현재 상태에서 subgoal까지의 state trajectory를 만듭니다.
3. IDM이 trajectory prefix를 action chunk proposal로 변환합니다.
4. TRL critic이 action chunk와 transitive value target을 학습합니다.
5. Actor는 critic-ranked proposal을 따라가도록 SPI objective로 업데이트됩니다.

현재 sweep의 핵심 config는 보통 다음 값을 가집니다.

```yaml
dynamics:
  subgoal_distribution: flow
  subgoal_stochastic_loss: mse
  planner_type: forward_bridge_residual
  subgoal_eval_selection: best_of_n_value

critic_agent:
  algorithm: trl
  critic_type: trl
  use_chunk_critic: false
  q_target_from_value: true
```

## 프로젝트 구조

| 경로 | 역할 |
|------|------|
| `main.py` | 학습 entry point. dynamics, critic, actor 동시 학습 |
| `agents/dynamics.py` | flow/gaussian/deterministic subgoal, bridge planner, IDM, proposal builder |
| `agents/critic.py` | DQC/IQL legacy path와 TRL critic |
| `agents/actor.py` | SPI-style action chunk actor |
| `utils/critic_sequence_dataset.py` | TRL critic batch와 transitive value fields 생성 |
| `utils/dynamics.py` | bridge schedule, posterior/model mean, residual helper |
| `utils/theta_schedules.py` | `linear_beta`, `prefix_progress` schedule |
| `eval_checkpoint.py` | checkpoint에서 eval만 재실행 |
| `rollout/` | checkpoint rollout/시각화 도구 |
| `config/` | 실험 YAML |
| `scripts/` | YAML 생성, sweep 실행, eval summary 도구 |
| `docs/` | 구현 설명과 분석 산출물 |

## 환경 설정

저장소 루트에서 실행합니다.

```bash
cd /home/svcho/Pathbridger_flow
export PYTHONPATH=.
export MUJOCO_GL=egl
```

GPU 학습은 CUDA용 JAX가 필요합니다. `scripts/with_jax_cuda.sh`는 pip/conda 환경의 CUDA shared library path를 잡은 뒤 명령을 실행하는 wrapper입니다.

확인:

```bash
python - <<'PY'
import jax
print(jax.default_backend())
print(jax.devices())
PY
```

## 기본 학습

```bash
PYTHONPATH=. MUJOCO_GL=egl python main.py \
  --run_config config/sweep_flow_trl_finaleval/p4_g5_w5_n1.yaml \
  --seed 0 \
  --async_prefetch
```

Resume:

```bash
PYTHONPATH=. MUJOCO_GL=egl python main.py \
  --run_config config/sweep_flow_trl_finaleval/p4_g5_w5_n1.yaml \
  --resume_run_dir runs/<run_dir> \
  --resume_epoch 200
```

Run directory:

```text
runs/<YYYYMMDD_HHMMSS>_seed<seed>_<env_name>/
  config_used.yaml
  flags.json
  train.csv
  run.log
  run_resume_from<E>_<timestamp>.log
  checkpoints/
    dynamics/params_<epoch>.pkl
    critic/params_<epoch>.pkl
    actor/params_<epoch>.pkl
  eval_results/
    epoch600_n<N>.json
    epoch600_t0p5_n<N>.json
```

## Config 핵심 옵션

Top-level:

| 키 | 기본/관례 | 설명 |
|----|-----------|------|
| `env_name` | required | OGBench env |
| `run_group` | required | sweep/run 식별자 |
| `train_epochs` | `600` | 학습 epoch |
| `batch_size` | `1024` | shared batch size |
| `horizon` | `25` 또는 `40` | `dynamics_N`, `subgoal_steps`, critic `full_chunk_horizon`에 동기화 |
| `plan_candidates` | `1` | bridge/action proposal candidate 수 |
| `eval_freq` | `100` | in-training eval 주기 |
| `eval_task_ids` | `1,2,3,4,5` | OGBench eval task ids |
| `eval_episodes_per_task` | `10` | 일반 eval episode 수 |
| `final_eval_episodes_per_task` | `25` | final epoch eval episode 수 |
| `final_eval_subgoal_eval_num_samples` | `1,2,4,8,16` | final epoch N sweep |

Flow subgoal:

| 키 | 설명 |
|----|------|
| `subgoal_flow_steps` | flow endpoint Euler step 수. 현재 sweep은 8 |
| `subgoal_flow_t_min` | flow time lower bound |
| `subgoal_flow_noise_scale` | flow sampling noise scale |
| `subgoal_temperature` | eval-time subgoal sampling temperature |
| `subgoal_eval_num_samples` | eval-time best-of-N 후보 수 |
| `subgoal_eval_selection` | 현재 주 경로는 `best_of_n_value` |
| `subgoal_value_gap_scale` | value-gap weighting scale |
| `subgoal_value_weight_max` | flow matching weight cap |

Subgoal SPI:

`subgoal_spi_enabled: true`는 기존 flow/gaussian subgoal을 대체하지 않습니다. 기존 `subgoal_net`은 그대로 proposal/teacher 역할을 하고, 별도의 deterministic `subgoal_spi_net`을 추가로 학습합니다.

권장 config:

```yaml
dynamics:
  subgoal_distribution: flow        # proposal/teacher subgoal net
  subgoal_spi_enabled: true         # add deterministic subgoal_spi_net
  subgoal_spi_num_samples: 16       # proposal candidates for SPI update
  subgoal_spi_beta: 1.0             # Boltzmann weight temperature inverse
  subgoal_spi_tau: 5.0              # proximal strength denominator
  subgoal_spi_energy_norm_eps: 1.0e-6
```

학습 objective는 flow `subgoal_net`에서 N개 후보 `z_i`를 샘플링하고, critic product energy로 Boltzmann target을 만듭니다.

```text
E(s,z,g) = V(s,z) * V(z,g)
rho_i = softmax(beta * E(s,z_i,g))
L_spi = -E(s,z_theta,g) / scale
        + sum_i rho_i * ||z_theta - stopgrad(z_i)||^2 / (2 * tau)
scale = stopgrad(mean(abs(E(s,z_theta,g))) + eps)
```

여기서 `z_theta`는 deterministic `subgoal_spi_net(s,g)` 출력입니다. `scale`은 energy 항의 수치 크기 안정화용이고, `eps`는 `subgoal_spi_energy_norm_eps`입니다. flow proposal net은 기존 flow matching/value-guided loss로 계속 학습됩니다.

eval에서 `subgoal_spi_enabled: true`이면 네 조합을 함께 기록합니다.

| metric prefix | subgoal source | low-level policy |
|---------------|----------------|------------------|
| `eval_flow_idm` | flow `subgoal_net` best-of-N | IDM |
| `eval_flow_actor` | flow `subgoal_net` best-of-N | actor |
| `eval_spi_subgoal_idm` | deterministic `subgoal_spi_net` 1회 forward | IDM |
| `eval_spi_subgoal_actor` | deterministic `subgoal_spi_net` 1회 forward | actor |

기존 flow-only 학습을 하려면 `subgoal_spi_enabled`를 생략하거나 `false`로 둡니다. 이 경우 `subgoal_spi_net`은 생성되지 않고, 기존 `subgoal_distribution: flow` 경로만 사용합니다.

TRL critic:

| 키 | 설명 |
|----|------|
| `action_chunk_horizon` | actor/IDM chunk horizon. 현재 sweep은 5 |
| `full_chunk_horizon` | critic full horizon. top-level `horizon`과 동기화 |
| `discount` | env별 gamma |
| `tau_v` | expectile/value 관련 coefficient |
| `lambda_v_self`, `lambda_v_base`, `lambda_v_tri` | TRL value loss weights |
| `value_base_horizon` | base value target horizon |
| `value_distance_weight_power` | value distance reweight |
| `subgoal_value_bonus_type` | 보통 `transitive_product` |

Goal sampling notation:

| 표기 | config field |
|------|--------------|
| `cur` | `value_p_curgoal` 또는 `actor_p_curgoal` |
| `geom` | `value_geom_sample` 또는 `actor_geom_sample` |
| `traj` | `value_p_trajgoal` 또는 `actor_p_trajgoal` |
| `rand` | `value_p_randomgoal` 또는 `actor_p_randomgoal` |

최근 puzzle 4x5/4x6 sweep은 다음 설정을 사용합니다.

- policy `(cur, geom, traj, rand) = (0, 0.5, 0, 0.5)`
- value TRL `(cur, geom, traj, rand) = (0, 0, 1, 0)`

## 주요 Sweep

### Flow+TRL final-eval sweep

```bash
python scripts/write_flow_trl_sweep_yaml.py
GPU_ID=0 nohup bash scripts/run_flow_trl_sweep.sh > nohup_logs/flow_trl_sweep.nohup.log 2>&1 &
```

- configs: `config/sweep_flow_trl_finaleval/`
- gap: `{1, 3, 5, 10}`
- wmax: `5`
- train N: `1`
- final eval N: `{1, 2, 4, 8, 16}`

### Puzzle 4x5 / 4x6

```bash
python scripts/write_flow_trl_puzzle_45_46_yaml.py
GPU_ID=0 nohup bash scripts/run_flow_trl_puzzle_45_46.sh > nohup_logs/flow_trl_p456.nohup.log 2>&1 &
```

- configs: `config/sweep_flow_trl_puzzle_45_46/`
- run group: `flow_trl_p456g999_*`
- envs: `puzzle-4x5-play-v0`, `puzzle-4x6-play-v0`
- gamma: `0.999`
- `value_distance_weight_power: 0.0`
- eval budget: 환경 max episode length

### Puzzle 3x3 / 4x4 Subgoal SPI

```bash
GPU_ID=0 nohup bash scripts/run_flow_trl_spi_p33_p44_500k.sh > nohup_logs/flow_trl_spi_p33_p44_500k.nohup.log 2>&1 &
```

- configs: `config/sweep_flow_trl_spi_p33_p44_500k/`
- subgoal: `subgoal_distribution: flow` + `subgoal_spi_enabled: true`
- deterministic SPI net: `subgoal_spi_net`
- proposal/teacher: flow `subgoal_net`
- train steps: `250000`, `500000`
- eval N: `{2, 16}`
- eval temperature: `{1.0, 0.5}`
- 250K eval uses `eval_episodes_per_task`; 500K eval uses `final_eval_episodes_per_task`

### K=40 best follow-up

```bash
python scripts/write_flow_trl_k40_best_yaml.py
GPU_ID=0 nohup bash scripts/run_flow_trl_k40_best.sh > nohup_logs/flow_trl_k40_best.nohup.log 2>&1 &
```

- configs: `config/sweep_flow_trl_k40_best/`
- horizon/full chunk: `40`
- selected params from prior K=25 best settings

### Antmaze-giant env-max eval

```bash
GPU_ID=0 nohup bash scripts/run_amg_m800_eval.sh > nohup_logs/amg_envmax_eval.nohup.log 2>&1 &
```

- temp: `1.0`, `0.5`
- eval N: `{2, 8, 16, 32}`

Chain giant eval then puzzle sweep:

```bash
GPU_ID=0 nohup bash scripts/run_amg_m800_then_p456.sh > nohup_logs/amg_envmax_then_p456.nohup.log 2>&1 &
```

## Eval

Checkpoint eval:

```bash
PYTHONPATH=. MUJOCO_GL=egl python eval_checkpoint.py \
  --run_dir runs/<run_dir> \
  --epoch 600 \
  --eval_task_ids "1,2,3,4,5" \
  --eval_episodes_per_task 25 \
  --subgoal_eval_num_samples 16 \
  --skip_if_saved
```

Temp / N override:

```bash
PYTHONPATH=. MUJOCO_GL=egl python eval_checkpoint.py \
  --run_dir runs/<run_dir> \
  --epoch 600 \
  --eval_episodes_per_task 25 \
  --subgoal_temperature 0.5 \
  --subgoal_eval_num_samples 32 \
  --skip_if_saved
```

## Rollout

통합 rollout:

```bash
PYTHONPATH=. MUJOCO_GL=egl python -m rollout.run \
  --run_dir runs/<run_dir> \
  --checkpoint_epoch 600 \
  --task_ids 1,2,3,4,5 \
  --mode all
```

직접 호출 가능한 entry point:

```bash
python -m rollout.subgoal --run_dir runs/<run_dir> --checkpoint_epoch 600
python -m rollout.idm --run_dir runs/<run_dir> --checkpoint_epoch 600
python -m rollout.actor --run_dir runs/<run_dir> --checkpoint_epoch 600
python -m rollout.manip_play_rollouts --run_dir runs/<run_dir> --checkpoint_epoch 600
python -m rollout.manip_play_state_rollout --run_dir runs/<run_dir> --checkpoint_epoch 600
```

Maze 계열은 state-space plot이 의미 있고, cube/puzzle ManipSpace 계열은 IDM/Actor MP4 중심입니다.

## 결과 요약

Generated docs CSV/MD는 `_7ch` suffix를 붙입니다. helper는 `scripts/docs_output_paths.py`입니다.

```bash
python scripts/summarize_feval_results.py
python scripts/summarize_runs.py
```

대표 출력:

- `docs/flow_trl_feval_results_7ch.csv`
- `docs/flow_trl_feval_results_7ch.md`
- `docs/runs_results_total_7ch.csv`
- `docs/douri_runs_results_total_7ch.csv`

`_7ch`가 없는 generated CSV/MD는 만들지 않는 규칙입니다. 수동 작성 문서와 figures는 별도입니다.

자세한 script별 역할은 `scripts/README.md`를 참고하세요. 예전 tune/plain-flow ablation runner와 완료된 일회성 eval helper는 `scripts/`에서 제거했습니다.

## 테스트

핵심 회귀 테스트:

```bash
PYTHONPATH=. python tests/test_exact_residual_dynamics.py
PYTHONPATH=. python tests/test_distributional_subgoal.py
PYTHONPATH=. python tests/test_forward_bridge_planner.py
PYTHONPATH=. python -m pytest tests/test_critic_modes.py -v
PYTHONPATH=. python tests/test_prefix_progress_schedule.py
```

Dynamics/planner/schedule을 바꾸면 forward bridge와 prefix progress 테스트를, critic을 바꾸면 `tests/test_critic_modes.py`를 확인하세요.

## 운영 주의사항

- 저장소 루트에서 실행하고 `PYTHONPATH=.`을 설정합니다.
- headless eval/rollout은 `MUJOCO_GL=egl`이 안정적입니다.
- 긴 sweep은 `nohup`으로 실행하고, 중복 process를 확인한 뒤 시작합니다.
- Runner들은 기존 run, gamma match, eval JSON completeness를 기준으로 skip/retrain 여부를 결정합니다.
- `flags.json`에는 `dynamics`, `critic_agent`, `actor` block이 모두 저장됩니다. TRL value sampling은 `critic_agent` block을 기준으로 봅니다.
- Generated CSV/MD, zip, ad-hoc 결과표는 로컬 분석 산출물입니다. 커밋할 때는 의도적으로 stage했는지 확인하세요.
