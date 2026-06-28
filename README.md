# PathBridger Experiment Guide

OGBench 기반 long-horizon goal-conditioned control 실험 코드입니다. 두 가지 학습 경로를 지원합니다.

- **Standard (single-run)**: Flow subgoal + forward-bridge residual dynamics + TRL critic + SPI actor를 한 번에 같이 학습합니다.
- **State-only Offline + Online Hybrid (`hybrid_phase`)**: offline에서 action 없이 dynamics(path) / value V / flow subgoal / deterministic subgoal-spi net을 학습하고, online에서 환경 rollout으로 IDM / action critic Q / SPI actor를 학습합니다. ([State-only Offline + Online Hybrid](#state-only-offline--online-hybrid-hybrid_phase) 참고)

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

## State-only Offline + Online Hybrid (`hybrid_phase`)

`--hybrid_phase {standard,offline,online}` 플래그로 동작을 전환합니다. offline과 online은 **별도 run**으로 실행하고, online이 offline 체크포인트를 불러옵니다.

### 핵심 아이디어

- **Offline (action-free)**: action 정보를 전혀 쓰지 않고 학습합니다.
  - `DynamicsAgent`: path-residual planner + flow subgoal net 학습 (IDM은 `idm_loss_weight=0`으로 비활성).
  - `CriticAgent`: value head V만 학습 (`trl_value_only=True`, action critic Q는 0).
  - `SubgoalSpiAgent`(`agents/subgoal_spi.py`): flow subgoal 후보를 critic의 transitive value `V(s,z)·V(z,g)/V(s,g)`로 점수화해, SPI objective로 **deterministic subgoal net**을 distill합니다 (SPI actor와 동일한 `-V/scale + prox/(2·tau)` 형태).
- **Online**: offline 체크포인트(dynamics, V, flow subgoal, subgoal-spi)를 불러와 frozen으로 두고, 실제 환경 rollout `(s, a, s')`로 다음을 학습합니다.
  - `DynamicsAgent`: IDM만 (path/flow subgoal은 frozen).
  - `CriticAgent`: action critic Q만 (offline V는 frozen, `lambda_v_*=0`).
  - `ActorAgent`: SPI actor.
  - subgoal policy는 deterministic subgoal-spi net을 사용합니다. eval은 subgoal-spi와 flow best-of-N(`--online_eval_flow_n_samples`)을 함께 측정해 비교합니다.

> 기존 600에포크 flow 체크포인트처럼 subgoal-spi net이 없는 경우, online에서 `--online_train_subgoal_spi=True`(기본값)로 frozen flow + frozen V로부터 subgoal-spi를 같이 distill합니다.

### Offline run

```bash
PYTHONPATH=. MUJOCO_GL=egl python main.py \
  --run_config config/hybrid_offline_antmaze_large.yaml \
  --hybrid_phase offline \
  --seed 0
```

### Online run

offline run 디렉토리(또는 위 구조의 flow 체크포인트 디렉토리)를 `--offline_run_dir`로 지정합니다.

```bash
PYTHONPATH=. MUJOCO_GL=egl python main.py \
  --run_config config/hybrid_online_antmaze_large.yaml \
  --hybrid_phase online \
  --offline_run_dir runs/<offline_run_dir> \
  --offline_load_step 600 \
  --seed 0
```

`--offline_run_dir`는 `checkpoints/{dynamics,critic}/params_<step>.pkl` 구조를 가진 디렉토리여야 하며, `subgoal_spi/params_<step>.pkl`이 있으면 함께 로드합니다.

### 주요 hybrid 플래그

| 플래그 | 기본 | 설명 |
|--------|------|------|
| `hybrid_phase` | `standard` | `standard` / `offline` / `online` |
| `offline_run_dir` | `""` | online에서 불러올 offline(또는 flow) 체크포인트 디렉토리 |
| `offline_load_step` | `-1` | 로드 step. `<0`이면 최신 step 자동 탐색 |
| `online_freeze_offline` | `True` | online에서 path/flow subgoal/V를 frozen으로 둠 |
| `online_train_subgoal_spi` | `True` | online에서 subgoal-spi net을 distill (subgoal-spi 체크포인트가 없을 때 필수) |
| `online_eval_flow_n_samples` | `8` | online eval 시 flow best-of-N 비교 (`<=0`이면 비활성) |
| `online_warmup_env_steps` | - | 학습 시작 전 수집할 warmup env step |
| `online_random_env_steps` | - | 초기 랜덤 행동 수집 step |
| `online_env_steps_per_update` | `1` | gradient step당 env step 수 |
| `online_rebuild_every_env_steps` | - | replay 데이터셋 재구축 주기 |
| `online_replay_capacity` | - | replay buffer 용량(step) |
| `online_collect_policy` | `actor` | 수집 정책 (`actor` / `idm`) |
| `online_explore_noise` | - | 수집 시 action gaussian noise scale |

학습/eval 주기는 step 기준 플래그(`train_steps`, `log_every_n_steps`, `save_every_n_steps`, `eval_every_n_steps`)로 제어합니다. 예: `train_steps=300000`, `eval_every_n_steps=100000`.

예시 config: `config/hybrid_offline_antmaze_large.yaml`, `config/hybrid_online_antmaze_large.yaml`.

## 프로젝트 구조

| 경로 | 역할 |
|------|------|
| `main.py` | 학습 entry point. standard 단일 학습 + `hybrid_phase` offline/online 오케스트레이션 |
| `agents/dynamics.py` | flow/gaussian/deterministic subgoal, bridge planner, IDM, proposal builder |
| `agents/critic.py` | DQC/IQL legacy path와 TRL critic (`trl_value_only`로 value-only 학습 지원) |
| `agents/actor.py` | SPI-style action chunk actor |
| `agents/subgoal_spi.py` | flow subgoal을 critic value로 SPI distill하는 deterministic subgoal net |
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
