# DOURI Dynamics Experiment Guide

OGBench 기반 오프라인 제어 실험 코드입니다. 메인 경로는 **linear-SDE dynamics + critic + SPI actor**의 동시 학습입니다.

- Dynamics 부분은 GOUB[^goub] 계열의 reverse mean matching을 따와서 시작했지만, 현재는 단일 exact linear-SDE bridge로 정리한 자체 구현입니다. `bridge_gamma_inv: 0.0`이면 hard endpoint bridge이며, 현재 sweep은 이 설정 위에서 deterministic point subgoal을 학습합니다.
- Critic 부분은 DQC[^dqc]의 chunk + action critic 구조를 가져와 SPI actor에 맞게 재구성한 것입니다.

[^goub]: Generalized Ornstein-Uhlenbeck Bridge.
[^dqc]: Decoupled Q Chunking.

## 프로젝트 구조

작업 디렉터리는 항상 저장소 루트(`douri`)이며, 실행 전 `PYTHONPATH=.`을 설정합니다.

| 경로 | 역할 |
|------|------|
| `main.py` | 학습 엔트리포인트 (dynamics + critic + actor 동시 학습) |
| `agents/dynamics.py` | linear-SDE dynamics agent (subgoal/IDM/path/rollout 손실, planner 분기) |
| `utils/dynamics.py` | bridge 스케줄, sampling, posterior/model mean, exact-residual 헬퍼 |
| `utils/theta_schedules.py` | linear-SDE θ 스케줄 (`linear_beta`, `prefix_progress`) |
| `agents/critic.py` | chunk + action critic |
| `agents/actor.py` | SPI actor |
| `eval_checkpoint.py` | 체크포인트에서 환경 평가만 재실행 |
| `rollout/` | checkpoint 기반 `subgoal`, `idm`, `actor` rollout/시각화 |
| `tests/` | 핵심 수치/분기 회귀 테스트 (`test_*.py`) |
| `config/` | 실험 YAML (legacy는 `config/legacy/`) |
| `scripts/` | sweep / eval summary / heatmap 스크립트 (legacy는 `scripts/legacy/`) |

## 설정

YAML은 agent default 위에 필요한 override만 적습니다. 기본값은 다음에서 옵니다.

- `agents/dynamics.get_dynamics_config()`
- `agents/critic.get_config()`
- `agents/actor.get_actor_config()`

### Top-level 옵션 (run/eval 공통)

| 키 | 기본 | 설명 |
|----|------|------|
| `env_name`, `seed`, `train_epochs` | — | OGBench 환경, 시드, 학습 epoch 수 |
| `batch_size` | `1024` | 학습 배치 크기 |
| `horizon` | — | dataset segment 길이 (`dynamics_N` / `subgoal_steps`와 정합) |
| `plan_candidates` | `1` | 후보 plan trajectory 수 (>1이면 critic이 rescoring) |
| `plan_noise_scale` | `0.01` | plan sampling 시 추가 노이즈 표준편차 |
| `eval_freq` | `100` | env 평가 주기(epoch) |
| `eval_task_ids` | `"1,2,3,4,5"` | OGBench task id 목록 |
| `eval_episodes_per_task` | `10` | task 당 epoch 평가 episode 수 |
| `final_eval_episodes_per_task` | `0` | `>0`이면 **마지막 epoch**에서만 episode 수를 이 값으로 override (보통 50) |
| `eval_max_chunks` | `200` | episode 당 최대 action chunk 수 |
| `eval_goal_tol`, `eval_goal_dims` | `0.5`, `"0,1"` | 평가용 goal 도달 판정 |
| `async_prefetch` | `true` | host-side 배치 샘플링을 GPU 학습과 오버랩 |

### `dynamics:` 핵심 옵션

YAML의 `dynamics:` 키 아래에 둡니다. 모든 옵션은 `get_dynamics_config()`에 default가 잡혀 있어 명시 안 해도 됩니다.

```yaml
dynamics:
  bridge_gamma_inv: 0.0
  subgoal_distribution: deterministic
  subgoal_value_alpha: 0.3
```

#### Bridge / 스케줄

| 키 | 기본 | 설명 |
|----|------|------|
| `dynamics_N` | `25` | linear-SDE 단계 수 (= subgoal까지의 forward step 수) |
| `dynamics_beta_min`, `dynamics_beta_max` | `0.1`, `20.0` | `linear_beta` 스케줄의 β 범위 |
| `dynamics_lambda` | `1.0` | OU stationary scale λ |
| `bridge_gamma_inv` | `0.0` | bridge denominator offset. `0.0`이면 hard endpoint bridge |
| `theta_schedule` | `linear_beta` | `linear_beta` (legacy) 또는 `prefix_progress` (아래 §Prefix-progress 스케줄 참조) |
| `theta_total` | `1.0` | `prefix_progress` 모드의 누적 rate $\Theta_K$ |
| `progress_alpha` | `0.8` | `prefix_progress` 모드의 진행 곡률, $c_i = (i/K)^\alpha$ |

#### Subgoal 손실

| 키 | 기본 | 설명 |
|----|------|------|
| `subgoal_distribution` | `deterministic` | hard-bridge sweep은 deterministic point subgoal. 분포 학습 ablation에서만 `gaussian` |
| `subgoal_loss_weight` | `1.0` | subgoal MSE/NLL 가중치 |
| `subgoal_value_alpha` | `0.1` | subgoal loss의 $V(\hat s_g, g)$ critic value bonus 계수. `0`이면 비활성화 |
| `subgoal_steps` | `25` | subgoal 추정에 사용할 future horizon |
| `clip_path_to_goal` | `true` | 가까운 goal에서 endpoint를 실제 goal로 clip/pad해 "도착 후 머물기"를 학습 |

#### Planner / model 모드

| 키 | 기본 | 설명 |
|----|------|------|
| `planner_type` | `reverse_score` | reverse mean matching 기반 학습 chain (legacy). 대안: `forward_bridge` (closed-form forward bridge mean, 학습 path 파라미터 없음), `forward_bridge_residual` (forward bridge mean + endpoint-preserving 학습 residual `PathResidualNet`) |
| `forward_bridge_mode` | `mean` | `forward_bridge*` 모드의 inference (`mean` / `sample`) |
| `forward_bridge_use_path_loss` | `true` | path-step 손실 활성화 |
| `dynamics_model_type` | `sde_euler` | reverse step 파라미터화. `sde_euler`는 학습된 model mean. `exact_residual`은 model mean을 `posterior_mean + sqrt(post_var) * eps_pred`로 재정의 (data residual 학습) |
| `exact_residual_scale` | `1.0` | `exact_residual` 모드의 residual 스케일 |
| `exact_residual_reg_weight` | `1e-4` | residual L2 정규화 |

#### Curved Centerline Bridge (옵션, ablation)

State-space subgoal proposal을 **endpoint-preserving 학습 곡선** $c_i = (1-\beta_i) s_0 + \beta_i s_K + \beta_i (1-\beta_i)\,h_\psi(s_0, s_K, g, i)$ 주위의 residual 좌표에서 풀게 만드는 옵션입니다. `dynamics_model_type=exact_residual` + `planner_type=reverse_score` 조합에서만 활성화되며, 그 외 조합에선 경고와 함께 무시됩니다.

| 키 | 기본 | 설명 |
|----|------|------|
| `use_curved_centerline` | `false` | true로 켜면 curved centerline bridge 활성화 |
| `centerline_hidden_dims` | `(256, 256)` | `CurvedCenterlineNet` MLP 구조 |
| `centerline_scale` | `1.0` | 학습 displacement $h$의 출력 스케일 |
| `centerline_zero_init` | `true` | true면 마지막 layer를 0 초기화 → 학습 초기에는 직선 bridge와 동치 |
| `centerline_beta_type` | `linear` | β 스케줄 (`linear` 또는 `hard_bridge`) |
| `centerline_use_goal` | `true` | $h_\psi$에 goal $g$ 추가 conditioning |
| `centerline_amp_coef` | `1e-4` | centerline 변위 amplitude 정규화 |
| `centerline_smooth_coef` | `1e-3` | centerline 시간축 smoothness 정규화 |
| `centerline_residual_use_hard_variance` | `true` | residual 좌표의 reverse 분산을 hard bridge 분산으로 고정 |
| `centerline_apply_to_state_dims` | `None` | None이면 모든 state dim에 적용, list[int]면 해당 dim에만 |

#### Prefix-progress θ 스케줄 (옵션)

기본 `linear_beta` 스케줄은 K=25 bridge의 초기 prefix를 매우 천천히 움직입니다. 그러나 actor / IDM / SPI 재평가는 보통 첫 `rollout_horizon=5` proposal state만 소비하므로, 짧은 prefix가 subgoal 변위의 의미 있는 비율을 차지하도록 보정할 필요가 있습니다.

`theta_schedule: prefix_progress`는 desired progress curve $c_i = (i/K)^\alpha$를 두고

$$\Theta_i = \mathrm{asinh}(c_i\, \sinh\Theta_K), \quad \theta_i = \Theta_{i+1} - \Theta_i$$

로 역산해 hard bridge marginal interpolation $\beta_i = \sinh(\Theta_i)/\sinh(\Theta_K)$가 정확히 $c_i$가 되게 합니다. 예: `K=25, alpha=0.8`이면 $c_5 \approx 0.276$ (5스텝이면 약 28% 진행).

- `linear_beta` 디폴트 동작은 bit-for-bit 보존되며, 기존 config 의미는 변하지 않습니다.
- `prefix_progress`로 학습한 run은 `dynamics/theta_schedule_id`, `dynamics/prefix_progress_target_5`, `dynamics/prefix_progress_actual_5` 등 진단 로그가 추가로 찍힙니다.
- 예시 config: `config/antmaze_large_navigate_prefix_progress.yaml`.

### `critic_agent:` / `actor:` override

대부분 그대로 두고 sweep 시 다음 정도만 자주 건드립니다.

```yaml
critic_agent:
  action_chunk_horizon: 10   # default 10. 짧게 잡으면 critic가 더 잦게 rescoring
  discount: 0.99             # giant 환경은 보통 0.995

actor:
  spi_tau: 10.0              # 작을수록 후보 chunk 쪽으로 더 강하게 당김
  spi_beta: 1.0
  spi_actor_layer_norm: true
```

## 학습 실행

```bash
cd /path/to/douri
export PYTHONPATH=.
python main.py --run_config=config/antmaze_large_navigate.yaml
```

Resume:

```bash
python main.py \
  --run_config=config/antmaze_large_navigate.yaml \
  --resume_run_dir=runs/<run_dir> \
  --resume_epoch=200
```

Resume 로그는 `run_resume_from<E>_<timestamp>.log`로 따로 저장됩니다. `flags.json`이 있으면 hyperparameter는 자동으로 그 스냅샷에서 불러옵니다 (legacy `joint_horizon` 같은 키는 새 키 `horizon`으로 자동 매핑). Resume 시 새 config로 override하면 epoch별 평가 episode 수 같은 값을 갈아끼울 수 있습니다 (예: `eval_episodes_per_task=10`, `final_eval_episodes_per_task=50`).

## 현재 Config 레이아웃

`config/sweep_dynamics_tau_alpha/`와 `scripts/launch_dynamics_tau_alpha_sweep.sh`로 돌리던 sweep는 정리됐고, 현재는 feature별 평탄한 config로 운영합니다.

- `antmaze_large_navigate.yaml` / `antmaze_medium_navigate.yaml` — 베이스 (기본 schedule, deterministic subgoal)
- `antmaze_large_navigate_exact_residual.yaml`, `antmaze_large_navigate_exact_residual_tau10p0_alpha0p3.yaml` — `dynamics_model_type=exact_residual` ablation
- `antmaze_large_navigate_prefix_progress.yaml` — `theta_schedule=prefix_progress` ablation
- `antmaze_giant_navigate_exact_residual*.yaml` — giant 환경 + `subgoal_value_alpha ∈ {0.01, 0.5, 0.8}` 변종, `final_eval_episodes_per_task=50` 적용

> 과거 sweep launcher (`scripts/launch_dynamics_tau_alpha_sweep*.sh`)는 삭제된 `config/sweep_dynamics_tau_alpha/` 디렉터리를 참조하므로 더 이상 동작하지 않습니다. 새 sweep을 짤 때는 위 평탄한 config들 중 하나를 베이스로 복제해서 쓰세요.

## Run Directory

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
```

> 과거 학습된 디렉토리들은 `..._joint_dqc_seed<seed>_<env_name>/` prefix를 그대로 사용합니다 (data 호환). eval / heatmap glob은 두 prefix를 모두 매치합니다.

## 학습 구성

Dynamics agent는 다음 손실을 함께 학습합니다.

- `phase1/loss_dynamics`: reverse mean matching (또는 `forward_bridge*` 모드의 forward bridge mean matching)
- `phase1/loss_path_step`: dataset segment와 step-aligned path loss
- `phase1/loss_roll`: short rollout consistency
- `phase1/loss_subgoal`: deterministic subgoal MSE와 critic value bonus (또는 distributional NLL+MSE)
- `phase1/loss_idm`: embedded inverse dynamics MSE
- `dynamics/centerline/{amp,smooth}`: curved centerline 활성 시에만 추가되는 정규화 항

Critic + SPI actor:

- Critic은 후보 action chunk를 평가합니다 (chunk + action heads).
- SPI actor는 critic score로 만든 soft target distribution에 대해 W2-style proximal loss를 씁니다.
- `spi_tau`가 작을수록 후보 chunk 쪽으로 더 강하게 당깁니다.
- Actor와 critic SPI 경로는 항상 dynamics가 예측한 subgoal(`spi_goals`)에 condition됩니다.

주요 dynamics 로그:

- `dynamics/bridge_gamma_inv`, `dynamics/gamma_inv`
- `dynamics/theta_schedule_id`, `dynamics/theta_total`, `dynamics/progress_alpha`
- `dynamics/prefix_progress_actual_5`, `dynamics/prefix_progress_target_5` (target은 `prefix_progress` 모드에서만)
- `phase1/mu_true_norm`, `phase1/mu_pred_norm`, `phase1/bridge_step_mean`

## 평가

Checkpoint eval:

```bash
PYTHONPATH=. MUJOCO_GL=egl python eval_checkpoint.py \
  --run_dir=runs/<run_dir> \
  --epoch=300 \
  --eval_task_ids="1,2,3,4,5" \
  --eval_episodes_per_task=10
```

Rollout 3종:

```bash
# state-space subgoal rollout
PYTHONPATH=. MUJOCO_GL=egl python rollout/subgoal.py \
  --run_dir=runs/<run_dir> \
  --checkpoint_epoch=300 \
  --task_id=1 \
  --max_steps=1000 \
  --out_path=runs/<run_dir>/rollouts/task1_subgoal.png

# IDM real-env rollout
PYTHONPATH=. MUJOCO_GL=egl python rollout/idm.py \
  --run_dir=runs/<run_dir> \
  --checkpoint_epoch=300 \
  --task_id=1 \
  --max_steps=1000 \
  --action_chunk_horizon=5 \
  --out_mp4=runs/<run_dir>/rollouts/task1_idm.mp4

# Actor real-env rollout
PYTHONPATH=. MUJOCO_GL=egl python rollout/actor.py \
  --run_dir=runs/<run_dir> \
  --checkpoint_epoch=300 \
  --task_id=1 \
  --max_chunks=1000 \
  --out_mp4=runs/<run_dir>/rollouts/task1_actor.mp4
```

## 테스트

핵심 회귀 테스트는 `tests/`에 있습니다. JAX가 깔린 환경(`offrl` conda 등)에서 그냥 파일을 직접 실행합니다.

```bash
cd /path/to/douri
PYTHONPATH=. python tests/test_exact_residual_dynamics.py
PYTHONPATH=. python tests/test_distributional_subgoal.py
PYTHONPATH=. python tests/test_forward_bridge_planner.py
PYTHONPATH=. python tests/test_curved_centerline_bridge.py
PYTHONPATH=. python tests/test_prefix_progress_schedule.py
```

각 테스트는 끝에 `OK: ...` 또는 `All tests passed.`를 찍어 통과 여부를 알려줍니다. dynamics 관련 변경(특히 schedule, planner, model_type, curved centerline)을 한 뒤에는 이 5개를 한 번에 돌려 회귀를 확인하세요.

## 주의사항

- 저장소 루트에서 `PYTHONPATH=.` 없이 실행하면 import가 깨질 수 있습니다.
- `MUJOCO_GL=egl`을 설정하면 headless rollout/영상 생성이 안정적입니다.
- Hard bridge sweep에서는 `bridge_gamma_inv: 0.0`과 `subgoal_distribution: deterministic`을 같이 둡니다.
- `use_curved_centerline=true`는 `dynamics_model_type=exact_residual` + `planner_type=reverse_score`에서만 동작합니다. 다른 조합에선 경고와 함께 무시됩니다.
- `theta_schedule=prefix_progress`로 학습한 체크포인트는 평가/롤아웃 시에도 같은 schedule 인자가 자동으로 `flags.json`에서 복원됩니다. 명시적으로 override할 일이 거의 없습니다.
- 옛 prefix(`*_joint_dqc_*`) 디렉토리들은 데이터 호환을 위해 변경하지 않았습니다. 새 학습 run은 `<ts>_seed<N>_<env>` prefix를 사용합니다.
