# DOURI 구현 개요와 아이디어

이 문서는 `README.md`와 현재 코드를 함께 읽고 정리한 프로젝트 설명입니다. 실행법이나 config 표는 README가 이미 잘 담당하고 있으므로, 여기서는 **왜 이런 구조인지**, **각 모듈이 어떤 역할을 하는지**, **구현상 어떤 선택이 들어갔는지**, **앞으로 정리하면 좋을 부분**에 초점을 둡니다.

## 한 줄 요약

DOURI는 OGBench의 long-horizon offline control 문제를 풀기 위해, 최종 goal로 바로 action을 예측하지 않고 다음 세 단계를 함께 학습합니다.

1. 현재 상태와 최종 goal로부터 도달할 만한 **subgoal**을 예측합니다.
2. 현재 상태에서 subgoal까지 이어지는 **linear-SDE bridge trajectory**를 생성합니다.
3. 그 trajectory를 inverse dynamics로 action chunk로 바꾼 뒤, critic이 고른 proposal을 따라가도록 **SPI actor**를 학습합니다.

즉, policy가 먼 goal까지 한 번에 뛰어가도록 강제하기보다, dynamics가 만든 중간 상태 후보와 critic의 가치 판단을 이용해 actor를 안정적으로 끌고 가는 구조입니다.

## 중심 아이디어

### 1. Long-horizon goal을 subgoal 문제로 낮추기

AntMaze, HumanoidMaze, Cube/Puzzle 계열 task는 최종 goal까지의 horizon이 길고 reward가 sparse합니다. 이 프로젝트는 actor가 `pi(a | s, g_final)`을 바로 학습하는 대신, 먼저 dynamics agent가 `s`와 `g_final`을 보고 `g_sub`를 예측하게 합니다.

기본 subgoal estimator는 deterministic point predictor입니다. `diag_gaussian` 모드에서는 `(mu, log_std)`를 예측해 여러 endpoint 후보를 뽑을 수도 있습니다. 이 분포형 subgoal은 uncertainty-aware proposal ablation을 위한 확장으로 보입니다.

### 2. Bridge로 상태 경로를 만들고 IDM으로 action화하기

Subgoal이 정해지면 dynamics는 현재 상태에서 subgoal까지의 상태 trajectory를 만듭니다. 기본 planner는 `exact_residual_chain`입니다.

- exact bridge posterior mean을 base transition으로 사용합니다.
- residual network가 dataset trajectory에 맞는 보정항을 예측합니다.
- `bridge_gamma_inv=0.0`인 hard endpoint bridge가 현재 기본 sweep의 중심입니다.
- `theta_schedule=prefix_progress`는 actor/IDM이 실제로 사용하는 짧은 prefix가 subgoal 방향으로 충분히 진행되도록 설계된 schedule입니다.

상태 trajectory만으로는 환경을 움직일 수 없으므로, dynamics agent 안의 IDM이 `(s_t, s_{t+1}) -> a_t`를 예측해 action chunk proposal을 만듭니다.

### 3. Critic으로 proposal을 고르고 actor를 SPI로 학습하기

Dynamics proposal은 가능한 action chunk 후보일 뿐이므로, critic이 후보를 평가합니다. Critic은 DQC 스타일의 full chunk critic, partial action critic, scalar value를 같이 갖습니다.

Actor는 proposal을 단순 behavior cloning하지 않습니다. Critic score로 proposal distribution을 만들고, actor가 높은 Q를 유지하면서 좋은 proposal 근처로 이동하도록 SPI-style objective를 사용합니다. 코드상 actor loss는 critic Q 항과 proposal proximity 항을 함께 씁니다.

## 전체 학습 흐름

`main.py`가 학습의 단일 entry point입니다. 한 step의 흐름은 다음과 같습니다.

1. `PathHGCDataset`과 `CriticSequenceDataset`에서 같은 start index를 공유해 dynamics batch와 critic batch를 샘플링합니다.
2. Dynamics가 `observations`와 `high_actor_goals`로 subgoal 후보를 예측합니다.
3. Subgoal 후보마다 bridge trajectory를 만들고, IDM이 actor horizon만큼 action chunk 후보를 만듭니다.
4. Dynamics는 path supervision, reverse dynamics, rollout consistency, subgoal loss, IDM loss를 합쳐 업데이트됩니다.
5. Critic은 offline dataset의 action chunk로 업데이트됩니다.
6. 방금 업데이트된 critic이 dynamics proposal을 다시 평가합니다.
7. 후보가 여러 개라면 전체 후보 축에서 global best proposal 하나를 고르고, 그 proposal에 대응하는 subgoal을 actor의 `spi_goals`로 넘깁니다.
8. Actor는 선택된 proposal 및 critic score 기반 SPI objective로 업데이트됩니다.

중요한 점은 proposal 생성은 dynamics update 전에 일어나지만, actor update에 들어가는 proposal score는 critic update 후 다시 계산된다는 것입니다. 그래서 actor는 항상 현재 critic 기준의 proposal ranking을 보게 됩니다.

## 데이터셋 구성

### `PathHGCDataset`

Dynamics 학습용 dataset입니다. 각 sample에는 다음이 포함됩니다.

- `observations`, `next_observations`
- `high_actor_goals`
- `high_actor_targets`
- `trajectory_segment`: `s_t, s_{t+1}, ..., s_{t+K}`

`clip_path_to_goal=True`가 기본입니다. 목표가 `K` step보다 가까우면 endpoint를 `s_{min(t+K, t_g)}`로 잡고, 그 뒤 segment tail은 goal state로 padding합니다. 이 선택은 가까운 goal에서 subgoal과 bridge가 goal을 지나쳐 버리는 overshoot를 줄이고, "도착 후 머무르기"를 학습시키려는 의도입니다.

### `CriticSequenceDataset`

Critic 학습용 dataset입니다. 같은 start state에서 다음 정보를 만듭니다.

- full chunk action: long backup용
- action chunk action: actor horizon과 맞는 partial action critic용
- value goal과 next observation
- horizon별 reward, mask, backup horizon

`clip_chunk_to_goal=True`이면 value goal이 chunk 안에 있을 때 backup horizon을 줄이고 mask를 0으로 둡니다. Goal-conditioned value backup에서 goal 도달을 terminal처럼 처리하는 셈입니다.

## 주요 모듈

### `agents/dynamics.py`

가장 많은 책임을 가진 핵심 파일입니다.

- `ResidualNet`: exact bridge posterior mean 위에 더할 residual 예측
- `SubgoalEstimatorNet`: deterministic subgoal 예측
- `DistributionalSubgoalEstimatorNet`: diagonal Gaussian subgoal 예측
- `PathResidualNet`: `forward_bridge_residual` planner용 endpoint-preserving residual
- bridge planning, subgoal inference, action proposal 생성
- dynamics phase-1 loss 전체
- embedded IDM loss와 inference

현재 기본값은 `exact_residual_chain + deterministic subgoal + prefix_progress + hard bridge`입니다.

### `agents/critic.py`

Goal-conditioned critic stack입니다.

- `BinaryChunkCritic`: full action chunk 또는 partial action chunk 평가
- `ScalarValueNet`: value/subgoal value bonus에 사용
- DQC 모드에서는 chunk critic으로 긴 horizon을 평가하고 partial action critic으로 actor horizon action을 평가합니다.
- IQL 모드도 남아 있으며, 이 경우 chunk critic을 끄고 action critic/value 중심으로 동작합니다.

### `agents/actor.py`

Deterministic chunk actor입니다.

- 입력: observation과 `spi_goals`
- 출력: `actor_chunk_horizon * action_dim` 크기의 action chunk
- loss: critic Q를 높이는 항 + critic score로 weighting된 proposal proximity 항

Actor가 직접 final goal을 보는 것이 아니라 dynamics가 예측한 subgoal을 condition으로 받는다는 점이 이 프로젝트의 coupling 핵심입니다.

### `utils/dynamics.py`, `utils/theta_schedules.py`

Linear-SDE bridge의 수치적 핵심입니다.

- bridge schedule 생성
- posterior mean/variance 계산
- exact residual model mean
- reverse sampling
- `linear_beta`, `prefix_progress` schedule

README에 자세히 적힌 것처럼, `prefix_progress`는 짧은 prefix가 subgoal displacement의 의미 있는 비율을 차지하도록 설계되어 있습니다.

### `rollout/`

Checkpoint를 로드해 subgoal, IDM, actor rollout을 수행하는 시각화/검증 경로입니다. Maze 계열은 state-space plot이 의미 있고, ManipSpace cube/puzzle 계열은 IDM/Actor MP4 중심입니다.

## Config 설계

Config는 YAML에서 필요한 override만 적고, 대부분은 agent default에서 가져오는 방식입니다.

Top-level flag는 run/eval 공통 제어를 담당합니다.

- `horizon`: dynamics `dynamics_N`, `subgoal_steps`, critic `full_chunk_horizon`에 같이 적용됩니다.
- `plan_candidates`: subgoal endpoint마다 만들 bridge/action 후보 수입니다.
- `eval_*`: OGBench task eval episode 수와 최대 chunk 수를 제어합니다.
- `async_prefetch`: host-side batch sampling과 GPU compute를 overlap합니다.

`dynamics:`, `critic_agent:`, `actor:` 아래에는 각 agent별 override를 둡니다. 환경별 baseline YAML은 주로 critic의 discount, kappa, action chunk horizon 차이를 담고, 알고리즘 default는 코드 default에 맡기는 방향입니다.

## 평가와 rollout

학습 중 eval과 `eval_checkpoint.py` 모두 같은 핵심 경로를 사용합니다.

- Actor policy: `infer_subgoal(obs, goal) -> actor.sample_actions(obs, subgoal)`
- IDM policy: `infer_subgoal(obs, goal) -> bridge plan -> IDM action chunk`

Success 판정은 OGBench env의 `info["success"]`를 기준으로 합니다.

## 구현상 좋은 점

- Dynamics와 critic이 같은 start index를 공유해 학습되므로, actor coupling에 쓰이는 상태 분포가 흔들리지 않습니다.
- `async_prefetch`가 single-worker `ThreadPoolExecutor`를 사용해 sampling order를 보존합니다. 재현성과 성능을 같이 챙긴 선택입니다.
- Metric accumulation에서 매 step마다 host sync하지 않고 epoch 말에 모아서 device-to-host transfer를 합니다.
- `PathHGCDataset.validate_sample_batch`처럼 trajectory alignment를 검사할 수 있는 hook이 있어 bridge supervision 관련 버그를 잡기 좋습니다.
- `eval_checkpoint.py`가 `flags.json`을 읽어 config를 복원하므로, 오래된 run도 같은 hyperparameter로 평가하기 쉽습니다.

## 수정 또는 정리 추천

### 1. README의 작은 stale 항목

README는 전체적으로 현재 구현을 잘 설명하지만, 몇 가지 세부값은 코드와 다릅니다.

- `plan_noise_scale` README 기본값은 `0.01`이지만 현재 flag 기본값은 `1.0`입니다.
- `plan_candidates` 설명은 endpoint별 best-of-N처럼 읽히지만, multi-candidate 경로는 전체 후보 축에서 global best 1개를 고릅니다.
- `horizon`은 실제 flag 기본값이 `25`입니다.
- `dataset_dir`, `subgoal_override_goal`, `eval_video_*`, `goal_representation=phi`는 README에 보강하면 좋습니다.

### 2. `agents/dynamics.py` 책임 분리

현재 dynamics 파일은 network, planner, subgoal distribution, IDM, loss, proposal builder를 모두 포함합니다. 실험 속도에는 좋지만 장기 유지보수에는 부담입니다.

분리 후보:

- `dynamics_networks.py`
- `dynamics_planner.py`
- `dynamics_losses.py`
- `dynamics_proposals.py`

다만 한 번에 크게 나누기보다는, 현재 public API인 `infer_subgoal`, `plan`, `build_actor_proposals`, `update`를 유지하면서 내부 helper를 천천히 옮기는 편이 안전합니다.

### 3. Config 디렉터리 정리

`config/`에 baseline과 ablation 파일이 섞여 있습니다. 현재처럼 실험이 빠르게 늘어나는 프로젝트에서는 나중에 어떤 config가 canonical인지 헷갈리기 쉽습니다.

추천 구조:

```text
config/
  baseline/
  ablation/
  archive/
```

또는 파일명 규칙을 README에 명시하는 것도 충분히 효과적입니다.

### 4. Single-candidate score logging

후보가 1개일 때 actor proposal score는 fast path에서 0으로 채워집니다. SPI softmax에는 문제가 없지만, log를 볼 때 실제 critic confidence로 오해하기 쉽습니다.

추천:

- dummy score임을 별도 key로 남기거나,
- 분석 run에서는 single candidate도 critic score를 계산하는 옵션을 둡니다.

### 5. Test runner 추가

README는 테스트 파일을 직접 실행하라고 안내합니다. 지금은 괜찮지만 테스트가 더 늘어나면 `scripts/run_tests.sh`나 `pytest` entry point가 있으면 회귀 확인이 쉬워집니다.

## 정리

이 프로젝트는 단순히 "dynamics model + actor"를 붙인 구조라기보다, **subgoal prediction, bridge-based state proposal, inverse dynamics actionization, critic-ranked SPI update**를 한 학습 루프 안에 묶은 실험 코드입니다.

핵심 실험 질문은 다음으로 요약할 수 있습니다.

> Offline long-horizon goal-conditioned control에서, learned bridge가 만든 중간 상태와 critic-ranked action proposal을 actor 학습의 guide로 쓰면 sparse goal task를 더 안정적으로 풀 수 있는가?

현재 구현은 그 질문을 여러 환경과 ablation에서 빠르게 검증할 수 있도록 구성되어 있습니다. README는 실행과 config reference로 유지하고, 이 문서는 구현 의도와 구조를 파악하는 companion note로 보면 좋습니다.
