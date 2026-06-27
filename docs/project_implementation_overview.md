# Pathbridger 구현 개요

이 문서는 현재 코드스페이스 기준으로 `main.py`, `agents/`, `utils/`, `scripts/`, `config/`를 함께 읽어 정리한 구현 설명입니다. 실행 명령이나 개별 실험 결과 표보다는, 현재 branch가 어떤 학습 구조를 구현하고 있고 파일들이 어떤 책임을 갖는지에 초점을 둡니다.

## 한 줄 요약

Pathbridger는 OGBench의 long-horizon offline goal-conditioned control을 위해, 최종 goal로 바로 action을 예측하지 않고 다음 흐름을 한 학습 루프에 묶습니다.

1. 현재 상태와 최종 goal에서 도달할 중간 endpoint, 즉 **subgoal**을 예측합니다.
2. `forward_bridge_residual` planner가 현재 상태에서 subgoal까지의 상태 trajectory를 만듭니다.
3. IDM이 상태 trajectory prefix를 action chunk로 바꿉니다.
4. TRL critic이 subgoal/value 및 action chunk를 평가합니다.
5. actor는 critic-ranked proposal 근처로 이동하면서 높은 Q를 얻도록 SPI-style objective로 학습됩니다.

현재 실험의 중심은 `subgoal_distribution: flow`와 `critic_type: trl`의 조합입니다. 예전 deterministic 또는 diagonal Gaussian subgoal 코드는 남아 있지만, 최신 sweep은 rectified-flow subgoal과 TRL value signal을 주로 사용합니다.

## 핵심 아이디어

### 1. Flow subgoal

`agents/dynamics.py`는 세 가지 subgoal mode를 지원합니다.

- `deterministic`: 하나의 endpoint를 직접 예측합니다.
- `diag_gaussian`: diagonal Gaussian endpoint distribution을 예측합니다.
- `flow`: rectified-flow velocity field를 학습하고 Euler sampling으로 subgoal endpoint를 생성합니다.

현재 Flow+TRL sweep의 기본값은 `flow`입니다. `scripts/flow_trl_sweep_common.py`의 `FLOW_DYNAMICS_BASE`가 이 실험군의 공통 설정을 정의합니다.

- `subgoal_flow_steps: 8`
- `subgoal_flow_t_min: 1e-4`
- `subgoal_flow_noise_scale: 1.0`
- `subgoal_eval_selection: best_of_n_value`
- `subgoal_eval_include_zero_candidate: false`

학습 중 flow loss는 단순 matching loss가 아니라 critic value gap으로 가중됩니다. `subgoal_value_gap_scale`, `subgoal_value_weight_max`, `subgoal_value_bonus_type: transitive_product`가 이 부분을 제어합니다.

### 2. Forward Bridge Residual

현재 planner는 `forward_bridge_residual`로 고정되어 있습니다. `agents/dynamics.py`의 `_VALID_PLANNER_TYPES`도 이 값을 기준으로 정리되어 있습니다.

Subgoal이 정해지면 bridge trajectory를 만들고, residual network가 dataset trajectory에 맞게 path를 보정합니다. `theta_schedule: prefix_progress`와 `progress_alpha`는 actor/IDM이 실제로 사용하는 짧은 prefix가 subgoal 방향으로 충분히 진행되도록 설계된 schedule입니다.

Bridge가 만든 상태 trajectory는 그대로 환경에 넣을 수 없으므로, dynamics agent 안의 IDM이 `(s_t, s_{t+1}) -> a_t`를 예측해 action chunk proposal로 바꿉니다.

### 3. TRL Critic

`agents/critic.py`는 TRL critic mode를 지원하고, TRL 여부는 `_is_trl_type()`으로 판별됩니다. TRL 학습용 batch는 `utils/critic_sequence_dataset.py`의 `CriticSequenceDataset`에서 생성됩니다.

TRL path에서는 `sample_trl_goals()`가 strictly future same-trajectory goal을 뽑고, `_sample_trl_fields()`가 다음 필드를 추가합니다.

- `value_base_goals`, `value_base_offsets`
- `trans_v_split_observations`
- `trans_v_left_goals`
- `trans_v_right_observations`, `trans_v_right_goals`
- `trans_v_valid_mask`, `trans_v_split_offsets`
- `q_goals`, `q_goal_offsets`

이 필드들은 transitive value target과 local Q target을 동시에 구성하는 데 쓰입니다. Flow subgoal의 value bonus도 TRL critic value를 사용합니다.

### 4. Critic-ranked SPI Actor

`agents/actor.py`는 proposal chunk와 proposal score를 외부에서 받아 actor update를 수행합니다. Actor는 final goal을 직접 따라가기보다, dynamics가 만든 subgoal/proposal을 condition으로 받습니다.

훈련 루프에서는 dynamics proposal을 만들고, critic update 이후 현재 critic으로 proposal을 다시 scoring합니다. 후보가 여러 개인 경우 proposal axis 전체에서 critic score가 높은 후보를 골라 actor batch에 전달합니다. 따라서 actor는 stale critic이 아니라 방금 업데이트된 critic 기준의 ranking을 봅니다.

## 전체 학습 흐름

`main.py`가 학습의 단일 entry point입니다.

1. YAML과 CLI flag를 합쳐 `dynamics_config`, `critic_config`, `actor_config`를 만듭니다.
2. `horizon`을 `dynamics_N`, `subgoal_steps`, critic `full_chunk_horizon`에 동기화합니다.
3. OGBench env와 dataset을 로드하고, 필요하면 env 기반 `max_goal_steps`를 resolve합니다.
4. `PathHGCDataset`과 `CriticSequenceDataset`에서 공통 valid start index를 맞춥니다.
5. Dynamics, critic, actor agent를 생성하거나 checkpoint에서 복원합니다.
6. 매 epoch마다 shared start index로 dynamics batch와 critic batch를 샘플링합니다.
7. Dynamics가 subgoal/bridge/IDM proposal을 만들고 dynamics loss를 업데이트합니다.
8. Critic은 offline action chunk 및 TRL fields로 업데이트됩니다.
9. 최신 critic으로 proposal을 rescore하고 actor batch를 구성합니다.
10. Actor는 SPI objective로 업데이트됩니다.
11. `eval_freq`마다 OGBench task eval을 수행하고, epoch 600에서는 N sweep final eval도 수행할 수 있습니다.

`async_prefetch`는 host-side batch sampling과 GPU compute를 겹치기 위한 단일 worker prefetch 경로입니다.

## Dataset 계층

### `PathHGCDataset`

Dynamics 학습용 dataset입니다. 각 sample은 다음을 포함합니다.

- `observations`, `next_observations`
- `high_actor_goals`
- `high_actor_targets`
- `trajectory_segment`
- path valid index와 bridge supervision에 필요한 trajectory 정보

`clip_path_to_goal=True`일 때 목표가 `K` step보다 가까우면 endpoint를 goal에 맞추고 tail을 goal state로 padding합니다. 가까운 goal에서 bridge가 goal을 지나치는 문제를 줄이기 위한 선택입니다.

### `CriticSequenceDataset`

Critic 학습용 dataset입니다. TRL mode에서는 일반 goal sampling 대신 `sample_trl_goals()`를 사용합니다.

일반적으로 다음을 만듭니다.

- full chunk action과 action chunk action
- value goal과 chunk backup target
- full/action chunk rewards, masks, backup horizon
- TRL transitive value tuple과 local Q goal fields

`value_geom_sample`의 의미는 중요합니다. `true`면 geometric future-goal sampling이고, `false`면 same-trajectory future goal을 uniform하게 뽑습니다. 최근 puzzle 4x5/4x6 sweep은 value TRL sampling을 `(cur, geom, traj, rand) = (0, 0, 1, 0)`으로 맞추며, 이는 `value_p_curgoal=0`, `value_geom_sample=false`, `value_p_trajgoal=1`, `value_p_randomgoal=0`에 해당합니다.

## 주요 모듈

### `agents/dynamics.py`

가장 큰 핵심 파일입니다.

- deterministic / Gaussian / flow subgoal network
- `forward_bridge_residual` trajectory planner
- flow endpoint sampling과 eval-time best-of-N selection
- IDM action decoding
- critic value를 사용한 subgoal value bonus
- dynamics phase-1 loss와 proposal construction

현재 Flow+TRL 경로에서는 rectified-flow loss와 transitive-product value weighting이 핵심입니다.

### `agents/critic.py`

Goal-conditioned critic stack입니다.

- DQC/IQL legacy path
- TRL critic path
- chunk critic, scalar value, transitive value 관련 helper
- TRL 여부 판별 및 config normalization

최근 sweep은 `algorithm: trl`, `critic_type: trl`, `use_chunk_critic: false`, `q_target_from_value: true`를 중심으로 사용합니다.

### `agents/actor.py`

Deterministic action chunk actor입니다.

- 입력: observation과 `spi_goals`
- 출력: `actor_chunk_horizon * action_dim` action chunk
- update 입력: `proposal_partial_chunks`, `proposal_scores`
- loss: critic Q를 높이는 항 + proposal proximity 항

Actor가 직접 final goal을 보는 대신 dynamics가 선택한 subgoal/proposal을 따라가는 것이 coupling의 핵심입니다.

### `utils/critic_sequence_dataset.py`

TRL critic 학습 batch를 만듭니다. 특히 `sample_trl_goals()`와 `_sample_trl_fields()`가 현재 TRL branch의 중요한 데이터 구성입니다.

### `utils/eval_results_io.py`, `eval_checkpoint.py`

Checkpoint 기반 eval 결과를 JSON으로 저장하고, 이미 저장된 결과는 `--skip_if_saved`로 건너뜁니다. Eval rollout은 별도 chunk budget 없이 환경의 max episode length까지 수행합니다.

### `rollout/`

Checkpoint를 로드해 subgoal, IDM, actor rollout을 수행하는 검증/시각화 경로입니다. ManipSpace 계열 cube/puzzle은 MP4 중심이고, maze 계열은 state-space plot이 의미 있습니다.

## Config와 Sweep 구조

### Flow+TRL 공통 sweep

`scripts/flow_trl_sweep_common.py`가 gap/wmax/N sweep의 공통 builder입니다.

- gap: `{1, 3, 5, 10}`
- wmax: `5`
- train N: `1`
- final eval N: `{1, 2, 4, 8, 16}`
- run order: `p3, p4, p45, p46, cs, cd, ct, amm, aml, amg`

`scripts/write_flow_trl_sweep_yaml.py`와 `config/sweep_flow_trl_finaleval/`가 일반 final-eval sweep을 담당합니다.

### Puzzle 4x5 / 4x6 sweep

최근 코드스페이스에는 `puzzle-4x5-play-v0`, `puzzle-4x6-play-v0` 전용 sweep이 추가되어 있습니다.

- writer: `scripts/write_flow_trl_puzzle_45_46_yaml.py`
- runner: `scripts/run_flow_trl_puzzle_45_46.sh`
- configs: `config/sweep_flow_trl_puzzle_45_46/`
- run_group prefix: `flow_trl_p456g999`
- gamma: `0.999`
- value distance weight power: `0`
- policy sampling `(cur, geom, traj, rand) = (0, 0.5, 0, 0.5)`
- value TRL sampling `(cur, geom, traj, rand) = (0, 0, 1, 0)`
- eval budget: 환경 max episode length

Runner는 기존 run을 찾고, eval JSON이 다 있으면 skip합니다. gamma mismatch가 감지되면 retrain 대상으로 처리합니다.

### K=40 best follow-up

`config/sweep_flow_trl_k40_best/`와 `scripts/write_flow_trl_k40_best_yaml.py`는 horizon 40 follow-up 실험용입니다. 기존 K=25 best parameter를 기반으로 selected env/config를 K=40으로 확장합니다.

### Antmaze-giant env-max eval

`scripts/run_amg_m800_eval.sh`는 antmaze-giant checkpoint를 환경 max episode length 기준으로 재평가합니다.

관련 chain script:

- `scripts/run_amg_m800_then_p456.sh`: giant env-max eval 후 puzzle 4x5/4x6 sweep 재개
- temp: `1.0`, `0.5`
- eval N: 현재 `{2, 8, 16, 32}`

## 평가와 결과 산출

학습 중 eval과 `eval_checkpoint.py`는 같은 core rollout 경로를 사용합니다.

- IDM policy: `infer_subgoal(obs, goal) -> bridge plan -> IDM action chunk`
- Actor policy: `infer_subgoal(obs, goal) -> actor.sample_actions(obs, subgoal)`
- success: OGBench env의 `info["success"]`

`scripts/summarize_feval_results.py`는 `runs/*/eval_results/*.json`을 모아 master CSV/MD를 만듭니다. 현재 코드스페이스의 규칙은 generated docs CSV/MD 파일명에 `_7ch`를 붙이는 것입니다.

- `scripts/docs_output_paths.py`: docs 출력 파일명 helper
- `flow_trl_feval_results_7ch.csv`
- `flow_trl_feval_results_7ch.md`
- `runs_results_*_7ch.*`
- `douri_runs_results_*_7ch.*`

`_7ch`가 붙지 않은 generated CSV/MD는 더 이상 만들지 않는 방향입니다. 수동 작성 문서와 figures는 이 규칙의 대상이 아닙니다.

## 구현상 좋은 점

- Dynamics와 critic이 공통 start index를 공유하므로 actor coupling에 쓰이는 상태 분포가 흔들리지 않습니다.
- Flow subgoal, Gaussian subgoal, deterministic subgoal을 같은 `DynamicsAgent` API 아래에 둬 ablation 전환이 쉽습니다.
- TRL critic batch가 별도 dataset class로 분리되어 transitive value target 구성이 명확합니다.
- `eval_checkpoint.py`가 `flags.json`을 읽어 checkpoint hyperparameter를 복원하므로, 훈련 이후 N/temp eval sweep을 붙이기 쉽습니다.
- sweep runner들이 `skip_if_saved`, `eval_results_complete`, `run_gamma_matches_config`를 사용해 긴 실험을 재개하기 쉽습니다.
- generated docs output path를 `docs_output_paths.py`로 묶어 `_7ch` 파일명 규칙을 한 곳에서 관리합니다.

## 주의할 점

### 1. Config와 `flags.json`의 block 구분

`flags.json`에는 `dynamics`, `critic_agent`, `actor` block이 모두 저장됩니다. TRL value goal sampling은 `critic_agent` block이 기준입니다. `dynamics` block에도 과거 default의 `value_geom_sample`이 보일 수 있으나, critic 학습에는 `critic_agent`의 값이 적용됩니다.

### 2. Generated docs는 commit 대상과 분리

CSV/MD summary는 로컬 분석 산출물입니다. 현재 규칙은 `_7ch` suffix를 붙여 생성하고, 필요할 때만 명시적으로 commit합니다. 대용량 checkpoint zip이나 ad-hoc 결과표는 기본적으로 commit 대상에서 제외하는 편이 안전합니다.

### 3. Long-running sweep 재개

Puzzle 4x5/4x6와 giant env-max eval 같은 sweep은 nohup chain으로 오래 돕니다. 중간에 중지/재시작할 때는 run_group, gamma, eval JSON completeness를 기준으로 skip/retrain 여부가 결정됩니다.

### 4. `agents/dynamics.py` 책임이 큼

`agents/dynamics.py`는 network, planner, flow sampling, IDM, subgoal value loss, proposal builder까지 포함합니다. 실험 속도에는 유리하지만, 장기적으로는 다음 단위로 나누면 유지보수가 쉬워집니다.

- `dynamics_networks.py`
- `dynamics_planner.py`
- `dynamics_flow.py`
- `dynamics_losses.py`
- `dynamics_proposals.py`

다만 지금은 public API인 `infer_subgoal`, `plan`, `build_actor_proposals`, `update`가 여러 스크립트에서 쓰이므로, 큰 리팩터보다 내부 helper부터 옮기는 편이 안전합니다.

## 정리

현재 코드스페이스의 중심 질문은 다음입니다.

> Offline long-horizon goal-conditioned control에서, learned flow subgoal과 forward-bridge residual path, IDM actionization, TRL critic-ranked proposal을 결합하면 sparse goal task를 더 안정적으로 풀 수 있는가?

이 branch는 그 질문을 gap/wmax/N, K=25/K=40, eval temperature, eval max chunks, puzzle 4x5/4x6 확장까지 빠르게 검증할 수 있도록 구성되어 있습니다. README는 실행 reference로 두고, 이 문서는 현재 구현 의도와 파일별 책임을 파악하기 위한 companion note로 보면 됩니다.
