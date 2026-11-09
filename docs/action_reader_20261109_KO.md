# Action-only parent와 interval reader: 구현 계약

2026-11-09 KST. 기반 소스: 3900308, codex/skill-subgoal-global-20261107.
변경 branch: codex/skill-action-reader-20261109.

이 문서는 상태를 포함한 부모 encoder 설계를 사용자가 지적·수정한 뒤의 기준이다.
기존 skill_main.py와 skill_extensions/{models,planner,runtime}.py는 변경하지 않는다.
서버의 최신 residual script 또는 100k donor checkpoint를 직접 확보한 구현은 아니다.

## 1. 입력과 의미

부모 encoder의 API는 encode(actions, lengths)이며 관측/goal을 받지 않는다.

\[
u=E_\theta(a_{0:20}),\quad z_j=R_\psi(u,20,5j,5),\quad
\widehat A_j=\operatorname{Dec}_0(s_{5j},z_j,5),\quad j=0,\ldots,3.
\]

같은 action sequence는 출처 상태와 goal을 바꿔도 같은 부모가 된다.
상태 trajectory는 decoder 시작 상태와 Q의 first-hit/경계 표적에 사용하며,
부모 encoding 경로에 넣지 않는다. 최종 goal은 proposal/Q만 조건화한다.

SkillInterval(root,L,a,b)는 원래 벡터와 시간 구간을 보관한다. 분해는
구간을 나누며 root를 재생성하지 않는다. child는 부모에서 독립한 canonical
latent가 아니라 view이다. 학습된 compact splitter/합성/시간 일반화의 완성이 아니다.

## 2. 초기화와 자산

- 부모 action-only BiGRU는 **새로 초기화**한다. 상태·행동 encoder의 입력층 삭제,
  부분 가중치 이식, optimizer 이식은 하지 않는다.
- reader 대 recursive 팔은 같은 seed의 동일한 새 encoder 초기값을 공유한다.
- 유일한 학습된 donor 이식은 짧은 ChunkDecoder와 그 **원래 관측 정규화**이다.
  새 archive에서 관측 정규화를 다시 fit해 frozen decoder에 적용하지 않는다.
- donor의 latent_dim, width, max_duration, observation_dim, action_dim, checkpoint SHA,
  representation_step과 smoke 여부를 검사한다. 상태·행동 encoder를 가져오지 않는다.
- 지원 donor schema는 pathbridger-variable-skill-v1이다. 최신 서버 schema가 다르면
  명시적인 export adapter가 필요하다. 이를 자동 추정하지 않는다.
- pickle 입력은 신뢰하는 본인 파일에 한정하며, 정확한 SHA 검사를 역직렬화 전에 한다.

단계 1에서 encoder/reader는 새로 학습하므로 과거 warm-start 부모 encoder의
20k 결과와 동일한 초기화 비교가 아니다. A/B를 같은 조건으로 새로 비교해야 한다.

## 3. 학습과 동결

| 단계 | 주 표적 | 갱신 | 동결 |
|---|---|---|---|
| representation | 기록 action MSE | 새 action encoder + reader 또는 recursive splitter | donor decoder와 원래 normalizer |
| G1 | 같은 source/start에서 첫 프로그램 실행 비교 | 없음 | 모든 파라미터 |
| proposal | 원래 부모 code 분포의 conditional FM | flow만 | encoder/reader/decoder |
| value | finite-candidate expected-max first-hit target | Q0/Q1 + target EMA | encoder/reader/decoder/flow |
| 배포 | 32개 후보 중 min-Q 최대 선택, 20-step 유지 | 없음 | 전부 |

단계별 optimizer state를 별도로 보관한다. active parameter subtree만 미분·갱신한다.
frozen decoder의 파라미터를 고정하되 입력 code에 대한 derivative는 유지한다.

### Representation

\[
\mathcal L_{\rm repr}=\frac1{20A}\sum_{j=0}^3\sum_{\ell=0}^4
\|\widehat a_{j,\ell}-a_{5j+\ell}\|_2^2.
\]

기록 시작 상태에서 네 chunk를 decode한다. code KL, prior KL, teacher imitation,
value auxiliary, 첫 chunk 특별 가중치를 합치지 않는다. full posterior action 복원은
기록 전이를 Q 표적으로 쓰는 근사를 겨냥하지만 transition 일치를 증명하지 않는다.

### Proposal

동결 부모를 training windows에서 streaming mean/std로 표준화한다. 표준편차 하한은
1e-3이다. validation을 쓰지 않고 sampler RNG도 소모하지 않는다.

\[
x_t=(1-t)\epsilon+t v,\quad
\mathcal L_{\rm FM}=\mathbb E\|F_\eta(s,g,x_t,t)-(v-\epsilon)\|^2/16.
\]

return 가중 없이 flow만 학습한 뒤 고정한다. Euler 8-step, noise 온도 1,
기본 후보 수 32이다. 정확한 dataset support 보장은 아니다.

### Value

\[
y=\mathrm{sg}\left[H_{20}^g+B_{20}^g
\max_{i\le32}\min_{q=0,1}\bar Q_q(s_{20},v'_i,g)\right],
\quad v'_i\sim P(\cdot\mid s_{20},g).
\]

H는 구간 안 first-hit의 gamma^T, B는 미도달·비흡수 상태의 gamma^20이다.
true failure terminal이면 B=0이다. **compact episode 경계와 true terminal은 다르다.**
--boundary-kind timeout|terminal|field를 명시하며 field는 true_terminals 필드를 요구한다.
시작 상태의 hit는 1이며 coincident terminal보다 success가 우선한다.

Q는 sigmoid logit BCE이다. target이 [0,1]이므로 조건부 평균의 proper loss로 정합적이다.
EMA는 0.005, target/배포 모두 같은 max-over-32 min-over-2 규칙이다.
별도 expectile V 또는 actor loss는 없다. encoded skill의 실제 전이가 기록 경계와
다를 수 있는 factual-window approximation은 남는다.

## 4. 데이터와 goal

명시적 local train/validation compact NPZ를 받는다. archive를 내려받거나
환경을 생성하지 않는다. 유효한 완전한 20-step window만 균등 추출하며
NaN terminal dummy action, invalid action row, episode reset을 넘는 window를 제외한다.
complete episode fingerprint가 겹치면 train/validation 구성을 거부한다.
이 검사는 물리적으로 비슷한 episode까지 검출하는 것은 아니다.

proposal/Q goal의 절반은 구간 endpoint이고 나머지는 기존 PB goal sampler의
actor_p와 training_objective_protocol을 사용한다. --goal-config JSON의 실제 값과
discount를 checkpoint에 기록한다.

Q의 hit predicate는 --goal-contract JSON이 지정하는 module:function이다.
입력은 states [B,21,D], full observation goals [B,D], parameters dict이며,
출력은 bool [B,21]이어야 한다. 환경 이름·검증 근거·predicate 소스 hash를 기록한다.
generic full-state 근접 판정을 benchmark 성공으로 대체하지 않는다.
내장 diagnostic_full_state_linf는 synthetic smoke 전용이며 formal Q에서 거부한다.
실제 환경 goal adapter는 현재 확보하지 않았으므로 정식 Q 실행 전 구현·검증해야 한다.

## 5. 배포와 G1

CommittedActionPolicy는 실제 상태에서 부모를 생성하고 20-step 동안 유지한다.
5-step 경계에서만 최신 관측으로 decoder를 다시 호출한다.
primitive step 사이에는 생성한 5-action chunk를 실행한다. root는 다시 인코딩하지 않는다.
episode goal이 중간에 바뀌면 reset을 요구한다. action clipping은 명시적인
환경 action bounds를 사용한다. 외부 환경 루프는 success/terminal/truncation에서 멈춰야 한다.

recorded_state_errors는 기록 행동 대비 chunk별 MSE와 L2를 구분한다.
이 함수는 **환경 검증 없이 G1 pass를 만들지 않는다.**
execute_fixed_program은 외부에서 동일 full simulator state로 복원한 환경에서
첫 20-step 프로그램 하나만 실행한다. Q 선택과 후속 재선택이 없고 first-hit에 멈춘다.
직접 child code열은 명시적인 oracle reference 인수이며 배포 정책 입력이 아니다.

formal proposal 단계는 실제 representation checkpoint SHA와 parameter hash에
연결된 G1 pass report를 요구한다. smoke는 이를 우회하지만 우회 사실을 기록하며
smoke checkpoint의 formal deployment를 거부한다.

## 6. 근거와 설계 의도

| 선택 | 근거 | 설명의 한계 |
|---|---|---|
| action-only encoder | 사용자가 확정한 action-program 정의; 복원 표적 자체를 입력 | 상태 효과·제어 의도 latent의 동등성을 주장하지 않음 |
| 불변 root + interval | 자체 정의: 같은 root/구간이면 분해 순서와 reader 입력이 동일 | 원본 행동 정보 보존/새 상태 성공은 검증 가설 |
| 새 encoder 초기화 | 상태 입력층 크기와 입력 의미가 바뀜 | 과거 warm-start 수치를 그대로 비교할 수 없음 |
| donor decoder + normalizer 동결 | executor와 표현 변경을 분리 | donor decoder 자체 한계가 남음 |
| 기록 action 단일 loss | factual-window label과 action 복원 표적 연결 | 환경 경계 일치의 충분조건으로 사용하지 않음 |
| 단계별 동결 | 표현 이동과 proposal/Q 변경을 분리 | 결합 end-to-end 최적성 보장 없음 |
| plain conditional FM | [R2] §3.2, 식 (14),(20)-(23), 기존 코드 | finite Euler의 정확한 support/분포 보장 없음 |
| sampled expected-max Q | [R1] §3.1-3.3, 식 (5),(10)-(11) | neural latent macro-action 수렴 보장 이식 없음 |
| 두 critic min/target | [R3] §4 식 (10), §5.2 및 기존 코드 | 불편 추정/정확한 보수적 bound 아님 |
| MLP GELU | 실제 utils.networks.MLP 재사용 | 기존 PDF의 ReLU 표기를 수정; 성능 개선 주장 없음 |
| 20/5, 16D, 200, batch256, lr3e-4 | 기존 진단·소스와 비교를 위한 고정값 | 최적값·필수 차원·통계적 power 주장 없음 |

[R1] Ghasemipour, Schuurmans, Gu (2021), EMaQ.
https://proceedings.mlr.press/v139/ghasemipour21a/ghasemipour21a.pdf

[R2] Lipman et al. (2023), Flow Matching for Generative Modeling, v2.
https://arxiv.org/html/2210.02747v2

[R3] Fujimoto, van Hoof, Meger (2018), Addressing Function Approximation Error in
Actor-Critic Methods, v3. https://arxiv.org/pdf/1802.09477

[R4] Li, Park, Levine (2025), Decoupled Q-Chunking, v2 §4.
https://arxiv.org/pdf/2512.10926v2
기록 실행과 실제 실행 분포의 구분에 사용한다. 5-step feedback을 사용하는 본 구조에
원문의 open-loop 보장을 그대로 적용하지 않는다.

## 7. 실행 경로

아래 명령은 실제 donor/data가 있는 호스트에서 사용할 명세이며, 이 문서 작성 중
서버에 제출하거나 정식 학습을 시작한 명령이 아니다.

~~~bash
python action_skill_main.py --stage representation \
  --train-npz "$TRAIN" --validation-npz "$VALIDATION" \
  --donor "$DONOR" --donor-sha256 "$DONOR_SHA256" \
  --config skill_configs/action_reader.json \
  --seed 0 --updates 20000 --batch-size 256 --output runs/action_reader/seed0/repr

python action_skill_main.py --stage proposal \
  --train-npz "$TRAIN" --validation-npz "$VALIDATION" \
  --input-checkpoint "$REPR" --input-sha256 "$REPR_SHA256" \
  --goal-config "$GOAL_CONFIG" --gate-report "$G1" --gate-sha256 "$G1_SHA256" \
  --seed 0 --updates 20000 --batch-size 256 --output runs/action_reader/seed0/flow

python action_skill_main.py --stage value \
  --train-npz "$TRAIN" --validation-npz "$VALIDATION" \
  --input-checkpoint "$FLOW" --input-sha256 "$FLOW_SHA256" \
  --goal-config "$GOAL_CONFIG" --goal-contract "$GOAL_CONTRACT" \
  --boundary-kind field \
  --seed 0 --updates 20000 --batch-size 256 --output runs/action_reader/seed0/q
~~~

recursive 대조는 첫 명령의 config만 skill_configs/action_recursive.json으로 바꾼다.
상태·행동 legacy encoder를 사용할 수 있는 옵션은 새 entry point에 없다.
output 디렉터리는 새 경로를 요구한다. 같은 단계의 input checkpoint는 optimizer,
sampler RNG와 model RNG를 복원하며, dataset/batch/goal/terminal 계약 변경을 거부한다.
의도적으로 바꾼 코드에서 strict source-bound checkpoint를 이어 학습하는 것은 지원하지 않는다.

## 8. 검증 및 남은 것

검증은 tests/test_action_program.py와
computations/action_reader_20261109/verification.json에 남긴다.
action-only 입력 불변성, padding, 분해 경로, A/B 동일 초기값,
동결 decoder의 입력 gradient, 세 단계 갱신 범위, EMA, 독립 손계산 target,
sampler/resume, train-only root 통계, 20/5 commitment와 3-stage CLI smoke가 대상이다.

정식 20k/seed 학습, 실제 donor import, GPU stack, OGBench G1 rollout/성공률은
아직 검증하지 않았다. synthetic CLI/수치 검증을 실제 환경 성공으로 보고하지 않는다.
균형 트리 백업, learned switching, prior/code KL, online data 추가는 구현하지 않았다.
