# PB 기반 가변 길이 skill: subgoal / global-goal 두 버전

2026-11-07. 기준: `SChoish/PathBridger@943819864628832c93ea0a35e6f549db50060849`.
사용자 지시대로 **두 실행 모드의 코드·오프라인 학습·평가 경로를 구현**했다.
실제 PB bundle·OGBench archive를 사용한 정식 학습/환경 평가는 아직 실행하지 않았다.
기존 PBF/PBG 16개 preset과 원래 agent/bridge/IDM 코드는 변경하지 않는다.

## 두 버전의 차이

| 단계 | Subgoal 모드 | Global 모드 |
|---|---|---|
| Skill이 지향할 목표 (g_\mathrm{skill}) | 고정 PB endpoint proposer의 후보를 PB의 (V(s,u)V(u,g))로 선택한 (u) | 환경에서 받은 최종 목표 (g) |
| Skill 후보 | (p_\xi(z\mid s,g_\mathrm{skill},d)) | 동일 모델/API |
| Skill 선택 | (\arg\max Q_d(s,z,g_\mathrm{skill})) | 동일 모델/API |
| 실행 | 상태 독립 분해의 앞 child를 5-step action chunk로 decode | 동일 |
| 이후 | 나머지 child를 버리고 실제 상태에서 새 계획 | 동일 |

두 버전은 **동일하게 학습한 encoder/분해기/decoder/proposal/Q checkpoint**를 사용한다.
따라서 두 학습 런을 따로 수행해야 하는 구성이 아니다. 추론에서 목표를 정하는 단계가 다르다.
Subgoal 모드는 bridge/IDM을 호출하지 않는다. Global 모드는 PB endpoint proposer도 호출하지 않는다.
Global의 최종 goal은 사용자/환경이 제공하며 중간 상태를 생성해 subgoal처럼 쓰지 않는다.
두 버전 모두 별도의 첫-slot selector/수리 network를 추가하지 않고 합의한 Q로 부모 후보를 선택한다.
사용자가 언급한 “스킬 셀렉터”는 작업 절차를 고르는 도구 지시이며 별도 RL 알고리즘 요청이 아니었다.

## 가변 길이 표현과 실행

부모는 (Z=(z,d)), (z\in\mathbb R^{16})이다. 초기 설정의 길이는

\[
d\in\{5,10,15,20,25,30,35,40\},\quad k=5.
\]

16차원이나 최대 길이 40이 최적이라는 결과는 없다. 초기 구현 설정이다.
길이는 벡터 norm에서 추정하지 않고 별도 입력한다. 매 (d-k)가 학습 길이 집합에 있도록 검사한다.
Masked BiGRU가 실제 가변 길이 관측·행동 구간을 Gaussian posterior로 인코딩한다.
공유 분해기 `D(z,d,k)`는 상태나 목표를 받지 않고 두 child의 평균·분산을 출력한다.
같은 관측·code로 매-step 행동을 생성하는 기존 OPAL 경로가 아니다.

\[
(z_L,z_R)=D_\phi(z,d,k),\qquad
A_{0:k}=\operatorname{Dec}_\theta(s,z_L,k).
\]

실행 시에는 앞 child만 decode한다. 원래 부모와 뒤 child를 끝까지 펼치지 않는다.
Decoder는 길이 입력을 받는 고정 최대 크기 action head이며, 유효한 앞 k개를 실행한다.
따라서 현재 구현이 출력 head 계산까지 k에 비례하도록 최적화됐다는 주장은 하지 않는다.
환경 경계/남은 episode 예산에서 실행을 멈추는 것은 기존 PB 평가 루프를 따른다.

## 학습 1: 실행 표현

정규화는 train archive로만 계산한다. 길이를 먼저 균등 표집하고, 그 길이가 실제로 존재하는
episode 내부의 시작점을 표집한다. terminal의 dummy action, `valids=0` 구간, reset을 넘지 않는다.
각 부모에서 앞 k/뒤 d-k의 direct posterior를 함께 계산한다.

- 부모의 전체 action NLL.
- 분해한 두 child의 해당 구간 action NLL. 뒤 child는 기록된 실제 경계 관측에서 decode한다.
- Direct child Gaussian과 분해 출력 Gaussian의 KL. Direct target은 stop-gradient이다.
- 부모 posterior와 표준 Gaussian의 KL, 기본 beta=0.01.

앞의 세 항 기본 계수는 1이다. 모든 NLL은 유효 시간·action 좌표 평균 후 샘플 평균한다.
가변 입력 mask는 recurrent projection 전에 적용하고 padding으로 NaN이 들어와도 제거한다.
좌표 KL이나 teacher-forced NLL의 개선을 feedback 실행 보존으로 간주하지 않는다.
합성 M이나 군 구조를 새로 구현했다고 부르지 않는다. 이번 범위는 학습된 분해와 실행 연결이다.

## 학습 2: 전체 계획 Q와 skill proposal

표현 학습 후 encoder/분해기/action decoder는 고정한다.
Q 2개와 conditional rectified-flow skill proposer만 별도의 Adam 상태로 학습한다.
PB 전체 파라미터는 두 단계 모두 고정하며 목표 가치만 사용한다.

\[
Q_d^{\bar V}(s,z,g)
=\mathbb E[\gamma^{T_g}\mathbf1\{T_g\le d\}
+\gamma^d\mathbf1\{T_g>d\}\bar V(s_d,g)].
\]

직접 anchor는 기록된 첫 도달 수익 또는 `gamma**d * frozen_PB_V(s_d,g)`이다.
분해 target은 앞 k 안에 도달했으면 그 수익, 아니면

\[
y_\mathrm{split}=\gamma^k\operatorname{sg}
[\min_i\bar Q_{d-k,i}(s_k,z_R,g)].
\]

뒤 child가 target에서 사라지지 않는다. 왼쪽의 연속 경계 상태 도달 Q를 새로 학습해 곱하지 않는다.
Anchor BCE와 split BCE의 기본 계수는 1이다. 전체 부모 외에 direct 앞/뒤 구간도 anchor로 감독한다.
Q target ensemble은 0.005 EMA이며, encoder/실행기의 파라미터를 변경하지 않는다.
`split_value_weight=0` 대조를 같은 sampler/초기값/예산으로 실행할 수 있다.

Proposal은 부모 posterior 평균을 flow target으로 사용한다. 시간과 Gaussian noise를 표집하고,
고정 PB 가치 대비 anchor return의 advantage에 따른 정규화 exponential weight를 사용한다.
기본 value scale 10이며 exponent를 [-10,10]으로 제한한다. 이는 새 제안 분포의 recipe다.
학습 goal 절반은 구간의 실제 endpoint, 나머지는 PB의 actor goal sampler에서 얻은 future goal이다.
Endpoint goal이 학습에 들어간다고 global 추론에서 endpoint/subgoal을 생성하는 것은 아니다.

두 Q는 full normalized observation goal의 tolerance 1e-5 첫 도달 척도를 사용한다.
이는 PB의 full-state 가치 입력과 일치시키기 위한 학습 정의이며,
환경의 공식 성공 반경이나 oracle goal feature를 새로 구현한 것이 아니다.
Compact `terminals`는 자료 경계로 취급하며, 실제 흡수 terminal이라는 증거 없이 bootstrap을 0으로 만들지 않는다.

## 중요한 근사와 판정 범위

**기록 구간을 posterior code에 연결한 Q label은 근사다.** 새 code/분해기가 기록 action을 정확히
실행했다는 증거가 아니다. 이 한계를 actor SHA 검증이나 unit test로 해결했다고 주장하지 않는다.
실제 제어 우위는 고정 모델의 환경 평가로 확인해야 한다.
Q는 고정 PB continuation 아래의 전체 skill 계획 점수이며, 5-step마다 계획을 바꾸는 배포 정책의
정확한 가치함수라고 주장하지 않는다. 압축 차원, 분해 일반화, 후보 coverage, 장기 Q 정확도는 미검증이다.

## 추론과 비교

기본 전체 skill 후보는 32개다. 길이를 순환 배정해 여덟 길이에 각 4개를 생성한다.
모든 부모를 Q로 평가한 뒤 한 부모만 한 번 분해·decode한다. future trajectory는 추론 입력이 아니다.
Subgoal 모드는 추가로 PB endpoint 32개를 평가하므로 총 inference 비용이 global과 같지는 않다.
같은 skill 후보 예산과 동일 episode seed로 비교하며, extra endpoint 작업은 별도로 기록한다.

`evaluate`는 기본적으로 subgoal/global/원래 PB 세 팔을 deterministic-v1으로 실행한다.
고정 모델과 원래 PB의 official task success를 같은 episode seed·환경 step 제한에서 비교한다.
모든 팔을 포함한 명시적 policy-step cap을 실행 전에 요구한다. 원시 상태·행동·success/goal을 저장하고,
skill 팔의 선택 길이·점수·실제 입력 goal을 기록한다. evaluation 중 optimizer update는 없다.
Checkpoint가 smoke이거나 표현/Q 학습이 비어 있으면 정식 평가를 거부한다.

## 실행

기존 `inference.py --export_path=...`로 **학습된 공식 PB checkpoint**를 inference bundle로 내보낸다.
초기화만 된 PB를 성공 baseline/장기 가치 teacher로 사용하면 안 된다.
아래 변수는 사용자가 가진 실제 파일로 지정해야 한다. 이번 작업에서는 해당 파일에 접근하지 못했다.

```bash
python skill_main.py train \
  --pb-bundle "$PB_BUNDLE" --pb-sha256 "$PB_SHA256" \
  --dataset-dir "$LOCAL_OGBENCH_DATA" \
  --config skill_configs/common.json --seed 0 \
  --output "$SKILL_RUN" --checkpoint-dir "$RAID_SKILL_CHECKPOINTS" \
  --representation-updates 20000 --value-updates 20000

python skill_main.py evaluate \
  --pb-bundle "$PB_BUNDLE" --pb-sha256 "$PB_SHA256" \
  --checkpoint "$RAID_SKILL_CHECKPOINTS/repr0020000_value0020000.pkl" \
  --output "$SKILL_EVAL" --episodes 50 --mode both --step-cap "$ALL_ARM_STEP_CAP"
```

`--mode subgoal` 또는 `--mode global`로 한 버전만 평가할 수 있고, `--skip-pb`로 원본 비교 팔을 뺄 수 있다.
CLI는 이미 있는 로컬 NPZ archive를 요구하며 데이터를 자동 다운로드하지 않는다.
기본 학습은 GPU를 요구한다. `--smoke`는 단계별 최대 3 update이며 정식 평가용이 아니다.
1,000 update마다 optimizer/JAX RNG/sampler RNG/단계·자료·teacher/source hash를 atomic 저장한다.
저장마다 같은 directory의 용량을 검사하고 최소 1 GiB 또는 checkpoint의 3배를 요구한다.
실제 서버에서 checkpoint directory는 RAID 경로를 명시한다. 이전 artifact를 덮어쓰지 않는다.
Train/validation content identity와 정확히 동일한 전체 episode의 중복을 검사한다.
이는 모든 물리적 overlap의 부재를 증명하는 검사가 아니다.

## 이번 검증의 범위

새 numerical tests는 fake PB teacher를 명시적으로 사용한다. 실제 JAX 업데이트·파라미터 scope,
정확한 optimizer/sampler 재개, full-parent Q argmax, 상태 독립 분해, 가변 길이 mask,
global의 endpoint 미호출, subgoal 입력, 짝지은 모의 환경 평가·예산·clipping을 검사한다.
실제 PB bundle 복원과 OGBench 성능·기존 전체 regression suite는 아직 검증하지 못했다.
필요한 의존성 설치가 자동 승인 심사에서 거절되어 추가 설치 없이 가능한 검사만 수행했다.
검증된 수치 동작과 실제 학습·성능을 구분한다. 새 정식 학습/rollout은 시작하지 않았다.
