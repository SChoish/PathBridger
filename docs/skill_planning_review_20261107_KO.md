# PB 기반 skill 학습: 학습 전 수학·구현 재검토

2026-11-07 KST. 검토 대상은 `SChoish/PathBridger@ede2b8a5b39f367f8ecf8f14003aa0ace880b18b`이다.
이번 검토는 동일 작성자의 **자기 검토**이며 독립 심사나 실제 제어 성능 검증이 아니다.
핵심 6개 소스는 해당 원격 commit과 로컬 bytes가 같은지 확인했다.

## 판정과 범위

**주 판정: `conditional on a named input` — 분해한 code의 실행과 factual 구간의 대응에 조건부다.**

기록된 한 경로의 할인 first-hit/terminal-potential 값을 나누는 식은 맞는다.
확인한 범위에서 상태 독립 분해, action-chunk 실행, 업데이트 범위와 할인 계산을 위반하는
새로운 중대 구현 결함은 확인하지 못했다. 작은 오프라인 pilot을 시도할 수 있는 구조다.
다만 현재 Q는 posterior-code와 factual 구간을 연결한 근사이므로, 학습된 실행기에 대한
정확한 Bellman 평가나 정책 개선 보장이 확보됐다고 말할 수 없다.

알고리즘 소스·기본 계수·학습 길이·checkpoint 선택 규칙은 이번 재검토에서 변경하지 않았다.
정식 학습과 실제 환경 실행도 시작하지 않았다. 회의록은 수정하지 않았다.

## 1. 검토한 주장과 의존 관계

부모 `Z=(z,d)`, `z∈R^16`, `d∈{5,10,...,40}`, 실행 cut `k=5`를 사용한다.

1. 기록 경로 τ의 값 F를 정의한다. 목표 도달은 정규화된 full observation의
   max-absolute 오차가 1e-5 이내인 첫 시점 T이며, 시작점 0과 끝점 d도 포함한다.
2. `F_d(τ,g)=γ^T` if `T≤d`; 미도달이면 `γ^d·V_PB(s_d,g)`이다.
3. 동일 경로와 동일 끝점 potential을 쓰면, 앞 k에서 도달한 경우 그 수익을 취하고,
   아니면 `γ^k·F_{d-k}(τ_R,g)`로 나누는 항등식이 성립한다.
4. 구현은 F의 직접 label을 인코딩한 부모·앞·뒤 code에 붙인다.
5. 분해 백업은 기록 경계 `s_k`와 **분해기가 예측한** suffix code를 target Q에 넣는다.
6. 배포는 전체 부모 Q로 후보를 고르고, 분해한 앞 child만 decode하여 5 step 실행한다.

3은 경로 함수의 항등식이다. 3에서 5·6의 정확한 정책 가치로 넘어가려면
code와 실행 경로의 대응, suffix Q의 정확도라는 별도 전제가 필요하다.
이를 KL/NLL 손실이 작다는 사실이나 단위검사 통과로 대체하지 않는다.

## 2. 확인 의무

| 의무 | 판정 | 근거 또는 남은 조건 |
|---|---|---|
| 같은 factual 경로의 할인·first-hit 분할 | passed | 아래 도출 및 유리수 1,764개 검사 |
| 시작점·cut·끝점 도달, 미도달, padding | passed, 유한 범위 | 실제 `first_hit` 196개 및 `return_target` 1,764개 비교 |
| 분해기 입력의 상태·목표 독립 | passed, 코드 범위 | `SkillSplitter(z,d,k)` 입력과 테스트 |
| chunk 내부에 매-step 상태·phase를 재입력하지 않음 | passed, 코드 범위 | 시작 상태 decoder와 기존 5-action 실행 루프 |
| 표현/Q/proposal/PB 업데이트 범위·정확 재개 | passed, CPU 모의 범위 | 기존 11개 테스트 재실행 |
| global 추론의 subgoal/PB 호출 없음 | passed, CPU 모의 범위 | 금지 호출을 넣은 검사 |
| 분해 child가 기록 prefix·suffix 실행을 보존 | conditional | 동결 실행기의 실제 경로 비교 필요 |
| predicted suffix code에서 Q가 직접 posterior만큼 정확 | conditional | 현재 직접 anchor는 direct posterior code에 존재 |
| 부모 전체 계획 점수가 재계획 정책의 정확한 가치 | not addressed | 두 객체를 동일하다고 주장하지 않는 설계 |
| 실제 PB bundle 복원·OGBench 연결 | not addressed | bundle/archive 미제공; 기존 regression은 의존성 부족 |
| 학습·제어 우위와 horizon reduction | out of scope | 이번에는 정식 학습·실환경 실행 없음 |

## 3. 할인 분할식의 도출

미도달을 `T=∞`로 쓰자. `T≤k`이면 두 식 모두 `γ^T`다.
`k<T≤d`이면 suffix의 첫 도달 시간은 `T-k`이므로

\[
\gamma^k F_{d-k}(\tau_R,g)
=\gamma^k\gamma^{T-k}=\gamma^T.
\]

`T>d`이면 양쪽이 같은 끝점과 같은 terminal potential을 사용하므로

\[
\gamma^k F_{d-k}(\tau_R,g)
=\gamma^k\gamma^{d-k}\bar V_{\rm PB}(s_d,g)
=\gamma^d\bar V_{\rm PB}(s_d,g).
\]

따라서 추가 discount나 왼쪽 경계 도달 Q의 곱은 필요 없다.
다만 실제 target Q는 위의 정답 F가 아니라 학습된 함수다. 두 개의 target Q 중 최솟값도
정답 또는 신뢰구간 하한이라고 보장되지 않는다.

## 4. 핵심 우려 A: factual label과 실행의 대응

위 식이 수학적으로 맞아도 `s_k`가 새 decoder의 실제 경계 상태인 것은 아니다.
부모 전체 decoder와 분해한 child decoder 역시 동일 실행 함수로 강제되지 않는다.
현재는 양쪽을 같은 기록 action에 맞추어 근사한다.

가장 작은 예로 결정론적 누산기 `s_{j+1}=s_j+a_j`, 시작 0, 목표 10을 생각한다.
기록 action 10개가 모두 1이면 10 step에 도달하며 label은 `0.99^10≈0.90438`이다.
분해한 prefix가 처음 5개 action을 0으로 출력하고 suffix는 1을 출력하면,
실제 경계는 기록의 5가 아니라 0이고 최종 위치는 5다. 실제 first-hit 수익은 0이다.
이는 새 모델에서 관측한 실패가 아니라 **실행 대응을 가정 없이 생략할 수 없다는 반례**다.

최소 확인 항목은 heldout의 direct-parent/direct-child/decomposed-child 행동 복원과,
동결 모델의 동일 시작점 실제 prefix 실행이다. Offline 오차에 임의 문턱을 정해
기능 평가를 영구 금지하지 않는다. 실제 task 성공은 별도 지표로 평가한다.

## 5. 핵심 우려 B: suffix query와 손실의 실제 비중

`planner.py`의 직접 anchor는 부모·앞 child·유효한 뒤 child를 합쳐 평균한다.
split 항은 suffix가 존재하는 부모만 평균한다. 따라서 두 계수를 1로 둬도
**부모 한 query에서 두 손실의 비중은 같지 않다.**

배치 크기 B, `d>5`인 표본 수 v라면, 각 Q head에서

\[
w_{\rm parent\ anchor}=\frac{1}{2(2B+v)},\qquad
w_{\rm parent\ split}=\frac{\lambda_{\rm split}}{2v}.
\]

여덟 길이가 동일 개수인 B=256 배치는 v=224이고,

\[
\frac{w_{\rm split}}{w_{\rm anchor}}
=\lambda_{\rm split}\frac{2B+v}{v}
=\lambda_{\rm split}\frac{23}{7}\approx3.286\lambda_{\rm split}.
\]

실제 무작위 배치는 v에 따라 달라진다. 모든 gradient의 크기가 3.286배라는 뜻은 아니다.
정규화에서 나오는 **같은 부모 출력에 대한 BCE 계수 비율**이다.

독립적으로 조절 가능한 부모 출력 하나에 anchor target y와 잘못된 split target b가 동시에
주어졌다고 가정하면 BCE 최소점은 `(a·y+c·b)/(a+c)`이다. 위 배치에서 b=0,
y=`0.99^20≈0.81791`이면 최소점은 `7y/30≈0.19084`다.
이는 실제 collapse 측정값이 아니라 auxiliary target 오류가 충분히 영향을 줄 수 있음을
보여주는 계산이다. 계수 1 자체가 버그라는 판정은 아니다.

특히 직접 anchor의 suffix는 `E(τ_R)`이지만 backup의 suffix는 `D_R(E(τ))`다.
두 code가 가까워도 해당 위치에서 Q가 정확하다는 보장은 없다. 이 값이 작아지면
직접 부모 label이 양수여도 부모 예측이 내려갈 수 있다.

이 효과를 판단하려면 **같은 표현 checkpoint에서** `split_value_weight=0`과 기본값 1을
비교하는 것이 가장 작은 대조다. 상태 독립 분해와 chunk 실행은 두 조건 모두 유지한다.
두 조건의 초기값·batch 순서·학습 예산을 맞춰야 한다.

## 6. 설계 선택이며 오류로 보지 않은 항목

- 16D와 최대 길이 40, 모든 5의 배수 길이는 초기 설정이다. 최적성 또는 임의 길이 일반화의 근거는 없다.
- 현재 cut은 앞 5 step이다. 임의 cut 분해나 결합법칙·합성 M·군 구조를 구현한 것이 아니다.
- Subgoal/global은 공유 checkpoint의 서로 다른 목표 조건부 실행이다. 별도 재학습이 필수는 아니다.
- full-observation exact-state 도달 정의는 공식 환경 성공과 다르다. 학습 BCE를 공식 성공률로 부르지 않는다.
- 두 모드는 skill 후보 32개로 같지만 subgoal 모드는 PB endpoint 후보 32개의 추가 비용이 있다.
- 길이가 긴 부모를 골라도 매번 실행은 5 step이다. 이 구조만으로 planning call이 줄어들지 않는다.
- full-plan score를 이용한 prefix 재계획은 MPC 형태의 근사 제어다. 학습 점수와 실제 재계획 정책 가치를 동일시하지 않는다.
- NLL과 Gaussian KL은 분산의 영향도 받는다. 평균 행동 RMSE와 duration-matched shuffled-code 대조를 함께 볼 필요가 있다.
- flow proposal은 가중 행동 모방 recipe이며, argmax Q 선택의 정책 개선 보장이나 OOD 배제 보장은 아니다.
- PB terminal value는 고정 학습 함수다. 특정 최적 정책의 정확한 가치로 인증한 입력이 아니다.

## 7. 실행한 검사와 재현

`tests/test_skill_planner.py`: **11 passed, 6 deprecation warnings**, CPU, 32.56초.
작은 JAX 업데이트와 모의 환경을 사용한 구현 검사이며 정식 optimizer budget과
OGBench 성능 평가로 세지 않는다.

새 수학 검사는 `computations/skill_planning_review_20261107/verify_contract.py`다.
실행 계약·runner argv·software version·입력 전후 해시·결과 해시는 같은 디렉터리의
`contract.json`, `run01/manifest.json`, `run01/result.json`에 보존했다.

- `run.status`: `completed`; manifest 검증 통과.
- exact partition: **1,764개**.
- 실제 first-hit 구현: **196개**.
- 실제 return target과 유리수 기대값: **1,764개**; 최대 float32 절대 오차 **2.56812e-7**.
- 기본 계수의 부모당 상대 비중과 위 결정론적 반례 재계산 통과.
- 새 수학 검사의 optimizer update와 environment step: **각 0**.

검사 범위는 d=5,...,40, k=5, 각 hit=0,...,d 및 미도달,
gamma∈{0.9,0.99,1}, terminal potential∈{0,0.3,1}다. 임의 신경망 또는 모든 환경의 검증이 아니다.

명령은 기존 의존성이 설치된 Python에서 repository root를 기준으로 실행한다.

```bash
JAX_PLATFORMS=cpu PYTHONPATH=. python -m pytest tests/test_skill_planner.py -q
JAX_PLATFORMS=cpu PYTHONPATH=. python computations/skill_planning_review_20261107/verify_contract.py --result /tmp/pb_skill_review_result.json
python -m ruff check .
git diff --check
```

기존 `tests/test_config.py`는 재시도했으나 `ml_collections`가 없어 collection 단계에서
실패했다. 실제 PB bundle과 OGBench archive도 이번 작업공간에 없으므로 실제 restore와
정식 GPU smoke·평가를 통과했다고 말하지 않는다. 의존성 설치를 재시도하지 않았다.

로컬 checkout은 이전 partial import commit에 새 구현이 미커밋으로 놓인 상태다.
따라서 runner의 local git commit을 원격 구현 commit으로 바꾸어 기록하지 않았다.
검토한 6개 원격 파일과 로컬의 byte 일치 확인을 별도 기록했다.

## 8. 학습 시도의 권장 범위

1. 실제 학습된 PB inference bundle을 복원하고, 기존 로컬 archive에서 두 단계의 짧은
   smoke 및 두 추론 모드의 finite 출력·shape를 확인한다. 이것은 연결 검증이다.
2. 동일한 표현 checkpoint로 split 백업 유무를 비교한다. 기본값 하나만으로 분해 백업의
   추가 이득 또는 실패 원인을 판정하지 않는다. 별도 계수 sweep은 우선 필요 없다.
3. 직접 suffix와 predicted suffix의 Q·anchor 오차, 길이별 선택 빈도, action 평균 오차와
   분산을 따로 기록한다. 현재 logger에 모든 세부 진단이 구현됐다는 뜻은 아니다.
4. Subgoal/global/원래 PB의 같은-seed 공식 task 평가를 수행한다. 작은 복원 개선을
   기능 성공으로 대신하지 않으며, 복원 개선 부재만으로 기능 평가를 금지하지 않는다.

이번 결론은 **현재 구조의 제한된 pilot은 타당하지만, 정확한 skill Bellman 모델이라는
보증은 아직 없다**는 것이다. 최저 비용의 다음 확인은 실제 bundle·archive smoke이며,
새 loss·새 selector·추가 online 수집을 자동으로 시작하지 않는다.
