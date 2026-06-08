# TRL 결과 정리 (PathBridger)

본 문서는 `critic_type/algorithm = trl`(state-pair transitive V + local subgoal-conditioned Q)
구성으로 학습한 run들의 평가 결과를 정리한다. 모든 수치는 `runs/<...>/run.log`의
`env_success_rate_mean`(OGBench `info['success']` any-step 기준)에서 추출했다.

## 평가 지표

각 eval epoch마다 두 정책을 동일 성공 정의로 측정한다.

- **IDM**: dynamics가 예측한 subgoal로 bridge plan → IDM action chunk.
- **Actor**: 학습된 SPI actor의 action chunk (`actor_type=action_chunk`).

표기: `last (best)` = 마지막 eval epoch 성공률 (학습 중 최고 eval 성공률).
모두 **단일 시드(seed 0)** 결과이며 분산 추정은 없다.

## AntMaze (navigate)

IDM last 내림차순 정렬 (동점은 best 내림차순).

| run (date) | env | epochs | gap | gmax | alpha | disc | n | goal | tgt | IDM last(best) | Actor last(best) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 20260607_100119 | antmaze-medium | 600 | 10 | 5 | 0 | 0.99 | 1 | geomV | disp | 0.96 (0.98) | 0.92 (0.96) |
| 20260606_233627 | antmaze-medium | 600 | 10 | 5 | 0 | 0.99 | 4 | mix | disp | 0.89 (0.94) | 0.95 (0.95) |
| 20260606_220257 | antmaze-large | 600 | 10 | 5 | 0 | 0.995 | 1 | mix | **abs** | 0.87 (0.88) | 0.88 (0.88) |
| 20260607_111203 | antmaze-large | 600 | 10 | 5 | 0 | 0.99 | 1 | geomV | disp | 0.81 (0.81) | 0.77 (0.86) |
| 20260531_015543 | antmaze-large | 600 | 3 | 0 | 0 | 0.995 | 1 | mix | abs | 0.78 (0.82) | 0.84 (0.92) |
| 20260530_230230 | antmaze-large | 200 | 0 | 0 | 0 | 0.995 | 1 | mix | abs | 0.73 (0.73) | 0.31 (0.31) |
| 20260530_222056 | antmaze-large | 200 | 0 | 0 | 0 | 0.995 | 1 | mix | abs | 0.70 (0.73) | 0.67 (0.67) |
| 20260607_014409 | antmaze-large | 600 | 10 | 5 | 0 | 0.99 | 4 | mix | disp | 0.69 (0.86) | 0.86 (0.86) |
| 20260531_002854 | antmaze-large | 600 | 1 | 0 | 0 | 0.99 | 1 | mix | abs | 0.68 (0.82) | 0.80 (0.86) |
| 20260530_234442 | antmaze-large | 600 | 5 | 0 | 0 | 0.99 | 1 | mix | abs | 0.68 (0.68) | 0.78 (0.80) |
| 20260530_214825 | antmaze-large | 200 | 0 | 0 | 1.0 | 0.995 | 1 | mix | abs | 0.65 (0.65) | 0.65 (0.65) |
| 20260607_123151 | antmaze-giant | 600 | 10 | 5 | 0 | 0.99 | 1 | geomV | disp | 0.14 (0.14) | 0.27 (0.27) |
| 20260607_035955 | antmaze-giant | 600 | 10 | 5 | 0 | 0.99 | 4 | mix | disp | 0.12 (0.18) | 0.27 (0.27) |
| 20260607_220842 | humanoidmaze-giant | 600 | 10 | 5 | 0 | 0.999 | 1 | trajV | disp | 0.00 (0.00) | 0.00 (0.00) |

## Manipulation (cube)

| run (date) | env | epochs | gap | gmax | dwp | disc | n | goal | tgt | IDM last(best) | Actor last(best) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 20260607_180838 | cube-single | 600 | 10 | 5 | 0.7 | 0.99 | 1 | geomV | disp | 1.00 (1.00) | 0.92 (1.00) |
| 20260607_191325 | cube-double | 600 | 10 | 5 | 1.0 | 0.99 | 1 | geomV | disp | 0.69 (0.76) | 0.76 (0.84) |
| 20260607_202811 | cube-triple | 300 | 10 | 5 | 1.0 | 0.995 | 1 | geomV | disp | 0.02 (0.02) | 0.00 (0.00) |

## Puzzle

IDM last 내림차순 정렬 (동점은 best 내림차순).

| run (date) | env | epochs | gap | gmax | dwp | disc | n | goal | tgt | IDM last(best) | Actor last(best) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 20260608_092417 | puzzle-3x3 | 600 | 3 | 5 | 0.5 | 0.99 | 1 | geomV* | disp | 0.13 (0.16) | 0.14 (0.14) |
| 20260607_062422 | puzzle-3x3 | 600 | 10 | 5 | 0.5 | 0.99 | 4 | mix | disp | 0.11 (0.16) | 0.08 (0.16) |
| 20260607_145301 | puzzle-3x3 | 600 | 10 | 5 | 0.5 | 0.99 | 1 | geomV* | disp | 0.11 (0.14) | 0.13 (0.14) |
| 20260608_104837 | puzzle-3x3 | 600 | 5 | 5 | 0.5 | 0.99 | 1 | geomV* | disp | 0.11 (0.14) | 0.14 (0.14) |
| 20260608_133838 | puzzle-3x3 | 600 | 10 | 5 | 0.5 | 0.995 | 1 | geomV* | disp | 0.07 (0.07) | 0.07 (0.16) |
| 20260608_121304 | puzzle-3x3 | 600 | 10 | 3 | 0.5 | 0.99 | 1 | geomV* | disp | 0.00 (0.12) | 0.02 (0.02) |
| 20260607_084543 | puzzle-4x4 | 600 | 10 | 5 | 2.0 | 0.99 | 4 | mix | disp | 0.00 (0.00) | 0.00 (0.00) |
| 20260608_020717 | puzzle-4x4 | 600 | 10 | 5 | 2.0 | 0.99 | 1 | geomV* | disp | 0.00 (0.00) | 0.00 (0.00) |
| 20260608_150400 | puzzle-4x4 | 600 | 3 | 5 | 2.0 | 0.99 | 1 | geomV* | disp | 0.00 (0.00) | 0.00 (0.00) |

컬럼 정의:
- `gap` = `subgoal_value_gap_scale` (subgoal MSE 가중 `exp(gap·ΔV)`의 scale).
- `gmax` = `subgoal_value_weight_max` (위 exp 가중의 상한 clip; **0 = clip 없음**).
- `alpha` = `subgoal_value_alpha`, `dwp` = `value_distance_weight_power`, `disc` = `discount`.
- `n` = `subgoal_num_samples` (subgoal value 스코어링 표본 수).
- `tgt` = `subgoal_target_mode` (`abs` = absolute, `disp` = displacement).
- `goal` = value/actor goal-sampling 분포:
  - `mix` = value goal `cur0.2/rnd0.3/traj0.5`, `value_geom_sample=off` (코드 기본값; config에 키가 없던 구버전 run 포함).
  - `geomV` = `value_geom_sample=on` (기하 trajectory goal), actor `trajgoal=1.0`.
  - `geomV*` = `geomV` + actor도 `actor_geom_sample=on`, `rnd0.5/traj0.5` (puzzle 후기 run).
  - `trajV` = `value_geom_sample=off`, value `trajgoal=1.0`.

공통: `subgoal_distribution=diag_gaussian`, `goal_representation=full`, `action_chunk_horizon=5`,
`full_chunk_horizon=25`, `value_base_horizon=5`, `lambda_q_local=1.0`, `te0/sn0`(time embedding·state
normalization off), seed 0. (humanoidmaze-giant는 `batch_size=8192`, cube-triple는 `4096`, 그 외 `1024`.)

## 동일 설정처럼 보였으나 다른 점수: 세부 차이 조사

`gap/gmax/disc`만 보면 같아 보이지만 점수가 다른 쌍들을 config_used.yaml·flags.json까지
대조한 결과, 차이는 거의 항상 **`subgoal_num_samples`(n)** 와 **value/actor goal-sampling 분포
(`goal`)**, 일부는 **`subgoal_target_mode`(tgt)** 에서 나왔다.

| 쌍 (동일 gap/disc) | run A | run B | 실제 다른 파라미터 | 점수 (IDM / Actor, last(best)) |
|---|---|---|---|---|
| antmaze-large g10/d0.99/disp | 20260607_014409 | 20260607_111203 | A: n=4, mix / B: n=1, geomV | A 0.69(0.86)/0.86 vs B 0.81(0.81)/0.77(0.86) |
| antmaze-medium g10/d0.99 | 20260607_100119 | 20260606_233627 | A: n=1, geomV / B: n=4, mix | A 0.96(0.98)/0.92(0.96) vs B 0.89(0.94)/0.95 |
| antmaze-giant g10/d0.99 | 20260607_035955 | 20260607_123151 | A: n=4, mix / B: n=1, geomV | A 0.12(0.18)/0.27 vs B 0.14(0.14)/0.27 |
| puzzle-3x3 g10/dwp0.5/d0.99 | 20260607_062422 | 20260607_145301 | A: n=4, mix / B: n=1, geomV* | A 0.11(0.16)/0.08(0.16) vs B 0.11(0.14)/0.13(0.14) |
| puzzle-4x4 g10/dwp2/d0.99 | 20260607_084543 | 20260608_020717 | A: n=4, mix, **resume@200** / B: n=1, geomV* | 둘 다 0.00 |

또한 antmaze-large에서 IDM last 최고였던 `20260606_220257`(0.87)은 다른 large run과 달리
**`subgoal_target_mode=absolute` + `disc=0.995`**였다(나머지 large gap10은 displacement). 즉
large에서 `tgt=abs`도 잠재적 상승 요인일 수 있다.

정리하면 학습기에 존재하던 두 "레짐"이 표면적으로 동일 설정에 섞여 있었다:

- **n4-mix**: `subgoal_num_samples=4`, value goal = HIQL식 혼합(`cur0.2/rnd0.3/traj0.5`), `value_geom_sample=off`.
- **n1-geomV**: `subgoal_num_samples=1`, `value_geom_sample=on`(기하 trajectory goal); puzzle 후기엔 actor도 `geom rnd0.5/traj0.5`.

두 레짐의 우열은 **일관적이지 않다**. medium은 n1-geomV가 IDM 더 높고(0.96 vs 0.89), large는
n1-geomV가 last는 높지만(0.81 vs 0.69) best는 동일(0.86), Actor는 오히려 n4-mix가 안정적(0.86 vs
0.77). 단일 시드 노이즈와 샘플링 분포 차이가 섞여 있어 둘 중 하나를 명확히 권장하기 어렵다 →
핵심 설정(특히 `n`, `goal`)은 다중 시드 재현이 필요하다.

## 핵심 관찰

1. **`subgoal_value_gap_scale`(subgoal MSE의 ΔV 가중)가 antmaze-large에서 가장 영향이 컸다.**
   gap=0 (IDM ~0.70) < gap=1/5 (~0.68–0.82) < **gap=3 (Actor best 0.92)** ≈ **gap=10 (IDM 0.87, Actor 0.88)**.
   gap을 키워 "value gap이 큰(=더 유익한) subgoal"에 MSE를 집중시키는 것이 안정적으로 도움이 됐다.

2. **`subgoal_value_alpha`(TRL product bonus `α·V(s,ẑ)·V(ẑ,g)`)는 large gap=0에서 0.65로,
   gap 기반 가중 대비 뚜렷한 이득이 없었다.** alpha 단독보다는 gap 가중이 핵심 레버였다.

3. **난이도 분포가 명확하다.**
   - 잘 됨: antmaze-medium(IDM 0.96 / Actor 0.95), antmaze-large(~0.87/0.88), cube-single(1.0), cube-double(~0.76/0.84).
   - 어려움: antmaze-giant(~0.14/0.27), cube-triple(~0.02), puzzle-3x3(~0.1–0.16), puzzle-4x4(0.0), humanoidmaze-giant(0.0).
   장거리 미로(giant/humanoid)와 조합 폭발이 큰 puzzle/멀티큐브에서 TRL transitive value가 충분히 학습되지 않았다.

4. **IDM과 Actor 정책 성능은 대체로 동행**하나, antmaze-large에서는 Actor가 IDM보다 약간 높게 끝나는
   경우가 많았고(예: 0.69 vs 0.86), 반대로 cube-single에서는 IDM이 더 안정적이었다.

5. **discount**: large는 0.995/0.99 모두 유사. humanoidmaze-giant는 0.999로도 0.0 → discount 문제라기보다
   장기 신뢰 할당·planning 한계로 보인다.

## 권장 기본값 (현재까지)

- AntMaze navigate: `subgoal_value_gap_scale=10`, `subgoal_value_weight_max=5`, `subgoal_value_alpha=0`,
  `value_distance_weight_power=0`, `discount=0.99~0.995`, `te0/sn0`, 600 epochs. (medium/large에서 0.85+ 재현)
  large는 `subgoal_target_mode=absolute`가 가장 높은 IDM(0.87)을 기록.
- Cube(single/double): `gap=10`, `gmax=5`, `value_distance_weight_power≈0.7~1.0`.
- `subgoal_num_samples`(n)·`goal`(value/actor goal-sampling)은 환경별로 우열이 일관되지 않으니
  고정 전 다중 시드로 확인 권장.
- Puzzle / giant / humanoid: 미해결. 추가 연구 필요(아래 참조).

## 미해결 과제

- **puzzle 계열 전반 저조**: binary button 상태에서 transitive value가 잘 안 잡힘. goal representation /
  distance reweight 재검토 필요.
- **giant/humanoid 장거리**: subgoal chaining horizon, plan_candidates, value_base_horizon 확장 실험 필요.
- **cube-triple**: 300 epoch만 수행됨 → 충분 학습 시 재평가 필요.
- 전 결과가 **단일 시드**라 분산 미반영. 핵심 설정은 다중 시드 재현 권장.

> 참고: 본 결과는 state-space SPI(`actor_type=state_subgoal/state_proposal`) 도입 **이전**의
> `action_chunk` actor 기준 TRL 성능이다. state-space SPI 비교 결과는 별도 정리 예정.
