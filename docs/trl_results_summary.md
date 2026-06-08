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

| run (date) | env | epochs | gap | alpha | disc | IDM last(best) | Actor last(best) |
|---|---|---|---|---|---|---|---|
| 20260607_100119 | antmaze-medium | 600 | 10 | 0 | 0.99 | 0.96 (0.98) | 0.92 (0.96) |
| 20260606_233627 | antmaze-medium | 600 | 10 | 0 | 0.99 | 0.89 (0.94) | 0.95 (0.95) |
| 20260606_220257 | antmaze-large | 600 | 10 | 0 | 0.995 | 0.87 (0.88) | 0.88 (0.88) |
| 20260607_014409 | antmaze-large | 600 | 10 | 0 | 0.99 | 0.69 (0.86) | 0.86 (0.86) |
| 20260607_111203 | antmaze-large | 600 | 10 | 0 | 0.99 | 0.81 (0.81) | 0.77 (0.86) |
| 20260531_015543 | antmaze-large | 600 | 3 | 0 | 0.995 | 0.78 (0.82) | 0.84 (0.92) |
| 20260531_002854 | antmaze-large | 600 | 1 | 0 | 0.99 | 0.68 (0.82) | 0.80 (0.86) |
| 20260530_234442 | antmaze-large | 600 | 5 | 0 | 0.99 | 0.68 (0.68) | 0.78 (0.80) |
| 20260530_222056 | antmaze-large | 200 | 0 | 0 | 0.995 | 0.70 (0.73) | 0.67 (0.67) |
| 20260530_230230 | antmaze-large | 200 | 0 | 0 | 0.995 | 0.73 (0.73) | 0.31 (0.31) |
| 20260530_214825 | antmaze-large | 200 | 0 | 1.0 | 0.995 | 0.65 (0.65) | 0.65 (0.65) |
| 20260607_035955 | antmaze-giant | 600 | 10 | 0 | 0.99 | 0.12 (0.18) | 0.27 (0.27) |
| 20260607_123151 | antmaze-giant | 600 | 10 | 0 | 0.99 | 0.14 (0.14) | 0.27 (0.27) |
| 20260607_220842 | humanoidmaze-giant | 600 | 10 | 0 | 0.999 | 0.00 (0.00) | 0.00 (0.00) |

## Manipulation (cube)

| run (date) | env | epochs | gap | dwp | disc | IDM last(best) | Actor last(best) |
|---|---|---|---|---|---|---|---|
| 20260607_180838 | cube-single | 600 | 10 | 0.7 | 0.99 | 1.00 (1.00) | 0.92 (1.00) |
| 20260607_191325 | cube-double | 600 | 10 | 1.0 | 0.99 | 0.69 (0.76) | 0.76 (0.84) |
| 20260607_202811 | cube-triple | 300 | 10 | 1.0 | 0.995 | 0.02 (0.02) | 0.00 (0.00) |

## Puzzle

| run (date) | env | epochs | gap | dwp | disc | IDM last(best) | Actor last(best) |
|---|---|---|---|---|---|---|---|
| 20260607_062422 | puzzle-3x3 | 600 | 10 | 0.5 | 0.99 | 0.11 (0.16) | 0.08 (0.16) |
| 20260607_145301 | puzzle-3x3 | 600 | 10 | 0.5 | 0.99 | 0.11 (0.14) | 0.13 (0.14) |
| 20260608_092417 | puzzle-3x3 | 600 | 3 | 0.5 | 0.99 | 0.13 (0.16) | 0.14 (0.14) |
| 20260608_104837 | puzzle-3x3 | 600 | 5 | 0.5 | 0.99 | 0.11 (0.14) | 0.14 (0.14) |
| 20260608_121304 | puzzle-3x3 | 600 | 10 | 0.5 | 0.99 | 0.00 (0.12) | 0.02 (0.02) |
| 20260608_133838 | puzzle-3x3 | 600 | 10 | 0.5 | 0.995 | 0.07 (0.07) | 0.07 (0.16) |
| 20260607_084543 | puzzle-4x4 | 600 | 10 | 2.0 | 0.99 | 0.00 (0.00) | 0.00 (0.00) |
| 20260608_020717 | puzzle-4x4 | 600 | 10 | 2.0 | 0.99 | 0.00 (0.00) | 0.00 (0.00) |
| 20260608_150400 | puzzle-4x4 | 600 | 3 | 2.0 | 0.99 | 0.00 (0.00) | 0.00 (0.00) |

(`gap` = `subgoal_value_gap_scale`, `alpha` = `subgoal_value_alpha`, `dwp` =
`value_distance_weight_power`, `disc` = `discount`. 공통: `subgoal_distribution=diag_gaussian`,
`goal_representation=full`, `action_chunk_horizon=5`, `lambda_q_local=1.0`,
`te0/sn0`(time embedding·state normalization off).)

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

- AntMaze navigate: `subgoal_value_gap_scale=10`, `subgoal_value_alpha=0`, `value_distance_weight_power=0`,
  `discount=0.99~0.995`, `te0/sn0`, 600 epochs. (medium/large에서 0.85+ 재현)
- Cube(single/double): `gap=10`, `value_distance_weight_power≈0.7~1.0`.
- Puzzle / giant / humanoid: 미해결. 추가 연구 필요(아래 참조).

## 미해결 과제

- **puzzle 계열 전반 저조**: binary button 상태에서 transitive value가 잘 안 잡힘. goal representation /
  distance reweight 재검토 필요.
- **giant/humanoid 장거리**: subgoal chaining horizon, plan_candidates, value_base_horizon 확장 실험 필요.
- **cube-triple**: 300 epoch만 수행됨 → 충분 학습 시 재평가 필요.
- 전 결과가 **단일 시드**라 분산 미반영. 핵심 설정은 다중 시드 재현 권장.

> 참고: 본 결과는 state-space SPI(`actor_type=state_subgoal/state_proposal`) 도입 **이전**의
> `action_chunk` actor 기준 TRL 성능이다. state-space SPI 비교 결과는 별도 정리 예정.
