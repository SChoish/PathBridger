# Pathbridger runs/ 결과 요약

자동 생성: 2026-06-27 16:53 · `scripts/summarize_runs.py`
소스: `/home/choi/Pathbridger_flow/scripts/../runs`

**포함:** TRL + DQC + Flow (TRL+Flow).

총 **79** runs — TRL **13**, DQC **16**, Flow **50**.

파라미터: **gap**, **maxgap**, **N**, **γ** · 성공률 = 마지막 `EVAL END`.
resume 로그(`run_resume*.log`)가 있으면 `run.log`에 이어 붙여 읽습니다.
전체 표 CSV: [`runs_results_total_choi.csv`](runs_results_total_choi.csv).

## 환경별 best (IDM)

### Flow

| env | gap | maxgap | N | γ | IDM | ACTOR | run_dir |
| --- | --- | --- | --- | --- | --- | --- | --- |
| antmaze-giant-navigate-v0 | 10 | 5 | 1 | 0.999 | 0.15 | 0.19 | 20260610_102004_seed0_antmaze-giant-navigate-v0 |
| antmaze-large-navigate-v0 | 10 | 5 | 1 | 0.995 | 0.66 | 0.56 | 20260610_040156_seed0_antmaze-large-navigate-v0 |
| antmaze-medium-navigate-v0 | 5 | 10 | 4 | 0.99 | 0.00 | 0.00 | 20260608_033008_seed0_antmaze-medium-navigate-v0 |
| cube-double-play-v0 | 20 | 5 | 1 | 0.99 | 0.70 | 0.70 | 20260624_051355_seed0_cube-double-play-v0 |
| cube-single-play-v0 | 20 | 5 | 1 | 0.99 | 0.68 | 0.82 | 20260624_022019_seed0_cube-single-play-v0 |
| cube-triple-play-v0 | 10 | 5 | 1 | 0.995 | 0.39 | 0.38 | 20260611_143919_seed0_cube-triple-play-v0 |
| humanoidmaze-large-navigate-v0 | 10 | 5 | 1 | 0.999 | 0.00 | 0.00 | 20260625_123122_seed0_humanoidmaze-large-navigate-v0 |
| humanoidmaze-medium-navigate-v0 | 10 | 5 | 1 | 0.999 | 0.00 | 0.02 | 20260626_160557_seed0_humanoidmaze-medium-navigate-v0 |
| puzzle-3x3-play-v0 | 5 | 5 | 1 | 0.99 | 0.44 | 0.54 | 20260619_021654_seed0_puzzle-3x3-play-v0 |
| puzzle-4x4-play-v0 | 5 | 5 | 1 | 0.99 | 0.54 | 0.54 | 20260619_113125_seed0_puzzle-4x4-play-v0 |

### TRL

| env | gap | maxgap | N | γ | IDM | ACTOR | run_dir |
| --- | --- | --- | --- | --- | --- | --- | --- |
| antmaze-giant-navigate-v0 | 10 | 5 | 4 | 0.98 | - | - | 20260608_093354_seed0_antmaze-giant-navigate-v0 |
| cube-triple-play-v0 | 1 | 10 | 1 | 0.999 | 0.22 | 0.22 | 20260609_144912_seed0_cube-triple-play-v0 |
| puzzle-3x3-play-v0 | 1 | 5 | 1 | 0.999 | 0.09 | 0.05 | 20260608_110811_seed0_puzzle-3x3-play-v0 |
| puzzle-4x4-play-v0 | 1 | 5 | 1 | 0.999 | 0.00 | 0.00 | 20260608_193247_seed0_puzzle-4x4-play-v0 |

### DQC

| env | gap | maxgap | N | γ | IDM | ACTOR | run_dir |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cube-double-play-v0 | 5 | 0 | 1 | 0.99 | 0.80 | 0.76 | 20260525_141920_seed0_cube-double-play-v0 |
| cube-single-play-v0 | 5 | 0 | 1 | 0.99 | 1.00 | 0.97 | 20260526_161909_seed0_cube-single-play-v0 |

## Flow

| run_dir | algo | env | ep | gap | maxgap | N | γ | eval_ep | IDM | ACTOR | done |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 20260607_181714_seed0_antmaze-giant-navigate-v0 | Flow | antmaze-giant-navigate-v0 | 400 | 5 | 10 | 4 | 0.99 | 400 | 0.00 | 0.00 | ✅ |
| 20260607_225528_seed0_antmaze-large-navigate-v0 | Flow | antmaze-large-navigate-v0 | 400 | 5 | 10 | 4 | 0.99 | 400 | 0.31 | 0.12 | ✅ |
| 20260608_033008_seed0_antmaze-medium-navigate-v0 | Flow | antmaze-medium-navigate-v0 | 400 | 5 | 10 | 4 | 0.99 | 400 | 0.00 | 0.00 | ✅ |
| 20260608_080608_seed0_cube-double-play-v0 | Flow | cube-double-play-v0 | 400 | 5 | 10 | 4 | 0.99 | 100 | 0.00 | 0.00 | ⏳ |
| 20260609_235715_seed0_antmaze-giant-navigate-v0 | Flow | antmaze-giant-navigate-v0 | 600 | 10 | 5 | 1 | 0.999 | 600 | 0.12 | 0.19 | ✅ |
| 20260610_040156_seed0_antmaze-large-navigate-v0 | Flow | antmaze-large-navigate-v0 | 600 | 10 | 5 | 1 | 0.995 | 600 | 0.66 | 0.56 | ✅ |
| 20260610_075615_seed0_cube-double-play-v0 | Flow | cube-double-play-v0 | 600 | 10 | 5 | 1 | 0.99 | 300 | 0.40 | 0.50 | ⏳ |
| 20260610_102004_seed0_antmaze-giant-navigate-v0 | Flow | antmaze-giant-navigate-v0 | 600 | 10 | 5 | 1 | 0.999 | 600 | 0.15 | 0.19 | ✅ |
| 20260610_195823_seed0_antmaze-large-navigate-v0 | Flow | antmaze-large-navigate-v0 | 600 | 10 | 5 | 1 | 0.995 | 600 | 0.49 | 0.60 | ✅ |
| 20260611_052711_seed0_cube-double-play-v0 | Flow | cube-double-play-v0 | 600 | 10 | 5 | 1 | 0.99 | 600 | 0.67 | 0.62 | ✅ |
| 20260611_143919_seed0_cube-triple-play-v0 | Flow | cube-triple-play-v0 | 600 | 10 | 5 | 1 | 0.995 | 600 | 0.39 | 0.38 | ✅ |
| 20260612_050602_seed0_puzzle-3x3-play-v0 | Flow | puzzle-3x3-play-v0 | 600 | 10 | 5 | 1 | 0.99 | 600 | 0.28 | 0.29 | ✅ |
| 20260612_100719_seed0_puzzle-4x4-play-v0 | Flow | puzzle-4x4-play-v0 | 600 | 10 | 5 | 1 | 0.995 | 600 | 0.15 | 0.24 | ✅ |
| 20260617_232434_seed0_cube-double-play-v0 | Flow | cube-double-play-v0 | 600 | 10 | 5 | 1 | 0.99 |  |  | - | ⏳ |
| 20260617_233620_seed0_cube-double-play-v0 | Flow | cube-double-play-v0 | 600 | 10 | 5 | 1 | 0.99 | 500 | 0.56 | 0.58 | ✅ |
| 20260618_025240_seed0_cube-triple-play-v0 | Flow | cube-triple-play-v0 | 600 | 10 | 5 | 1 | 0.995 | 500 | 0.14 | 0.20 | ✅ |
| 20260618_201259_seed0_puzzle-3x3-play-v0 | Flow | puzzle-3x3-play-v0 | 600 | 5 | 5 | 1 | 0.99 |  |  | - | ⏳ |
| 20260618_201856_seed0_puzzle-3x3-play-v0 | Flow | puzzle-3x3-play-v0 | 600 | 5 | 5 | 1 | 0.99 | 500 | 0.42 | 0.28 | ✅ |
| 20260619_021654_seed0_puzzle-3x3-play-v0 | Flow | puzzle-3x3-play-v0 | 600 | 5 | 5 | 1 | 0.99 | 500 | 0.44 | 0.54 | ✅ |
| 20260619_113125_seed0_puzzle-4x4-play-v0 | Flow | puzzle-4x4-play-v0 | 600 | 5 | 5 | 1 | 0.99 | 500 | 0.54 | 0.54 | ⏳ |
| 20260619_173141_seed0_puzzle-3x3-play-v0 | Flow | puzzle-3x3-play-v0 | 600 | 5 | 5 | 1 | 0.99 | 500 | 0.36 | 0.32 | ⏳ |
| 20260619_225308_seed0_puzzle-3x3-play-v0 | Flow | puzzle-3x3-play-v0 | 600 | 5 | 5 | 1 | 0.99 | 500 | 0.12 | 0.06 | ✅ |
| 20260620_025143_seed0_puzzle-3x3-play-v0 | Flow | puzzle-3x3-play-v0 | 600 | 5 | 5 | 1 | 0.99 | 500 | 0.08 | 0.12 | ✅ |
| 20260620_071224_seed0_puzzle-4x4-play-v0 | Flow | puzzle-4x4-play-v0 | 600 | 5 | 5 | 1 | 0.99 | 500 | 0.00 | 0.02 | ✅ |
| 20260620_112559_seed0_puzzle-4x4-play-v0 | Flow | puzzle-4x4-play-v0 | 600 | 5 | 5 | 1 | 0.99 | 500 | 0.00 | 0.02 | ✅ |
| 20260620_160219_seed0_cube-single-play-v0 | Flow | cube-single-play-v0 | 600 | 10 | 5 | 1 | 0.99 |  |  | - | ⏳ |
| 20260620_161231_seed0_puzzle-3x3-play-v0 | Flow | puzzle-3x3-play-v0 | 600 | 5 | 5 | 1 | 0.99 | 200 | 0.20 | 0.16 | ⏳ |
| 20260620_182946_seed0_puzzle-3x3-play-v0 | Flow | puzzle-3x3-play-v0 | 600 | 5 | 5 | 1 | 0.99 | 500 | 0.02 | 0.08 | ✅ |
| 20260620_215024_seed0_puzzle-4x4-play-v0 | Flow | puzzle-4x4-play-v0 | 600 | 5 | 5 | 1 | 0.99 | 500 | 0.00 | 0.00 | ✅ |
| 20260621_012515_seed0_cube-single-play-v0 | Flow | cube-single-play-v0 | 600 | 10 | 5 | 1 | 0.99 | 500 | 0.62 | 0.58 | ✅ |
| 20260621_041808_seed0_cube-double-play-v0 | Flow | cube-double-play-v0 | 600 | 10 | 5 | 1 | 0.99 | 500 | 0.38 | 0.26 | ✅ |
| 20260621_072838_seed0_cube-triple-play-v0 | Flow | cube-triple-play-v0 | 600 | 10 | 5 | 1 | 0.995 | 500 | 0.06 | 0.06 | ✅ |
| 20260621_155942_seed0_antmaze-medium-navigate-v0 | Flow | antmaze-medium-navigate-v0 | 600 | 5 | 5 | 1 | 0.99 | 500 | 0.00 | 0.02 | ✅ |
| 20260621_191938_seed0_antmaze-large-navigate-v0 | Flow | antmaze-large-navigate-v0 | 600 | 5 | 5 | 1 | 0.995 | 500 | 0.00 | 0.00 | ✅ |
| 20260621_223959_seed0_antmaze-giant-navigate-v0 | Flow | antmaze-giant-navigate-v0 | 600 | 5 | 5 | 1 | 0.99 | 500 | 0.00 | 0.00 | ✅ |
| 20260623_225006_seed0_puzzle-3x3-play-v0 | Flow | puzzle-3x3-play-v0 | 600 | 0.5 | 5 | 1 | 0.99 | 500 | 0.04 | 0.04 | ✅ |
| 20260624_022019_seed0_cube-single-play-v0 | Flow | cube-single-play-v0 | 600 | 20 | 5 | 1 | 0.99 | 500 | 0.68 | 0.82 | ✅ |
| 20260624_051355_seed0_cube-double-play-v0 | Flow | cube-double-play-v0 | 600 | 20 | 5 | 1 | 0.99 | 500 | 0.70 | 0.70 | ✅ |
| 20260624_082614_seed0_cube-triple-play-v0 | Flow | cube-triple-play-v0 | 600 | 20 | 5 | 1 | 0.995 | 500 | 0.32 | 0.30 | ✅ |
| 20260624_165737_seed0_antmaze-large-navigate-v0 | Flow | antmaze-large-navigate-v0 | 600 | 20 | 5 | 1 | 0.99 | 500 | 0.28 | 0.24 | ✅ |
| 20260625_000429_seed0_antmaze-large-navigate-v0 | Flow | antmaze-large-navigate-v0 | 600 | 10 | 5 | 1 | 0.99 |  |  | - | ⏳ |
| 20260625_000705_seed0_humanoidmaze-large-navigate-v0 | Flow | humanoidmaze-large-navigate-v0 | 600 | 10 | 5 | 1 | 0.999 |  |  | - | ⏳ |
| 20260625_035918_seed0_humanoidmaze-large-navigate-v0 | Flow | humanoidmaze-large-navigate-v0 | 600 | 10 | 5 | 1 | 0.999 |  |  | - | ⏳ |
| 20260625_123122_seed0_humanoidmaze-large-navigate-v0 | Flow | humanoidmaze-large-navigate-v0 | 600 | 10 | 5 | 1 | 0.999 | 500 | 0.00 | 0.00 | ✅ |
| 20260625_192606_seed0_humanoidmaze-large-navigate-v0 | Flow | humanoidmaze-large-navigate-v0 | 600 | 10 | 5 | 1 | 0.999 | 500 | 0.00 | 0.00 | ✅ |
| 20260626_022012_seed0_humanoidmaze-large-navigate-v0 | Flow | humanoidmaze-large-navigate-v0 | 600 | 5 | 5 | 1 | 0.999 | 500 | 0.00 | 0.00 | ✅ |
| 20260626_091305_seed0_humanoidmaze-large-navigate-v0 | Flow | humanoidmaze-large-navigate-v0 | 600 | 5 | 5 | 1 | 0.999 | 500 | 0.00 | 0.00 | ✅ |
| 20260626_160557_seed0_humanoidmaze-medium-navigate-v0 | Flow | humanoidmaze-medium-navigate-v0 | 600 | 10 | 5 | 1 | 0.999 | 500 | 0.00 | 0.02 | ✅ |
| 20260627_065306_seed0_humanoidmaze-medium-navigate-v0 | Flow | humanoidmaze-medium-navigate-v0 | 600 | 10 | 5 | 1 | 0.999 | 500 | 0.00 | 0.00 | ✅ |
| 20260627_155820_seed0_humanoidmaze-medium-navigate-v0 | Flow | humanoidmaze-medium-navigate-v0 | 600 | 5 | 5 | 1 | 0.999 |  |  | - | ⏳ |

## TRL

| run_dir | algo | env | ep | gap | maxgap | N | γ | eval_ep | IDM | ACTOR | done |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 20260608_093354_seed0_antmaze-giant-navigate-v0 | TRL | antmaze-giant-navigate-v0 | 400 | 10 | 5 | 4 | 0.98 |  |  | - | ⏳ |
| 20260608_110811_seed0_puzzle-3x3-play-v0 | TRL | puzzle-3x3-play-v0 | 600 | 1 | 5 | 1 | 0.999 | 600 | 0.09 | 0.05 | ✅ |
| 20260608_131431_seed0_puzzle-3x3-play-v0 | TRL | puzzle-3x3-play-v0 | 600 | 1 | 5 | 1 | 0.995 | 600 | 0.07 | 0.05 | ✅ |
| 20260608_152048_seed0_puzzle-3x3-play-v0 | TRL | puzzle-3x3-play-v0 | 600 | 1 | 10 | 1 | 0.999 | 600 | 0.06 | 0.10 | ✅ |
| 20260608_172657_seed0_puzzle-3x3-play-v0 | TRL | puzzle-3x3-play-v0 | 600 | 1 | 10 | 1 | 0.995 | 600 | 0.07 | 0.02 | ✅ |
| 20260608_193247_seed0_puzzle-4x4-play-v0 | TRL | puzzle-4x4-play-v0 | 600 | 1 | 5 | 1 | 0.999 | 600 | 0.00 | 0.00 | ✅ |
| 20260608_214627_seed0_puzzle-4x4-play-v0 | TRL | puzzle-4x4-play-v0 | 600 | 1 | 5 | 1 | 0.995 | 600 | 0.00 | 0.00 | ✅ |
| 20260609_000006_seed0_puzzle-4x4-play-v0 | TRL | puzzle-4x4-play-v0 | 600 | 1 | 10 | 1 | 0.999 | 600 | 0.00 | 0.00 | ✅ |
| 20260609_021354_seed0_puzzle-4x4-play-v0 | TRL | puzzle-4x4-play-v0 | 600 | 1 | 10 | 1 | 0.995 | 600 | 0.00 | 0.00 | ✅ |
| 20260609_042719_seed0_cube-triple-play-v0 | TRL | cube-triple-play-v0 | 600 | 1 | 5 | 1 | 0.999 | 600 | 0.18 | 0.23 | ✅ |
| 20260609_093823_seed0_cube-triple-play-v0 | TRL | cube-triple-play-v0 | 600 | 1 | 5 | 1 | 0.995 | 600 | 0.21 | 0.22 | ✅ |
| 20260609_144912_seed0_cube-triple-play-v0 | TRL | cube-triple-play-v0 | 600 | 1 | 10 | 1 | 0.999 | 600 | 0.22 | 0.22 | ✅ |
| 20260609_195951_seed0_cube-triple-play-v0 | TRL | cube-triple-play-v0 | 600 | 1 | 10 | 1 | 0.995 | 300 | 0.00 | 0.00 | ⏳ |

## DQC

| run_dir | algo | env | ep | gap | maxgap | N | γ | eval_ep | IDM | ACTOR | done |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 20260525_141920_seed0_cube-double-play-v0 | DQC | cube-double-play-v0 | 400 | 5 | 0 | 1 | 0.99 | 400 | 0.80 | 0.76 | ✅ |
| 20260525_163103_seed0_cube-double-play-v0 | DQC | cube-double-play-v0 | 400 | 5 | 0 | 1 | 0.99 | 400 | 0.27 | 0.33 | ✅ |
| 20260525_184554_seed0_cube-double-play-v0 | DQC | cube-double-play-v0 | 400 | 5 | 0 | 1 | 0.99 | 400 | 0.76 | 0.66 | ✅ |
| 20260525_205728_seed0_cube-double-play-v0 | DQC | cube-double-play-v0 | 400 | 5 | 0 | 1 | 0.99 | 400 | 0.22 | 0.32 | ✅ |
| 20260525_231150_seed0_cube-double-play-v0 | DQC | cube-double-play-v0 | 400 | 5 | 0 | 1 | 0.99 | 400 | 0.76 | 0.67 | ✅ |
| 20260526_012346_seed0_cube-double-play-v0 | DQC | cube-double-play-v0 | 400 | 5 | 0 | 1 | 0.99 | 400 | 0.21 | 0.25 | ✅ |
| 20260526_033907_seed0_cube-double-play-v0 | DQC | cube-double-play-v0 | 400 | 5 | 0 | 1 | 0.99 | 400 | 0.78 | 0.72 | ✅ |
| 20260526_055032_seed0_cube-double-play-v0 | DQC | cube-double-play-v0 | 400 | 5 | 0 | 1 | 0.99 | 400 | 0.13 | 0.11 | ✅ |
| 20260526_080605_seed0_cube-single-play-v0 | DQC | cube-single-play-v0 | 400 | 1 | 0 | 1 | 0.99 | 400 | 0.98 | 0.80 | ✅ |
| 20260526_100904_seed0_cube-single-play-v0 | DQC | cube-single-play-v0 | 400 | 1 | 0 | 1 | 0.99 | 400 | 0.94 | 0.76 | ✅ |
| 20260526_121232_seed0_cube-single-play-v0 | DQC | cube-single-play-v0 | 400 | 1 | 0 | 1 | 0.99 | 400 | 0.98 | 0.90 | ✅ |
| 20260526_141549_seed0_cube-single-play-v0 | DQC | cube-single-play-v0 | 400 | 1 | 0 | 1 | 0.99 | 400 | 0.98 | 0.87 | ✅ |
| 20260526_161909_seed0_cube-single-play-v0 | DQC | cube-single-play-v0 | 400 | 5 | 0 | 1 | 0.99 | 400 | 1.00 | 0.97 | ✅ |
| 20260526_182231_seed0_cube-single-play-v0 | DQC | cube-single-play-v0 | 400 | 5 | 0 | 1 | 0.99 | 400 | 1.00 | 0.88 | ✅ |
| 20260526_202530_seed0_cube-single-play-v0 | DQC | cube-single-play-v0 | 400 | 5 | 0 | 1 | 0.99 | 400 | 1.00 | 0.89 | ✅ |
| 20260526_222826_seed0_cube-single-play-v0 | DQC | cube-single-play-v0 | 400 | 5 | 0 | 1 | 0.99 | 400 | 1.00 | 0.89 | ✅ |

## 환경별 상세 (task별)

### antmaze-giant-navigate-v0

| run_dir | algo | gap | maxgap | N | γ | eval_ep | IDM | ACTOR | tasks IDM | tasks ACTOR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 20260610_102004_seed0_antmaze-giant-navigate-v0 | Flow | 10 | 5 | 1 | 0.999 | 600 | 0.15 | 0.19 | 0.04,0.24,0.00,0.12,0.36 | 0.04,0.16,0.24,0.04,0.48 |
| 20260609_235715_seed0_antmaze-giant-navigate-v0 | Flow | 10 | 5 | 1 | 0.999 | 600 | 0.12 | 0.19 | 0.04,0.12,0.08,0.04,0.32 | 0.00,0.16,0.12,0.28,0.40 |
| 20260607_181714_seed0_antmaze-giant-navigate-v0 | Flow | 5 | 10 | 4 | 0.99 | 400 | 0.00 | 0.00 | 0.00,0.00,0.00,0.00,0.00 | 0.00,0.00,0.00,0.00,0.00 |
| 20260621_223959_seed0_antmaze-giant-navigate-v0 | Flow | 5 | 5 | 1 | 0.99 | 500 | 0.00 | 0.00 | 0.00,0.00,0.00,0.00,0.00 | 0.00,0.00,0.00,0.00,0.00 |
| 20260608_093354_seed0_antmaze-giant-navigate-v0 | TRL | 10 | 5 | 4 | 0.98 |  |  | - |  |  |

### antmaze-large-navigate-v0

| run_dir | algo | gap | maxgap | N | γ | eval_ep | IDM | ACTOR | tasks IDM | tasks ACTOR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 20260610_040156_seed0_antmaze-large-navigate-v0 | Flow | 10 | 5 | 1 | 0.995 | 600 | 0.66 | 0.56 | 0.52,0.68,0.96,0.56,0.56 | 0.32,0.44,0.96,0.52,0.56 |
| 20260610_195823_seed0_antmaze-large-navigate-v0 | Flow | 10 | 5 | 1 | 0.995 | 600 | 0.49 | 0.60 | 0.28,0.44,0.80,0.48,0.44 | 0.48,0.40,0.84,0.72,0.56 |
| 20260607_225528_seed0_antmaze-large-navigate-v0 | Flow | 5 | 10 | 4 | 0.99 | 400 | 0.31 | 0.12 | 0.26,0.12,0.70,0.30,0.16 | 0.12,0.00,0.34,0.06,0.08 |
| 20260624_165737_seed0_antmaze-large-navigate-v0 | Flow | 20 | 5 | 1 | 0.99 | 500 | 0.28 | 0.24 | 0.50,0.30,0.40,0.20,0.00 | 0.30,0.20,0.30,0.10,0.30 |
| 20260621_191938_seed0_antmaze-large-navigate-v0 | Flow | 5 | 5 | 1 | 0.995 | 500 | 0.00 | 0.00 | 0.00,0.00,0.00,0.00,0.00 | 0.00,0.00,0.00,0.00,0.00 |
| 20260625_000429_seed0_antmaze-large-navigate-v0 | Flow | 10 | 5 | 1 | 0.99 |  |  | - |  |  |

### antmaze-medium-navigate-v0

| run_dir | algo | gap | maxgap | N | γ | eval_ep | IDM | ACTOR | tasks IDM | tasks ACTOR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 20260608_033008_seed0_antmaze-medium-navigate-v0 | Flow | 5 | 10 | 4 | 0.99 | 400 | 0.00 | 0.00 | 0.00,0.00,0.00,0.00,0.00 | 0.00,0.00,0.00,0.00,0.00 |
| 20260621_155942_seed0_antmaze-medium-navigate-v0 | Flow | 5 | 5 | 1 | 0.99 | 500 | 0.00 | 0.02 | 0.00,0.00,0.00,0.00,0.00 | 0.00,0.00,0.10,0.00,0.00 |

### cube-double-play-v0

| run_dir | algo | gap | maxgap | N | γ | eval_ep | IDM | ACTOR | tasks IDM | tasks ACTOR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 20260525_141920_seed0_cube-double-play-v0 | DQC | 5 | 0 | 1 | 0.99 | 400 | 0.80 | 0.76 | 1.00,0.92,0.96,0.26,0.88 | 0.98,0.94,0.86,0.32,0.70 |
| 20260526_033907_seed0_cube-double-play-v0 | DQC | 5 | 0 | 1 | 0.99 | 400 | 0.78 | 0.72 | 1.00,1.00,0.98,0.20,0.74 | 1.00,0.84,0.88,0.26,0.60 |
| 20260525_184554_seed0_cube-double-play-v0 | DQC | 5 | 0 | 1 | 0.99 | 400 | 0.76 | 0.66 | 1.00,0.96,0.98,0.14,0.72 | 0.98,0.86,0.82,0.32,0.32 |
| 20260525_231150_seed0_cube-double-play-v0 | DQC | 5 | 0 | 1 | 0.99 | 400 | 0.76 | 0.67 | 1.00,0.96,0.98,0.32,0.56 | 0.98,0.72,0.86,0.36,0.42 |
| 20260624_051355_seed0_cube-double-play-v0 | Flow | 20 | 5 | 1 | 0.99 | 500 | 0.70 | 0.70 | 1.00,0.60,0.80,0.30,0.80 | 1.00,0.70,0.90,0.00,0.90 |
| 20260611_052711_seed0_cube-double-play-v0 | Flow | 10 | 5 | 1 | 0.99 | 600 | 0.67 | 0.62 | 0.96,0.72,0.68,0.28,0.72 | 0.88,0.76,0.72,0.20,0.56 |
| 20260617_233620_seed0_cube-double-play-v0 | Flow | 10 | 5 | 1 | 0.99 | 500 | 0.56 | 0.58 | 0.80,0.50,0.70,0.00,0.80 | 0.80,0.50,0.60,0.30,0.70 |
| 20260610_075615_seed0_cube-double-play-v0 | Flow | 10 | 5 | 1 | 0.99 | 300 | 0.40 | 0.50 | 0.90,0.30,0.40,0.20,0.20 | 0.90,0.60,0.20,0.30,0.50 |
| 20260621_041808_seed0_cube-double-play-v0 | Flow | 10 | 5 | 1 | 0.99 | 500 | 0.38 | 0.26 | 0.90,0.20,0.30,0.20,0.30 | 0.60,0.30,0.20,0.00,0.20 |
| 20260525_163103_seed0_cube-double-play-v0 | DQC | 5 | 0 | 1 | 0.99 | 400 | 0.27 | 0.33 | 0.70,0.26,0.26,0.10,0.04 | 0.82,0.34,0.24,0.14,0.12 |
| 20260525_205728_seed0_cube-double-play-v0 | DQC | 5 | 0 | 1 | 0.99 | 400 | 0.22 | 0.32 | 0.52,0.30,0.18,0.00,0.12 | 0.76,0.36,0.24,0.10,0.12 |
| 20260526_012346_seed0_cube-double-play-v0 | DQC | 5 | 0 | 1 | 0.99 | 400 | 0.21 | 0.25 | 0.60,0.12,0.24,0.02,0.06 | 0.80,0.16,0.20,0.06,0.02 |
| 20260526_055032_seed0_cube-double-play-v0 | DQC | 5 | 0 | 1 | 0.99 | 400 | 0.13 | 0.11 | 0.36,0.10,0.14,0.04,0.02 | 0.38,0.12,0.00,0.02,0.02 |
| 20260608_080608_seed0_cube-double-play-v0 | Flow | 5 | 10 | 4 | 0.99 | 100 | 0.00 | 0.00 | 0.00,0.00,0.00,0.00,0.00 | 0.00,0.00,0.00,0.00,0.00 |
| 20260617_232434_seed0_cube-double-play-v0 | Flow | 10 | 5 | 1 | 0.99 |  |  | - |  |  |

### cube-single-play-v0

| run_dir | algo | gap | maxgap | N | γ | eval_ep | IDM | ACTOR | tasks IDM | tasks ACTOR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 20260526_161909_seed0_cube-single-play-v0 | DQC | 5 | 0 | 1 | 0.99 | 400 | 1.00 | 0.97 | 1.00,1.00,1.00,1.00,1.00 | 0.96,0.90,1.00,1.00,1.00 |
| 20260526_182231_seed0_cube-single-play-v0 | DQC | 5 | 0 | 1 | 0.99 | 400 | 1.00 | 0.88 | 1.00,1.00,1.00,1.00,1.00 | 0.82,0.86,0.96,0.90,0.84 |
| 20260526_202530_seed0_cube-single-play-v0 | DQC | 5 | 0 | 1 | 0.99 | 400 | 1.00 | 0.89 | 1.00,1.00,1.00,1.00,1.00 | 0.92,0.86,1.00,0.96,0.72 |
| 20260526_222826_seed0_cube-single-play-v0 | DQC | 5 | 0 | 1 | 0.99 | 400 | 1.00 | 0.89 | 1.00,0.98,1.00,1.00,1.00 | 0.94,0.94,0.94,0.86,0.76 |
| 20260526_080605_seed0_cube-single-play-v0 | DQC | 1 | 0 | 1 | 0.99 | 400 | 0.98 | 0.80 | 1.00,1.00,0.98,0.94,1.00 | 0.78,0.74,0.88,0.82,0.80 |
| 20260526_121232_seed0_cube-single-play-v0 | DQC | 1 | 0 | 1 | 0.99 | 400 | 0.98 | 0.90 | 1.00,1.00,0.92,1.00,0.98 | 0.94,0.78,0.98,0.90,0.88 |
| 20260526_141549_seed0_cube-single-play-v0 | DQC | 1 | 0 | 1 | 0.99 | 400 | 0.98 | 0.87 | 1.00,0.98,0.98,0.94,1.00 | 0.96,0.84,0.90,0.84,0.82 |
| 20260526_100904_seed0_cube-single-play-v0 | DQC | 1 | 0 | 1 | 0.99 | 400 | 0.94 | 0.76 | 0.98,1.00,0.86,0.98,0.88 | 0.88,0.68,0.74,0.84,0.66 |
| 20260624_022019_seed0_cube-single-play-v0 | Flow | 20 | 5 | 1 | 0.99 | 500 | 0.68 | 0.82 | 0.70,0.60,0.80,0.60,0.70 | 0.80,0.90,0.80,0.80,0.80 |
| 20260621_012515_seed0_cube-single-play-v0 | Flow | 10 | 5 | 1 | 0.99 | 500 | 0.62 | 0.58 | 0.80,0.80,0.60,0.30,0.60 | 0.70,0.70,0.60,0.60,0.30 |
| 20260620_160219_seed0_cube-single-play-v0 | Flow | 10 | 5 | 1 | 0.99 |  |  | - |  |  |

### cube-triple-play-v0

| run_dir | algo | gap | maxgap | N | γ | eval_ep | IDM | ACTOR | tasks IDM | tasks ACTOR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 20260611_143919_seed0_cube-triple-play-v0 | Flow | 10 | 5 | 1 | 0.995 | 600 | 0.39 | 0.38 | 0.88,0.44,0.44,0.12,0.08 | 0.88,0.48,0.52,0.00,0.04 |
| 20260624_082614_seed0_cube-triple-play-v0 | Flow | 20 | 5 | 1 | 0.995 | 500 | 0.32 | 0.30 | 1.00,0.20,0.40,0.00,0.00 | 0.80,0.20,0.40,0.00,0.10 |
| 20260609_144912_seed0_cube-triple-play-v0 | TRL | 1 | 10 | 1 | 0.999 | 600 | 0.22 | 0.22 | 0.92,0.08,0.04,0.04,0.00 | 1.00,0.04,0.00,0.04,0.00 |
| 20260609_093823_seed0_cube-triple-play-v0 | TRL | 1 | 5 | 1 | 0.995 | 600 | 0.21 | 0.22 | 1.00,0.00,0.00,0.04,0.00 | 1.00,0.00,0.08,0.00,0.00 |
| 20260609_042719_seed0_cube-triple-play-v0 | TRL | 1 | 5 | 1 | 0.999 | 600 | 0.18 | 0.23 | 0.88,0.00,0.04,0.00,0.00 | 0.96,0.16,0.04,0.00,0.00 |
| 20260618_025240_seed0_cube-triple-play-v0 | Flow | 10 | 5 | 1 | 0.995 | 500 | 0.14 | 0.20 | 0.40,0.10,0.20,0.00,0.00 | 0.80,0.00,0.20,0.00,0.00 |
| 20260621_072838_seed0_cube-triple-play-v0 | Flow | 10 | 5 | 1 | 0.995 | 500 | 0.06 | 0.06 | 0.20,0.00,0.10,0.00,0.00 | 0.30,0.00,0.00,0.00,0.00 |
| 20260609_195951_seed0_cube-triple-play-v0 | TRL | 1 | 10 | 1 | 0.995 | 300 | 0.00 | 0.00 | 0.00,0.00,0.00,0.00,0.00 | 0.00,0.00,0.00,0.00,0.00 |

### humanoidmaze-large-navigate-v0

| run_dir | algo | gap | maxgap | N | γ | eval_ep | IDM | ACTOR | tasks IDM | tasks ACTOR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 20260625_123122_seed0_humanoidmaze-large-navigate-v0 | Flow | 10 | 5 | 1 | 0.999 | 500 | 0.00 | 0.00 | 0.00,0.00,0.00,0.00,0.00 | 0.00,0.00,0.00,0.00,0.00 |
| 20260625_192606_seed0_humanoidmaze-large-navigate-v0 | Flow | 10 | 5 | 1 | 0.999 | 500 | 0.00 | 0.00 | 0.00,0.00,0.00,0.00,0.00 | 0.00,0.00,0.00,0.00,0.00 |
| 20260626_022012_seed0_humanoidmaze-large-navigate-v0 | Flow | 5 | 5 | 1 | 0.999 | 500 | 0.00 | 0.00 | 0.00,0.00,0.00,0.00,0.00 | 0.00,0.00,0.00,0.00,0.00 |
| 20260626_091305_seed0_humanoidmaze-large-navigate-v0 | Flow | 5 | 5 | 1 | 0.999 | 500 | 0.00 | 0.00 | 0.00,0.00,0.00,0.00,0.00 | 0.00,0.00,0.00,0.00,0.00 |
| 20260625_000705_seed0_humanoidmaze-large-navigate-v0 | Flow | 10 | 5 | 1 | 0.999 |  |  | - |  |  |
| 20260625_035918_seed0_humanoidmaze-large-navigate-v0 | Flow | 10 | 5 | 1 | 0.999 |  |  | - |  |  |

### humanoidmaze-medium-navigate-v0

| run_dir | algo | gap | maxgap | N | γ | eval_ep | IDM | ACTOR | tasks IDM | tasks ACTOR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 20260626_160557_seed0_humanoidmaze-medium-navigate-v0 | Flow | 10 | 5 | 1 | 0.999 | 500 | 0.00 | 0.02 | 0.00,0.00,0.00,0.00,0.00 | 0.00,0.00,0.00,0.10,0.00 |
| 20260627_065306_seed0_humanoidmaze-medium-navigate-v0 | Flow | 10 | 5 | 1 | 0.999 | 500 | 0.00 | 0.00 | 0.00,0.00,0.00,0.00,0.00 | 0.00,0.00,0.00,0.00,0.00 |
| 20260627_155820_seed0_humanoidmaze-medium-navigate-v0 | Flow | 5 | 5 | 1 | 0.999 |  |  | - |  |  |

### puzzle-3x3-play-v0

| run_dir | algo | gap | maxgap | N | γ | eval_ep | IDM | ACTOR | tasks IDM | tasks ACTOR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 20260619_021654_seed0_puzzle-3x3-play-v0 | Flow | 5 | 5 | 1 | 0.99 | 500 | 0.44 | 0.54 | 0.80,0.50,0.30,0.50,0.10 | 0.90,0.40,0.40,0.50,0.50 |
| 20260618_201856_seed0_puzzle-3x3-play-v0 | Flow | 5 | 5 | 1 | 0.99 | 500 | 0.42 | 0.28 | 1.00,0.30,0.20,0.40,0.20 | 0.80,0.20,0.00,0.30,0.10 |
| 20260619_173141_seed0_puzzle-3x3-play-v0 | Flow | 5 | 5 | 1 | 0.99 | 500 | 0.36 | 0.32 | 0.80,0.50,0.00,0.20,0.30 | 0.70,0.30,0.00,0.30,0.30 |
| 20260612_050602_seed0_puzzle-3x3-play-v0 | Flow | 10 | 5 | 1 | 0.99 | 600 | 0.28 | 0.29 | 0.88,0.20,0.16,0.04,0.12 | 0.76,0.24,0.16,0.16,0.12 |
| 20260620_161231_seed0_puzzle-3x3-play-v0 | Flow | 5 | 5 | 1 | 0.99 | 200 | 0.20 | 0.16 | 0.70,0.20,0.10,0.00,0.00 | 0.50,0.10,0.10,0.00,0.10 |
| 20260619_225308_seed0_puzzle-3x3-play-v0 | Flow | 5 | 5 | 1 | 0.99 | 500 | 0.12 | 0.06 | 0.30,0.10,0.00,0.00,0.20 | 0.20,0.10,0.00,0.00,0.00 |
| 20260608_110811_seed0_puzzle-3x3-play-v0 | TRL | 1 | 5 | 1 | 0.999 | 600 | 0.09 | 0.05 | 0.36,0.04,0.00,0.00,0.04 | 0.24,0.00,0.00,0.00,0.00 |
| 20260620_025143_seed0_puzzle-3x3-play-v0 | Flow | 5 | 5 | 1 | 0.99 | 500 | 0.08 | 0.12 | 0.20,0.10,0.00,0.00,0.10 | 0.30,0.00,0.00,0.20,0.10 |
| 20260608_131431_seed0_puzzle-3x3-play-v0 | TRL | 1 | 5 | 1 | 0.995 | 600 | 0.07 | 0.05 | 0.16,0.12,0.04,0.00,0.04 | 0.16,0.04,0.00,0.04,0.00 |
| 20260608_172657_seed0_puzzle-3x3-play-v0 | TRL | 1 | 10 | 1 | 0.995 | 600 | 0.07 | 0.02 | 0.28,0.04,0.00,0.00,0.04 | 0.08,0.00,0.00,0.00,0.00 |
| 20260608_152048_seed0_puzzle-3x3-play-v0 | TRL | 1 | 10 | 1 | 0.999 | 600 | 0.06 | 0.10 | 0.32,0.00,0.00,0.00,0.00 | 0.44,0.00,0.00,0.00,0.08 |
| 20260623_225006_seed0_puzzle-3x3-play-v0 | Flow | 0.5 | 5 | 1 | 0.99 | 500 | 0.04 | 0.04 | 0.20,0.00,0.00,0.00,0.00 | 0.20,0.00,0.00,0.00,0.00 |
| 20260620_182946_seed0_puzzle-3x3-play-v0 | Flow | 5 | 5 | 1 | 0.99 | 500 | 0.02 | 0.08 | 0.10,0.00,0.00,0.00,0.00 | 0.40,0.00,0.00,0.00,0.00 |
| 20260618_201259_seed0_puzzle-3x3-play-v0 | Flow | 5 | 5 | 1 | 0.99 |  |  | - |  |  |

### puzzle-4x4-play-v0

| run_dir | algo | gap | maxgap | N | γ | eval_ep | IDM | ACTOR | tasks IDM | tasks ACTOR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 20260619_113125_seed0_puzzle-4x4-play-v0 | Flow | 5 | 5 | 1 | 0.99 | 500 | 0.54 | 0.54 | 0.90,0.30,0.60,0.50,0.40 | 0.80,0.60,0.70,0.30,0.30 |
| 20260612_100719_seed0_puzzle-4x4-play-v0 | Flow | 10 | 5 | 1 | 0.995 | 600 | 0.15 | 0.24 | 0.16,0.20,0.08,0.12,0.20 | 0.32,0.20,0.16,0.24,0.28 |
| 20260608_193247_seed0_puzzle-4x4-play-v0 | TRL | 1 | 5 | 1 | 0.999 | 600 | 0.00 | 0.00 | 0.00,0.00,0.00,0.00,0.00 | 0.00,0.00,0.00,0.00,0.00 |
| 20260608_214627_seed0_puzzle-4x4-play-v0 | TRL | 1 | 5 | 1 | 0.995 | 600 | 0.00 | 0.00 | 0.00,0.00,0.00,0.00,0.00 | 0.00,0.00,0.00,0.00,0.00 |
| 20260609_000006_seed0_puzzle-4x4-play-v0 | TRL | 1 | 10 | 1 | 0.999 | 600 | 0.00 | 0.00 | 0.00,0.00,0.00,0.00,0.00 | 0.00,0.00,0.00,0.00,0.00 |
| 20260609_021354_seed0_puzzle-4x4-play-v0 | TRL | 1 | 10 | 1 | 0.995 | 600 | 0.00 | 0.00 | 0.00,0.00,0.00,0.00,0.00 | 0.00,0.00,0.00,0.00,0.00 |
| 20260620_071224_seed0_puzzle-4x4-play-v0 | Flow | 5 | 5 | 1 | 0.99 | 500 | 0.00 | 0.02 | 0.00,0.00,0.00,0.00,0.00 | 0.10,0.00,0.00,0.00,0.00 |
| 20260620_112559_seed0_puzzle-4x4-play-v0 | Flow | 5 | 5 | 1 | 0.99 | 500 | 0.00 | 0.02 | 0.00,0.00,0.00,0.00,0.00 | 0.00,0.10,0.00,0.00,0.00 |
| 20260620_215024_seed0_puzzle-4x4-play-v0 | Flow | 5 | 5 | 1 | 0.99 | 500 | 0.00 | 0.00 | 0.00,0.00,0.00,0.00,0.00 | 0.00,0.00,0.00,0.00,0.00 |

