# Scripts

이 디렉터리는 현재 Flow+TRL 실험 운용에 필요한 스크립트만 유지합니다.

## 공통

| 파일 | 역할 |
| --- | --- |
| `with_jax_cuda.sh`, `jax_cuda_env.sh` | JAX CUDA library path 설정 후 명령 실행 |
| `yaml_run_config.py` | canonical YAML field order를 맞추는 run-config builder |
| `docs_output_paths.py` | generated docs CSV/MD에 `_choi` suffix를 붙이는 output path helper |

## Flow+TRL Sweep

| 파일 | 역할 |
| --- | --- |
| `flow_trl_sweep_common.py` | Flow+TRL variant/config/run-dir helper |
| `write_flow_trl_sweep_yaml.py` | legacy-compatible Flow+TRL sweep YAML 생성 |
| `generate_flow_trl_gap_tune_configs.py` | gap tune config 생성 |
| `generate_flow_trl_antmaze_dw_sweep_configs.py` | antmaze distance-weight sweep config 생성 |
| `generate_flow_trl_humanoidmaze_dw_sweep_configs.py` | humanoidmaze distance-weight sweep config 생성 |
| `generate_flow_gap10_n1_cube_configs.py` | cube gap10/w5/N1 follow-up config 생성 |
| `generate_flow_k_sweep_configs.py` | K/horizon follow-up config 생성 |
| `generate_flow_h25_ha10_configs.py` | h=25, action-horizon=10 follow-up config 생성 |

## 실행 스크립트

| 파일 | 역할 |
| --- | --- |
| `run_flow_trl_sweep.sh` | base Flow+TRL final eval sweep 실행 |
| `run_flow_trl_gap_tune.sh` | gap tune sweep 실행 |
| `run_flow_trl_antmaze_dw_sweep.sh` | antmaze distance-weight sweep 실행 |
| `run_flow_trl_humanoidmaze_dw_sweep.sh` | humanoidmaze distance-weight sweep 실행 |
| `run_flow_trl_eval_giant_cap4000.sh` | antmaze-giant full-cap eval-only pass |
| `run_flow_trl_giant_then_humanoidmaze.sh` | giant eval 후 humanoid sweep chain |
| `run_flow_h25_ha10.sh` | h=25/action-horizon=10 follow-up 실행 |
| `run_flow_k_sweep.sh`, `run_flow_k_sweep_puzzle44.sh` | K/horizon follow-up 실행 |
| `run_flow_gap10_n1_cube.sh` | cube gap10/w5/N1 follow-up 실행 |
| `run_flow_cube_vzg_eval.sh` | cube goal-value score eval helper |
| `run_flow_trl_eval_temp_p3_cd_ct.sh` | selected temp eval helper |

## 결과 요약

| 파일 | 출력 |
| --- | --- |
| `summarize_feval_results.py` | `docs/flow_trl_feval_results_choi.csv/.md` |
| `summarize_runs.py` | `docs/*runs_results*_choi.csv/.md` |
| `export_trl_completed_results.py` | `docs/trl_completed_results_choi.csv` |

## 제거된 Legacy 범위

예전 TRL tune sweep, plain Flow gap5/gap10 ablation, 완료된 일회성 temp/rerun helper는 이 로컬에서 유지하지 않습니다. 필요한 경우 git history에서 복원합니다.
