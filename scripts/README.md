# scripts/ 정리

이 폴더는 실험 YAML 생성, 긴 sweep 실행, checkpoint 재평가, 결과 요약을 위한 운영 스크립트를 모아 둔 곳입니다. 저장소 루트에서 실행하는 것을 기본으로 합니다.

```bash
cd /home/svcho/Pathbridger_flow
export PYTHONPATH=.
```

GPU 실행 스크립트는 보통 `GPU_ID`, `PYTHON_BIN`, `SEED` 환경변수를 읽고 로그를 `nohup_logs/`에 씁니다.

## 공통 유틸

| 파일 | 역할 |
|------|------|
| `with_jax_cuda.sh` | JAX CUDA 환경을 세팅한 뒤 명령 실행 |
| `jax_cuda_env.sh` | CUDA/JAX library path 설정 조각 |
| `yaml_run_config.py` | YAML run config를 안정적인 key order로 생성하는 builder |
| `docs_output_paths.py` | `docs/`에 생성하는 CSV/MD 파일명에 `_7ch` suffix를 붙이는 helper |

## Flow+TRL sweep

현재 주 실험군입니다. `subgoal_distribution=flow`, `critic_type=trl`, gap/wmax/N sweep을 다룹니다.

| 파일 | 역할 |
|------|------|
| `flow_trl_sweep_common.py` | Flow+TRL sweep 공통 상수와 run/eval completion helper |
| `write_flow_trl_sweep_yaml.py` | `config/sweep_flow_trl_finaleval/` YAML 생성 |
| `run_flow_trl_sweep.sh` | Flow+TRL final-eval sweep 실행 및 fallback eval |
| `write_flow_trl_puzzle_45_46_yaml.py` | puzzle 4x5/4x6 gap sweep YAML 생성 |
| `run_flow_trl_puzzle_45_46.sh` | puzzle 4x5/4x6 train+eval sweep 실행 |
| `write_flow_trl_k40_best_yaml.py` | K=40 best-param follow-up YAML 생성 |
| `run_flow_trl_k40_best.sh` | K=40 follow-up 학습/eval 실행 |

대표 실행:

```bash
GPU_ID=0 nohup bash scripts/run_flow_trl_sweep.sh > nohup_logs/flow_trl_sweep.nohup.log 2>&1 &
GPU_ID=0 nohup bash scripts/run_flow_trl_puzzle_45_46.sh > nohup_logs/flow_trl_p456.nohup.log 2>&1 &
```

## 후속 eval / rerun

| 파일 | 역할 |
|------|------|
| `run_amg_m800_eval.sh` | antmaze-giant checkpoint를 `eval_max_chunks=800`으로 재평가 |
| `run_amg_m800_then_p456.sh` | giant m800 eval 후 puzzle 4x5/4x6 sweep 이어서 실행 |

완료된 일회성 temp/gamma/N rerun helper는 제거했습니다. 필요하면 `eval_checkpoint.py`와 `summarize_feval_results.py` 조합으로 새 runner를 짧게 작성합니다.

## 결과 요약

모든 generated CSV/MD는 `docs/` 아래에 `_7ch` suffix로 씁니다.

| 파일 | 출력 |
|------|------|
| `summarize_feval_results.py` | `docs/flow_trl_feval_results_7ch.csv/.md` |
| `summarize_runs.py` | `docs/runs_results_*_7ch.*`, `docs/douri_runs_results_*_7ch.*` |

예시:

```bash
python scripts/summarize_feval_results.py
python scripts/summarize_runs.py
```

## 운영 메모

- 긴 sweep은 `nohup`으로 실행하고, 중복 process가 없는지 확인한 뒤 시작합니다.
- Runner는 가능한 경우 기존 run, gamma match, eval JSON completeness를 확인해 skip합니다.
- `docs/`에 생성되는 CSV/MD는 로컬 분석 산출물입니다. 커밋 대상인지 명시적으로 확인한 뒤 stage합니다.
