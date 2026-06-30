# env_best_runs bundle

Pathbridger_flow 프로젝트에서 지금까지 수집된 Flow+TRL feval 결과 기준,
환경별 best run의 **학습 하이퍼파라미터**, **eval 설정**, **epoch 600 checkpoint**를 한곳에 모았습니다.

## 파일

| 파일 | 설명 |
| --- | --- |
| `env_best_runs_choi.csv` | 환경당 1행 flat table |
| `env_best_runs_choi.md` | 요약表 + 환경별 상세 |
| `env_best_params.json` | programmatic JSON |
| `checkpoints/<env>/` | `flags.json` + `checkpoints/{dynamics,critic,actor}/params_600.pkl` |
| `configs/<env>.yaml` | best run의 `config_used.yaml` 복사 |
| `eval/<env>_best.json` | best eval JSON 스냅샷 |

## 재생성

```bash
PYTHONPATH=.:scripts python scripts/export_env_best_runs.py
```

생성 시각: 2026-06-27T21:00:43
환경 수: 10

