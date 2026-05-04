#!/usr/bin/env bash
# 여러 run_config 를 **한 번에 하나씩** 끝까지 학습한다. 각 단계는 nohup 으로 백그라운드에 올린 뒤
# `wait` 으로 완료를 기다리므로, 동시에 두 개의 main.py 가 돌지 않는다.
#
# 사용 예:
#   cd /path/to/douri
#   nohup bash scripts/run_configs_sequential_nohup.sh > nohup_logs/sequential_orchestrator.log 2>&1 &
#
# 순서만 바꾸고 싶으면 아래 CONFIGS 배열을 수정한다.
# 모든 런에 공통으로 넘길 인자(예: train_epochs)는 스크립트 뒤에 붙인다:
#   bash scripts/run_configs_sequential_nohup.sh --train_epochs=500
#
# 환경 변수:
#   PYTHON_BIN, RUNS_ROOT, MUJOCO_GL, LOG_DIR — run_cube_single_detached.sh 와 동일 의미

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

LOG_DIR="${LOG_DIR:-${ROOT_DIR}/nohup_logs}"
RUNS_ROOT="${RUNS_ROOT:-${ROOT_DIR}/runs}"
PYTHON_BIN="${PYTHON_BIN:-/home/choi/miniconda3/envs/offrl/bin/python}"
mkdir -p "${LOG_DIR}"

# --- 순서: 필요에 맞게 편집 (상대 경로는 ROOT_DIR 기준 config/) ---
CONFIGS=(
  "config/puzzle_3x3.yaml"
  "config/puzzle_4x4.yaml"
  "config/cube_double.yaml"
  "config/cube_triple.yaml"
)

export PYTHONPATH=.
export MUJOCO_GL="${MUJOCO_GL:-egl}"
unset JAX_PLATFORM_NAME JAX_PLATFORMS CUDA_VISIBLE_DEVICES 2>/dev/null || true

orch_ts="$(date +%Y%m%d_%H%M%S)"
echo "[${orch_ts}] sequential trainer root=${ROOT_DIR}"
echo "[${orch_ts}] python=${PYTHON_BIN} runs_root=${RUNS_ROOT} mujoco_gl=${MUJOCO_GL}"
echo "[${orch_ts}] stages=${#CONFIGS[@]} extra_args=$*"

stage=0
for CONFIG_PATH in "${CONFIGS[@]}"; do
  stage=$((stage + 1))
  if [[ ! -f "${CONFIG_PATH}" ]]; then
    echo "[stage ${stage}] SKIP missing file: ${CONFIG_PATH}" >&2
    exit 1
  fi

  stem="$(basename "${CONFIG_PATH}" .yaml)"
  ts="$(date +%Y%m%d_%H%M%S)"
  step_log="${LOG_DIR}/seq${stage}_${stem}_${ts}.log"

  echo "[stage ${stage}/${#CONFIGS[@]}] start config=${CONFIG_PATH}"
  echo "[stage ${stage}] log=${step_log}"

  cmd=(
    "${PYTHON_BIN}" "${ROOT_DIR}/main.py"
    "--run_config=${CONFIG_PATH}"
    "--runs_root=${RUNS_ROOT}"
    "$@"
  )

  # nohup: 터미널 끊겨도 해당 학습 프로세스는 유지. wait 로 다음 단계는 이전 완료 후에만 시작.
  nohup "${cmd[@]}" >>"${step_log}" 2>&1 &
  pid=$!
  echo "[stage ${stage}] pid=${pid}"
  if ! wait "${pid}"; then
    echo "[stage ${stage}] FAILED exit!=0 pid=${pid} log=${step_log}" >&2
    exit 1
  fi
  echo "[stage ${stage}] done"
done

echo "[${orch_ts}] all stages completed ok (${#CONFIGS[@]} runs)"
