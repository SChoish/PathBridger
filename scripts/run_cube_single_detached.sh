#!/usr/bin/env bash
# Cube-single 학습을 터미널과 완전히 분리해 실행한다.
# 기본 사용:
#   ./scripts/run_cube_single_detached.sh
#
# 다른 config 사용:
#   ./scripts/run_cube_single_detached.sh --config config/cube_double.yaml
#
# 포그라운드 실행(디버깅용):
#   ./scripts/run_cube_single_detached.sh --foreground
#
# 참고:
# - Cursor/IDE 터미널 종료 시에는 단순 nohup 만으로는 부모 세션 정리 신호를 피하지 못할 수 있어
#   launcher 단계에서 setsid 로 새 세션을 만든다.
# - 실제 학습 stdout/stderr 는 nohup_logs/ 아래 launch log 에 남고,
#   코드 내부 로그는 runs/.../run.log 에 별도로 남는다.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/nohup_logs}"
RUNS_ROOT="${RUNS_ROOT:-${ROOT_DIR}/runs}"
PYTHON_BIN="${PYTHON_BIN:-/home/choi/miniconda3/envs/offrl/bin/python}"
DEFAULT_CONFIG="config/cube_single.yaml"
PID_DIR="${LOG_DIR}/pids"

mkdir -p "${LOG_DIR}" "${PID_DIR}" "${RUNS_ROOT}"

CONFIG_PATH="${DEFAULT_CONFIG}"
FOREGROUND=0
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG_PATH="${2:?missing value for --config}"
      shift 2
      ;;
    --foreground)
      FOREGROUND=1
      shift
      ;;
    --worker)
      shift
      exec env RUNNER_WORKER=1 "$0" "$@"
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

cd "${ROOT_DIR}"

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "missing config: ${CONFIG_PATH}" >&2
  exit 1
fi

if [[ "${RUNNER_WORKER:-0}" != "1" && "${FOREGROUND}" -eq 0 ]]; then
  ts="$(date +%Y%m%d_%H%M%S)"
  stem="$(basename "${CONFIG_PATH}" .yaml)"
  launch_log="${LOG_DIR}/${stem}_${ts}.log"
  pid_file="${PID_DIR}/${stem}_${ts}.pid"

  setsid env \
    RUNNER_WORKER=1 \
    PYTHON_BIN="${PYTHON_BIN}" \
    LOG_DIR="${LOG_DIR}" \
    RUNS_ROOT="${RUNS_ROOT}" \
    MUJOCO_GL="${MUJOCO_GL:-egl}" \
    bash "$0" --config "${CONFIG_PATH}" "${EXTRA_ARGS[@]}" \
    </dev/null >>"${launch_log}" 2>&1 &

  launcher_pid=$!
  printf '%s\n' "${launcher_pid}" > "${pid_file}"

  echo "detached started"
  echo "pid=${launcher_pid}"
  echo "pid_file=${pid_file}"
  echo "launch_log=${launch_log}"
  exit 0
fi

unset JAX_PLATFORM_NAME JAX_PLATFORMS CUDA_VISIBLE_DEVICES 2>/dev/null || true
export PYTHONPATH=.
export MUJOCO_GL="${MUJOCO_GL:-egl}"

ts="$(date +%Y%m%d_%H%M%S)"
stem="$(basename "${CONFIG_PATH}" .yaml)"
echo "[${ts}] start config=${CONFIG_PATH}"
echo "[${ts}] python=${PYTHON_BIN}"
echo "[${ts}] runs_root=${RUNS_ROOT}"
echo "[${ts}] mujoco_gl=${MUJOCO_GL}"

cmd=(
  "${PYTHON_BIN}" "${ROOT_DIR}/main.py"
  "--run_config=${CONFIG_PATH}"
  "--runs_root=${RUNS_ROOT}"
)

if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  cmd+=("${EXTRA_ARGS[@]}")
fi

printf '[%s] cmd=' "${ts}"
printf ' %q' "${cmd[@]}"
printf '\n'

exec "${cmd[@]}"
