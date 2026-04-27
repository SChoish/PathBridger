#!/usr/bin/env bash
# Giant navigate — four runs (2×2): (α=0.3|0.4) × (critic discount 0.99|0.995), τ=10, 1000 epochs.
# Matching run under runs/ → resume from latest common goub/critic/actor checkpoint.
#
# Foreground (터미널 붙잡음):
#   cd /path/to/douri && ./scripts/launch_giant_tau10_alpha_discount_1000ep.sh
#
# nohup (SSH 끊어도 계속; 권장):
#   cd /path/to/douri && ./scripts/launch_giant_tau10_alpha_discount_1000ep.sh --nohup
#
# 수동으로 해도 됨:
#   nohup env MUJOCO_GL=egl ./scripts/launch_giant_tau10_alpha_discount_1000ep.sh \\
#     > nohup_logs/giant_tau10_alpha_discount_1k_\$(date +%Y%m%d_%H%M%S).log 2>&1 &
#
# Optional env: MUJOCO_GL (default egl), PYTHON_BIN, RUNS_ROOT, LOG_DIR
#
# CPU 주의: 다른 터미널에서 rollout_*_cpu_*.sh 등을 실행하면 JAX_PLATFORM_NAME=cpu 가
# 셸에 남을 수 있고, nohup 으로 이 스크립트를 띄우면 그대로 물려 GPU를 안 씀.
# 아래에서 명시적으로 제거한다.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/nohup_logs}"
mkdir -p "${LOG_DIR}"

if [[ "${1:-}" == "--nohup" ]]; then
  shift
  LAUNCH_LOG="${LOG_DIR}/giant_tau10_alpha_discount_1k_seq_$(date +%Y%m%d_%H%M%S).log"
  cd "${ROOT_DIR}"
  # 이전 세션의 CPU-only JAX 설정이 자식에게 전달되지 않게 제거
  nohup env -u JAX_PLATFORM_NAME -u JAX_PLATFORMS -u CUDA_VISIBLE_DEVICES \
    PYTHONPATH=. \
    MUJOCO_GL="${MUJOCO_GL:-egl}" \
    PYTHON_BIN="${PYTHON_BIN:-/home/choi/miniconda3/envs/offrl/bin/python}" \
    RUNS_ROOT="${RUNS_ROOT:-${ROOT_DIR}/runs}" \
    LOG_DIR="${LOG_DIR}" \
    bash "${ROOT_DIR}/scripts/launch_giant_tau10_alpha_discount_1000ep.sh" "$@" >>"${LAUNCH_LOG}" 2>&1 &
  echo "nohup started PID=$!  launcher_log=${LAUNCH_LOG}"
  exit 0
fi

cd "${ROOT_DIR}"
export PYTHONPATH=.

unset JAX_PLATFORM_NAME JAX_PLATFORMS CUDA_VISIBLE_DEVICES 2>/dev/null || true

PYTHON_BIN="${PYTHON_BIN:-/home/choi/miniconda3/envs/offrl/bin/python}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
RUNS_ROOT="${RUNS_ROOT:-${ROOT_DIR}/runs}"

TARGET_EPOCHS=1000
SPI_TAU=10.0
ENV_NAME="antmaze-giant-navigate-v0"

CONFIGS=(
  "config/sweep_dynamics_tau_alpha/antmaze_giant_navigate_dynamics_tau10_a0p3_disc0p99.yaml"
  "config/sweep_dynamics_tau_alpha/antmaze_giant_navigate_dynamics_tau10_a0p3_disc0p995.yaml"
  "config/sweep_dynamics_tau_alpha/antmaze_giant_navigate_dynamics_tau10_a0p4_disc0p99.yaml"
  "config/sweep_dynamics_tau_alpha/antmaze_giant_navigate_dynamics_tau10_a0p4_disc0p995.yaml"
)

for cfg in "${CONFIGS[@]}"; do
  if [[ ! -f "${cfg}" ]]; then
    echo "missing ${cfg}" >&2
    exit 1
  fi

  mapfile -t _ad < <("${PYTHON_BIN}" - <<PY
import yaml
from pathlib import Path
d = yaml.safe_load(Path("${cfg}").read_text())
print(d["goub"]["subgoal_value_alpha"])
print(d["critic_agent"]["discount"])
PY
)
  ALPHA="${_ad[0]}"
  DISCOUNT="${_ad[1]}"

  echo ""
  echo "======== ${cfg}  (alpha=${ALPHA} discount=${DISCOUNT}) ========"

  RESUME_TXT="$("${PYTHON_BIN}" "${ROOT_DIR}/scripts/resolve_resume.py" \
    --runs-root="${RUNS_ROOT}" \
    --env-name="${ENV_NAME}" \
    --spi-tau="${SPI_TAU}" \
    --alpha="${ALPHA}" \
    --discount="${DISCOUNT}" \
    --target-epochs="${TARGET_EPOCHS}")"
  STATUS="$(printf '%s\n' "${RESUME_TXT}" | sed -n 's/^STATUS=//p')"
  RUN_DIR="$(printf '%s\n' "${RESUME_TXT}" | sed -n 's/^RUN_DIR=//p')"
  RESUME_EPOCH="$(printf '%s\n' "${RESUME_TXT}" | sed -n 's/^RESUME_EPOCH=//p')"

  stem="$(basename "${cfg}" .yaml)"
  ts="$(date +%Y%m%d_%H%M%S)"
  tee_log="${LOG_DIR}/${stem}_${ts}.log"

  if [[ "${STATUS}" == "complete" ]]; then
    echo "skip: already have checkpoints up to ${RESUME_EPOCH} (>= ${TARGET_EPOCHS}): ${RUN_DIR}"
    continue
  fi

  cmd=(
    "${PYTHON_BIN}" "${ROOT_DIR}/main.py"
    "--train_epochs=${TARGET_EPOCHS}"
    "--save_every_n_epochs=100"
    "--eval_freq=100"
  )

  if [[ "${STATUS}" == "resume" ]]; then
    echo "resume: dir=${RUN_DIR} epoch=${RESUME_EPOCH} -> ${TARGET_EPOCHS}"
    cmd+=(
      "--resume_run_dir=${RUN_DIR}"
      "--resume_epoch=${RESUME_EPOCH}"
      "--run_config=${cfg}"
    )
  else
    echo "new run from yaml (no matching checkpointed run)"
    cmd+=("--run_config=${cfg}")
  fi

  echo "log=${tee_log}"
  "${cmd[@]}" 2>&1 | tee "${tee_log}"
done

echo "all four jobs finished (or were skipped as complete)."
