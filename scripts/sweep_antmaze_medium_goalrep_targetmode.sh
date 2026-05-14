#!/usr/bin/env bash
# 2×2 sweep: (goal_representation phi|full) × (subgoal_target_mode displacement|absolute).
# Each run: train_epochs=400 (set in yaml). Logs final idm/actor env_success_rate_mean at epoch 400.
#
#   cd /home/choi/douri
#   nohup bash scripts/sweep_antmaze_medium_goalrep_targetmode.sh \
#     > nohup_logs/sweep_medium_goal_target_$(date +%Y%m%d_%H%M%S).log 2>&1 &
#
# Env: PYTHON_BIN, RUNS_ROOT, LOG_DIR, MUJOCO_GL.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

LOG_DIR="${LOG_DIR:-${ROOT_DIR}/nohup_logs}"
RUNS_ROOT="${RUNS_ROOT:-${ROOT_DIR}/runs}"
PYTHON_BIN="${PYTHON_BIN:-/home/choi/miniconda3/envs/offrl/bin/python}"
mkdir -p "${LOG_DIR}"

CONFIGS=(
  "config/antmaze_medium_navigate_table_phi_disp.yaml"
  "config/antmaze_medium_navigate_table_phi_abs.yaml"
  "config/antmaze_medium_navigate_table_full_disp.yaml"
  "config/antmaze_medium_navigate_table_full_abs.yaml"
)

export PYTHONPATH=.
export MUJOCO_GL="${MUJOCO_GL:-egl}"
unset JAX_PLATFORM_NAME JAX_PLATFORMS 2>/dev/null || true

orch_ts="$(date +%Y%m%d_%H%M%S)"
echo "[${orch_ts}] sweep root=${ROOT_DIR} stages=${#CONFIGS[@]}"

stage=0
for CONFIG_PATH in "${CONFIGS[@]}"; do
  stage=$((stage + 1))
  if [[ ! -f "${CONFIG_PATH}" ]]; then
    echo "[stage ${stage}] missing: ${CONFIG_PATH}" >&2
    exit 1
  fi
  stem="$(basename "${CONFIG_PATH}" .yaml)"
  ts="$(date +%Y%m%d_%H%M%S)"
  step_log="${LOG_DIR}/sweep${stage}_${stem}_${ts}.log"
  echo "[stage ${stage}/4] start ${CONFIG_PATH} log=${step_log}"
  "${PYTHON_BIN}" "${ROOT_DIR}/main.py" "--run_config=${CONFIG_PATH}" "--runs_root=${RUNS_ROOT}" >>"${step_log}" 2>&1
  echo "[stage ${stage}/4] done"
done

echo "[$(date +%Y%m%d_%H%M%S)] sweep completed; extract last EVAL lines from each run dir under ${RUNS_ROOT}"
