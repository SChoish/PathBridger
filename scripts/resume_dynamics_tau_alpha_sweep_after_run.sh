#!/usr/bin/env bash
# Continue `launch_dynamics_tau_alpha_sweep.sh` after an interrupted first job:
# 1) Resume an existing run dir from the latest checkpoint (default epoch 100).
# 2) Run the remaining sweep configs in the same order as the launch script (skip config #1).
#
# Usage (foreground + tee):
#   ./scripts/resume_dynamics_tau_alpha_sweep_after_run.sh
#
# Usage (nohup):
#   nohup env RESUME_RUN_DIR="$PWD/runs/..." RESUME_EPOCH=100 \
#     ./scripts/resume_dynamics_tau_alpha_sweep_after_run.sh \
#     > nohup_logs/resume_sweep_continue_$(date +%Y%m%d_%H%M%S).log 2>&1 &
#
# Defaults match run `20260426_004705_joint_dqc_seed0_antmaze-large-navigate-v0`
# (large, tau=5, alpha=0.1) — override with RESUME_RUN_DIR / RESUME_EPOCH if needed.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/choi/miniconda3/envs/offrl/bin/python}"
LOG_DIR="${ROOT_DIR}/nohup_logs"
mkdir -p "${LOG_DIR}"

RESUME_RUN_DIR="${RESUME_RUN_DIR:-${ROOT_DIR}/runs/20260426_004705_joint_dqc_seed0_antmaze-large-navigate-v0}"
RESUME_EPOCH="${RESUME_EPOCH:-100}"

CONFIGS=(
  "config/sweep_dynamics_tau_alpha/antmaze_large_navigate_dynamics_tau5p0_alpha0p1.yaml"
  "config/sweep_dynamics_tau_alpha/antmaze_large_navigate_dynamics_tau10p0_alpha0p1.yaml"
  "config/sweep_dynamics_tau_alpha/antmaze_large_navigate_dynamics_tau5p0_alpha0p3.yaml"
  "config/sweep_dynamics_tau_alpha/antmaze_large_navigate_dynamics_tau5p0_alpha0p5.yaml"
  "config/sweep_dynamics_tau_alpha/antmaze_large_navigate_dynamics_tau10p0_alpha0p5.yaml"
  "config/sweep_dynamics_tau_alpha/antmaze_giant_navigate_dynamics_tau5p0_alpha0p1.yaml"
  "config/sweep_dynamics_tau_alpha/antmaze_giant_navigate_dynamics_tau10p0_alpha0p1.yaml"
  "config/sweep_dynamics_tau_alpha/antmaze_giant_navigate_dynamics_tau5p0_alpha0p3.yaml"
  "config/sweep_dynamics_tau_alpha/antmaze_giant_navigate_dynamics_tau10p0_alpha0p3.yaml"
  "config/sweep_dynamics_tau_alpha/antmaze_giant_navigate_dynamics_tau5p0_alpha0p5.yaml"
  "config/sweep_dynamics_tau_alpha/antmaze_giant_navigate_dynamics_tau10p0_alpha0p5.yaml"
)

cd "${ROOT_DIR}"
export PYTHONPATH=.
export MUJOCO_GL="${MUJOCO_GL:-egl}"

_ckpt300() {
  local d="${RESUME_RUN_DIR}"
  [[ -f "${d}/checkpoints/goub/params_300.pkl" ]] \
    && [[ -f "${d}/checkpoints/critic/params_300.pkl" ]] \
    && [[ -f "${d}/checkpoints/actor/params_300.pkl" ]]
}

ts="$(date +%Y%m%d_%H%M%S)"
log="${LOG_DIR}/resume_sweep_continue_${ts}.log"

{
  if _ckpt300; then
    echo "=== RESUME SKIPPED: epoch-300 checkpoints already exist in ${RESUME_RUN_DIR} ==="
  else
    echo "=== RESUME epoch=${RESUME_EPOCH} run_dir=${RESUME_RUN_DIR} start $(date -Iseconds) ==="
    echo "=== (Do not start a second resume on the same run_dir in parallel.) ==="
    "${PYTHON_BIN}" main.py \
      --resume_run_dir="${RESUME_RUN_DIR}" \
      --resume_epoch="${RESUME_EPOCH}"
  fi

  echo "=== REMAINING SWEEP (skip first config; $((${#CONFIGS[@]} - 1)) jobs) start $(date -Iseconds) ==="
  skip_first=1
  for cfg in "${CONFIGS[@]}"; do
    if ((skip_first)); then
      skip_first=0
      echo "=== skip (already resumed): ${cfg} ==="
      continue
    fi
    echo "=== ${cfg} ==="
    "${PYTHON_BIN}" main.py --run_config="${cfg}"
  done
  echo "=== ALL DONE $(date -Iseconds) ==="
} 2>&1 | tee "${log}"
