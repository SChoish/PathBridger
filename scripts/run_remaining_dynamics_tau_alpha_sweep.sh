#!/usr/bin/env bash
# Run configs 2..N of `launch_dynamics_tau_alpha_sweep.sh` (skip large tau5 alpha0.1).
# Use after the first sweep job finished (e.g. resumed run reached 300 epochs).

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/choi/miniconda3/envs/offrl/bin/python}"
LOG_DIR="${ROOT_DIR}/nohup_logs"
mkdir -p "${LOG_DIR}"

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

ts="$(date +%Y%m%d_%H%M%S)"
log="${LOG_DIR}/remaining_dynamics_sweep_${ts}.log"

{
  echo "=== REMAINING SWEEP only ($((${#CONFIGS[@]} - 1)) jobs) start $(date -Iseconds) ==="
  skip_first=1
  for cfg in "${CONFIGS[@]}"; do
    if ((skip_first)); then
      skip_first=0
      echo "=== skip first config: ${cfg} ==="
      continue
    fi
    echo "=== ${cfg} ==="
    "${PYTHON_BIN}" main.py --run_config="${cfg}"
  done
  echo "=== ALL DONE $(date -Iseconds) ==="
} 2>&1 | tee "${log}"
