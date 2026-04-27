#!/usr/bin/env bash
# AntMaze **giant** linear dynamics τ × α only (500 epochs each).
# Uses `config/sweep_dynamics_tau_alpha/antmaze_giant_navigate_dynamics_*.yaml`
# with critic_agent.discount=0.995 (same sweep order as the tail of
# `launch_dynamics_tau_alpha_sweep.sh`).

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/choi/miniconda3/envs/offrl/bin/python}"
LOG_DIR="${ROOT_DIR}/nohup_logs"
mkdir -p "${LOG_DIR}"

CONFIGS=(
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

for cfg in "${CONFIGS[@]}"; do
  stem="$(basename "${cfg}" .yaml)"
  ts="$(date +%Y%m%d_%H%M%S)"
  log="${LOG_DIR}/${stem}_${ts}.log"
  echo "=== ${cfg} ==="
  echo "log=${log}"
  "${PYTHON_BIN}" main.py --run_config="${cfg}" 2>&1 | tee "${log}"
done
