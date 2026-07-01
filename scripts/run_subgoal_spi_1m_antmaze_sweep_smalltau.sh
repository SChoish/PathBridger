#!/usr/bin/env bash
# Subgoal SPI antmaze sweep with small tau (1e-3, 1e-2).
#
# Usage:
#   bash scripts/run_subgoal_spi_1m_antmaze_sweep_smalltau.sh [seed] [tau1 tau2 ...]
#
# Eval: eval_spi_subgoal_only=true (skip frozen flow baselines).

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}:${ROOT}/scripts:${PYTHONPATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export EVAL_SPI_SUBGOAL_ONLY="${EVAL_SPI_SUBGOAL_ONLY:-true}"

SEED="${1:-0}"
shift || true

if [[ "$#" -gt 0 ]]; then
  TAUS=("$@")
else
  TAUS=(0.001 0.01)
fi

CKPT_ROOT="${CKPT_ROOT:-checkpoints/1m_env_best}"
PRETRAINED_EPOCH="${PRETRAINED_EPOCH:-1000000}"
LOG_DIR="${ROOT}/nohup_logs/subgoal_spi/1m_antmaze_smalltau"
MASTER="${LOG_DIR}/master_seed${SEED}.log"
mkdir -p "${LOG_DIR}"

declare -A ENVS=(
  [antmaze-medium-navigate-v0]="amm_antmaze-medium"
  [antmaze-large-navigate-v0]="aml_antmaze-large"
  [antmaze-giant-navigate-v0]="amg_antmaze-giant"
)

echo "[$(date -Is)] subgoal_spi smalltau start seed=${SEED} taus=${TAUS[*]} eval_spi_only=${EVAL_SPI_SUBGOAL_ONLY}" | tee -a "${MASTER}"

for env in antmaze-medium-navigate-v0 antmaze-large-navigate-v0 antmaze-giant-navigate-v0; do
  bundle="${CKPT_ROOT}/${ENVS[$env]}"
  if [[ ! -d "${bundle}" ]]; then
    echo "[$(date -Is)] SKIP missing bundle ${bundle}" | tee -a "${MASTER}"
    continue
  fi
  echo "[$(date -Is)] START_ENV ${env} bundle=${bundle}" | tee -a "${MASTER}"
  PRETRAINED_CKPT_DIR="${bundle}" \
  PRETRAINED_EPOCH="${PRETRAINED_EPOCH}" \
  USE_BEST_PARAMS_BUNDLE=false \
  EVAL_SPI_SUBGOAL_ONLY="${EVAL_SPI_SUBGOAL_ONLY}" \
  bash scripts/sweep_subgoal_spi_tau.sh "${env}" "${SEED}" "${TAUS[@]}"
  echo "[$(date -Is)] DONE_ENV ${env}" | tee -a "${MASTER}"
done

echo "[$(date -Is)] subgoal_spi smalltau complete seed=${SEED}" | tee -a "${MASTER}"
