#!/usr/bin/env bash
# Actor SPI tau sweep for 1M puzzle checkpoints only.
#
# Usage:
#   bash scripts/run_actor_spi_1m_puzzle_sweep.sh [seed] [tau1 tau2 ...]

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}:${ROOT}/scripts:${PYTHONPATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

SEED="${1:-0}"
shift || true

if [[ "$#" -gt 0 ]]; then
  TAUS=("$@")
else
  TAUS=(0.5 1 3 10)
fi

CKPT_ROOT="${CKPT_ROOT:-checkpoints/1m_env_best}"
PRETRAINED_EPOCH="${PRETRAINED_EPOCH:-1000000}"
LOG_DIR="${ROOT}/nohup_logs/actor_spi/1m_puzzle"
MASTER="${LOG_DIR}/master_seed${SEED}.log"
mkdir -p "${LOG_DIR}"

declare -A ENVS=(
  [puzzle-3x3-play-v0]="p3_puzzle-3x3"
  [puzzle-4x4-play-v0]="p4_puzzle-4x4"
)

echo "[$(date -Is)] actor_spi 1m puzzle sweep start seed=${SEED} taus=${TAUS[*]} epoch=${PRETRAINED_EPOCH}" | tee -a "${MASTER}"

for env in puzzle-3x3-play-v0 puzzle-4x4-play-v0; do
  bundle="${CKPT_ROOT}/${ENVS[$env]}"
  if [[ ! -d "${bundle}" ]]; then
    echo "[$(date -Is)] SKIP missing bundle ${bundle}" | tee -a "${MASTER}"
    continue
  fi
  echo "[$(date -Is)] START_ENV ${env} bundle=${bundle}" | tee -a "${MASTER}"
  PRETRAINED_CKPT_DIR="${bundle}" \
  PRETRAINED_EPOCH="${PRETRAINED_EPOCH}" \
  USE_BEST_PARAMS_BUNDLE=false \
  INIT_ACTOR=true \
  USE_SPI_SUBGOAL_PIPELINE=true \
  bash scripts/sweep_actor_spi_tau.sh "${env}" "${SEED}" "${TAUS[@]}"
  echo "[$(date -Is)] DONE_ENV ${env}" | tee -a "${MASTER}"
done

echo "[$(date -Is)] actor_spi 1m puzzle sweep complete seed=${SEED}" | tee -a "${MASTER}"
