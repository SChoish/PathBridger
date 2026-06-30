#!/usr/bin/env bash
# Actor SPI tau sweep for cube envs from checkpoints/1m_env_best/.
#
# Usage:
#   bash scripts/run_actor_spi_1m_cube_sweep.sh [seed] [tau1 tau2 ...]

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SEED="${1:-0}"
if [[ "$#" -gt 0 ]]; then
  shift || true
fi

if [[ "$#" -gt 0 ]]; then
  TAUS=("$@")
else
  TAUS=(0.5 1 3 10)
fi

CKPT_ROOT="${CKPT_ROOT:-checkpoints/1m_env_best}"
PRETRAINED_EPOCH="${PRETRAINED_EPOCH:-1000000}"
LOG_DIR="${ROOT}/nohup_logs/actor_spi/1m_cube"
MASTER="${LOG_DIR}/master_seed${SEED}.log"
mkdir -p "${LOG_DIR}"

declare -A ENVS=(
  [cube-single-play-v0]="cs_cube-single"
  [cube-double-play-v0]="cd_cube-double"
  [cube-triple-play-v0]="ct_cube-triple"
)

echo "[$(date -Is)] actor_spi 1m cube sweep start seed=${SEED} taus=${TAUS[*]}" | tee -a "${MASTER}"

for env in cube-single-play-v0 cube-double-play-v0 cube-triple-play-v0; do
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

echo "[$(date -Is)] actor_spi 1m cube sweep complete seed=${SEED}" | tee -a "${MASTER}"
