#!/usr/bin/env bash
# Subgoal SPI tau sweep for antmaze only (checkpoints/1m_env_best/).
#
# Usage:
#   bash scripts/run_subgoal_spi_1m_antmaze_sweep.sh [seed] [tau1 tau2 ...]
#
# Same setup as run_subgoal_spi_1m_sweep.sh / actor SPI: 1M ckpt, N/T from
# best_eval_meta.yaml, 100K steps, eval @ 50K/100K, tau {0.5,1,3,10}.

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
LOG_DIR="${ROOT}/nohup_logs/subgoal_spi/1m_antmaze"
MASTER="${LOG_DIR}/master_seed${SEED}.log"
mkdir -p "${LOG_DIR}"

declare -A ENVS=(
  [antmaze-medium-navigate-v0]="amm_antmaze-medium"
  [antmaze-large-navigate-v0]="aml_antmaze-large"
  [antmaze-giant-navigate-v0]="amg_antmaze-giant"
)

echo "[$(date -Is)] subgoal_spi 1m antmaze start seed=${SEED} taus=${TAUS[*]} epoch=${PRETRAINED_EPOCH}" | tee -a "${MASTER}"

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
  bash scripts/sweep_subgoal_spi_tau.sh "${env}" "${SEED}" "${TAUS[@]}"
  echo "[$(date -Is)] DONE_ENV ${env}" | tee -a "${MASTER}"
done

echo "[$(date -Is)] subgoal_spi 1m antmaze complete seed=${SEED}" | tee -a "${MASTER}"
