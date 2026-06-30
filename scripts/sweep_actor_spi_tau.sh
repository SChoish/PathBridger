#!/usr/bin/env bash
# Sweep spi_tau for actor-only SPI finetuning from the best-IDM eval checkpoint in runs/.
#
# Usage:
#   bash scripts/sweep_actor_spi_tau.sh <env_name> <seed> [tau1 tau2 ...]
#
# env_name accepts full OGBench names or short tokens, e.g. puzzle-3x3.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}:${ROOT}/scripts:${PYTHONPATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

ENV_NAME="${1:?usage: sweep_actor_spi_tau.sh <env_name> <seed> [taus...]}"
SEED="${2:?usage: sweep_actor_spi_tau.sh <env_name> <seed> [taus...]}"
shift 2 || true

if [[ "$#" -gt 0 ]]; then
  TAUS=("$@")
else
  TAUS=(0.01 0.03 0.1 0.3 1.0 3.0 10.0)
fi

GPU_ID="${GPU_ID:-0}"
PYTHON_BIN="${PYTHON_BIN:-/home/choi/miniconda3/envs/offrl/bin/python}"
WITH_CUDA="${ROOT}/scripts/with_jax_cuda.sh"
ACTOR_SPI_STEPS="${ACTOR_SPI_STEPS:-100000}"
ACTOR_SPI_LR="${ACTOR_SPI_LR:-0.0003}"
ACTOR_SPI_BATCH_SIZE="${ACTOR_SPI_BATCH_SIZE:-0}"
EVAL_INTERVAL="${EVAL_INTERVAL:-50000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-50000}"
EVAL_EPISODES="${EVAL_EPISODES:-25}"
PRETRAINED_CKPT_DIR="${PRETRAINED_CKPT_DIR:-}"
BEST_RUNS_ROOT="${BEST_RUNS_ROOT:-runs}"
LOG_DIR="${ROOT}/nohup_logs/actor_spi/${ENV_NAME}"
MASTER="${LOG_DIR}/sweep_seed${SEED}.log"

mkdir -p "${LOG_DIR}"
echo "[$(date -Is)] actor_spi sweep env=${ENV_NAME} seed=${SEED} taus=${TAUS[*]} GPU=${GPU_ID}" | tee -a "${MASTER}"

for tau in "${TAUS[@]}"; do
  tau_tag="${tau//./p}"
  run_log="${LOG_DIR}/seed${SEED}_tau${tau_tag}.log"
  extra_args=()
  if [[ -n "${PRETRAINED_CKPT_DIR}" ]]; then
    extra_args+=(--pretrained_ckpt_dir="${PRETRAINED_CKPT_DIR}")
  fi
  echo "[$(date -Is)] START tau=${tau}" | tee -a "${MASTER}"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" bash "${WITH_CUDA}" "${PYTHON_BIN}" -u train_actor_spi.py \
    --env_name="${ENV_NAME}" \
    --seed="${SEED}" \
    --spi_tau="${tau}" \
    --actor_spi_steps="${ACTOR_SPI_STEPS}" \
    --actor_spi_lr="${ACTOR_SPI_LR}" \
    --actor_spi_batch_size="${ACTOR_SPI_BATCH_SIZE}" \
    --eval_interval="${EVAL_INTERVAL}" \
    --save_interval="${SAVE_INTERVAL}" \
    --actor_spi_eval_episodes="${EVAL_EPISODES}" \
    --best_runs_root="${BEST_RUNS_ROOT}" \
    --mujoco_gl="${MUJOCO_GL}" \
    --nouse_wandb \
    "${extra_args[@]}" \
    2>&1 | tee "${run_log}"
  echo "[$(date -Is)] DONE tau=${tau}" | tee -a "${MASTER}"
done

echo "[$(date -Is)] actor_spi sweep complete env=${ENV_NAME} seed=${SEED}" | tee -a "${MASTER}"
