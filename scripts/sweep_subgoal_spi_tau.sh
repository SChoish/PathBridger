#!/usr/bin/env bash
# Sweep subgoal_spi_tau for subgoal SPI finetuning from a pretrained checkpoint.
#
# Usage:
#   bash scripts/sweep_subgoal_spi_tau.sh <env_name> <seed> [tau1 tau2 ...]

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}:${ROOT}/scripts:${PYTHONPATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

ENV_NAME="${1:?usage: sweep_subgoal_spi_tau.sh <env_name> <seed> [taus...]}"
SEED="${2:?usage: sweep_subgoal_spi_tau.sh <env_name> <seed> [taus...]}"
shift 2 || true

if [[ "$#" -gt 0 ]]; then
  TAUS=("$@")
else
  TAUS=(0.5 1 3 10)
fi

GPU_ID="${GPU_ID:-0}"
PYTHON_BIN="${PYTHON_BIN:-/home/svcho/anaconda3/envs/offrl/bin/python}"
WITH_CUDA="${ROOT}/scripts/with_jax_cuda.sh"
SUBGOAL_SPI_CONFIG="${SUBGOAL_SPI_CONFIG:-config/subgoal_spi/subgoal_spi_default.yaml}"
EVAL_SPI_SUBGOAL_ONLY="${EVAL_SPI_SUBGOAL_ONLY:-true}"
SUBGOAL_SPI_STEPS="${SUBGOAL_SPI_STEPS:-100000}"
SUBGOAL_SPI_LR="${SUBGOAL_SPI_LR:-0.0003}"
SUBGOAL_SPI_BATCH_SIZE="${SUBGOAL_SPI_BATCH_SIZE:-0}"
EVAL_INTERVAL="${EVAL_INTERVAL:-50000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-50000}"
EVAL_EPISODES="${EVAL_EPISODES:-25}"
PRETRAINED_CKPT_DIR="${PRETRAINED_CKPT_DIR:-}"
PRETRAINED_EPOCH="${PRETRAINED_EPOCH:--1}"
BEST_RUNS_ROOT="${BEST_RUNS_ROOT:-runs}"
USE_BEST_PARAMS_BUNDLE="${USE_BEST_PARAMS_BUNDLE:-false}"
LOG_DIR="${ROOT}/nohup_logs/subgoal_spi/${ENV_NAME}"
MASTER="${LOG_DIR}/sweep_seed${SEED}.log"

mkdir -p "${LOG_DIR}"
echo "[$(date -Is)] subgoal_spi sweep env=${ENV_NAME} seed=${SEED} taus=${TAUS[*]} GPU=${GPU_ID}" | tee -a "${MASTER}"

for tau in "${TAUS[@]}"; do
  tau_tag="${tau//./p}"
  run_log="${LOG_DIR}/seed${SEED}_tau${tau_tag}.log"
  extra_args=()
  if [[ -n "${PRETRAINED_CKPT_DIR}" ]]; then
    extra_args+=(--pretrained_ckpt_dir="${PRETRAINED_CKPT_DIR}")
  fi
  if [[ "${PRETRAINED_EPOCH}" -ge 0 ]]; then
    extra_args+=(--pretrained_epoch="${PRETRAINED_EPOCH}")
  fi
  if [[ "${USE_BEST_PARAMS_BUNDLE}" == "true" ]]; then
    extra_args+=(--use_best_params_bundle)
  else
    extra_args+=(--nouse_best_params_bundle)
  fi
  if [[ "${EVAL_SPI_SUBGOAL_ONLY}" == "true" ]]; then
    extra_args+=(--eval_spi_subgoal_only)
  else
    extra_args+=(--noeval_spi_subgoal_only)
  fi
  echo "[$(date -Is)] START tau=${tau}" | tee -a "${MASTER}"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" bash "${WITH_CUDA}" "${PYTHON_BIN}" -u train_subgoal_spi.py \
    --env_name="${ENV_NAME}" \
    --seed="${SEED}" \
    --subgoal_spi_config="${SUBGOAL_SPI_CONFIG}" \
    --subgoal_spi_tau="${tau}" \
    --subgoal_spi_steps="${SUBGOAL_SPI_STEPS}" \
    --subgoal_spi_lr="${SUBGOAL_SPI_LR}" \
    --subgoal_spi_batch_size="${SUBGOAL_SPI_BATCH_SIZE}" \
    --eval_interval="${EVAL_INTERVAL}" \
    --save_interval="${SAVE_INTERVAL}" \
    --subgoal_spi_eval_episodes="${EVAL_EPISODES}" \
    --best_runs_root="${BEST_RUNS_ROOT}" \
    --mujoco_gl="${MUJOCO_GL}" \
    --nouse_wandb \
    "${extra_args[@]}" \
    2>&1 | tee "${run_log}"
  echo "[$(date -Is)] DONE tau=${tau}" | tee -a "${MASTER}"
done

echo "[$(date -Is)] subgoal_spi sweep complete env=${ENV_NAME} seed=${SEED}" | tee -a "${MASTER}"
