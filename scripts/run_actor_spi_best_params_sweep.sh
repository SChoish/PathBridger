#!/usr/bin/env bash
# Actor SPI from env_best_params bundle: random-init actor, N/T subgoal pipeline, tau sweep.
#
# Usage:
#   bash scripts/run_actor_spi_best_params_sweep.sh [seed] [tau1 tau2 ...]
#
# Defaults: all 10 envs from docs/env_best_runs_choi/env_best_params.json
#           tau in {0.5, 1, 3, 10}, 100K steps, eval/save at 50K and 100K.

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

GPU_ID="${GPU_ID:-0}"
PYTHON_BIN="${PYTHON_BIN:-/home/choi/miniconda3/envs/offrl/bin/python}"
WITH_CUDA="${ROOT}/scripts/with_jax_cuda.sh"
BEST_PARAMS_JSON="${BEST_PARAMS_JSON:-docs/env_best_runs_choi/env_best_params.json}"
BUNDLE_ROOT="${BUNDLE_ROOT:-docs/env_best_runs_choi/checkpoints}"
ACTOR_SPI_STEPS="${ACTOR_SPI_STEPS:-100000}"
ACTOR_SPI_LR="${ACTOR_SPI_LR:-0.0003}"
EVAL_INTERVAL="${EVAL_INTERVAL:-50000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-50000}"
EVAL_EPISODES="${EVAL_EPISODES:-25}"

ENVS=(
  antmaze-giant-navigate-v0
  antmaze-large-navigate-v0
  antmaze-medium-navigate-v0
  cube-double-play-v0
  cube-single-play-v0
  cube-triple-play-v0
  humanoidmaze-large-navigate-v0
  humanoidmaze-medium-navigate-v0
  puzzle-3x3-play-v0
  puzzle-4x4-play-v0
)

LOG_DIR="${ROOT}/nohup_logs/actor_spi/best_params"
MASTER="${LOG_DIR}/sweep_seed${SEED}.log"
mkdir -p "${LOG_DIR}"

echo "[$(date -Is)] actor_spi best_params sweep start seed=${SEED} taus=${TAUS[*]} GPU=${GPU_ID}" | tee -a "${MASTER}"

for env in "${ENVS[@]}"; do
  for tau in "${TAUS[@]}"; do
    tau_tag="${tau//./p}"
    run_log="${LOG_DIR}/${env}_seed${SEED}_tau${tau_tag}.log"
    echo "[$(date -Is)] START env=${env} tau=${tau}" | tee -a "${MASTER}"
    CUDA_VISIBLE_DEVICES="${GPU_ID}" bash "${WITH_CUDA}" "${PYTHON_BIN}" -u train_actor_spi.py \
      --env_name="${env}" \
      --seed="${SEED}" \
      --spi_tau="${tau}" \
      --actor_spi_steps="${ACTOR_SPI_STEPS}" \
      --actor_spi_lr="${ACTOR_SPI_LR}" \
      --eval_interval="${EVAL_INTERVAL}" \
      --save_interval="${SAVE_INTERVAL}" \
      --actor_spi_eval_episodes="${EVAL_EPISODES}" \
      --best_params_json="${BEST_PARAMS_JSON}" \
      --checkpoint_bundle_root="${BUNDLE_ROOT}" \
      --use_best_params_bundle \
      --init_actor \
      --use_spi_subgoal_pipeline \
      --mujoco_gl="${MUJOCO_GL}" \
      --nouse_wandb \
      2>&1 | tee "${run_log}"
    echo "[$(date -Is)] DONE env=${env} tau=${tau}" | tee -a "${MASTER}"
  done
done

echo "[$(date -Is)] actor_spi best_params sweep complete seed=${SEED}" | tee -a "${MASTER}"
