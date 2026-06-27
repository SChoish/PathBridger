#!/usr/bin/env bash
# Supplemental K-sweep for puzzle-4x4 (K=5,10). Chained after main sweep if already running.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH=".:${ROOT}/scripts:${PYTHONPATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-0}"
PYTHON_BIN="${PYTHON_BIN:-/home/choi/miniconda3/envs/offrl/bin/python}"
WITH_CUDA="${ROOT}/scripts/with_jax_cuda.sh"
LOG_DIR="${ROOT}/nohup_logs"
CONFIG_DIR="${ROOT}/config/flow_k_sweep"
LOG_TAG="flow_k_sweep_p4"

"${PYTHON_BIN}" "${ROOT}/scripts/generate_flow_k_sweep_configs.py"

mkdir -p "${LOG_DIR}"
MASTER_LOG="${LOG_DIR}/${LOG_TAG}_master.log"
echo "[$(date -Is)] ${LOG_TAG} start GPU=${GPU_ID} seed=${SEED}" | tee -a "${MASTER_LOG}"

configs=(
  "${CONFIG_DIR}/flow_g5_w5_k5_trl_puzzle_4x4_rd_sd.yaml"
  "${CONFIG_DIR}/flow_g5_w5_k10_trl_puzzle_4x4_rd_sd.yaml"
)

for cfg in "${configs[@]}"; do
  if [[ ! -f "${cfg}" ]]; then
    echo "Missing config: ${cfg}" | tee -a "${MASTER_LOG}"
    exit 1
  fi
  stem="$(basename "${cfg}" .yaml)"
  env_name="$("${PYTHON_BIN}" -c "import yaml; print(yaml.safe_load(open('${cfg}'))['env_name'])")"
  log="${LOG_DIR}/${LOG_TAG}_${stem}.log"
  echo "[$(date -Is)] START ${stem} env=${env_name}" | tee -a "${MASTER_LOG}"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" bash "${WITH_CUDA}" "${PYTHON_BIN}" -u main.py \
    --run_config "${cfg}" \
    --env_name "${env_name}" \
    --seed "${SEED}" \
    --async_prefetch \
    --nouse_wandb \
    --nouse_tqdm \
    2>&1 | tee "${log}" | tee -a "${MASTER_LOG}"
  echo "[$(date -Is)] DONE ${stem}" | tee -a "${MASTER_LOG}"
done

echo "[$(date -Is)] ${LOG_TAG} complete; summarize" | tee -a "${MASTER_LOG}"
"${PYTHON_BIN}" "${ROOT}/scripts/summarize_feval_results.py" | tee -a "${MASTER_LOG}"
