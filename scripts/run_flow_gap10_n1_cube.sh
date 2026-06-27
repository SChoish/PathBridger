#!/usr/bin/env bash
# Flow + TRL cube retrain: gap=10, wmax=5, train N=1, final eval N=1,4,8,16.
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
CONFIG_DIR="${ROOT}/config/flow_gap10_w5_n1_cube"

"${PYTHON_BIN}" "${ROOT}/scripts/generate_flow_gap10_n1_cube_configs.py"

mkdir -p "${LOG_DIR}"
MASTER_LOG="${LOG_DIR}/flow_gap10_w5_n1_cube_master.log"
echo "[$(date -Is)] flow gap10 w5 N1 cube retrain start GPU=${GPU_ID} seed=${SEED}" | tee -a "${MASTER_LOG}"

for cfg in "${CONFIG_DIR}"/flow_gap10_w5_n1_trl_*_rd_sd.yaml; do
  stem="$(basename "${cfg}" .yaml)"
  env_name="$("${PYTHON_BIN}" -c "import yaml; print(yaml.safe_load(open('${cfg}'))['env_name'])")"
  log="${LOG_DIR}/${stem}.log"
  echo "[$(date -Is)] START ${stem} env=${env_name}" | tee -a "${MASTER_LOG}"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" bash "${WITH_CUDA}" "${PYTHON_BIN}" -u main.py \
    --run_config "${cfg}" \
    --env_name "${env_name}" \
    --seed "${SEED}" \
    --async_prefetch \
    --nouse_wandb \
    --nouse_tqdm \
    2>&1 | tee "${log}"
  echo "[$(date -Is)] DONE ${stem}" | tee -a "${MASTER_LOG}"
done

echo "[$(date -Is)] flow gap10 w5 N1 cube retrain complete" | tee -a "${MASTER_LOG}"
"${PYTHON_BIN}" "${ROOT}/scripts/summarize_feval_results.py" | tee -a "${MASTER_LOG}"
