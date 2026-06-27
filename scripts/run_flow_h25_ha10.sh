#!/usr/bin/env bash
# Flow + TRL: horizon K=25, h_a=10 across puzzle/cube/antmaze; final eval N=1,4,8,16.
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
CONFIG_DIR="${ROOT}/config/flow_h25_ha10"
LOG_TAG="flow_h25_ha10"

"${PYTHON_BIN}" "${ROOT}/scripts/generate_flow_h25_ha10_configs.py"

mkdir -p "${LOG_DIR}"
MASTER_LOG="${LOG_DIR}/${LOG_TAG}_master.log"
echo "[$(date -Is)] ${LOG_TAG} start GPU=${GPU_ID} seed=${SEED}" | tee -a "${MASTER_LOG}"

mapfile -t configs < <(
  CONFIG_DIR="${CONFIG_DIR}" "${PYTHON_BIN}" - <<'PY'
import glob
import os

config_dir = os.environ['CONFIG_DIR']
order = [
    'puzzle_3x3', 'puzzle_4x4',
    'cube_single', 'cube_double', 'cube_triple',
    'antmaze_medium', 'antmaze_large', 'antmaze_giant',
]

def sort_key(path: str) -> int:
    base = os.path.basename(path)
    stem = base.replace('flow_g', '').replace('_rd_sd.yaml', '')
    parts = stem.split('_trl_')
    if len(parts) != 2:
        return 99
    env = parts[1]
    return order.index(env) if env in order else 99

paths = glob.glob(os.path.join(config_dir, 'flow_g*_h25_ha10_n1_trl_*_rd_sd.yaml'))
for p in sorted(paths, key=sort_key):
    print(p)
PY
)

if ((${#configs[@]} == 0)); then
  echo "No configs in ${CONFIG_DIR}" | tee -a "${MASTER_LOG}"
  exit 1
fi

echo "[$(date -Is)] ${#configs[@]} configs queued" | tee -a "${MASTER_LOG}"

for cfg in "${configs[@]}"; do
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
