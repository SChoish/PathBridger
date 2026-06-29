#!/usr/bin/env bash
# Flow+TRL: puzzle-4x5/4x6 K=40, gap10/wmax5/N1, final eval only.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}:${ROOT}/scripts:${PYTHONPATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-0}"
EVAL_ONLY="${EVAL_ONLY:-0}"
PYTHON_BIN="${PYTHON_BIN:-/home/offrl/miniconda3/envs/offrl/bin/python}"
WITH_CUDA="${ROOT}/scripts/with_jax_cuda.sh"
LOG_DIR="${ROOT}/nohup_logs"
RUNS_ROOT="${ROOT}/runs"
CONFIG_DIR="${ROOT}/config/sweep_flow_trl_p456_k40_g10"
LOG_TAG="flow_trl_p456_k40_g10"
FINAL_STEP="${FINAL_STEP:-600000}"

mkdir -p "${LOG_DIR}"
MASTER_LOG="${LOG_DIR}/${LOG_TAG}_master.log"

mapfile -t configs < <(
  CONFIG_DIR="${CONFIG_DIR}" "${PYTHON_BIN}" - <<'PY'
import os
from pathlib import Path

config_dir = Path(os.environ['CONFIG_DIR'])
for name in ('p45_k40_g10_w5_n1.yaml', 'p46_k40_g10_w5_n1.yaml'):
    print(config_dir / name)
PY
)

mapfile -t eval_ns < <(
  "${PYTHON_BIN}" - <<'PY'
from flow_trl_sweep_common import FINAL_EVAL_N_VALUES
for n in FINAL_EVAL_N_VALUES:
    print(n)
PY
)

resolve_run_dir() {
  local cfg="$1"
  CONFIG_PATH="${cfg}" RUNS_ROOT="${RUNS_ROOT}" SEED="${SEED}" FINAL_STEP="${FINAL_STEP}" \
    "${PYTHON_BIN}" - <<'PY'
import os
import sys

sys.path.insert(0, os.path.join(os.getcwd(), 'scripts'))
from flow_trl_sweep_common import resolve_run_dir

print(
    resolve_run_dir(
        config_path=os.environ['CONFIG_PATH'],
        runs_root=os.environ['RUNS_ROOT'],
        seed=int(os.environ['SEED']),
        final_epoch=int(os.environ['FINAL_STEP']),
        require_checkpoint=True,
    )
)
PY
}

eval_results_complete() {
  RUN_DIR="$1" FINAL_STEP="${FINAL_STEP}" \
    "${PYTHON_BIN}" - <<'PY'
import os
import sys

sys.path.insert(0, os.path.join(os.getcwd(), 'scripts'))
from flow_trl_sweep_common import eval_results_complete

print('1' if eval_results_complete(os.environ['RUN_DIR'], epoch=int(os.environ['FINAL_STEP'])) else '0')
PY
}

run_final_evals() {
  local base="$1"
  local run_dir="$2"
  for eval_n in "${eval_ns[@]}"; do
    eval_log="${LOG_DIR}/${LOG_TAG}_${base}.eval_n${eval_n}.log"
    echo "[$(date -Is)] START_EVAL ${base} eval_n=${eval_n} run_dir=${run_dir}" | tee -a "${MASTER_LOG}"
    CUDA_VISIBLE_DEVICES="${GPU_ID}" bash "${WITH_CUDA}" "${PYTHON_BIN}" -u eval_checkpoint.py \
      --run_dir "${run_dir}" \
      --epoch "${FINAL_STEP}" \
      --seed "${SEED}" \
      --eval_episodes_per_task 25 \
      --subgoal_eval_num_samples "${eval_n}" \
      --mujoco_gl "${MUJOCO_GL}" \
      --skip_if_saved \
      2>&1 | tee "${eval_log}"
    echo "[$(date -Is)] DONE_EVAL ${base} eval_n=${eval_n}" | tee -a "${MASTER_LOG}"
  done
}

mode_label="train+final-eval"
if [[ "${EVAL_ONLY}" == "1" ]]; then
  mode_label="eval-only"
fi
echo "[$(date -Is)] ${LOG_TAG} start mode=${mode_label} GPU=${GPU_ID} seed=${SEED}" | tee -a "${MASTER_LOG}"

for cfg in "${configs[@]}"; do
  if [[ ! -f "${cfg}" ]]; then
    echo "Missing config: ${cfg}" | tee -a "${MASTER_LOG}"
    exit 1
  fi
  base="$(basename "$cfg" .yaml)"
  env_name="$("${PYTHON_BIN}" -c "import yaml; print(yaml.safe_load(open('${cfg}'))['env_name'])")"
  run_dir="$(resolve_run_dir "${cfg}")"

  if [[ "${EVAL_ONLY}" != "1" ]]; then
    if [[ -n "${run_dir}" ]] && [[ "$(eval_results_complete "${run_dir}")" == "1" ]]; then
      echo "[$(date -Is)] SKIP_COMPLETE ${base} run_dir=${run_dir}" | tee -a "${MASTER_LOG}"
      continue
    fi
    if [[ -n "${run_dir}" ]]; then
      echo "[$(date -Is)] SKIP_TRAIN_INCOMPLETE_EVAL ${base} run_dir=${run_dir}" | tee -a "${MASTER_LOG}"
    else
      train_log="${LOG_DIR}/${LOG_TAG}_${base}.train.log"
      echo "[$(date -Is)] START_TRAIN ${base} env=${env_name}" | tee -a "${MASTER_LOG}"
      CUDA_VISIBLE_DEVICES="${GPU_ID}" bash "${WITH_CUDA}" "${PYTHON_BIN}" -u main.py \
        --run_config "${cfg}" \
        --env_name "${env_name}" \
        --seed "${SEED}" \
        --async_prefetch \
        --nouse_wandb \
        --nouse_tqdm \
        2>&1 | tee "${train_log}"
      run_dir="$(resolve_run_dir "${cfg}")"
    fi
  fi

  if [[ -z "${run_dir}" ]]; then
    echo "[$(date -Is)] SKIP_EVAL_NO_RUNDIR ${base}" | tee -a "${MASTER_LOG}"
    continue
  fi
  if [[ "$(eval_results_complete "${run_dir}")" == "1" ]]; then
    echo "[$(date -Is)] EVAL_ALREADY_COMPLETE ${base} run_dir=${run_dir}" | tee -a "${MASTER_LOG}"
    continue
  fi
  echo "[$(date -Is)] RESOLVED_RUNDIR ${base} run_dir=${run_dir}" | tee -a "${MASTER_LOG}"
  run_final_evals "${base}" "${run_dir}"
done

echo "[$(date -Is)] SUMMARIZE feval results" | tee -a "${MASTER_LOG}"
"${PYTHON_BIN}" "${ROOT}/scripts/summarize_feval_results.py" | tee -a "${MASTER_LOG}"
echo "[$(date -Is)] ${LOG_TAG} complete (${#configs[@]} configs)" | tee -a "${MASTER_LOG}"
