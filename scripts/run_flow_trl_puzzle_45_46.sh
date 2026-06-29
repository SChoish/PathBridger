#!/usr/bin/env bash
# Flow+TRL sweep: puzzle-4x5 and puzzle-4x6, gap={1,3,5,10}, vdist_pow=0.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}:${ROOT}/scripts:${PYTHONPATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-0}"
EVAL_ONLY="${EVAL_ONLY:-0}"
PYTHON_BIN="${PYTHON_BIN:-/home/svcho/anaconda3/bin/python}"
WITH_CUDA="${ROOT}/scripts/with_jax_cuda.sh"
LOG_DIR="${ROOT}/nohup_logs"
RUNS_ROOT="${ROOT}/runs"
CONFIG_DIR="${ROOT}/config/sweep_flow_trl_puzzle_45_46"
WRITER="${ROOT}/scripts/write_flow_trl_puzzle_45_46_yaml.py"
LOG_TAG="flow_trl_p456"
FINAL_EPOCH="${FINAL_EPOCH:-600}"

mkdir -p "${LOG_DIR}"
MASTER_LOG="${LOG_DIR}/${LOG_TAG}_master.log"
"${PYTHON_BIN}" "${WRITER}"
"${PYTHON_BIN}" "${WRITER}" --probe

mode_label="train+eval"
if [[ "${EVAL_ONLY}" == "1" ]]; then
  mode_label="eval-only"
fi
echo "[$(date -Is)] ${LOG_TAG} sweep start mode=${mode_label} GPU=${GPU_ID} seed=${SEED}" | tee -a "${MASTER_LOG}"

mapfile -t configs < <(
  CONFIG_DIR="${CONFIG_DIR}" "${PYTHON_BIN}" - <<'PY'
import glob
import os
import sys

sys.path.insert(0, os.path.join(os.getcwd(), 'scripts'))
from flow_trl_sweep_common import config_sort_key

config_dir = os.environ['CONFIG_DIR']
paths = [
    p for p in glob.glob(os.path.join(config_dir, '*.yaml'))
    if not os.path.basename(p).startswith('_')
]
for p in sorted(paths, key=lambda x: config_sort_key(os.path.basename(x))):
    print(p)
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
  CONFIG_PATH="${cfg}" RUNS_ROOT="${RUNS_ROOT}" SEED="${SEED}" FINAL_EPOCH="${FINAL_EPOCH}" \
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
        final_epoch=int(os.environ['FINAL_EPOCH']),
        require_checkpoint=True,
    )
)
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
      --epoch "${FINAL_EPOCH}" \
      --seed "${SEED}" \
      --eval_episodes_per_task 25 \
      --subgoal_eval_num_samples "${eval_n}" \
      --mujoco_gl "${MUJOCO_GL}" \
      --skip_if_saved \
      2>&1 | tee "${eval_log}"
    echo "[$(date -Is)] DONE_EVAL ${base} eval_n=${eval_n}" | tee -a "${MASTER_LOG}"
  done
}

eval_results_complete() {
  RUN_DIR="$1" FINAL_EPOCH="${FINAL_EPOCH}" \
    "${PYTHON_BIN}" - <<'PY'
import os
import sys

sys.path.insert(0, os.path.join(os.getcwd(), 'scripts'))
from flow_trl_sweep_common import eval_results_complete

print('1' if eval_results_complete(os.environ['RUN_DIR'], epoch=int(os.environ['FINAL_EPOCH'])) else '0')
PY
}

run_gamma_matches_config() {
  CONFIG_PATH="$1" RUN_DIR="$2" \
    "${PYTHON_BIN}" - <<'PY'
import os
import sys

sys.path.insert(0, os.path.join(os.getcwd(), 'scripts'))
from flow_trl_sweep_common import run_gamma_matches_config

print('1' if run_gamma_matches_config(
    config_path=os.environ['CONFIG_PATH'],
    run_dir=os.environ['RUN_DIR'],
) else '0')
PY
}

if ((${#configs[@]} == 0)); then
  echo "No configs in ${CONFIG_DIR}" | tee -a "${MASTER_LOG}"
  exit 1
fi

for cfg in "${configs[@]}"; do
  base="$(basename "$cfg" .yaml)"
  env_name="$("${PYTHON_BIN}" -c "import yaml; print(yaml.safe_load(open('${cfg}'))['env_name'])")"
  run_dir="$(resolve_run_dir "${cfg}")"

  if [[ "${EVAL_ONLY}" != "1" ]]; then
    if [[ -n "${run_dir}" ]] && [[ "$(eval_results_complete "${run_dir}")" == "1" ]] \
        && [[ "$(run_gamma_matches_config "${cfg}" "${run_dir}")" == "1" ]]; then
      echo "[$(date -Is)] SKIP_COMPLETE ${base} run_dir=${run_dir}" | tee -a "${MASTER_LOG}"
      continue
    fi
    if [[ -n "${run_dir}" ]] && [[ "$(eval_results_complete "${run_dir}")" == "1" ]] \
        && [[ "$(run_gamma_matches_config "${cfg}" "${run_dir}")" != "1" ]]; then
      echo "[$(date -Is)] GAMMA_MISMATCH_RETRAIN ${base} run_dir=${run_dir}" | tee -a "${MASTER_LOG}"
      run_dir=""
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
    if [[ "${EVAL_ONLY}" == "1" ]]; then
      echo "[$(date -Is)] SKIP_EVAL_NO_RUNDIR ${base}" | tee -a "${MASTER_LOG}"
    else
      echo "[$(date -Is)] TRAIN_DONE_NO_RUNDIR ${base}" | tee -a "${MASTER_LOG}"
    fi
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
echo "[$(date -Is)] ${LOG_TAG} sweep complete (${#configs[@]} configs)" | tee -a "${MASTER_LOG}"
