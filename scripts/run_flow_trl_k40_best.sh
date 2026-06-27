#!/usr/bin/env bash
# Sequential K=40 Flow subgoal + TRL best-param follow-up.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}:${ROOT}/scripts:${PYTHONPATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-0}"
PYTHON_BIN="${PYTHON_BIN:-/home/svcho/anaconda3/bin/python}"
WITH_CUDA="${ROOT}/scripts/with_jax_cuda.sh"
LOG_DIR="${ROOT}/nohup_logs"
RUNS_ROOT="${ROOT}/runs"
CONFIG_DIR="${ROOT}/config/sweep_flow_trl_k40_best"
WRITER="${ROOT}/scripts/write_flow_trl_k40_best_yaml.py"
LOG_TAG="flow_trl_k40_best"
FINAL_EPOCH="${FINAL_EPOCH:-600}"

mkdir -p "${LOG_DIR}"
MASTER_LOG="${LOG_DIR}/${LOG_TAG}_master.log"
"${PYTHON_BIN}" "${WRITER}"

echo "[$(date -Is)] ${LOG_TAG} sweep start GPU=${GPU_ID} seed=${SEED}" | tee -a "${MASTER_LOG}"

mapfile -t configs < <(
  CONFIG_DIR="${CONFIG_DIR}" "${PYTHON_BIN}" - <<'PY'
import os

manifest = os.path.join(os.environ['CONFIG_DIR'], '_manifest.txt')
with open(manifest, encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        print(os.path.abspath(line))
PY
)

resolve_run_dir() {
  local cfg="$1"
  local require_checkpoint="$2"
  CONFIG_PATH="${cfg}" RUNS_ROOT="${RUNS_ROOT}" SEED="${SEED}" FINAL_EPOCH="${FINAL_EPOCH}" \
  REQUIRE_CHECKPOINT="${require_checkpoint}" "${PYTHON_BIN}" - <<'PY'
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
        require_checkpoint=os.environ['REQUIRE_CHECKPOINT'] == '1',
    )
)
PY
}

latest_checkpoint_epoch() {
  RUN_DIR="$1" "${PYTHON_BIN}" - <<'PY'
import glob
import os
import re

run_dir = os.environ['RUN_DIR']
epoch_sets = []
for name in ['dynamics', 'critic', 'actor']:
    ckpt_dir = os.path.join(run_dir, 'checkpoints', name)
    epochs = set()
    for path in glob.glob(os.path.join(ckpt_dir, 'params_*.pkl')):
        m = re.search(r'params_(\d+)\.pkl$', os.path.basename(path))
        if m:
            epochs.add(int(m.group(1)))
    epoch_sets.append(epochs)
common = set.intersection(*epoch_sets) if epoch_sets else set()
print(max(common) if common else 0)
PY
}

eval_n_for_config() {
  CONFIG_PATH="$1" "${PYTHON_BIN}" - <<'PY'
import os
import yaml

with open(os.environ['CONFIG_PATH'], encoding='utf-8') as f:
    cfg = yaml.safe_load(f)
print(str(cfg['final_eval_subgoal_eval_num_samples']).split(',')[0])
PY
}

env_for_config() {
  CONFIG_PATH="$1" "${PYTHON_BIN}" - <<'PY'
import os
import yaml

with open(os.environ['CONFIG_PATH'], encoding='utf-8') as f:
    cfg = yaml.safe_load(f)
print(cfg['env_name'])
PY
}

eval_result_exists() {
  local run_dir="$1"
  local eval_n="$2"
  [[ -f "${run_dir}/eval_results/epoch${FINAL_EPOCH}_n${eval_n}.json" ]]
}

run_final_eval() {
  local base="$1"
  local run_dir="$2"
  local eval_n="$3"
  local eval_log="${LOG_DIR}/${LOG_TAG}_${base}.eval_n${eval_n}.log"
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
}

if ((${#configs[@]} == 0)); then
  echo "No configs in ${CONFIG_DIR}" | tee -a "${MASTER_LOG}"
  exit 1
fi

for cfg in "${configs[@]}"; do
  base="$(basename "$cfg" .yaml)"
  env_name="$(env_for_config "${cfg}")"
  eval_n="$(eval_n_for_config "${cfg}")"
  run_dir="$(resolve_run_dir "${cfg}" 1)"

  if [[ -n "${run_dir}" ]] && eval_result_exists "${run_dir}" "${eval_n}"; then
    echo "[$(date -Is)] SKIP_COMPLETE ${base} run_dir=${run_dir}" | tee -a "${MASTER_LOG}"
    continue
  fi

  if [[ -z "${run_dir}" ]]; then
    partial_run_dir="$(resolve_run_dir "${cfg}" 0)"
    train_log="${LOG_DIR}/${LOG_TAG}_${base}.train.log"
    if [[ -n "${partial_run_dir}" ]]; then
      latest_ep="$(latest_checkpoint_epoch "${partial_run_dir}")"
    else
      latest_ep="0"
    fi

    if (( latest_ep > 0 && latest_ep < FINAL_EPOCH )); then
      echo "[$(date -Is)] RESUME_TRAIN ${base} env=${env_name} epoch=${latest_ep} run_dir=${partial_run_dir}" | tee -a "${MASTER_LOG}"
      CUDA_VISIBLE_DEVICES="${GPU_ID}" bash "${WITH_CUDA}" "${PYTHON_BIN}" -u main.py \
        --run_config "${cfg}" \
        --env_name "${env_name}" \
        --seed "${SEED}" \
        --resume_run_dir "${partial_run_dir}" \
        --resume_epoch "${latest_ep}" \
        --async_prefetch \
        --nouse_wandb \
        --nouse_tqdm \
        2>&1 | tee "${train_log}"
    else
      echo "[$(date -Is)] START_TRAIN ${base} env=${env_name}" | tee -a "${MASTER_LOG}"
      CUDA_VISIBLE_DEVICES="${GPU_ID}" bash "${WITH_CUDA}" "${PYTHON_BIN}" -u main.py \
        --run_config "${cfg}" \
        --env_name "${env_name}" \
        --seed "${SEED}" \
        --async_prefetch \
        --nouse_wandb \
        --nouse_tqdm \
        2>&1 | tee "${train_log}"
    fi
    run_dir="$(resolve_run_dir "${cfg}" 1)"
  fi

  if [[ -z "${run_dir}" ]]; then
    echo "[$(date -Is)] TRAIN_DONE_NO_RUNDIR ${base}; skipping eval" | tee -a "${MASTER_LOG}"
    continue
  fi

  if eval_result_exists "${run_dir}" "${eval_n}"; then
    echo "[$(date -Is)] EVAL_ALREADY_COMPLETE ${base} eval_n=${eval_n} run_dir=${run_dir}" | tee -a "${MASTER_LOG}"
  else
    run_final_eval "${base}" "${run_dir}" "${eval_n}"
  fi
done

echo "[$(date -Is)] SUMMARIZE feval results" | tee -a "${MASTER_LOG}"
"${PYTHON_BIN}" "${ROOT}/scripts/summarize_feval_results.py" | tee -a "${MASTER_LOG}"

echo "[$(date -Is)] ${LOG_TAG} sweep complete (${#configs[@]} configs)" | tee -a "${MASTER_LOG}"
