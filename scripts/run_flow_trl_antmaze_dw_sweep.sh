#!/usr/bin/env bash
# Train + final eval: antmaze medium/large × gap {5,10} × distance_weight {0.0,0.1}.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}:${ROOT}/scripts:${PYTHONPATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-0}"
EVAL_ONLY="${EVAL_ONLY:-0}"
PYTHON_BIN="${PYTHON_BIN:-/home/choi/miniconda3/envs/offrl/bin/python}"
WITH_CUDA="${ROOT}/scripts/with_jax_cuda.sh"
LOG_DIR="${ROOT}/nohup_logs"
RUNS_ROOT="${ROOT}/runs"
CONFIG_DIR="${ROOT}/config/sweep_flow_trl_antmaze_dw"
FINAL_EPOCH="${FINAL_EPOCH:-600}"
LOG_TAG="flow_trl_antmaze_dw"
FINAL_EVAL_NS=(2 8 16 32)

mkdir -p "${LOG_DIR}"
MASTER_LOG="${LOG_DIR}/${LOG_TAG}_master.log"

"${PYTHON_BIN}" "${ROOT}/scripts/generate_flow_trl_antmaze_dw_sweep_configs.py"

mode_label="train+eval"
if [[ "${EVAL_ONLY}" == "1" ]]; then
  mode_label="eval-only"
fi
echo "[$(date -Is)] ${LOG_TAG} start mode=${mode_label} GPU=${GPU_ID} seed=${SEED} final_eval_N=${FINAL_EVAL_NS[*]}" | tee -a "${MASTER_LOG}"

mapfile -t configs < <(find "${CONFIG_DIR}" -maxdepth 1 -name '*.yaml' ! -name '_*' | sort)

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

find_train_state() {
  local cfg="$1"
  CONFIG_PATH="${cfg}" RUNS_ROOT="${RUNS_ROOT}" SEED="${SEED}" FINAL_EPOCH="${FINAL_EPOCH}" \
    "${PYTHON_BIN}" - <<'PY'
import os
import sys

sys.path.insert(0, os.path.join(os.getcwd(), 'scripts'))
from flow_trl_sweep_common import find_run_dir_for_config, resolve_train_target
import yaml

cfg = yaml.safe_load(open(os.environ['CONFIG_PATH'], encoding='utf-8'))
target_step = resolve_train_target(cfg)
if int(cfg.get('train_steps', 0) or 0) <= 0:
    target_step = int(os.environ.get('FINAL_EPOCH', target_step))

run_dir, latest_step = find_run_dir_for_config(
    config_path=os.environ['CONFIG_PATH'],
    runs_root=os.environ['RUNS_ROOT'],
    seed=int(os.environ['SEED']),
    final_step=target_step,
)
if not run_dir:
    print('fresh||0')
elif latest_step >= target_step:
    print(f'finished|{run_dir}|{latest_step}')
else:
    print(f'resume|{run_dir}|{latest_step}')
PY
}

eval_done() {
  local run_dir="$1"
  local eval_n="$2"
  [[ -f "${run_dir}/eval_results/epoch${FINAL_EPOCH}_n${eval_n}.json" ]]
}

eval_all_done() {
  local run_dir="$1"
  local eval_n
  for eval_n in "${FINAL_EVAL_NS[@]}"; do
    if ! eval_done "${run_dir}" "${eval_n}"; then
      return 1
    fi
  done
}

run_final_evals() {
  local base="$1"
  local run_dir="$2"
  local eval_n
  for eval_n in "${FINAL_EVAL_NS[@]}"; do
    if eval_done "${run_dir}" "${eval_n}"; then
      echo "[$(date -Is)] SKIP_EVAL_SAVED ${base} eval_n=${eval_n}" | tee -a "${MASTER_LOG}"
      continue
    fi
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

for cfg in "${configs[@]}"; do
  base="$(basename "$cfg" .yaml)"
  env_name="$("${PYTHON_BIN}" -c "import yaml; print(yaml.safe_load(open('${cfg}'))['env_name'])")"
  read -r train_state partial_run_dir resume_epoch <<< "$(find_train_state "${cfg}" | tr '|' ' ')"
  run_dir=""
  if [[ "${train_state}" == "finished" ]]; then
    run_dir="${partial_run_dir}"
  fi

  if [[ "${EVAL_ONLY}" != "1" ]]; then
    if [[ -n "${run_dir}" ]] && eval_all_done "${run_dir}"; then
      echo "[$(date -Is)] SKIP_COMPLETE ${base} run_dir=${run_dir}" | tee -a "${MASTER_LOG}"
      continue
    fi
    train_log="${LOG_DIR}/${LOG_TAG}_${base}.train.log"
    if [[ "${train_state}" == "resume" ]]; then
      echo "[$(date -Is)] RESUME_TRAIN ${base} env=${env_name} run_dir=${partial_run_dir} epoch=${resume_epoch}" | tee -a "${MASTER_LOG}"
      CUDA_VISIBLE_DEVICES="${GPU_ID}" bash "${WITH_CUDA}" "${PYTHON_BIN}" -u main.py \
        --run_config "${cfg}" \
        --env_name "${env_name}" \
        --seed "${SEED}" \
        --resume_run_dir "${partial_run_dir}" \
        --resume_epoch "${resume_epoch}" \
        --async_prefetch \
        --nouse_wandb \
        --nouse_tqdm \
        2>&1 | tee -a "${train_log}"
      run_dir="${partial_run_dir}"
    elif [[ "${train_state}" == "finished" ]]; then
      echo "[$(date -Is)] SKIP_TRAIN_HAVE_CKPT ${base} run_dir=${run_dir}" | tee -a "${MASTER_LOG}"
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
      run_dir="$(resolve_run_dir "${cfg}")"
    fi
  else
    run_dir="$(resolve_run_dir "${cfg}")"
    if [[ -z "${run_dir}" && "${train_state}" == "resume" ]]; then
      run_dir="${partial_run_dir}"
    fi
  fi

  if [[ -z "${run_dir}" ]]; then
    echo "[$(date -Is)] SKIP_EVAL_NO_RUNDIR ${base}" | tee -a "${MASTER_LOG}"
    continue
  fi
  if eval_all_done "${run_dir}"; then
    echo "[$(date -Is)] EVAL_ALREADY_COMPLETE ${base} run_dir=${run_dir}" | tee -a "${MASTER_LOG}"
    continue
  fi

  run_final_evals "${base}" "${run_dir}"
done

echo "[$(date -Is)] ${LOG_TAG} complete (${#configs[@]} configs)" | tee -a "${MASTER_LOG}"
