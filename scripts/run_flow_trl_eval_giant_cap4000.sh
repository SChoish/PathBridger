#!/usr/bin/env bash
# Re-eval distinct antmaze-giant ep600 runs at full env cap (4000 env steps).
#
# Skips duplicate configs (e.g. two identical FlowGap10W5 reruns).
# Temperatures: 1.0, 0.5
# eval_n: 2, 8, 16, 32 · 25 episodes/task
#
# Env: GPU_ID, SEED (default 0), EPOCH (default 600), GIANT_ENV_STEPS (default 4000)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH=".:${ROOT}/scripts:${PYTHONPATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-0}"
EPOCH="${EPOCH:-600}"
GIANT_ENV_STEPS="${GIANT_ENV_STEPS:-4000}"
PYTHON_BIN="${PYTHON_BIN:-/home/choi/miniconda3/envs/offrl/bin/python}"
WITH_CUDA="${ROOT}/scripts/with_jax_cuda.sh"
LOG_DIR="${ROOT}/nohup_logs"
LOG_TAG="flow_trl_eval_giant_cap4000"
MASTER_LOG="${LOG_DIR}/${LOG_TAG}_master.log"
DOC_LOG="${LOG_DIR}/${LOG_TAG}_summary.log"

TEMPS=(1.0 0.5)
EVAL_NS=(2 8 16 32)

# run_dir|label|note
JOBS=(
  "checkpoints/flow_trl_best_epoch600/antmaze_giant_amg_g3_w5_n1_evalN8|amg_g3|feval gap=3 h_a=5"
  "runs/20260610_102004_seed0_antmaze-giant-navigate-v0|gap10_w5|FlowGap10W5TRL gap=10 train_N=4 h_a=5"
  "runs/20260621_223959_seed0_antmaze-giant-navigate-v0|h25_ha10|FlowG5W5H25Ha10 gap=5 h_a=10"
)

ha_and_max_chunks() {
  local run_dir="$1"
  RUN_DIR="${run_dir}" GIANT_ENV_STEPS="${GIANT_ENV_STEPS}" "${PYTHON_BIN}" - <<'PY'
import json
import os
from pathlib import Path

run_dir = Path(os.environ["RUN_DIR"])
root = json.loads((run_dir / "flags.json").read_text(encoding="utf-8"))
h = int(root.get("critic_agent", {}).get("action_chunk_horizon", 5))
max_steps = int(os.environ["GIANT_ENV_STEPS"])
chunks = max(1, (max_steps + h - 1) // h)
print(h, chunks)
PY
}

mkdir -p "${LOG_DIR}"
{
  echo "start $(date -Is)"
  echo "GPU=${GPU_ID} epoch=${EPOCH} giant_env_steps=${GIANT_ENV_STEPS}"
  echo "jobs=${#JOBS[@]} temps=${TEMPS[*]} eval_ns=${EVAL_NS[*]}"
  echo "total_eval_jobs=$(( ${#JOBS[@]} * ${#EVAL_NS[@]} * ${#TEMPS[@]} ))"
} | tee "${DOC_LOG}" | tee -a "${MASTER_LOG}"

for spec in "${JOBS[@]}"; do
  IFS='|' read -r run_dir label note <<< "${spec}"
  if [[ ! -f "${run_dir}/flags.json" ]]; then
    echo "SKIP missing flags: ${run_dir}" | tee -a "${DOC_LOG}" "${MASTER_LOG}"
    continue
  fi
  if [[ ! -f "${run_dir}/checkpoints/dynamics/params_${EPOCH}.pkl" ]]; then
    echo "SKIP no ep${EPOCH} checkpoint: ${run_dir}" | tee -a "${DOC_LOG}" "${MASTER_LOG}"
    continue
  fi
  read -r idm_h max_chunks <<< "$(ha_and_max_chunks "${run_dir}")"
  echo "" | tee -a "${DOC_LOG}" "${MASTER_LOG}"
  echo "===== ${label} (${note}) run_dir=${run_dir} h_a=${idm_h} max_chunks=${max_chunks} =====" \
    | tee -a "${DOC_LOG}" "${MASTER_LOG}"

  for eval_n in "${EVAL_NS[@]}"; do
    for temp in "${TEMPS[@]}"; do
      temp_tag="${temp//./p}"
      suffix="cap4000_t${temp_tag}"
      log="${LOG_DIR}/${LOG_TAG}_${label}_n${eval_n}_${suffix}.log"
      echo "--- ${label} eval_n=${eval_n} temp=${temp} ---" | tee -a "${DOC_LOG}" "${MASTER_LOG}"
      CUDA_VISIBLE_DEVICES="${GPU_ID}" bash "${WITH_CUDA}" "${PYTHON_BIN}" -u eval_checkpoint.py \
        --run_dir "${run_dir}" \
        --epoch "${EPOCH}" \
        --seed "${SEED}" \
        --eval_episodes_per_task 25 \
        --eval_max_chunks "${max_chunks}" \
        --idm_action_chunk_horizon "${idm_h}" \
        --subgoal_eval_num_samples "${eval_n}" \
        --subgoal_temperature "${temp}" \
        --eval_result_suffix "${suffix}" \
        --mujoco_gl "${MUJOCO_GL}" \
        --skip_if_saved \
        2>&1 | tee "${log}" | tee -a "${DOC_LOG}" "${MASTER_LOG}"
      echo "[$(date -Is)] DONE ${label} n=${eval_n} temp=${temp}" | tee -a "${MASTER_LOG}"
    done
  done
done

echo "[$(date -Is)] SUMMARIZE feval results" | tee -a "${DOC_LOG}" "${MASTER_LOG}"
"${PYTHON_BIN}" "${ROOT}/scripts/summarize_feval_results.py" | tee -a "${MASTER_LOG}"

echo "[$(date -Is)] ${LOG_TAG} complete" | tee -a "${DOC_LOG}" "${MASTER_LOG}"
