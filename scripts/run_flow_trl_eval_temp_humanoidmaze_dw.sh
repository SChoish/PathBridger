#!/usr/bin/env bash
# Temperature ablation (0.5, 0.25) on completed humanoidmaze_dw sweep runs (ep600).
# Uses best ACTOR eval_n from cap2000 feval per config.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH=".:${ROOT}/scripts:${PYTHONPATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-0}"
EPOCH="${EPOCH:-600}"
PYTHON_BIN="${PYTHON_BIN:-/home/choi/miniconda3/envs/offrl/bin/python}"
WITH_CUDA="${ROOT}/scripts/with_jax_cuda.sh"
LOG_DIR="${ROOT}/nohup_logs"
LOG_TAG="flow_trl_eval_temp_hmm_dw"
MASTER_LOG="${LOG_DIR}/${LOG_TAG}_master.log"
DOC_LOG="${LOG_DIR}/${LOG_TAG}_summary.log"
CAP_SUFFIX="cap2000"

# run_dir|label|eval_n|eval_max_chunks
JOBS=(
  "runs/20260627_155820_seed0_humanoidmaze-medium-navigate-v0|hmm_g5_dwp0|8|400"
  "runs/20260628_010448_seed0_humanoidmaze-medium-navigate-v0|hmm_g5_dwp0p1|32|400"
  "runs/20260626_160557_seed0_humanoidmaze-medium-navigate-v0|hmm_g10_dwp0|32|400"
  "runs/20260627_065306_seed0_humanoidmaze-medium-navigate-v0|hmm_g10_dwp0p1|32|400"
  "runs/20260626_022012_seed0_humanoidmaze-large-navigate-v0|hml_g5_dwp0|32|400"
  "runs/20260626_091305_seed0_humanoidmaze-large-navigate-v0|hml_g5_dwp0p1|32|400"
  "runs/20260625_123122_seed0_humanoidmaze-large-navigate-v0|hml_g10_dwp0|16|400"
  "runs/20260625_192606_seed0_humanoidmaze-large-navigate-v0|hml_g10_dwp0p1|32|400"
)

TEMPS=(0.5 0.25)

mkdir -p "${LOG_DIR}"
{
  echo "start $(date -Is)"
  echo "GPU=${GPU_ID} epoch=${EPOCH} temps=${TEMPS[*]} cap=${CAP_SUFFIX}"
  echo "jobs=${#JOBS[@]} total_evals=$(( ${#JOBS[@]} * ${#TEMPS[@]} ))"
} | tee "${DOC_LOG}" | tee -a "${MASTER_LOG}"

for spec in "${JOBS[@]}"; do
  IFS='|' read -r run_dir label eval_n max_chunks <<< "${spec}"
  if [[ ! -f "${run_dir}/checkpoints/dynamics/params_${EPOCH}.pkl" ]]; then
    echo "SKIP no ep${EPOCH} ckpt: ${run_dir}" | tee -a "${DOC_LOG}" "${MASTER_LOG}"
    continue
  fi
  for temp in "${TEMPS[@]}"; do
    temp_tag="${temp//./p}"
    suffix="${CAP_SUFFIX}_t${temp_tag}"
    log="${LOG_DIR}/${LOG_TAG}_${label}_n${eval_n}_${suffix}.log"
    echo "" | tee -a "${DOC_LOG}" "${MASTER_LOG}"
    echo "===== ${label} temp=${temp} eval_n=${eval_n} max_chunks=${max_chunks} suffix=${suffix} =====" \
      | tee -a "${DOC_LOG}" "${MASTER_LOG}"
    CUDA_VISIBLE_DEVICES="${GPU_ID}" bash "${WITH_CUDA}" "${PYTHON_BIN}" -u eval_checkpoint.py \
      --run_dir "${run_dir}" \
      --epoch "${EPOCH}" \
      --seed "${SEED}" \
      --eval_episodes_per_task 25 \
      --eval_max_chunks "${max_chunks}" \
      --subgoal_eval_num_samples "${eval_n}" \
      --subgoal_temperature "${temp}" \
      --eval_result_suffix "${suffix}" \
      --mujoco_gl "${MUJOCO_GL}" \
      --skip_if_saved \
      2>&1 | tee "${log}" | tee -a "${DOC_LOG}" "${MASTER_LOG}"
    echo "[$(date -Is)] DONE ${label} temp=${temp} n=${eval_n}" | tee -a "${MASTER_LOG}"
  done
done

echo "[$(date -Is)] ${LOG_TAG} complete" | tee -a "${DOC_LOG}" "${MASTER_LOG}"
