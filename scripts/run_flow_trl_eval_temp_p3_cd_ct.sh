#!/usr/bin/env bash
# Temperature ablation (0.5, 0.25) on best-params checkpoints + gap-tune runs for p3/cd/ct.
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
LOG_TAG="flow_trl_eval_temp_p3_cd_ct"
MASTER_LOG="${LOG_DIR}/${LOG_TAG}_master.log"
DOC_LOG="${ROOT}/docs/flow_trl_eval_temp_p3_cd_ct.log"

# run_dir|label|eval_n
JOBS=(
  "checkpoints/flow_trl_best_epoch600/puzzle_3x3_p3_g1_w5_n1_evalN32|best_p3_g1|32"
  "checkpoints/flow_trl_best_epoch600/cube_double_cd_g10_w5_n1_evalN8|best_cd_g10|8"
  "checkpoints/flow_trl_best_epoch600/cube_triple_ct_g10_w5_n1_evalN8|best_ct_g10|8"
  "runs/20260623_225006_seed0_puzzle-3x3-play-v0|tune_p3_g0p5|32"
  "runs/20260624_051355_seed0_cube-double-play-v0|tune_cd_g20|8"
  "runs/20260624_082614_seed0_cube-triple-play-v0|tune_ct_g20|16"
)

TEMPS=(0.5 0.25)

mkdir -p "${LOG_DIR}"
{
  echo "start $(date -Is)"
  echo "GPU=${GPU_ID} epoch=${EPOCH} temps=${TEMPS[*]}"
} | tee "${DOC_LOG}" | tee -a "${MASTER_LOG}"

for spec in "${JOBS[@]}"; do
  IFS='|' read -r run_dir label eval_n <<< "${spec}"
  for temp in "${TEMPS[@]}"; do
    suffix="t${temp//./p}"
    log="${LOG_DIR}/${LOG_TAG}_${label}_${suffix}.log"
    echo "" | tee -a "${DOC_LOG}" "${MASTER_LOG}"
    echo "===== ${label} temp=${temp} eval_n=${eval_n} =====" | tee -a "${DOC_LOG}" "${MASTER_LOG}"
    CUDA_VISIBLE_DEVICES="${GPU_ID}" bash "${WITH_CUDA}" "${PYTHON_BIN}" -u eval_checkpoint.py \
      --run_dir "${run_dir}" \
      --epoch "${EPOCH}" \
      --seed "${SEED}" \
      --eval_episodes_per_task 25 \
      --subgoal_eval_num_samples "${eval_n}" \
      --subgoal_temperature "${temp}" \
      --eval_result_suffix "${suffix}" \
      --mujoco_gl "${MUJOCO_GL}" \
      --skip_if_saved \
      2>&1 | tee "${log}" | tee -a "${DOC_LOG}" "${MASTER_LOG}"
    echo "[$(date -Is)] DONE ${label} temp=${temp}" | tee -a "${MASTER_LOG}"
  done
done

echo "[$(date -Is)] ${LOG_TAG} complete" | tee -a "${DOC_LOG}" "${MASTER_LOG}"
