#!/usr/bin/env bash
# Post-train eval with V(z,g) scoring (not V*V/V): N=4,8,16,32 on flow cube checkpoints.
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
EVAL_NS=(4 8 16 32)

RUNS=(
  "runs/20260617_233620_seed0_cube-double-play-v0|cube_double"
  "runs/20260618_025240_seed0_cube-triple-play-v0|cube_triple"
)

mkdir -p "${LOG_DIR}"
MASTER_LOG="${LOG_DIR}/flow_cube_vzg_eval_master.log"
echo "[$(date -Is)] V(z,g) eval start GPU=${GPU_ID} epoch=${EPOCH} N=${EVAL_NS[*]}" | tee -a "${MASTER_LOG}"

for spec in "${RUNS[@]}"; do
  run_dir="${spec%%|*}"
  base="${spec##*|}"
  for eval_n in "${EVAL_NS[@]}"; do
    log="${LOG_DIR}/flow_cube_vzg_${base}_n${eval_n}.log"
    echo "[$(date -Is)] START ${base} eval_n=${eval_n} score=goal_value" | tee -a "${MASTER_LOG}"
    CUDA_VISIBLE_DEVICES="${GPU_ID}" bash "${WITH_CUDA}" "${PYTHON_BIN}" -u eval_checkpoint.py \
      --run_dir "${run_dir}" \
      --epoch "${EPOCH}" \
      --seed "${SEED}" \
      --eval_episodes_per_task 25 \
      --subgoal_eval_num_samples "${eval_n}" \
      --subgoal_eval_selection best_of_n_value \
      --subgoal_eval_score_type goal_value \
      --mujoco_gl "${MUJOCO_GL}" \
      --skip_if_saved \
      2>&1 | tee "${log}" | tee -a "${MASTER_LOG}"
    echo "[$(date -Is)] DONE ${base} eval_n=${eval_n}" | tee -a "${MASTER_LOG}"
  done
done

echo "[$(date -Is)] ALL DONE" | tee -a "${MASTER_LOG}"
