#!/usr/bin/env bash
# Fork new run dir: copy epoch-500 checkpoints from 20260427_101011 (plan_candidates=1),
# then train with plan_candidates=8 to train_epochs (default 1000).
#
# main.py always resumes in-place; there is no "new dir" flag, so we copy ckpts into a
# fresh runs/<ts>_..._resume_pc8_from101011/ and pass --run_config=...plan_candidates_8.yaml
# so hyperparameters (incl. plan_candidates) come from YAML, not the old flags.json snapshot.
#
# Usage:
#   cd /path/to/douri && ./scripts/launch_giant_plan8_from_101011_ep500.sh
#   ./scripts/launch_giant_plan8_from_101011_ep500.sh --nohup
#
# Optional: SOURCE_RUN_DIR, RESUME_EPOCH, PYTHON_BIN, MUJOCO_GL, LOG_DIR

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/nohup_logs}"
mkdir -p "${LOG_DIR}"

if [[ "${1:-}" == "--nohup" ]]; then
  shift
  LAUNCH_LOG="${LOG_DIR}/giant_plan8_from101011_ep500_$(date +%Y%m%d_%H%M%S).log"
  cd "${ROOT_DIR}"
  nohup env -u JAX_PLATFORM_NAME -u JAX_PLATFORMS -u CUDA_VISIBLE_DEVICES \
    PYTHONPATH=. \
    MUJOCO_GL="${MUJOCO_GL:-egl}" \
    PYTHON_BIN="${PYTHON_BIN:-/home/choi/miniconda3/envs/offrl/bin/python}" \
    bash "${ROOT_DIR}/scripts/launch_giant_plan8_from_101011_ep500.sh" "$@" >>"${LAUNCH_LOG}" 2>&1 &
  echo "nohup started PID=$!  log=${LAUNCH_LOG}"
  exit 0
fi

cd "${ROOT_DIR}"
export PYTHONPATH=.
unset JAX_PLATFORM_NAME JAX_PLATFORMS CUDA_VISIBLE_DEVICES 2>/dev/null || true

PYTHON_BIN="${PYTHON_BIN:-/home/choi/miniconda3/envs/offrl/bin/python}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

SOURCE_RUN_DIR="${SOURCE_RUN_DIR:-${ROOT_DIR}/runs/20260427_101011_joint_dqc_seed0_antmaze-giant-navigate-v0}"
RESUME_EPOCH="${RESUME_EPOCH:-500}"
CFG="config/sweep_dynamics_tau_alpha/antmaze_giant_navigate_dynamics_tau10_a0p3_disc0p995_plan_candidates_8.yaml"

for sub in goub critic actor; do
  if [[ ! -f "${SOURCE_RUN_DIR}/checkpoints/${sub}/params_${RESUME_EPOCH}.pkl" ]]; then
    echo "missing checkpoint: ${SOURCE_RUN_DIR}/checkpoints/${sub}/params_${RESUME_EPOCH}.pkl" >&2
    exit 1
  fi
done

ts="$(date +%Y%m%d_%H%M%S)"
NEW_RUN_DIR="${ROOT_DIR}/runs/${ts}_joint_dqc_seed0_antmaze-giant-navigate-v0_resume_pc8_from101011"
mkdir -p "${NEW_RUN_DIR}/checkpoints/goub" "${NEW_RUN_DIR}/checkpoints/critic" "${NEW_RUN_DIR}/checkpoints/actor"

for sub in goub critic actor; do
  cp -a "${SOURCE_RUN_DIR}/checkpoints/${sub}/params_${RESUME_EPOCH}.pkl" "${NEW_RUN_DIR}/checkpoints/${sub}/"
done

echo "new_run_dir=${NEW_RUN_DIR}"
echo "resume_epoch=${RESUME_EPOCH}  plan_candidates=8  source=${SOURCE_RUN_DIR}"

"${PYTHON_BIN}" "${ROOT_DIR}/main.py" \
  --train_epochs=1000 \
  --save_every_n_epochs=100 \
  --eval_freq=100 \
  --resume_run_dir="${NEW_RUN_DIR}" \
  --resume_epoch="${RESUME_EPOCH}" \
  --run_config="${CFG}"
