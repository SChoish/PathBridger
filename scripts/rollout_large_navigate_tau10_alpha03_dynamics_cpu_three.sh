#!/usr/bin/env bash
# AntMaze **large** navigate — **τ=10, α=0.3** (300-epoch checkpoints 100/200/300).
#
# Linear-dynamics sweep cell (`run_group: antmaze_navigate_dynamics_tau_alpha_sweep`) for τ10/α0.3 is
# `runs/20260426_165239_joint_dqc_seed0_antmaze-large-navigate-v0`, but that run stopped early and has
# **no** `params_*.pkl` under `checkpoints/goub/`. Re-train or set RUN_DIR after checkpoints exist.
#
# Default below: `runs/20260425_224314_*` — same τ/α with `bridge_type: theta_linear` (has checkpoints).
#
# Three rollout kinds on **CPU**, **all tasks** 1..N (default N from env `num_tasks`).
#
#   ./scripts/rollout_large_navigate_tau10_alpha03_dynamics_cpu_three.sh
#
# Optional env: RUN_DIR, CHECKPOINT_EPOCH (default 300), TASK_IDS, SUBGOAL_MAX_STEPS (default 1000),
#   IDM_MAX_STEPS, ACTOR_MAX_CHUNKS, OUT_SUBDIR

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
export PYTHONPATH=.

PYTHON_BIN="${PYTHON_BIN:-/home/choi/miniconda3/envs/offrl/bin/python}"

export JAX_PLATFORM_NAME=cpu
export JAX_PLATFORMS=cpu
export CUDA_VISIBLE_DEVICES=""
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MUJOCO_GL=osmesa

RUN_DIR="${RUN_DIR:-${ROOT_DIR}/runs/20260425_224314_joint_dqc_seed0_antmaze-large-navigate-v0}"
CHECKPOINT_EPOCH="${CHECKPOINT_EPOCH:-300}"
OUT_SUBDIR="${OUT_SUBDIR:-rollouts_cpu_ep${CHECKPOINT_EPOCH}}"
OUT_BASE="${RUN_DIR}/${OUT_SUBDIR}"
mkdir -p "${OUT_BASE}"

SUBGOAL_MAX_STEPS="${SUBGOAL_MAX_STEPS:-1000}"
IDM_MAX_STEPS="${IDM_MAX_STEPS:-200}"
ACTOR_MAX_CHUNKS="${ACTOR_MAX_CHUNKS:-200}"

if [[ ! -f "${RUN_DIR}/config_used.yaml" ]]; then
  echo "missing ${RUN_DIR}/config_used.yaml" >&2
  exit 1
fi

if [[ -n "${TASK_IDS:-}" ]]; then
  read -r -a TASK_ARR <<< "${TASK_IDS}"
else
  N_TASKS="$("${PYTHON_BIN}" - <<PY
import yaml
from pathlib import Path
from utils.env_utils import make_env_and_datasets

p = Path("${RUN_DIR}") / "config_used.yaml"
cfg = yaml.safe_load(p.read_text())
name = str(cfg.get("env_name") or "")
fs = (cfg.get("critic_agent") or {}).get("frame_stack")
env, _, _ = make_env_and_datasets(name, frame_stack=fs)
print(int(getattr(env.unwrapped, "num_tasks", 5)))
PY
)"
  TASK_ARR=()
  for ((i = 1; i <= N_TASKS; i++)); do
    TASK_ARR+=("$i")
  done
fi

echo "RUN_DIR=${RUN_DIR}"
echo "OUT_BASE=${OUT_BASE}  tasks=${TASK_ARR[*]}  epoch=${CHECKPOINT_EPOCH}"
echo "SUBGOAL_MAX_STEPS=${SUBGOAL_MAX_STEPS}  IDM_MAX_STEPS=${IDM_MAX_STEPS}  ACTOR_MAX_CHUNKS=${ACTOR_MAX_CHUNKS}"

for TASK_ID in "${TASK_ARR[@]}"; do
  echo "######## task_id=${TASK_ID} ########"

  echo "=== subgoal (state space, max_steps=${SUBGOAL_MAX_STEPS}) ==="
  "${PYTHON_BIN}" rollout/subgoal.py \
    --run_dir="${RUN_DIR}" \
    --checkpoint_epoch="${CHECKPOINT_EPOCH}" \
    --task_id="${TASK_ID}" \
    --max_steps="${SUBGOAL_MAX_STEPS}" \
    --out_path="${OUT_BASE}/subgoal_task${TASK_ID}_ep${CHECKPOINT_EPOCH}.png" \
    --no_mp4 \
    --no-value_heatmap

  echo "=== idm (env) ==="
  "${PYTHON_BIN}" rollout/idm.py \
    --run_dir="${RUN_DIR}" \
    --checkpoint_epoch="${CHECKPOINT_EPOCH}" \
    --task_id="${TASK_ID}" \
    --max_steps="${IDM_MAX_STEPS}" \
    --navigator=snap \
    --out_path="${OUT_BASE}/idm_task${TASK_ID}_ep${CHECKPOINT_EPOCH}.png" \
    --out_mp4="${OUT_BASE}/idm_task${TASK_ID}_ep${CHECKPOINT_EPOCH}.mp4" \
    --fps=30 \
    --no-value_heatmap \
    --mujoco_gl=osmesa

  echo "=== actor (env) ==="
  "${PYTHON_BIN}" rollout/actor.py \
    --run_dir="${RUN_DIR}" \
    --checkpoint_epoch="${CHECKPOINT_EPOCH}" \
    --task_id="${TASK_ID}" \
    --max_chunks="${ACTOR_MAX_CHUNKS}" \
    --navigator=snap \
    --out_path="${OUT_BASE}/actor_task${TASK_ID}_ep${CHECKPOINT_EPOCH}.png" \
    --out_mp4="${OUT_BASE}/actor_task${TASK_ID}_ep${CHECKPOINT_EPOCH}.mp4" \
    --fps=30 \
    --no-value_heatmap \
    --mujoco_gl=osmesa
done

echo "done: ls ${OUT_BASE}"
ls -la "${OUT_BASE}"
