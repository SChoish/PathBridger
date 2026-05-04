#!/usr/bin/env bash
# OGBench cube-*-play 전용: 상태공간(큐브 축) PNG + IDM/액터 **환경 RGB** MP4.
# (기존 rollout/subgoal.py·idm.py·actor.py 는 미로/qpos 동기화 전제라 큐브에 그대로 쓰지 않음.)
#
#   RUN_DIR=runs/... CHECKPOINT_EPOCH=1000 ./scripts/rollout_cube_play_cpu_three.sh
#
# 환경 변수: RUN_DIR, CHECKPOINT_EPOCH, TASK_IDS (기본 1–5), PYTHON_BIN, MUJOCO_GL

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON_BIN:-/home/choi/miniconda3/envs/offrl/bin/python}"
export JAX_PLATFORM_NAME=cpu
export JAX_PLATFORMS=cpu
export CUDA_VISIBLE_DEVICES=""
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"

RUN_DIR="${RUN_DIR:-${ROOT_DIR}/runs/20260503_111509_seed0_cube-single-play-v0}"
CHECKPOINT_EPOCH="${CHECKPOINT_EPOCH:-1000}"
TASK_IDS="${TASK_IDS:-1,2,3,4,5}"

"${PYTHON_BIN}" -m rollout.manip_play_rollouts \
  --run_dir="${RUN_DIR}" \
  --checkpoint_epoch="${CHECKPOINT_EPOCH}" \
  --task_ids="${TASK_IDS}" \
  --mujoco_gl="${MUJOCO_GL}"

echo "outputs under ${RUN_DIR}/rollouts_manip_<env_slug>_ep${CHECKPOINT_EPOCH}/task*/"
