#!/usr/bin/env bash
# 큐브 플레이 IDM/액터 RGB MP4 (manip_play_rollouts 통합 entrypoint).
#
#   ./scripts/rollout_cube_single_cpu_three.sh
#
# 환경 변수:
#   RUN_DIR, CHECKPOINT_EPOCH, TASK_IDS, PYTHON_BIN, MUJOCO_GL
#   IDM_MAX_CHUNKS, ACTOR_MAX_CHUNKS

set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export RUN_DIR="${RUN_DIR:-${ROOT_DIR}/runs/20260503_111509_seed0_cube-single-play-v0}"
export CHECKPOINT_EPOCH="${CHECKPOINT_EPOCH:-1000}"
export TASK_IDS="${TASK_IDS:-1,2,3,4,5}"
exec "${ROOT_DIR}/scripts/rollout_cube_play_cpu_three.sh"
