#!/usr/bin/env bash
# Maze 계열(antmaze-*-navigate, antmaze-*-teleport, humanoidmaze-*) 체크포인트 rollout
# 3종(subgoal/idm/actor)을 CPU(JAX + MuJoCo software GL)로 모두 돌립니다.
#
# 사용 예:
#   RUN_DIR=runs/<run_dir> CHECKPOINT_EPOCH=500 \
#     ./scripts/rollout_maze_cpu_three.sh
#
# 또는 첫 위치 인자로 RUN_DIR을 줘도 됩니다:
#   ./scripts/rollout_maze_cpu_three.sh runs/<run_dir>
#
# 출력 위치: ${RUN_DIR}/${OUT_SUBDIR}/ (기본 OUT_SUBDIR=rollouts_cpu_ep<EP>)
#   - subgoal_task<i>_ep<EP>.png
#   - idm_task<i>_ep<EP>.{png,mp4}
#   - actor_task<i>_ep<EP>.{png,mp4}
#
# 환경 변수:
#   RUN_DIR             (필수)  체크포인트가 있는 학습 run 디렉토리
#   CHECKPOINT_EPOCH    (=1000) rollout 대상 epoch
#   TASK_IDS            (자동)  공백 분리. 비우면 env.num_tasks 기반 1..N
#   SUBGOAL_MAX_STEPS   (=1000) state-space open-loop rollout 최대 step
#   IDM_MAX_STEPS       (=200)  IDM replan cap (~5 env step/chunk)
#   OUT_SUBDIR          (=rollouts_cpu_ep<EP>)
#   PYTHON_BIN, MUJOCO_GL
#
# Actor replan budget은 env TimeLimit에서 자동 산출됩니다.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
export PYTHONPATH=.

PYTHON_BIN="${PYTHON_BIN:-/home/choi/miniconda3/envs/offrl/bin/python}"

export JAX_PLATFORM_NAME=cpu
export JAX_PLATFORMS=cpu
export CUDA_VISIBLE_DEVICES=""
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"

RUN_DIR="${RUN_DIR:-${1:-}}"
if [[ -z "${RUN_DIR}" ]]; then
  echo "RUN_DIR이 비어 있습니다. 환경변수나 첫 위치 인자로 전달하세요." >&2
  exit 1
fi
RUN_DIR="$(cd "${RUN_DIR}" && pwd)"

CHECKPOINT_EPOCH="${CHECKPOINT_EPOCH:-1000}"
OUT_SUBDIR="${OUT_SUBDIR:-rollouts_cpu_ep${CHECKPOINT_EPOCH}}"
OUT_BASE="${RUN_DIR}/${OUT_SUBDIR}"
mkdir -p "${OUT_BASE}"

SUBGOAL_MAX_STEPS="${SUBGOAL_MAX_STEPS:-1000}"
IDM_MAX_STEPS="${IDM_MAX_STEPS:-200}"

if [[ ! -f "${RUN_DIR}/config_used.yaml" ]]; then
  echo "missing ${RUN_DIR}/config_used.yaml" >&2
  exit 1
fi

# Default task ids 1..N (OGBench multi-task)
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
echo "SUBGOAL_MAX_STEPS=${SUBGOAL_MAX_STEPS}  IDM_MAX_STEPS=${IDM_MAX_STEPS}  ACTOR_MAX_CHUNKS=auto(env TimeLimit)"

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
    --out_path="${OUT_BASE}/idm_task${TASK_ID}_ep${CHECKPOINT_EPOCH}.png" \
    --out_mp4="${OUT_BASE}/idm_task${TASK_ID}_ep${CHECKPOINT_EPOCH}.mp4" \
    --fps=30 \
    --no-value_heatmap \
    --mujoco_gl="${MUJOCO_GL}"

  echo "=== actor (env) ==="
  "${PYTHON_BIN}" rollout/actor.py \
    --run_dir="${RUN_DIR}" \
    --checkpoint_epoch="${CHECKPOINT_EPOCH}" \
    --task_id="${TASK_ID}" \
    --out_path="${OUT_BASE}/actor_task${TASK_ID}_ep${CHECKPOINT_EPOCH}.png" \
    --out_mp4="${OUT_BASE}/actor_task${TASK_ID}_ep${CHECKPOINT_EPOCH}.mp4" \
    --fps=30 \
    --no-value_heatmap \
    --mujoco_gl="${MUJOCO_GL}"
done

echo "done: ls ${OUT_BASE}"
ls -la "${OUT_BASE}"
