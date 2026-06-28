#!/usr/bin/env bash
# Sequentially run the state-only-offline + online hybrid ONLINE phase for the
# three cube envs, loading the 600-epoch flow_subgoal checkpoints. Each run does
# 300K online updates with eval every 100K (see config/hybrid_online_cube_*.yaml).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PYTHON="${PYTHON:-/home/offrl/miniconda3/envs/offrl/bin/python}"
export PYTHONPATH=.
export MUJOCO_GL=egl
# shellcheck disable=SC1091
source scripts/jax_cuda_env.sh

PY="$PYTHON"
CKPT_ROOT="${CKPT_ROOT:-/home/offrl/Pathbridger_hybrid/runs/flow_subgoal/checkpoints}"
SEED="${SEED:-0}"
STEP="${STEP:-600}"

ENVS=(cube-single-play-v0 cube-double-play-v0 cube-triple-play-v0)
TOKS=(cube_single_play cube_double_play cube_triple_play)

for i in "${!ENVS[@]}"; do
  envname="${ENVS[$i]}"
  tok="${TOKS[$i]}"
  cfg="config/hybrid_online_${tok}.yaml"
  off="${CKPT_ROOT}/${envname}"
  echo "=== [$(date '+%F %T')] ONLINE HYBRID start env=${envname} cfg=${cfg} off=${off} ==="
  "$PY" main.py \
    --run_config "$cfg" \
    --hybrid_phase online \
    --offline_run_dir "$off" \
    --offline_load_step "$STEP" \
    --seed "$SEED"
  echo "=== [$(date '+%F %T')] ONLINE HYBRID done env=${envname} ==="
done
echo "=== [$(date '+%F %T')] ALL CUBE ONLINE HYBRID RUNS DONE ==="
