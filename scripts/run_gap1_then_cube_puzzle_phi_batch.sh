#!/usr/bin/env bash
# Sequential training (offrl JAX env):
#   1) Resume three runs to train_epochs=1000 (from last saved dynamics checkpoint, or epoch 0 if none).
#   2) Fresh 1000-epoch runs for puzzle-*-play envs (phi recipe YAMLs).
#
# Resume targets (under repo runs/):
#   - antmaze-giant (gap5 run 182720)
#   - cube-double (213424)
#   - cube-triple (223109; was interrupted → may resume from 0 in same dir)
#
# Override: TRAIN_EPOCHS=1000 PYTHON=... ./scripts/run_gap1_then_cube_puzzle_phi_batch.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

if [[ -n "${PYTHON:-}" ]]; then
  PY="${PYTHON}"
elif [[ -x "${HOME}/miniconda3/envs/offrl/bin/python" ]]; then
  PY="${HOME}/miniconda3/envs/offrl/bin/python"
else
  PY="python3"
fi

RUNS_ROOT="${RUNS_ROOT:-$ROOT/runs}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-1000}"

# Largest dynamics checkpoint suffix in run_dir (0 if none).
max_saved_epoch() {
  local rd="$1/checkpoints/dynamics"
  local best=0
  shopt -s nullglob
  local f n
  for f in "$rd"/params_*.pkl; do
    n="${f##*/params_}"
    n="${n%.pkl}"
    [[ "$n" =~ ^[0-9]+$ ]] || continue
    if (( n > best )); then best=$n; fi
  done
  shopt -u nullglob
  echo "$best"
}

resume_to_total_epochs() {
  local rel="$1"
  local run_dir="$RUNS_ROOT/$rel"
  if [[ ! -d "$run_dir" ]]; then
    echo "ERROR: missing run_dir: $run_dir" >&2
    return 1
  fi
  local ep
  ep="$(max_saved_epoch "$run_dir")"
  echo "=== resume rel=$rel last_ckpt_epoch=$ep -> train_epochs=$TRAIN_EPOCHS ==="
  "$PY" main.py \
    --resume_run_dir="$run_dir" \
    --resume_epoch="$ep" \
    --train_epochs="$TRAIN_EPOCHS"
}

echo "=== Phase A: resume listed runs to ${TRAIN_EPOCHS} epochs ==="
resume_to_total_epochs "20260510_182720_seed0_antmaze-giant-navigate-v0"
resume_to_total_epochs "20260510_213424_seed0_cube-double-play-v0"
resume_to_total_epochs "20260510_223109_seed0_cube-triple-play-v0"

# 퍼즐 phi YAML: 베이스와 맞추는 건 critic kappa_b/kappa_d 뿐(각 config/puzzle_*.yaml). 나머지는 phi 레시피 그대로.
PUZZLE_CFGS=(
  config/puzzle_3x3_phi_u4_nll_alpha03_gap20_envgoal_300.yaml
  config/puzzle_4x4_phi_u4_nll_alpha03_gap20_envgoal_300.yaml
  config/puzzle_4x5_phi_u4_nll_alpha03_gap20_envgoal_300.yaml
  config/puzzle_4x6_phi_u4_nll_alpha03_gap20_envgoal_300.yaml
)

echo "=== Phase B: puzzle envs, ${TRAIN_EPOCHS} epochs (fresh run dirs) ==="
for cfg in "${PUZZLE_CFGS[@]}"; do
  echo "=== $cfg ==="
  "$PY" main.py --run_config="$cfg" --train_epochs="$TRAIN_EPOCHS" --seed=0
done

echo "All jobs finished."
