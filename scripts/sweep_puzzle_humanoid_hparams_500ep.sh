#!/usr/bin/env bash
# Sweep kappa_b, kappa_d, subgoal_value_alpha, subgoal_value_gap_scale for
# puzzle-3x3-play-v0 and humanoidmaze-medium-navigate-v0 (500 epochs each, sequential).
#
# Logs: scripts/sweep_logs/sweep_master_*.log and per-run tee under the same dir.
#
# Usage:
#   bash scripts/sweep_puzzle_humanoid_hparams_500ep.sh
# Optional: CUDA_VISIBLE_DEVICES=0 bash scripts/...

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-/home/offrl/miniconda3/envs/offrl/bin/python}"
GEN="$REPO_ROOT/scripts/write_sweep_run_yaml.py"
OUTDIR="$REPO_ROOT/scripts/sweep_generated"
LOGDIR="$REPO_ROOT/scripts/sweep_logs"
TS="$(date +%Y%m%d_%H%M%S)"
MASTER_LOG="$LOGDIR/sweep_master_${TS}.log"

mkdir -p "$OUTDIR" "$LOGDIR"

if [[ ! -x "$PYTHON" ]]; then
  echo "Python not found at $PYTHON; set PYTHON=..." | tee -a "$MASTER_LOG"
  exit 1
fi

run_one () {
  local family="$1" tag="$2" kb="$3" kd="$4" alpha="$5" gap="$6"
  local ypath="$OUTDIR/${family}_${tag}.yaml"
  local rlog="$LOGDIR/${family}_${tag}_${TS}.log"
  echo "=== $(date -Is) START $family $tag kb=$kb kd=$kd alpha=$alpha gap=$gap ===" | tee -a "$MASTER_LOG"
  "$PYTHON" "$GEN" --family "$family" --out "$ypath" --tag "$tag" \
    --kappa_b "$kb" --kappa_d "$kd" \
    --subgoal_value_alpha "$alpha" --subgoal_value_gap_scale "$gap"
  (
    cd "$REPO_ROOT"
    "$PYTHON" main.py --run_config="$ypath"
  ) 2>&1 | tee -a "$rlog" | tee -a "$MASTER_LOG"
  echo "=== $(date -Is) END $family $tag ===" | tee -a "$MASTER_LOG"
}

# --- Puzzle (6): 로그 기준 0.7/0.7 안정 + gap 1 vs 5 경험, antmaze식 고 kappa·gap 스케일 탐색 ---
run_one puzzle p_kb07_kd07_a03_g1    0.7  0.7  0.3  1.0
run_one puzzle p_kb93_kd08_a03_g5  0.93 0.8  0.3  5.0
run_one puzzle p_kb93_kd08_a03_g20 0.93 0.8  0.3  20.0
run_one puzzle p_kb09_kd05_a03_g5  0.9  0.5  0.3  5.0
run_one puzzle p_kb85_kd08_a01_g5  0.85 0.8  0.1  5.0
run_one puzzle p_kb93_kd07_a04_g10 0.93 0.7  0.4  10.0

# --- Humanoid (6): 기존 yaml(0.93/0.8, gap5) 중심으로 gap·kappa·alpha 변형 ---
run_one humanoid h_kb93_kd08_a03_g5   0.93 0.8  0.3  5.0
run_one humanoid h_kb93_kd08_a03_g1  0.93 0.8  0.3  1.0
run_one humanoid h_kb93_kd08_a03_g20 0.93 0.8  0.3  20.0
run_one humanoid h_kb85_kd08_a03_g5  0.85 0.8  0.3  5.0
run_one humanoid h_kb93_kd08_a05_g5  0.93 0.8  0.5  5.0
run_one humanoid h_kb93_kd08_a015_g10 0.93 0.8  0.15 10.0

echo "All sweep jobs finished. Master log: $MASTER_LOG"
