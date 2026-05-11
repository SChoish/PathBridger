#!/usr/bin/env bash
# Sequential training:
#   (1) Resume in-place: hyperparameters come from the run snapshot by default
#       (`main.py`: `flags.json` → temp YAML if present, else `config_used.yaml`)
#       unless you set PHASE1_RUN_CONFIG or pass `--run_config` on argv (see main.py).
#   (2) Fresh run: uses RUN_CONFIG (repo YAML).
#
# Environment (override as needed):
#   RESUME_RUN_DIR   — run folder with checkpoints/*/params_${RESUME_EPOCH}.pkl
#   RESUME_EPOCH     — default 200
#   PHASE1_EPOCHS    — total train_epochs for phase 1 (inclusive); default 1000
#   PHASE1_RUN_CONFIG — if set, phase 1 uses this YAML instead of the run snapshot
#   RUN_CONFIG       — phase 2 default: repo config/antmaze_large_navigate.yaml
#   PHASE2_TRAIN_EPOCHS — if set, passed as --train_epochs=... for phase 2 only
#
# Conda: `conda activate offrl` after `conda shell.bash hook`.
#
# Example (nohup, snapshot hparams for resume; pick any existing dir for the log):
#   nohup env RESUME_RUN_DIR="$PWD/runs/20260424_005514_joint_dqc_seed0_antmaze-large-navigate-v0" \
#     bash scripts/antmaze_resume_then_fresh.sh \
#     > "$RESUME_RUN_DIR/chain_$(date +%Y%m%d_%H%M%S).out" 2>&1 &
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${PYTHONPATH:-.}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

RUN_CONFIG="${RUN_CONFIG:-$ROOT/config/antmaze_large_navigate.yaml}"
RESUME_RUN_DIR="${RESUME_RUN_DIR:-$ROOT/runs/20260424_005514_joint_dqc_seed0_antmaze-large-navigate-v0}"
RESUME_EPOCH="${RESUME_EPOCH:-200}"
PHASE1_EPOCHS="${PHASE1_EPOCHS:-1000}"

if [[ ! -d "$RESUME_RUN_DIR" ]]; then
  echo "ERROR: RESUME_RUN_DIR is not a directory: $RESUME_RUN_DIR" >&2
  exit 1
fi
for agent in goub critic actor; do
  ckpt="$RESUME_RUN_DIR/checkpoints/$agent/params_${RESUME_EPOCH}.pkl"
  if [[ ! -f "$ckpt" ]]; then
    echo "ERROR: missing checkpoint: $ckpt" >&2
    exit 1
  fi
done

# shellcheck disable=SC1091
eval "$(conda shell.bash hook)"
conda activate offrl

echo "=========================================="
echo "Phase 1: resume in-place (weights + hparams from run snapshot unless PHASE1_RUN_CONFIG set)"
echo "  run_dir=$RESUME_RUN_DIR"
echo "  resume_epoch=$RESUME_EPOCH  train_epochs=$PHASE1_EPOCHS"
if [[ -n "${PHASE1_RUN_CONFIG:-}" ]]; then
  echo "  run_config=$PHASE1_RUN_CONFIG (overrides snapshot)"
else
  echo "  run_config: (omit) → main.py uses run_dir/flags.json (else config_used.yaml)"
fi
echo "=========================================="
phase1=(python main.py --resume_run_dir="$RESUME_RUN_DIR" --resume_epoch="$RESUME_EPOCH" --train_epochs="$PHASE1_EPOCHS")
if [[ -n "${PHASE1_RUN_CONFIG:-}" ]]; then
  phase1+=(--run_config="$PHASE1_RUN_CONFIG")
fi
"${phase1[@]}"

echo "=========================================="
echo "Phase 2: fresh run (new run_dir under runs/)"
echo "  run_config=$RUN_CONFIG"
if [[ -n "${PHASE2_TRAIN_EPOCHS:-}" ]]; then
  echo "  --train_epochs=$PHASE2_TRAIN_EPOCHS (overrides YAML)"
else
  echo "  train_epochs from YAML only"
fi
echo "=========================================="
phase2_args=(--run_config="$RUN_CONFIG")
if [[ -n "${PHASE2_TRAIN_EPOCHS:-}" ]]; then
  phase2_args+=(--train_epochs="$PHASE2_TRAIN_EPOCHS")
fi
python main.py "${phase2_args[@]}"

echo "=========================================="
echo "All phases finished OK."
echo "=========================================="
