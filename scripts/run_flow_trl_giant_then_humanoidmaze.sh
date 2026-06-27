#!/usr/bin/env bash
# Chain: giant cap4000 re-eval → humanoidmaze dw sweep (train + final eval).
#
# Usage (nohup):
#   nohup bash scripts/run_flow_trl_giant_then_humanoidmaze.sh \
#     > nohup_logs/flow_trl_giant_then_humanoidmaze_nohup.out 2>&1 &
#
# If giant eval is already running, set WAIT_FOR_GIANT=1 to skip re-launch and
# wait for the in-flight job instead:
#   WAIT_FOR_GIANT=1 nohup bash scripts/run_flow_trl_giant_then_humanoidmaze.sh ...
#
# Env forwarded to both stages: GPU_ID, SEED, PYTHON_BIN, MUJOCO_GL, EPOCH, GIANT_ENV_STEPS
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-0}"
WAIT_FOR_GIANT="${WAIT_FOR_GIANT:-0}"
LOG_DIR="${ROOT}/nohup_logs"
LOG_TAG="flow_trl_giant_then_humanoidmaze"
MASTER_LOG="${LOG_DIR}/${LOG_TAG}_master.log"

mkdir -p "${LOG_DIR}"

log() {
  echo "[$(date -Is)] $*" | tee -a "${MASTER_LOG}"
}

wait_for_giant_eval() {
  log "waiting for in-flight giant cap4000 eval to finish..."
  while true; do
    if pgrep -f 'run_flow_trl_eval_giant_cap4000\.sh' >/dev/null 2>&1; then
      sleep 30
      continue
    fi
    if pgrep -f 'eval_checkpoint\.py.*cap4000' >/dev/null 2>&1; then
      sleep 30
      continue
    fi
    break
  done
  log "in-flight giant cap4000 eval finished"
}

log "chain start GPU=${GPU_ID} seed=${SEED} WAIT_FOR_GIANT=${WAIT_FOR_GIANT}"

if [[ "${WAIT_FOR_GIANT}" == "1" ]]; then
  wait_for_giant_eval
else
  log "STAGE 1/2: giant cap4000 re-eval"
  GPU_ID="${GPU_ID}" SEED="${SEED}" bash "${ROOT}/scripts/run_flow_trl_eval_giant_cap4000.sh"
  log "STAGE 1/2 complete"
fi

log "STAGE 2/2: humanoidmaze dw sweep (train + final eval N=2 8 16 32)"
GPU_ID="${GPU_ID}" SEED="${SEED}" bash "${ROOT}/scripts/run_flow_trl_humanoidmaze_dw_sweep.sh"
log "chain complete"
