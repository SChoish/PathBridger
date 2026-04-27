#!/usr/bin/env bash
# Wait until run_dir has epoch-300 checkpoints (goub/critic/actor), then run
# `run_remaining_dynamics_tau_alpha_sweep.sh`.
#
# Example (nohup):
#   nohup ./scripts/queue_remaining_after_ckpt300.sh \
#     > nohup_logs/queue_remaining_$(date +%Y%m%d_%H%M%S).log 2>&1 &

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="${RUN_DIR:-${ROOT_DIR}/runs/20260426_004705_joint_dqc_seed0_antmaze-large-navigate-v0}"
POLL_SEC="${POLL_SEC:-60}"
MAX_WAIT_SEC="${MAX_WAIT_SEC:-864000}"

_ckpt300() {
  [[ -f "${RUN_DIR}/checkpoints/goub/params_300.pkl" ]] \
    && [[ -f "${RUN_DIR}/checkpoints/critic/params_300.pkl" ]] \
    && [[ -f "${RUN_DIR}/checkpoints/actor/params_300.pkl" ]]
}

elapsed=0
echo "[queue] waiting for params_300 in ${RUN_DIR} (poll=${POLL_SEC}s, max=${MAX_WAIT_SEC}s)"
while ! _ckpt300; do
  sleep "${POLL_SEC}"
  elapsed=$((elapsed + POLL_SEC))
  if ((elapsed > MAX_WAIT_SEC)); then
    echo "[queue] TIMEOUT after ${elapsed}s" >&2
    exit 1
  fi
done

echo "[queue] checkpoints ready; starting remaining sweep $(date -Iseconds)"
exec "${ROOT_DIR}/scripts/run_remaining_dynamics_tau_alpha_sweep.sh"
