#!/usr/bin/env bash
# 1) antmaze-giant m800 eval (temp 1.0 / 0.5)
# 2) resume puzzle 4x5/4x6 flow_trl_p456g999 sweep

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
LOG_DIR="${ROOT}/nohup_logs"
MASTER_LOG="${LOG_DIR}/amg_m800_then_p456_master.log"

echo "[$(date -Is)] amg_m800_then_p456 start" | tee -a "${MASTER_LOG}"

bash "${ROOT}/scripts/run_amg_m800_eval.sh" 2>&1 | tee -a "${MASTER_LOG}"

echo "[$(date -Is)] RESUME puzzle 4x5/4x6 sweep" | tee -a "${MASTER_LOG}"
bash "${ROOT}/scripts/run_flow_trl_puzzle_45_46.sh" 2>&1 | tee -a "${MASTER_LOG}"

echo "[$(date -Is)] amg_m800_then_p456 complete" | tee -a "${MASTER_LOG}"
