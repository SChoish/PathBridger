#!/usr/bin/env bash
# Sequential actor-only SPI tau sweeps for every env with a best-IDM eval checkpoint in runs/.
#
# Usage:
#   bash scripts/run_actor_spi_best_idm_sweep.sh [seed] [tau1 tau2 ...]

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SEED="${1:-0}"
if [[ "$#" -gt 0 ]]; then
  shift || true
fi

if [[ "$#" -gt 0 ]]; then
  TAUS=("$@")
else
  TAUS=(0.5 1 3 10)
fi

PYTHON_BIN="${PYTHON_BIN:-/home/choi/miniconda3/envs/offrl/bin/python}"
LOG_DIR="${ROOT}/nohup_logs/actor_spi/all_best"
MASTER="${LOG_DIR}/master_seed${SEED}.log"
mkdir -p "${LOG_DIR}"

mapfile -t ENVS < <(
  "${PYTHON_BIN}" - <<'PY'
import glob
import json
import os

best = {}
for path in glob.glob('runs/*/eval_results/*.json'):
    try:
        with open(path, encoding='utf-8') as f:
            rec = json.load(f)
    except Exception:
        continue
    env = rec.get('env_name')
    if not env:
        continue
    try:
        idm = float(rec['idm_success_rate_mean'])
    except Exception:
        continue
    actor = float(rec.get('actor_success_rate_mean', -1.0))
    epoch = int(rec.get('epoch', 0) or 0)
    eval_n = int(rec.get('subgoal_eval_num_samples', 0) or 0)
    run_dir = rec.get('run_dir') or os.path.dirname(os.path.dirname(path))
    if not os.path.exists(os.path.join(run_dir, 'checkpoints', 'dynamics', f'params_{epoch}.pkl')):
        continue
    if not os.path.exists(os.path.join(run_dir, 'checkpoints', 'critic', f'params_{epoch}.pkl')):
        continue
    if not os.path.exists(os.path.join(run_dir, 'checkpoints', 'actor', f'params_{epoch}.pkl')):
        continue
    key = (idm, actor, epoch, eval_n)
    if env not in best or key > best[env]:
        best[env] = key

for env in sorted(best):
    print(env)
PY
)

echo "[$(date -Is)] actor_spi all-best start seed=${SEED} envs=${ENVS[*]} taus=${TAUS[*]}" | tee -a "${MASTER}"

for env in "${ENVS[@]}"; do
  echo "[$(date -Is)] START_ENV ${env}" | tee -a "${MASTER}"
  bash scripts/sweep_actor_spi_tau.sh "${env}" "${SEED}" "${TAUS[@]}"
  echo "[$(date -Is)] DONE_ENV ${env}" | tee -a "${MASTER}"
done

echo "[$(date -Is)] actor_spi all-best complete seed=${SEED}" | tee -a "${MASTER}"
