#!/usr/bin/env bash
# Run missing K=40 evals for subgoal_eval_num_samples in {4,8,16,32}.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}:${ROOT}/scripts:${PYTHONPATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-0}"
FINAL_EPOCH="${FINAL_EPOCH:-600}"
PYTHON_BIN="${PYTHON_BIN:-/home/svcho/anaconda3/bin/python}"
WITH_CUDA="${ROOT}/scripts/with_jax_cuda.sh"
LOG_DIR="${ROOT}/nohup_logs"
LOG_TAG="k40_multi_n_eval"
EVAL_NS=(4 8 16 32)

mkdir -p "${LOG_DIR}"
MASTER_LOG="${LOG_DIR}/${LOG_TAG}_master.log"

echo "[$(date -Is)] ${LOG_TAG} start GPU=${GPU_ID}" | tee -a "${MASTER_LOG}"

mapfile -t jobs < <(
  EVAL_NS="${EVAL_NS[*]}" "${PYTHON_BIN}" - <<'PY'
import glob
import json
import os
from pathlib import Path

import yaml

eval_ns = [int(x) for x in os.environ['EVAL_NS'].split()]
for cfg in sorted(glob.glob('config/sweep_flow_trl_k40_best/*.yaml')):
    if os.path.basename(cfg).startswith('_'):
        continue
    c = yaml.safe_load(open(cfg, encoding='utf-8'))
    env = c['env_name']
    rg = c['run_group']
    matches = []
    for rd in Path('runs').glob(f'*_seed0_{env}'):
        fp = rd / 'flags.json'
        if fp.exists() and json.load(open(fp, encoding='utf-8')).get('flags', {}).get('run_group') == rg:
            matches.append(rd)
    if not matches:
        continue
    rd = max(matches, key=lambda p: p.stat().st_mtime)
    base = os.path.basename(cfg).removesuffix('.yaml')
    for n in eval_ns:
        out = rd / 'eval_results' / f'epoch600_n{n}.json'
        if out.is_file():
            continue
        print(f"{base}\t{rd}\t{n}")
PY
)

if ((${#jobs[@]} == 0)); then
  echo "[$(date -Is)] nothing to run (all eval JSONs exist)" | tee -a "${MASTER_LOG}"
else
  for job in "${jobs[@]}"; do
    IFS=$'\t' read -r base run_dir eval_n <<< "${job}"
    eval_log="${LOG_DIR}/${LOG_TAG}_${base}.n${eval_n}.log"
    echo "[$(date -Is)] START ${base} eval_n=${eval_n} run_dir=${run_dir}" | tee -a "${MASTER_LOG}"
    CUDA_VISIBLE_DEVICES="${GPU_ID}" bash "${WITH_CUDA}" "${PYTHON_BIN}" -u eval_checkpoint.py \
      --run_dir "${run_dir}" \
      --epoch "${FINAL_EPOCH}" \
      --seed "${SEED}" \
      --eval_episodes_per_task 25 \
      --subgoal_eval_num_samples "${eval_n}" \
      --mujoco_gl "${MUJOCO_GL}" \
      --skip_if_saved \
      2>&1 | tee "${eval_log}"
    echo "[$(date -Is)] DONE ${base} eval_n=${eval_n}" | tee -a "${MASTER_LOG}"
  done
fi

echo "[$(date -Is)] SUMMARIZE k40 multi-N table" | tee -a "${MASTER_LOG}"
"${PYTHON_BIN}" "${ROOT}/scripts/summarize_k40_multi_n_results.py" | tee -a "${MASTER_LOG}"
echo "[$(date -Is)] ${LOG_TAG} complete (${#jobs[@]} eval jobs)" | tee -a "${MASTER_LOG}"
