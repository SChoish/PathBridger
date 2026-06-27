#!/usr/bin/env bash
# Eval K=25 and K=40 best checkpoints with subgoal_temperature=0.5 at N=2,4,8,16,32.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}:${ROOT}/scripts:${PYTHONPATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-0}"
FINAL_EPOCH="${FINAL_EPOCH:-600}"
TEMP="${SUBGOAL_TEMPERATURE:-0.5}"
PYTHON_BIN="${PYTHON_BIN:-/home/svcho/anaconda3/bin/python}"
WITH_CUDA="${ROOT}/scripts/with_jax_cuda.sh"
LOG_DIR="${ROOT}/nohup_logs"
LOG_TAG="temp05_k25_k40_eval"
EVAL_NS=(2 4 8 16 32)

mkdir -p "${LOG_DIR}"
MASTER_LOG="${LOG_DIR}/${LOG_TAG}_master.log"

echo "[$(date -Is)] ${LOG_TAG} start GPU=${GPU_ID} temp=${TEMP}" | tee -a "${MASTER_LOG}"

mapfile -t jobs < <(
  TEMP="${TEMP}" EVAL_NS="${EVAL_NS[*]}" "${PYTHON_BIN}" - <<'PY'
import glob
import json
import os
from pathlib import Path

import yaml

temp = float(os.environ['TEMP'])
eval_ns = [int(x) for x in os.environ['EVAL_NS'].split()]

K25_SPECS = [
    ('k25', 'puzzle-3x3-play-v0', 'flow_trl_feval_p3_g1_w5_n1'),
    ('k25', 'puzzle-4x4-play-v0', 'flow_trl_feval_p4_g5_w5_n1'),
    ('k25', 'cube-single-play-v0', 'flow_trl_feval_cs_g10_w5_n1'),
    ('k25', 'cube-double-play-v0', 'flow_trl_feval_cd_g10_w5_n1'),
    ('k25', 'cube-triple-play-v0', 'flow_trl_feval_ct_g10_w5_n1'),
    ('k25', 'antmaze-medium-navigate-v0', 'flow_trl_feval_amm_g3_w5_n1'),
    ('k25', 'antmaze-large-navigate-v0', 'flow_trl_feval_aml_g10_w5_n1'),
    ('k25', 'antmaze-giant-navigate-v0', 'flow_trl_feval_amg_g3_w5_n1'),
]

def resolve_run(env: str, run_group: str) -> Path | None:
    matches = []
    for rd in Path('runs').glob(f'*_seed0_{env}'):
        fp = rd / 'flags.json'
        if not fp.is_file():
            continue
        flags = json.load(open(fp, encoding='utf-8'))
        if flags.get('flags', {}).get('run_group') != run_group:
            continue
        ckpt = rd / 'checkpoints' / 'dynamics' / 'params_600.pkl'
        if not ckpt.is_file():
            continue
        matches.append(rd)
    if not matches:
        return None
    return max(matches, key=lambda p: p.stat().st_mtime)

def temp_tag(t: float) -> str:
    if abs(t - round(t)) < 1e-9:
        return str(int(round(t)))
    return format(t, 'g').replace('.', 'p')

def emit(horizon: str, env: str, run_dir: Path, eval_n: int) -> None:
    out = run_dir / 'eval_results' / f'epoch600_t{temp_tag(temp)}_n{eval_n}.json'
    if out.is_file():
        return
    base = f'{horizon}_{env.split("-")[0]}'
    print(f'{base}\t{run_dir}\t{eval_n}')

for horizon, env, rg in K25_SPECS:
    rd = resolve_run(env, rg)
    if rd is None:
        continue
    for n in eval_ns:
        emit(horizon, env, rd, n)

for cfg in sorted(glob.glob('config/sweep_flow_trl_k40_best/*.yaml')):
    if os.path.basename(cfg).startswith('_'):
        continue
    c = yaml.safe_load(open(cfg, encoding='utf-8'))
    env = str(c['env_name'])
    rg = str(c['run_group'])
    rd = resolve_run(env, rg)
    if rd is None:
        continue
    for n in eval_ns:
        emit('k40', env, rd, n)
PY
)

if ((${#jobs[@]} == 0)); then
  echo "[$(date -Is)] nothing to run (all temp=${TEMP} eval JSONs exist)" | tee -a "${MASTER_LOG}"
else
  for job in "${jobs[@]}"; do
    IFS=$'\t' read -r base run_dir eval_n <<< "${job}"
    eval_log="${LOG_DIR}/${LOG_TAG}_${base}.n${eval_n}.log"
    echo "[$(date -Is)] START ${base} eval_n=${eval_n} temp=${TEMP} run_dir=${run_dir}" | tee -a "${MASTER_LOG}"
    CUDA_VISIBLE_DEVICES="${GPU_ID}" bash "${WITH_CUDA}" "${PYTHON_BIN}" -u eval_checkpoint.py \
      --run_dir "${run_dir}" \
      --epoch "${FINAL_EPOCH}" \
      --seed "${SEED}" \
      --eval_episodes_per_task 25 \
      --subgoal_eval_num_samples "${eval_n}" \
      --subgoal_temperature "${TEMP}" \
      --mujoco_gl "${MUJOCO_GL}" \
      --skip_if_saved \
      2>&1 | tee "${eval_log}"
    echo "[$(date -Is)] DONE ${base} eval_n=${eval_n}" | tee -a "${MASTER_LOG}"
  done
fi

echo "[$(date -Is)] SUMMARIZE temp=${TEMP} results" | tee -a "${MASTER_LOG}"
TEMP="${TEMP}" "${PYTHON_BIN}" "${ROOT}/scripts/summarize_temp05_k25_k40_results.py" | tee -a "${MASTER_LOG}"
echo "[$(date -Is)] ${LOG_TAG} complete (${#jobs[@]} eval jobs)" | tee -a "${MASTER_LOG}"
