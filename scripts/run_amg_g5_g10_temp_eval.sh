#!/usr/bin/env bash
# Eval amg_g5_w5_n1 and amg_g10_w5_n1 at subgoal_temperature=0.5 and 0.25, N=2,4,8,16,32.

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
LOG_TAG="amg_g5_g10_temp_eval"
TEMPS=(0.5 0.25)
EVAL_NS=(2 4 8 16 32)

mkdir -p "${LOG_DIR}"
MASTER_LOG="${LOG_DIR}/${LOG_TAG}_master.log"

echo "[$(date -Is)] ${LOG_TAG} start GPU=${GPU_ID} temps=${TEMPS[*]}" | tee -a "${MASTER_LOG}"

mapfile -t jobs < <(
  TEMPS="${TEMPS[*]}" EVAL_NS="${EVAL_NS[*]}" "${PYTHON_BIN}" - <<'PY'
import json
import os
from pathlib import Path

temps = [float(x) for x in os.environ['TEMPS'].split()]
eval_ns = [int(x) for x in os.environ['EVAL_NS'].split()]

SPECS = [
    ('amg_g5', 'antmaze-giant-navigate-v0', 'flow_trl_feval_amg_g5_w5_n1'),
    ('amg_g10', 'antmaze-giant-navigate-v0', 'flow_trl_feval_amg_g10_w5_n1'),
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


for base, env, rg in SPECS:
    rd = resolve_run(env, rg)
    if rd is None:
        print(f'# missing run: {base} {rg}', file=__import__('sys').stderr)
        continue
    for temp in temps:
        for n in eval_ns:
            out = rd / 'eval_results' / f'epoch600_t{temp_tag(temp)}_n{n}.json'
            if out.is_file():
                continue
            print(f'{base}\t{rd}\t{n}\t{temp}')
PY
)

if ((${#jobs[@]} == 0)); then
  echo "[$(date -Is)] nothing to run (all eval JSONs exist)" | tee -a "${MASTER_LOG}"
else
  for job in "${jobs[@]}"; do
    IFS=$'\t' read -r base run_dir eval_n temp <<< "${job}"
    temp_tag="${temp/./p}"
    eval_log="${LOG_DIR}/${LOG_TAG}_${base}.t${temp_tag}.n${eval_n}.log"
    echo "[$(date -Is)] START ${base} eval_n=${eval_n} temp=${temp} run_dir=${run_dir}" | tee -a "${MASTER_LOG}"
    CUDA_VISIBLE_DEVICES="${GPU_ID}" bash "${WITH_CUDA}" "${PYTHON_BIN}" -u eval_checkpoint.py \
      --run_dir "${run_dir}" \
      --epoch "${FINAL_EPOCH}" \
      --seed "${SEED}" \
      --eval_episodes_per_task 25 \
      --subgoal_eval_num_samples "${eval_n}" \
      --subgoal_temperature "${temp}" \
      --mujoco_gl "${MUJOCO_GL}" \
      --skip_if_saved \
      2>&1 | tee "${eval_log}"
    echo "[$(date -Is)] DONE ${base} eval_n=${eval_n} temp=${temp}" | tee -a "${MASTER_LOG}"
  done
fi

echo "[$(date -Is)] SUMMARIZE feval results" | tee -a "${MASTER_LOG}"
"${PYTHON_BIN}" "${ROOT}/scripts/summarize_feval_results.py" | tee -a "${MASTER_LOG}"
echo "[$(date -Is)] ${LOG_TAG} complete (${#jobs[@]} eval jobs)" | tee -a "${MASTER_LOG}"
