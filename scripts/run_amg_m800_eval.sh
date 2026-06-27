#!/usr/bin/env bash
# Re-eval antmaze-giant checkpoints to the env max episode length, temp=1.0 and 0.5.
# JSON: eval_results/epoch600[_t0p5]_n<N>.json

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
LOG_TAG="amg_envmax_eval"
TEMPS=(1.0 0.5)
EVAL_NS=(2 8 16 32)
FORCE_RERUN="${FORCE_RERUN:-0}"

mkdir -p "${LOG_DIR}"
MASTER_LOG="${LOG_DIR}/${LOG_TAG}_master.log"

echo "[$(date -Is)] ${LOG_TAG} start GPU=${GPU_ID} budget=env_max_episode_steps temps=${TEMPS[*]} N=${EVAL_NS[*]} force=${FORCE_RERUN}" | tee -a "${MASTER_LOG}"

if [[ "${FORCE_RERUN}" == "1" ]]; then
  echo "[$(date -Is)] remove prior env-max giant eval JSONs" | tee -a "${MASTER_LOG}"
  rm -f runs/*antmaze-giant*/eval_results/epoch*_n*.json
fi

mapfile -t jobs < <(
  TEMPS="${TEMPS[*]}" EVAL_NS="${EVAL_NS[*]}" FINAL_EPOCH="${FINAL_EPOCH}" FORCE_RERUN="${FORCE_RERUN}" \
    "${PYTHON_BIN}" - <<'PY'
import json
import os
from pathlib import Path

temps = [float(x) for x in os.environ['TEMPS'].split()]
eval_ns = [int(x) for x in os.environ['EVAL_NS'].split()]
final_epoch = int(os.environ['FINAL_EPOCH'])
force = os.environ.get('FORCE_RERUN', '0') == '1'

SPECS = [
    ('amg_g1', 'flow_trl_feval_amg_g1_w5_n1'),
    ('amg_g3', 'flow_trl_feval_amg_g3_w5_n1'),
    ('amg_g5', 'flow_trl_feval_amg_g5_w5_n1'),
    ('amg_g10', 'flow_trl_feval_amg_g10_w5_n1'),
    ('amg_k40', 'flow_trl_k40_best_amg_g3_en8'),
]


def resolve_run(run_group: str) -> Path | None:
    matches = []
    for rd in Path('runs').glob('*_seed0_antmaze-giant-navigate-v0'):
        fp = rd / 'flags.json'
        if not fp.is_file():
            continue
        rg = json.load(open(fp, encoding='utf-8')).get('flags', {}).get('run_group', '')
        if rg != run_group:
            continue
        ckpt = rd / 'checkpoints' / 'dynamics' / f'params_{final_epoch}.pkl'
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


def out_path(run_dir: Path, *, temp: float, n: int) -> Path:
    base = f'epoch{final_epoch}'
    if temp != 1.0:
        base += f'_t{temp_tag(temp)}'
    base += f'_n{n}.json'
    return run_dir / 'eval_results' / base


for label, run_group in SPECS:
    rd = resolve_run(run_group)
    if rd is None:
        print(f'# missing run: {label} {run_group}', flush=True)
        continue
    for temp in temps:
        for n in eval_ns:
            if not force and out_path(rd, temp=temp, n=n).is_file():
                continue
            print(f'{label}\t{rd}\t{n}\t{temp}')
PY
)

if ((${#jobs[@]} == 0)); then
  echo "[$(date -Is)] nothing to run (all env-max eval JSONs exist)" | tee -a "${MASTER_LOG}"
else
  for job in "${jobs[@]}"; do
    [[ "${job}" == \#* ]] && continue
    IFS=$'\t' read -r label run_dir eval_n temp <<< "${job}"
    temp_tag="${temp/./p}"
    eval_log="${LOG_DIR}/${LOG_TAG}_${label}.t${temp_tag}.n${eval_n}.log"
    echo "[$(date -Is)] START ${label} eval_n=${eval_n} temp=${temp} budget=env_max_episode_steps run_dir=${run_dir}" | tee -a "${MASTER_LOG}"
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
    echo "[$(date -Is)] DONE ${label} eval_n=${eval_n} temp=${temp}" | tee -a "${MASTER_LOG}"
  done
fi

echo "[$(date -Is)] SUMMARIZE feval results" | tee -a "${MASTER_LOG}"
"${PYTHON_BIN}" "${ROOT}/scripts/summarize_feval_results.py" | tee -a "${MASTER_LOG}"
echo "[$(date -Is)] ${LOG_TAG} complete (${#jobs[@]} eval jobs)" | tee -a "${MASTER_LOG}"
