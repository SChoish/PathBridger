#!/usr/bin/env bash
# One shared short executor, then paired reader/recursive representation seeds.
# No PB checkpoint, proposal/Q updates, environment rollouts, or G1 bypass.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
PYTHON="${PYTHON:-python}"
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
OUT_ROOT="${OUT_ROOT:-$PWD/runs/action_scratch_$(date -u +%Y%m%dT%H%M%SZ)}"
DECODER_UPDATES="${DECODER_UPDATES:-100000}"
REPRESENTATION_UPDATES="${REPRESENTATION_UPDATES:-20000}"
BATCH_SIZE="${BATCH_SIZE:-256}"
SMOKE="${SMOKE:-0}"
export OUT_ROOT

if [[ -e "$OUT_ROOT" ]]; then
  echo "Use a new OUT_ROOT; existing run will not be overwritten: $OUT_ROOT" >&2
  exit 1
fi
if [[ -n "${TRAIN:-}" || -n "${VALIDATION:-}" ]]; then
  : "${TRAIN:?Both TRAIN and VALIDATION are required}"
  : "${VALIDATION:?Both TRAIN and VALIDATION are required}"
else
  : "${ENV_NAME:?Set ENV_NAME for OGBench data preparation, or supply TRAIN and VALIDATION}"
fi

if [[ "$SMOKE" == 1 ]]; then
  if (( DECODER_UPDATES > 3 || REPRESENTATION_UPDATES > 3 )); then
    echo "SMOKE=1 requires DECODER_UPDATES and REPRESENTATION_UPDATES <= 3" >&2
    exit 1
  fi
  stage_flags=(--smoke)
else
  "$PYTHON" -c 'import jax; print(jax.devices()); assert jax.default_backend() == "gpu", "Formal training requires a working JAX GPU environment"'
  stage_flags=()
fi
mkdir -p "$OUT_ROOT/logs"
"$PYTHON" -m pip freeze > "$OUT_ROOT/python_packages.txt"
git rev-parse HEAD > "$OUT_ROOT/code_commit.txt"

if [[ -z "${TRAIN:-}" ]]; then
  export ENV_NAME
  export DATA_DIR="${DATA_DIR:-$OUT_ROOT/ogbench_cache}"
  "$PYTHON" - <<'PY'
import json
import os
from pathlib import Path

import numpy as np

from envs.env_utils import make_env_and_datasets
from skill_extensions.action_data import dataset_contract

root = Path(os.environ['OUT_ROOT']) / 'data'
root.mkdir()
env, train, validation = make_env_and_datasets(os.environ['ENV_NAME'], os.environ['DATA_DIR'])
env.close()
identity = dataset_contract(train, validation)
for name, dataset in (('train', train), ('validation', validation)):
    np.savez(root / f'{name}.npz', **dict(dataset))
(root / 'provenance.json').write_text(json.dumps({
    'environment': os.environ['ENV_NAME'], 'dataset_dir': os.environ['DATA_DIR'],
    'datasets': identity, 'environment_steps': 0,
}, indent=2) + '\n')
PY
  TRAIN="$OUT_ROOT/data/train.npz"
  VALIDATION="$OUT_ROOT/data/validation.npz"
fi

"$PYTHON" -u action_skill_main.py \
  --stage decoder \
  --train-npz "$TRAIN" --validation-npz "$VALIDATION" \
  --seed 0 --updates "$DECODER_UPDATES" --batch-size "$BATCH_SIZE" \
  --checkpoint-every 10000 --output "$OUT_ROOT/decoder_seed0" \
  "${stage_flags[@]}" 2>&1 | tee "$OUT_ROOT/logs/decoder_seed0.log"

DONOR="$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["last_checkpoint"]["path"])' "$OUT_ROOT/decoder_seed0/manifest.json")"
DONOR_SHA256="$(sha256sum "$DONOR" | awk '{print $1}')"

for seed in 0 1 2; do
  for arm in reader recursive; do
    "$PYTHON" -u action_skill_main.py \
      --stage representation \
      --train-npz "$TRAIN" --validation-npz "$VALIDATION" \
      --donor "$DONOR" --donor-sha256 "$DONOR_SHA256" \
      --config "skill_configs/action_${arm}.json" \
      --seed "$seed" --updates "$REPRESENTATION_UPDATES" --batch-size "$BATCH_SIZE" \
      --checkpoint-every 1000 --output "$OUT_ROOT/$arm/seed${seed}/representation" \
      "${stage_flags[@]}" 2>&1 | tee "$OUT_ROOT/logs/${arm}_seed${seed}.log"
  done
done

echo "Completed shared decoder + six representation runs: $OUT_ROOT"
echo "Next: paired first-program G1 environment evaluation. Proposal/Q have not been trained."
