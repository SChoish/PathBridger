#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-/home/choi/miniconda3/envs/offrl/bin/python}"

configs=(
  "config/antmaze_giant_navigate.yaml"
  "config/antmaze_teleport_navigate.yaml"
)

latest_resume_point() {
  local cfg="$1"
  "$PYTHON_BIN" - "$cfg" <<'PY'
import re
import sys
from pathlib import Path

import yaml

cfg_path = Path(sys.argv[1])
cfg = yaml.safe_load(cfg_path.read_text()) or {}
env_name = str(cfg["env_name"])
run_group = str(cfg["run_group"])
rows = []
for run_dir in Path("runs").glob(f"*_{env_name}"):
    used_path = run_dir / "config_used.yaml"
    if not used_path.is_file():
        continue
    used = yaml.safe_load(used_path.read_text()) or {}
    if str(used.get("run_group", "")) != run_group:
        continue
    epoch_sets = []
    for agent in ("dynamics", "critic", "actor"):
        ckpt_dir = run_dir / "checkpoints" / agent
        epochs = set()
        for p in ckpt_dir.glob("params_*.pkl"):
            m = re.fullmatch(r"params_(\d+)\.pkl", p.name)
            if m:
                epochs.add(int(m.group(1)))
        epoch_sets.append(epochs)
    common = set.intersection(*epoch_sets) if epoch_sets else set()
    if common:
        rows.append((max(common), run_dir.stat().st_mtime, run_dir))

if not rows:
    print("")
else:
    epoch, _mtime, run_dir = max(rows, key=lambda x: (x[0], x[1]))
    print(f"{run_dir} {epoch}")
PY
}

for cfg in "${configs[@]}"; do
  echo "=== $(date '+%F %T') :: preparing ${cfg} ==="
  resume_info="$(latest_resume_point "$cfg")"
  if [[ -n "$resume_info" ]]; then
    run_dir="${resume_info% *}"
    epoch="${resume_info##* }"
    echo "=== resume ${cfg} from ${run_dir} epoch ${epoch} ==="
    "$PYTHON_BIN" main.py \
      --run_config="$cfg" \
      --resume_run_dir="$run_dir" \
      --resume_epoch="$epoch" \
      --resume_use_run_snapshot_config=false
  else
    echo "=== start new ${cfg} ==="
    "$PYTHON_BIN" main.py --run_config="$cfg"
  fi
done

echo "=== $(date '+%F %T') :: all queued runs complete ==="
