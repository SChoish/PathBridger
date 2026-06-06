#!/usr/bin/env bash
# Sequential plain Flow-BC + eval BoN4 runs across env × target-mode grid.
#
# Grid: residual_target_mode × subgoal_target_mode ∈ {displacement, absolute}^2
# Tags: rd_sd, rd_sa, ra_sd, ra_sa
#
# Usage:
#   bash scripts/run_flow_plain_bon4_by_env.sh
#   GPU_ID=1 bash scripts/run_flow_plain_bon4_by_env.sh
#
# Regenerate YAMLs first:
#   python scripts/generate_flow_plain_bon4_configs.py

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH=".:${PYTHONPATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

GPU_ID="${GPU_ID:-0}"
CONFIG_DIR="${ROOT}/config/flow_plain_bon4_by_env"
GEN_PY="${ROOT}/scripts/generate_flow_plain_bon4_configs.py"
PYTHON_BIN="${PYTHON_BIN:-python}"
WITH_CUDA="${ROOT}/scripts/with_jax_cuda.sh"

"${PYTHON_BIN}" "${GEN_PY}"

run_one() {
  local cfg="$1"
  local env_name="$2"
  local epochs="$3"
  local run_group="$4"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" bash "${WITH_CUDA}" "${PYTHON_BIN}" main.py \
    --run_config "${cfg}" \
    --env_name "${env_name}" \
    --run_group "${run_group}" \
    --use_wandb True \
    --train_epochs "${epochs}" \
    --eval_freq 100 \
    --eval_episodes_per_task 10 \
    --final_eval_episodes_per_task 50 \
    --save_every_n_epochs 100
}

while IFS=$'\t' read -r cfg env_name epochs run_group; do
  run_one "${cfg}" "${env_name}" "${epochs}" "${run_group}"
done < <("${PYTHON_BIN}" - <<'PY'
import yaml
from pathlib import Path

ENV_STEMS = [
    "puzzle_3x3",
    "antmaze_medium",
    "antmaze_large",
    "antmaze_giant",
    "cube_single",
    "cube_double",
    "cube_triple",
]
MODE_TAGS = ["rd_sd", "rd_sa", "ra_sd", "ra_sa"]
root = Path("config/flow_plain_bon4_by_env")
for stem in ENV_STEMS:
    for tag in MODE_TAGS:
        path = root / f"flow_plain_bon4_{stem}_{tag}_from_diag_best.yaml"
        if not path.is_file():
            raise SystemExit(f"missing config: {path}")
        cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        print(
            "\t".join(
                [
                    str(path),
                    str(cfg.get("env_name", "")),
                    str(cfg.get("train_epochs", "")),
                    str(cfg.get("run_group", "FlowPlainBoN4")),
                ]
            )
        )
PY
)
