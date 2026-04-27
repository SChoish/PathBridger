#!/usr/bin/env python3
"""Find a training run matching hyperparameters for resume.

Prints shell-friendly lines: STATUS=... RUN_DIR=... RESUME_EPOCH=...
  STATUS=new       — no usable checkpoints
  STATUS=resume    — RESUME_EPOCH>0, RUN_DIR set; caller should pass --train_epochs etc.
  STATUS=complete  — already at or past --target-epochs (same RUN_DIR / max ckpt)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


def _suffixes(ckpt_dir: Path) -> set[int]:
    out: set[int] = set()
    if not ckpt_dir.is_dir():
        return out
    for p in ckpt_dir.glob("params_*.pkl"):
        m = re.search(r"params_(\d+)\.pkl$", p.name)
        if m:
            out.add(int(m.group(1)))
    return out


def max_common_ckpt(run_dir: Path) -> int:
    root = run_dir / "checkpoints"
    g = _suffixes(root / "dynamics")
    c = _suffixes(root / "critic")
    a = _suffixes(root / "actor")
    common = g & c & a
    return max(common) if common else 0


def flags_match(data: dict, *, env_name: str, spi_tau: float, subgoal_value_alpha: float, discount: float) -> bool:
    fg = data.get("flags") or {}
    if str(fg.get("env_name")) != env_name:
        return False
    g = data.get("dynamics") or {}
    cr = data.get("critic_agent") or {}
    act = data.get("actor") or {}
    if abs(float(act.get("spi_tau", -1.0)) - spi_tau) > 1e-5:
        return False
    if abs(float(g.get("subgoal_value_alpha", -1.0)) - subgoal_value_alpha) > 1e-5:
        return False
    if abs(float(cr.get("discount", -1.0)) - discount) > 1e-5:
        return False
    return True


def find_resume(
    runs_root: Path,
    *,
    env_name: str,
    spi_tau: float,
    alpha: float,
    discount: float,
    target_epochs: int,
) -> tuple[str, str, int]:
    """Returns (status, run_dir_str, resume_epoch)."""
    pattern = f"*_{env_name.replace('/', '-')}"  # env token in folder name
    # Folder names use underscores; env_name antmaze-giant-navigate-v0 -> antmaze-giant-navigate-v0
    env_tok = env_name.replace("/", "_")
    candidates: list[tuple[int, float, Path]] = []
    for run_dir in sorted(runs_root.glob(f"*_seed*_{env_tok}")):
        if not run_dir.is_dir():
            continue
        fj = run_dir / "flags.json"
        if not fj.is_file():
            continue
        try:
            data = json.loads(fj.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if not flags_match(data, env_name=env_name, spi_tau=spi_tau, subgoal_value_alpha=alpha, discount=discount):
            continue
        m = max_common_ckpt(run_dir)
        if m <= 0:
            continue
        try:
            mt = fj.stat().st_mtime
        except OSError:
            mt = 0.0
        candidates.append((m, mt, run_dir))

    if not candidates:
        return "new", "", 0

    # Prefer highest checkpoint; tie-break by newer flags.json mtime.
    candidates.sort(key=lambda x: (x[0], x[1]))
    best_m, _mt, best_dir = candidates[-1]

    if best_m >= target_epochs:
        return "complete", str(best_dir), best_m

    return "resume", str(best_dir), best_m


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs-root", type=Path, default=Path("runs"))
    p.add_argument("--env-name", default="antmaze-giant-navigate-v0")
    p.add_argument("--spi-tau", type=float, default=10.0)
    p.add_argument("--alpha", type=float, required=True)
    p.add_argument("--discount", type=float, required=True)
    p.add_argument("--target-epochs", type=int, default=1000)
    args = p.parse_args()

    status, run_dir, ep = find_resume(
        args.runs_root.resolve(),
        env_name=args.env_name,
        spi_tau=args.spi_tau,
        alpha=args.alpha,
        discount=args.discount,
        target_epochs=args.target_epochs,
    )
    print(f"STATUS={status}")
    print(f"RUN_DIR={run_dir}")
    print(f"RESUME_EPOCH={ep}")


if __name__ == "__main__":
    main()
