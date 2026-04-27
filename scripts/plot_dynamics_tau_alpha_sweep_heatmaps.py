#!/usr/bin/env python3
"""Linear dynamics τ × α 스윕 (`antmaze_navigate_dynamics_tau_alpha_sweep`) 히트맵.

- **large** `antmaze-large-navigate-v0`: eval epoch 100 / 200 / 300 (300-step 학습)
- **giant** `antmaze-giant-navigate-v0`: eval epoch 100 … 500 (500-step 학습)

각 run의 ``run*.log``에서 ``idm/actor success_rate_mean``을 읽어 (α × τ) 그리드에 채웁니다.
기본은 ``run_group: antmaze_navigate_dynamics_tau_alpha_sweep`` 만 사용합니다.
large에서 **τ=10, α=0.3** 만 셀이 비어 있으면, 동일 환경·하이퍼를 쓴
``runs/20260425_224314_joint_dqc_seed0_antmaze-large-navigate-v0`` (linear-SDE dynamics) eval 로그로만 채웁니다.

Usage:
  cd /path/to/douri && PYTHONPATH=. python scripts/plot_dynamics_tau_alpha_sweep_heatmaps.py

  Giant만 ``critic_agent.discount == 0.99`` 런으로 고정 PNG:
  ``python scripts/plot_dynamics_tau_alpha_sweep_heatmaps.py --giant-discount-snapshot 0.99``
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib import patheffects as pe

RUN_GROUP = "antmaze_navigate_dynamics_tau_alpha_sweep"
TAUS = (5.0, 10.0)
ALPHAS = (0.1, 0.3, 0.5)

# Same env / spi_tau / subgoal_value_alpha as the missing large dynamics cell; not dynamics sweep.
LARGE_TAU10_ALPHA03_ALIAS_RUN = (
    "20260425_224314_joint_dqc_seed0_antmaze-large-navigate-v0"
)


def parse_eval_means(log_text: str) -> dict[int, tuple[float, float]]:
    out: dict[int, tuple[float, float]] = {}
    lines = log_text.splitlines()
    i = 0
    while i < len(lines):
        m = re.search(r"=== EVAL START epoch=(\d+)", lines[i])
        if not m:
            i += 1
            continue
        ep = int(m.group(1))
        idm_mean = actor_mean = None
        i += 1
        while i < len(lines):
            if "=== EVAL END" in lines[i]:
                break
            lm = re.search(r"idm success_rate_mean=([\d.]+)", lines[i])
            if lm:
                idm_mean = float(lm.group(1))
            lm = re.search(r"actor success_rate_mean=([\d.]+)", lines[i])
            if lm:
                actor_mean = float(lm.group(1))
            i += 1
        if idm_mean is not None and actor_mean is not None:
            out[ep] = (idm_mean, actor_mean)
        i += 1
    return out


def read_run_logs(run_dir: Path) -> str:
    parts: list[str] = []
    for logf in sorted(run_dir.glob("run*.log")):
        try:
            parts.append(logf.read_text(encoding="utf-8", errors="ignore"))
        except OSError:
            continue
    return "\n".join(parts)


def collect(
    runs_root: Path,
    env_substr: str,
    epochs: tuple[int, ...],
    critic_discount: float | None = None,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], list[str]]:
    idm_mats = {ep: np.full((len(ALPHAS), len(TAUS)), np.nan) for ep in epochs}
    actor_mats = {ep: np.full((len(ALPHAS), len(TAUS)), np.nan) for ep in epochs}
    notes: list[str] = []
    tau_set = set(TAUS)
    alpha_set = set(ALPHAS)

    for d in sorted(runs_root.iterdir()):
        if not d.is_dir():
            continue
        cfg_path = d / "config_used.yaml"
        if not cfg_path.is_file():
            continue
        with open(cfg_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        if cfg.get("run_group") != RUN_GROUP:
            continue
        env = str(cfg.get("env_name") or "")
        if env_substr not in env:
            continue
        if critic_discount is not None:
            raw_disc = (cfg.get("critic_agent") or {}).get("discount")
            try:
                disc = float(raw_disc)
            except (TypeError, ValueError):
                notes.append(f"skip {d.name}: critic discount missing")
                continue
            if abs(disc - float(critic_discount)) > 1e-5:
                notes.append(f"skip {d.name}: critic discount={disc} (filter {critic_discount})")
                continue
        g = cfg.get("dynamics") or {}
        a = cfg.get("actor") or {}
        try:
            tau = float(a.get("spi_tau"))
            alpha = float(g.get("subgoal_value_alpha"))
        except (TypeError, ValueError):
            notes.append(f"skip {d.name}: missing tau/alpha")
            continue
        if tau not in tau_set or alpha not in alpha_set:
            notes.append(f"skip {d.name}: tau={tau} alpha={alpha}")
            continue
        ti = TAUS.index(tau)
        ai = ALPHAS.index(alpha)
        means = parse_eval_means(read_run_logs(d))
        for ep in epochs:
            pair = means.get(ep)
            if pair is None:
                continue
            idm_mats[ep][ai, ti] = pair[0]
            actor_mats[ep][ai, ti] = pair[1]

    # Large only: fill (τ=10, α=0.3) from theta-linear run if dynamics sweep left it empty.
    if env_substr == "large-navigate":
        alias = runs_root / LARGE_TAU10_ALPHA03_ALIAS_RUN
        cfg_path = alias / "config_used.yaml"
        if cfg_path.is_file():
            with open(cfg_path, encoding="utf-8") as f:
                acfg = yaml.safe_load(f) or {}
            env = str(acfg.get("env_name") or "")
            g = acfg.get("dynamics") or {}
            a = acfg.get("actor") or {}
            try:
                tau = float(a.get("spi_tau"))
                alpha = float(g.get("subgoal_value_alpha"))
            except (TypeError, ValueError):
                tau = alpha = None
            if "large-navigate" in env and tau == 10.0 and alpha == 0.3:
                ti = TAUS.index(10.0)
                ai = ALPHAS.index(0.3)
                means = parse_eval_means(read_run_logs(alias))
                filled = False
                for ep in epochs:
                    pair = means.get(ep)
                    if pair is None:
                        continue
                    if np.isnan(idm_mats[ep][ai, ti]):
                        idm_mats[ep][ai, ti] = pair[0]
                        filled = True
                    if np.isnan(actor_mats[ep][ai, ti]):
                        actor_mats[ep][ai, ti] = pair[1]
                        filled = True
                if filled:
                    notes.append(
                        f"large τ=10 α=0.3: filled from {LARGE_TAU10_ALPHA03_ALIAS_RUN} "
                        "(theta_linear sweep; not dynamics_tau_alpha_sweep)"
                    )

    return idm_mats, actor_mats, notes


def actor_uplift(idm: np.ndarray, actor: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        out = (actor - idm) / idm
    out = np.where(np.isfinite(out), out, np.nan)
    return out


def plot_composite(
    idm_mats: dict[int, np.ndarray],
    actor_mats: dict[int, np.ndarray],
    epochs: tuple[int, ...],
    out_path: Path,
    suptitle: str,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ncols = len(epochs)
    fig_w = 5.2 * ncols
    fig, axes = plt.subplots(3, ncols, figsize=(fig_w, 9.5), constrained_layout=True)
    pe_white = [pe.withStroke(linewidth=2, foreground="white")]

    abs_kwargs = {"cmap": "viridis", "vmin": 0.0, "vmax": 1.0}
    abs_fmt = lambda v: "—" if not np.isfinite(v) else f"{v:.2f}"
    pct_fmt = lambda v: "—" if not np.isfinite(v) else f"{v * 100:+.0f}%"

    for col, ep in enumerate(epochs):
        idm = idm_mats[ep]
        actor = actor_mats[ep]
        uplift = actor_uplift(idm, actor)

        rows = (
            ("IDM task-mean success", idm, abs_kwargs, abs_fmt),
            ("Actor task-mean success", actor, abs_kwargs, abs_fmt),
            (
                "Actor uplift vs IDM  (A−I)/I",
                uplift,
                {"cmap": "RdBu_r", "vmin": -1.0, "vmax": 1.0},
                pct_fmt,
            ),
        )

        last_row = len(rows) - 1
        for row, (name, mat, im_kwargs, fmt) in enumerate(rows):
            ax = axes[row, col]
            Z = np.ma.masked_invalid(mat)
            im = ax.imshow(
                Z,
                origin="lower",
                aspect="auto",
                extent=[-0.5, len(TAUS) - 0.5, -0.5, len(ALPHAS) - 0.5],
                **im_kwargs,
            )
            ax.set_xticks(range(len(TAUS)))
            ax.set_xticklabels([str(t) for t in TAUS])
            ax.set_yticks(range(len(ALPHAS)))
            ax.set_yticklabels([str(a) for a in ALPHAS])
            if row == last_row:
                ax.set_xlabel(r"actor.spi_tau ($\tau$)")
            if col == 0:
                ax.set_ylabel("dynamics.subgoal_value_alpha (α)")
            ax.set_title(f"{name}\nepoch {ep}")

            for i in range(len(ALPHAS)):
                for j in range(len(TAUS)):
                    ax.text(
                        j,
                        i,
                        fmt(mat[i, j]),
                        ha="center",
                        va="center",
                        color="black",
                        fontsize=9,
                        path_effects=pe_white,
                    )
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    fig.suptitle(suptitle, fontsize=11)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs-root", type=Path, default=Path(__file__).resolve().parent.parent / "runs")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "docs" / "figures",
    )
    p.add_argument(
        "--giant-discount-snapshot",
        type=float,
        metavar="GAMMA",
        default=None,
        help="If set, only write giant heatmap PNG using runs with critic_agent.discount == GAMMA "
        "(e.g. 0.99 before re-sweep at 0.995). Large heatmap is skipped.",
    )
    args = p.parse_args()
    out_dir = args.out_dir

    if args.giant_discount_snapshot is None:
        epochs_large = (100, 200, 300)
        idm_l, actor_l, notes_l = collect(args.runs_root, "large-navigate", epochs_large)
        for n in notes_l:
            print(n)
        plot_composite(
            idm_l,
            actor_l,
            epochs_large,
            out_dir / "dynamics_tau_alpha_sweep_antmaze_large_heatmaps.png",
            "AntMaze **large** — linear dynamics τ × α  (run_group: dynamics_tau_alpha_sweep)\n"
            "Rows: IDM success, Actor success, (Actor−IDM)/IDM.  "
            "τ=10,α=0.3 may use 20260425_224314 (theta_linear) eval — see plot script.",
        )
        print("wrote", out_dir / "dynamics_tau_alpha_sweep_antmaze_large_heatmaps.png")

    epochs_giant = (100, 200, 300, 400, 500)
    gdisc = args.giant_discount_snapshot
    idm_g, actor_g, notes_g = collect(
        args.runs_root, "giant-navigate", epochs_giant, critic_discount=gdisc
    )
    for n in notes_g:
        print(n)
    if gdisc is not None:
        tag = str(gdisc).replace(".", "p")  # 0.99 -> 0p99
        giant_name = f"dynamics_tau_alpha_sweep_antmaze_giant_heatmaps_gamma{tag}.png"
        gamma_note = f"critic γ={gdisc} only; "
    else:
        giant_name = "dynamics_tau_alpha_sweep_antmaze_giant_heatmaps.png"
        gamma_note = ""
    plot_composite(
        idm_g,
        actor_g,
        epochs_giant,
        out_dir / giant_name,
        "AntMaze **giant** — linear dynamics τ × α  (500 epochs; partial data if runs incomplete)\n"
        f"{gamma_note}"
        "Rows: IDM success, Actor success, (Actor−IDM)/IDM.",
    )
    print("wrote", out_dir / giant_name)


if __name__ == "__main__":
    main()
