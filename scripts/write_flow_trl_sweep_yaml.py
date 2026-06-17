#!/usr/bin/env python3
"""Emit Flow subgoal + TRL training configs.

Training sweep:
  env order: puzzle -> cube -> antmaze
  gap: 1, 3, 5, 10
  wmax: 5
  train subgoal_num_samples: 1

The runner performs post-training final evals with subgoal_eval_num_samples
1, 2, 4, 8, and 16.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from flow_trl_sweep_common import (
    ENV_ORDER,
    FINAL_EVAL_N_VALUES,
    FLOW_TRL_VARIANTS,
    GAP_VALUES,
    TRAIN_N,
    TRAIN_WMAX,
    build_flow_trl_sweep_config,
)
from utils.trl_critic_config import TRL_ENV_SPECS
from yaml_run_config import dump_run_config_yaml

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / 'config' / 'sweep_flow_trl_finaleval'


def _probe_env(env_name: str) -> tuple[bool, str]:
    try:
        os.environ.setdefault('MUJOCO_GL', 'egl')
        import ogbench

        out = ogbench.make_env_and_datasets(str(env_name), compact_dataset=True, env_only=True)
        env = out[0] if isinstance(out, tuple) else out
        while hasattr(env, 'env'):
            env = env.env
        obs_dim = int(env.observation_space.shape[0])
        return True, f'{env_name}: obs_dim={obs_dim} ok'
    except Exception as e:
        return False, f'{env_name}: probe failed: {e!r}'


def _probe_all_envs() -> None:
    seen: set[str] = set()
    for prefix in ENV_ORDER:
        name = str(TRL_ENV_SPECS[prefix]['env_name'])
        if name in seen:
            continue
        seen.add(name)
        ok, msg = _probe_env(name)
        print(('OK' if ok else 'WARN'), msg)


def write_sweep(outdir: Path | None = None) -> list[str]:
    out = outdir or OUT
    out.mkdir(parents=True, exist_ok=True)
    manifest: list[str] = []
    written_paths: set[Path] = set()

    for env_prefix in ENV_ORDER:
        for variant_suffix, variant in FLOW_TRL_VARIANTS.items():
            cfg = build_flow_trl_sweep_config(
                env_prefix=env_prefix,
                variant_suffix_key=variant_suffix,
                variant=variant,
                run_group_prefix='flow_trl_feval_',
            )
            out_path = out / f'{env_prefix}_{variant_suffix}.yaml'
            note = str(variant.get('note', '')).strip()
            note_line = f'# {note}\n' if note else ''
            cri = cfg['critic_agent']
            header = (
                '# Flow subgoal + TRL sweep: train N=1, final eval N sweep\n'
                f'# env={cfg["env_name"]} gap={variant["gap"]} wmax={variant["wmax"]} '
                f'train_N={variant["n"]} gamma={cri["discount"]} '
                f'vdist_pow={cri["value_distance_weight_power"]}\n'
                f'# final_eval_subgoal_eval_num_samples={FINAL_EVAL_N_VALUES}; episodes_per_task=25\n'
                f'{note_line}'
            )
            dump_run_config_yaml(out_path, cfg, header=header)
            written_paths.add(out_path)
            manifest.append(str(out_path.relative_to(REPO)))
            print(out_path)

    for stale in out.glob('*.yaml'):
        if stale not in written_paths:
            stale.unlink()
            print(f'removed stale: {stale}')

    manifest_path = out / '_manifest.txt'
    with open(manifest_path, 'w', encoding='utf-8') as f:
        f.write(f'# {len(manifest)} configs (8 envs × {len(GAP_VALUES)} gaps)\n')
        f.write('# order: puzzle (p3,p4) -> cube (cs,cd,ct) -> antmaze (amm,aml,amg)\n')
        f.write('# critic: utils/trl_critic_config.py gap10 baseline\n')
        f.write(f'# train: gap={GAP_VALUES}; wmax={TRAIN_WMAX}; subgoal_num_samples={TRAIN_N}\n')
        f.write(f'# post-train final eval: subgoal_eval_num_samples={FINAL_EVAL_N_VALUES}; episodes_per_task=25\n')
        for line in manifest:
            f.write(line + '\n')
    print(f'Manifest: {manifest_path}')
    return manifest


def main() -> None:
    p = argparse.ArgumentParser(description='Generate Flow+TRL final-eval sweep YAML configs.')
    p.add_argument('--probe', action='store_true', help='Probe env obs dims only (no YAML write).')
    p.add_argument('--outdir', type=Path, default=None, help='Override output directory.')
    args = p.parse_args()

    if args.probe:
        _probe_all_envs()
        return
    write_sweep(outdir=args.outdir)


if __name__ == '__main__':
    main()
