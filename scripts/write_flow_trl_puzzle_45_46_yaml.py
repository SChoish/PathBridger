#!/usr/bin/env python3
"""Flow+TRL configs for puzzle-4x5 and puzzle-4x6.

Gap sweep matches puzzle-4x4 feval: gap={1,3,5,10}, wmax=5, train_N=1.
Critic uses value_distance_weight_power=0 (distance lambda=0), unlike p4 (2.0).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from flow_trl_sweep_common import (
    FINAL_EVAL_N_VALUES,
    FLOW_DYNAMICS_BASE,
    FLOW_TRL_VARIANTS,
    final_eval_n_values_csv,
)
from utils.trl_critic_config import trl_critic_agent_config
from yaml_run_config import build_trl_run_config, dump_run_config_yaml

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / 'config' / 'sweep_flow_trl_puzzle_45_46'

PUZZLE_LARGE_ENVS: dict[str, dict] = {
    'p45': {
        'env_name': 'puzzle-4x5-play-v0',
        'stem': 'puzzle_4x5',
        'discount': 0.999,
        'value_distance_weight_power': 0.0,
        'batch_size': 1024,
    },
    'p46': {
        'env_name': 'puzzle-4x6-play-v0',
        'stem': 'puzzle_4x6',
        'discount': 0.999,
        'value_distance_weight_power': 0.0,
        'batch_size': 1024,
    },
}

# Policy (cur, geom, traj, rand) = (0, 0.5, 0, 0.5) for puzzle.
PUZZLE_LARGE_ACTOR_SAMPLING: dict[str, float | bool] = {
    'actor_p_curgoal': 0.0,
    'actor_p_trajgoal': 0.5,
    'actor_p_randomgoal': 0.5,
    'actor_geom_sample': True,
}

# Value TRL (cur, geom, traj, rand) = (0, 0, 1, 0).
PUZZLE_LARGE_VALUE_SAMPLING: dict[str, float | bool] = {
    'value_p_curgoal': 0.0,
    'value_p_trajgoal': 1.0,
    'value_p_randomgoal': 0.0,
    'value_geom_sample': False,
}

ENV_ORDER = ['p45', 'p46']


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


def build_config(*, env_prefix: str, variant_suffix: str, variant: dict) -> dict:
    spec = PUZZLE_LARGE_ENVS[env_prefix]
    from copy import deepcopy

    dynamics_overrides = deepcopy(FLOW_DYNAMICS_BASE)
    dynamics_overrides.update(
        {
            'subgoal_value_gap_scale': float(variant['gap']),
            'subgoal_value_weight_max': float(variant['wmax']),
            'subgoal_num_samples': int(variant['n']),
            'subgoal_eval_num_samples': int(variant['n']),
        }
    )
    critic_cfg = trl_critic_agent_config('p4')
    critic_cfg['discount'] = float(spec['discount'])
    critic_cfg['value_distance_weight_power'] = float(spec['value_distance_weight_power'])
    critic_cfg.update(PUZZLE_LARGE_VALUE_SAMPLING)

    cfg = build_trl_run_config(
        env_name=str(spec['env_name']),
        run_group=f'flow_trl_p456g999_{env_prefix}_{variant_suffix}',
        gap_scale=float(variant['gap']),
        weight_max=float(variant['wmax']),
        discount=float(spec['discount']),
        value_distance_weight_power=float(spec['value_distance_weight_power']),
        batch_size=int(spec['batch_size']),
        train_epochs=600,
        dynamics_overrides=dynamics_overrides,
        critic_overrides=critic_cfg,
        value_goal_sampling=deepcopy(PUZZLE_LARGE_VALUE_SAMPLING),
        actor_goal_sampling=deepcopy(PUZZLE_LARGE_ACTOR_SAMPLING),
    )
    cfg['final_eval_subgoal_eval_num_samples'] = final_eval_n_values_csv()
    return cfg


def write_sweep(outdir: Path | None = None) -> list[str]:
    out = outdir or OUT
    out.mkdir(parents=True, exist_ok=True)
    manifest: list[str] = []
    written: set[Path] = set()

    for env_prefix in ENV_ORDER:
        spec = PUZZLE_LARGE_ENVS[env_prefix]
        for variant_suffix, variant in FLOW_TRL_VARIANTS.items():
            cfg = build_config(
                env_prefix=env_prefix,
                variant_suffix=variant_suffix,
                variant=variant,
            )
            out_path = out / f'{env_prefix}_{variant_suffix}.yaml'
            header = (
                '# Flow subgoal + TRL: puzzle 4x5/4x6 gap sweep (4x4 feval gaps)\n'
                f'# env={cfg["env_name"]} gap={variant["gap"]} wmax={variant["wmax"]} '
                f'train_N={variant["n"]} gamma={cfg["critic_agent"]["discount"]} '
                f'vdist_pow={cfg["critic_agent"]["value_distance_weight_power"]}\n'
                '# policy (cur,geom,traj,rand)=(0,0.5,0,0.5); value TRL (0,0,1,0)\n'
                f'# final_eval_subgoal_eval_num_samples={FINAL_EVAL_N_VALUES}; episodes_per_task=25\n'
            )
            dump_run_config_yaml(out_path, cfg, header=header)
            written.add(out_path)
            manifest.append(str(out_path.relative_to(REPO)))
            print(out_path)

    for stale in out.glob('*.yaml'):
        if stale not in written:
            stale.unlink()
            print(f'removed stale: {stale}')

    manifest_path = out / '_manifest.txt'
    with open(manifest_path, 'w', encoding='utf-8') as f:
        f.write('# 8 configs: puzzle-4x5 + puzzle-4x6 × gap {1,3,5,10}\n')
        f.write('# gap/wmax/N match sweep_flow_trl_finaleval/p4; vdist_pow=0; gamma=0.999\n')
        f.write('# policy (cur,geom,traj,rand)=(0,0.5,0,0.5); value TRL (0,0,1,0)\n')
        for line in manifest:
            f.write(line + '\n')
    print(f'Manifest: {manifest_path}')
    return manifest


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--probe', action='store_true')
    p.add_argument('--outdir', type=Path, default=None)
    args = p.parse_args()
    if args.probe:
        for prefix in ENV_ORDER:
            ok, msg = _probe_env(PUZZLE_LARGE_ENVS[prefix]['env_name'])
            print(('OK' if ok else 'WARN'), msg)
        return
    write_sweep(outdir=args.outdir)


if __name__ == '__main__':
    main()
