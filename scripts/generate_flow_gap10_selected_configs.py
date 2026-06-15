#!/usr/bin/env python3
"""Emit the selected Flow + TRL configs requested for gap=10, wmax=5."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from yaml_run_config import (
    MAZE_ACTOR_GOAL_SAMPLING,
    MAZE_VALUE_GOAL_SAMPLING,
    PUZZLE_ACTOR_GOAL_SAMPLING,
    PUZZLE_VALUE_GOAL_SAMPLING,
    build_trl_run_config,
    dump_run_config_yaml,
)

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / 'config' / 'flow_gap10_selected'

GAP = 10.0
WMAX = 5.0

FLOW_DYNAMICS_PATCH: dict[str, Any] = {
    'subgoal_distribution': 'flow',
    'subgoal_stochastic_loss': 'mse',
    'subgoal_num_samples': 4,
    'subgoal_value_alpha': 0.0,
    'subgoal_value_gap_scale': GAP,
    'subgoal_value_weight_max': WMAX,
    'subgoal_flow_steps': 8,
    'subgoal_flow_t_min': 1.0e-4,
    'subgoal_flow_noise_scale': 1.0,
    'subgoal_temperature': 1.0,
    'subgoal_eval_selection': 'best_of_n_value',
    'subgoal_eval_num_samples': 4,
    'subgoal_eval_include_zero_candidate': False,
    'subgoal_eval_seed': 0,
}

SPECS: list[dict[str, Any]] = [
    {
        'stem': 'antmaze_large',
        'env_name': 'antmaze-large-navigate-v0',
        'discount': 0.995,
        'batch_size': 1024,
        'value_distance_weight_power': 0.0,
        'kappa_b': 0.9,
        'kappa_d': 0.9,
        'value_goal_sampling': MAZE_VALUE_GOAL_SAMPLING,
        'actor_goal_sampling': MAZE_ACTOR_GOAL_SAMPLING,
    },
    {
        'stem': 'antmaze_giant',
        'env_name': 'antmaze-giant-navigate-v0',
        'discount': 0.999,
        'batch_size': 1024,
        'value_distance_weight_power': 0.0,
        'kappa_b': 0.8,
        'kappa_d': 0.8,
        'value_goal_sampling': MAZE_VALUE_GOAL_SAMPLING,
        'actor_goal_sampling': MAZE_ACTOR_GOAL_SAMPLING,
    },
    {
        'stem': 'cube_double',
        'env_name': 'cube-double-play-v0',
        'discount': 0.99,
        'batch_size': 1024,
        'value_distance_weight_power': 1.0,
        'kappa_b': 0.6,
        'kappa_d': 0.6,
        'value_goal_sampling': MAZE_VALUE_GOAL_SAMPLING,
        'actor_goal_sampling': MAZE_ACTOR_GOAL_SAMPLING,
    },
    {
        'stem': 'cube_triple',
        'env_name': 'cube-triple-play-v0',
        'discount': 0.995,
        'batch_size': 4096,
        'value_distance_weight_power': 1.0,
        'kappa_b': 0.8,
        'kappa_d': 0.8,
        'value_goal_sampling': MAZE_VALUE_GOAL_SAMPLING,
        'actor_goal_sampling': MAZE_ACTOR_GOAL_SAMPLING,
    },
    {
        'stem': 'puzzle_3x3',
        'env_name': 'puzzle-3x3-play-v0',
        'discount': 0.99,
        'batch_size': 1024,
        'value_distance_weight_power': 0.5,
        'kappa_b': 0.6,
        'kappa_d': 0.6,
        'value_goal_sampling': PUZZLE_VALUE_GOAL_SAMPLING,
        'actor_goal_sampling': PUZZLE_ACTOR_GOAL_SAMPLING,
    },
    {
        'stem': 'puzzle_4x4',
        'env_name': 'puzzle-4x4-play-v0',
        'discount': 0.995,
        'batch_size': 1024,
        'value_distance_weight_power': 2.0,
        'kappa_b': 0.9,
        'kappa_d': 0.9,
        'value_goal_sampling': PUZZLE_VALUE_GOAL_SAMPLING,
        'actor_goal_sampling': PUZZLE_ACTOR_GOAL_SAMPLING,
    },
]


def build_config(spec: dict[str, Any]) -> dict[str, Any]:
    return build_trl_run_config(
        env_name=str(spec['env_name']),
        run_group=f'FlowGap10W5TRL_rd_sd_{spec["stem"]}',
        gap_scale=GAP,
        weight_max=WMAX,
        discount=float(spec['discount']),
        value_distance_weight_power=float(spec['value_distance_weight_power']),
        batch_size=int(spec['batch_size']),
        dynamics_overrides=deepcopy(FLOW_DYNAMICS_PATCH),
        critic_overrides={
            'kappa_b': float(spec['kappa_b']),
            'kappa_d': float(spec['kappa_d']),
        },
        value_goal_sampling=deepcopy(spec['value_goal_sampling']),
        actor_goal_sampling=deepcopy(spec['actor_goal_sampling']),
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest: list[str] = []
    for spec in SPECS:
        cfg = build_config(spec)
        out = OUT / f'flow_gap10_w5_trl_{spec["stem"]}_rd_sd.yaml'
        header = (
            '# Plain Flow-BC + TRL selected sweep, gap=10.0, wmax=5.0, rd_sd\n'
            f'# env={cfg["env_name"]} gamma={cfg["critic_agent"]["discount"]}\n'
        )
        dump_run_config_yaml(out, cfg, header=header)
        manifest.append(str(out.relative_to(REPO)))
        print(out)

    manifest_path = OUT / '_manifest.txt'
    with open(manifest_path, 'w', encoding='utf-8') as f:
        f.write(f'# {len(manifest)} selected Flow configs, gap=10.0, wmax=5.0\n')
        for line in manifest:
            f.write(line + '\n')
    print(f'Manifest: {manifest_path}')


if __name__ == '__main__':
    main()
