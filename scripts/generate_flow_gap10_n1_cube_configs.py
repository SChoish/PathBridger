#!/usr/bin/env python3
"""Flow + TRL cube configs: gap=10, wmax=5, train N=1, final eval N=1,4,8,16."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from yaml_run_config import MAZE_ACTOR_GOAL_SAMPLING, MAZE_VALUE_GOAL_SAMPLING, build_trl_run_config, dump_run_config_yaml

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / 'config' / 'flow_gap10_w5_n1_cube'

GAP = 10.0
WMAX = 5.0
TRAIN_N = 1
FINAL_EVAL_NS = '1,4,8,16'

FLOW_DYNAMICS_PATCH: dict[str, Any] = {
    'subgoal_distribution': 'flow',
    'subgoal_stochastic_loss': 'mse',
    'subgoal_num_samples': TRAIN_N,
    'subgoal_value_alpha': 0.0,
    'subgoal_value_gap_scale': GAP,
    'subgoal_value_weight_max': WMAX,
    'subgoal_flow_steps': 8,
    'subgoal_flow_t_min': 1.0e-4,
    'subgoal_flow_noise_scale': 1.0,
    'subgoal_temperature': 1.0,
    'subgoal_eval_selection': 'best_of_n_value',
    'subgoal_eval_num_samples': TRAIN_N,
    'subgoal_eval_include_zero_candidate': False,
    'subgoal_eval_seed': 0,
}

SPECS: list[dict[str, Any]] = [
    {
        'stem': 'cube_double',
        'env_name': 'cube-double-play-v0',
        'discount': 0.99,
        'batch_size': 1024,
        'value_distance_weight_power': 1.0,
        'kappa_b': 0.6,
        'kappa_d': 0.6,
    },
    {
        'stem': 'cube_triple',
        'env_name': 'cube-triple-play-v0',
        'discount': 0.995,
        'batch_size': 4096,
        'value_distance_weight_power': 1.0,
        'kappa_b': 0.8,
        'kappa_d': 0.8,
    },
]


def build_config(spec: dict[str, Any]) -> dict[str, Any]:
    cfg = build_trl_run_config(
        env_name=str(spec['env_name']),
        run_group=f'FlowGap10W5N1TRL_rd_sd_{spec["stem"]}',
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
        value_goal_sampling=deepcopy(MAZE_VALUE_GOAL_SAMPLING),
        actor_goal_sampling=deepcopy(MAZE_ACTOR_GOAL_SAMPLING),
    )
    cfg['final_eval_subgoal_eval_num_samples'] = FINAL_EVAL_NS
    return cfg


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest: list[str] = []
    for spec in SPECS:
        cfg = build_config(spec)
        out = OUT / f'flow_gap10_w5_n1_trl_{spec["stem"]}_rd_sd.yaml'
        header = (
            '# Flow-BC + TRL: gap=10, wmax=5, train_N=1, final_eval_N=1,4,8,16\n'
            f'# env={cfg["env_name"]} gamma={cfg["critic_agent"]["discount"]}\n'
        )
        dump_run_config_yaml(out, cfg, header=header)
        manifest.append(str(out.relative_to(REPO)))
        print(out)

    manifest_path = OUT / '_manifest.txt'
    with open(manifest_path, 'w', encoding='utf-8') as f:
        f.write(f'# {len(manifest)} cube Flow configs (gap10/w5/N1 train)\n')
        for line in manifest:
            f.write(line + '\n')
    print(f'Manifest: {manifest_path}')


if __name__ == '__main__':
    main()
