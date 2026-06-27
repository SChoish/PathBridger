#!/usr/bin/env python3
"""Flow + TRL horizon sweep: bridge/subgoal path length K in {5, 10}.

``horizon`` drives dynamics_N, subgoal_steps, and full_chunk_horizon (h_a=5 fixed).
Train subgoal_num_samples stays at 4; final eval sweeps N=1,4,8,16 @ epoch 600.

Envs (gap/wmax per family):
  - puzzle-3x3 / puzzle-4x4: gap=5, wmax=5
  - cube single/double/triple: gap=10, wmax=5
  - antmaze medium/large/giant: gap=5, wmax=5
"""

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
OUT = REPO / 'config' / 'flow_k_sweep'

WMAX = 5.0
HORIZON_VALUES = (5, 10)
TRAIN_SUBGOAL_SAMPLES = 4
FINAL_EVAL_NS = '1,4,8,16'

FLOW_DYNAMICS_BASE: dict[str, Any] = {
    'subgoal_distribution': 'flow',
    'subgoal_stochastic_loss': 'mse',
    'subgoal_value_alpha': 0.0,
    'subgoal_flow_steps': 8,
    'subgoal_flow_t_min': 1.0e-4,
    'subgoal_flow_noise_scale': 1.0,
    'subgoal_temperature': 1.0,
    'subgoal_eval_selection': 'best_of_n_value',
    'subgoal_eval_include_zero_candidate': False,
    'subgoal_eval_seed': 0,
}

# puzzle -> cube -> antmaze
ENV_SPECS: list[dict[str, Any]] = [
    {
        'stem': 'puzzle_3x3',
        'env_name': 'puzzle-3x3-play-v0',
        'gap': 5.0,
        'discount': 0.99,
        'batch_size': 1024,
        'value_distance_weight_power': 0.5,
        'value_goal_sampling': PUZZLE_VALUE_GOAL_SAMPLING,
        'actor_goal_sampling': PUZZLE_ACTOR_GOAL_SAMPLING,
        'kappa_b': 0.6,
        'kappa_d': 0.6,
    },
    {
        'stem': 'puzzle_4x4',
        'env_name': 'puzzle-4x4-play-v0',
        'gap': 5.0,
        'discount': 0.99,
        'batch_size': 1024,
        'value_distance_weight_power': 2.0,
        'value_goal_sampling': PUZZLE_VALUE_GOAL_SAMPLING,
        'actor_goal_sampling': PUZZLE_ACTOR_GOAL_SAMPLING,
        'kappa_b': 0.9,
        'kappa_d': 0.9,
    },
    {
        'stem': 'cube_single',
        'env_name': 'cube-single-play-v0',
        'gap': 10.0,
        'discount': 0.99,
        'batch_size': 1024,
        'value_distance_weight_power': 1.0,
        'kappa_b': 0.9,
        'kappa_d': 0.9,
    },
    {
        'stem': 'cube_double',
        'env_name': 'cube-double-play-v0',
        'gap': 10.0,
        'discount': 0.99,
        'batch_size': 1024,
        'value_distance_weight_power': 1.0,
        'kappa_b': 0.6,
        'kappa_d': 0.6,
    },
    {
        'stem': 'cube_triple',
        'env_name': 'cube-triple-play-v0',
        'gap': 10.0,
        'discount': 0.995,
        'batch_size': 4096,
        'value_distance_weight_power': 1.0,
        'kappa_b': 0.8,
        'kappa_d': 0.8,
    },
    {
        'stem': 'antmaze_medium',
        'env_name': 'antmaze-medium-navigate-v0',
        'gap': 5.0,
        'discount': 0.99,
        'batch_size': 1024,
        'value_distance_weight_power': 0.0,
        'kappa_b': 0.7,
        'kappa_d': 0.7,
    },
    {
        'stem': 'antmaze_large',
        'env_name': 'antmaze-large-navigate-v0',
        'gap': 5.0,
        'discount': 0.995,
        'batch_size': 1024,
        'value_distance_weight_power': 0.0,
        'kappa_b': 0.9,
        'kappa_d': 0.9,
    },
    {
        'stem': 'antmaze_giant',
        'env_name': 'antmaze-giant-navigate-v0',
        'gap': 5.0,
        'discount': 0.99,
        'batch_size': 1024,
        'value_distance_weight_power': 0.0,
        'kappa_b': 0.8,
        'kappa_d': 0.8,
    },
]


def _gap_tag(gap: float) -> str:
    return str(int(gap)) if abs(gap - round(gap)) < 1e-9 else format(gap, 'g').replace('.', 'p')


def build_config(spec: dict[str, Any], *, horizon: int) -> dict[str, Any]:
    gap = float(spec['gap'])
    dynamics_overrides = deepcopy(FLOW_DYNAMICS_BASE)
    dynamics_overrides.update(
        {
            'subgoal_num_samples': TRAIN_SUBGOAL_SAMPLES,
            'subgoal_eval_num_samples': TRAIN_SUBGOAL_SAMPLES,
            'subgoal_value_gap_scale': gap,
            'subgoal_value_weight_max': WMAX,
        }
    )
    value_sampling = spec.get('value_goal_sampling', MAZE_VALUE_GOAL_SAMPLING)
    actor_sampling = spec.get('actor_goal_sampling', MAZE_ACTOR_GOAL_SAMPLING)
    gap_tag = _gap_tag(gap)
    cfg = build_trl_run_config(
        env_name=str(spec['env_name']),
        run_group=f'FlowG{gap_tag}W5H{horizon}TRL_rd_sd_{spec["stem"]}',
        gap_scale=gap,
        weight_max=WMAX,
        discount=float(spec['discount']),
        value_distance_weight_power=float(spec['value_distance_weight_power']),
        batch_size=int(spec['batch_size']),
        horizon=int(horizon),
        dynamics_overrides=dynamics_overrides,
        critic_overrides={
            'kappa_b': float(spec['kappa_b']),
            'kappa_d': float(spec['kappa_d']),
        },
        value_goal_sampling=deepcopy(value_sampling),
        actor_goal_sampling=deepcopy(actor_sampling),
    )
    cfg['final_eval_subgoal_eval_num_samples'] = FINAL_EVAL_NS
    return cfg


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    # Remove stale sample-count sweep configs.
    for old in OUT.glob('flow_g*_w5_k*_trl_*_rd_sd.yaml'):
        old.unlink()

    manifest: list[str] = []
    for spec in ENV_SPECS:
        for horizon in HORIZON_VALUES:
            cfg = build_config(spec, horizon=horizon)
            gap_tag = _gap_tag(float(spec['gap']))
            out = OUT / f'flow_g{gap_tag}_w5_h{horizon}_trl_{spec["stem"]}_rd_sd.yaml'
            header = (
                f'# Flow-BC + TRL: gap={spec["gap"]}, wmax=5, horizon_K={horizon}, '
                f'train_N={TRAIN_SUBGOAL_SAMPLES}, final_eval_N={FINAL_EVAL_NS}, h_a=5\n'
                f'# env={cfg["env_name"]} gamma={cfg["critic_agent"]["discount"]}\n'
            )
            dump_run_config_yaml(out, cfg, header=header)
            manifest.append(str(out.relative_to(REPO)))
            print(out)

    manifest_path = OUT / '_manifest.txt'
    with open(manifest_path, 'w', encoding='utf-8') as f:
        f.write(f'# {len(manifest)} Flow horizon-sweep configs (K={list(HORIZON_VALUES)})\n')
        for line in manifest:
            f.write(line + '\n')
    print(f'Manifest: {manifest_path}')


if __name__ == '__main__':
    main()
