"""TRL critic_agent presets for sweep YAML generation (gap10 baseline)."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

MAZE_ACTOR_GOAL_SAMPLING: dict[str, Any] = {
    'actor_p_curgoal': 0.0,
    'actor_p_trajgoal': 1.0,
    'actor_p_randomgoal': 0.0,
    'actor_geom_sample': False,
}

PUZZLE_ACTOR_GOAL_SAMPLING: dict[str, Any] = {
    'actor_p_curgoal': 0.0,
    'actor_p_trajgoal': 0.5,
    'actor_p_randomgoal': 0.5,
    'actor_geom_sample': True,
}

TRL_CRITIC_COMMON: dict[str, Any] = {
    'algorithm': 'trl',
    'critic_type': 'trl',
    'use_chunk_critic': False,
    'action_chunk_horizon': 5,
    'full_chunk_horizon': 25,
    'value_hidden_dims': [512, 512, 512],
    'layer_norm': True,
    'num_qs': 2,
    'q_agg': 'mean',
    'target_tau': 0.005,
    'tau_v': 0.7,
    'lambda_v_self': 1.0,
    'lambda_v_base': 1.0,
    'lambda_v_tri': 1.0,
    'value_base_horizon': 5,
    'value_transitive_reweight': True,
    'value_distance_weight_clip_min': 0.05,
    'value_distance_weight_clip_max': 1.0,
    'lambda_q_local': 1.0,
    'q_target_from_value': True,
    'goal_representation': 'full',
    'max_goal_steps_from_env': False,
    'clip_chunk_to_goal': True,
    'subgoal_value_bonus_type': 'transitive_product',
    'subgoal_value_log_eps': 1.0e-6,
    'subgoal_value_ratio_eps': 1.0e-3,
    'subgoal_value_ratio_clip': 5.0,
}

STANDARD_VALUE_GOAL_SAMPLING: dict[str, Any] = {
    'value_p_curgoal': 0.0,
    'value_p_trajgoal': 1.0,
    'value_p_randomgoal': 0.0,
    'value_geom_sample': True,
}

LONG_HORIZON_VALUE_GOAL_SAMPLING: dict[str, Any] = {
    'value_p_curgoal': 0.0,
    'value_p_trajgoal': 0.0,
    'value_p_randomgoal': 0.0,
    'value_geom_sample': False,
}

# gap10 baseline (write_trl_gap10_g099_sweep_yaml.py Table 5).
TRL_ENV_SPECS: dict[str, dict[str, Any]] = {
    'amm': {
        'env_name': 'antmaze-medium-navigate-v0',
        'stem': 'antmaze_medium',
        'regime': 'standard',
        'discount': 0.99,
        'value_distance_weight_power': 0.0,
        'batch_size': 1024,
        'actor_goal_sampling': MAZE_ACTOR_GOAL_SAMPLING,
    },
    'aml': {
        'env_name': 'antmaze-large-navigate-v0',
        'stem': 'antmaze_large',
        'regime': 'standard',
        'discount': 0.995,
        'value_distance_weight_power': 0.0,
        'batch_size': 1024,
        'actor_goal_sampling': MAZE_ACTOR_GOAL_SAMPLING,
    },
    'amg': {
        'env_name': 'antmaze-giant-navigate-v0',
        'stem': 'antmaze_giant',
        'regime': 'standard',
        'discount': 0.999,
        'value_distance_weight_power': 0.0,
        'batch_size': 1024,
        'actor_goal_sampling': MAZE_ACTOR_GOAL_SAMPLING,
    },
    'p3': {
        'env_name': 'puzzle-3x3-play-v0',
        'stem': 'puzzle_3x3',
        'regime': 'standard',
        'discount': 0.99,
        'value_distance_weight_power': 0.5,
        'batch_size': 1024,
        'actor_goal_sampling': PUZZLE_ACTOR_GOAL_SAMPLING,
    },
    'p4': {
        'env_name': 'puzzle-4x4-play-v0',
        'stem': 'puzzle_4x4',
        'regime': 'standard',
        'discount': 0.99,
        'value_distance_weight_power': 2.0,
        'batch_size': 1024,
        'actor_goal_sampling': PUZZLE_ACTOR_GOAL_SAMPLING,
    },
    'cs': {
        'env_name': 'cube-single-play-v0',
        'stem': 'cube_single',
        'regime': 'standard',
        'discount': 0.99,
        'value_distance_weight_power': 0.7,
        'batch_size': 1024,
        'actor_goal_sampling': MAZE_ACTOR_GOAL_SAMPLING,
    },
    'cd': {
        'env_name': 'cube-double-play-v0',
        'stem': 'cube_double',
        'regime': 'standard',
        'discount': 0.99,
        'value_distance_weight_power': 1.0,
        'batch_size': 1024,
        'actor_goal_sampling': MAZE_ACTOR_GOAL_SAMPLING,
    },
    'ct': {
        'env_name': 'cube-triple-play-v0',
        'stem': 'cube_triple',
        'regime': 'standard',
        'discount': 0.995,
        'value_distance_weight_power': 1.0,
        'batch_size': 4096,
        'actor_goal_sampling': MAZE_ACTOR_GOAL_SAMPLING,
    },
}

ENV_ORDER = ['amm', 'aml', 'amg', 'p3', 'p4', 'cs', 'cd', 'ct']


def trl_critic_agent_config(env_slug: str) -> dict[str, Any]:
    """Return full critic_agent block for env_slug (gap10 TRL baseline)."""
    spec = TRL_ENV_SPECS[env_slug]
    cfg = deepcopy(TRL_CRITIC_COMMON)
    cfg['discount'] = float(spec['discount'])
    cfg['value_distance_weight_power'] = float(spec['value_distance_weight_power'])
    if str(spec['regime']) == 'long_horizon':
        cfg.update(LONG_HORIZON_VALUE_GOAL_SAMPLING)
    else:
        cfg.update(STANDARD_VALUE_GOAL_SAMPLING)
    return cfg


def trl_env_spec(env_slug: str) -> dict[str, Any]:
    return TRL_ENV_SPECS[env_slug]
