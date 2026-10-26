"""Complete default schema shared by the PBF and PBG environment presets."""

from __future__ import annotations

import ml_collections


def get_config() -> ml_collections.ConfigDict:
    """Return the state-PBF AntMaze-medium settings as clean defaults."""

    return ml_collections.ConfigDict(
        dict(
            env_name='antmaze-medium-navigate-v0',
            endpoint_distribution='flow',
            training_objective_protocol='reported-run-legacy-v1',
            horizon=25,
            discount=0.99,
            actor_p=(0.0, 0.0, 1.0, 0.0),
            critic_p=(0.0, 1.0, 0.0, 0.0),
            endpoint_value_scale=10.0,
            value_distance_weight_power=0.0,
            path_weight_beta=0.0,
            path_weight_min=0.1,
            path_weight_warmup=100_000,
            path_weight_ramp=100_000,
            path_distance_cap_multiplier=2.0,
            prefix_model='deterministic',
            prefix_loss_weight=1.0,
            prefix_rank=8,
            prefix_sigma_floor=1e-3,
            prefix_scale_floor=1e-3,
            prefix_flow_steps=8,
            eval_num_candidates=1,
            eval_temperature=1.0,
            eval_prefix_selection='sample_one',
            eval_num_prefix_samples=1,
            eval_prefix_temperature=1.0,
            eval_include_deterministic_prefix=False,
            offline_action_free=False,
        )
    )


__all__ = ['get_config']
