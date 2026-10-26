"""Paper PBG cube-double configuration.

Runtime: legacy Python tree represented by 12bc5ab723ad84fde1610873bd1625b3105cce63.
Training source: the recovered July run config used by the canonical PBG runs.
"""

from configs._base import best_config


def get_config():
    config = best_config(
        env_name='cube-double-play-v0',
        endpoint_distribution='gaussian',
        horizon=25,
        discount=0.99,
        endpoint_value_scale=10.0,
        value_distance_weight_power=1.0,
        eval_num_candidates=1,
        eval_temperature=0.0,
    )
    config.training_objective_protocol = (
        'reported-run-legacy-v1'
    )
    config.actor_p = (0.0, 0.0, 1.0, 0.0)
    config.critic_p = (0.0, 1.0, 0.0, 0.0)
    return config
