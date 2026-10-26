"""Paper PBG puzzle-3x3 configuration.

Unlike the general/data-size preset, the July paper run used a 50/50 geometric
trajectory/random endpoint-goal mixture.  In the unified four-tuple order this
is ``(current, geometric, trajectory, random) = (0, .5, 0, .5)``.
"""

from configs._base import best_config


def get_config():
    config = best_config(
        env_name='puzzle-3x3-play-v0',
        endpoint_distribution='gaussian',
        horizon=40,
        discount=0.99,
        endpoint_value_scale=10.0,
        value_distance_weight_power=0.5,
        eval_num_candidates=2,
        eval_temperature=0.25,
    )
    config.training_objective_protocol = (
        'reported-run-legacy-v1'
    )
    config.actor_p = (0.0, 0.5, 0.0, 0.5)
    config.critic_p = (0.0, 1.0, 0.0, 0.0)
    return config
