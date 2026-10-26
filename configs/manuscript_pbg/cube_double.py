"""Cube-double candidate with recovered paper settings and 6a objective."""

from configs._base import best_config
from utils.training_protocol import MANUSCRIPT_TRAINING_PROTOCOL


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
    config.training_objective_protocol = MANUSCRIPT_TRAINING_PROTOCOL
    config.actor_p = (0.0, 0.0, 1.0, 0.0)
    config.critic_p = (0.0, 1.0, 0.0, 0.0)
    return config
