"""Puzzle-3x3 candidate with recovered paper settings and 6a objective."""

from configs._base import best_config
from utils.training_protocol import MANUSCRIPT_TRAINING_PROTOCOL


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
    config.training_objective_protocol = MANUSCRIPT_TRAINING_PROTOCOL
    config.actor_p = (0.0, 0.5, 0.0, 0.5)
    config.critic_p = (0.0, 1.0, 0.0, 0.0)
    return config
