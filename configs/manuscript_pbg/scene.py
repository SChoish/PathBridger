"""Provenance-limited scene candidate with the 6a objective."""

from configs._base import best_config
from utils.training_protocol import MANUSCRIPT_TRAINING_PROTOCOL


def get_config():
    config = best_config(
        env_name='scene-play-v0',
        endpoint_distribution='gaussian',
        horizon=40,
        discount=0.99,
        endpoint_value_scale=5.0,
        value_distance_weight_power=1.0,
        eval_num_candidates=16,
        eval_temperature=0.5,
    )
    config.training_objective_protocol = MANUSCRIPT_TRAINING_PROTOCOL
    config.actor_p = (0.0, 0.0, 1.0, 0.0)
    config.critic_p = (0.0, 1.0, 0.0, 0.0)
    return config
