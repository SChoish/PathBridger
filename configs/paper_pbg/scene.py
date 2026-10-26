"""Best-supported reconstruction of the paper PBG scene configuration.

PB_logs fixes gap=5, gamma=.99, K=40, Best-(N,T)=(16,.5), and geometric
critic goals.  The original offrl config file is not present locally, so the
ordinary trajectory endpoint mix follows the surviving July K=40 scene config
and remains explicitly provenance-limited in the experiment manifest.
"""

from configs._base import best_config


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
    config.training_objective_protocol = (
        'reported-run-legacy-v1'
    )
    config.actor_p = (0.0, 0.0, 1.0, 0.0)
    config.critic_p = (0.0, 1.0, 0.0, 0.0)
    return config
