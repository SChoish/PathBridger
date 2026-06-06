#!/usr/bin/env python3
"""Smoke checks for plain Flow-BC + eval-time value BoN subgoal selection."""

from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import jax
import jax.numpy as jnp
import numpy as np

from agents.critic import CriticAgent, get_config as get_critic_config
from agents.dynamics import DynamicsAgent, get_dynamics_config


def _dynamics_config():
    cfg = get_dynamics_config()
    cfg.batch_size = 4
    cfg.dynamics_N = 3
    cfg.subgoal_steps = 3
    cfg.subgoal_hidden_dims = (16, 16)
    cfg.path_residual_hidden_dims = (16, 16)
    cfg.residual_model_hidden_dims = (16, 16)
    cfg.idm_hidden_dims = (16, 16)
    cfg.subgoal_value_hidden_dims = (16, 16)
    cfg.layer_norm = False
    cfg.subgoal_value_layer_norm = False
    cfg.subgoal_goal_representation = 'full'
    cfg.subgoal_value_goal_representation = 'full'
    cfg.subgoal_distribution = 'flow'
    cfg.subgoal_target_mode = 'displacement'
    cfg.residual_target_mode = 'displacement'
    cfg.subgoal_flow_energy_weighted = False
    cfg.subgoal_flow_use_value_bonus = False
    cfg.subgoal_flow_steps = 2
    cfg.subgoal_num_samples = 4
    cfg.subgoal_eval_selection = 'best_of_n_value'
    cfg.subgoal_eval_num_samples = 4
    cfg.subgoal_eval_include_zero_candidate = False
    cfg.subgoal_eval_seed = 0
    return cfg


def _critic_agent():
    cfg = get_critic_config()
    cfg.action_chunk_horizon = 2
    cfg.full_chunk_horizon = 4
    cfg.value_hidden_dims = (16, 16)
    cfg.action_dim = 2
    ex_obs = np.zeros((4, 5), dtype=np.float32)
    ex_full = np.zeros((4, cfg.full_chunk_horizon * cfg.action_dim), dtype=np.float32)
    ex_part = np.zeros((4, cfg.action_chunk_horizon * cfg.action_dim), dtype=np.float32)
    return CriticAgent.create(
        seed=1,
        ex_observations=ex_obs,
        ex_full_chunk_actions=ex_full,
        ex_action_chunk_actions=ex_part,
        config=cfg,
        ex_goals=ex_obs,
    )


def main() -> None:
    batch_size = 4
    state_dim = 5
    action_dim = 2
    ex_obs = jnp.zeros((batch_size, state_dim), dtype=jnp.float32)
    ex_act = jnp.zeros((batch_size, action_dim), dtype=jnp.float32)
    agent = DynamicsAgent.create(0, ex_obs, _dynamics_config(), ex_actions=ex_act)
    critic = _critic_agent()

    obs = jnp.linspace(-1.0, 1.0, batch_size * state_dim, dtype=jnp.float32).reshape(batch_size, state_dim)
    goals = obs + 0.5
    rng = jax.random.PRNGKey(0)

    candidates, mu = agent.sample_subgoal_candidates(
        obs,
        goals,
        rng,
        num_candidates=4,
        include_mean=False,
    )
    assert candidates.shape == (batch_size, 4, state_dim), candidates.shape
    assert mu.shape == (batch_size, state_dim), mu.shape

    selected = agent.infer_subgoal_for_eval(obs, goals, critic_agent=critic, rng=rng)
    assert selected.shape == (batch_size, state_dim), selected.shape
    assert bool(jnp.all(jnp.isfinite(selected))), selected

    batch = {
        'observations': obs,
        'next_observations': obs + 0.05,
        'actions': jnp.ones((batch_size, action_dim), dtype=jnp.float32) * 0.1,
        'high_actor_goals': goals,
        'high_actor_targets': obs + 0.25,
        'trajectory_segment': obs[:, None, :] + jnp.linspace(0.0, 1.0, 4, dtype=jnp.float32)[None, :, None] * 0.25,
    }
    _, info = agent.update(batch)
    assert bool(jnp.isfinite(info['phase1/loss'])), info['phase1/loss']
    assert float(info['phase1/subgoal_flow_energy_weighted']) == 0.0
    print('plain flow BoN smoke passed')


if __name__ == '__main__':
    main()
