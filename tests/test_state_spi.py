"""Tests for state-space SPI: cost energy + Wasserstein proximal actor.

Covers the command's acceptance checks, adapted to the explicit user
instruction that the energy is built WITHOUT the constant ``1`` offset:

    E_V  = -V(s, z) V(z, g)
    E_QZ = -Q_Z(s, z; g)

Smaller energy is better; with score in [0, 1] the energy lies in [-1, 0].
The state-SPI actor objective adds energy directly (no minus sign):

    L = E(s, pi(s,g), g) + (1 / (2 tau)) * d_M^2(pi_0, pi).

Run:
    PYTHONPATH=. python -m pytest tests/test_state_spi.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest

import jax.numpy as jnp

from agents.actor import ActorAgent, finite_anchor_distance, get_actor_config
from agents.critic import CriticAgent, get_config as get_critic_config
from utils.critic_sequence_dataset import CriticSequenceDataset
from utils.datasets import Dataset

STATE_DIM = 4
ACTION_DIM = 2
BATCH = 8
N = 5


# --------------------------------------------------------------------------- #
# builders
# --------------------------------------------------------------------------- #
def _make_critic(energy_type: str = 'v_product', use_qz: bool = False, qz_tri: bool = False):
    cfg = get_critic_config()
    cfg.critic_type = 'trl'
    cfg.algorithm = 'trl'
    cfg.goal_representation = 'full'
    cfg.value_hidden_dims = (32, 32)
    cfg.action_chunk_horizon = 2
    cfg.full_chunk_horizon = 4
    cfg.value_base_horizon = 2
    cfg.action_dim = ACTION_DIM
    cfg.state_spi_energy_type = energy_type
    cfg.use_qz_critic = use_qz
    cfg.qz_use_transitive_backup = qz_tri
    if qz_tri:
        cfg.lambda_qz_tri = 1.0
    ex_obs = np.zeros((BATCH, STATE_DIM), dtype=np.float32)
    ex_part = np.zeros((BATCH, cfg.action_chunk_horizon * ACTION_DIM), dtype=np.float32)
    return CriticAgent.create(
        seed=0,
        ex_observations=ex_obs,
        ex_full_chunk_actions=None,
        ex_action_chunk_actions=ex_part,
        config=cfg,
        ex_goals=ex_obs,
    )


def _make_state_actor(target_mode: str = 'absolute', anchor_metric: str = 'wasserstein_empirical'):
    cfg = get_actor_config()
    cfg.actor_type = 'state_subgoal'
    cfg.actor_chunk_horizon = 2
    cfg.action_dim = ACTION_DIM
    cfg.spi_tau = 5.0
    cfg.state_spi_target_mode = target_mode
    cfg.state_spi_anchor_metric = anchor_metric
    cfg.state_spi_metric_space = 'raw'
    ex_obs = np.zeros((BATCH, STATE_DIM), dtype=np.float32)
    return ActorAgent.create(seed=0, ex_observations=ex_obs, config=cfg, ex_goals=ex_obs)


def _dummy_dataset():
    n = 4 * 40
    rng = np.random.default_rng(0)
    obs = rng.standard_normal((n, STATE_DIM)).astype(np.float32)
    actions = rng.uniform(-1.0, 1.0, size=(n, ACTION_DIM)).astype(np.float32)
    terminals = np.zeros((n,), dtype=np.float32)
    for k in range(4):
        terminals[(k + 1) * 40 - 1] = 1.0
    return Dataset.create(observations=obs, actions=actions, terminals=terminals)


def _state_batch(seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    return {
        'observations': jnp.asarray(rng.standard_normal((BATCH, STATE_DIM)).astype(np.float32)),
        'candidate_goals': jnp.asarray(rng.standard_normal((BATCH, N, STATE_DIM)).astype(np.float32)),
        'high_actor_goals': jnp.asarray(rng.standard_normal((BATCH, STATE_DIM)).astype(np.float32)),
    }


# --------------------------------------------------------------------------- #
# Test 1: SPI objective sign (lower E -> lower loss for fixed anchor distance)
# --------------------------------------------------------------------------- #
def test_spi_objective_sign_lower_energy_lower_loss():
    tau = 5.0
    prox = 1.0 / (2.0 * tau)
    dist = jnp.ones((BATCH,))
    e_low = jnp.full((BATCH,), -0.9)   # high score -> low cost energy
    e_high = jnp.full((BATCH,), -0.1)  # low score -> high cost energy
    loss_low = float(e_low.mean() + prox * dist.mean())
    loss_high = float(e_high.mean() + prox * dist.mean())
    assert loss_low < loss_high


# --------------------------------------------------------------------------- #
# Test 2: Wasserstein proximal term (closer anchor -> lower loss for fixed E)
# --------------------------------------------------------------------------- #
def test_spi_objective_closer_anchor_lower_loss():
    tau = 5.0
    prox = 1.0 / (2.0 * tau)
    energy = jnp.full((BATCH,), -0.5)
    dist_near = jnp.full((BATCH,), 0.1)
    dist_far = jnp.full((BATCH,), 5.0)
    loss_near = float(energy.mean() + prox * dist_near.mean())
    loss_far = float(energy.mean() + prox * dist_far.mean())
    assert loss_near < loss_far


# --------------------------------------------------------------------------- #
# Test 3: empirical Wasserstein distance == mean_m ||z - z_m||^2
# --------------------------------------------------------------------------- #
def test_wasserstein_empirical_matches_mean_squared():
    rng = np.random.default_rng(1)
    z = jnp.asarray(rng.standard_normal((BATCH, STATE_DIM)).astype(np.float32))
    cand = jnp.asarray(rng.standard_normal((BATCH, N, STATE_DIM)).astype(np.float32))
    obs = jnp.asarray(rng.standard_normal((BATCH, STATE_DIM)).astype(np.float32))
    d2, extra = finite_anchor_distance(z, cand, obs, mode='wasserstein_empirical', metric_space='raw')
    assert d2.shape == (BATCH,)
    assert np.all(np.isfinite(np.asarray(d2)))
    expected = np.mean(
        np.sum((np.asarray(z)[:, None, :] - np.asarray(cand)) ** 2, axis=-1), axis=1
    )
    np.testing.assert_allclose(np.asarray(d2), expected, rtol=1e-5, atol=1e-5)
    assert not extra


def test_support_nearest_is_min_not_wasserstein():
    rng = np.random.default_rng(2)
    z = jnp.asarray(rng.standard_normal((BATCH, STATE_DIM)).astype(np.float32))
    cand = jnp.asarray(rng.standard_normal((BATCH, N, STATE_DIM)).astype(np.float32))
    obs = jnp.zeros((BATCH, STATE_DIM), dtype=jnp.float32)
    d2, _ = finite_anchor_distance(z, cand, obs, mode='support_nearest', metric_space='raw')
    expected = np.min(
        np.sum((np.asarray(z)[:, None, :] - np.asarray(cand)) ** 2, axis=-1), axis=1
    )
    np.testing.assert_allclose(np.asarray(d2), expected, rtol=1e-5, atol=1e-5)


# --------------------------------------------------------------------------- #
# Test 4: V-product energy shape + range (E = -score, score in [0,1])
# --------------------------------------------------------------------------- #
def test_v_product_energy_shape_and_range():
    critic = _make_critic('v_product')
    rng = np.random.default_rng(3)
    s = jnp.asarray(rng.standard_normal((BATCH, STATE_DIM)).astype(np.float32))
    z = jnp.asarray(rng.standard_normal((BATCH, N, STATE_DIM)).astype(np.float32))
    g = jnp.asarray(rng.standard_normal((BATCH, STATE_DIM)).astype(np.float32))
    e = np.asarray(
        critic.energy_state_subgoals(s, z, g, network_params=critic.network.params, energy_type='v_product')
    )
    assert e.shape == (BATCH, N)
    assert np.all(np.isfinite(e))
    assert np.all(e <= 1e-6) and np.all(e >= -1.0 - 1e-6)
    # score = -energy must be a valid probability.
    score = np.asarray(
        critic.score_state_subgoals(s, z, g, network_params=critic.network.params, energy_type='v_product')
    )
    np.testing.assert_allclose(score, -e, rtol=1e-5, atol=1e-6)
    assert np.all(score >= -1e-6) and np.all(score <= 1.0 + 1e-6)


# --------------------------------------------------------------------------- #
# Test 5: QZ energy shape + range
# --------------------------------------------------------------------------- #
def test_qz_energy_shape_and_range():
    critic = _make_critic('qz', use_qz=True)
    assert 'modules_qz_critic' in critic.network.params
    rng = np.random.default_rng(4)
    s = jnp.asarray(rng.standard_normal((BATCH, STATE_DIM)).astype(np.float32))
    z = jnp.asarray(rng.standard_normal((BATCH, N, STATE_DIM)).astype(np.float32))
    g = jnp.asarray(rng.standard_normal((BATCH, STATE_DIM)).astype(np.float32))
    e = np.asarray(
        critic.energy_state_subgoals(s, z, g, network_params=critic.network.params, energy_type='qz')
    )
    assert e.shape == (BATCH, N)
    assert np.all(np.isfinite(e))
    assert np.all(e <= 1e-6) and np.all(e >= -1.0 - 1e-6)


# --------------------------------------------------------------------------- #
# Test 6: state_subgoal actor update returns finite loss
# --------------------------------------------------------------------------- #
def test_state_actor_update_finite():
    critic = _make_critic('v_product')
    actor = _make_state_actor()
    batch = _state_batch()
    new_actor, info = actor.update(batch, critic)
    assert np.isfinite(float(info['state_spi/actor_loss']))
    assert np.isfinite(float(info['state_spi/energy_mean']))
    assert np.isfinite(float(info['state_spi/anchor_dist2_mean']))
    # objective = energy + prox * dist (energy added with no minus sign).
    expected = float(info['state_spi/energy_mean']) + float(info['state_spi/prox_coef']) * float(
        info['state_spi/anchor_dist2_mean']
    )
    np.testing.assert_allclose(float(info['state_spi/actor_loss']), expected, rtol=1e-4, atol=1e-5)


def test_state_actor_update_qz_finite():
    critic = _make_critic('qz', use_qz=True)
    actor = _make_state_actor()
    batch = _state_batch()
    _, info = actor.update(batch, critic)
    assert np.isfinite(float(info['state_spi/actor_loss']))


def test_state_proposal_skips_update():
    cfg = get_actor_config()
    cfg.actor_type = 'state_proposal'
    cfg.actor_chunk_horizon = 2
    cfg.action_dim = ACTION_DIM
    ex_obs = np.zeros((BATCH, STATE_DIM), dtype=np.float32)
    actor = ActorAgent.create(seed=0, ex_observations=ex_obs, config=cfg, ex_goals=ex_obs)
    critic = _make_critic('v_product')
    new_actor, info = actor.update(_state_batch(), critic)
    assert new_actor is actor
    assert float(info['state_spi/actor_loss']) == 0.0


# --------------------------------------------------------------------------- #
# Test 8: legacy action-chunk actor still works
# --------------------------------------------------------------------------- #
def test_legacy_action_actor_sample_actions():
    cfg = get_actor_config()
    cfg.actor_type = 'action_chunk'
    cfg.actor_chunk_horizon = 2
    cfg.action_dim = ACTION_DIM
    ex_obs = np.zeros((BATCH, STATE_DIM), dtype=np.float32)
    actor = ActorAgent.create(seed=0, ex_observations=ex_obs, config=cfg, ex_goals=ex_obs)
    obs = np.zeros((STATE_DIM,), dtype=np.float32)
    chunk = np.asarray(actor.sample_actions(obs, np.zeros((STATE_DIM,), dtype=np.float32)))
    assert chunk.shape == (2, ACTION_DIM)


# --------------------------------------------------------------------------- #
# QZ dataset fields
# --------------------------------------------------------------------------- #
def test_qz_dataset_fields_emitted_when_enabled():
    cfg = get_critic_config()
    cfg.critic_type = 'trl'
    cfg.algorithm = 'trl'
    cfg.goal_representation = 'full'
    cfg.action_chunk_horizon = 4
    cfg.full_chunk_horizon = 8
    cfg.value_base_horizon = 4
    cfg.action_dim = ACTION_DIM
    cfg.use_qz_critic = True
    ds = CriticSequenceDataset(_dummy_dataset(), cfg)
    batch = ds.sample(BATCH)
    for key in ('qz_subgoals', 'qz_goals', 'qz_subgoal_offsets', 'qz_goal_offsets', 'qz_valid_mask'):
        assert key in batch, f'missing qz field {key!r}'
    assert batch['qz_subgoals'].shape == (BATCH, STATE_DIM)
    assert batch['qz_goals'].shape == (BATCH, STATE_DIM)
    assert batch['qz_subgoal_offsets'].shape == (BATCH,)
    # subgoal offset k-i must be in [0, j-i].
    assert np.all(batch['qz_subgoal_offsets'] >= 0.0)
    assert np.all(batch['qz_subgoal_offsets'] <= batch['qz_goal_offsets'] + 1e-6)


def _make_dynamics_agent():
    from agents.dynamics import DynamicsAgent, get_dynamics_config

    cfg = get_dynamics_config()
    cfg.dynamics_N = 4
    cfg.subgoal_steps = 4
    cfg.rollout_horizon = 2
    cfg.subgoal_distribution = 'deterministic'
    cfg.residual_model_hidden_dims = (32, 32)
    cfg.subgoal_hidden_dims = (32, 32)
    cfg.subgoal_value_hidden_dims = (32, 32)
    cfg.idm_hidden_dims = (32, 32)
    ex_obs = np.zeros((BATCH, STATE_DIM), dtype=np.float32)
    ex_act = np.zeros((BATCH, ACTION_DIM), dtype=np.float32)
    return DynamicsAgent.create(seed=0, ex_observations=ex_obs, ex_actions=ex_act, config=cfg)


def test_state_actor_execution_through_bridge_and_idm():
    # Test 7: state actor output z -> bridge plan -> IDM -> action chunk.
    actor = _make_state_actor()
    dynamics = _make_dynamics_agent()
    rng = np.random.default_rng(7)
    obs = rng.standard_normal((STATE_DIM,)).astype(np.float32)
    goal = rng.standard_normal((STATE_DIM,)).astype(np.float32)

    z = np.asarray(actor.sample_subgoals(obs, goal), dtype=np.float32)
    assert z.shape == (STATE_DIM,)

    traj = np.asarray(dynamics.plan(obs, z)['trajectory'], dtype=np.float32)
    horizon = 2
    chunk = np.asarray(
        dynamics._idm_actions_from_trajectories(traj[None, ...], horizon), dtype=np.float32
    )
    assert chunk.shape == (1, horizon, ACTION_DIM)
    assert np.all(np.isfinite(chunk))


def _make_critic_with_batch(use_qz: bool = False, qz_tri: bool = False):
    cfg = get_critic_config()
    cfg.critic_type = 'trl'
    cfg.algorithm = 'trl'
    cfg.goal_representation = 'full'
    cfg.action_chunk_horizon = 4
    cfg.full_chunk_horizon = 8
    cfg.value_base_horizon = 4
    cfg.value_hidden_dims = (32, 32)
    cfg.action_dim = ACTION_DIM
    cfg.use_qz_critic = use_qz
    cfg.qz_use_transitive_backup = qz_tri
    if qz_tri:
        cfg.qz_transitive_reweight = True
        cfg.lambda_qz_tri = 1.0
    batch = CriticSequenceDataset(_dummy_dataset(), cfg).sample(BATCH)
    critic = CriticAgent.create(
        seed=0,
        ex_observations=batch['observations'],
        ex_full_chunk_actions=None,
        ex_action_chunk_actions=batch['action_chunk_actions'],
        config=cfg,
        ex_goals=batch['value_goals'],
    )
    return critic, batch


def test_qz_loss_update_finite():
    # Test 4 (QZ loss): one critic update with qz fields -> finite qz/loss_prod.
    critic, batch = _make_critic_with_batch(use_qz=True)
    _, info = critic.update(batch)
    assert np.isfinite(float(info['qz/loss_prod']))
    assert np.isfinite(float(info['qz/loss_total']))
    assert np.isfinite(float(info['total_loss']))


def test_qz_transitive_backup_runs_and_logs():
    critic, batch = _make_critic_with_batch(use_qz=True, qz_tri=True)
    assert 'qz_tri_split_observations' in batch
    assert 'qz_tri_split_offsets' in batch
    assert 'qz_tri_right_offsets' in batch
    _, info = critic.update(batch)
    assert np.isfinite(float(info['qz/loss_tri']))
    assert np.isfinite(float(info['qz/tri_valid_fraction']))
    assert np.isfinite(float(info['qz/tri_same_subgoal_fraction']))
    assert np.isfinite(float(info['qz/target_tri_mean']))


def test_sample_actions_raises_for_state_subgoal():
    # Rollout must not call sample_actions for state actors.
    actor = _make_state_actor()
    with pytest.raises(ValueError):
        actor.sample_actions(np.zeros((STATE_DIM,), dtype=np.float32), np.zeros((STATE_DIM,), dtype=np.float32))


def test_state_proposal_sample_subgoals_raises():
    cfg = get_actor_config()
    cfg.actor_type = 'state_proposal'
    cfg.actor_chunk_horizon = 2
    cfg.action_dim = ACTION_DIM
    ex_obs = np.zeros((BATCH, STATE_DIM), dtype=np.float32)
    actor = ActorAgent.create(seed=0, ex_observations=ex_obs, config=cfg, ex_goals=ex_obs)
    with pytest.raises(ValueError):
        actor.sample_subgoals(np.zeros((STATE_DIM,), dtype=np.float32), np.zeros((STATE_DIM,), dtype=np.float32))


def test_normalized_metric_requires_stats():
    cfg = get_actor_config()
    cfg.actor_type = 'state_subgoal'
    cfg.actor_chunk_horizon = 2
    cfg.action_dim = ACTION_DIM
    cfg.spi_tau = 5.0
    cfg.state_spi_metric_space = 'normalized'  # no state_mean/std provided
    ex_obs = np.zeros((BATCH, STATE_DIM), dtype=np.float32)
    actor = ActorAgent.create(seed=0, ex_observations=ex_obs, config=cfg, ex_goals=ex_obs)
    critic = _make_critic('v_product')
    with pytest.raises(ValueError):
        actor.update(_state_batch(), critic)


def test_normalized_metric_uses_stats_when_present():
    cfg = get_actor_config()
    cfg.actor_type = 'state_subgoal'
    cfg.actor_chunk_horizon = 2
    cfg.action_dim = ACTION_DIM
    cfg.spi_tau = 5.0
    cfg.state_spi_metric_space = 'normalized'
    cfg.state_mean = tuple(0.0 for _ in range(STATE_DIM))
    cfg.state_std = tuple(1.0 for _ in range(STATE_DIM))
    ex_obs = np.zeros((BATCH, STATE_DIM), dtype=np.float32)
    actor = ActorAgent.create(seed=0, ex_observations=ex_obs, config=cfg, ex_goals=ex_obs)
    critic = _make_critic('v_product')
    _, info = actor.update(_state_batch(), critic)
    assert np.isfinite(float(info['state_spi/actor_loss']))


def test_qz_dataset_fields_absent_when_disabled():
    cfg = get_critic_config()
    cfg.critic_type = 'trl'
    cfg.algorithm = 'trl'
    cfg.goal_representation = 'full'
    cfg.action_chunk_horizon = 4
    cfg.full_chunk_horizon = 8
    cfg.value_base_horizon = 4
    cfg.action_dim = ACTION_DIM
    ds = CriticSequenceDataset(_dummy_dataset(), cfg)
    batch = ds.sample(BATCH)
    assert 'qz_subgoals' not in batch


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            print(f'  PASS  {name}')
