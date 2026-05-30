"""Unit tests for the distributional-subgoal + linear dynamics refactor.

These tests are intentionally lightweight (no offline dataset, no actor / critic
training) so they can be run standalone with::

    PYTHONPATH=. python -m pytest tests/test_distributional_subgoal.py
    PYTHONPATH=. python tests/test_distributional_subgoal.py    # also works

They cover the contract changes only:
1. deterministic subgoal mode
2. linear dynamics schedule exposes gamma_inv and bridge arrays
3. distributional-subgoal sampling shape correctness
4. critic ``score_action_chunks`` accepts both ``[B, D]`` and ``[B, N, D]`` goals
5. ``plan_candidates=1`` and ``plan_candidates>1`` both succeed
6. all U*N bridge candidates are reduced to one best proposal before SPI
7. distributional subgoal loss path is finite (no NaNs / Infs)
8. dynamics-config defaults remain usable
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest

import jax
import jax.numpy as jnp

from utils.dynamics import bridge_sample, make_dynamics_schedule, posterior_mean
from agents.dynamics import (
    DynamicsAgent,
    get_dynamics_config,
)
from agents.critic import CriticAgent, get_config as get_critic_config
from main import _rescore_actor_batch_for_update


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

STATE_DIM = 4
ACTION_DIM = 2
BATCH = 4
ENV_NAME = 'antmaze-medium-navigate-v0'
_VALUE_SHARED = {
    'env_name': ENV_NAME,
    'subgoal_value_hidden_dims': (32, 32),
    'subgoal_value_goal_representation': 'phi',
}


def _make_critic(critic_type: str = 'dqc'):
    cfg = get_critic_config()
    cfg.action_chunk_horizon = 2
    cfg.full_chunk_horizon = 4
    cfg.value_hidden_dims = (32, 32)
    cfg.action_dim = ACTION_DIM
    cfg.env_name = ENV_NAME
    cfg.critic_type = critic_type
    if critic_type == 'trl':
        cfg.algorithm = 'trl'
        cfg.use_chunk_critic = False
    ex_obs = np.zeros((BATCH, STATE_DIM), dtype=np.float32)
    ex_part = np.zeros((BATCH, cfg.action_chunk_horizon * ACTION_DIM), dtype=np.float32)
    ex_full = None if critic_type != 'dqc' else np.zeros((BATCH, cfg.full_chunk_horizon * ACTION_DIM), np.float32)
    return CriticAgent.create(
        seed=0,
        ex_observations=ex_obs,
        ex_full_chunk_actions=ex_full,
        ex_action_chunk_actions=ex_part,
        config=cfg,
        ex_goals=ex_obs,
    )


def _make_dynamics_agent(
    subgoal_distribution: str,
    bridge_gamma_inv: float = 0.0,
    subgoal_num_samples: int = 1,
    config_updates: dict | None = None,
):
    cfg = get_dynamics_config()
    cfg.dynamics_N = 4
    cfg.subgoal_steps = 4
    cfg.rollout_horizon = 2
    cfg.subgoal_distribution = subgoal_distribution
    cfg.subgoal_num_samples = subgoal_num_samples
    cfg.bridge_gamma_inv = bridge_gamma_inv
    cfg.residual_model_hidden_dims = (32, 32)
    cfg.subgoal_hidden_dims = (32, 32)
    cfg.subgoal_value_hidden_dims = (32, 32)
    cfg.idm_hidden_dims = (32, 32)
    if config_updates:
        for key, value in config_updates.items():
            cfg[key] = value
    ex_obs = np.zeros((BATCH, STATE_DIM), dtype=np.float32)
    ex_act = np.zeros((BATCH, ACTION_DIM), dtype=np.float32)
    return DynamicsAgent.create(seed=0, ex_observations=ex_obs, ex_actions=ex_act, config=cfg)


# ---------------------------------------------------------------------------
# 1. deterministic mode
# ---------------------------------------------------------------------------

def test_deterministic_subgoal_api():
    agent = _make_dynamics_agent('deterministic')
    obs = jnp.zeros((BATCH, STATE_DIM), dtype=jnp.float32)
    g = jnp.zeros((BATCH, STATE_DIM), dtype=jnp.float32)
    sg = agent.predict_subgoal(obs, g)
    assert sg.shape == (BATCH, STATE_DIM), sg.shape
    # infer_subgoal must remain a backward-compatible alias.
    sg2 = agent.infer_subgoal(obs, g)
    assert jnp.allclose(sg, sg2)
    mu, log_std = agent.infer_subgoal_distribution(obs, g)
    # In deterministic mode the distribution helper must still return a mean;
    # log_std is filled with the configured floor.
    assert mu.shape == (BATCH, STATE_DIM)
    assert log_std.shape == (BATCH, STATE_DIM)
    assert jnp.allclose(mu, sg)


# ---------------------------------------------------------------------------
# 2. linear dynamics schedule
# ---------------------------------------------------------------------------

def test_linear_dynamics_schedule():
    schedule = make_dynamics_schedule(N=8, beta_min=0.1, beta_max=20.0, lambda_=1.0, bridge_gamma_inv=0.0)
    assert schedule['bridge_w'].shape == (9,)
    assert schedule['bridge_var'].shape == (9,)
    assert 'dynamics_phi_iK' in schedule
    assert float(schedule['gamma_inv']) == 0.0

    s_soft = make_dynamics_schedule(N=8, bridge_gamma_inv=0.5)
    assert abs(float(s_soft['gamma_inv']) - 0.5) < 1e-6

    with pytest.raises(ValueError):
        make_dynamics_schedule(N=4, bridge_gamma_inv=-1.0)

    rng = jax.random.PRNGKey(0)
    x0 = jax.random.normal(rng, (BATCH, STATE_DIM))
    xT = jax.random.normal(jax.random.fold_in(rng, 1), (BATCH, STATE_DIM))
    n = jnp.full((BATCH,), 3, dtype=jnp.int32)
    x_n = bridge_sample(x0, xT, n, schedule, rng)
    mu = posterior_mean(x_n, x0, xT, n, schedule)
    assert np.all(np.isfinite(np.asarray(x_n)))
    assert np.all(np.isfinite(np.asarray(mu)))


# ---------------------------------------------------------------------------
# 3. distributional subgoal sampling shape correctness
# ---------------------------------------------------------------------------

def test_distributional_subgoal_sampling_shapes():
    agent = _make_dynamics_agent('diag_gaussian')
    obs = jnp.zeros((BATCH, STATE_DIM), dtype=jnp.float32)
    g = jnp.zeros((BATCH, STATE_DIM), dtype=jnp.float32)
    mu, log_std = agent.infer_subgoal_distribution(obs, g)
    assert mu.shape == (BATCH, STATE_DIM)
    assert log_std.shape == (BATCH, STATE_DIM)
    # sample_subgoal_candidates returns ([B, N, D], [B, D])
    cand, mu2 = agent.sample_subgoal_candidates(
        obs, g, jax.random.PRNGKey(0), num_candidates=5, include_mean=True,
    )
    assert cand.shape == (BATCH, 5, STATE_DIM)
    assert mu2.shape == (BATCH, STATE_DIM)
    # In diag_gaussian mode candidate 0 is pinned to the mean when include_mean=True.
    np.testing.assert_allclose(np.asarray(cand[:, 0, :]), np.asarray(mu2), rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# 4. critic accepts per-candidate goals [B, N, D]
# ---------------------------------------------------------------------------

def test_critic_score_action_chunks_with_per_candidate_goals():
    critic = _make_critic()
    n_cand = 4
    ha = int(critic.config['action_chunk_horizon'])
    obs = jnp.zeros((BATCH, STATE_DIM), dtype=jnp.float32)
    # Shape required by score_action_chunks for candidates: [B, N, ha, A].
    chunks = jnp.zeros((BATCH, n_cand, ha, ACTION_DIM), dtype=jnp.float32)

    shared_goals = jnp.zeros((BATCH, STATE_DIM), dtype=jnp.float32)
    per_cand_goals = jnp.zeros((BATCH, n_cand, STATE_DIM), dtype=jnp.float32)

    s_shared = critic.score_action_chunks(obs, shared_goals, chunks, use_partial_critic=True)
    s_per = critic.score_action_chunks(obs, per_cand_goals, chunks, use_partial_critic=True)
    assert s_shared.shape == (BATCH, n_cand)
    assert s_per.shape == (BATCH, n_cand)
    # With identical goals they should match numerically.
    np.testing.assert_allclose(np.asarray(s_shared), np.asarray(s_per), rtol=1e-5, atol=1e-6)

    # Mismatched candidate count must raise.
    bad_goals = jnp.zeros((BATCH, n_cand + 1, STATE_DIM), dtype=jnp.float32)
    raised = False
    try:
        critic.score_action_chunks(obs, bad_goals, chunks, use_partial_critic=True)
    except Exception:
        raised = True
    assert raised, 'expected per-candidate goal/chunk mismatch to raise'


# ---------------------------------------------------------------------------
# 5. plan_candidates=1 and >1 both succeed (deterministic + diag_gaussian)
# ---------------------------------------------------------------------------

def _check_build_actor_proposals(agent, plan_candidates: int, expected_candidates: int | None = None):
    obs = jnp.zeros((BATCH, STATE_DIM), dtype=jnp.float32)
    g = jnp.zeros((BATCH, STATE_DIM), dtype=jnp.float32)
    mu, cand_actions, cand_goals, _ = agent.build_actor_proposals(
        obs, g, jax.random.PRNGKey(0),
        proposal_horizon=2, plan_candidates=plan_candidates, sample_noise_scale=0.0,
    )
    if expected_candidates is None:
        expected_candidates = plan_candidates
    assert mu.shape == (BATCH, STATE_DIM)
    assert cand_actions.shape == (BATCH, expected_candidates, 2, ACTION_DIM)
    assert cand_goals.shape == (BATCH, expected_candidates, STATE_DIM)


@pytest.mark.parametrize(
    'distribution,plan_candidates,subgoal_num_samples,expected_candidates',
    [
        ('deterministic', 1, 1, 1),
        ('diag_gaussian', 4, 3, 12),
    ],
)
def test_build_actor_proposals_shapes(distribution, plan_candidates, subgoal_num_samples, expected_candidates):
    agent = _make_dynamics_agent(distribution, subgoal_num_samples=subgoal_num_samples)
    _check_build_actor_proposals(agent, plan_candidates=plan_candidates, expected_candidates=expected_candidates)


def test_rescore_keeps_global_best_proposal_before_spi():
    subgoal_samples = 3
    plan_candidates = 4
    agent = _make_dynamics_agent('diag_gaussian', subgoal_num_samples=subgoal_samples)
    critic = _make_critic()
    obs = jnp.zeros((BATCH, STATE_DIM), dtype=jnp.float32)
    g = jnp.zeros((BATCH, STATE_DIM), dtype=jnp.float32)
    mu, cand_actions, cand_goals, _ = agent.build_actor_proposals(
        obs,
        g,
        jax.random.PRNGKey(0),
        proposal_horizon=2,
        plan_candidates=plan_candidates,
        sample_noise_scale=0.0,
    )
    assert cand_actions.shape[1] == subgoal_samples * plan_candidates
    actor_batch = {
        'observations': obs,
        'spi_goals': mu,
        'candidate_partial_chunks': cand_actions,
        'candidate_goals': cand_goals,
        'candidate_group_size': plan_candidates,
        'valids': jnp.ones((BATCH, 2), dtype=jnp.float32),
    }
    out_batch, stats = _rescore_actor_batch_for_update(actor_batch, critic, actor_config={})

    scores = critic.score_action_chunks(obs, cand_goals, cand_actions, use_partial_critic=True)
    best_idx = jnp.argmax(scores, axis=1)
    expected_goals = jnp.take_along_axis(cand_goals, best_idx[:, None, None], axis=1)[:, 0, :]

    assert out_batch['proposal_partial_chunks'].shape == (BATCH, 1, 2, ACTION_DIM)
    assert out_batch['proposal_scores'].shape == (BATCH, 1)
    np.testing.assert_allclose(np.asarray(out_batch['spi_goals']), np.asarray(expected_goals), rtol=1e-5, atol=1e-6)
    assert float(stats['proposal_best_of_n']) == float(subgoal_samples * plan_candidates)
    assert float(stats['proposal_pre_best_count']) == float(subgoal_samples * plan_candidates)
    assert float(stats['proposal_post_best_count']) == 1.0


# ---------------------------------------------------------------------------
# 7. distributional subgoal loss has no NaN / Inf
# ---------------------------------------------------------------------------

def _make_phase1_batch():
    rng = np.random.default_rng(0)
    obs = rng.standard_normal((BATCH, STATE_DIM)).astype(np.float32)
    target = rng.standard_normal((BATCH, STATE_DIM)).astype(np.float32)
    # 'trajectory_segment' must have N+1 = 5 states.
    seg = rng.standard_normal((BATCH, 5, STATE_DIM)).astype(np.float32)
    actions = rng.standard_normal((BATCH, ACTION_DIM)).astype(np.float32)
    next_obs = rng.standard_normal((BATCH, STATE_DIM)).astype(np.float32)
    return {
        'observations': jnp.asarray(obs),
        'next_observations': jnp.asarray(next_obs),
        'high_actor_goals': jnp.asarray(obs),
        'high_actor_targets': jnp.asarray(target),
        'trajectory_segment': jnp.asarray(seg),
        'actions': jnp.asarray(actions),
    }


def _assert_phase1_finite(info):
    for k, v in info.items():
        assert np.all(np.isfinite(np.asarray(v))), f'non-finite log value at {k}: {v}'


@pytest.mark.parametrize(
    'distribution,extra_updates,expected_mode,expected_stochastic_mode',
    [
        ('deterministic', {}, 0.0, 0.0),
        ('diag_gaussian', {}, 1.0, 0.0),
        ('diag_gaussian', {'subgoal_stochastic_loss': 'nll'}, 1.0, 1.0),
    ],
)
def test_phase1_subgoal_loss_finite(distribution, extra_updates, expected_mode, expected_stochastic_mode):
    agent = _make_dynamics_agent(distribution, config_updates=extra_updates)
    _, info = agent.update(_make_phase1_batch(), critic_value_params=None)
    _assert_phase1_finite(info)
    assert float(info['phase1/subgoal_mode']) == expected_mode
    assert float(info['phase1/subgoal_stochastic_loss_mode']) == expected_stochastic_mode


def test_subgoal_expectile_value_style_weights_by_gap_sign():
    agent = _make_dynamics_agent(
        'deterministic',
        config_updates={
            'subgoal_value_style': 'expectile',
            'subgoal_value_alpha': 0.1,
            'subgoal_value_expectile': 0.3,
        },
    )
    gap = jnp.asarray([0.2, -0.1, 0.0], dtype=jnp.float32)
    weight = agent._subgoal_mse_weight_from_gap(gap)
    np.testing.assert_allclose(np.asarray(weight), np.asarray([0.3, 0.7, 0.7]), rtol=1e-6)


# ---------------------------------------------------------------------------
# 8. dynamics-config defaults are usable
# ---------------------------------------------------------------------------

def test_dynamics_config_defaults_are_usable():
    cfg = get_dynamics_config()
    assert str(cfg.subgoal_distribution) == 'deterministic'
    assert str(cfg.subgoal_stochastic_loss) == 'mse'
    assert bool(cfg.subgoal_use_mean_for_actor_goal) is True
    assert int(cfg.subgoal_num_samples) == 1
    assert str(cfg.subgoal_value_style) == 'exponential'
    assert float(cfg.subgoal_value_expectile) == 0.7
    assert float(cfg.subgoal_value_gap_scale) == 1.0


def test_invalid_subgoal_stochastic_loss_rejected():
    raised = False
    try:
        _make_dynamics_agent('diag_gaussian', config_updates={'subgoal_stochastic_loss': 'bad'})
    except ValueError:
        raised = True
    assert raised


# ---------------------------------------------------------------------------
# 9. subgoal_target_mode='displacement'
# ---------------------------------------------------------------------------

def _displacement_agent(subgoal_distribution='deterministic', **updates):
    return _make_dynamics_agent(
        subgoal_distribution,
        config_updates={'subgoal_target_mode': 'displacement', **updates},
    )


def test_displacement_mode():
    agent = _displacement_agent('deterministic')
    rng = np.random.default_rng(0)
    obs = jnp.asarray(rng.standard_normal((BATCH, STATE_DIM)).astype(np.float32))
    g = jnp.asarray(rng.standard_normal((BATCH, STATE_DIM)).astype(np.float32))
    endpoint = obs + jnp.asarray(rng.standard_normal((BATCH, STATE_DIM)).astype(np.float32))

    sg = np.asarray(agent.predict_subgoal(obs, g))
    raw = np.asarray(agent._subgoal_forward(obs, g))
    np.testing.assert_allclose(sg, np.asarray(obs) + raw, rtol=1e-5, atol=1e-6)

    traj = np.asarray(agent.plan(obs, endpoint)['trajectory'])
    np.testing.assert_allclose(traj[:, 0, :], np.asarray(obs), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(traj[:, -1, :], np.asarray(endpoint), rtol=1e-5, atol=1e-6)

    mu, _, cand_goals, _ = agent.build_actor_proposals(
        obs, g, jax.random.PRNGKey(0), proposal_horizon=2, plan_candidates=2, sample_noise_scale=0.0,
    )
    np.testing.assert_allclose(np.asarray(mu) - np.asarray(obs), raw, rtol=1e-5, atol=1e-6)
    assert cand_goals.shape[-1] == STATE_DIM

    _, info = agent.update(_make_phase1_batch(), critic_value_params=None)
    _assert_phase1_finite(info)
    assert float(info['dynamics/subgoal_target_mode']) == 1.0

    zK = jnp.asarray(rng.standard_normal((BATCH, STATE_DIM)).astype(np.float32))
    t_norm = jnp.full((BATCH, 1), 0.5, dtype=jnp.float32)
    s1_a = jnp.asarray(rng.standard_normal((BATCH, STATE_DIM)).astype(np.float32))
    s1_b = jnp.asarray(rng.standard_normal((BATCH, STATE_DIM)).astype(np.float32))
    eps_a = np.asarray(agent.network.select('path_residual_net')(s1_a, zK, t_norm))
    eps_b = np.asarray(agent.network.select('path_residual_net')(s1_b, zK, t_norm))
    assert not np.allclose(eps_a, eps_b, atol=1e-6)
    np.testing.assert_allclose(np.asarray(agent._bridge_anchor(obs)), np.asarray(obs), rtol=1e-6, atol=1e-6)


def _subgoal_value_terms(agent, value_params, rng_seed=17):
    rng = np.random.default_rng(rng_seed)
    s = jnp.asarray(rng.standard_normal((BATCH, STATE_DIM)).astype(np.float32))
    sg = jnp.asarray(rng.standard_normal((BATCH, STATE_DIM)).astype(np.float32))
    target = jnp.asarray(rng.standard_normal((BATCH, STATE_DIM)).astype(np.float32))
    g = jnp.asarray(rng.standard_normal((BATCH, STATE_DIM)).astype(np.float32))
    return agent._subgoal_value_terms(s, sg, target, g, value_params)


def test_subgoal_value_bonus_by_critic_mode():
    dqc = _make_critic('dqc')
    dqc_agent = _make_dynamics_agent(
        'diag_gaussian',
        config_updates={**_VALUE_SHARED, 'critic_type': 'dqc', 'subgoal_value_alpha': 0.5},
    )
    pred, _, _, bonus, _, _, _, v_s_sg, _ = _subgoal_value_terms(
        dqc_agent, dqc.network.params['modules_value'], rng_seed=19,
    )
    np.testing.assert_allclose(np.asarray(bonus), 0.5 * np.asarray(pred), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(np.asarray(v_s_sg), 0.0, atol=1e-6)

    trl_critic = _make_critic('trl')
    trl_agent = _make_dynamics_agent(
        'diag_gaussian',
        config_updates={
            **_VALUE_SHARED,
            'critic_type': 'trl',
            'algorithm': 'trl',
            'subgoal_value_alpha': 1.0,
            'subgoal_value_bonus_type': 'transitive_ratio',
            'subgoal_value_ratio_eps': 1e-3,
            'subgoal_value_ratio_clip': 5.0,
        },
    )
    pred, obs_value, _, bonus, _, _, _, v_s_sg, v_sg_g = _subgoal_value_terms(
        trl_agent, trl_critic.network.params['modules_value'], rng_seed=23,
    )
    expected_ratio = np.clip(np.asarray(v_s_sg * v_sg_g / (obs_value + 1e-3)), 0.0, 5.0)
    np.testing.assert_allclose(np.asarray(bonus), np.asarray(expected_ratio), rtol=1e-5, atol=1e-6)


if __name__ == '__main__':
    failures = []
    for name, fn in list(globals().items()):
        if name.startswith('test_') and callable(fn):
            try:
                fn()
                print(f'  PASS  {name}')
            except Exception as e:  # pragma: no cover
                failures.append((name, e))
                print(f'  FAIL  {name}: {type(e).__name__}: {e}')
    if failures:
        sys.exit(1)
    print('\nAll tests passed.')
