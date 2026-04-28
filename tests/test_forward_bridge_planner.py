"""Smoke tests for the linear-SDE dynamics ``forward_bridge`` planner mode.

Covers:
1. ``forward_bridge_coefficients`` returns ``(K+1,)`` arrays with exact
   endpoint constraints (``a[0]=1, b[0]=0, std[0]=0`` and
   ``a[K]=0, b[K]=1, std[K]=0``).
2. ``forward_bridge_plan`` returns ``[B, K+1, D]`` paths with
   ``path[:, 0] == z0`` and ``path[:, -1] == zK``.
3. ``planner_type='reverse_score'`` (default) still runs and produces
   trajectories with the same shape as before this change.
4. ``planner_type='forward_bridge'`` and
   ``planner_type='forward_bridge_residual'`` go through ``plan()`` /
   ``sample_plan()`` without raising and produce endpoint-respecting
   paths.
5. ``DynamicsAgent.update`` runs one step in each planner mode
   without NaNs.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

import jax
import jax.numpy as jnp

from utils.dynamics import forward_bridge_coefficients
from agents.dynamics import DynamicsAgent, get_dynamics_config


STATE_DIM = 4
ACTION_DIM = 2
BATCH = 8


def _make_agent(planner_type: str = 'reverse_score'):
    cfg = get_dynamics_config()
    cfg.dynamics_N = 4
    cfg.subgoal_steps = 4
    cfg.rollout_horizon = 2
    cfg.eps_hidden_dims = (32, 32)
    cfg.subgoal_hidden_dims = (32, 32)
    cfg.subgoal_value_hidden_dims = (32, 32)
    cfg.idm_hidden_dims = (32, 32)
    cfg.residual_hidden_dims = (32, 32)
    cfg.planner_type = planner_type
    ex_obs = np.zeros((BATCH, STATE_DIM), dtype=np.float32)
    ex_act = np.zeros((BATCH, ACTION_DIM), dtype=np.float32)
    return DynamicsAgent.create(seed=0, ex_observations=ex_obs, ex_actions=ex_act, config=cfg)


def _make_batch(K: int):
    rng = np.random.default_rng(0)
    obs = rng.standard_normal((BATCH, STATE_DIM)).astype(np.float32)
    targets = rng.standard_normal((BATCH, STATE_DIM)).astype(np.float32)
    goals = rng.standard_normal((BATCH, STATE_DIM)).astype(np.float32)
    next_obs = rng.standard_normal((BATCH, STATE_DIM)).astype(np.float32)
    actions = rng.standard_normal((BATCH, ACTION_DIM)).astype(np.float32)

    seg = np.zeros((BATCH, K + 1, STATE_DIM), dtype=np.float32)
    seg[:, 0] = obs
    seg[:, -1] = targets
    if K >= 2:
        for i in range(1, K):
            t = i / float(K)
            seg[:, i] = (1.0 - t) * obs + t * targets + 0.05 * rng.standard_normal((BATCH, STATE_DIM)).astype(np.float32)

    return dict(
        observations=obs,
        next_observations=next_obs,
        actions=actions,
        high_actor_goals=goals,
        high_actor_targets=targets,
        trajectory_segment=seg,
    )


def test_forward_bridge_coefficients_endpoints():
    K = 8
    a, b, std = forward_bridge_coefficients(K, beta_min=0.1, beta_max=20.0, lambda_=1.0)
    a = np.asarray(a)
    b = np.asarray(b)
    std = np.asarray(std)
    assert a.shape == (K + 1,)
    assert b.shape == (K + 1,)
    assert std.shape == (K + 1,)

    assert np.allclose(a[0], 1.0, atol=1e-6), f'a[0] = {a[0]} (expected 1.0)'
    assert np.allclose(b[0], 0.0, atol=1e-6), f'b[0] = {b[0]} (expected 0.0)'
    assert np.allclose(std[0], 0.0, atol=1e-6), f'std[0] = {std[0]} (expected 0.0)'
    assert np.allclose(a[-1], 0.0, atol=1e-6), f'a[-1] = {a[-1]} (expected 0.0)'
    assert np.allclose(b[-1], 1.0, atol=1e-6), f'b[-1] = {b[-1]} (expected 1.0)'
    assert np.allclose(std[-1], 0.0, atol=1e-6), f'std[-1] = {std[-1]} (expected 0.0)'

    # Coefficients should obey monotonic transitions: a starts at 1 and
    # decreases to 0, b starts at 0 and increases to 1.
    assert a[0] > a[1] > a[-1]
    assert b[0] < b[1] < b[-1]
    # Bridge std is zero at endpoints and positive in the interior.
    assert np.all(std[1:-1] > 0.0)


def test_forward_bridge_coefficients_use_bridge_gamma_inv():
    K = 8
    a_hard, b_hard, std_hard = forward_bridge_coefficients(
        K,
        beta_min=0.1,
        beta_max=20.0,
        lambda_=1.0,
        bridge_gamma_inv=0.0,
        theta_schedule='prefix_progress',
        theta_total=1.0,
    )
    a_soft, b_soft, std_soft = forward_bridge_coefficients(
        K,
        beta_min=0.1,
        beta_max=20.0,
        lambda_=1.0,
        bridge_gamma_inv=0.5,
        theta_schedule='prefix_progress',
        theta_total=1.0,
    )

    # The finite-gamma bridge should affect the interior marginal coefficients
    # and variance, while explicit endpoint clamps keep planner outputs pinned.
    assert not np.allclose(np.asarray(b_hard[1:-1]), np.asarray(b_soft[1:-1]))
    assert not np.allclose(np.asarray(std_hard[1:-1]), np.asarray(std_soft[1:-1]))
    a_soft_np = np.asarray(a_soft)
    b_soft_np = np.asarray(b_soft)
    std_soft_np = np.asarray(std_soft)
    np.testing.assert_allclose(a_soft_np[[0, -1]], np.asarray([1.0, 0.0]), atol=1e-6)
    np.testing.assert_allclose(b_soft_np[[0, -1]], np.asarray([0.0, 1.0]), atol=1e-6)
    np.testing.assert_allclose(std_soft_np[[0, -1]], np.asarray([0.0, 0.0]), atol=1e-6)


def test_forward_bridge_plan_shapes_and_endpoints():
    agent = _make_agent('forward_bridge')
    K = int(agent.config['dynamics_N'])
    z0 = jnp.asarray(np.random.RandomState(1).randn(BATCH, STATE_DIM).astype(np.float32))
    zK = jnp.asarray(np.random.RandomState(2).randn(BATCH, STATE_DIM).astype(np.float32))

    path = agent.forward_bridge_plan(z0, zK, sample=False, noise_scale=0.0)
    path_np = np.asarray(path)
    assert path_np.shape == (BATCH, K + 1, STATE_DIM)
    np.testing.assert_allclose(path_np[:, 0], np.asarray(z0), atol=1e-5)
    np.testing.assert_allclose(path_np[:, -1], np.asarray(zK), atol=1e-5)

    # Sampled mode must still respect endpoints (clamp after noise).
    path_s = agent.forward_bridge_plan(z0, zK, sample=True, noise_scale=0.5, rng=jax.random.PRNGKey(7))
    path_s_np = np.asarray(path_s)
    assert path_s_np.shape == (BATCH, K + 1, STATE_DIM)
    np.testing.assert_allclose(path_s_np[:, 0], np.asarray(z0), atol=1e-5)
    np.testing.assert_allclose(path_s_np[:, -1], np.asarray(zK), atol=1e-5)


def test_agent_forward_bridge_uses_configured_bridge_gamma_inv():
    cfg_hard = get_dynamics_config()
    cfg_hard.dynamics_N = 8
    cfg_hard.subgoal_steps = 8
    cfg_hard.bridge_gamma_inv = 0.0
    cfg_hard.theta_schedule = 'prefix_progress'
    cfg_hard.theta_total = 1.0
    agent_hard = DynamicsAgent.create(
        seed=0,
        ex_observations=np.zeros((BATCH, STATE_DIM), dtype=np.float32),
        ex_actions=np.zeros((BATCH, ACTION_DIM), dtype=np.float32),
        config=cfg_hard,
    )

    cfg_soft = get_dynamics_config()
    cfg_soft.dynamics_N = 8
    cfg_soft.subgoal_steps = 8
    cfg_soft.bridge_gamma_inv = 0.5
    cfg_soft.theta_schedule = 'prefix_progress'
    cfg_soft.theta_total = 1.0
    agent_soft = DynamicsAgent.create(
        seed=0,
        ex_observations=np.zeros((BATCH, STATE_DIM), dtype=np.float32),
        ex_actions=np.zeros((BATCH, ACTION_DIM), dtype=np.float32),
        config=cfg_soft,
    )

    _, b_hard, std_hard = agent_hard.forward_bridge_coefficients(8)
    _, b_soft, std_soft = agent_soft.forward_bridge_coefficients(8)
    assert not np.allclose(np.asarray(b_hard[1:-1]), np.asarray(b_soft[1:-1]))
    assert not np.allclose(np.asarray(std_hard[1:-1]), np.asarray(std_soft[1:-1]))


def test_reverse_score_plan_unchanged():
    agent = _make_agent('reverse_score')
    K = int(agent.config['dynamics_N'])
    obs = jnp.asarray(np.random.RandomState(3).randn(BATCH, STATE_DIM).astype(np.float32))
    goal = jnp.asarray(np.random.RandomState(4).randn(BATCH, STATE_DIM).astype(np.float32))

    out_det = agent.plan(obs, goal)
    traj = np.asarray(out_det['trajectory'])
    assert traj.shape == (BATCH, K + 1, STATE_DIM)
    np.testing.assert_allclose(traj[:, 0], np.asarray(obs), atol=1e-5)

    out_stoch = agent.sample_plan(obs, goal, jax.random.PRNGKey(11), noise_scale=0.5)
    traj_s = np.asarray(out_stoch['trajectory'])
    assert traj_s.shape == (BATCH, K + 1, STATE_DIM)


def test_forward_bridge_planner_dispatch():
    for planner in ('forward_bridge', 'forward_bridge_residual'):
        agent = _make_agent(planner)
        K = int(agent.config['dynamics_N'])
        obs = jnp.asarray(np.random.RandomState(5).randn(BATCH, STATE_DIM).astype(np.float32))
        goal = jnp.asarray(np.random.RandomState(6).randn(BATCH, STATE_DIM).astype(np.float32))

        out_det = agent.plan(obs, goal)
        traj = np.asarray(out_det['trajectory'])
        assert traj.shape == (BATCH, K + 1, STATE_DIM)
        np.testing.assert_allclose(traj[:, 0], np.asarray(obs), atol=1e-5)
        np.testing.assert_allclose(traj[:, -1], np.asarray(goal), atol=1e-5)

        next_step_np = np.asarray(out_det['next_step'])
        assert next_step_np.shape == (BATCH, STATE_DIM)
        np.testing.assert_allclose(next_step_np, traj[:, 1], atol=1e-5)


def test_forward_bridge_total_loss_finite():
    for planner in ('reverse_score', 'forward_bridge', 'forward_bridge_residual'):
        agent = _make_agent(planner)
        batch = _make_batch(int(agent.config['dynamics_N']))
        agent2, info = agent.update(batch)
        loss_val = float(info['phase1/loss'])
        assert np.isfinite(loss_val), f'planner={planner} produced non-finite loss {loss_val}'
        # Endpoint MSE should be exactly 0 for forward-bridge (clamped).
        if planner in ('forward_bridge', 'forward_bridge_residual'):
            assert float(info['forward_bridge/endpoint_start_mse']) < 1e-10
            assert float(info['forward_bridge/endpoint_end_mse']) < 1e-10


if __name__ == '__main__':
    test_forward_bridge_coefficients_endpoints()
    test_forward_bridge_coefficients_use_bridge_gamma_inv()
    test_forward_bridge_plan_shapes_and_endpoints()
    test_agent_forward_bridge_uses_configured_bridge_gamma_inv()
    test_reverse_score_plan_unchanged()
    test_forward_bridge_planner_dispatch()
    test_forward_bridge_total_loss_finite()
    print('OK: all forward_bridge planner smoke tests passed.')
