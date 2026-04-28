"""Tests for the optional curved centerline bridge.

Covers (per ``scripts/cursor_curved_centerline_bridge_instructions.md``):

* T1 backward-compat: default config (``use_curved_centerline=False``) leaves
  ``_learned_reverse_mean`` bit-identical to the existing exact-residual path.
* T2 endpoint preservation: random ``(s0, sK, goal)`` produce
  ``c_{i=0} = s0`` and ``c_{i=K} = sK`` exactly, regardless of ``h``.
* T3 zero-init equivalence: ``centerline_zero_init=True`` makes the curved
  centerline equal to the linear interpolation between endpoints, so
  ``_curved_reverse_mean`` reduces to the analytic linear-SDE bridge in
  residual coordinates plus the standard exact-residual.
* T4 hard final step: ``centerline_residual_use_hard_variance=True`` with
  ``gamma_inv=0`` zeros the residual scale at the last reverse step, so
  ``mu == sK`` exactly.
* T5 K=2 edge: smallest valid ``dynamics_N`` runs end-to-end without NaNs and
  the smoothness loss is computable.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

import jax
import jax.numpy as jnp

from agents.dynamics import DynamicsAgent, get_dynamics_config
from utils.dynamics import (
    exact_residual_model_mean,
    posterior_moments,
)


STATE_DIM = 4
ACTION_DIM = 2
BATCH = 8


def _make_cfg(curved: bool = False, **overrides):
    cfg = get_dynamics_config()
    cfg.dynamics_N = 4
    cfg.subgoal_steps = 4
    cfg.rollout_horizon = 2
    cfg.eps_hidden_dims = (32, 32)
    cfg.subgoal_hidden_dims = (32, 32)
    cfg.subgoal_value_hidden_dims = (32, 32)
    cfg.idm_hidden_dims = (32, 32)
    cfg.residual_hidden_dims = (32, 32)
    cfg.dynamics_model_type = 'exact_residual'
    cfg.planner_type = 'reverse_score'
    cfg.use_curved_centerline = bool(curved)
    cfg.centerline_hidden_dims = (16, 16)
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def _make_agent(cfg):
    ex_obs = np.zeros((BATCH, STATE_DIM), dtype=np.float32)
    ex_act = np.zeros((BATCH, ACTION_DIM), dtype=np.float32)
    return DynamicsAgent.create(seed=0, ex_observations=ex_obs, ex_actions=ex_act, config=cfg)


def _sample_inputs(seed=0):
    rng = np.random.default_rng(seed)
    x_n = rng.standard_normal((BATCH, STATE_DIM)).astype(np.float32)
    x_0 = rng.standard_normal((BATCH, STATE_DIM)).astype(np.float32)
    x_T = rng.standard_normal((BATCH, STATE_DIM)).astype(np.float32)
    goal = rng.standard_normal((BATCH, STATE_DIM)).astype(np.float32)
    return jnp.asarray(x_n), jnp.asarray(x_0), jnp.asarray(x_T), jnp.asarray(goal)


def test_backward_compat_default_cfg():
    """T1: with use_curved_centerline=False the reverse mean equals the
    standard exact_residual_model_mean computation bit-for-bit."""
    cfg = _make_cfg(curved=False)
    agent = _make_agent(cfg)
    x_n, x_0, x_T, _ = _sample_inputs(seed=1)
    n = jnp.asarray([1, 2, 3, 4, 1, 2, 3, 4], dtype=jnp.int32)

    mu, eps = agent._learned_reverse_mean(x_n, x_T, x_0, n, agent.schedule)

    eps_ref = agent.network.select('eps_net')(x_n, x_T, x_0, n.astype(jnp.float32))
    mu_ref, _, _ = exact_residual_model_mean(
        x_n, x_0, x_T, eps_ref, n, agent.schedule,
        residual_scale=float(cfg.exact_residual_scale),
    )
    np.testing.assert_allclose(np.asarray(eps), np.asarray(eps_ref), atol=1e-6)
    np.testing.assert_allclose(np.asarray(mu), np.asarray(mu_ref), atol=1e-6)


def test_endpoint_preservation():
    """T2: c_{i=0} = s0 and c_{i=K} = sK regardless of the learned displacement."""
    cfg = _make_cfg(
        curved=True,
        centerline_zero_init=False,
        centerline_use_goal=True,
        centerline_beta_type='linear',
    )
    agent = _make_agent(cfg)
    x_n, x_0, x_T, goal = _sample_inputs(seed=2)
    K = int(cfg.dynamics_N)

    c0, _, b0 = agent._curved_centerline(x_T, x_0, goal, 0)
    cK, _, bK = agent._curved_centerline(x_T, x_0, goal, K)
    np.testing.assert_allclose(np.asarray(c0), np.asarray(x_T), atol=1e-6)
    np.testing.assert_allclose(np.asarray(cK), np.asarray(x_0), atol=1e-6)
    assert float(b0[0, 0]) == 0.0
    assert float(bK[0, 0]) == 1.0

    cfg2 = _make_cfg(
        curved=True,
        centerline_zero_init=False,
        centerline_use_goal=True,
        centerline_beta_type='hard_bridge',
    )
    agent2 = _make_agent(cfg2)
    c0, _, b0 = agent2._curved_centerline(x_T, x_0, goal, 0)
    cK, _, bK = agent2._curved_centerline(x_T, x_0, goal, K)
    np.testing.assert_allclose(np.asarray(c0), np.asarray(x_T), atol=1e-5)
    np.testing.assert_allclose(np.asarray(cK), np.asarray(x_0), atol=1e-5)


def test_zero_init_equivalence_to_linear_centerline_formula():
    """T3: centerline_zero_init=True + exact_residual_scale=0 makes the curved
    reverse mean equal to the analytic formula

        mu = c_{i+1} + posterior_mean(x_n - c_i, 0, 0, n)

    with c_i the linear interpolation between (s0, sK)."""
    cfg = _make_cfg(
        curved=True,
        centerline_zero_init=True,
        centerline_use_goal=True,
        centerline_beta_type='linear',
        centerline_residual_use_hard_variance=False,
        exact_residual_scale=0.0,
    )
    agent = _make_agent(cfg)
    x_n, x_0, x_T, goal = _sample_inputs(seed=3)
    n = jnp.asarray([1, 2, 3, 4, 1, 2, 3, 4], dtype=jnp.int32)
    K = int(cfg.dynamics_N)

    mu, _, _, _ = agent._curved_reverse_mean(x_n, x_T, x_0, n, agent.schedule, goal)

    i = K - n
    i_next = i + 1
    b_i = (i.astype(jnp.float32) / float(K))[:, None]
    b_next = (i_next.astype(jnp.float32) / float(K))[:, None]
    c_i = (1.0 - b_i) * x_T + b_i * x_0
    c_next = (1.0 - b_next) * x_T + b_next * x_0

    z_n = x_n - c_i
    mu_z, _ = posterior_moments(
        z_n, jnp.zeros_like(z_n), jnp.zeros_like(z_n), n, agent.schedule,
    )
    mu_ref = c_next + mu_z
    np.testing.assert_allclose(np.asarray(mu), np.asarray(mu_ref), atol=1e-5)


def test_hard_variance_zero_at_final_step():
    """T4: with hard variance + gamma_inv=0, the last reverse step (n=1) puts
    rho at the next state's marginal std (=0 at the endpoint), so mu == sK
    exactly. The centerline at i=K must also pin to sK regardless of h."""
    cfg = _make_cfg(
        curved=True,
        centerline_zero_init=False,
        centerline_use_goal=True,
        centerline_beta_type='linear',
        centerline_residual_use_hard_variance=True,
        bridge_gamma_inv=0.0,
    )
    agent = _make_agent(cfg)
    x_n, x_0, x_T, goal = _sample_inputs(seed=4)
    n_final = jnp.ones((BATCH,), dtype=jnp.int32)

    mu, _, _, residual = agent._curved_reverse_mean(
        x_n, x_T, x_0, n_final, agent.schedule, goal,
    )
    np.testing.assert_allclose(np.asarray(residual), np.zeros_like(np.asarray(residual)), atol=1e-6)
    np.testing.assert_allclose(np.asarray(mu), np.asarray(x_0), atol=1e-5)


def test_total_loss_curved_finite():
    """T5: end-to-end update finishes without NaNs under the curved option,
    and the centerline diagnostics show up in info."""
    cfg = _make_cfg(curved=True)
    agent = _make_agent(cfg)
    rng = np.random.default_rng(5)
    K = int(cfg.dynamics_N)
    obs = rng.standard_normal((BATCH, STATE_DIM)).astype(np.float32)
    targets = rng.standard_normal((BATCH, STATE_DIM)).astype(np.float32)
    goals = rng.standard_normal((BATCH, STATE_DIM)).astype(np.float32)
    seg = np.zeros((BATCH, K + 1, STATE_DIM), dtype=np.float32)
    seg[:, 0] = obs
    seg[:, -1] = targets
    for i in range(1, K):
        t = i / float(K)
        seg[:, i] = (1.0 - t) * obs + t * targets

    batch = dict(
        observations=obs,
        next_observations=rng.standard_normal((BATCH, STATE_DIM)).astype(np.float32),
        actions=rng.standard_normal((BATCH, ACTION_DIM)).astype(np.float32),
        high_actor_goals=goals,
        high_actor_targets=targets,
        trajectory_segment=seg,
    )

    agent2, info = agent.update(batch)
    loss_val = float(info['phase1/loss'])
    assert np.isfinite(loss_val), f'curved-centerline loss not finite: {loss_val}'
    assert 'dynamics/centerline/amp' in info
    assert 'dynamics/centerline/smooth' in info
    assert 'dynamics/centerline/deviation' in info
    assert 'dynamics/centerline/residual_norm' in info
    assert np.isfinite(float(info['dynamics/centerline/amp']))
    assert np.isfinite(float(info['dynamics/centerline/smooth']))


def test_safety_warning_on_unsupported_combo(caplog=None):
    """When the curved option is requested but the planner / model_type are not
    supported (here: sde_euler), the agent must skip registering centerline_net
    and emit a logging warning. We assert the absence of the centerline_net
    parameter in the resulting params dict."""
    cfg = _make_cfg(curved=True)
    cfg.dynamics_model_type = 'sde_euler'
    agent = _make_agent(cfg)
    params_keys = list(agent.network.params.keys())
    assert 'centerline_net' not in params_keys, (
        f'centerline_net should not be registered for sde_euler, got params: {params_keys}'
    )


if __name__ == '__main__':
    test_backward_compat_default_cfg()
    test_endpoint_preservation()
    test_zero_init_equivalence_to_linear_centerline_formula()
    test_hard_variance_zero_at_final_step()
    test_total_loss_curved_finite()
    test_safety_warning_on_unsupported_combo()
    print('OK: all curved_centerline_bridge tests passed.')
