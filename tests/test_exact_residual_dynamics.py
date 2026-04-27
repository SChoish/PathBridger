"""Sanity tests for the ``exact_residual`` dynamics mode.

Covers:
1. ``posterior_mean`` is bit-equivalent to the mean returned by
   ``posterior_moments`` (numerical refactor preserves output).
2. ``posterior_moments`` returns near-zero variance at the terminal endpoint
   ``n=1`` (where ``bridge_var[0]=0``).
3. ``exact_residual_model_mean`` with ``eps_pred=0`` collapses to
   ``posterior_mean``.
4. ``_dynamics_model_type`` validates the config string and rejects unknowns.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import jax
import jax.numpy as jnp

try:
    import pytest  # noqa: F401
    _HAS_PYTEST = True
except ImportError:  # pragma: no cover - allows running as a plain script.
    _HAS_PYTEST = False

from utils.dynamics import (
    exact_residual_model_mean,
    make_dynamics_schedule,
    posterior_mean,
    posterior_moments,
)


def _sample_inputs(B=4, D=3, key_seed=0):
    key = jax.random.PRNGKey(key_seed)
    k1, k2, k3 = jax.random.split(key, 3)
    x_n = jax.random.normal(k1, (B, D))
    x_0 = jax.random.normal(k2, (B, D))
    x_T = jax.random.normal(k3, (B, D))
    return x_n, x_0, x_T


def test_posterior_mean_matches_moments_mean():
    B, D, N = 4, 3, 10
    x_n, x_0, x_T = _sample_inputs(B, D, 0)
    n = jnp.asarray([1, 2, 5, 10], dtype=jnp.int32)
    schedule = make_dynamics_schedule(N=N)

    m0 = posterior_mean(x_n, x_0, x_T, n, schedule)
    m1, v1 = posterior_moments(x_n, x_0, x_T, n, schedule)

    assert m0.shape == (B, D)
    assert v1.shape == (B, 1)
    assert jnp.max(jnp.abs(m0 - m1)) < 1e-6
    assert jnp.min(v1) >= 0.0


def test_posterior_var_zero_at_terminal_step():
    """At n=1, bridge_var[0]=0 in hard-endpoint mode -> posterior var = 0."""
    B, D, N = 4, 3, 10
    x_n, x_0, x_T = _sample_inputs(B, D, 1)
    n = jnp.ones((B,), dtype=jnp.int32)
    schedule = make_dynamics_schedule(N=N)

    _, var = posterior_moments(x_n, x_0, x_T, n, schedule)
    assert jnp.max(var) < 1e-6


def test_exact_residual_zero_eps_equals_posterior_mean():
    B, D, N = 4, 3, 10
    x_n, x_0, x_T = _sample_inputs(B, D, 2)
    eps = jnp.zeros_like(x_n)
    n = jnp.asarray([1, 3, 7, 10], dtype=jnp.int32)
    schedule = make_dynamics_schedule(N=N)

    mu, base, var = exact_residual_model_mean(x_n, x_0, x_T, eps, n, schedule)
    post = posterior_mean(x_n, x_0, x_T, n, schedule)

    assert jnp.max(jnp.abs(base - post)) < 1e-6
    assert jnp.max(jnp.abs(mu - post)) < 1e-6
    assert jnp.min(var) >= 0.0


def test_exact_residual_uses_post_var_as_scale():
    """At n=1 (post_var=0) the residual must vanish even with non-zero eps."""
    B, D, N = 4, 3, 10
    x_n, x_0, x_T = _sample_inputs(B, D, 3)
    eps = jnp.ones_like(x_n)
    n = jnp.ones((B,), dtype=jnp.int32)
    schedule = make_dynamics_schedule(N=N)

    mu, base, var = exact_residual_model_mean(
        x_n, x_0, x_T, eps, n, schedule, residual_scale=10.0,
    )
    assert jnp.max(jnp.abs(mu - base)) < 1e-6
    assert jnp.max(var) < 1e-6


def test_dynamics_model_type_validation():
    """The config helper must accept the two valid modes and reject unknowns."""
    from agents.dynamics import _dynamics_model_type

    assert _dynamics_model_type({}) == 'sde_euler'
    assert _dynamics_model_type({'dynamics_model_type': 'sde_euler'}) == 'sde_euler'
    assert _dynamics_model_type({'dynamics_model_type': 'EXACT_RESIDUAL'}) == 'exact_residual'
    raised = False
    try:
        _dynamics_model_type({'dynamics_model_type': 'no_such_mode'})
    except ValueError:
        raised = True
    assert raised, 'expected ValueError for unknown dynamics_model_type'


if __name__ == '__main__':
    test_posterior_mean_matches_moments_mean()
    test_posterior_var_zero_at_terminal_step()
    test_exact_residual_zero_eps_equals_posterior_mean()
    test_exact_residual_uses_post_var_as_scale()
    test_dynamics_model_type_validation()
    print('exact_residual sanity tests OK.')
