"""Neural modules for PathBridger value, endpoints, bridges, and inverse dynamics."""

from __future__ import annotations

from typing import Sequence

import flax.linen as nn
import jax.numpy as jnp

from utils.goal_representation import goal_representation
from utils.networks import MLP

_HIDDEN_DIMS = (512, 512, 512)
_LAYER_NORM = True
_GAUSSIAN_LOG_STD_MIN = -5.0
_GAUSSIAN_LOG_STD_MAX = 1.0


class ScalarTransitiveValue(nn.Module):
    """A bounded state-goal value, returned as a logit."""

    hidden_dims: Sequence[int] = _HIDDEN_DIMS

    @nn.compact
    def __call__(self, observations: jnp.ndarray, goals: jnp.ndarray) -> jnp.ndarray:
        goal_inputs = goal_representation(goals, 'full')
        inputs = jnp.concatenate([observations, goal_inputs], axis=-1)
        return MLP(
            (*self.hidden_dims, 1),
            activate_final=False,
            layer_norm=_LAYER_NORM,
        )(inputs).squeeze(-1)


class GaussianEndpointProposer(nn.Module):
    """Diagonal-Gaussian model of the K-step endpoint displacement."""

    state_dim: int
    env_name: str
    phi_goal_obs_indices: tuple[int, ...]
    hidden_dims: Sequence[int] = _HIDDEN_DIMS

    @nn.compact
    def __call__(
        self,
        observations: jnp.ndarray,
        goals: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        goal_inputs = goal_representation(
            goals,
            'phi',
            self.phi_goal_obs_indices,
            env_name=self.env_name,
        )
        inputs = jnp.concatenate([observations, goal_inputs], axis=-1)
        hidden = MLP(
            tuple(self.hidden_dims),
            activate_final=True,
            layer_norm=_LAYER_NORM,
        )(inputs)
        mean = nn.Dense(self.state_dim, name='mean')(hidden)
        log_std = nn.Dense(self.state_dim, name='log_std')(hidden)
        log_std = jnp.clip(
            log_std,
            _GAUSSIAN_LOG_STD_MIN,
            _GAUSSIAN_LOG_STD_MAX,
        )
        return mean, log_std


class FlowEndpointProposer(nn.Module):
    """Conditional rectified-flow velocity field for endpoint displacements."""

    state_dim: int
    env_name: str
    phi_goal_obs_indices: tuple[int, ...]
    hidden_dims: Sequence[int] = _HIDDEN_DIMS

    @nn.compact
    def __call__(
        self,
        observations: jnp.ndarray,
        goals: jnp.ndarray,
        noisy_displacements: jnp.ndarray,
        times: jnp.ndarray,
    ) -> jnp.ndarray:
        goal_inputs = goal_representation(
            goals,
            'phi',
            self.phi_goal_obs_indices,
            env_name=self.env_name,
        )
        times = jnp.asarray(times, dtype=jnp.float32)
        if times.ndim == 2 and times.shape[-1] == 1:
            time_scalars = times[..., 0]
        else:
            time_scalars = times
        time_inputs = time_scalars[..., None].astype(jnp.float32)
        inputs = jnp.concatenate(
            [observations, goal_inputs, noisy_displacements, time_inputs],
            axis=-1,
        )
        return MLP(
            (*self.hidden_dims, self.state_dim),
            activate_final=False,
            layer_norm=_LAYER_NORM,
        )(inputs)


class BridgeResidual(nn.Module):
    """Interior deformation of a fixed, endpoint-pinned reference bridge."""

    state_dim: int
    hidden_dims: Sequence[int] = _HIDDEN_DIMS

    @nn.compact
    def __call__(
        self,
        observations: jnp.ndarray,
        endpoint_displacements: jnp.ndarray,
        times: jnp.ndarray,
    ) -> jnp.ndarray:
        times = jnp.asarray(times, dtype=jnp.float32)
        observations = jnp.broadcast_to(
            observations[:, None, :],
            (*times.shape, observations.shape[-1]),
        )
        endpoint_displacements = jnp.broadcast_to(
            endpoint_displacements[:, None, :],
            (*times.shape, endpoint_displacements.shape[-1]),
        )
        inputs = jnp.concatenate(
            [observations, endpoint_displacements, times[..., None]],
            axis=-1,
        )
        return MLP(
            (*self.hidden_dims, self.state_dim),
            activate_final=False,
            layer_norm=_LAYER_NORM,
        )(inputs)


class InverseDynamics(nn.Module):
    """Decode one desired state transition into an action."""

    action_dim: int
    hidden_dims: Sequence[int] = _HIDDEN_DIMS

    @nn.compact
    def __call__(
        self,
        observations: jnp.ndarray,
        next_observations: jnp.ndarray,
    ) -> jnp.ndarray:
        inputs = jnp.concatenate([observations, next_observations], axis=-1)
        return MLP(
            (*self.hidden_dims, self.action_dim),
            activate_final=False,
            layer_norm=_LAYER_NORM,
        )(inputs)


__all__ = [
    'ScalarTransitiveValue',
    'GaussianEndpointProposer',
    'FlowEndpointProposer',
    'BridgeResidual',
    'InverseDynamics',
]
