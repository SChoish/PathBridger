"""Optional PathBridger extensions and historical inference compatibility.

The sixteen released presets use deterministic bridges and an IDM. This module
keeps path weighting, stochastic prefix models, state-only planning, and the
staged inference shim out of the core value/endpoint/bridge/IDM implementation.
The mixin owns no state: serialization and the public agent type stay unchanged.
"""

from __future__ import annotations

from functools import partial
from typing import Any

import jax
import jax.numpy as jnp

from agents._constants import (
    _ACTION_HORIZON,
    _PREFIX_STREAM_ID,
    _SUBGOAL_VALUE_RATIO_EPS,
    _VALUE_EPS,
)
from agents.prefix_generators import JointFlowPrefix, LowRankGaussianPrefix, low_rank_gaussian_nll


def _temporal_path_weights(
    distances: jnp.ndarray,
    active_transitions: jnp.ndarray,
    strength: jnp.ndarray,
    min_weight: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return one-sided geodesic energies and path weights."""

    active = jnp.asarray(active_transitions, dtype=jnp.float32)
    current = distances[:, :-1]
    following = distances[:, 1:]
    distance_deltas = following - current
    progress_targets = jnp.minimum(1.0, jax.lax.stop_gradient(current))
    defects = jnp.where(
        active > 0.0,
        progress_targets + distance_deltas,
        0.0,
    )
    active_counts = jnp.maximum(jnp.sum(active, axis=1), 1.0)
    energies = jnp.sum(jax.nn.relu(defects) * active, axis=1) / active_counts
    weights = jnp.clip(
        jnp.exp(-jax.lax.stop_gradient(strength * energies)),
        float(min_weight),
        1.0,
    )
    return jax.lax.stop_gradient(weights), energies, defects, distance_deltas


class PathBridgerExtensions:
    """Stateless optional methods inherited by the public PathBridger agent."""

    # Temporal path weighting.

    def _path_weight_strength(self) -> jnp.ndarray:
        """Linearly enable temporal path weighting after value warm-up."""

        beta = float(self.config['path_weight_beta'])
        warmup = int(self.config['path_weight_warmup'])
        ramp = int(self.config['path_weight_ramp'])
        step = jnp.asarray(self.network.step, dtype=jnp.float32)
        if ramp == 0:
            progress = jnp.asarray(step >= warmup, dtype=jnp.float32)
        else:
            progress = jnp.clip((step - warmup) / float(ramp), 0.0, 1.0)
        return jnp.asarray(beta, dtype=jnp.float32) * progress

    def _dataset_path_weights(
        self,
        observations: jnp.ndarray,
        bridge_targets: jnp.ndarray,
        endpoints: jnp.ndarray,
        *,
        full_metrics: bool,
    ) -> tuple[
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
    ]:
        """Score the real five-step prefix against its padded endpoint."""

        prefix = jnp.concatenate(
            [observations[:, None, :], bridge_targets],
            axis=1,
        )
        current_states = prefix[:, :-1, :]
        active = jnp.any(
            jnp.abs(current_states - endpoints[:, None, :]) > 1e-6,
            axis=-1,
        ).astype(jnp.float32)
        strength = self._path_weight_strength()

        def evaluate_geometry(_):
            batch_size, prefix_length, state_dim = prefix.shape
            flat_states = prefix.reshape(batch_size * prefix_length, state_dim)
            flat_endpoints = jnp.broadcast_to(
                endpoints[:, None, :],
                prefix.shape,
            ).reshape(batch_size * prefix_length, state_dim)
            values = jax.nn.sigmoid(
                self.network.select('target_value')(
                    flat_states,
                    flat_endpoints,
                )
            ).reshape(batch_size, prefix_length)
            values = jax.lax.stop_gradient(values)
            safe_values = jnp.clip(values, _VALUE_EPS, 1.0)
            distances = jnp.log(safe_values) / jnp.log(
                jnp.asarray(self.config['discount'], dtype=jnp.float32)
            )
            distance_cap = float(self.config['path_distance_cap_multiplier']) * float(
                self.config['horizon']
            )
            distances = jnp.clip(distances, 0.0, distance_cap)
            return _temporal_path_weights(
                distances,
                active,
                strength,
                float(self.config['path_weight_min']),
            )

        def skip_geometry(_):
            batch_size = observations.shape[0]
            transition_shape = (batch_size, _ACTION_HORIZON)
            return (
                jnp.ones((batch_size,), dtype=jnp.float32),
                jnp.zeros((batch_size,), dtype=jnp.float32),
                jnp.zeros(transition_shape, dtype=jnp.float32),
                jnp.zeros(transition_shape, dtype=jnp.float32),
            )

        # In the common unweighted configuration, keep the geometry branch out
        # of the compiled training executable altogether.  Full-metric updates
        # still evaluate it so diagnostics retain their previous semantics.
        if float(self.config['path_weight_beta']) == 0.0 and not full_metrics:
            weights, energies, defects, distance_deltas = skip_geometry(None)
        else:
            should_evaluate = jnp.logical_or(
                strength > 0.0,
                jnp.asarray(full_metrics),
            )
            weights, energies, defects, distance_deltas = jax.lax.cond(
                should_evaluate,
                evaluate_geometry,
                skip_geometry,
                operand=None,
            )
        return weights, energies, defects, distance_deltas, active, strength

    # Stochastic prefixes.

    def prefix_distribution_loss(
        self,
        batch: dict[str, jnp.ndarray],
        grad_params: Any,
        rng: jax.Array,
        full_metrics: bool = True,
        reference: jnp.ndarray | None = None,
    ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        """Learn an unweighted joint distribution of normalized bridge residuals."""

        prefix_model = self.config['prefix_model']
        if prefix_model == 'deterministic':
            zero = jnp.zeros((), dtype=batch['observations'].dtype)
            return zero, {}

        event_steps = _ACTION_HORIZON - int(int(self.config['horizon']) == _ACTION_HORIZON)
        observations = batch['observations']
        endpoints = batch['endpoint_targets']
        if reference is None:
            reference = self._construct_bridge_at_indices(
                observations,
                endpoints,
                jnp.arange(1, event_steps + 1),
                params=grad_params,
            )
        reference = jax.lax.stop_gradient(reference[:, :event_steps, :])
        residual_targets = (
            batch['bridge_targets'][:, :event_steps, :] - reference
        ) / self.state_scale[None, None, :]
        flat_targets = residual_targets.reshape(residual_targets.shape[0], -1)
        endpoint_displacements = endpoints - observations

        if prefix_model == 'low_rank_gaussian':
            means, sigmas, factors = self.network.select('prefix')(
                observations,
                endpoint_displacements,
                params=grad_params,
            )
            nll = low_rank_gaussian_nll(flat_targets, means, sigmas, factors)
            loss = nll.mean()
            info = {'prefix/loss': loss}
            if full_metrics:
                info.update(
                    {
                        'prefix/nll': loss,
                        'prefix/target_rms': jnp.sqrt(jnp.mean(jnp.square(flat_targets))),
                        'prefix/mean_rms': jnp.sqrt(jnp.mean(jnp.square(means))),
                        'prefix/sigma_mean': sigmas.mean(),
                        'prefix/factor_rms': jnp.sqrt(jnp.mean(jnp.square(factors))),
                    }
                )
            return loss, info

        noise_rng, time_rng = jax.random.split(rng)
        noise = jax.random.normal(
            noise_rng,
            flat_targets.shape,
            dtype=flat_targets.dtype,
        )
        times = jax.random.uniform(
            time_rng,
            (flat_targets.shape[0], 1),
            dtype=flat_targets.dtype,
        )
        noisy_residuals = (1.0 - times) * noise + times * flat_targets
        target_velocities = flat_targets - noise
        predicted_velocities = self.network.select('prefix')(
            observations,
            endpoint_displacements,
            noisy_residuals,
            times,
            params=grad_params,
        )
        per_example_errors = jnp.sum(
            jnp.square(predicted_velocities - target_velocities),
            axis=-1,
        )
        loss = per_example_errors.mean()
        info = {'prefix/loss': loss}
        if full_metrics:
            info.update(
                {
                    'prefix/flow_matching_loss': loss,
                    'prefix/target_rms': jnp.sqrt(jnp.mean(jnp.square(flat_targets))),
                    'prefix/noise_rms': jnp.sqrt(jnp.mean(jnp.square(noise))),
                    'prefix/velocity_rms': jnp.sqrt(jnp.mean(jnp.square(predicted_velocities))),
                }
            )
        return loss, info

    def _sample_stochastic_prefixes(
        self,
        observations: jnp.ndarray,
        endpoints: jnp.ndarray,
        seed: jax.Array,
        *,
        num_samples: int,
        temperature: float,
        reference: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """Sample absolute prefixes for one already-selected endpoint per row."""

        batch_size, state_dim = observations.shape
        event_steps = _ACTION_HORIZON - int(int(self.config['horizon']) == _ACTION_HORIZON)
        event_dim = event_steps * state_dim
        endpoint_displacements = endpoints - observations
        if reference is None:
            reference = self._construct_bridge_at_indices(
                observations,
                endpoints,
                jnp.arange(1, _ACTION_HORIZON + 1),
            )
        temperature_array = jnp.asarray(temperature, dtype=observations.dtype)

        if self.config['prefix_model'] == 'low_rank_gaussian':
            low_rank_rng, diagonal_rng = jax.random.split(seed)
            means, sigmas, factors = self.network.select('prefix')(
                observations,
                endpoint_displacements,
            )
            low_rank_noise = jax.random.normal(
                low_rank_rng,
                (batch_size, num_samples, int(self.config['prefix_rank'])),
                dtype=observations.dtype,
            )
            diagonal_noise = jax.random.normal(
                diagonal_rng,
                (batch_size, num_samples, event_dim),
                dtype=observations.dtype,
            )
            correlated = jnp.einsum(
                'bpr,bmr->bmp',
                factors,
                low_rank_noise,
            )
            normalized = means[:, None, :] + temperature_array * (
                correlated + sigmas[:, None, :] * diagonal_noise
            )
        else:
            normalized = temperature_array * jax.random.normal(
                seed,
                (batch_size, num_samples, event_dim),
                dtype=observations.dtype,
            )
            flat_observations = jnp.broadcast_to(
                observations[:, None, :],
                (batch_size, num_samples, state_dim),
            ).reshape(batch_size * num_samples, state_dim)
            flat_displacements = jnp.broadcast_to(
                endpoint_displacements[:, None, :],
                (batch_size, num_samples, state_dim),
            ).reshape(batch_size * num_samples, state_dim)
            flat_normalized = normalized.reshape(batch_size * num_samples, event_dim)
            flow_steps = int(self.config['prefix_flow_steps'])
            step_size = jnp.asarray(1.0 / flow_steps, dtype=observations.dtype)
            for step in range(flow_steps):
                times = jnp.full(
                    (batch_size * num_samples, 1),
                    step / flow_steps,
                    dtype=observations.dtype,
                )
                velocities = self.network.select('prefix')(
                    flat_observations,
                    flat_displacements,
                    flat_normalized,
                    times,
                )
                flat_normalized = flat_normalized + step_size * velocities
            normalized = flat_normalized.reshape(batch_size, num_samples, event_dim)

        normalized = normalized.reshape(
            batch_size,
            num_samples,
            event_steps,
            state_dim,
        )
        prefixes = jnp.broadcast_to(
            reference[:, None, :, :],
            (batch_size, num_samples, _ACTION_HORIZON, state_dim),
        )
        stochastic_states = (
            reference[:, None, :event_steps, :] + self.state_scale[None, None, None, :] * normalized
        )
        prefixes = prefixes.at[:, :, :event_steps, :].set(stochastic_states)
        starts = jnp.broadcast_to(
            observations[:, None, None, :],
            (batch_size, num_samples, 1, state_dim),
        )
        return jnp.concatenate([starts, prefixes], axis=2)

    def _sample_prefixes(
        self,
        observations: jnp.ndarray,
        endpoints: jnp.ndarray,
        seed: jax.Array,
        *,
        num_samples: int,
        temperature: float,
        include_deterministic: bool,
    ) -> jnp.ndarray:
        if self.config['prefix_model'] == 'deterministic':
            return self._construct_bridge_at_indices(
                observations,
                endpoints,
                jnp.arange(0, _ACTION_HORIZON + 1),
            )[:, None, :, :]

        deterministic_states = None
        if include_deterministic:
            deterministic_states = self._construct_bridge_at_indices(
                observations,
                endpoints,
                jnp.arange(0, _ACTION_HORIZON + 1),
            )
            reference = deterministic_states[:, 1:, :]
        else:
            reference = self._construct_bridge_at_indices(
                observations,
                endpoints,
                jnp.arange(1, _ACTION_HORIZON + 1),
            )

        stochastic = self._sample_stochastic_prefixes(
            observations,
            endpoints,
            seed,
            num_samples=num_samples,
            temperature=temperature,
            reference=reference,
        )
        if include_deterministic:
            return jnp.concatenate(
                [deterministic_states[:, None, :, :], stochastic],
                axis=1,
            )
        return stochastic

    @partial(
        jax.jit,
        static_argnames=('num_samples', 'temperature', 'include_deterministic'),
    )
    def sample_prefixes(
        self,
        observations: jnp.ndarray,
        endpoints: jnp.ndarray,
        seed: jax.Array | None = None,
        *,
        num_samples: int = 1,
        temperature: float = 1.0,
        include_deterministic: bool = False,
    ) -> jnp.ndarray:
        """Sample endpoint-conditioned absolute state prefixes."""

        if num_samples < 1:
            raise ValueError('num_samples must be at least one.')
        if temperature < 0.0:
            raise ValueError('temperature must be non-negative.')
        if seed is None:
            seed = jax.random.PRNGKey(0)
        squeeze = observations.ndim == 1
        if squeeze:
            observations = observations[None, :]
            endpoints = endpoints[None, :]
        prefixes = self._sample_prefixes(
            observations,
            endpoints,
            seed,
            num_samples=num_samples,
            temperature=temperature,
            include_deterministic=include_deterministic,
        )
        return prefixes[0] if squeeze else prefixes

    def _transv_chain_scores(
        self,
        prefixes: jnp.ndarray,
        endpoints: jnp.ndarray,
    ) -> jnp.ndarray:
        """Score prefix candidates with the EMA TransV multiplicative chain."""

        batch_size, num_prefixes, _, state_dim = prefixes.shape
        current = prefixes[:, :, :-1, :].reshape(-1, state_dim)
        following = prefixes[:, :, 1:, :].reshape(-1, state_dim)
        final_states = prefixes[:, :, -1, :].reshape(-1, state_dim)
        repeated_endpoints = jnp.broadcast_to(
            endpoints[:, None, :],
            (batch_size, num_prefixes, state_dim),
        ).reshape(-1, state_dim)
        value_logits = self.network.select('target_value')(
            jnp.concatenate([current, final_states], axis=0),
            jnp.concatenate([following, repeated_endpoints], axis=0),
        )
        transition_logits, remainder_logits = jnp.split(
            value_logits,
            [current.shape[0]],
            axis=0,
        )
        transition_values = jax.nn.sigmoid(transition_logits).reshape(
            batch_size,
            num_prefixes,
            _ACTION_HORIZON,
        )
        remainder_values = jax.nn.sigmoid(remainder_logits).reshape(
            batch_size,
            num_prefixes,
        )
        return jnp.sum(jnp.log(jnp.clip(transition_values, _VALUE_EPS, 1.0)), axis=-1) + jnp.log(
            jnp.clip(remainder_values, _VALUE_EPS, 1.0)
        )

    # State-only planning.

    @partial(
        jax.jit,
        static_argnames=('num_candidates', 'temperature'),
    )
    def sample_state_prefix(
        self,
        observations: jnp.ndarray,
        goals: jnp.ndarray,
        seed: jax.Array | None = None,
        num_candidates: int | None = None,
        temperature: float | None = None,
    ) -> jnp.ndarray:
        """Plan one absolute six-state prefix without decoding actions."""

        if seed is None:
            seed = jax.random.PRNGKey(0)
        if num_candidates is None:
            num_candidates = int(self.config['eval_num_candidates'])
        if temperature is None:
            temperature = float(self.config['eval_temperature'])
        if num_candidates < 1:
            raise ValueError('num_candidates must be at least one.')
        if temperature < 0.0:
            raise ValueError('temperature must be non-negative.')

        squeeze = observations.ndim == 1
        if squeeze:
            observations = observations[None, :]
            goals = goals[None, :]

        candidates = self._sample_endpoint_candidates(
            observations,
            goals,
            seed,
            num_candidates=num_candidates,
            temperature=temperature,
        )
        batch_size, _, state_dim = candidates.shape
        repeated_observations = jnp.broadcast_to(observations[:, None, :], candidates.shape)
        repeated_goals = jnp.broadcast_to(
            goals[:, None, :],
            (batch_size, num_candidates, goals.shape[-1]),
        )
        flat_observations = repeated_observations.reshape(batch_size * num_candidates, state_dim)
        flat_candidates = candidates.reshape(batch_size * num_candidates, state_dim)
        flat_goals = repeated_goals.reshape(batch_size * num_candidates, goals.shape[-1])
        values_to_endpoint = jax.nn.sigmoid(
            self.network.select('value')(flat_observations, flat_candidates)
        )
        values_to_goal = jax.nn.sigmoid(self.network.select('value')(flat_candidates, flat_goals))
        values_direct = jax.nn.sigmoid(self.network.select('value')(flat_observations, flat_goals))
        scores = (
            values_to_endpoint * values_to_goal / (values_direct + _SUBGOAL_VALUE_RATIO_EPS)
        ).reshape(batch_size, num_candidates)
        best_indices = jnp.argmax(scores, axis=1)
        selected_endpoints = jnp.take_along_axis(candidates, best_indices[:, None, None], axis=1)[
            :, 0, :
        ]

        prefix_seed = jax.random.fold_in(seed, _PREFIX_STREAM_ID)
        prefix = self._select_prefix(observations, selected_endpoints, prefix_seed)

        return prefix[0] if squeeze else prefix

    # Historical staged inference.

    @partial(
        jax.jit,
        static_argnames=('num_candidates', 'temperature'),
    )
    def _legacy_endpoint_stage(
        self,
        observations: jnp.ndarray,
        goals: jnp.ndarray,
        seed: jax.Array,
        *,
        num_candidates: int,
        temperature: float,
    ) -> jnp.ndarray:
        return self._sample_endpoint_candidates(
            observations,
            goals,
            seed,
            num_candidates=num_candidates,
            temperature=temperature,
        )

    @jax.jit
    def _legacy_selector_stage(
        self,
        observations: jnp.ndarray,
        candidates: jnp.ndarray,
        goals: jnp.ndarray,
    ) -> jnp.ndarray:
        num_candidates = candidates.shape[1]
        observations = jnp.asarray(observations, dtype=jnp.float32)
        candidates = jnp.asarray(candidates, dtype=jnp.float32)
        goals = jnp.asarray(goals, dtype=jnp.float32)
        flat_observations = jnp.repeat(observations[:, None, :], num_candidates, axis=1).reshape(
            observations.shape[0] * num_candidates, -1
        )
        flat_candidates = candidates.reshape(candidates.shape[0] * num_candidates, -1)
        flat_goals = jnp.repeat(goals[:, None, :], num_candidates, axis=1).reshape(
            goals.shape[0] * num_candidates, -1
        )
        values_to_endpoint = jax.nn.sigmoid(
            self.network.select('value')(flat_observations, flat_candidates)
        )
        values_to_goal = jax.nn.sigmoid(self.network.select('value')(flat_candidates, flat_goals))
        values_direct = jax.nn.sigmoid(self.network.select('value')(flat_observations, flat_goals))
        return (
            values_to_endpoint * values_to_goal / (values_direct + _SUBGOAL_VALUE_RATIO_EPS)
        ).reshape(observations.shape[0], num_candidates)

    def _legacy_idm_stage(self, trajectories: jnp.ndarray) -> jnp.ndarray:
        current = trajectories[:, :_ACTION_HORIZON, :]
        following = trajectories[:, 1 : _ACTION_HORIZON + 1, :]
        state_dim = trajectories.shape[-1]
        actions = self.network.select('idm')(
            current.reshape(-1, state_dim),
            following.reshape(-1, state_dim),
        )
        return actions.reshape(trajectories.shape[0], _ACTION_HORIZON, -1)

    def _sample_action_chunks_legacy_staged(
        self,
        observations: jnp.ndarray,
        goals: jnp.ndarray,
        seed: jax.Array | None,
        *,
        num_candidates: int | None,
        temperature: float | None,
    ) -> jnp.ndarray:
        if self.config['prefix_model'] != 'deterministic':
            raise ValueError('Legacy staged inference supports deterministic prefixes only.')
        if seed is None:
            seed = jax.random.PRNGKey(0)
        if num_candidates is None:
            num_candidates = int(self.config['eval_num_candidates'])
        if temperature is None:
            temperature = float(self.config['eval_temperature'])
        if num_candidates < 1:
            raise ValueError('num_candidates must be at least one.')
        if temperature < 0.0:
            raise ValueError('temperature must be non-negative.')

        squeeze = observations.ndim == 1
        if squeeze:
            observations = observations[None, :]
            goals = goals[None, :]
        candidates = self._legacy_endpoint_stage(
            observations,
            goals,
            seed,
            num_candidates=num_candidates,
            temperature=temperature,
        )
        scores = self._legacy_selector_stage(observations, candidates, goals)
        best_indices = jnp.argmax(scores, axis=1)
        selected = jnp.take_along_axis(candidates, best_indices[:, None, None], axis=1)[:, 0, :]
        trajectories = self.construct_bridge(observations, selected)
        actions = self._legacy_idm_stage(trajectories)
        return actions[0] if squeeze else actions

    def _select_prefix(
        self,
        observations: jnp.ndarray,
        selected_endpoints: jnp.ndarray,
        prefix_seed: jax.Array,
    ) -> jnp.ndarray:
        """Select a deterministic or stochastic prefix for fixed endpoints."""

        if self.config['prefix_model'] == 'deterministic':
            prefix_candidates = self._sample_prefixes(
                observations,
                selected_endpoints,
                prefix_seed,
                num_samples=1,
                temperature=0.0,
                include_deterministic=True,
            )
            prefix = prefix_candidates[:, 0, :, :]
        elif self.config['eval_prefix_selection'] == 'sample_one':
            prefix_candidates = self._sample_prefixes(
                observations,
                selected_endpoints,
                prefix_seed,
                num_samples=1,
                temperature=float(self.config['eval_prefix_temperature']),
                include_deterministic=False,
            )
            prefix = prefix_candidates[:, 0, :, :]
        else:
            prefix_candidates = self._sample_prefixes(
                observations,
                selected_endpoints,
                prefix_seed,
                num_samples=int(self.config['eval_num_prefix_samples']),
                temperature=float(self.config['eval_prefix_temperature']),
                include_deterministic=bool(self.config['eval_include_deterministic_prefix']),
            )
            chain_scores = self._transv_chain_scores(
                prefix_candidates,
                selected_endpoints,
            )
            best_prefix_indices = jnp.argmax(chain_scores, axis=1)
            prefix = jnp.take_along_axis(
                prefix_candidates,
                best_prefix_indices[:, None, None, None],
                axis=1,
            )[:, 0, :, :]
        return prefix


def create_prefix_module(config, observations):
    """Build an optional prefix module without altering core parameter names."""

    prefix_model = config['prefix_model']
    horizon = int(config['horizon'])
    state_dim = int(observations.shape[-1])
    event_steps = _ACTION_HORIZON - int(horizon == _ACTION_HORIZON)
    prefix_event_dim = event_steps * state_dim
    prefix_def = None
    prefix_args = None
    if prefix_model == 'low_rank_gaussian':
        prefix_def = LowRankGaussianPrefix(
            event_dim=prefix_event_dim,
            rank=int(config['prefix_rank']),
            sigma_floor=float(config['prefix_sigma_floor']),
        )
        prefix_args = (observations, jnp.zeros_like(observations))
    elif prefix_model == 'joint_flow':
        prefix_def = JointFlowPrefix(event_dim=prefix_event_dim)
        prefix_args = (
            observations,
            jnp.zeros_like(observations),
            jnp.zeros(
                (observations.shape[0], prefix_event_dim),
                dtype=jnp.float32,
            ),
            jnp.zeros((observations.shape[0], 1), dtype=jnp.float32),
        )
    return prefix_def, prefix_args


__all__ = ['PathBridgerExtensions', 'create_prefix_module']
