"""PathBridger training, bridge construction, and policy sampling.

The released agent deliberately contains no actor or action-conditioned
critic.  It learns exactly four policy-side components:

* a scalar transitive value with an EMA target,
* a Gaussian or rectified-flow endpoint proposer,
* an endpoint-pinned residual bridge, and
* an inverse-dynamics model (IDM).

Neural module definitions live in :mod:`agents.models`; the default schema
lives in :mod:`agents.defaults`. Optional features live in :mod:`agents.extensions`.
Existing agent methods and import paths remain available.
"""

from __future__ import annotations

from functools import partial
from typing import Any

import flax
import jax
import jax.numpy as jnp
import optax

from agents._constants import (
    _ACTION_HORIZON,
    _PREFIX_STREAM_ID,
    _SUBGOAL_VALUE_RATIO_EPS,
    _VALUE_EPS,
)
from agents.defaults import get_config as get_config
from agents.extensions import (
    PathBridgerExtensions,
    _temporal_path_weights as _temporal_path_weights,
    create_prefix_module,
)
from agents.models import (
    BridgeResidual as BridgeResidual,
    FlowEndpointProposer as FlowEndpointProposer,
    GaussianEndpointProposer as GaussianEndpointProposer,
    InverseDynamics as InverseDynamics,
    ScalarTransitiveValue as ScalarTransitiveValue,
)
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.goal_representation import (
    assert_phi_goal_obs_indices,
    infer_phi_goal_obs_indices,
)
from utils.training_protocol import (
    MANUSCRIPT_TRAINING_PROTOCOL,
    REPORTED_RUN_LEGACY_TRAINING_PROTOCOL,
    normalize_training_protocol,
)

# These are algorithm constants, not experiment options.
_LEARNING_RATE = 3e-4
_BASE_HORIZON = 5
_TAU_V = 0.7
_TARGET_TAU = 0.005
_ENDPOINT_WEIGHT_CAP = 5.0
_FLOW_STEPS = 8
_BRIDGE_ALPHA_POWER = 0.8
_VALUE_DISTANCE_WEIGHT_CLIP_MIN = 0.05
_VALUE_DISTANCE_WEIGHT_CLIP_MAX = 1.0
_FLOW_TIME_EPSILON = 1e-4

_REQUIRED_BATCH_KEYS = (
    'observations',
    'next_observations',
    'actions',
    'bridge_targets',
    'endpoint_goals',
    'endpoint_targets',
    'value_goals',
    'value_offsets',
    'base_goals',
    'base_offsets',
    'transitive_subgoals',
    'transitive_offsets',
    'transitive_valids',
)
_REQUIRED_STATE_BATCH_KEYS = tuple(key for key in _REQUIRED_BATCH_KEYS if key != 'actions')


def _legacy_bridge_progress(horizon: int) -> jnp.ndarray:
    """Reproduce the legacy hard-bridge beta calculation, including rounding."""

    indices = jnp.arange(horizon + 1, dtype=jnp.float32)
    desired = jnp.power(indices / float(horizon), _BRIDGE_ALPHA_POWER)
    total = jnp.asarray(1.0, dtype=jnp.float32)
    cumulative = jnp.arcsinh(desired * jnp.sinh(total))
    theta = jnp.maximum(cumulative[1:] - cumulative[:-1], 1e-12)
    transition = jnp.exp(theta)
    step_variance = 1.0 - jnp.exp(-2.0 * theta)
    innovation_variance = step_variance * jnp.square(transition)

    variances = [jnp.asarray(0.0, dtype=jnp.float32)]
    for step in range(horizon):
        variances.append(jnp.square(transition[step]) * variances[-1] + innovation_variance[step])
    variance = jnp.stack(variances)
    tails = [None] * (horizon + 1)
    tails[horizon] = jnp.asarray(1.0, dtype=jnp.float32)
    for step in range(horizon - 1, -1, -1):
        tails[step] = transition[step] * tails[step + 1]
    beta = variance * jnp.stack(tails) / jnp.maximum(variance[-1], 1e-12)
    return beta.at[0].set(0.0).at[-1].set(1.0)


def _expectile_bce(
    logits: jnp.ndarray,
    targets: jnp.ndarray,
) -> jnp.ndarray:
    """Expectile-weighted BCE used by the transitive constituent target."""

    predictions = jax.nn.sigmoid(logits)
    weights = jnp.where(predictions > targets, 1.0 - _TAU_V, _TAU_V)
    return weights * optax.sigmoid_binary_cross_entropy(logits, targets)


def _replace_module_params(params: Any, module_name: str, value: Any) -> Any:
    """Replace one ModuleDict subtree without changing the container type."""

    is_frozen = isinstance(params, flax.core.FrozenDict)
    mutable = flax.core.unfreeze(params) if is_frozen else dict(params)
    mutable[f'modules_{module_name}'] = value
    return flax.core.freeze(mutable) if is_frozen else mutable


class PathBridgerAgent(PathBridgerExtensions, flax.struct.PyTreeNode):
    """Actor-free PathBridger training and evaluation state."""

    rng: Any
    network: TrainState
    state_scale: Any
    config: Any = nonpytree_field()

    def _distance_weight_from_values(
        self,
        values: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Compute the protocol-specific inverse-distance weight."""

        values = jax.lax.stop_gradient(values)
        safe_values = jnp.clip(values, _VALUE_EPS, 1.0)
        distance = jnp.log(safe_values) / jnp.log(
            jnp.asarray(self.config['discount'], dtype=jnp.float32)
        )
        legacy = self.config['training_objective_protocol'] == REPORTED_RUN_LEGACY_TRAINING_PROTOCOL
        if legacy:
            distance = jnp.maximum(distance, 0.0)
        weights = jnp.power(
            1.0 + distance,
            -jnp.asarray(
                self.config['value_distance_weight_power'],
                dtype=jnp.float32,
            ),
        )
        if legacy:
            weights = jnp.clip(
                weights,
                _VALUE_DISTANCE_WEIGHT_CLIP_MIN,
                _VALUE_DISTANCE_WEIGHT_CLIP_MAX,
            )
        return jax.lax.stop_gradient(weights), values

    def _manuscript_value_loss(
        self,
        batch: dict[str, jnp.ndarray],
        grad_params: Any,
        full_metrics: bool,
    ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        """6a1806b scalar-value objective on a caller-supplied batch."""

        observations = batch['observations']
        discount = jnp.asarray(self.config['discount'], dtype=jnp.float32)
        online_logits = self.network.select('value')(
            jnp.concatenate([observations, observations, observations], axis=0),
            jnp.concatenate(
                [observations, batch['base_goals'], batch['value_goals']],
                axis=0,
            ),
            params=grad_params,
        )
        self_logits, base_logits, transitive_logits = jnp.split(online_logits, 3, axis=0)
        self_loss = optax.sigmoid_binary_cross_entropy(
            self_logits, jnp.ones_like(self_logits)
        ).mean()

        base_offsets = jnp.asarray(batch['base_offsets'], dtype=jnp.float32)
        base_targets = jnp.power(discount, base_offsets)
        base_bce = optax.sigmoid_binary_cross_entropy(base_logits, base_targets)

        goals = batch['value_goals']
        subgoals = batch['transitive_subgoals']
        left_offsets = jnp.asarray(batch['transitive_offsets'], dtype=jnp.float32)
        total_offsets = jnp.asarray(batch['value_offsets'], dtype=jnp.float32)
        right_offsets = total_offsets - left_offsets

        use_distance_weights = float(self.config['value_distance_weight_power']) != 0.0
        need_distance_values = use_distance_weights or full_metrics
        target_states = [observations, subgoals]
        target_goals = [subgoals, goals]
        if need_distance_values:
            target_states.extend([observations, observations])
            target_goals.extend([batch['base_goals'], goals])
        target_logits = self.network.select('target_value')(
            jnp.concatenate(target_states, axis=0),
            jnp.concatenate(target_goals, axis=0),
        )
        target_values = jnp.split(jax.nn.sigmoid(target_logits), len(target_states), axis=0)
        target_left, target_right = target_values[:2]
        if use_distance_weights:
            base_distance_weights, _ = self._distance_weight_from_values(target_values[2])
            transitive_distance_weights, _ = self._distance_weight_from_values(target_values[3])
        else:
            base_distance_weights = jnp.ones_like(base_bce)
            transitive_distance_weights = jnp.ones_like(base_bce)
        base_loss = jnp.mean(base_distance_weights * base_bce)

        exact_left = jnp.power(discount, left_offsets)
        exact_right = jnp.power(discount, right_offsets)
        mixed_left = jnp.where(left_offsets <= _BASE_HORIZON, exact_left, target_left)
        mixed_right = jnp.where(right_offsets <= _BASE_HORIZON, exact_right, target_right)
        transitive_targets = jax.lax.stop_gradient(mixed_left * mixed_right)
        transitive_bce = _expectile_bce(transitive_logits, transitive_targets)
        transitive_valids = jnp.asarray(batch['transitive_valids'], dtype=jnp.float32)
        if transitive_valids.ndim > 1:
            transitive_valids = transitive_valids.reshape(transitive_valids.shape[0], -1)[:, 0]
        transitive_loss = jnp.sum(
            transitive_valids * transitive_distance_weights * transitive_bce
        ) / jnp.maximum(jnp.sum(transitive_valids), 1.0)

        loss = self_loss + base_loss + transitive_loss
        info = {
            'value/loss': loss,
            'value/self_loss': self_loss,
            'value/base_loss': base_loss,
            'value/transitive_loss': transitive_loss,
        }
        if full_metrics:
            info.update(
                {
                    'value/self_mean': jax.nn.sigmoid(self_logits).mean(),
                    'value/base_mean': jax.nn.sigmoid(base_logits).mean(),
                    'value/base_target_mean': base_targets.mean(),
                    'value/base_target_value_mean': target_values[2].mean(),
                    'value/base_distance_weight_mean': (base_distance_weights.mean()),
                    'value/transitive_mean': (jax.nn.sigmoid(transitive_logits).mean()),
                    'value/transitive_target_mean': transitive_targets.mean(),
                    'value/transitive_target_value_mean': target_values[3].mean(),
                    'value/transitive_distance_weight_mean': (transitive_distance_weights.mean()),
                    'value/transitive_valid_fraction': transitive_valids.mean(),
                }
            )
        return loss, info

    # 1. Transitive value.

    def value_loss(
        self,
        batch: dict[str, jnp.ndarray],
        grad_params: Any,
        full_metrics: bool = True,
    ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        """Dispatch the versioned scalar-value objective."""

        if self.config['training_objective_protocol'] == MANUSCRIPT_TRAINING_PROTOCOL:
            return self._manuscript_value_loss(batch, grad_params, full_metrics)

        observations = batch['observations']
        discount = jnp.asarray(self.config['discount'], dtype=jnp.float32)

        # Keep the calls separate. The frozen split implementation compiled
        # these as three independent module calls; batching them changes the
        # reduction boundary and therefore the float32 gradient bits.
        self_logits = self.network.select('value')(
            observations,
            observations,
            params=grad_params,
        )
        base_logits = self.network.select('value')(
            observations,
            batch['base_goals'],
            params=grad_params,
        )
        transitive_logits = self.network.select('value')(
            observations,
            batch['value_goals'],
            params=grad_params,
        )
        self_targets = jnp.ones_like(self_logits)
        self_loss = optax.sigmoid_binary_cross_entropy(
            self_logits,
            self_targets,
        ).mean()

        base_offsets = jnp.asarray(batch['base_offsets'], dtype=jnp.float32)
        base_targets = jnp.clip(
            jnp.power(discount, base_offsets),
            _VALUE_EPS,
            1.0,
        )
        base_bce = optax.sigmoid_binary_cross_entropy(
            base_logits,
            base_targets,
        )
        base_loss = base_bce.mean()

        goals = batch['value_goals']
        subgoals = batch['transitive_subgoals']
        left_offsets = jnp.asarray(
            batch['transitive_offsets'],
            dtype=jnp.float32,
        )
        total_offsets = jnp.asarray(batch['value_offsets'], dtype=jnp.float32)
        right_offsets = total_offsets - left_offsets

        target_left = jnp.clip(
            jax.nn.sigmoid(self.network.select('target_value')(observations, subgoals)),
            _VALUE_EPS,
            1.0,
        )
        target_right = jnp.clip(
            jax.nn.sigmoid(self.network.select('target_value')(subgoals, goals)),
            _VALUE_EPS,
            1.0,
        )
        transitive_values = jax.nn.sigmoid(transitive_logits)
        transitive_distance_weights, _ = self._distance_weight_from_values(transitive_values)

        exact_left = jnp.clip(jnp.power(discount, left_offsets), _VALUE_EPS, 1.0)
        exact_right = jnp.clip(jnp.power(discount, right_offsets), _VALUE_EPS, 1.0)
        mixed_left = jnp.where(
            left_offsets <= _BASE_HORIZON,
            exact_left,
            target_left,
        )
        mixed_right = jnp.where(
            right_offsets <= _BASE_HORIZON,
            exact_right,
            target_right,
        )
        transitive_targets = jax.lax.stop_gradient(
            jnp.clip(mixed_left * mixed_right, _VALUE_EPS, 1.0)
        )

        transitive_bce = _expectile_bce(
            transitive_logits,
            transitive_targets,
        )
        transitive_valids = jnp.asarray(
            batch['transitive_valids'],
            dtype=jnp.float32,
        )
        if transitive_valids.ndim > 1:
            transitive_valids = transitive_valids.reshape(
                transitive_valids.shape[0],
                -1,
            )[:, 0]
        transitive_numerator = jnp.sum(
            transitive_valids * transitive_distance_weights * transitive_bce
        )
        transitive_loss = transitive_numerator / jnp.maximum(
            jnp.sum(transitive_valids * transitive_distance_weights),
            _VALUE_EPS,
        )

        loss = self_loss + base_loss + transitive_loss
        info = {
            'value/loss': loss,
            'value/self_loss': self_loss,
            'value/base_loss': base_loss,
            'value/transitive_loss': transitive_loss,
        }
        if full_metrics:
            info.update(
                {
                    'value/self_mean': jax.nn.sigmoid(self_logits).mean(),
                    'value/base_mean': jax.nn.sigmoid(base_logits).mean(),
                    'value/base_target_mean': base_targets.mean(),
                    'value/transitive_mean': jax.nn.sigmoid(transitive_logits).mean(),
                    'value/transitive_target_mean': transitive_targets.mean(),
                    'value/transitive_distance_weight_mean': (transitive_distance_weights.mean()),
                    'value/transitive_valid_fraction': transitive_valids.mean(),
                }
            )
        return loss, info

    def _endpoint_weights(
        self,
        observations: jnp.ndarray,
        goals: jnp.ndarray,
        endpoint_targets: jnp.ndarray,
        predicted_endpoints: jnp.ndarray | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        if (
            self.config.get(
                'training_objective_protocol',
                REPORTED_RUN_LEGACY_TRAINING_PROTOCOL,
            )
            == MANUSCRIPT_TRAINING_PROTOCOL
        ):
            value_logits = self.network.select('target_value')(
                jnp.concatenate([observations, endpoint_targets], axis=0),
                jnp.concatenate([goals, goals], axis=0),
            )
            current_values, endpoint_values = jnp.split(jax.nn.sigmoid(value_logits), 2, axis=0)
        else:
            # Split DynamicsAgent borrowed the critic's online value and also
            # evaluated the predicted endpoint, whose result it discarded.
            if predicted_endpoints is None:
                predicted_endpoints = endpoint_targets
            value_logits = self.network.select('value')(
                jnp.concatenate(
                    [predicted_endpoints, observations, endpoint_targets],
                    axis=0,
                ),
                jnp.concatenate([goals, goals, goals], axis=0),
            )
            _, current_values, endpoint_values = jnp.split(jax.nn.sigmoid(value_logits), 3, axis=0)
        value_gap = endpoint_values - current_values
        weights = jnp.minimum(
            _ENDPOINT_WEIGHT_CAP,
            jnp.exp(
                jnp.asarray(
                    self.config['endpoint_value_scale'],
                    dtype=jnp.float32,
                )
                * value_gap
            ),
        )
        return jax.lax.stop_gradient(weights), jax.lax.stop_gradient(value_gap)

    # 2. Endpoint proposer.

    def endpoint_loss(
        self,
        batch: dict[str, jnp.ndarray],
        grad_params: Any,
        rng: jax.Array,
        full_metrics: bool = True,
    ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        """Weighted PBG likelihood or PBF generator-matching loss."""

        observations = batch['observations']
        goals = batch['endpoint_goals']
        endpoint_targets = batch['endpoint_targets']
        displacement_targets = endpoint_targets - observations
        manuscript = self.config['training_objective_protocol'] == MANUSCRIPT_TRAINING_PROTOCOL
        if manuscript:
            weights, value_gap = self._endpoint_weights(observations, goals, endpoint_targets)
        if self.config['endpoint_distribution'] == 'gaussian':
            means, log_stds = self.network.select('endpoint')(
                observations,
                goals,
                params=grad_params,
            )
            if not manuscript:
                _, sampling_rng = jax.random.split(rng, 2)
                noise = jax.random.normal(
                    sampling_rng,
                    means.shape,
                    dtype=means.dtype,
                )
                predicted_endpoints = observations + means + noise * jnp.exp(log_stds)
                weights, value_gap = self._endpoint_weights(
                    observations,
                    goals,
                    endpoint_targets,
                    predicted_endpoints,
                )
            inverse_variances = jnp.exp(-2.0 * log_stds)
            nll = 0.5 * jnp.sum(
                jnp.square(displacement_targets - means) * inverse_variances
                + 2.0 * log_stds
                + jnp.log(2.0 * jnp.pi),
                axis=-1,
            )
            loss = jnp.mean(weights * nll)
            info = {
                'endpoint/loss': loss,
            }
            if full_metrics:
                info['endpoint/nll'] = nll.mean()
                info['endpoint/std_mean'] = jnp.exp(log_stds).mean()
        else:
            if manuscript:
                noise_rng, time_rng = jax.random.split(rng)
                sample_rng = None
            else:
                # Split DynamicsAgent first split the update loss key, then
                # split the resulting flow key three ways.
                _, flow_rng = jax.random.split(rng, 2)
                noise_rng, time_rng, sample_rng = jax.random.split(flow_rng, 3)
            noise = jax.random.normal(
                noise_rng,
                displacement_targets.shape,
                dtype=displacement_targets.dtype,
            )
            times = jax.random.uniform(
                time_rng,
                (displacement_targets.shape[0], 1),
                minval=_FLOW_TIME_EPSILON,
                maxval=1.0 - _FLOW_TIME_EPSILON,
                dtype=displacement_targets.dtype,
            )
            noisy_displacements = (1.0 - times) * noise + times * displacement_targets
            target_velocities = displacement_targets - noise
            predicted_velocities = self.network.select('endpoint')(
                observations,
                goals,
                noisy_displacements,
                times,
                params=grad_params,
            )
            if not manuscript:
                sample_noise = jax.random.normal(
                    sample_rng,
                    displacement_targets.shape,
                    dtype=displacement_targets.dtype,
                )
                sampled_displacements = self._flow_endpoint_samples(
                    observations,
                    goals,
                    sample_noise[:, None, :],
                    params=grad_params,
                )[:, 0, :]
                weights, value_gap = self._endpoint_weights(
                    observations,
                    goals,
                    endpoint_targets,
                    observations + sampled_displacements,
                )
            squared_flow_errors = jnp.square(predicted_velocities - target_velocities)
            flow_errors = (
                jnp.sum(squared_flow_errors, axis=-1)
                if manuscript
                else jnp.mean(squared_flow_errors, axis=-1)
            )
            loss = jnp.mean(weights * flow_errors)
            info = {
                'endpoint/loss': loss,
            }
            if full_metrics:
                info['endpoint/flow_matching_loss'] = flow_errors.mean()
                info['endpoint/flow_time_mean'] = times.mean()

        if full_metrics:
            info.update(
                {
                    'endpoint/weight_mean': weights.mean(),
                    'endpoint/weight_max': weights.max(),
                    'endpoint/value_gap_mean': value_gap.mean(),
                }
            )
        return loss, info

    def _construct_bridge_at_indices(
        self,
        observations: jnp.ndarray,
        endpoints: jnp.ndarray,
        indices: jnp.ndarray,
        *,
        params: Any | None = None,
    ) -> jnp.ndarray:
        """Construct only the requested bridge indices."""

        if self.config['training_objective_protocol'] == MANUSCRIPT_TRAINING_PROTOCOL:
            horizon = int(self.config['horizon'])
            indices = jnp.asarray(indices, dtype=jnp.float32)
            times = indices / jnp.asarray(horizon, dtype=jnp.float32)
            times = jnp.broadcast_to(
                times[None, :],
                (observations.shape[0], indices.shape[0]),
            )
            alphas = jnp.power(times, _BRIDGE_ALPHA_POWER)
            masks = (
                indices
                * (jnp.asarray(horizon, dtype=jnp.float32) - indices)
                / jnp.asarray(horizon * horizon, dtype=jnp.float32)
            )
            endpoint_displacements = endpoints - observations
            residuals = self.network.select('bridge')(
                observations,
                endpoint_displacements,
                times,
                params=params,
            )
            displacements = (
                alphas[..., None] * endpoint_displacements[:, None, :]
                + masks[None, :, None] * residuals
            )
            bridge = observations[:, None, :] + displacements
            bridge = jnp.where(
                (indices == 0)[None, :, None],
                observations[:, None, :],
                bridge,
            )
            bridge = jnp.where(
                (indices == horizon)[None, :, None],
                endpoints[:, None, :],
                bridge,
            )
            return bridge

        displacements = self._construct_bridge_displacements_at_indices(
            observations,
            endpoints,
            indices,
            params=params,
        )
        indices = jnp.asarray(indices, dtype=jnp.float32)
        horizon = int(self.config['horizon'])
        bridge = observations[:, None, :] + displacements
        bridge = jnp.where(
            (indices == 0)[None, :, None],
            observations[:, None, :],
            bridge,
        )
        bridge = jnp.where(
            (indices == horizon)[None, :, None],
            endpoints[:, None, :],
            bridge,
        )
        return bridge

    def _construct_bridge_displacements_at_indices(
        self,
        observations: jnp.ndarray,
        endpoints: jnp.ndarray,
        indices: jnp.ndarray,
        *,
        params: Any | None = None,
    ) -> jnp.ndarray:
        """Construct requested bridge indices in the legacy local frame."""

        horizon = int(self.config['horizon'])
        indices = jnp.asarray(indices, dtype=jnp.float32)
        times = indices / jnp.asarray(horizon, dtype=jnp.float32)
        times = jnp.broadcast_to(
            times[None, :],
            (observations.shape[0], indices.shape[0]),
        )
        if self.config['training_objective_protocol'] == MANUSCRIPT_TRAINING_PROTOCOL:
            alphas = jnp.power(times, _BRIDGE_ALPHA_POWER)
        else:
            progress = _legacy_bridge_progress(horizon)
            alphas = progress[indices.astype(jnp.int32)]
        masks = (
            indices
            * (jnp.asarray(horizon, dtype=jnp.float32) - indices)
            / jnp.asarray(horizon * horizon, dtype=jnp.float32)
        )
        endpoint_displacements = endpoints - observations
        residuals = self.network.select('bridge')(
            observations,
            endpoint_displacements,
            times,
            params=params,
        )
        displacements = (
            alphas[..., None] * endpoint_displacements[:, None, :]
            + masks[None, :, None] * residuals
        )
        if self.config['training_objective_protocol'] != MANUSCRIPT_TRAINING_PROTOCOL:
            displacements = jnp.where(
                (indices == 0)[None, :, None],
                jnp.zeros_like(displacements),
                displacements,
            )
            displacements = jnp.where(
                (indices == horizon)[None, :, None],
                endpoint_displacements[:, None, :],
                displacements,
            )
        return displacements

    def _construct_bridge(
        self,
        observations: jnp.ndarray,
        endpoints: jnp.ndarray,
        *,
        params: Any | None = None,
    ) -> jnp.ndarray:
        """Construct the full endpoint-pinned bridge for inference."""

        return self._construct_bridge_at_indices(
            observations,
            endpoints,
            jnp.arange(int(self.config['horizon']) + 1),
            params=params,
        )

    @jax.jit
    def construct_bridge(
        self,
        observations: jnp.ndarray,
        endpoints: jnp.ndarray,
    ) -> jnp.ndarray:
        """Construct a deterministic endpoint-pinned absolute-state bridge."""

        squeeze = observations.ndim == 1
        if squeeze:
            observations = observations[None, :]
            endpoints = endpoints[None, :]
        bridge = self._construct_bridge(observations, endpoints)
        return bridge[0] if squeeze else bridge

    # 3. Endpoint-pinned bridge.

    def bridge_loss(
        self,
        batch: dict[str, jnp.ndarray],
        grad_params: Any,
        full_metrics: bool = True,
        prefix: jnp.ndarray | None = None,
        prefix_displacements: jnp.ndarray | None = None,
    ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        """Mean L1 reconstruction for exactly bridge indices 1 through 5."""

        bridge_targets = batch['bridge_targets']
        manuscript = self.config['training_objective_protocol'] == MANUSCRIPT_TRAINING_PROTOCOL
        if manuscript:
            if prefix is None:
                prefix = self._construct_bridge_at_indices(
                    batch['observations'],
                    batch['endpoint_targets'],
                    jnp.arange(1, _ACTION_HORIZON + 1),
                    params=grad_params,
                )
            prefix_errors = jnp.sum(
                jnp.abs(prefix - bridge_targets),
                axis=-1,
            )
            per_sample_loss = prefix_errors.mean(axis=1)
        else:
            if prefix_displacements is None:
                if prefix is None:
                    prefix_displacements = self._construct_bridge_displacements_at_indices(
                        batch['observations'],
                        batch['endpoint_targets'],
                        jnp.arange(1, _ACTION_HORIZON + 1),
                        params=grad_params,
                    )
                else:
                    prefix_displacements = prefix - batch['observations'][:, None, :]
            if prefix is None:
                prefix = batch['observations'][:, None, :] + prefix_displacements
            target_displacements = bridge_targets - batch['observations'][:, None, :]
            prefix_errors = jnp.sum(
                jnp.abs(prefix_displacements - target_displacements),
                axis=-1,
            )
            # The released launcher appended an exact endpoint before its mean.
            if int(self.config['horizon']) > _ACTION_HORIZON:
                per_sample_loss = prefix_errors.sum(axis=1) / float(_ACTION_HORIZON + 1)
            else:
                per_sample_loss = prefix_errors[:, :-1].mean(axis=1)
        (
            path_weights,
            path_energies,
            defects,
            distance_deltas,
            active,
            strength,
        ) = self._dataset_path_weights(
            batch['observations'],
            bridge_targets,
            batch['endpoint_targets'],
            full_metrics=full_metrics,
        )
        if manuscript:
            loss = jnp.sum(path_weights * per_sample_loss) / jnp.maximum(
                jnp.sum(path_weights), _VALUE_EPS
            )
        elif float(self.config['path_weight_beta']) == 0.0:
            if int(self.config['horizon']) > _ACTION_HORIZON:
                endpoint_displacements = (batch['endpoint_targets'] - batch['observations'])[
                    :, None, :
                ]
                legacy_predictions = jnp.concatenate(
                    [prefix_displacements, endpoint_displacements],
                    axis=1,
                )
                legacy_targets = jnp.concatenate(
                    [target_displacements, endpoint_displacements],
                    axis=1,
                )
            else:
                legacy_predictions = prefix_displacements[:, :-1, :]
                legacy_targets = target_displacements[:, :-1, :]
            # Preserve the frozen implementation's nested reduction.  Moving
            # the appended endpoint zero after the state-axis sum changes the
            # XLA reduction tree and the reported float32 scalar.
            loss = jnp.mean(jnp.abs(legacy_predictions - legacy_targets).sum(axis=-1))
        else:
            loss = jnp.sum(path_weights * per_sample_loss) / jnp.maximum(
                jnp.sum(path_weights),
                _VALUE_EPS,
            )
        info = {
            'bridge/loss': loss,
        }
        if full_metrics:
            active_count = jnp.maximum(jnp.sum(active), 1.0)
            weight_sum = jnp.sum(path_weights)
            ess = jnp.square(weight_sum) / jnp.maximum(
                path_weights.shape[0] * jnp.sum(jnp.square(path_weights)),
                _VALUE_EPS,
            )
            info.update(
                {
                    'bridge/prefix_l1': prefix_errors.mean(),
                    'bridge/endpoint_error': jnp.zeros((), dtype=loss.dtype),
                    'bridge/path_weight_strength': strength,
                    'bridge/path_weight_mean': path_weights.mean(),
                    'bridge/path_weight_min': path_weights.min(),
                    'bridge/path_weight_ess_fraction': ess,
                    'bridge/path_energy_mean': path_energies.mean(),
                    'bridge/path_active_fraction': active.mean(),
                    'bridge/path_monotonic_violation_fraction': jnp.sum(
                        (distance_deltas > 0.0) * active
                    )
                    / active_count,
                    'bridge/path_negative_defect_fraction': jnp.sum((defects < 0.0) * active)
                    / active_count,
                }
            )
        return loss, info

    # 4. Inverse dynamics.

    def idm_loss(
        self,
        batch: dict[str, jnp.ndarray],
        grad_params: Any,
        full_metrics: bool = True,
    ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        """Adjacent-transition inverse-dynamics MSE."""

        predicted_actions = self.network.select('idm')(
            batch['observations'],
            batch['next_observations'],
            params=grad_params,
        )
        squared_errors = jnp.square(predicted_actions - batch['actions'])
        if self.config['training_objective_protocol'] == MANUSCRIPT_TRAINING_PROTOCOL:
            loss = jnp.sum(squared_errors, axis=-1).mean()
        else:
            loss = squared_errors.mean()
        info = {
            'idm/loss': loss,
            'idm/squared_l2': loss,
        }
        if full_metrics:
            info['idm/action_mse'] = squared_errors.mean()
        return loss, info

    @partial(jax.jit, static_argnames=('full_metrics',))
    # Joint objective and update.

    def total_loss(
        self,
        batch: dict[str, jnp.ndarray],
        grad_params: Any,
        rng: jax.Array | None = None,
        full_metrics: bool = True,
    ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        """Sum all four fixed-coefficient PathBridger objectives."""

        rng = self.rng if rng is None else rng
        if self.config['training_objective_protocol'] == MANUSCRIPT_TRAINING_PROTOCOL:
            value_loss, value_info = self.value_loss(batch, grad_params, full_metrics)
            endpoint_loss, endpoint_info = self.endpoint_loss(
                batch,
                grad_params,
                rng,
                full_metrics,
            )
            bridge_prefix = self._construct_bridge_at_indices(
                batch['observations'],
                batch['endpoint_targets'],
                jnp.arange(1, _ACTION_HORIZON + 1),
                params=grad_params,
            )
            bridge_loss, bridge_info = self.bridge_loss(
                batch,
                grad_params,
                full_metrics,
                prefix=bridge_prefix,
            )
            if bool(self.config.get('offline_action_free', False)):
                idm_loss = jnp.zeros((), dtype=value_loss.dtype)
                idm_info = {}
            else:
                idm_loss, idm_info = self.idm_loss(batch, grad_params, full_metrics)
            prefix_rng = jax.random.fold_in(rng, _PREFIX_STREAM_ID)
            prefix_loss, prefix_info = self.prefix_distribution_loss(
                batch,
                grad_params,
                prefix_rng,
                full_metrics,
                reference=bridge_prefix,
            )
            weighted_prefix_loss = (
                jnp.asarray(
                    self.config['prefix_loss_weight'],
                    dtype=prefix_loss.dtype,
                )
                * prefix_loss
            )
            loss = value_loss + endpoint_loss + bridge_loss + idm_loss + weighted_prefix_loss
            info = {
                'loss/total': loss,
                **value_info,
                **endpoint_info,
                **bridge_info,
                **idm_info,
                **prefix_info,
            }
            if self.config['prefix_model'] != 'deterministic':
                info['loss/prefix_weighted'] = weighted_prefix_loss
            return loss, info

        horizon = int(self.config['horizon'])
        if horizon > _ACTION_HORIZON:
            bridge_indices = jnp.concatenate(
                [
                    jnp.arange(0, _ACTION_HORIZON + 1),
                    jnp.asarray([horizon]),
                ]
            )
        else:
            bridge_indices = jnp.arange(0, horizon + 1)
        bridge_training_displacements = self._construct_bridge_displacements_at_indices(
            batch['observations'],
            batch['endpoint_targets'],
            bridge_indices,
            params=grad_params,
        )
        bridge_prefix_displacements = bridge_training_displacements[:, 1 : _ACTION_HORIZON + 1, :]
        bridge_prefix = batch['observations'][:, None, :] + bridge_prefix_displacements
        bridge_loss, bridge_info = self.bridge_loss(
            batch,
            grad_params,
            full_metrics,
            prefix=bridge_prefix,
            prefix_displacements=bridge_prefix_displacements,
        )
        endpoint_loss, endpoint_info = self.endpoint_loss(
            batch,
            grad_params,
            rng,
            full_metrics,
        )
        if bool(self.config.get('offline_action_free', False)):
            idm_loss = jnp.zeros((), dtype=bridge_loss.dtype)
            idm_info = {}
        else:
            idm_loss, idm_info = self.idm_loss(batch, grad_params, full_metrics)
        value_loss, value_info = self.value_loss(batch, grad_params, full_metrics)
        prefix_rng = jax.random.fold_in(rng, _PREFIX_STREAM_ID)
        prefix_loss, prefix_info = self.prefix_distribution_loss(
            batch,
            grad_params,
            prefix_rng,
            full_metrics,
            reference=bridge_prefix,
        )
        weighted_prefix_loss = (
            jnp.asarray(self.config['prefix_loss_weight'], dtype=prefix_loss.dtype) * prefix_loss
        )
        dynamics_loss = bridge_loss + endpoint_loss + idm_loss
        loss = value_loss + dynamics_loss + weighted_prefix_loss
        info = {
            'loss/total': loss,
            'loss/dynamics_total': dynamics_loss,
            **value_info,
            **endpoint_info,
            **bridge_info,
            **idm_info,
            **prefix_info,
        }
        if self.config['prefix_model'] != 'deterministic':
            info['loss/prefix_weighted'] = weighted_prefix_loss
        return loss, info

    def _ema_target_value(self, network: TrainState) -> TrainState:
        online_params = network.params['modules_value']
        target_params = network.params['modules_target_value']
        updated_target = jax.tree_util.tree_map(
            lambda online, target: _TARGET_TAU * online + (1.0 - _TARGET_TAU) * target,
            online_params,
            target_params,
        )
        return network.replace(
            params=_replace_module_params(
                network.params,
                'target_value',
                updated_target,
            )
        )

    def update(
        self,
        batch: dict[str, jnp.ndarray],
        full_metrics: bool = True,
    ) -> tuple['PathBridgerAgent', dict[str, jnp.ndarray]]:
        """Update the shared value, endpoint, bridge, prefix, and IDM modules."""

        required = (
            _REQUIRED_STATE_BATCH_KEYS
            if bool(self.config.get('offline_action_free', False))
            else _REQUIRED_BATCH_KEYS
        )
        missing = [key for key in required if key not in batch]
        if missing:
            raise KeyError(f'PathBridger batch is missing keys: {missing}')
        if int(batch['bridge_targets'].shape[1]) != _ACTION_HORIZON:
            raise ValueError(
                f'bridge_targets must have shape [B, 5, D]; got {batch["bridge_targets"].shape}.'
            )
        return self._update_impl(batch, full_metrics=full_metrics)

    @partial(jax.jit, static_argnames=('full_metrics',))
    def _update_impl(
        self,
        batch: dict[str, jnp.ndarray],
        full_metrics: bool = True,
    ) -> tuple['PathBridgerAgent', dict[str, jnp.ndarray]]:
        new_rng, loss_rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(
                batch,
                grad_params,
                rng=loss_rng,
                full_metrics=full_metrics,
            )

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        new_network = self._ema_target_value(new_network)
        return self.replace(rng=new_rng, network=new_network), info

    # Policy sampling.

    def _flow_endpoint_samples(
        self,
        observations: jnp.ndarray,
        goals: jnp.ndarray,
        initial_noise: jnp.ndarray,
        *,
        params: Any | None = None,
    ) -> jnp.ndarray:
        """Integrate the endpoint flow with the fixed eight Euler steps."""

        batch_size, num_candidates, state_dim = initial_noise.shape
        flat_observations = jnp.broadcast_to(
            observations[:, None, :],
            (batch_size, num_candidates, state_dim),
        ).reshape(batch_size * num_candidates, state_dim)
        flat_goals = jnp.broadcast_to(
            goals[:, None, :],
            (batch_size, num_candidates, goals.shape[-1]),
        ).reshape(batch_size * num_candidates, goals.shape[-1])
        initial = initial_noise.reshape(
            batch_size * num_candidates,
            state_dim,
        )
        step_size = jnp.asarray(1.0 / _FLOW_STEPS, dtype=jnp.float32)

        def body(step, displacements):
            times = jnp.full(
                (displacements.shape[0], 1),
                step.astype(jnp.float32) / jnp.asarray(_FLOW_STEPS, dtype=jnp.float32),
            )
            velocities = self.network.select('endpoint')(
                flat_observations,
                flat_goals,
                displacements,
                times,
                params=params,
            )
            return displacements + step_size * velocities

        displacements = jax.lax.fori_loop(
            0,
            _FLOW_STEPS,
            body,
            jnp.asarray(initial, dtype=jnp.float32),
        )
        return displacements.reshape(batch_size, num_candidates, state_dim)

    def _sample_endpoint_candidates(
        self,
        observations: jnp.ndarray,
        goals: jnp.ndarray,
        seed: jax.Array,
        *,
        num_candidates: int,
        temperature: float,
    ) -> jnp.ndarray:
        batch_size, state_dim = observations.shape
        temperature_array = jnp.asarray(temperature, dtype=observations.dtype)

        if self.config['endpoint_distribution'] == 'gaussian':
            means, log_stds = self.network.select('endpoint')(
                observations,
                goals,
            )
            standard_deviations = jnp.exp(log_stds) * temperature_array
            if self.config['training_objective_protocol'] == MANUSCRIPT_TRAINING_PROTOCOL:
                noise = jax.random.normal(
                    seed,
                    (batch_size, num_candidates, state_dim),
                    dtype=observations.dtype,
                )
                displacements = means[:, None, :] + noise * standard_deviations[:, None, :]
            else:
                noise = jax.random.normal(
                    seed,
                    (num_candidates, batch_size, state_dim),
                    dtype=observations.dtype,
                )
                sampled = means[None, :, :] + noise * standard_deviations[None, :, :]
                displacements = jnp.swapaxes(sampled, 0, 1)
        elif self.config['endpoint_distribution'] == 'flow':
            noise = jax.random.normal(
                seed,
                (batch_size, num_candidates, state_dim),
                dtype=observations.dtype,
            )
            displacements = self._flow_endpoint_samples(
                observations,
                goals,
                temperature_array * noise,
            )
        return observations[:, None, :] + displacements

    @partial(
        jax.jit,
        static_argnames=('num_candidates', 'temperature'),
    )
    def _sample_action_chunks_compiled(
        self,
        observations: jnp.ndarray,
        goals: jnp.ndarray,
        seed: jax.Array | None = None,
        num_candidates: int | None = None,
        temperature: float | None = None,
    ) -> jnp.ndarray:
        """Select an endpoint first, sample its prefix, then decode actions."""

        if bool(self.config.get('offline_action_free', False)):
            raise ValueError(
                'An action-free PBF checkpoint has no IDM. Use sample_state_prefix '
                'with a separately trained online IDM.'
            )

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

        batch_size, state_dim = observations.shape
        manuscript_gaussian = (
            self.config['endpoint_distribution'] == 'gaussian'
            and self.config['training_objective_protocol'] == MANUSCRIPT_TRAINING_PROTOCOL
        )
        if manuscript_gaussian and num_candidates == 1 and temperature == 0.0:
            # The sole deterministic candidate always wins. Avoid random
            # sampling and the otherwise unused value selector.
            mean_displacements, _ = self.network.select('endpoint')(
                observations,
                goals,
            )
            selected_endpoints = observations + mean_displacements
        else:
            candidates = self._sample_endpoint_candidates(
                observations,
                goals,
                seed,
                num_candidates=num_candidates,
                temperature=temperature,
            )
            if manuscript_gaussian and num_candidates == 1:
                # Sampling can be stochastic, but selection cannot be.
                selected_endpoints = candidates[:, 0, :]
            else:
                repeated_observations = jnp.broadcast_to(
                    observations[:, None, :],
                    candidates.shape,
                )
                repeated_goals = jnp.broadcast_to(
                    goals[:, None, :],
                    (batch_size, num_candidates, goals.shape[-1]),
                )
                flat_observations = repeated_observations.reshape(
                    batch_size * num_candidates,
                    state_dim,
                )
                flat_candidates = candidates.reshape(
                    batch_size * num_candidates,
                    state_dim,
                )
                flat_goals = repeated_goals.reshape(
                    batch_size * num_candidates,
                    goals.shape[-1],
                )
                if manuscript_gaussian:
                    value_logits = self.network.select('value')(
                        jnp.concatenate([flat_observations, flat_candidates], axis=0),
                        jnp.concatenate([flat_candidates, flat_goals], axis=0),
                    )
                    values_to_endpoint, values_to_goal = jnp.split(
                        jax.nn.sigmoid(value_logits), 2, axis=0
                    )
                    scores = values_to_endpoint * values_to_goal
                else:
                    # Preserve reported-run call boundaries and ratio scoring.
                    values_to_endpoint = jax.nn.sigmoid(
                        self.network.select('value')(flat_observations, flat_candidates)
                    )
                    values_to_goal = jax.nn.sigmoid(
                        self.network.select('value')(flat_candidates, flat_goals)
                    )
                    values_direct = jax.nn.sigmoid(
                        self.network.select('value')(flat_observations, flat_goals)
                    )
                    scores = (
                        values_to_endpoint
                        * values_to_goal
                        / (values_direct + _SUBGOAL_VALUE_RATIO_EPS)
                    )
                best_indices = jnp.argmax(scores.reshape(batch_size, num_candidates), axis=1)
                selected_endpoints = jnp.take_along_axis(
                    candidates,
                    best_indices[:, None, None],
                    axis=1,
                )[:, 0, :]

        prefix_seed = jax.random.fold_in(seed, _PREFIX_STREAM_ID)
        prefix = self._select_prefix(observations, selected_endpoints, prefix_seed)

        current_states = prefix[:, :-1, :].reshape(
            batch_size * _ACTION_HORIZON,
            state_dim,
        )
        next_states = prefix[:, 1:, :].reshape(
            batch_size * _ACTION_HORIZON,
            state_dim,
        )
        actions = self.network.select('idm')(current_states, next_states)
        actions = actions.reshape(batch_size, _ACTION_HORIZON, -1)
        return actions[0] if squeeze else actions

    def sample_action_chunks(
        self,
        observations: jnp.ndarray,
        goals: jnp.ndarray,
        seed: jax.Array | None = None,
        num_candidates: int | None = None,
        temperature: float | None = None,
    ) -> jnp.ndarray:
        """Run the optimized kernel or an explicit legacy-stage shim."""

        if bool(self.config.get('legacy_inference_staging', False)):
            return self._sample_action_chunks_legacy_staged(
                observations,
                goals,
                seed,
                num_candidates=num_candidates,
                temperature=temperature,
            )
        return self._sample_action_chunks_compiled(
            observations,
            goals,
            seed,
            num_candidates=num_candidates,
            temperature=temperature,
        )

    # Initialization.

    @classmethod
    def create(
        cls,
        seed: int,
        ex_observations: jnp.ndarray,
        ex_actions: jnp.ndarray | None,
        config: dict[str, Any],
        state_scale: jnp.ndarray | None = None,
    ) -> 'PathBridgerAgent':
        """Initialize the shared PBF/PBG modules and optimizer."""

        config = dict(config)
        config.setdefault('offline_action_free', False)
        config.setdefault(
            'training_objective_protocol',
            REPORTED_RUN_LEGACY_TRAINING_PROTOCOL,
        )
        config['training_objective_protocol'] = normalize_training_protocol(
            config['training_objective_protocol']
        )
        endpoint_distribution = str(config['endpoint_distribution']).lower()
        if endpoint_distribution not in ('flow', 'gaussian'):
            raise ValueError(
                "endpoint_distribution must be 'flow' or 'gaussian', "
                f'got {endpoint_distribution!r}.'
            )
        config['endpoint_distribution'] = endpoint_distribution

        prefix_model = str(config['prefix_model']).lower()
        if prefix_model not in ('deterministic', 'low_rank_gaussian', 'joint_flow'):
            raise ValueError(
                "prefix_model must be 'deterministic', 'low_rank_gaussian', or "
                f"'joint_flow', got {prefix_model!r}."
            )
        config['prefix_model'] = prefix_model
        if float(config['prefix_loss_weight']) < 0.0:
            raise ValueError('prefix_loss_weight cannot be negative.')
        if int(config['prefix_rank']) < 1:
            raise ValueError('prefix_rank must be at least one.')
        if float(config['prefix_sigma_floor']) <= 0.0:
            raise ValueError('prefix_sigma_floor must be positive.')
        if float(config['prefix_scale_floor']) <= 0.0:
            raise ValueError('prefix_scale_floor must be positive.')
        if int(config['prefix_flow_steps']) < 1:
            raise ValueError('prefix_flow_steps must be at least one.')
        prefix_selection = str(config['eval_prefix_selection']).lower()
        if prefix_selection not in ('sample_one', 'transv_chain'):
            raise ValueError("eval_prefix_selection must be 'sample_one' or 'transv_chain'.")
        config['eval_prefix_selection'] = prefix_selection
        if int(config['eval_num_prefix_samples']) < 1:
            raise ValueError('eval_num_prefix_samples must be at least one.')
        if float(config['eval_prefix_temperature']) < 0.0:
            raise ValueError('eval_prefix_temperature must be non-negative.')

        horizon = int(config['horizon'])
        if horizon < _ACTION_HORIZON:
            raise ValueError(f'horizon must be at least {_ACTION_HORIZON}, got {horizon}.')
        discount = float(config['discount'])
        if not 0.0 < discount < 1.0:
            raise ValueError(f'discount must be in (0, 1), got {discount}.')
        if float(config['path_weight_beta']) < 0.0:
            raise ValueError('path_weight_beta cannot be negative.')
        if not 0.0 < float(config['path_weight_min']) <= 1.0:
            raise ValueError('path_weight_min must lie in (0, 1].')
        if int(config['path_weight_warmup']) < 0:
            raise ValueError('path_weight_warmup cannot be negative.')
        if int(config['path_weight_ramp']) < 0:
            raise ValueError('path_weight_ramp cannot be negative.')
        if float(config['path_distance_cap_multiplier']) <= 0.0:
            raise ValueError('path_distance_cap_multiplier must be positive.')
        if int(config['eval_num_candidates']) < 1:
            raise ValueError('eval_num_candidates must be at least one.')
        if float(config['eval_temperature']) < 0.0:
            raise ValueError('eval_temperature must be non-negative.')

        observations = jnp.asarray(ex_observations, dtype=jnp.float32)
        action_free = bool(config['offline_action_free'])
        if observations.ndim != 2:
            raise ValueError('ex_observations must be a batched rank-2 array.')
        actions = None if ex_actions is None else jnp.asarray(ex_actions, dtype=jnp.float32)
        if not action_free:
            if actions is None or actions.ndim != 2:
                raise ValueError(
                    'Action-conditioned PathBridger requires batched rank-2 ex_actions.'
                )
            if observations.shape[0] != actions.shape[0]:
                raise ValueError('ex_observations and ex_actions must have equal batch sizes.')
        state_dim = int(observations.shape[-1])
        if state_scale is None:
            state_scale_array = jnp.ones((state_dim,), dtype=jnp.float32)
        else:
            state_scale_array = jnp.asarray(state_scale, dtype=jnp.float32)
        if state_scale_array.shape != (state_dim,):
            raise ValueError(
                f'state_scale must have shape ({state_dim},), got {state_scale_array.shape}.'
            )
        if not bool(jnp.all(jnp.isfinite(state_scale_array))):
            raise ValueError('state_scale must contain only finite values.')
        if not bool(jnp.all(state_scale_array > 0.0)):
            raise ValueError('state_scale values must be positive.')
        env_name = str(config['env_name'])
        phi_goal_obs_indices = infer_phi_goal_obs_indices(env_name, state_dim)
        assert_phi_goal_obs_indices(
            state_dim,
            'phi',
            phi_goal_obs_indices,
            where='PathBridgerAgent.create (endpoint goal representation)',
            env_name=env_name,
        )
        value_def = ScalarTransitiveValue()
        target_value_def = ScalarTransitiveValue()
        if endpoint_distribution == 'gaussian':
            endpoint_def = GaussianEndpointProposer(
                state_dim=state_dim,
                env_name=env_name,
                phi_goal_obs_indices=phi_goal_obs_indices,
            )
            endpoint_args = (observations, observations)
        else:
            endpoint_def = FlowEndpointProposer(
                state_dim=state_dim,
                env_name=env_name,
                phi_goal_obs_indices=phi_goal_obs_indices,
            )
            endpoint_args = (
                observations,
                observations,
                jnp.zeros_like(observations),
                jnp.zeros((observations.shape[0], 1), dtype=jnp.float32),
            )
        bridge_def = BridgeResidual(state_dim=state_dim)
        prefix_def, prefix_args = create_prefix_module(config, observations)

        bridge_times = jnp.broadcast_to(
            jnp.linspace(
                0.0,
                1.0,
                horizon + 1,
                dtype=jnp.float32,
            )[None, :],
            (observations.shape[0], horizon + 1),
        )
        network_info = {
            'value': (value_def, (observations, observations)),
            'target_value': (
                target_value_def,
                (observations, observations),
            ),
            'bridge': (
                bridge_def,
                (observations, jnp.zeros_like(observations), bridge_times),
            ),
        }
        network_info['endpoint'] = (endpoint_def, endpoint_args)
        if not action_free:
            idm_def = InverseDynamics(action_dim=int(actions.shape[-1]))
            network_info['idm'] = (idm_def, (observations, observations))
        if prefix_def is not None:
            network_info['prefix'] = (prefix_def, prefix_args)
        network_def = ModuleDict(
            {name: definition for name, (definition, _) in network_info.items()}
        )
        network_args = {name: arguments for name, (_, arguments) in network_info.items()}

        rng = jax.random.PRNGKey(int(seed))
        rng, init_rng = jax.random.split(rng)
        network_params = network_def.init(
            init_rng,
            **network_args,
        )['params']
        network_params = _replace_module_params(
            network_params,
            'target_value',
            network_params['modules_value'],
        )
        network_tx = optax.adam(_LEARNING_RATE)
        network = TrainState.create(
            network_def,
            network_params,
            tx=network_tx,
        )
        return cls(
            rng=rng,
            network=network,
            state_scale=state_scale_array,
            config=flax.core.FrozenDict(config),
        )


__all__ = [
    'BridgeResidual',
    'FlowEndpointProposer',
    'GaussianEndpointProposer',
    'InverseDynamics',
    'PathBridgerAgent',
    'ScalarTransitiveValue',
    'get_config',
]
