"""G1 diagnostics: fixed source program, no Q selection, no subsequent replanning."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np


def recorded_state_errors(agent, batch):
    """Source actions form the parent; observations are used only by the executor."""
    root = agent.encode_actions(batch['actions'])
    leaves = agent.apply(agent.state.params, 'leaves', root)
    predictions = []
    for index in range(4):
        predictions.append(
            np.asarray(
                agent.decode(
                    jnp.asarray(batch['states'][:, index * 5]),
                    leaves[:, index],
                )
            )
        )
    predicted = np.stack(predictions, axis=1)
    target = np.asarray(batch['actions']).reshape((len(root), 4, 5, agent.action_dim))
    delta = predicted - target
    return {
        'recorded_action_mse_by_chunk': np.mean(delta**2, axis=(0, 2, 3)).tolist(),
        'recorded_action_l2_by_chunk': np.linalg.norm(delta.reshape((len(root), 4, -1)), axis=-1)
        .mean(0)
        .tolist(),
        'root': np.asarray(root),
        'predicted_actions': predicted,
        'status': 'descriptive_only_not_gate_pass',
    }


def execute_fixed_program(
    env,
    agent,
    root,
    initial_observation,
    *,
    direct_child_codes=None,
):
    """Run exactly one chosen program on an already restored environment.

    Restore the same full simulator state for each paired arm before calling.
    Merely sharing a reset seed is not a guarantee of paired simulator state.
    direct_child_codes, when supplied, are an explicitly oracle reference.
    """
    observation = np.asarray(initial_observation)
    if observation.shape != (agent.observation_dim,):
        raise ValueError('initial observation dimension mismatch')
    if direct_child_codes is not None and np.shape(direct_child_codes) != (
        4,
        agent.config['latent_dim'],
    ):
        raise ValueError('four reference child codes required')
    states, actions, boundary_steps = [observation.copy()], [], [0]
    success_step = None
    terminated = truncated = False
    for chunk in range(4):
        if direct_child_codes is None:
            sequence = agent.decode_program_chunk(
                jnp.asarray(observation[None]), jnp.asarray(root).reshape((1, -1)), chunk
            )
        else:
            sequence = agent.decode(
                jnp.asarray(observation[None]),
                jnp.asarray(direct_child_codes[chunk : chunk + 1]),
            )
        for action in np.asarray(sequence)[0]:
            action = np.clip(action, env.action_space.low, env.action_space.high)
            observation, _, terminated, truncated, info = env.step(action)
            if not np.isfinite(action).all() or not np.isfinite(observation).all():
                raise FloatingPointError('nonfinite fixed-program execution')
            actions.append(np.asarray(action).copy())
            states.append(np.asarray(observation).copy())
            if bool(info.get('success', False)):
                success_step = len(actions)
            if success_step is not None or terminated or truncated:
                break
        boundary_steps.append(len(actions))
        if success_step is not None or terminated or truncated:
            break
    return {
        'states': np.asarray(states),
        'actions': np.asarray(actions),
        'boundary_steps': boundary_steps,
        'first_hit_step': success_step,
        'terminated': bool(terminated),
        'truncated': bool(truncated),
        'programs_selected': 1,
        'replanning_calls': 0,
        'reference': 'direct_child_oracle' if direct_child_codes is not None else 'encoded_actions',
    }
