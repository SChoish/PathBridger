"""State-based multitask OGBench evaluation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

import jax
import numpy as np

DEFAULT_TASK_IDS = (1, 2, 3, 4, 5)
ACTION_CHUNK_HORIZON = 5
HISTORICAL_EVALUATION_PROTOCOL = 'historical-v1'
DETERMINISTIC_EVALUATION_PROTOCOL = 'deterministic-v1'
EVALUATION_PROTOCOLS = (
    HISTORICAL_EVALUATION_PROTOCOL,
    DETERMINISTIC_EVALUATION_PROTOCOL,
)


def normalize_evaluation_protocol(protocol: str | None) -> str:
    """Resolve versionless/short names without changing legacy behavior."""

    aliases = {
        None: HISTORICAL_EVALUATION_PROTOCOL,
        '': HISTORICAL_EVALUATION_PROTOCOL,
        'historical': HISTORICAL_EVALUATION_PROTOCOL,
        HISTORICAL_EVALUATION_PROTOCOL: HISTORICAL_EVALUATION_PROTOCOL,
        'deterministic': DETERMINISTIC_EVALUATION_PROTOCOL,
        DETERMINISTIC_EVALUATION_PROTOCOL: DETERMINISTIC_EVALUATION_PROTOCOL,
    }
    try:
        return aliases[protocol]
    except KeyError as error:
        raise ValueError(
            f'Unknown evaluation protocol {protocol!r}; expected one of {EVALUATION_PROTOCOLS}.'
        ) from error


def evaluation_protocol_metadata(
    env: Any,
    *,
    protocol: str | None,
    task_ids: Sequence[int],
    episodes_per_task: int,
    num_candidates: int,
    temperature: float,
    seed: int,
) -> dict[str, Any]:
    """Return the effective, content-addressed evaluation contract."""

    protocol = normalize_evaluation_protocol(protocol)
    task_ids = tuple(int(task_id) for task_id in task_ids)
    deterministic = protocol == DETERMINISTIC_EVALUATION_PROTOCOL
    settings = {
        'protocol_version': protocol,
        'task_ids': list(task_ids),
        'task_episode_order': 'episode-major-round-robin',
        'episodes_per_task': int(episodes_per_task),
        'total_episodes': len(task_ids) * int(episodes_per_task),
        'max_episode_steps': _max_episode_steps(env),
        'action_chunk_horizon': ACTION_CHUNK_HORIZON,
        'num_candidates': int(num_candidates),
        'temperature': float(temperature),
        'policy_jax_seed': int(seed),
        'environment_reset': (
            'seed=seed*100000+task_id*1000+episode,options' if deterministic else 'options-only'
        ),
        'action_space_seed': (
            'stable evaluation action-space instance seeded with seed*100000+task_id*1000+episode'
            if deterministic
            else None
        ),
        'legacy_numpy_reset_seed': (
            'temporarily seed global NumPy for reset, then restore caller state'
            if deterministic
            else None
        ),
        'termination': 'terminated-or-truncated-or-max_episode_steps',
        'success': 'any info[success] within episode',
        'task_denominator': int(episodes_per_task),
        'overall_aggregation': 'unweighted-mean-of-task-success-rates',
    }
    canonical = json.dumps(
        settings, sort_keys=True, separators=(',', ':'), ensure_ascii=True
    ).encode('utf-8')
    return {
        **settings,
        'settings_sha256': hashlib.sha256(canonical).hexdigest(),
    }


def _max_episode_steps(env: Any) -> int:
    spec = getattr(env, 'spec', None)
    max_steps = getattr(spec, 'max_episode_steps', None)
    if max_steps is None:
        max_steps = getattr(env, '_max_episode_steps', None)
    if max_steps is None:
        raise ValueError('Evaluation environment must expose `spec.max_episode_steps`.')
    max_steps = int(max_steps)
    if max_steps < 1:
        raise ValueError(f'Environment max episode length must be positive, got {max_steps}.')
    return max_steps


def _info_success(info: Any) -> bool:
    if not isinstance(info, Mapping):
        return False
    value = np.asarray(info.get('success', False))
    return bool(np.any(value))


def _array_sha256(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode('ascii'))
    digest.update(str(array.shape).encode('ascii'))
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


def _stable_action_space(env: Any) -> Any:
    """Return the action-space instance that reset-time sampling will use.

    Most Gymnasium environments own one action-space object. OGBench's
    manipulation environments instead construct a fresh ``Box`` on every
    property access and also call ``self.action_space.sample()`` inside reset.
    Seeding the wrapper-visible temporary object therefore has no effect on
    those reset-time samples. For deterministic evaluation only, install a
    per-instance subclass whose property returns one cached space. This does
    not alter the shared OGBench class or simulator state.
    """

    unwrapped = getattr(env, 'unwrapped', env)
    cached = getattr(unwrapped, '_pathbridger_evaluation_action_space', None)
    if cached is not None:
        return cached

    first = unwrapped.action_space
    if unwrapped.action_space is first:
        return first

    original_class = type(unwrapped)
    stable_class = type(
        f'_PathBridgerStableActionSpace_{original_class.__name__}_{id(unwrapped)}',
        (original_class,),
        {
            '__module__': __name__,
            'action_space': property(
                lambda instance: instance._pathbridger_evaluation_action_space
            ),
        },
    )
    try:
        unwrapped._pathbridger_evaluation_action_space = first
        unwrapped.__class__ = stable_class
    except (AttributeError, TypeError) as error:
        raise RuntimeError(
            'deterministic-v1 cannot seed this environment action space: '
            'the environment returns a new space per access and cannot be '
            'adapted per instance.'
        ) from error
    if unwrapped.action_space is not first or env.action_space is not first:
        raise RuntimeError('deterministic-v1 failed to install a stable action-space instance.')
    return first


def _deterministic_reset(
    env: Any,
    *,
    episode_seed: int,
    reset_options: dict[str, Any],
) -> tuple[Any, Any]:
    """Apply every deterministic reset seed without leaking global RNG state."""

    action_space = _stable_action_space(env)
    action_space.seed(episode_seed)
    numpy_state = np.random.get_state()
    try:
        # Some supported OGBench resets still use legacy global ``np.random``
        # for goal noise even though Gymnasium's ``reset(seed=...)`` seeds the
        # environment-owned Generator. Keep that use deterministic while
        # leaving the caller's global stream untouched.
        np.random.seed(episode_seed)
        return env.reset(seed=episode_seed, options=reset_options)
    finally:
        np.random.set_state(numpy_state)


def _sample_action_chunk(
    agent: Any,
    observation: np.ndarray,
    goal: np.ndarray,
    *,
    num_candidates: int,
    temperature: float,
    seed,
) -> np.ndarray:
    action_chunks = agent.sample_action_chunks(
        observations=observation.reshape(1, -1),
        goals=goal.reshape(1, -1),
        num_candidates=num_candidates,
        temperature=temperature,
        seed=seed,
    )
    action_chunks = np.asarray(jax.device_get(action_chunks), dtype=np.float32)
    if action_chunks.ndim != 3 or action_chunks.shape[0] != 1:
        raise ValueError(
            'agent.sample_action_chunks must return shape [1, 5, action_dim], '
            f'got {action_chunks.shape}.'
        )
    action_chunk = np.squeeze(action_chunks, axis=0)
    if action_chunk.shape[0] != ACTION_CHUNK_HORIZON:
        raise ValueError(
            f'PathBridger executes {ACTION_CHUNK_HORIZON}-step action chunks, got shape {action_chunk.shape}.'
        )
    return action_chunk


def _evaluate_episode(
    agent: Any,
    env: Any,
    observation: np.ndarray,
    goal: np.ndarray,
    *,
    action_low: np.ndarray,
    action_high: np.ndarray,
    max_episode_steps: int,
    num_candidates: int,
    temperature: float,
    rng,
) -> tuple[bool, Any, int]:
    observation = np.asarray(observation, dtype=np.float32).reshape(-1)
    goal = np.asarray(goal, dtype=np.float32).reshape(-1)
    success = False
    terminated = False
    truncated = False
    step = 0

    while step < max_episode_steps and not (terminated or truncated):
        rng, sample_seed = jax.random.split(rng)
        action_chunk = _sample_action_chunk(
            agent,
            observation,
            goal,
            num_candidates=num_candidates,
            temperature=temperature,
            seed=sample_seed,
        )

        for action in action_chunk:
            if step >= max_episode_steps or terminated or truncated:
                break
            action = np.clip(action, action_low, action_high)
            observation, _, terminated, truncated, info = env.step(action)
            observation = np.asarray(observation, dtype=np.float32).reshape(-1)
            success = success or _info_success(info)
            terminated = bool(terminated)
            truncated = bool(truncated)
            step += 1

    return success, rng, step


def evaluate(
    agent: Any,
    env: Any,
    *,
    task_ids: Sequence[int] = DEFAULT_TASK_IDS,
    episodes_per_task: int = 10,
    num_candidates: int = 1,
    temperature: float = 1.0,
    seed: int = 0,
    protocol: str | None = None,
    episode_records: list[dict[str, Any]] | None = None,
) -> dict[str, float | int]:
    """Evaluate PathBridger on OGBench tasks using five-step IDM chunks.

    Each replanning call receives batched state and goal arrays and a fresh JAX
    random key. An episode succeeds when the environment reports
    ``info['success']`` on any step.
    """
    protocol = normalize_evaluation_protocol(protocol)
    task_ids = tuple(int(task_id) for task_id in task_ids)
    if not task_ids:
        raise ValueError('task_ids must contain at least one task.')
    if int(episodes_per_task) < 1:
        raise ValueError('episodes_per_task must be at least 1.')
    if int(num_candidates) < 1:
        raise ValueError('num_candidates must be at least 1.')
    if float(temperature) < 0:
        raise ValueError('temperature must be non-negative.')

    action_low = np.asarray(env.action_space.low, dtype=np.float32)
    action_high = np.asarray(env.action_space.high, dtype=np.float32)
    max_episode_steps = _max_episode_steps(env)
    rng = jax.random.PRNGKey(int(seed))
    metrics: dict[str, float | int] = {}
    task_successes: dict[int, list[float]] = {task_id: [] for task_id in task_ids}

    # Round-robin tasks within each episode index (1→…→T, repeat).
    for episode in range(int(episodes_per_task)):
        for task_id in task_ids:
            reset_options = {'task_id': task_id, 'render_goal': False}
            episode_seed = None
            if protocol == DETERMINISTIC_EVALUATION_PROTOCOL:
                episode_seed = int(seed) * 100_000 + task_id * 1_000 + episode
                observation, info = _deterministic_reset(
                    env,
                    episode_seed=episode_seed,
                    reset_options=reset_options,
                )
            else:
                observation, info = env.reset(options=reset_options)
            if not isinstance(info, Mapping) or 'goal' not in info:
                raise RuntimeError(
                    f'Environment reset for task {task_id} did not return info["goal"].'
                )
            initial_observation = np.asarray(observation, dtype=np.float32)
            initial_goal = np.asarray(info['goal'], dtype=np.float32)
            success, rng, steps_executed = _evaluate_episode(
                agent,
                env,
                observation,
                info['goal'],
                action_low=action_low,
                action_high=action_high,
                max_episode_steps=max_episode_steps,
                num_candidates=int(num_candidates),
                temperature=float(temperature),
                rng=rng,
            )
            task_successes[task_id].append(float(success))
            if episode_records is not None:
                episode_records.append(
                    {
                        'protocol_version': protocol,
                        'episode_index': int(episode),
                        'task_id': int(task_id),
                        'environment_seed': episode_seed,
                        'initial_observation_sha256': _array_sha256(initial_observation),
                        'goal_sha256': _array_sha256(initial_goal),
                        'steps_executed': int(steps_executed),
                        'success': bool(success),
                    }
                )

    task_success_rates = []
    for task_id in task_ids:
        task_success = float(np.mean(task_successes[task_id]))
        metrics[f'task_{task_id}_success'] = task_success
        task_success_rates.append(task_success)

    metrics['overall_success'] = float(np.mean(task_success_rates))
    metrics['num_tasks'] = len(task_ids)
    metrics['episodes_per_task'] = int(episodes_per_task)
    return metrics


__all__ = [
    'ACTION_CHUNK_HORIZON',
    'DEFAULT_TASK_IDS',
    'DETERMINISTIC_EVALUATION_PROTOCOL',
    'EVALUATION_PROTOCOLS',
    'HISTORICAL_EVALUATION_PROTOCOL',
    'evaluate',
    'evaluation_protocol_metadata',
    'normalize_evaluation_protocol',
]
