"""Chunked env eval rollouts: success is decided **only** by ``info['success']`` from the env.

No user-defined distance tolerance is consulted here; the env is the single source of truth for
whether an evaluation episode succeeded.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from utils.ogbench_eval_helpers import append_ogbench_render, update_episode_env_success


def execute_action_chunk_eval(
    env: Any,
    obs: np.ndarray,
    action_chunk: np.ndarray,
    *,
    low: np.ndarray,
    high: np.ndarray,
    render_buf: list[np.ndarray] | None = None,
    goal_frame: np.ndarray | None = None,
    should_render: bool = False,
    video_frame_skip: int = 4,
    step_counter: list[int] | None = None,
) -> tuple[np.ndarray, bool, bool, bool]:
    """Advance env for one action chunk; stop stepping only on ``terminated`` or ``truncated``.

    Returns ``(next_obs, saw_env_success, terminated, truncated)``. ``saw_env_success`` is
    ``info['success']`` aggregated across the chunk (env's own judgement, no extra tolerance).
    """
    saw_env_success = False
    terminated = False
    truncated = False
    for action in np.asarray(action_chunk, dtype=np.float32):
        if terminated or truncated:
            break
        clipped = np.clip(action, low, high)
        step_ix = int(step_counter[0]) if step_counter is not None else 0
        _ob, _reward, term, trunc, info = env.step(clipped)
        obs = np.asarray(_ob, dtype=np.float32).reshape(-1)
        terminated = bool(term)
        truncated = bool(trunc)
        saw_env_success = update_episode_env_success(saw_env_success, info)
        if step_counter is not None and render_buf is not None:
            done = bool(terminated or truncated)
            append_ogbench_render(
                render_buf,
                env,
                goal_frame,
                should_render=bool(should_render),
                step=step_ix,
                done=done,
                video_frame_skip=int(video_frame_skip),
            )
            step_counter[0] = step_ix + 1
    return obs, saw_env_success, terminated, truncated


def rollout_chunked_eval_episode(
    env: Any,
    obs0: np.ndarray,
    goal0: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    max_chunks: int,
    *,
    sample_action_chunk: Callable[[np.ndarray, np.ndarray], np.ndarray],
    render_buf: list[np.ndarray] | None = None,
    goal_frame: np.ndarray | None = None,
    should_render: bool = False,
    video_frame_skip: int = 4,
) -> bool:
    """Replanning rollout until ``terminated``/``truncated`` or chunk budget.

    Returns ``True`` iff the env reported ``info['success']`` at any step.
    """
    obs = np.asarray(obs0, dtype=np.float32).reshape(-1)
    goal = np.asarray(goal0, dtype=np.float32).reshape(-1)
    step_counter = [0] if (should_render and render_buf is not None) else None
    cum_env = False
    terminated = False
    truncated = False
    for _ in range(max(1, int(max_chunks))):
        if terminated or truncated:
            break
        chunk = sample_action_chunk(obs, goal)
        obs, saw_e, term, trunc = execute_action_chunk_eval(
            env,
            obs,
            chunk,
            low=low,
            high=high,
            render_buf=render_buf,
            goal_frame=goal_frame,
            should_render=bool(should_render and render_buf is not None),
            video_frame_skip=video_frame_skip,
            step_counter=step_counter,
        )
        cum_env = cum_env or saw_e
        terminated = terminated or term
        truncated = truncated or trunc
    return bool(cum_env)
