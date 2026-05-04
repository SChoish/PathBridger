"""Regression tests for OGBench-aligned eval helpers and chunked rollout.

Env eval success is decided **only** by ``info['success']`` from the env; no
user-defined distance tolerance is consulted by the eval runner.
"""

from __future__ import annotations

import numpy as np
from gymnasium.spaces import Box

from utils.ogbench_eval_helpers import append_ogbench_render, info_success, update_episode_env_success
from utils.ogbench_eval_rollout import rollout_chunked_eval_episode


def test_info_success_numpy_scalar():
    assert info_success({'success': np.array(1.0)}) is True
    assert info_success({'success': np.array(0.0)}) is False


def test_update_episode_success_or_mid_episode():
    assert update_episode_env_success(False, {'success': True}) is True
    assert update_episode_env_success(True, {'success': False}) is True


class _FakeEnvA:
    """A: success True on step 1 only; terminates at step 3."""

    action_space = Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

    def __init__(self) -> None:
        self.t = 0
        self._obs = np.zeros((2,), dtype=np.float32)
        self.goal = np.zeros((2,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        self.t = 0
        return self._obs.copy(), {'goal': self.goal.copy()}

    def render(self):
        return np.zeros((2, 2, 3), dtype=np.uint8)

    def step(self, action):
        self.t += 1
        succ = self.t == 1
        term = self.t >= 3
        return self._obs.copy(), 0.0, bool(term), False, {'success': succ}


def test_rollout_any_step_success_persists_after_false():
    env = _FakeEnvA()
    low = np.asarray(env.action_space.low, dtype=np.float32).reshape(-1)
    high = np.asarray(env.action_space.high, dtype=np.float32).reshape(-1)
    obs, info = env.reset(options=dict(task_id=1, render_goal=False))
    goal = np.asarray(info['goal'], dtype=np.float32).reshape(-1)

    def chunk(_o, _g):
        return np.ones((1, 1), dtype=np.float32)

    ok_env = rollout_chunked_eval_episode(
        env,
        obs,
        goal,
        low,
        high,
        max_chunks=10,
        sample_action_chunk=chunk,
    )
    assert ok_env is True


class _FakeEnvB:
    """B: obs moves close to goal but env never reports info['success']; ends on truncation."""

    action_space = Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

    def __init__(self) -> None:
        self.t = 0
        self.goal = np.array([0.0, 0.0], dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        self.t = 0
        obs = np.array([10.0, 10.0], dtype=np.float32)
        return obs, {'goal': self.goal.copy()}

    def render(self):
        return np.zeros((2, 2, 3), dtype=np.uint8)

    def step(self, action):
        self.t += 1
        obs = np.array([0.0, 0.0], dtype=np.float32)
        return obs, 0.0, False, self.t >= 2, {'success': False}


def test_rollout_without_env_success_is_failure_regardless_of_distance():
    env = _FakeEnvB()
    low = np.asarray(env.action_space.low, dtype=np.float32).reshape(-1)
    high = np.asarray(env.action_space.high, dtype=np.float32).reshape(-1)
    obs, info = env.reset(options=dict(task_id=1, render_goal=False))
    goal = np.asarray(info['goal'], dtype=np.float32).reshape(-1)

    def chunk(_o, _g):
        return np.ones((1, 1), dtype=np.float32)

    ok_env = rollout_chunked_eval_episode(
        env,
        obs,
        goal,
        low,
        high,
        max_chunks=10,
        sample_action_chunk=chunk,
    )
    assert ok_env is False


def test_video_episode_stats_exclusion_pattern():
    num_eval, num_video = 2, 1
    stats: list[int] = []
    for ep_ix in range(num_eval + num_video):
        if ep_ix < num_eval:
            stats.append(ep_ix)
    assert stats == [0, 1]


def test_append_render_doubles_height_with_goal_frame():
    class E:
        def render(self):
            return np.full((4, 4, 3), 200, dtype=np.uint8)

    buf: list[np.ndarray] = []
    gf = np.full((4, 4, 3), 100, dtype=np.uint8)
    append_ogbench_render(buf, E(), gf, should_render=True, step=0, done=False, video_frame_skip=1)
    assert len(buf) == 1
    assert buf[0].shape[0] == 8
    assert buf[0].dtype == np.uint8


if __name__ == '__main__':
    import inspect
    import sys

    failures = []
    mod = sys.modules[__name__]
    for name, fn in inspect.getmembers(mod, inspect.isfunction):
        if not name.startswith('test_'):
            continue
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            failures.append((name, repr(e)))
            print(f'  FAIL {name}: {e!r}')
        else:
            print(f'  PASS {name}')
    if failures:
        raise SystemExit(f'{len(failures)} test(s) failed.')
    print('All tests passed.')
