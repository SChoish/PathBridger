"""Twenty-step commitment with a fresh decoder observation every five steps."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


class CommittedActionPolicy:
    def __init__(self, agent, *, seed, action_low, action_high):
        self.agent = agent
        self.key = jax.random.PRNGKey(seed)
        self.low, self.high = np.asarray(action_low), np.asarray(action_high)
        if (
            self.low.shape != (agent.action_dim,)
            or self.high.shape != self.low.shape
            or not np.isfinite(self.low).all()
            or not np.isfinite(self.high).all()
            or np.any(self.low >= self.high)
        ):
            raise ValueError('finite per-coordinate environment action bounds required')
        self.reset()

    def reset(self):
        self.root, self.goal = None, None
        self.cursor, self.plan_count = 20, 0
        self.actions = None

    def action(self, observation, goal):
        observation, goal = np.asarray(observation), np.asarray(goal)
        if observation.shape != (self.agent.observation_dim,) or goal.shape != observation.shape:
            raise ValueError('full observation and full goal observation vectors required')
        if not np.isfinite(observation).all() or not np.isfinite(goal).all():
            raise ValueError('nonfinite policy input')
        if self.goal is not None and not np.array_equal(goal, self.goal):
            raise ValueError('goal changed inside an episode; reset the committed policy')
        if self.cursor == 20:
            self.key, key = jax.random.split(self.key)
            root, self.last_plan = self.agent.select_program(
                jnp.asarray(observation[None]), jnp.asarray(goal[None]), key
            )
            self.root, self.goal = root, goal.copy()
            self.cursor = 0
            self.plan_count += 1
        if self.cursor % 5 == 0:
            self.actions = np.asarray(
                self.agent.decode_program_chunk(
                    jnp.asarray(observation[None]), self.root, self.cursor // 5
                )
            )[0]
            if not np.isfinite(self.actions).all():
                raise FloatingPointError('nonfinite decoded actions')
        action = np.clip(self.actions[self.cursor % 5], self.low, self.high)
        self.cursor += 1
        return action
