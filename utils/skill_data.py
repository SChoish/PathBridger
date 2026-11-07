"""Variable-length factual windows in PB's compact episode layout."""

from __future__ import annotations

import hashlib

import numpy as np

from utils.datasets import Dataset, PathBridgerDataset


class SkillDataset:
    def __init__(self, dataset: Dataset, pb_config, config, *, seed):
        self.dataset, self.config = dataset, dict(config)
        self.base = PathBridgerDataset(dataset, {**dict(pb_config), 'horizon': 5}, seed=seed)
        self.rng = np.random.RandomState(seed)
        self.starts = {}
        observations, actions = dataset['observations'], dataset['actions']
        if (
            actions.ndim != 2
            or not np.isfinite(actions).all()
            or not np.isfinite(observations).all()
        ):
            raise ValueError('skill training requires finite state vectors and recorded actions')
        invalid_prefix = None
        if 'valids' in dataset:
            invalid_prefix = np.r_[0, np.cumsum(np.asarray(dataset['valids']).reshape(-1) <= 0)]
        for duration in config['durations']:
            parts = []
            for first, final in zip(self.base.initial_locs, self.base.terminal_locs):
                if final - first >= duration:
                    candidates = np.arange(first, final - duration + 1)
                    # Compact terminal actions are never used. Respect invalid action rows.
                    if invalid_prefix is not None:
                        candidates = candidates[
                            invalid_prefix[candidates + duration] == invalid_prefix[candidates]
                        ]
                    parts.append(candidates)
            self.starts[duration] = np.concatenate(parts) if parts else np.array([], np.int64)
            if not len(self.starts[duration]):
                raise ValueError(f'no complete valid training window for duration {duration}')

    def get_rng_state(self):
        return self.rng.get_state(), self.base.get_rng_state()

    def set_rng_state(self, state):
        self.rng.set_state(state[0])
        self.base.set_rng_state(state[1])

    def sample(self, batch_size):
        if batch_size < 1:
            raise ValueError('positive batch size required')
        choices = self.rng.choice(self.config['durations'], size=batch_size)
        starts = np.asarray([self.rng.choice(self.starts[int(d)]) for d in choices])
        base = self.base.sample(batch_size, idxs=starts)
        maximum, k = max(self.config['durations']), self.config['execution_horizon']
        offsets = np.arange(maximum + 1)[None]
        indices = starts[:, None] + np.minimum(offsets, choices[:, None])
        observations = np.asarray(self.dataset['observations'][indices], np.float32)
        action_indices = starts[:, None] + np.minimum(
            np.arange(maximum)[None], choices[:, None] - 1
        )
        actions = np.asarray(self.dataset['actions'][action_indices], np.float32)
        mask = np.arange(maximum)[None] < choices[:, None]
        actions = np.where(mask[..., None], actions, 0.0)
        goals = np.asarray(base['endpoint_goals'], np.float32).copy()
        # Include reachable in-window anchors and longer-horizon goals. The exact
        # endpoint half guarantees positive anchors without pretending random goals
        # are permanently unreachable. Both inference modes use identical training.
        endpoint_rows = np.arange(batch_size) % 2 == 0
        goals[endpoint_rows] = observations[endpoint_rows, choices[endpoint_rows]]
        return {
            'states': observations,
            'actions': actions,
            'durations': choices.astype(np.int32),
            'cuts': np.minimum(choices, k).astype(np.int32),
            'goals': goals,
        }


def fit_skill_normalization(dataset):
    states = np.asarray(dataset['observations'], np.float64)
    return states.mean(0).astype(np.float32), np.maximum(states.std(0), 1e-3).astype(np.float32)


def episode_fingerprints(dataset):
    """Detect exactly duplicated complete episodes, not all possible physical overlap."""
    finals = np.flatnonzero(np.asarray(dataset['terminals']).reshape(-1) > 0)
    if not len(finals) or finals[-1] != len(dataset['observations']) - 1:
        raise ValueError('explicit compact episode boundaries required')
    identities, first = set(), 0
    for final in finals:
        h = hashlib.sha256()
        for values in (dataset['observations'][first : final + 1], dataset['actions'][first:final]):
            array = np.ascontiguousarray(values)
            h.update(str((array.dtype.str, array.shape)).encode())
            h.update(memoryview(array).cast('B'))
        identities.add(h.hexdigest())
        first = final + 1
    return identities
