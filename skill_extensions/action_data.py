"""Offline windows and explicit goal/terminal contracts, without environment imports."""

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
from pathlib import Path

import numpy as np

from utils.datasets import Dataset, PathBridgerDataset, dataset_identity
from utils.skill_data import episode_fingerprints


class ActionWindowSampler:
    def __init__(self, dataset, *, seed, goal_config=None, boundary_kind=None, horizon=20):
        self.dataset = dataset
        self.rng = np.random.RandomState(seed)
        if horizon not in (5, 20):
            raise ValueError('only five-step decoder or 20-step program windows are supported')
        if horizon != 20 and (goal_config is not None or boundary_kind is not None):
            raise ValueError('short decoder windows do not carry macro-Q labels')
        self.horizon = horizon
        self.boundary_kind = boundary_kind
        if boundary_kind not in (None, 'timeout', 'terminal', 'field'):
            raise ValueError('boundary_kind must be timeout, terminal, or field')
        obs, actions = dataset['observations'], dataset['actions']
        if obs.ndim != 2 or actions.ndim != 2 or not np.isfinite(obs).all():
            raise ValueError('finite state/action matrices required')
        finals = np.flatnonzero(np.asarray(dataset['terminals']).reshape(-1) > 0)
        if not len(finals) or finals[-1] != len(obs) - 1:
            raise ValueError('explicit final compact episode boundary required')
        firsts = np.r_[0, finals[:-1] + 1]
        valid_action = np.isfinite(actions).all(axis=1)
        if 'valids' in dataset:
            valid_action &= np.asarray(dataset['valids']).reshape(-1) > 0
        for final in finals:
            valid_action[final] = False
        bad_prefix = np.r_[0, np.cumsum(~valid_action)]
        starts = []
        for first, final in zip(firsts, finals, strict=True):
            candidates = np.arange(first, final - horizon + 1)
            starts.extend(candidates[bad_prefix[candidates + horizon] == bad_prefix[candidates]])
        self.starts = np.asarray(starts, np.int64)
        if not len(self.starts):
            raise ValueError(f'no complete valid {horizon}-step windows')
        self.allowed_starts = np.zeros(len(obs), bool)
        self.allowed_starts[self.starts] = True
        if boundary_kind == 'field' and 'true_terminals' not in dataset:
            raise ValueError('field boundary_kind requires true_terminals')
        if 'true_terminals' in dataset:
            true_rows = np.asarray(dataset['true_terminals']).reshape(-1) > 0
            if np.any(true_rows & ~(np.asarray(dataset['terminals']).reshape(-1) > 0)):
                raise ValueError('a true terminal must also be an episode boundary')
        self.goal_sampler = (
            PathBridgerDataset(dataset, {**dict(goal_config), 'horizon': 5}, seed=seed + 1)
            if goal_config is not None
            else None
        )

    def get_rng_state(self):
        return (
            self.rng.get_state(),
            None if self.goal_sampler is None else self.goal_sampler.get_rng_state(),
        )

    def set_rng_state(self, state):
        self.rng.set_state(state[0])
        if self.goal_sampler is not None:
            self.goal_sampler.set_rng_state(state[1])
        elif state[1] is not None:
            raise ValueError('checkpoint expects a goal sampler')

    def windows(self, starts):
        starts = np.asarray(starts, np.int64)
        if starts.ndim != 1 or not len(starts):
            raise ValueError('nonempty start vector required')
        # Membership prevents crossing episodes even for caller-supplied contexts.
        if (
            np.any(starts < 0)
            or np.any(starts >= len(self.allowed_starts))
            or not self.allowed_starts[starts].all()
        ):
            raise ValueError('invalid or cross-episode source window')
        states = np.asarray(
            self.dataset['observations'][starts[:, None] + np.arange(self.horizon + 1)]
        )
        actions = np.asarray(self.dataset['actions'][starts[:, None] + np.arange(self.horizon)])
        result = {'states': states.astype(np.float32), 'actions': actions.astype(np.float32)}
        if self.goal_sampler is not None:
            goals = np.asarray(
                self.goal_sampler.sample(len(starts), idxs=starts)['endpoint_goals']
            ).copy()
            goals[::2] = states[::2, 20]
            result['goals'] = goals.astype(np.float32)
        if self.boundary_kind is not None:
            if self.boundary_kind == 'field':
                true = np.asarray(self.dataset['true_terminals'])[starts + 20] > 0
            elif self.boundary_kind == 'terminal':
                true = np.asarray(self.dataset['terminals'])[starts + 20] > 0
            else:
                true = np.zeros(len(starts), bool)
            result['true_terminals'] = true.reshape(-1)
        return result

    def sample(self, count):
        if count < 1:
            raise ValueError('positive batch size required')
        return self.windows(self.rng.choice(self.starts, count))


def load_archive(path):
    with np.load(path, allow_pickle=False) as data:
        return Dataset.create(**{key: data[key] for key in data.files})


def dataset_contract(train, validation):
    if episode_fingerprints(train) & episode_fingerprints(validation):
        raise ValueError('training and validation contain identical complete episodes')
    return {'train': dataset_identity(train), 'validation': dataset_identity(validation)}


def load_goal_contract(path, *, smoke=False):
    """No undocumented default tolerance/feature projection is accepted for Q.

    The external predicate receives states [B,T,D], goals [B,D], parameters dict;
    it must return bool [B,T] under the benchmark's goal semantics.
    """
    specification = json.loads(Path(path).read_text())
    required = {'callable', 'parameters', 'environment', 'validation_evidence'}
    if required - set(specification):
        raise ValueError(f'goal contract missing {sorted(required - set(specification))}')
    if not smoke and not specification['validation_evidence']:
        raise ValueError('formal Q requires environment goal-predicate validation evidence')
    module_name, name = specification['callable'].split(':', 1)
    module = importlib.import_module(module_name)
    predicate = getattr(module, name)
    if not smoke and predicate is diagnostic_full_state_linf:
        raise ValueError('diagnostic full-state tolerance is not a formal benchmark goal adapter')
    source = inspect.getsourcefile(predicate)
    if source is None:
        raise ValueError('goal predicate must have an inspectable source file')
    specification['source_sha256'] = hashlib.sha256(Path(source).read_bytes()).hexdigest()
    return predicate, specification


def add_value_labels(batch, predicate, parameters, discount):
    if 'true_terminals' not in batch:
        raise ValueError('true terminal semantics must be explicit before Q learning')
    if not 0 < discount < 1:
        raise ValueError('discount must lie in (0,1)')
    hits = np.asarray(predicate(batch['states'], batch['goals'], parameters))
    if hits.shape != batch['states'].shape[:2] or hits.dtype != np.bool_:
        raise ValueError('goal predicate must return bool [batch,21]')
    first = np.where(hits.any(1), hits.argmax(1), -1).astype(np.int32)
    return {**batch, 'hits': first, 'discount': np.float32(discount)}


def diagnostic_full_state_linf(states, goals, parameters):
    """Synthetic/diagnostic only; not advertised as an OGBench success rule."""
    tolerance = float(parameters['tolerance'])
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError('invalid diagnostic tolerance')
    return np.max(np.abs(states - goals[:, None]), axis=-1) <= tolerance
