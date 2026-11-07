"""Full-plan skill Q on a frozen PB terminal value; prefix-only deployment.

The Q labels are posterior-code/factual-window approximations, not certified
transitions of a newly decoded skill. Subgoal and global modes share learning.
"""

from __future__ import annotations

from functools import partial
from typing import Any

import flax
import flax.struct
import jax
import jax.numpy as jnp
import numpy as np
import optax

from skill_extensions.models import SkillNetwork

FORMAT = 'pathbridger-variable-skill-v1'
DEFAULTS = dict(
    latent_dim=16,
    width=200,
    durations=tuple(range(5, 41, 5)),
    execution_horizon=5,
    learning_rate=3e-4,
    beta=0.01,
    split_code_weight=1.0,
    split_action_weight=1.0,
    split_value_weight=1.0,
    proposal_weight=1.0,
    target_rate=0.005,
    flow_steps=8,
    candidate_count=32,
    endpoint_candidates=32,
    temperature=1.0,
    proposal_value_scale=10.0,
    goal_tolerance=1e-5,
)


def validate_config(config):
    unknown = set(config) - set(DEFAULTS)
    if unknown:
        raise ValueError(f'unknown skill options: {sorted(unknown)}')
    c = {**DEFAULTS, **dict(config)}
    c['durations'] = tuple(c['durations'])
    integer_keys = (
        'latent_dim',
        'width',
        'execution_horizon',
        'flow_steps',
        'candidate_count',
        'endpoint_candidates',
    )
    if any(not isinstance(c[k], int) or isinstance(c[k], bool) or c[k] < 1 for k in integer_keys):
        raise ValueError('positive integer skill dimensions/budgets required')
    if c['execution_horizon'] != 5:
        raise ValueError(
            'PB runner executes five-step chunks; other execution horizons are not connected'
        )
    if not c['durations'] or tuple(sorted(set(c['durations']))) != c['durations']:
        raise ValueError('durations must be unique and increasing')
    k = c['execution_horizon']
    if any(not isinstance(d, int) or d < k or d % k for d in c['durations']):
        raise ValueError('durations must be positive multiples of execution_horizon')
    if c['durations'] != tuple(range(k, max(c['durations']) + 1, k)):
        raise ValueError('durations must include every residual d-k queried by the split target')
    if c['candidate_count'] < len(c['durations']):
        raise ValueError('candidate budget must cover every duration')
    if (
        not all(np.isfinite(c[x]) for x in ('target_rate', 'learning_rate', 'goal_tolerance'))
        or not 0 < c['target_rate'] <= 1
        or c['learning_rate'] <= 0
        or c['goal_tolerance'] < 0
    ):
        raise ValueError('invalid optimizer/goal settings')
    for key in (
        'beta',
        'split_code_weight',
        'split_action_weight',
        'split_value_weight',
        'proposal_weight',
        'temperature',
        'proposal_value_scale',
    ):
        if not np.isfinite(c[key]) or c[key] < 0:
            raise ValueError(f'invalid nonnegative weight: {key}')
    return c


def first_hit(states, lengths, goals, scale, tolerance):
    times = jnp.arange(states.shape[1])[None]
    valid = times <= lengths[:, None]
    residual = jnp.where(valid[..., None], (states - goals[:, None]) / scale, jnp.inf)
    hits = valid & (jnp.max(jnp.abs(residual), axis=-1) <= tolerance)
    result = jnp.min(jnp.where(hits, times, states.shape[1]), axis=1)
    return jnp.where(jnp.any(hits, axis=1), result, -1)


def return_target(hit, duration, continuation, gamma):
    return jax.lax.stop_gradient(
        jnp.where(hit >= 0, gamma ** jnp.maximum(hit, 0), gamma**duration * continuation)
    )


def nll(actions, mean, log_std, lengths):
    mask = jnp.arange(actions.shape[1])[None] < lengths[:, None]
    target = jnp.where(mask[..., None], actions, 0.0)
    terms = 0.5 * ((target - mean) * jnp.exp(-log_std)) ** 2 + log_std + 0.5 * jnp.log(2 * jnp.pi)
    value = jnp.mean(terms, axis=-1)
    return jnp.sum(jnp.where(mask, value, 0.0), axis=1) / jnp.maximum(lengths, 1)


def gaussian_kl(q_mean, q_log_std, p_mean, p_log_std):
    return jnp.mean(
        p_log_std
        - q_log_std
        + 0.5
        * (
            jnp.exp(2 * (q_log_std - p_log_std))
            + (q_mean - p_mean) ** 2 * jnp.exp(-2 * p_log_std)
            - 1
        ),
        axis=-1,
    )


def right_window(batch):
    maximum = batch['actions'].shape[1]
    offsets = jnp.arange(maximum + 1)[None]
    ids = jnp.minimum(batch['cuts'][:, None] + offsets, batch['durations'][:, None])
    states = jnp.take_along_axis(batch['states'], ids[..., None], axis=1)
    action_ids = jnp.minimum(batch['cuts'][:, None] + offsets[:, :-1], maximum - 1)
    actions = jnp.take_along_axis(batch['actions'], action_ids[..., None], axis=1)
    lengths = batch['durations'] - batch['cuts']
    return states, jnp.where((offsets[:, :-1] < lengths[:, None])[..., None], actions, 0), lengths


@flax.struct.dataclass
class SkillState:
    params: Any
    targets: Any
    repr_optimizer: Any
    value_optimizer: Any
    representation_step: jax.Array
    value_step: jax.Array


@flax.struct.dataclass
class SkillPlanner:
    state: SkillState
    pb: Any
    mean: jax.Array
    scale: jax.Array
    rng: jax.Array
    model: Any = flax.struct.field(pytree_node=False)
    tx: Any = flax.struct.field(pytree_node=False)
    config: Any = flax.struct.field(pytree_node=False)
    mode: str = flax.struct.field(pytree_node=False, default='global')

    @classmethod
    def create(cls, pb, observation_dim, action_dim, mean, scale, *, seed, config, mode='global'):
        c = validate_config(config)
        if mode not in ('subgoal', 'global'):
            raise ValueError('mode must be subgoal or global')
        mean, scale = np.asarray(mean, np.float32), np.asarray(scale, np.float32)
        if mean.shape != (observation_dim,) or scale.shape != mean.shape:
            raise ValueError('normalization shape mismatch')
        if not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
            raise ValueError('normalization must be finite with positive scales')
        model = SkillNetwork(action_dim, c['latent_dim'], max(c['durations']), c['width'])
        maximum = max(c['durations'])
        params = model.init(
            jax.random.PRNGKey(seed),
            jnp.zeros((1, maximum + 1, observation_dim)),
            jnp.zeros((1, maximum, action_dim)),
            jnp.array([maximum]),
            jnp.zeros((1, observation_dim)),
        )['params']
        tx = optax.chain(optax.clip_by_global_norm(10.0), optax.adam(c['learning_rate']))
        state = SkillState(
            params, params, tx.init(params), tx.init(params), jnp.int32(0), jnp.int32(0)
        )
        return cls(
            state,
            pb,
            jnp.asarray(mean),
            jnp.asarray(scale),
            jax.random.PRNGKey(seed),
            model,
            tx,
            flax.core.FrozenDict(c),
            mode,
        )

    def apply(self, params, method, *args):
        return self.model.apply({'params': params}, *args, method=getattr(self.model, method))

    def normalize(self, states):
        return (states - self.mean) / self.scale

    def terminal_value(self, states, goals):
        return jax.lax.stop_gradient(jax.nn.sigmoid(self.pb.network.select('value')(states, goals)))

    def representation_loss(self, params, batch, key):
        states, actions, durations, cuts = (
            batch[k] for k in ('states', 'actions', 'durations', 'cuts')
        )
        normalized = self.normalize(states)
        mu, log_std = self.apply(params, 'encode', normalized, actions, durations)
        z = mu + jnp.exp(log_std) * jax.random.normal(key, mu.shape)
        mean, ls = self.apply(params, 'decode', normalized[:, 0], z, durations)
        whole = nll(actions, mean, ls, durations)
        left, right, left_ls, right_ls = self.apply(params, 'split', z, durations, cuts)
        r_states, r_actions, r_lengths = right_window(batch)
        l_mu, l_ls = self.apply(params, 'encode', normalized, actions, cuts)
        r_mu, r_ls = self.apply(params, 'encode', self.normalize(r_states), r_actions, r_lengths)
        left_mean, left_log_std = self.apply(params, 'decode', normalized[:, 0], left, cuts)
        middle = jnp.take_along_axis(normalized, cuts[:, None, None], 1)[:, 0]
        right_mean, right_log_std = self.apply(params, 'decode', middle, right, r_lengths)
        has_right = r_lengths > 0
        child = (
            nll(actions, left_mean, left_log_std, cuts)
            + nll(r_actions, right_mean, right_log_std, r_lengths)
        ) / (1 + has_right)
        alignment = gaussian_kl(
            jax.lax.stop_gradient(l_mu), jax.lax.stop_gradient(l_ls), left, left_ls
        )
        alignment += gaussian_kl(
            jax.lax.stop_gradient(r_mu), jax.lax.stop_gradient(r_ls), right, right_ls
        )
        alignment = jnp.where(has_right, alignment / 2, 0)
        regularizer = gaussian_kl(mu, log_std, jnp.zeros_like(mu), jnp.zeros_like(log_std))
        loss = jnp.mean(
            whole
            + self.config['split_action_weight'] * child
            + self.config['split_code_weight'] * alignment
            + self.config['beta'] * regularizer
        )
        return loss, {
            'loss': loss,
            'whole_nll': whole.mean(),
            'split_nll': child.mean(),
            'split_kl': alignment.mean(),
            'prior_kl': regularizer.mean(),
        }

    def value_loss(self, params, batch, key):
        states, actions, d, k, goals = (
            batch[x] for x in ('states', 'actions', 'durations', 'cuts', 'goals')
        )
        frozen = self.state.params
        norm = self.normalize(states)
        z, _ = self.apply(frozen, 'encode', norm, actions, d)
        r_states, r_actions, r_d = right_window(batch)
        l_z, _ = self.apply(frozen, 'encode', norm, actions, k)
        r_z, _ = self.apply(frozen, 'encode', self.normalize(r_states), r_actions, r_d)
        _, predicted_right, *_ = self.apply(frozen, 'split', z, d, k)
        middle = r_states[:, 0]
        end = jnp.take_along_axis(states, d[:, None, None], 1)[:, 0]
        gamma = float(self.pb.config['discount'])
        full_hit = first_hit(states, d, goals, self.scale, self.config['goal_tolerance'])
        left_hit = first_hit(states, k, goals, self.scale, self.config['goal_tolerance'])
        right_hit = first_hit(r_states, r_d, goals, self.scale, self.config['goal_tolerance'])
        end_v = self.terminal_value(end, goals)
        middle_v = self.terminal_value(middle, goals)
        full_y = return_target(full_hit, d, end_v, gamma)
        anchor_y = jnp.concatenate(
            (
                full_y,
                return_target(left_hit, k, middle_v, gamma),
                return_target(right_hit, r_d, end_v, gamma),
            )
        )
        all_z = jnp.concatenate((z, l_z, r_z))
        all_s = self.normalize(jnp.concatenate((states[:, 0], states[:, 0], middle)))
        all_d = jnp.concatenate((d, k, r_d))
        all_g = self.normalize(jnp.concatenate((goals, goals, goals)))
        logits = self.apply(params, 'q', all_s, all_z, all_d, all_g)
        weights = jnp.concatenate((jnp.ones_like(d), jnp.ones_like(d), r_d > 0))
        anchor = jnp.sum(
            optax.sigmoid_binary_cross_entropy(logits, anchor_y[None]) * weights[None]
        ) / (2 * weights.sum())
        right_values = jax.nn.sigmoid(
            self.apply(
                self.state.targets,
                'q',
                self.normalize(middle),
                predicted_right,
                r_d,
                self.normalize(goals),
            )
        ).min(0)
        split_y = return_target(left_hit, k, right_values, gamma)
        valid = r_d > 0
        split_bce = optax.sigmoid_binary_cross_entropy(logits[:, : len(d)], split_y[None])
        split = jnp.sum(split_bce * valid[None]) / jnp.maximum(2 * valid.sum(), 1)
        noise_key, time_key = jax.random.split(key)
        noise = jax.random.normal(noise_key, z.shape)
        time = jax.random.uniform(time_key, (len(z), 1))
        point = (1 - time) * noise + time * z
        velocity = self.apply(params, 'flow', norm[:, 0], self.normalize(goals), d, point, time)
        advantage = full_y - self.terminal_value(states[:, 0], goals)
        proposal_weights = jnp.exp(
            jnp.clip(self.config['proposal_value_scale'] * advantage, -10, 10)
        )
        proposal_weights = jax.lax.stop_gradient(proposal_weights / proposal_weights.mean())
        flow = jnp.mean(proposal_weights * jnp.mean((velocity - (z - noise)) ** 2, -1))
        loss = (
            anchor
            + self.config['split_value_weight'] * split
            + self.config['proposal_weight'] * flow
        )
        return loss, {
            'loss': loss,
            'anchor_bce': anchor,
            'split_bce': split,
            'flow_loss': flow,
            'anchor_mean': full_y.mean(),
            'split_target_mean': jnp.sum(split_y * valid) / jnp.maximum(valid.sum(), 1),
        }

    @partial(jax.jit, static_argnames=('stage',))
    def update(self, batch, *, stage):
        if stage not in ('representation', 'value'):
            raise ValueError('unknown skill training stage')
        rng, key = jax.random.split(self.rng)
        fn = self.representation_loss if stage == 'representation' else self.value_loss
        (_, metrics), grads = jax.value_and_grad(fn, has_aux=True)(self.state.params, batch, key)
        active = (
            ('encoder', 'splitter', 'decoder')
            if stage == 'representation'
            else ('q0', 'q1', 'proposal')
        )
        grads = {
            name: tree if name in active else jax.tree_util.tree_map(jnp.zeros_like, tree)
            for name, tree in grads.items()
        }
        opt_state = (
            self.state.repr_optimizer if stage == 'representation' else self.state.value_optimizer
        )
        changes, opt_state = self.tx.update(grads, opt_state, self.state.params)
        proposed = optax.apply_updates(self.state.params, changes)
        params = {
            name: proposed[name] if name in active else value
            for name, value in self.state.params.items()
        }
        if stage == 'representation':
            state = self.state.replace(
                params=params,
                targets=params,
                repr_optimizer=opt_state,
                representation_step=self.state.representation_step + 1,
            )
        else:
            rate = self.config['target_rate']
            targets = {
                name: jax.tree_util.tree_map(
                    lambda t, p: (1 - rate) * t + rate * p, self.state.targets[name], value
                )
                if name in ('q0', 'q1')
                else value
                for name, value in params.items()
            }
            state = self.state.replace(
                params=params,
                targets=targets,
                value_optimizer=opt_state,
                value_step=self.state.value_step + 1,
            )
        return self.replace(state=state, rng=rng), {
            **metrics,
            'grad_norm': optax.global_norm(grads),
        }

    def select_goal(self, observations, goals, seed):
        if self.mode == 'global':
            return goals
        candidates = self.pb._sample_endpoint_candidates(
            observations,
            goals,
            seed,
            num_candidates=self.config['endpoint_candidates'],
            temperature=self.config['temperature'],
        )
        b, n, s = candidates.shape
        starts = jnp.broadcast_to(observations[:, None], (b, n, s)).reshape(b * n, s)
        targets = jnp.broadcast_to(goals[:, None], (b, n, s)).reshape(b * n, s)
        flat = candidates.reshape(b * n, s)
        score = self.terminal_value(starts, flat) * self.terminal_value(flat, targets)
        best = jnp.argmax(score.reshape(b, n), axis=1)
        return jnp.take_along_axis(candidates, best[:, None, None], axis=1)[:, 0]

    @partial(jax.jit, static_argnames=('num_candidates', 'temperature'))
    def plan(self, observations, goals, seed, *, num_candidates, temperature):
        selected_goal = self.select_goal(observations, goals, jax.random.fold_in(seed, 11))
        b, s = observations.shape
        n = num_candidates
        ds = jnp.asarray(self.config['durations'], jnp.int32)[
            jnp.arange(n) % len(self.config['durations'])
        ]
        lengths = jnp.broadcast_to(ds[None], (b, n)).reshape(-1)
        starts = self.normalize(
            jnp.broadcast_to(observations[:, None], (b, n, s)).reshape(b * n, s)
        )
        targets = self.normalize(
            jnp.broadcast_to(selected_goal[:, None], (b, n, s)).reshape(b * n, s)
        )
        z = temperature * jax.random.normal(
            jax.random.fold_in(seed, 22), (b * n, self.config['latent_dim'])
        )
        steps = self.config['flow_steps']

        def advance(i, value):
            time = jnp.full((b * n, 1), i / steps)
            return (
                value
                + self.apply(self.state.params, 'flow', starts, targets, lengths, value, time)
                / steps
            )

        z = jax.lax.fori_loop(0, steps, advance, z)
        scores = (
            jax.nn.sigmoid(self.apply(self.state.params, 'q', starts, z, lengths, targets))
            .min(0)
            .reshape(b, n)
        )
        best = jnp.argmax(scores, axis=1)
        chosen_z = z.reshape(b, n, -1)[jnp.arange(b), best]
        chosen_d = ds[best]
        cuts = jnp.full((b,), self.config['execution_horizon'], jnp.int32)
        left, *_ = self.apply(self.state.params, 'split', chosen_z, chosen_d, cuts)
        mean, _ = self.apply(self.state.params, 'decode', self.normalize(observations), left, cuts)
        actions = mean[:, : self.config['execution_horizon']]
        return actions, {
            'scores': scores,
            'durations': ds,
            'selected': best,
            'selected_duration': chosen_d,
            'selected_goal': selected_goal,
            'candidate_codes': z.reshape(b, n, -1),
        }

    def sample_action_chunks(
        self, observations, goals, seed=None, num_candidates=None, temperature=None
    ):
        observations, goals = jnp.asarray(observations), jnp.asarray(goals)
        squeeze = observations.ndim == 1
        if squeeze:
            observations, goals = observations[None], goals[None]
        if (
            observations.ndim != 2
            or observations.shape != goals.shape
            or observations.shape[-1] != len(self.mean)
        ):
            raise ValueError('skill inference needs matching state/goal vector batches')
        n = self.config['candidate_count'] if num_candidates is None else num_candidates
        temp = self.config['temperature'] if temperature is None else temperature
        if n < len(self.config['durations']) or temp < 0:
            raise ValueError(
                'candidate count must cover all lengths and temperature must be nonnegative'
            )
        actions, _ = self.plan(
            observations,
            goals,
            jax.random.PRNGKey(0) if seed is None else seed,
            num_candidates=n,
            temperature=temp,
        )
        return actions[0] if squeeze else actions
