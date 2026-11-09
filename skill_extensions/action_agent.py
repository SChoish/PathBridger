"""Frozen-decoder action programs, plain FM, and finite-candidate macro Q.

Definition/rationale: docs/action_reader_20261109_KO.md.
New parent encoders are initialized from scratch, never from a state/action encoder.
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

from skill_extensions.action_models import ActionProgramNetwork
from skill_extensions.models import ChunkDecoder

DEFAULTS = dict(
    latent_dim=16,
    width=200,
    program_length=20,
    chunk=5,
    representation='reader',
    learning_rate=3e-4,
    target_rate=0.005,
    flow_steps=8,
    candidate_count=32,
)


def validate_config(config):
    if set(config) - set(DEFAULTS):
        raise ValueError(f'unknown action-program keys: {sorted(set(config) - set(DEFAULTS))}')
    c = {**DEFAULTS, **dict(config)}
    for key in ('latent_dim', 'width', 'program_length', 'chunk', 'flow_steps', 'candidate_count'):
        if isinstance(c[key], bool) or not isinstance(c[key], int) or c[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    if c['program_length'] != 20 or c['chunk'] != 5:
        raise ValueError('this pilot fixes a 20-step program and five-step execution')
    if c['representation'] not in ('reader', 'recursive'):
        raise ValueError('representation must be reader or recursive')
    if not np.isfinite(c['learning_rate']) or c['learning_rate'] <= 0:
        raise ValueError('invalid learning rate')
    if not np.isfinite(c['target_rate']) or not 0 < c['target_rate'] <= 1:
        raise ValueError('invalid target rate')
    return flax.core.FrozenDict(c)


def expected_max_target(hits, true_terminals, next_scores, gamma, horizon=20):
    """hits: -1 for no hit; next_scores: [ensemble,batch,candidates].

    First-hit at the initial state is 1; success dominates a coincident terminal.
    Compact episode boundaries/time limits must not be passed as true terminals.
    """
    best = jnp.max(jnp.min(next_scores, axis=0), axis=-1)
    bootstrap = jnp.where(true_terminals, 0.0, gamma**horizon * best)
    return jax.lax.stop_gradient(jnp.where(hits >= 0, gamma ** jnp.maximum(hits, 0), bootstrap))


@flax.struct.dataclass
class ActionProgramState:
    params: Any
    targets: Any
    repr_optimizer: Any
    flow_optimizer: Any
    q_optimizer: Any
    rng: Any
    root_mean: Any
    root_scale: Any
    representation_step: Any
    proposal_step: Any
    value_step: Any


@flax.struct.dataclass
class ActionProgramAgent:
    state: ActionProgramState
    decoder_params: Any
    observation_mean: Any
    observation_scale: Any
    config: Any = flax.struct.field(pytree_node=False)
    model: Any = flax.struct.field(pytree_node=False)
    decoder: Any = flax.struct.field(pytree_node=False)
    tx: Any = flax.struct.field(pytree_node=False)
    stage: str = flax.struct.field(pytree_node=False, default='representation')
    observation_dim: int = flax.struct.field(pytree_node=False, default=0)
    action_dim: int = flax.struct.field(pytree_node=False, default=0)
    decoder_max_duration: int = flax.struct.field(pytree_node=False, default=0)
    decoder_width: int = flax.struct.field(pytree_node=False, default=0)

    @classmethod
    def create(cls, asset, *, seed=0, config=None):
        c = validate_config(config or {})
        if c['latent_dim'] != asset['latent_dim'] or asset['max_duration'] < 5:
            raise ValueError('frozen decoder latent/duration contract mismatch')
        od, ad = int(asset['observation_dim']), int(asset['action_dim'])
        mean, scale = np.asarray(asset['mean']), np.asarray(asset['scale'])
        if (
            mean.shape != (od,)
            or scale.shape != (od,)
            or not np.isfinite(mean).all()
            or not np.isfinite(scale).all()
            or np.any(scale <= 0)
        ):
            raise ValueError('invalid donor observation normalization')
        model = ActionProgramNetwork(
            c['latent_dim'],
            c['width'],
            c['program_length'],
            c['chunk'],
            asset['max_duration'],
            c['representation'],
        )
        params = flax.core.freeze(
            model.init(
                jax.random.PRNGKey(seed),
                jnp.zeros((1, 20, ad)),
                jnp.array([20]),
                jnp.zeros((1, od)),
                jnp.zeros((1, od)),
            )['params']
        )
        tx = optax.chain(optax.clip_by_global_norm(10.0), optax.adam(c['learning_rate']))

        def select(names):
            return flax.core.freeze({name: params[name] for name in names})

        state = ActionProgramState(
            params=params,
            targets=select(('q0', 'q1')),
            repr_optimizer=tx.init(select(('encoder', 'reader'))),
            flow_optimizer=tx.init(select(('proposal',))),
            q_optimizer=tx.init(select(('q0', 'q1'))),
            rng=jax.random.PRNGKey(seed + 1),
            root_mean=jnp.zeros(c['latent_dim']),
            root_scale=jnp.ones(c['latent_dim']),
            representation_step=jnp.int32(0),
            proposal_step=jnp.int32(0),
            value_step=jnp.int32(0),
        )
        agent = cls(
            state=state,
            decoder_params=flax.core.freeze(asset['params']),
            observation_mean=jnp.asarray(mean, jnp.float32),
            observation_scale=jnp.asarray(scale, jnp.float32),
            config=c,
            model=model,
            decoder=ChunkDecoder(ad, asset['max_duration'], asset['width']),
            tx=tx,
            observation_dim=od,
            action_dim=ad,
            decoder_max_duration=asset['max_duration'],
            decoder_width=asset['width'],
        )
        # Detect incompatible transplant shapes before starting an optimizer.
        result = agent.decode(jnp.zeros((1, od)), jnp.zeros((1, c['latent_dim'])))
        if result.shape != (1, 5, ad) or not np.isfinite(result).all():
            raise ValueError('invalid frozen decoder forward pass')
        return agent

    def apply(self, params, method, *args):
        return self.model.apply({'params': params}, *args, method=getattr(self.model, method))

    def normalize(self, states):
        return (states - self.observation_mean) / self.observation_scale

    def encode_actions(self, actions):
        actions = jnp.asarray(actions, jnp.float32)
        if actions.ndim != 3 or actions.shape[1:] != (20, self.action_dim):
            raise ValueError('encoder requires [batch,20,action_dim], with no state input')
        return self.apply(
            self.state.params, 'encode', actions, jnp.full((len(actions),), 20, jnp.int32)
        )

    def normalize_root(self, root):
        return (root - self.state.root_mean) / self.state.root_scale

    def raw_root(self, value):
        return self.state.root_mean + self.state.root_scale * value

    def decode(self, states, code):
        # Parameters are frozen, but the derivative with respect to code is essential.
        mean, _ = self.decoder.apply(
            {'params': self.decoder_params},
            self.normalize(states),
            code,
            jnp.full((len(code),), 5, jnp.int32),
        )
        return mean[:, :5]

    def representation_loss(self, params, batch, key=None):
        del key
        actions, states = batch['actions'], batch['states']
        root = self.apply(params, 'encode', actions, jnp.full((len(actions),), 20))
        leaves = self.apply(params, 'leaves', root)
        predicted = self.decode(
            states[:, :20:5].reshape((-1, self.observation_dim)),
            leaves.reshape((-1, self.config['latent_dim'])),
        ).reshape(actions.shape)
        squared = (predicted - actions) ** 2
        chunks = squared.reshape((len(actions), 4, 5, self.action_dim)).mean((0, 2, 3))
        metrics = {'loss': squared.mean()}
        metrics.update({f'action_mse_{i}': chunks[i] for i in range(4)})
        return squared.mean(), metrics

    def flow_loss(self, params, batch, key):
        noise_key, time_key = jax.random.split(key)
        target = jax.lax.stop_gradient(self.normalize_root(self.encode_actions(batch['actions'])))
        noise = jax.random.normal(noise_key, target.shape)
        time = jax.random.uniform(time_key, (len(target), 1))
        point = (1 - time) * noise + time * target
        velocity = self.apply(
            params,
            'flow',
            self.normalize(batch['states'][:, 0]),
            self.normalize(batch['goals']),
            point,
            time,
        )
        loss = jnp.mean((velocity - (target - noise)) ** 2)
        return loss, {'loss': loss}

    def candidates(self, params, states, goals, key):
        n = self.config['candidate_count']
        batch = len(states)
        starts = jnp.repeat(self.normalize(states), n, axis=0)
        targets = jnp.repeat(self.normalize(goals), n, axis=0)
        point = jax.random.normal(key, (batch * n, self.config['latent_dim']))
        steps = self.config['flow_steps']

        def advance(i, x):
            time = jnp.full((len(x), 1), i / steps)
            return x + self.apply(params, 'flow', starts, targets, x, time) / steps

        return jax.lax.fori_loop(0, steps, advance, point).reshape((batch, n, -1))

    def scores(self, params, states, candidates, goals):
        batch, count, dim = candidates.shape
        logits = self.apply(
            params,
            'q',
            jnp.repeat(self.normalize(states), count, axis=0),
            candidates.reshape((-1, dim)),
            jnp.repeat(self.normalize(goals), count, axis=0),
        )
        return jax.nn.sigmoid(logits).reshape((2, batch, count))

    def value_loss(self, params, batch, key):
        # flow is frozen, target Q uses the same finite-candidate rule as deployment.
        candidates = self.candidates(self.state.params, batch['states'][:, 20], batch['goals'], key)
        target_params = self.state.params.copy(self.state.targets)
        next_scores = self.scores(target_params, batch['states'][:, 20], candidates, batch['goals'])
        target = expected_max_target(
            batch['hits'], batch['true_terminals'], next_scores, batch['discount']
        )
        root = jax.lax.stop_gradient(self.normalize_root(self.encode_actions(batch['actions'])))
        logits = self.apply(
            params,
            'q',
            self.normalize(batch['states'][:, 0]),
            root,
            self.normalize(batch['goals']),
        )
        loss = optax.sigmoid_binary_cross_entropy(logits, target[None]).mean()
        return loss, {'loss': loss, 'target_mean': target.mean()}

    def seal_representation(self, mean, scale):
        if self.stage != 'representation' or int(self.state.representation_step) < 1:
            raise ValueError('representation must be trained before sealing')
        mean, scale = np.asarray(mean), np.asarray(scale)
        if (
            mean.shape != (self.config['latent_dim'],)
            or scale.shape != mean.shape
            or not np.isfinite(mean).all()
            or not np.isfinite(scale).all()
            or np.any(scale <= 0)
        ):
            raise ValueError('invalid training-only root statistics')
        return self.replace(
            state=self.state.replace(root_mean=jnp.asarray(mean), root_scale=jnp.asarray(scale)),
            stage='proposal',
        )

    def seal_proposal(self):
        if self.stage != 'proposal' or int(self.state.proposal_step) < 1:
            raise ValueError('proposal must be trained before sealing')
        return self.replace(stage='value')

    def update(self, batch, *, stage):
        if stage != self.stage:
            raise ValueError(f'active stage is {self.stage}, cannot update {stage}')
        if batch['actions'].shape[1:] != (20, self.action_dim):
            raise ValueError('wrong action window')
        if batch['states'].shape[1:] != (21, self.observation_dim):
            raise ValueError('wrong state window')
        if stage == 'value':
            gamma = float(batch['discount'])
            if not 0 < gamma < 1:
                raise ValueError('discount must lie in (0,1)')
        return self._update(batch, stage=stage)

    @partial(jax.jit, static_argnames=('stage',))
    def _update(self, batch, *, stage):
        names = {
            'representation': ('encoder', 'reader'),
            'proposal': ('proposal',),
            'value': ('q0', 'q1'),
        }[stage]
        active = flax.core.freeze({name: self.state.params[name] for name in names})
        rng, key = jax.random.split(self.state.rng)
        loss_fn = {
            'representation': self.representation_loss,
            'proposal': self.flow_loss,
            'value': self.value_loss,
        }[stage]

        def objective(current):
            return loss_fn(self.state.params.copy(current), batch, key)

        (_, metrics), grads = jax.value_and_grad(objective, has_aux=True)(active)
        optimizer_name = {
            'representation': 'repr_optimizer',
            'proposal': 'flow_optimizer',
            'value': 'q_optimizer',
        }[stage]
        changes, optimizer = self.tx.update(grads, getattr(self.state, optimizer_name), active)
        updated = optax.apply_updates(active, changes)
        params = self.state.params.copy(updated)
        targets = self.state.targets
        if stage == 'value':
            tau = self.config['target_rate']
            targets = jax.tree_util.tree_map(
                lambda old, new: (1 - tau) * old + tau * new, targets, updated
            )
        step_name = f'{stage}_step'
        state = self.state.replace(
            params=params,
            targets=targets,
            rng=rng,
            **{
                optimizer_name: optimizer,
                step_name: getattr(self.state, step_name) + 1,
            },
        )
        return self.replace(state=state), {**metrics, 'grad_norm': optax.global_norm(grads)}

    @jax.jit
    def select_program(self, states, goals, key):
        candidates = self.candidates(self.state.params, states, goals, key)
        scores = self.scores(self.state.params, states, candidates, goals).min(0)
        selected = jnp.argmax(scores, axis=1)
        value = candidates[jnp.arange(len(states)), selected]
        return self.raw_root(value), {
            'scores': scores,
            'selected': selected,
            'candidate_codes': candidates,
        }

    @jax.jit
    def decode_program_chunk(self, states, root, index):
        leaves = self.apply(self.state.params, 'leaves', root)
        return self.decode(states, leaves[:, index])
