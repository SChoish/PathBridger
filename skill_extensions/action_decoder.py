"""From-scratch five-step action autoencoder, before the frozen-executor A/B.

Only actions enter the encoder. Observations condition the chunk decoder.
The sole objective is recorded-action mean squared error; there is no PB,
goal, critic, splitter, KL, or proposal in this prerequisite stage.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState

from skill_extensions.action_data import ActionWindowSampler, dataset_contract, load_archive
from skill_extensions.action_models import ActionSequenceEncoder
from skill_extensions.action_runtime import atomic_json, digest, source_identity
from skill_extensions.models import ChunkDecoder
from skill_extensions.runtime import write_checkpoint
from utils.skill_data import fit_skill_normalization

FORMAT = 'pathbridger-action-decoder-v1'


class ShortActionAutoencoder(nn.Module):
    action_dim: int
    latent_dim: int
    width: int

    def setup(self):
        self.encoder = ActionSequenceEncoder(self.latent_dim, self.width, 5)
        self.decoder = ChunkDecoder(self.action_dim, 5, self.width)

    def encode(self, actions):
        return self.encoder(actions, jnp.full((len(actions),), 5, jnp.int32))

    def __call__(self, states, actions):
        mean, _ = self.decoder(states, self.encode(actions), jnp.full((len(actions),), 5))
        return mean


@flax.struct.dataclass
class DecoderAgent:
    state: TrainState
    mean: jax.Array
    scale: jax.Array
    model: ShortActionAutoencoder = flax.struct.field(pytree_node=False)

    @classmethod
    def create(cls, observation_dim, action_dim, mean, scale, *, seed, config):
        model = ShortActionAutoencoder(action_dim, config['latent_dim'], config['width'])
        params = flax.core.freeze(
            model.init(
                jax.random.PRNGKey(seed),
                jnp.zeros((1, observation_dim)),
                jnp.zeros((1, 5, action_dim)),
            )['params']
        )
        tx = optax.chain(optax.clip_by_global_norm(10.0), optax.adam(config['learning_rate']))
        state = TrainState.create(apply_fn=None, params=params, tx=tx)
        return cls(state, jnp.asarray(mean), jnp.asarray(scale), model)

    def loss(self, params, batch):
        prediction = self.model.apply(
            {'params': params}, (batch['states'][:, 0] - self.mean) / self.scale, batch['actions']
        )
        return jnp.mean((prediction - batch['actions']) ** 2)

    @partial(jax.jit)
    def update(self, batch):
        loss, grads = jax.value_and_grad(self.loss)(self.state.params, batch)
        return self.replace(state=self.state.apply_gradients(grads=grads)), {'action_mse': loss}


def run_decoder_train(args):
    """Called after common budget/GPU guards in action_skill_main.run_train."""
    import json
    import pickle

    output = Path(args.output)
    if output.exists():
        raise FileExistsError('use a fresh decoder output directory, including for resume')
    if (
        args.donor
        or args.donor_sha256
        or args.goal_config
        or args.goal_contract
        or args.boundary_kind
    ):
        raise ValueError('decoder pretraining uses no donor, PB, goals, or terminal-value labels')
    if bool(args.input_checkpoint) != bool(args.input_sha256):
        raise ValueError('input checkpoint and its SHA256 must be supplied together')
    train, validation = load_archive(args.train_npz), load_archive(args.validation_npz)
    identity = dataset_contract(train, validation)
    sampler = ActionWindowSampler(train, seed=args.seed, horizon=5)
    validation_sampler = ActionWindowSampler(validation, seed=args.seed + 100000, horizon=5)
    od, ad = train['observations'].shape[1], train['actions'].shape[1]
    if (validation['observations'].shape[1], validation['actions'].shape[1]) != (od, ad):
        raise ValueError('train/validation dimensions differ')
    # Exactly the donor's train-fitted normalization is exported and later frozen.
    mean, scale = fit_skill_normalization(train)
    config = dict(latent_dim=16, width=200, max_duration=5, learning_rate=3e-4)
    previous = None
    if args.input_checkpoint:
        if args.config or digest(args.input_checkpoint) != args.input_sha256:
            raise ValueError('resume requires unchanged config and exact checkpoint SHA256')
        with Path(args.input_checkpoint).open('rb') as stream:
            previous = pickle.load(stream)
        if previous.get('format') != FORMAT or previous['sources'] != source_identity():
            raise ValueError('decoder checkpoint format/source identity mismatch')
        config = previous['config']
    elif args.config:
        supplied = json.loads(Path(args.config).read_text())
        if set(supplied) - set(config):
            raise ValueError('unknown short decoder config keys')
        config.update(supplied)
    if (
        any(type(config[k]) is not int or config[k] < 1 for k in ('latent_dim', 'width'))
        or config['max_duration'] != 5
        or not np.isfinite(config['learning_rate'])
        or config['learning_rate'] <= 0
    ):
        raise ValueError('invalid five-step action decoder configuration')
    metadata = dict(
        seed=args.seed,
        smoke=args.smoke,
        datasets=identity,
        batch_size=args.batch_size,
        environment_steps=0,
        training_kind='action-only-five-step-autoencoder',
        label_contract='recorded five-step action MSE; observations only condition decoder',
    )
    agent = DecoderAgent.create(od, ad, mean, scale, seed=args.seed, config=config)
    if previous:
        if previous['metadata'] != metadata:
            raise ValueError('decoder resume data/seed/batch/smoke contract changed')
        if not np.array_equal(previous['mean'], mean) or not np.array_equal(
            previous['scale'], scale
        ):
            raise ValueError('decoder resume normalization changed')
        agent = agent.replace(
            state=flax.serialization.from_state_dict(agent.state, previous['state'])
        )
        sampler.set_rng_state(previous['sampler_rng'])
    current = int(agent.state.step)
    if current >= args.updates:
        raise ValueError('checkpoint has already reached the decoder stage budget')
    output.mkdir(parents=True)
    manifest = {
        **metadata,
        'stage': 'decoder',
        'config': config,
        'status': 'running',
        'history': [],
    }
    atomic_json(output / 'manifest.json', manifest)
    try:
        for index in range(current, args.updates):
            batch = {k: jnp.asarray(v) for k, v in sampler.sample(args.batch_size).items()}
            agent, metrics = agent.update(batch)
            row = {'step': index + 1, **{k: float(v) for k, v in metrics.items()}}
            if not all(np.isfinite(value) for value in row.values()):
                raise FloatingPointError('nonfinite short decoder training metrics')
            if index == current or (index + 1) % 100 == 0 or index + 1 == args.updates:
                manifest['history'].append(row)
                print(json.dumps(row), flush=True)
            if (index + 1) % args.checkpoint_every == 0 or index + 1 == args.updates:
                payload = dict(
                    format=FORMAT,
                    sources=source_identity(),
                    state=flax.serialization.to_state_dict(jax.device_get(agent.state)),
                    config=config,
                    metadata=metadata,
                    smoke=args.smoke,
                    decoder_step=int(agent.state.step),
                    observation_dim=od,
                    action_dim=ad,
                    mean=np.asarray(agent.mean),
                    scale=np.asarray(agent.scale),
                    sampler_rng=sampler.get_rng_state(),
                )
                manifest['last_checkpoint'] = write_checkpoint(
                    output / f'decoder_{index + 1:07d}.pkl', payload, smoke=args.smoke
                )
                atomic_json(output / 'manifest.json', manifest)
        batch = {k: jnp.asarray(v) for k, v in validation_sampler.sample(args.batch_size).items()}
        manifest.update(
            status='completed',
            fixed_validation={'action_mse': float(agent.loss(agent.state.params, batch))},
            validation_use='fixed diagnostic only; no checkpoint/model selection',
        )
    except BaseException as error:
        manifest.update(status='failed', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        atomic_json(output / 'manifest.json', manifest)
    return manifest
