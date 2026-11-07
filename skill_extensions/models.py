"""Variable-duration skill models. Algebra has no state/goal inputs."""

from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp

from utils.networks import MLP


class GaussianHead(nn.Module):
    width: int

    @nn.compact
    def __call__(self, x):
        return nn.Dense(self.width, name='mean')(x), jnp.clip(
            nn.Dense(self.width, name='log_std')(x), -5.0, 1.0
        )


class SequenceEncoder(nn.Module):
    latent_dim: int
    width: int
    max_duration: int

    @nn.compact
    def __call__(self, states, actions, lengths):
        mask = jnp.arange(actions.shape[1])[None] < lengths[:, None]
        # Invalid padding must not enter a recurrent projection, even as NaNs.
        states = jnp.where(mask[..., None], states[:, :-1], 0.0)
        actions = jnp.where(mask[..., None], actions, 0.0)
        x = nn.relu(nn.Dense(self.width)(jnp.concatenate((states, actions), axis=-1)))
        safe_lengths = jnp.maximum(lengths, 1)
        forward = nn.RNN(nn.GRUCell(self.width), name='forward')(x, seq_lengths=safe_lengths)
        backward = nn.RNN(nn.GRUCell(self.width), reverse=True, keep_order=True, name='backward')(
            x, seq_lengths=safe_lengths
        )
        last = jnp.take_along_axis(forward, (safe_lengths - 1)[:, None, None], axis=1)[:, 0]
        pooled = jnp.concatenate((last, backward[:, 0], lengths[:, None] / self.max_duration), -1)
        return GaussianHead(self.latent_dim)(pooled)


class SkillSplitter(nn.Module):
    latent_dim: int
    width: int
    max_duration: int

    @nn.compact
    def __call__(self, z, durations, cuts):
        times = jnp.stack((durations, cuts, durations - cuts), axis=-1) / self.max_duration
        hidden = MLP((self.width, self.width), activate_final=True)(jnp.concatenate((z, times), -1))
        mean, log_std = GaussianHead(2 * self.latent_dim)(hidden)
        left, right = jnp.split(mean, 2, axis=-1)
        ls_left, ls_right = jnp.split(log_std, 2, axis=-1)
        # At the base duration no split/re-encoding is needed.
        closed = (durations <= cuts)[:, None]
        return jnp.where(closed, z, left), right, ls_left, ls_right


class ChunkDecoder(nn.Module):
    action_dim: int
    max_duration: int
    width: int

    @nn.compact
    def __call__(self, initial_states, z, lengths):
        x = jnp.concatenate((initial_states, z, lengths[:, None] / self.max_duration), -1)
        x = MLP((self.width, self.width), activate_final=True)(x)
        mean, log_std = GaussianHead(self.max_duration * self.action_dim)(x)
        shape = (len(z), self.max_duration, self.action_dim)
        return mean.reshape(shape), log_std.reshape(shape)


class SkillFlow(nn.Module):
    latent_dim: int
    max_duration: int
    width: int

    @nn.compact
    def __call__(self, states, goals, lengths, noisy_z, times):
        x = jnp.concatenate(
            (states, goals, lengths[:, None] / self.max_duration, noisy_z, times), -1
        )
        return MLP((self.width, self.width, self.latent_dim), activate_final=False)(x)


class SkillQ(nn.Module):
    max_duration: int
    width: int

    @nn.compact
    def __call__(self, states, z, lengths, goals):
        x = jnp.concatenate((states, z, lengths[:, None] / self.max_duration, goals), -1)
        return MLP((self.width, self.width, 1), activate_final=False)(x)[..., 0]


class SkillNetwork(nn.Module):
    action_dim: int
    latent_dim: int
    max_duration: int
    width: int

    def setup(self):
        self.encoder = SequenceEncoder(self.latent_dim, self.width, self.max_duration)
        self.splitter = SkillSplitter(self.latent_dim, self.width, self.max_duration)
        self.decoder = ChunkDecoder(self.action_dim, self.max_duration, self.width)
        self.proposal = SkillFlow(self.latent_dim, self.max_duration, self.width)
        self.q0 = SkillQ(self.max_duration, self.width)
        self.q1 = SkillQ(self.max_duration, self.width)

    def encode(self, states, actions, lengths):
        return self.encoder(states, actions, lengths)

    def split(self, z, lengths, cuts):
        return self.splitter(z, lengths, cuts)

    def decode(self, states, z, lengths):
        return self.decoder(states, z, lengths)

    def flow(self, states, goals, lengths, z, times):
        return self.proposal(states, goals, lengths, z, times)

    def q(self, states, z, lengths, goals):
        return jnp.stack((self.q0(states, z, lengths, goals), self.q1(states, z, lengths, goals)))

    def __call__(self, states, actions, lengths, goals):
        z, _ = self.encode(states, actions, lengths)
        left, *_ = self.split(z, lengths, jnp.minimum(lengths, 5))
        self.decode(states[:, 0], left, lengths)
        self.flow(states[:, 0], goals, lengths, z, jnp.zeros((len(z), 1)))
        return self.q(states[:, 0], z, lengths, goals)
