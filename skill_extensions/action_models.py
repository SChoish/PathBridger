"""Action-only programs. No observation or goal can enter their encoder/reader."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import flax.linen as nn
import jax.numpy as jnp

from utils.networks import MLP


class ActionSequenceEncoder(nn.Module):
    latent_dim: int
    width: int
    max_duration: int

    @nn.compact
    def __call__(self, actions, lengths):
        mask = jnp.arange(actions.shape[1])[None] < lengths[:, None]
        clean = jnp.where(mask[..., None], actions, 0.0)
        features = nn.relu(nn.Dense(self.width)(clean))
        forward = nn.RNN(nn.GRUCell(self.width), name='forward')(features, seq_lengths=lengths)
        backward = nn.RNN(nn.GRUCell(self.width), reverse=True, keep_order=True, name='backward')(
            features, seq_lengths=lengths
        )
        last = jnp.take_along_axis(forward, (lengths - 1)[:, None, None], axis=1)[:, 0]
        pooled = jnp.concatenate(
            (last, backward[:, 0], lengths[:, None] / self.max_duration), axis=-1
        )
        return nn.Dense(self.latent_dim, name='mean')(pooled)


class IntervalReader(nn.Module):
    latent_dim: int
    width: int
    time_scale: int

    @nn.compact
    def __call__(self, root, root_lengths, offsets, lengths):
        times = jnp.stack((root_lengths, offsets, lengths), axis=-1) / self.time_scale
        return MLP((self.width, self.width, self.latent_dim))(
            jnp.concatenate((root, times), axis=-1)
        )


class RecursiveSplitter(nn.Module):
    """Matched deterministic action-loss control; no posterior/prior KL."""

    latent_dim: int
    width: int
    time_scale: int

    @nn.compact
    def __call__(self, code, lengths, cuts):
        times = jnp.stack((lengths, cuts, lengths - cuts), -1) / self.time_scale
        children = MLP((self.width, self.width, 2 * self.latent_dim))(
            jnp.concatenate((code, times), -1)
        )
        left, right = jnp.split(children, 2, axis=-1)
        return jnp.where((lengths <= cuts)[:, None], code, left), right


class ActionProgramNetwork(nn.Module):
    latent_dim: int
    width: int
    program_length: int
    chunk: int
    time_scale: int
    representation: str

    def setup(self):
        self.encoder = ActionSequenceEncoder(self.latent_dim, self.width, self.program_length)
        self.reader = (
            IntervalReader(self.latent_dim, self.width, self.time_scale)
            if self.representation == 'reader'
            else RecursiveSplitter(self.latent_dim, self.width, self.time_scale)
        )
        self.proposal = MLP((self.width, self.width, self.latent_dim))
        self.q0 = MLP((self.width, self.width, 1))
        self.q1 = MLP((self.width, self.width, 1))

    def encode(self, actions, lengths):
        return self.encoder(actions, lengths)

    def leaves(self, root):
        batch = len(root)
        full = jnp.full((batch,), self.program_length, jnp.int32)
        cuts = jnp.full((batch,), self.chunk, jnp.int32)
        result, residual = [], root
        for offset in range(0, self.program_length, self.chunk):
            if self.representation == 'reader':
                child = self.reader(root, full, jnp.full_like(full, offset), cuts)
            else:
                child, residual = self.reader(residual, full - offset, cuts)
            result.append(child)
        return jnp.stack(result, axis=1)

    def flow(self, states, goals, point, times):
        return self.proposal(jnp.concatenate((states, goals, point, times), -1))

    def q(self, states, root, goals):
        inputs = jnp.concatenate((states, root, goals), -1)
        return jnp.stack((self.q0(inputs)[..., 0], self.q1(inputs)[..., 0]))

    def __call__(self, actions, lengths, states, goals):
        root = self.encode(actions, lengths)
        self.leaves(root)
        self.flow(states, goals, root, jnp.zeros((len(root), 1)))
        return self.q(states, root, goals)


@dataclass(frozen=True)
class SkillInterval:
    """A view contains the vector itself, never an archive/source lookup key."""

    root: Any
    root_length: int
    start: int
    stop: int
    chunk: int = 5

    def __post_init__(self):
        if (
            self.chunk < 1
            or not 0 <= self.start < self.stop <= self.root_length
            or any(x % self.chunk for x in (self.start, self.stop, self.root_length))
        ):
            raise ValueError('invalid chunk-aligned interval')

    def split(self, length):
        if (
            not isinstance(length, int)
            or length <= 0
            or length >= self.stop - self.start
            or length % self.chunk
        ):
            raise ValueError('cut must be a strictly interior chunk-aligned length')
        middle = self.start + length
        return (
            SkillInterval(self.root, self.root_length, self.start, middle, self.chunk),
            SkillInterval(self.root, self.root_length, middle, self.stop, self.chunk),
        )
