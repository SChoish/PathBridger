"""SPI-distilled deterministic subgoal network.

This mirrors the SPI actor (``agents/actor.py``) but operates on *subgoal
endpoints* instead of action chunks.  It is the offline "deterministic subgoal
net" of the state-only-offline + online hybrid:

* proposals are subgoal-endpoint candidates sampled from the (frozen-or-
  co-trained) flow subgoal net (``DynamicsAgent.sample_subgoal_candidates``),
* each candidate is scored by the critic's transitive value ratio
  ``V(s,z) * V(z,g) / V(s,g)`` (``CriticAgent.score_transitive_subgoals``),
* the deterministic net ``mu_theta(s, g)`` is pulled toward the score-weighted
  proposals (softmax over scores) while maximizing the transitive value of its
  own prediction.

It never consumes actions, so it belongs to the action-free offline phase.
The SPI objective matches the actor's ``-Q/scale + prox/(2*tau)`` form with the
critic action-Q replaced by the subgoal transitive value.
"""

from __future__ import annotations

from typing import Any, Sequence

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax

from utils.flax_utils import TrainState, nonpytree_field
from utils.networks import MLP


class DeterministicSubgoalNet(nn.Module):
    """Deterministic ``(observation, goal) -> absolute subgoal endpoint`` net."""

    hidden_dims: Sequence[int]
    subgoal_dim: int
    layer_norm: bool = True

    @nn.compact
    def __call__(self, observations: jnp.ndarray, goals: jnp.ndarray | None = None) -> jnp.ndarray:
        xs = [observations]
        if goals is not None:
            xs.append(goals)
        x = jnp.concatenate(xs, axis=-1)
        return MLP((*self.hidden_dims, self.subgoal_dim), activate_final=False, layer_norm=self.layer_norm)(x)


class SubgoalSpiAgent(flax.struct.PyTreeNode):
    """SPI distillation of a flow subgoal net into a deterministic subgoal net."""

    rng: Any
    subgoal: Any
    config: Any = nonpytree_field()

    def predict_subgoal(self, observations, goals=None) -> jnp.ndarray:
        """Return the deterministic subgoal endpoint (absolute frame).

        Handles both single (rank-1) and batched (rank-2) observations so it can
        be used as a drop-in subgoal policy at rollout/eval time.
        """
        observations = jnp.asarray(observations, dtype=jnp.float32)
        squeeze = observations.ndim == 1
        if squeeze:
            observations = observations[None]
            goals = None if goals is None else jnp.asarray(goals, dtype=jnp.float32)[None]
        elif goals is not None:
            goals = jnp.asarray(goals, dtype=jnp.float32)
        out = self.subgoal(observations, goals)
        if squeeze:
            out = out[0]
        return out

    def infer_subgoal(self, observations, goals=None) -> jnp.ndarray:
        """Alias matching ``DynamicsAgent.infer_subgoal`` so eval/rollout code can
        treat this agent as the subgoal policy."""
        return self.predict_subgoal(observations, goals)

    def subgoal_loss(self, batch: dict, subgoal_params: dict, critic_agent: Any) -> tuple[jnp.ndarray, dict]:
        obs = jnp.asarray(batch['observations'], dtype=jnp.float32)
        goals = jnp.asarray(batch['high_actor_goals'], dtype=jnp.float32)
        candidates = jax.lax.stop_gradient(jnp.asarray(batch['subgoal_candidates'], dtype=jnp.float32))  # [B, N, D]
        scores = batch.get('subgoal_scores', None)
        if scores is None:
            raise ValueError(
                'SubgoalSpiAgent requires subgoal_scores precomputed from the current critic snapshot. '
                'Score the flow proposals in the training loop before the subgoal-spi update.'
            )
        scores = jax.lax.stop_gradient(jnp.asarray(scores, dtype=jnp.float32))  # [B, N]
        if scores.ndim != 2:
            raise ValueError(f'subgoal_scores must be rank-2 [B, N], got shape={scores.shape}')
        if scores.shape[:2] != candidates.shape[:2]:
            raise ValueError(
                'subgoal_scores must align with subgoal_candidates, '
                f'got scores={scores.shape} candidates={candidates.shape}.'
            )

        rho = jax.nn.softmax(float(self.config['spi_beta']) * scores, axis=1)
        rho = jax.lax.stop_gradient(rho)

        mu = self.subgoal(obs, goals, params=subgoal_params)  # [B, D]
        # Transitive value of the deterministic subgoal (gradients flow through mu).
        mu_value = critic_agent.score_transitive_subgoals(
            obs,
            mu[:, None, :],
            goals,
            network_params=critic_agent.network.params,
        )[:, 0]

        diff = mu[:, None, :] - candidates
        sqdist = jnp.sum(diff**2, axis=-1)
        prox = jnp.sum(rho * sqdist, axis=1)
        v_eps = jnp.asarray(float(self.config.get('spi_q_norm_eps', 1e-6)), dtype=jnp.float32)
        v_scale = jax.lax.stop_gradient(jnp.mean(jnp.abs(mu_value)) + v_eps)
        value_scaled = mu_value / v_scale
        loss = jnp.mean(-value_scaled + prox / (2.0 * float(self.config['spi_tau'])))

        rho_eps = 1e-8
        rho_entropy = -jnp.sum(rho * jnp.log(rho + rho_eps), axis=1).mean()
        return loss, {
            'subgoal_spi/loss': loss,
            'subgoal_spi/value_mean': mu_value.mean(),
            'subgoal_spi/value_max': mu_value.max(),
            'subgoal_spi/value_min': mu_value.min(),
            'subgoal_spi/prox_mean': prox.mean(),
            'subgoal_spi/prox_max': prox.max(),
            'subgoal_spi/score_mean': scores.mean(),
            'subgoal_spi/rho_entropy': rho_entropy,
            'subgoal_spi/rho_max_mean': jnp.max(rho, axis=1).mean(),
        }

    @jax.jit
    def update(self, batch: dict, critic_agent: Any):
        new_rng, _ = jax.random.split(self.rng)
        batch = jax.tree_util.tree_map(lambda x: jnp.asarray(x), batch)

        def loss_fn(subgoal_params):
            return self.subgoal_loss(batch, subgoal_params=subgoal_params, critic_agent=critic_agent)

        new_subgoal, info = self.subgoal.apply_loss_fn(loss_fn=loss_fn)
        return self.replace(rng=new_rng, subgoal=new_subgoal), info

    @classmethod
    def create(cls, seed: int, ex_observations: np.ndarray, config: dict, ex_goals: np.ndarray | None = None):
        rng = jax.random.PRNGKey(int(seed))
        rng, init_rng = jax.random.split(rng)
        ex_obs = jnp.asarray(ex_observations, dtype=jnp.float32)
        ex_goal = None if ex_goals is None else jnp.asarray(ex_goals, dtype=jnp.float32)

        config = dict(config)
        subgoal_dim = int(config.get('subgoal_dim') or ex_obs.shape[-1])
        config['subgoal_dim'] = subgoal_dim

        subgoal_def = DeterministicSubgoalNet(
            hidden_dims=tuple(int(x) for x in config['subgoal_spi_hidden_dims']),
            subgoal_dim=subgoal_dim,
            layer_norm=bool(config['subgoal_spi_layer_norm']),
        )
        subgoal_params = subgoal_def.init(init_rng, ex_obs, ex_goal)['params']
        subgoal = TrainState.create(subgoal_def, subgoal_params, tx=optax.adam(float(config['lr'])))
        return cls(rng=rng, subgoal=subgoal, config=flax.core.FrozenDict(**config))


def get_subgoal_spi_config():
    return ml_collections.ConfigDict(
        dict(
            lr=3e-4,
            spi_tau=5.0,
            spi_beta=1.0,
            spi_q_norm_eps=1e-6,
            subgoal_spi_layer_norm=True,
            subgoal_spi_hidden_dims=(512, 512, 512),
            # Number of flow subgoal candidates used as SPI proposals per update.
            subgoal_spi_num_samples=8,
            # Resolved to the observation dim at create() time.
            subgoal_dim=ml_collections.config_dict.placeholder(int),
        )
    )


__all__ = [
    'DeterministicSubgoalNet',
    'SubgoalSpiAgent',
    'get_subgoal_spi_config',
]
