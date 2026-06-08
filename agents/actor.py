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


class DeterministicChunkActor(nn.Module):
    """Deterministic actor that outputs a flattened action chunk."""

    hidden_dims: Sequence[int]
    action_dim: int
    layer_norm: bool = True

    @nn.compact
    def __call__(self, observations: jnp.ndarray, goals: jnp.ndarray | None = None) -> jnp.ndarray:
        xs = [observations]
        if goals is not None:
            xs.append(goals)
        x = jnp.concatenate(xs, axis=-1)
        out = MLP((*self.hidden_dims, self.action_dim), activate_final=False, layer_norm=self.layer_norm)(x)
        return jnp.clip(out, -1.0, 1.0)


class StateSubgoalActor(nn.Module):
    """Deterministic state-subgoal actor ``z = pi_Z(s, g)``.

    Outputs an absolute state subgoal.  When ``target_mode='displacement'`` the
    network predicts a delta and the public output is ``s + delta``; with
    ``target_mode='absolute'`` the raw network output is the absolute subgoal.
    The actor never outputs action chunks; execution converts ``z`` into an
    action chunk via bridge planning + IDM.
    """

    hidden_dims: Sequence[int]
    state_dim: int
    layer_norm: bool = True
    target_mode: str = 'absolute'

    @nn.compact
    def __call__(self, observations: jnp.ndarray, goals: jnp.ndarray | None = None) -> jnp.ndarray:
        xs = [observations]
        if goals is not None:
            xs.append(goals)
        x = jnp.concatenate(xs, axis=-1)
        raw = MLP((*self.hidden_dims, self.state_dim), activate_final=False, layer_norm=self.layer_norm)(x)
        if str(self.target_mode).lower() == 'displacement':
            return observations + raw
        return raw


_ANCHOR_METRIC_IDS = {
    'wasserstein_empirical': 0.0,
    'wasserstein_weighted': 1.0,
    'support_nearest': 2.0,
    'support_soft_nearest': 3.0,
}
_METRIC_SPACE_IDS = {
    'raw': 0.0,
    'normalized': 1.0,
    'displacement': 2.0,
}


def finite_anchor_distance(
    z_pi: jnp.ndarray,
    candidate_goals: jnp.ndarray,
    observations: jnp.ndarray,
    mode: str = 'wasserstein_empirical',
    metric_space: str = 'raw',
    state_mean: jnp.ndarray | None = None,
    state_std: jnp.ndarray | None = None,
    softmin_tau: float = 1.0,
    anchor_weights: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, dict]:
    """Proximal distance between a Dirac state actor and a finite anchor set.

    The deterministic actor ``pi = delta_{z_pi}`` and the dynamics proposal
    policy ``pi_0 ~ (1/N) sum_m delta_{z_m}`` (``candidate_goals``).  Under the
    metric ``phi_M`` the default ``wasserstein_empirical`` mode returns the exact
    squared 2-Wasserstein distance between the Dirac mass and the empirical
    anchor distribution, ``(1/N) sum_m ||phi(z_pi) - phi(z_m)||^2``.

    Gradients flow through ``z_pi``; ``candidate_goals`` is stop-gradient.

    Returns
    -------
    dist2 : ``[B]`` squared proximal distance.
    extra : dict with optional diagnostics (e.g. ``softmin_entropy``).
    """
    z = jnp.asarray(z_pi, dtype=jnp.float32)[:, None, :]  # [B, 1, D]
    cand = jax.lax.stop_gradient(jnp.asarray(candidate_goals, dtype=jnp.float32))  # [B, N, D]
    obs = jnp.asarray(observations, dtype=jnp.float32)[:, None, :]  # [B, 1, D]
    metric_space = str(metric_space).lower()
    mode = str(mode).lower()

    # phi(z_pi) - phi(z_m); the offset terms (mu for normalized, s for
    # displacement) cancel in the difference, so we operate directly on diffs.
    if metric_space == 'normalized' and state_mean is not None and state_std is not None:
        denom = jnp.asarray(state_std, dtype=jnp.float32) + 1e-6
        diff = (z - cand) / denom
    elif metric_space == 'displacement':
        # phi(z; s) = z - s  ->  phi(z_pi) - phi(z_m) = z_pi - z_m.
        diff = z - cand + 0.0 * obs
    else:  # 'raw', or 'normalized' without stats (fall back to raw).
        diff = z - cand
    d2 = jnp.sum(diff**2, axis=-1)  # [B, N]

    extra: dict = {}
    if mode == 'wasserstein_empirical':
        dist2 = jnp.mean(d2, axis=1)
    elif mode == 'wasserstein_weighted':
        if anchor_weights is not None:
            w = jax.lax.stop_gradient(jnp.asarray(anchor_weights, dtype=jnp.float32))
            w = w / jnp.maximum(jnp.sum(w, axis=1, keepdims=True), 1e-8)
            dist2 = jnp.sum(w * d2, axis=1)
        else:
            dist2 = jnp.mean(d2, axis=1)
    elif mode == 'support_nearest':
        dist2 = jnp.min(d2, axis=1)
    elif mode == 'support_soft_nearest':
        omega = jax.nn.softmax(-d2 / jnp.maximum(float(softmin_tau), 1e-6), axis=1)
        dist2 = jnp.sum(omega * d2, axis=1)
        extra['softmin_entropy'] = -jnp.sum(omega * jnp.log(omega + 1e-8), axis=1).mean()
    else:
        raise ValueError(
            "state_spi_anchor_metric must be one of 'wasserstein_empirical', "
            "'wasserstein_weighted', 'support_nearest', 'support_soft_nearest', "
            f"got {mode!r}"
        )
    return dist2, extra


class ActorAgent(flax.struct.PyTreeNode):
    """SPI actor trained against an external critic scorer."""

    rng: Any
    actor: Any
    config: Any = nonpytree_field()

    def _actor_type(self) -> str:
        return str(self.config.get('actor_type', 'action_chunk')).lower()

    def _state_goals(self, batch: dict) -> jnp.ndarray:
        """Goal ``g`` for the state-SPI energy ``E(s, z, g)``.

        Uses the true high-level goal so ``V(z, g)`` / ``Q_Z(s, z; g)`` measure
        reachability to the actual goal, not to a provisional subgoal.
        """
        goals = batch.get('high_actor_goals', None)
        if goals is None:
            goals = batch.get('spi_goals', None)
        if goals is None:
            goals = batch.get('value_goals', None)
        if goals is None:
            raise ValueError('state_subgoal actor update requires high_actor_goals/spi_goals.')
        return jnp.asarray(goals, dtype=jnp.float32)

    def _goals(self, batch: dict) -> jnp.ndarray | None:
        # ``spi_goals`` is the dynamics-predicted subgoal (or broadcast) from
        # ``main._build_actor_batch_from_dynamics``. π and actor Q in ``actor_loss``
        # share this vector so the policy stays consistent with the selected subgoal.
        goals = batch.get('spi_goals', None)
        if goals is None:
            goals = batch.get('value_goals', None)
        return None if goals is None else jnp.asarray(goals, dtype=jnp.float32)

    def _proposal_q_goals(self, batch: dict) -> jnp.ndarray | None:
        """Goals used to score proposal action chunks for the SPI target distribution."""
        goals = batch.get('proposal_goals', None)
        if goals is None:
            return self._goals(batch)
        return jnp.asarray(goals, dtype=jnp.float32)

    def _chunk_dim(self) -> int:
        return int(self.config['actor_chunk_horizon']) * int(self.config['action_dim'])

    def _dim_mask(self, batch: dict, chunk_dim: int) -> jnp.ndarray:
        valids = batch.get('valids', None)
        if valids is None:
            return jnp.ones((batch['observations'].shape[0], chunk_dim), dtype=jnp.float32)
        valids = jnp.asarray(valids, dtype=jnp.float32)
        if valids.ndim == 1:
            return jnp.repeat(valids[:, None], chunk_dim, axis=1)
        steps = valids.shape[-1]
        if chunk_dim % steps != 0:
            return jnp.ones((batch['observations'].shape[0], chunk_dim), dtype=jnp.float32)
        rep = chunk_dim // steps
        return jnp.repeat(valids, rep, axis=-1)

    def _proposal_chunks(self, batch: dict) -> jnp.ndarray:
        external = batch.get('proposal_partial_chunks', None)
        if external is None:
            raise ValueError('Actor update requires proposal_partial_chunks in the batch.')
        external = jnp.asarray(external, dtype=jnp.float32)
        if external.ndim == 4:
            external = external.reshape(external.shape[0], external.shape[1], -1)
        if external.ndim != 3:
            raise ValueError(f'proposal_partial_chunks must be rank-3/4, got shape={external.shape}')
        chunk_dim = self._chunk_dim()
        if external.shape[-1] != chunk_dim:
            raise ValueError(f'proposal_partial_chunks last dim must be {chunk_dim}, got {external.shape[-1]}.')
        return jax.lax.stop_gradient(external)

    def actor_loss(self, batch: dict, actor_params: dict, critic_agent: Any) -> tuple[jnp.ndarray, dict]:
        proposal_chunks = self._proposal_chunks(batch)
        goals = self._goals(batch)
        proposal_q_goals = self._proposal_q_goals(batch)
        proposal_q = critic_agent.score_action_chunks(
            batch['observations'],
            proposal_q_goals,
            proposal_chunks,
            network_params=critic_agent.network.params,
            use_partial_critic=True,
        )
        proposal_q = jax.lax.stop_gradient(jnp.asarray(proposal_q, dtype=jnp.float32))
        if proposal_q.shape != proposal_chunks.shape[:2]:
            raise ValueError(
                'proposal Q scores must align with proposal_partial_chunks, '
                f'got q={proposal_q.shape} proposals={proposal_chunks.shape}.'
            )
        rho = jax.nn.softmax(float(self.config['spi_beta']) * proposal_q, axis=1)
        rho = jax.lax.stop_gradient(rho)

        actor_chunk = self.actor(batch['observations'], goals, params=actor_params)
        actor_q = critic_agent.score_action_chunks(
            batch['observations'],
            goals,
            actor_chunk,
            network_params=critic_agent.network.params,
            use_partial_critic=True,
        )[:, 0]

        dim_mask = self._dim_mask(batch, proposal_chunks.shape[-1])
        diff = (actor_chunk[:, None, :] - proposal_chunks) * dim_mask[:, None, :]
        sqdist = jnp.sum(diff**2, axis=-1)
        prox = jnp.sum(rho * sqdist, axis=1)
        # Scale critic Q by batch-mean |Q| so the SPI term is ``-Q / (mean|Q| + eps)``.
        q_eps = jnp.asarray(float(self.config.get('spi_q_norm_eps', 1e-6)), dtype=jnp.float32)
        q_scale = jax.lax.stop_gradient(jnp.mean(jnp.abs(actor_q)) + q_eps)
        actor_q_scaled = actor_q / q_scale
        actor_loss = jnp.mean(-actor_q_scaled + prox / (2.0 * float(self.config['spi_tau'])))

        rho_eps = 1e-8
        rho_entropy = -jnp.sum(rho * jnp.log(rho + rho_eps), axis=1).mean()
        return actor_loss, {
            'spi_actor/actor_loss': actor_loss,
            'spi_actor/proposal_q_mean': proposal_q.mean(),
            'spi_actor/proposal_q_max': proposal_q.max(),
            'spi_actor/proposal_q_min': proposal_q.min(),
            'spi_actor/q_mean': actor_q.mean(),
            'spi_actor/q_max': actor_q.max(),
            'spi_actor/q_min': actor_q.min(),
            'spi_actor/prox_mean': prox.mean(),
            'spi_actor/prox_max': prox.max(),
            'spi_actor/prox_min': prox.min(),
            'spi_actor/rho_entropy': rho_entropy,
            'spi_actor/rho_max_mean': jnp.max(rho, axis=1).mean(),
        }

    def state_actor_loss(self, batch: dict, actor_params: dict, critic_agent: Any) -> tuple[jnp.ndarray, dict]:
        """State-space SPI loss with explicit cost energy + Wasserstein prox.

            L = E(s, pi_Z(s, g), g) + (1 / (2 tau_SPI)) * d_M^2(pi_0, pi).

        ``E`` is a probability-scale cost energy (smaller is better); the
        objective adds it directly (no minus sign). ``d_M^2`` defaults to the
        empirical W2^2 between the Dirac actor and the finite dynamics proposal
        set ``candidate_goals``. Critic params are constants here, so no critic
        gradient flows from the actor update.
        """
        obs = jnp.asarray(batch['observations'], dtype=jnp.float32)
        goals = self._state_goals(batch)
        candidate_goals = jnp.asarray(batch['candidate_goals'], dtype=jnp.float32)  # [B, N, D]

        z_pi = self.actor(obs, goals, params=actor_params)  # [B, D] absolute subgoal

        energy = critic_agent.energy_state_subgoals(
            obs,
            z_pi,
            goals,
            network_params=critic_agent.network.params,
            energy_type=str(critic_agent.config.get('state_spi_energy_type', 'v_product')),
        )[:, 0]

        metric_space = str(self.config.get('state_spi_metric_space', 'raw'))
        state_mean = None
        state_std = None
        if metric_space == 'normalized':
            sm = self.config.get('state_mean', None)
            ss = self.config.get('state_std', None)
            if sm is None or ss is None or len(sm) == 0 or len(ss) == 0:
                raise ValueError(
                    "state_spi_metric_space='normalized' requires state_mean/state_std in the "
                    "actor config (mirrored from dynamics state normalization stats). Set "
                    "state_normalization=true on dynamics or use state_spi_metric_space='raw'."
                )
            state_mean = jnp.asarray(sm, dtype=jnp.float32)
            state_std = jnp.asarray(ss, dtype=jnp.float32)

        anchor_dist2, extra = finite_anchor_distance(
            z_pi,
            candidate_goals,
            obs,
            mode=str(self.config.get('state_spi_anchor_metric', 'wasserstein_empirical')),
            metric_space=metric_space,
            state_mean=state_mean,
            state_std=state_std,
            softmin_tau=float(self.config.get('state_spi_anchor_softmin_tau', 1.0)),
            anchor_weights=batch.get('anchor_weights', None),
        )

        prox_coef = 1.0 / (2.0 * float(self.config['spi_tau']))
        loss = energy.mean() + prox_coef * anchor_dist2.mean()

        info = {
            'state_spi/actor_loss': loss,
            'state_spi/energy_mean': energy.mean(),
            'state_spi/energy_min': energy.min(),
            'state_spi/energy_max': energy.max(),
            'state_spi/anchor_dist2_mean': anchor_dist2.mean(),
            'state_spi/anchor_dist2_min': anchor_dist2.min(),
            'state_spi/anchor_dist2_max': anchor_dist2.max(),
            'state_spi/prox_coef': jnp.asarray(prox_coef, dtype=jnp.float32),
            'state_spi/z_pred_norm': jnp.linalg.norm(z_pi, axis=-1).mean(),
            'state_spi/candidate_goal_norm_mean': jnp.linalg.norm(candidate_goals, axis=-1).mean(),
            'state_spi/anchor_metric_id': jnp.asarray(
                _ANCHOR_METRIC_IDS.get(str(self.config.get('state_spi_anchor_metric', 'wasserstein_empirical')).lower(), -1.0),
                dtype=jnp.float32,
            ),
            'state_spi/metric_space_id': jnp.asarray(
                _METRIC_SPACE_IDS.get(str(self.config.get('state_spi_metric_space', 'raw')).lower(), -1.0),
                dtype=jnp.float32,
            ),
        }
        if 'softmin_entropy' in extra:
            info['state_spi/anchor_softmin_entropy'] = extra['softmin_entropy']
        return loss, info

    def update(self, batch: dict, critic_agent: Any):
        actor_type = self._actor_type()
        if actor_type == 'state_proposal':
            # Nonparametric proposal policy: no learned actor update.
            return self, {'state_spi/actor_loss': jnp.asarray(0.0, dtype=jnp.float32)}
        if actor_type == 'state_subgoal':
            return self._update_state(batch, critic_agent)
        return self._update_action(batch, critic_agent)

    @jax.jit
    def _update_action(self, batch: dict, critic_agent: Any):
        new_rng, _ = jax.random.split(self.rng)
        batch = jax.tree_util.tree_map(lambda x: jnp.asarray(x), batch)

        def loss_fn(actor_params):
            return self.actor_loss(batch, actor_params=actor_params, critic_agent=critic_agent)

        new_actor, info = self.actor.apply_loss_fn(loss_fn=loss_fn)
        return self.replace(rng=new_rng, actor=new_actor), info

    @jax.jit
    def _update_state(self, batch: dict, critic_agent: Any):
        new_rng, _ = jax.random.split(self.rng)
        batch = jax.tree_util.tree_map(lambda x: jnp.asarray(x), batch)

        def loss_fn(actor_params):
            return self.state_actor_loss(batch, actor_params=actor_params, critic_agent=critic_agent)

        new_actor, info = self.actor.apply_loss_fn(loss_fn=loss_fn)
        return self.replace(rng=new_rng, actor=new_actor), info

    def sample_actions(self, observations, goals=None):
        if self._actor_type() != 'action_chunk':
            raise ValueError(
                f"sample_actions is only valid for actor_type='action_chunk'; "
                f"actor_type={self._actor_type()!r} should use sample_subgoals + bridge/IDM."
            )
        observations = jnp.asarray(observations, dtype=jnp.float32)
        squeeze = observations.ndim == 1
        if squeeze:
            observations = observations[None]
            goals = None if goals is None else jnp.asarray(goals, dtype=jnp.float32)[None]
        elif goals is not None:
            goals = jnp.asarray(goals, dtype=jnp.float32)

        chunk = self.actor(observations, goals)
        horizon = int(self.config['actor_chunk_horizon'])
        action_dim = int(self.config['action_dim'])
        chunk = chunk.reshape(chunk.shape[0], horizon, action_dim)
        if squeeze:
            chunk = chunk[0]
        return chunk

    def sample_subgoals(self, observations, goals=None):
        """Return absolute state subgoal ``z = pi_Z(s, g)`` for the learned state actor.

        Only valid for ``actor_type='state_subgoal'``. ``state_proposal`` is a
        nonparametric mode with no learned actor: proposals must be selected
        through dynamics generation + critic state-space energy, never through a
        (randomly initialized / never-trained) actor head.
        """
        if self._actor_type() == 'state_proposal':
            raise ValueError(
                "actor_type='state_proposal' has no learned actor; select subgoals via "
                "dynamics proposals scored by critic_agent.energy_state_subgoals, not sample_subgoals."
            )
        if self._actor_type() != 'state_subgoal':
            raise ValueError(
                f"sample_subgoals requires actor_type='state_subgoal', got {self._actor_type()!r}."
            )
        observations = jnp.asarray(observations, dtype=jnp.float32)
        squeeze = observations.ndim == 1
        if squeeze:
            observations = observations[None]
            goals = None if goals is None else jnp.asarray(goals, dtype=jnp.float32)[None]
        elif goals is not None:
            goals = jnp.asarray(goals, dtype=jnp.float32)
        z = self.actor(observations, goals)
        if squeeze:
            z = z[0]
        return z

    @classmethod
    def create(cls, seed: int, ex_observations: np.ndarray, config: dict, ex_goals: np.ndarray | None = None):
        rng = jax.random.PRNGKey(int(seed))
        rng, init_rng = jax.random.split(rng)
        ex_obs = jnp.asarray(ex_observations, dtype=jnp.float32)
        ex_goal = None if ex_goals is None else jnp.asarray(ex_goals, dtype=jnp.float32)

        config = dict(config)
        actor_type = str(config.get('actor_type', 'action_chunk')).lower()

        if actor_type in ('state_subgoal', 'state_proposal'):
            actor_def = StateSubgoalActor(
                hidden_dims=(512, 512, 512),
                state_dim=int(ex_obs.shape[-1]),
                layer_norm=bool(config['state_spi_actor_layer_norm']),
                target_mode=str(config.get('state_spi_target_mode', 'absolute')),
            )
        else:
            actor_def = DeterministicChunkActor(
                hidden_dims=(512, 512, 512),
                action_dim=int(config['actor_chunk_horizon']) * int(config['action_dim']),
                layer_norm=bool(config['spi_actor_layer_norm']),
            )
        actor_params = actor_def.init(init_rng, ex_obs, ex_goal)['params']
        actor = TrainState.create(actor_def, actor_params, tx=optax.adam(float(config['lr'])))
        return cls(rng=rng, actor=actor, config=flax.core.FrozenDict(**config))


def get_actor_config():
    return ml_collections.ConfigDict(
        dict(
            lr=3e-4,
            spi_tau=5.0,
            spi_beta=1.0,
            spi_actor_layer_norm=True,
            spi_q_norm_eps=1e-6,
            # 'action_chunk' (default): actor outputs an action chunk A^h.
            # 'state_subgoal': actor outputs a state subgoal z = pi_Z(s, g);
            #     trained with E + W2^2/(2 tau); executed via bridge + IDM.
            # 'state_proposal': no learned actor; pick lowest-energy dynamics proposal.
            actor_type='action_chunk',
            # State-subgoal actor settings.
            state_spi_actor_layer_norm=True,
            state_spi_target_mode='absolute',  # 'absolute' | 'displacement'
            state_spi_anchor_metric='wasserstein_empirical',
            state_spi_metric_space='raw',  # 'raw' | 'normalized' | 'displacement'
            state_spi_anchor_softmin_tau=1.0,
            # Mirrored from dynamics state normalization stats by main._prepare_configs /
            # _attach_state_normalization_stats; required when metric_space='normalized'.
            state_mean=(),
            state_std=(),
            # π and Q in the SPI loss are always conditioned on the dynamics-predicted
            # subgoal (``spi_goals`` from ``main._build_actor_batch_from_dynamics``).
            actor_chunk_horizon=ml_collections.config_dict.placeholder(int),
            action_dim=2,
        )
    )
