"""Light smoke tests for critic modes (dqc / iql / trl)."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
import jax
import jax.numpy as jnp

from agents.critic import CriticAgent, get_config as get_critic_config, validate_config
from utils.critic_sequence_dataset import CriticSequenceDataset
from utils.datasets import Dataset

STATE_DIM = 6
ACTION_DIM = 3
BATCH = 4


def _dummy_dataset():
    n = 4 * 40
    rng = np.random.default_rng(0)
    obs = rng.standard_normal((n, STATE_DIM)).astype(np.float32)
    actions = rng.uniform(-1.0, 1.0, size=(n, ACTION_DIM)).astype(np.float32)
    terminals = np.zeros((n,), dtype=np.float32)
    for k in range(4):
        terminals[(k + 1) * 40 - 1] = 1.0
    return Dataset.create(observations=obs, actions=actions, terminals=terminals)


def _cfg(critic_type: str):
    cfg = get_critic_config()
    cfg.action_chunk_horizon = 4
    cfg.full_chunk_horizon = 8
    cfg.value_hidden_dims = (16, 16)
    cfg.action_dim = ACTION_DIM
    cfg.frame_stack = None
    cfg.critic_type = critic_type
    if critic_type in ('iql', 'trl'):
        cfg.use_chunk_critic = False
    if critic_type == 'trl':
        cfg.algorithm = 'trl'
        cfg.goal_representation = 'full'
        cfg.value_base_horizon = 4
    return cfg


def _build(critic_type: str):
    cfg = _cfg(critic_type)
    batch = CriticSequenceDataset(_dummy_dataset(), cfg).sample(BATCH)
    critic = CriticAgent.create(
        seed=0,
        ex_observations=batch['observations'],
        ex_full_chunk_actions=batch['full_chunk_actions'] if critic_type == 'dqc' else None,
        ex_action_chunk_actions=batch['action_chunk_actions'],
        config=cfg,
        ex_goals=batch['value_goals'],
    )
    return critic, batch


@pytest.mark.parametrize('critic_type', ('dqc', 'iql', 'trl'))
def test_critic_mode_update_and_scoring(critic_type):
    critic, batch = _build(critic_type)
    if critic_type == 'dqc':
        assert 'modules_chunk_critic' in critic.network.params
    else:
        assert 'modules_chunk_critic' not in critic.network.params
    _, info = critic.update(batch)
    assert np.isfinite(float(info['total_loss']))
    scores = critic.score_action_chunks(
        jnp.asarray(batch['observations']),
        jnp.asarray(batch['value_goals']),
        np.random.uniform(
            -1, 1,
            size=(BATCH, 2, int(critic.config['action_chunk_horizon']), ACTION_DIM),
        ).astype(np.float32),
        network_params=critic.network.params,
    )
    assert scores.shape == (BATCH, 2)


def test_trl_transitive_contract():
    critic, batch = _build('trl')
    assert critic._is_trl()
    for key in ('trans_v_valid_mask', 'value_base_goals', 'q_goals', 'trans_v_split_offsets'):
        assert key in batch

    ds = CriticSequenceDataset(_dummy_dataset(), critic.config)
    fields = ds._sample_trl_fields(ds.valid_starts[:BATCH], ds.valid_starts[:BATCH] + 1)
    assert np.all(fields['trans_v_valid_mask'] == 0.0)

    ex_batch = dict(batch)
    ex_batch['trans_v_valid_mask'] = np.zeros_like(batch['trans_v_valid_mask'])
    _, info = critic.update(ex_batch)
    assert float(info['value/tri_loss']) == 0.0

    def loss_fn(params):
        return critic.total_loss(batch, params)[0]

    grads = jax.grad(loss_fn)(critic.network.params)
    leaf_sum = lambda tree: float(sum(np.asarray(jnp.sum(jnp.abs(x))) for x in jax.tree_util.tree_leaves(tree)))
    assert leaf_sum(grads['modules_value']) > 0.0
    assert leaf_sum(grads['modules_target_value']) == 0.0


def test_validate_config():
    for critic_type in ('iql', 'trl'):
        cfg = _cfg(critic_type)
        cfg.use_chunk_critic = True
        validate_config(cfg)
        assert bool(cfg.use_chunk_critic) is False

    cfg = _cfg('dqc')
    cfg.critic_type = 'direct_chunk_trl'
    validate_config(cfg)
    assert cfg['critic_type'] == 'trl'

    cfg = _cfg('dqc')
    cfg.critic_type = 'foo'
    with pytest.raises(ValueError, match='critic_type'):
        validate_config(cfg)
