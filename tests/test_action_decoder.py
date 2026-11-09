"""From-scratch prerequisites; synthetic CPU checks are not environment evidence."""

import argparse
import inspect
import json
import pickle

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from action_skill_main import run_train
from skill_extensions.action_agent import ActionProgramAgent
from skill_extensions.action_data import ActionWindowSampler
from skill_extensions.action_decoder import ShortActionAutoencoder
from skill_extensions.action_runtime import digest, import_decoder
from utils.datasets import Dataset


def same_tree(left, right):
    assert jax.tree_util.tree_structure(left) == jax.tree_util.tree_structure(right)
    for a, b in zip(jax.tree_util.tree_leaves(left), jax.tree_util.tree_leaves(right), strict=True):
        np.testing.assert_array_equal(a, b)


def args_for(tmp_path):
    rng = np.random.RandomState(10)
    for name in ('train', 'validation'):
        actions = rng.normal(size=(48, 2)).astype(np.float32)
        actions[[23, 47]] = np.nan
        np.savez(
            tmp_path / f'{name}.npz',
            observations=rng.normal(size=(48, 3)).astype(np.float32),
            actions=actions,
            terminals=np.tile(np.r_[np.zeros(23), 1], 2).astype(np.float32),
        )
    config = tmp_path / 'decoder.json'
    config.write_text(json.dumps(dict(latent_dim=4, width=8)))
    return argparse.Namespace(
        stage='decoder',
        train_npz=tmp_path / 'train.npz',
        validation_npz=tmp_path / 'validation.npz',
        output=tmp_path / 'decoder',
        config=config,
        donor=None,
        donor_sha256=None,
        input_checkpoint=None,
        input_sha256=None,
        goal_config=None,
        goal_contract=None,
        boundary_kind=None,
        gate_report=None,
        gate_sha256=None,
        seed=0,
        updates=1,
        batch_size=3,
        checkpoint_every=1,
        smoke=True,
    )


def test_short_window_includes_episode_tail_without_consuming_dummy_action():
    data = Dataset.create(
        observations=np.arange(24, dtype=np.float32).reshape(8, 3),
        actions=np.r_[np.ones((7, 2)), [[np.nan, np.nan]]],
        terminals=np.r_[np.zeros(7), 1],
    )
    sampler = ActionWindowSampler(data, seed=0, horizon=5)
    np.testing.assert_array_equal(sampler.starts, [0, 1, 2])
    tail = sampler.windows([2])
    np.testing.assert_array_equal(tail['states'][0, -1], data['observations'][7])
    assert tail['actions'].shape == (1, 5, 2) and np.isfinite(tail['actions']).all()
    with pytest.raises(ValueError, match='cross-episode'):
        sampler.windows([3])


def test_short_encoder_only_accepts_actions():
    assert tuple(inspect.signature(ShortActionAutoencoder.encode).parameters) == ('self', 'actions')
    model = ShortActionAutoencoder(2, 4, 8)
    actions = jnp.arange(20, dtype=jnp.float32).reshape(2, 5, 2) / 20
    params = model.init(jax.random.PRNGKey(0), jnp.zeros((2, 3)), actions)
    code = model.apply(params, actions, method=model.encode)
    assert code.shape == (2, 4) and np.isfinite(code).all()


def test_no_checkpoint_bootstrap_resume_and_frozen_donor_handoff(tmp_path):
    args = args_for(tmp_path)
    first = run_train(args)
    receipt = first['last_checkpoint']
    with pytest.raises(ValueError, match='non-smoke'):
        import_decoder(receipt['path'], receipt['sha256'])
    asset = import_decoder(receipt['path'], receipt['sha256'], allow_smoke=True)
    agent = ActionProgramAgent.create(asset, config=dict(latent_dim=4, width=8))
    assert agent.decode(jnp.zeros((2, 3)), jnp.zeros((2, 4))).shape == (2, 5, 2)
    assert asset['max_duration'] == 5 and first['environment_steps'] == 0
    with open(receipt['path'], 'rb') as stream:
        payload = pickle.load(stream)
    assert set(payload['state']['params']) == {'encoder', 'decoder'}
    assert payload['metadata']['training_kind'] == 'action-only-five-step-autoencoder'
    assert payload['decoder_step'] == 1
    same_tree(payload['state']['params']['decoder'], asset['params'])
    initial = ShortActionAutoencoder(2, 4, 8).init(
        jax.random.PRNGKey(0), jnp.zeros((1, 3)), jnp.zeros((1, 5, 2))
    )['params']
    for name in ('encoder', 'decoder'):
        assert any(
            not np.array_equal(a, b)
            for a, b in zip(
                jax.tree_util.tree_leaves(initial[name]),
                jax.tree_util.tree_leaves(payload['state']['params'][name]),
                strict=True,
            )
        )

    # Independent uninterrupted and restored two-update runs must match optimizer/RNG.
    args.updates, args.output = 2, tmp_path / 'uninterrupted'
    whole = run_train(args)
    args.input_checkpoint, args.input_sha256 = receipt['path'], receipt['sha256']
    args.config, args.output = None, tmp_path / 'resumed'
    resumed = run_train(args)
    with open(whole['last_checkpoint']['path'], 'rb') as stream:
        whole_payload = pickle.load(stream)
    with open(resumed['last_checkpoint']['path'], 'rb') as stream:
        resumed_payload = pickle.load(stream)
    same_tree(whole_payload['state'], resumed_payload['state'])
    assert pickle.dumps(whole_payload['sampler_rng']) == pickle.dumps(
        resumed_payload['sampler_rng']
    )
    assert whole['fixed_validation'] == resumed['fixed_validation']
    assert digest(resumed['last_checkpoint']['path']) == resumed['last_checkpoint']['sha256']

    before = agent.decoder_params
    args.stage, args.output = 'representation', tmp_path / 'representation'
    args.input_checkpoint = args.input_sha256 = None
    args.donor, args.donor_sha256 = receipt['path'], receipt['sha256']
    args.config = tmp_path / 'program.json'
    args.config.write_text(json.dumps(dict(latent_dim=4, width=8)))
    representation = run_train(args)
    from skill_extensions.action_runtime import restore

    trained, _ = restore(
        representation['last_checkpoint']['path'],
        expected_sha256=representation['last_checkpoint']['sha256'],
    )
    same_tree(before, trained.decoder_params)
    np.testing.assert_array_equal(payload['mean'], trained.observation_mean)
    np.testing.assert_array_equal(payload['scale'], trained.observation_scale)
