"""Numerical/contract verification only; synthetic checks are not RL performance."""

import argparse
import inspect
import json
import pickle

import flax
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from action_skill_main import root_statistics, run_train
from skill_extensions.action_agent import ActionProgramAgent, expected_max_target
from skill_extensions.action_data import (
    ActionWindowSampler,
    add_value_labels,
    dataset_contract,
    diagnostic_full_state_linf,
)
from skill_extensions.action_diagnostics import execute_fixed_program, recorded_state_errors
from skill_extensions.action_models import ActionSequenceEncoder, SkillInterval
from skill_extensions.action_policy import CommittedActionPolicy
from skill_extensions.action_runtime import digest, import_decoder, restore, save, tree_digest
from skill_extensions.models import ChunkDecoder
from utils.datasets import Dataset


def same_tree(a, b):
    assert jax.tree_util.tree_structure(a) == jax.tree_util.tree_structure(b)
    for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b), strict=True):
        np.testing.assert_array_equal(x, y)


def changed(a, b):
    return any(
        not np.array_equal(x, y)
        for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b), strict=True)
    )


@pytest.fixture(scope='module')
def asset():
    decoder = ChunkDecoder(2, 40, 8)
    params = decoder.init(
        jax.random.PRNGKey(4), jnp.zeros((1, 4)), jnp.zeros((1, 4)), jnp.array([5])
    )['params']
    return dict(
        params=params,
        latent_dim=4,
        width=8,
        max_duration=40,
        observation_dim=4,
        action_dim=2,
        mean=np.zeros(4),
        scale=np.ones(4),
        donor_smoke=True,
        donor_sha256='synthetic',
        decoder_parameter_sha256=tree_digest(params),
        donor_representation_step=1,
    )


def config(mode='reader'):
    return dict(latent_dim=4, width=8, representation=mode, flow_steps=2, candidate_count=3)


def batch(seed=3):
    rng = np.random.RandomState(seed)
    states = rng.normal(size=(3, 21, 4)).astype(np.float32)
    return dict(
        actions=rng.uniform(-0.5, 0.5, (3, 20, 2)).astype(np.float32),
        states=states,
        goals=states[:, 20],
    )


def dataset(seed=0):
    rng = np.random.RandomState(seed)
    actions = rng.uniform(-1, 1, (48, 2)).astype(np.float32)
    actions[[23, 47]] = np.nan  # Compact terminal dummy actions must never be consumed.
    return Dataset.create(
        observations=rng.normal(size=(48, 4)).astype(np.float32),
        actions=actions,
        terminals=np.tile(np.r_[np.zeros(23), 1], 2).astype(np.float32),
    )


def goal_config():
    return dict(
        horizon=5,
        discount=0.99,
        actor_p=[0, 0, 1, 0],
        critic_p=[0, 1, 0, 0],
        training_objective_protocol='manuscript-v1',
        env_name='synthetic-diagnostic',
    )


def test_encoder_has_no_state_input_and_ignores_poisoned_padding():
    assert tuple(inspect.signature(ActionSequenceEncoder.__call__).parameters) == (
        'self',
        'actions',
        'lengths',
    )
    model = ActionSequenceEncoder(4, 8, 20)
    actions = jnp.asarray(batch()['actions'])
    lengths = jnp.array([5, 10, 20])
    params = model.init(jax.random.PRNGKey(0), actions, lengths)
    expected = model.apply(params, actions, lengths)
    poisoned = jnp.where((jnp.arange(20)[None] < lengths[:, None])[..., None], actions, jnp.nan)
    np.testing.assert_array_equal(expected, model.apply(params, poisoned, lengths))
    assert np.isfinite(expected).all()


def test_root_is_independent_of_recorded_states_and_goals(asset):
    agent = ActionProgramAgent.create(asset, seed=0, config=config())
    original = batch()
    changed_context = {
        **original,
        'states': original['states'] + 100,
        'goals': -original['goals'],
    }
    same_tree(
        agent.encode_actions(original['actions']),
        agent.encode_actions(changed_context['actions']),
    )
    # Negative control: observations still matter in the frozen conditional executor.
    a = agent.representation_loss(agent.state.params, original)[0]
    b = agent.representation_loss(agent.state.params, changed_context)[0]
    assert float(a) != float(b)


def test_interval_paths_agree_without_parent_rewriting():
    root = object()
    whole = SkillInterval(root, 20, 0, 20)
    _, right_half = whole.split(10)
    by_half, _ = right_half.split(5)
    _, suffix = whole.split(5)
    _, suffix = suffix.split(5)
    by_prefix, _ = suffix.split(5)
    assert (by_half.start, by_half.stop) == (by_prefix.start, by_prefix.stop) == (10, 15)
    assert by_half.root is by_prefix.root is root
    with pytest.raises(ValueError, match='chunk-aligned'):
        whole.split(7)


def test_action_only_initialization_is_shared_between_arms(asset):
    a = ActionProgramAgent.create(asset, seed=7, config=config('recursive'))
    b = ActionProgramAgent.create(asset, seed=7, config=config('reader'))
    same_tree(a.state.params['encoder'], b.state.params['encoder'])
    same_tree(a.decoder_params, b.decoder_params)


def test_frozen_decoder_keeps_the_code_derivative(asset):
    agent = ActionProgramAgent.create(asset, config=config())
    states = jnp.asarray(batch()['states'][:1, 0])
    code = jnp.full((1, 4), 0.2)

    def objective(z):
        return jnp.sum(agent.decode(states, z) ** 2)

    gradient = jax.grad(objective)(code)
    coordinate = np.unravel_index(np.argmax(np.abs(gradient)), gradient.shape)
    delta = jnp.zeros_like(code).at[coordinate].set(1e-3)
    numerical = (objective(code + delta) - objective(code - delta)) / 2e-3
    assert abs(float(gradient[coordinate])) > 1e-6
    np.testing.assert_allclose(gradient[coordinate], numerical, rtol=2e-2, atol=2e-4)


@pytest.mark.parametrize('mode', ['reader', 'recursive'])
def test_stages_change_only_owned_parameters_and_target_ema(asset, mode):
    original = ActionProgramAgent.create(asset, config=config(mode))
    represented, metrics = original.update(batch(), stage='representation')
    assert np.isfinite([float(x) for x in metrics.values()]).all()
    assert changed(original.state.params['encoder'], represented.state.params['encoder'])
    assert changed(original.state.params['reader'], represented.state.params['reader'])
    for name in ('proposal', 'q0', 'q1'):
        same_tree(original.state.params[name], represented.state.params[name])
    same_tree(original.decoder_params, represented.decoder_params)
    flow = represented.seal_representation(np.zeros(4), np.ones(4))
    with pytest.raises(ValueError, match='cannot update'):
        flow.update(batch(), stage='representation')
    flowed, _ = flow.update(batch(), stage='proposal')
    assert changed(flow.state.params['proposal'], flowed.state.params['proposal'])
    for name in ('encoder', 'reader', 'q0', 'q1'):
        same_tree(flow.state.params[name], flowed.state.params[name])
    value = flowed.seal_proposal()
    labelled = add_value_labels(
        {**batch(), 'true_terminals': np.array([False, False, True])},
        diagnostic_full_state_linf,
        {'tolerance': 1e-5},
        0.99,
    )
    updated, metrics = value.update(labelled, stage='value')
    assert np.isfinite([float(x) for x in metrics.values()]).all()
    for name in ('encoder', 'reader', 'proposal'):
        same_tree(value.state.params[name], updated.state.params[name])
    same_tree(value.decoder_params, updated.decoder_params)
    assert changed(value.state.params['q0'], updated.state.params['q0'])
    tau = value.config['target_rate']
    expected = jax.tree_util.tree_map(
        lambda old, new: (1 - tau) * old + tau * new,
        value.state.targets,
        flax.core.freeze({name: updated.state.params[name] for name in ('q0', 'q1')}),
    )
    for x, y in zip(
        jax.tree_util.tree_leaves(expected),
        jax.tree_util.tree_leaves(updated.state.targets),
        strict=True,
    ):
        # JIT may fuse multiply/add; independently computed float32 EMA can differ by 1 ULP.
        np.testing.assert_allclose(x, y, rtol=2e-7, atol=2e-8)


def test_expected_max_target_hand_computation_and_no_target_gradient():
    scores = jnp.array(
        [
            [[0.9, 0.3], [0.8, 0.4], [0.9, 0.3], [0.2, 0.5]],
            [[0.2, 0.7], [0.3, 0.7], [0.4, 0.8], [0.6, 0.4]],
        ]
    )
    hits = jnp.array([3, -1, -1, 0])
    terminal = jnp.array([False, False, True, True])
    actual = expected_max_target(hits, terminal, scores, 0.9)
    np.testing.assert_allclose(actual, [0.9**3, 0.9**20 * 0.4, 0, 1], rtol=1e-6)
    np.testing.assert_array_equal(
        jax.grad(lambda x: expected_max_target(hits, terminal, x, 0.9).sum())(scores),
        jnp.zeros_like(scores),
    )


def test_sampler_boundary_masks_and_exact_rng_resume():
    data = dataset()
    sampler = ActionWindowSampler(data, seed=4, goal_config=goal_config(), boundary_kind='timeout')
    saved = sampler.get_rng_state()
    first = sampler.sample(30)
    sampler.set_rng_state(saved)
    same_tree(first, sampler.sample(30))
    assert np.isfinite(first['actions']).all()
    assert not first['true_terminals'].any()
    for row in first['states']:
        index = np.flatnonzero(np.all(data['observations'] == row[0], axis=1))[0]
        assert index // 24 == (index + 20) // 24
    with pytest.raises(ValueError, match='cross-episode'):
        sampler.windows([10])
    with pytest.raises(ValueError, match='identical complete episodes'):
        dataset_contract(data, data)
    terminal = ActionWindowSampler(data, seed=0, boundary_kind='terminal')
    assert terminal.windows([3, 27])['true_terminals'].all()


def test_training_statistics_do_not_consume_sampler_rng(asset):
    agent = ActionProgramAgent.create(asset, config=config())
    represented, _ = agent.update(batch(), stage='representation')
    sampler = ActionWindowSampler(dataset(), seed=3, goal_config=goal_config())
    rng_before = pickle.dumps(sampler.get_rng_state())
    mean, scale = root_statistics(represented, sampler, batch_size=3)
    assert pickle.dumps(sampler.get_rng_state()) == rng_before
    roots = represented.encode_actions(sampler.windows(sampler.starts)['actions'])
    np.testing.assert_allclose(mean, np.asarray(roots).mean(0), rtol=2e-6, atol=2e-7)
    np.testing.assert_allclose(scale, np.maximum(np.asarray(roots).std(0), 1e-3), rtol=2e-6)


def test_resume_restores_optimizer_parameters_and_rng_exactly(asset, tmp_path):
    agent = ActionProgramAgent.create(asset, config=config())
    data = dataset()
    sampler = ActionWindowSampler(data, seed=8)
    trained, _ = agent.update(sampler.sample(3), stage='representation')
    path = tmp_path / 'checkpoint.pkl'
    save(path, trained, sampler, asset, {'seed': 0, 'smoke': True}, smoke=True)
    resumed, payload = restore(path, expected_sha256=digest(path))
    repeated_sampler = ActionWindowSampler(data, seed=0)
    repeated_sampler.set_rng_state(payload['sampler_rng'])
    uninterrupted, _ = trained.update(sampler.sample(3), stage='representation')
    continued, _ = resumed.update(repeated_sampler.sample(3), stage='representation')
    same_tree(uninterrupted.state, continued.state)
    with pytest.raises(ValueError, match='SHA256'):
        restore(path, expected_sha256='incorrect')
    with pytest.raises(ValueError, match='non-smoke'):
        restore(path, expected_sha256=digest(path), require_deployable=True)


def test_policy_commitment_and_five_step_observation_refresh():
    class FakeAgent:
        action_dim, observation_dim = 2, 4

        def __init__(self):
            self.plans, self.reads = 0, []

        def select_program(self, states, goals, key):
            self.plans += 1
            return jnp.zeros((1, 4)), {}

        def decode_program_chunk(self, states, root, index):
            self.reads.append((float(states[0, 0]), index))
            return jnp.full((1, 5, 2), states[0, 0] + index)

    agent = FakeAgent()
    policy = CommittedActionPolicy(
        agent,
        seed=0,
        action_low=np.full(2, -100),
        action_high=np.full(2, 100),
    )
    observed = [policy.action(np.array([step, 0, 0, 0]), np.zeros(4))[0] for step in range(26)]
    assert agent.plans == policy.plan_count == 2
    assert agent.reads == [(0, 0), (5, 1), (10, 2), (15, 3), (20, 0), (25, 1)]
    assert observed[:10] == [0] * 5 + [6] * 5
    with pytest.raises(ValueError, match='goal changed'):
        policy.action(np.zeros(4), np.ones(4))


def test_g1_runs_only_one_fixed_program_and_stops_on_first_hit(asset):
    agent = ActionProgramAgent.create(asset, config=config())
    errors = recorded_state_errors(agent, batch())
    assert errors['status'] == 'descriptive_only_not_gate_pass'
    assert len(errors['recorded_action_mse_by_chunk']) == 4

    class Space:
        low, high = np.full(2, -1), np.full(2, 1)

    class Env:
        action_space = Space()

        def __init__(self):
            self.steps = 0

        def step(self, action):
            self.steps += 1
            return (
                np.full(4, self.steps, np.float32),
                0,
                False,
                False,
                {
                    'success': self.steps == 7,
                },
            )

    env = Env()
    result = execute_fixed_program(env, agent, errors['root'][0], np.zeros(4))
    assert env.steps == result['first_hit_step'] == 7
    assert result['boundary_steps'] == [0, 5, 7]
    assert result['replanning_calls'] == 0


def test_cli_three_stage_smoke_and_donor_guards(asset, tmp_path):
    train, valid = dataset(20), dataset(21)
    for name, data in (('train', train), ('validation', valid)):
        np.savez(tmp_path / f'{name}.npz', **dict(data))
    donor_path = tmp_path / 'donor.pkl'
    donor = dict(
        format='pathbridger-variable-skill-v1',
        representation_step=1,
        smoke=True,
        config=dict(latent_dim=4, width=8, durations=list(range(5, 41, 5))),
        state={'params': {'decoder': asset['params']}},
        mean=asset['mean'],
        scale=asset['scale'],
        observation_dim=4,
        action_dim=2,
    )
    donor_path.write_bytes(pickle.dumps(donor))
    with pytest.raises(ValueError, match='non-smoke'):
        import_decoder(donor_path, digest(donor_path))
    tiny = tmp_path / 'tiny.json'
    tiny.write_text(json.dumps(config()))
    goals = tmp_path / 'goals.json'
    goals.write_text(json.dumps(goal_config()))
    goal_contract = tmp_path / 'goal_contract.json'
    goal_contract.write_text(
        json.dumps(
            dict(
                callable='skill_extensions.action_data:diagnostic_full_state_linf',
                parameters={'tolerance': 1e-5},
                environment='synthetic-diagnostic',
                validation_evidence='synthetic test only; not an OGBench adapter',
            )
        )
    )
    common = dict(
        train_npz=tmp_path / 'train.npz',
        validation_npz=tmp_path / 'validation.npz',
        donor=donor_path,
        donor_sha256=digest(donor_path),
        input_checkpoint=None,
        input_sha256=None,
        config=tiny,
        seed=0,
        updates=1,
        batch_size=3,
        checkpoint_every=1,
        smoke=True,
        goal_config=None,
        goal_contract=None,
        boundary_kind=None,
        gate_report=None,
        gate_sha256=None,
    )
    reports = []
    for stage in ('representation', 'proposal', 'value'):
        args = argparse.Namespace(
            **{
                **common,
                'stage': stage,
                'output': tmp_path / stage,
            }
        )
        report = run_train(args)
        assert report['status'] == 'completed' and report['environment_steps'] == 0
        checkpoint = report['last_checkpoint']
        reports.append(report)
        common.update(
            input_checkpoint=checkpoint['path'],
            input_sha256=checkpoint['sha256'],
            config=None,
            goal_config=goals,
        )
        if stage == 'proposal':
            common.update(goal_contract=goal_contract, boundary_kind='timeout')
    final, _ = restore(
        reports[-1]['last_checkpoint']['path'],
        expected_sha256=reports[-1]['last_checkpoint']['sha256'],
    )
    assert [
        int(getattr(final.state, f'{name}_step'))
        for name in ('representation', 'proposal', 'value')
    ] == [1, 1, 1]
