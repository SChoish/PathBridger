"""Numerical/contract tests with an explicit fake PB teacher, not OGBench scores."""

import json

import flax
import flax.struct
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from skill_extensions.planner import (
    SkillPlanner,
    first_hit,
    return_target,
    validate_config,
)
from skill_extensions.runtime import restore
from utils.datasets import Dataset
from utils.skill_data import SkillDataset, episode_fingerprints, fit_skill_normalization


@flax.struct.dataclass
class FakeNetwork:
    params: jax.Array

    def select(self, name):
        assert name == 'value', 'skill path must never call the PB bridge/IDM'
        return lambda s, g: 1.0 - jnp.sum((s - g) ** 2, axis=-1) + self.params


@flax.struct.dataclass
class FakePB:
    network: FakeNetwork
    config: object = flax.struct.field(pytree_node=False)

    def _sample_endpoint_candidates(
        self, observations, goals, seed, *, num_candidates, temperature
    ):
        del seed, temperature
        weight = jnp.linspace(0.25, 0.75, num_candidates)[None, :, None]
        return observations[:, None] + weight * (goals - observations)[:, None]


def pb():
    return FakePB(
        FakeNetwork(jnp.float32(0.1)),
        flax.core.FrozenDict(
            dict(
                horizon=10,
                discount=0.99,
                actor_p=(0.0, 0.0, 1.0, 0.0),
                critic_p=(0.0, 1.0, 0.0, 0.0),
                training_objective_protocol='manuscript-v1',
                env_name='antmaze-medium-navigate-v0',
            )
        ),
    )


def config():
    return validate_config(
        dict(
            latent_dim=4,
            width=8,
            durations=(5, 10),
            flow_steps=2,
            candidate_count=4,
            endpoint_candidates=3,
        )
    )


def dataset(seed=0):
    rng = np.random.RandomState(seed)
    return Dataset.create(
        observations=rng.normal(size=(48, 4)).astype(np.float32),
        actions=rng.uniform(-1, 1, (48, 2)).astype(np.float32),
        terminals=np.tile(np.r_[np.zeros(23), 1], 2).astype(np.float32),
    )


def agent_and_batch():
    data = dataset()
    sampler = SkillDataset(data, pb().config, config(), seed=0)
    mean, scale = fit_skill_normalization(data)
    agent = SkillPlanner.create(pb(), 4, 2, mean, scale, seed=0, config=config())
    return agent, sampler.sample(4)


def same_tree(a, b):
    for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b), strict=True):
        np.testing.assert_array_equal(x, y)


def test_config_rejects_missing_residual_lengths_and_unreviewed_keys():
    with pytest.raises(ValueError, match='every residual'):
        validate_config({'durations': (5, 10, 20)})
    with pytest.raises(ValueError, match='unknown'):
        validate_config({'selector_magic': True})
    with pytest.raises(ValueError, match='five-step'):
        validate_config({'execution_horizon': 10})


def test_episode_overlap_detects_shared_episode_with_different_dataset_hash():
    original, other = dataset(0), dataset(1)
    combined = Dataset.create(
        **{name: np.concatenate((original[name][:24], other[name][24:])) for name in original}
    )
    assert episode_fingerprints(original) & episode_fingerprints(combined)
    assert not episode_fingerprints(original) & episode_fingerprints(other)


def test_variable_windows_never_cross_boundaries_and_sampler_resume_is_exact():
    data = dataset()
    sampler = SkillDataset(data, pb().config, config(), seed=7)
    saved = sampler.get_rng_state()
    batch = sampler.sample(32)
    sampler.set_rng_state(saved)
    repeat = sampler.sample(32)
    same_tree(batch, repeat)
    assert set(batch['durations']) == {5, 10}
    for row, d in enumerate(batch['durations']):
        first = np.flatnonzero(np.all(data['observations'] == batch['states'][row, 0], axis=1))[0]
        assert first // 24 == (first + d) // 24
        np.testing.assert_array_equal(batch['actions'][row, :d], data['actions'][first : first + d])
        np.testing.assert_array_equal(
            batch['states'][row, : d + 1], data['observations'][first : first + d + 1]
        )
        assert np.all(batch['actions'][row, d:] == 0)


def test_first_hit_discount_and_terminal_continuation_hand_calculation():
    states = jnp.zeros((3, 11, 2)).at[:, :, 0].set(jnp.arange(11)[None])
    hits = first_hit(
        states,
        jnp.array([10, 5, 10]),
        jnp.array([[3.0, 0.0], [8.0, 0.0], [20.0, 0.0]]),
        jnp.ones(2),
        1e-5,
    )
    np.testing.assert_array_equal(hits, [3, -1, -1])
    target = return_target(hits, jnp.array([10, 5, 10]), jnp.array([0.7, 0.4, 0.0]), 0.9)
    np.testing.assert_allclose(target, [0.9**3, 0.9**5 * 0.4, 0], rtol=1e-6)
    assert (
        float(
            jax.grad(lambda v: return_target(jnp.array([-1]), jnp.array([5]), v, 0.9).sum())(
                jnp.array([0.4])
            )[0]
        )
        == 0
    )


def test_representation_shape_padding_and_state_independent_split():
    agent, batch = agent_and_batch()
    original = agent.apply(
        agent.state.params,
        'encode',
        agent.normalize(batch['states']),
        batch['actions'],
        batch['durations'],
    )
    poisoned = {k: np.array(v, copy=True) for k, v in batch.items()}
    for row, d in enumerate(batch['durations']):
        poisoned['states'][row, d:] = np.nan
        poisoned['actions'][row, d:] = np.nan
    # Encoder consumes state/action pairs 0..d-1, never endpoint/future states.
    clean = agent.apply(
        agent.state.params,
        'encode',
        agent.normalize(poisoned['states']),
        poisoned['actions'],
        batch['durations'],
    )
    same_tree(original, clean)
    z = original[0]
    parts = agent.apply(agent.state.params, 'split', z, batch['durations'], batch['cuts'])
    assert all(x.shape == (4, 4) for x in parts)
    predicted = agent.apply(
        agent.state.params, 'decode', agent.normalize(batch['states'][:, 0]), z, batch['durations']
    )
    assert predicted[0].shape == (4, 10, 2)
    np.testing.assert_array_equal(parts[0][batch['durations'] == 5], z[batch['durations'] == 5])


def test_real_jax_updates_have_disjoint_scopes_and_frozen_pb():
    agent, batch = agent_and_batch()
    represented, metrics = agent.update(batch, stage='representation')
    assert all(np.isfinite(float(v)) for v in metrics.values())
    assert int(represented.state.representation_step) == 1
    for key in ('q0', 'q1', 'proposal'):
        same_tree(agent.state.params[key], represented.state.params[key])
    valued, metrics = represented.update(batch, stage='value')
    assert all(np.isfinite(float(v)) for v in metrics.values())
    for key in ('encoder', 'splitter', 'decoder'):
        same_tree(represented.state.params[key], valued.state.params[key])
    same_tree(agent.pb, valued.pb)
    assert int(valued.state.value_step) == 1
    assert any(
        not np.array_equal(a, b)
        for a, b in zip(
            jax.tree_util.tree_leaves(represented.state.params['q0']),
            jax.tree_util.tree_leaves(valued.state.params['q0']),
        )
    )


def test_global_mode_does_not_call_endpoint_or_pb_runtime_and_q_selects_argmax(monkeypatch):
    agent, batch = agent_and_batch()
    observations, goals = jnp.array(batch['states'][:2, 0]), jnp.array(batch['goals'][:2])

    def forbidden(*args, **kwargs):
        raise AssertionError('global planning must not generate subgoals or call PB value')

    with monkeypatch.context() as m:
        m.setattr(FakePB, '_sample_endpoint_candidates', forbidden)
        m.setattr(FakeNetwork, 'select', forbidden)
        actions, info = agent.plan(
            observations, goals, jax.random.PRNGKey(99), num_candidates=4, temperature=0.5
        )
    assert actions.shape == (2, 5, 2)
    np.testing.assert_array_equal(info['selected_goal'], goals)
    np.testing.assert_array_equal(info['selected'], np.argmax(info['scores'], axis=1))
    np.testing.assert_array_equal(info['durations'], [5, 10, 5, 10])
    # Q is recomputed from the complete parent; no prefix-only re-ranking.
    starts = jnp.repeat(agent.normalize(observations), 4, axis=0)
    targets = jnp.repeat(agent.normalize(goals), 4, axis=0)
    codes = info['candidate_codes'].reshape(8, 4)
    ds = jnp.tile(info['durations'], 2)
    expected = (
        jax.nn.sigmoid(agent.apply(agent.state.params, 'q', starts, codes, ds, targets))
        .min(0)
        .reshape(2, 4)
    )
    np.testing.assert_allclose(info['scores'], expected, rtol=1e-6, atol=1e-7)


def test_subgoal_mode_uses_selected_endpoint_and_same_skill_parameters():
    agent, batch = agent_and_batch()
    observations, goals = jnp.array(batch['states'][:2, 0]), jnp.array(batch['goals'][:2])
    subgoal = agent.replace(mode='subgoal')
    actions, info = subgoal.plan(
        observations, goals, jax.random.PRNGKey(9), num_candidates=4, temperature=0.5
    )
    assert actions.shape == (2, 5, 2)
    assert not np.array_equal(info['selected_goal'], goals)
    same_tree(subgoal.state, agent.state)
    same_tree(subgoal.pb, agent.pb)


def test_offline_stage_pipeline_checkpoint_and_exact_resume(tmp_path):
    from skill_main import train

    teacher = pb()
    kwargs = dict(
        config=config(),
        seed=0,
        pb_sha256='fake-pb',
        representation_updates=2,
        value_updates=2,
        batch_size=4,
        checkpoint_every=1,
        smoke=True,
    )
    full, report = train(
        teacher,
        dataset(0),
        dataset(1),
        output=tmp_path / 'full',
        checkpoint_dir=tmp_path / 'checkpoints',
        **kwargs,
    )
    assert report['status'] == 'completed' and report['environment_steps'] == 0
    checkpoint = tmp_path / 'checkpoints/repr0000001_value0000000.pkl'
    resumed, _ = train(
        teacher,
        dataset(0),
        dataset(1),
        output=tmp_path / 'resumed',
        checkpoint_dir=tmp_path / 'resumed_checkpoints',
        resume=checkpoint,
        **kwargs,
    )
    same_tree(full.state, resumed.state)
    same_tree(full.rng, resumed.rng)
    final = tmp_path / 'checkpoints/repr0000002_value0000002.pkl'
    with pytest.raises(ValueError, match='non-smoke'):
        restore(final, teacher, 'fake-pb')
    with pytest.raises(ValueError, match='binding'):
        restore(final, teacher, 'other', require_trained=False)
    loaded, _ = restore(final, teacher, 'fake-pb', require_trained=False, mode='subgoal')
    same_tree(loaded.state, full.state)
    assert loaded.mode == 'subgoal'
    assert json.loads((tmp_path / 'full/manifest.json').read_text())['shared_training'] is True


def test_matched_two_mode_rollout_budget_clipping_and_frozen_weights(tmp_path):
    from types import SimpleNamespace

    from skill_main import evaluate

    agent, _ = agent_and_batch()
    before = flax.serialization.to_state_dict(agent.state)

    class Space:
        low = np.array([-0.1, -0.1], np.float32)
        high = np.array([0.1, 0.1], np.float32)

        def seed(self, seed):
            self.seed_value = seed

    class Env:
        def __init__(self):
            self.action_space = Space()
            self.unwrapped = self
            self.spec = SimpleNamespace(max_episode_steps=7)
            self.seeds, self.actions = [], []
            self.n = 0

        def reset(self, *, seed, options):
            self.seeds.append((seed, options['task_id']))
            self.n = 0
            return np.zeros(4, np.float32), {'goal': np.array([1, 0, 0, 0], np.float32)}

        def step(self, action):
            assert np.max(np.abs(action)) <= 0.1 + 1e-7
            self.actions.append(action.copy())
            self.n += 1
            obs = np.zeros(4, np.float32)
            obs[0] = self.n / 5
            return obs, 0, False, self.n >= 7, {'success': self.n == 5}

    env = Env()
    with pytest.raises(ValueError, match='cap'):
        evaluate(
            agent,
            env,
            tmp_path / 'reject',
            seed=3,
            episodes=1,
            task_ids=(1, 2),
            step_cap=27,
            include_pb=False,
        )
    assert not env.seeds
    result = evaluate(
        agent,
        env,
        tmp_path / 'evaluation',
        seed=3,
        episodes=1,
        task_ids=(1, 2),
        step_cap=28,
        include_pb=False,
    )
    assert result['status'] == 'completed' and result['policy_steps'] == 28
    assert env.seeds[:2] == env.seeds[2:]
    assert result['arms']['subgoal']['metrics']['overall_success'] == 1
    assert result['arms']['global']['metrics']['overall_success'] == 1
    for mode in ('subgoal', 'global'):
        assert len(result['arms'][mode]['plans']) == 4  # two five-step planning calls per episode
        assert 'skill_goal' in result['arms'][mode]['plans'][0]
    assert len(list((tmp_path / 'evaluation').glob('episode_*.npz'))) == 4
    same_tree(before, flax.serialization.to_state_dict(agent.state))


def test_invalid_padded_actions_do_not_change_representation_gradient():
    agent, batch = agent_and_batch()
    mutated = {k: np.array(v, copy=True) for k, v in batch.items()}
    # Guarantee a short row in this test independently of the random sampler.
    batch['durations'][0], batch['cuts'][0] = 5, 5
    mutated['durations'][0], mutated['cuts'][0] = 5, 5
    batch['actions'][0, 5:] = 0
    mutated['actions'][0, 5:] = np.nan
    key = jax.random.PRNGKey(12)
    clean = jax.value_and_grad(lambda p: agent.representation_loss(p, batch, key)[0])(
        agent.state.params
    )
    poisoned = jax.value_and_grad(lambda p: agent.representation_loss(p, mutated, key)[0])(
        agent.state.params
    )
    same_tree(clean, poisoned)
