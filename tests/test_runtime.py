"""Bounded CPU checks for training, optional extensions, and serialization."""

import tempfile
import unittest
from pathlib import Path

import flax
import jax
import jax.numpy as jnp
import numpy as np

from agents import get_config
from utils.datasets import Dataset, action_free_view
from utils.deployment import export_inference_bundle, restore_inference_bundle
from utils.flax_utils import restore_agent, save_agent
from utils.runtime import build_state_runtime


def assert_tree_equal(left, right):
    left = flax.serialization.to_state_dict(left)
    right = flax.serialization.to_state_dict(right)
    left_leaves, left_structure = jax.tree_util.tree_flatten(left)
    right_leaves, right_structure = jax.tree_util.tree_flatten(right)
    if left_structure != right_structure:
        raise AssertionError('serialized tree structure changed')
    for actual, expected in zip(left_leaves, right_leaves):
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(2417)
        self.data = Dataset.create(
            observations=rng.normal(size=(96, 4)).astype(np.float32),
            actions=rng.uniform(-1.0, 1.0, size=(96, 2)).astype(np.float32),
            terminals=np.tile(np.r_[np.zeros(47), 1], 2).astype(np.float32),
        )
        self.observations = jnp.asarray(self.data['observations'][:2])
        self.endpoints = jnp.asarray(self.data['observations'][10:12])
        self.key = jax.random.PRNGKey(51)

    def tearDown(self):
        jax.clear_caches()

    def runtime(self, **overrides):
        config = get_config()
        config.update(overrides)
        data = action_free_view(self.data) if config.offline_action_free else self.data
        return build_state_runtime(data, config, seed=17)

    def test_core_update_resume_and_inference_bundle(self):
        for distribution in ('flow', 'gaussian'):
            with self.subTest(endpoint=distribution), tempfile.TemporaryDirectory() as directory:
                agent, sampler, _ = self.runtime(endpoint_distribution=distribution)
                updated, metrics = agent.update(sampler.sample(2), full_metrics=False)
                self.assertTrue(np.isfinite(np.asarray(metrics['loss/total'])))
                self.assertEqual(int(updated.network.step), int(agent.network.step) + 1)
                self.assertFalse(np.array_equal(np.asarray(agent.rng), np.asarray(updated.rng)))
                for online, old_target, target in zip(
                    jax.tree_util.tree_leaves(updated.network.params['modules_value']),
                    jax.tree_util.tree_leaves(agent.network.params['modules_target_value']),
                    jax.tree_util.tree_leaves(updated.network.params['modules_target_value']),
                ):
                    np.testing.assert_allclose(
                        np.asarray(target),
                        0.005 * np.asarray(online) + 0.995 * np.asarray(old_target),
                        rtol=1e-6,
                        atol=1e-7,
                    )
                actions = updated.sample_action_chunks(
                    self.observations, self.endpoints, self.key, num_candidates=3, temperature=0.5
                )
                self.assertEqual(actions.shape, (2, 5, 2))
                self.assertTrue(np.all(np.isfinite(np.asarray(actions))))
                bridge = updated.construct_bridge(self.observations, self.endpoints)
                np.testing.assert_array_equal(
                    np.asarray(bridge[:, 0]), np.asarray(self.observations)
                )
                np.testing.assert_array_equal(np.asarray(bridge[:, -1]), np.asarray(self.endpoints))

                save_agent(updated, directory, 1, sampler=sampler)
                expected_batch = sampler.sample(2)
                restored = restore_agent(agent, directory, 1, sampler=sampler)
                assert_tree_equal(restored, updated)
                assert_tree_equal(sampler.sample(2), expected_batch)
                continued, _ = updated.update(expected_batch, full_metrics=False)
                resumed, _ = restored.update(expected_batch, full_metrics=False)
                assert_tree_equal(continued, resumed)

                bundle = Path(directory) / 'inference.pkl'
                export_inference_bundle(updated, bundle, action_dim=2)
                inference = restore_inference_bundle(agent, bundle)
                self.assertIsNone(inference.network.opt_state)
                np.testing.assert_array_equal(
                    np.asarray(
                        inference.sample_action_chunks(
                            self.observations,
                            self.endpoints,
                            self.key,
                            num_candidates=3,
                            temperature=0.5,
                        )
                    ),
                    np.asarray(actions),
                )
                different_protocol = agent.replace(
                    config=agent.config.copy({'training_objective_protocol': 'manuscript-v1'})
                )
                with self.assertRaises(ValueError):
                    restore_agent(different_protocol, directory, 1)

    def test_stochastic_prefixes_keep_endpoints_and_repeat_with_fixed_seed(self):
        for model in ('low_rank_gaussian', 'joint_flow'):
            with self.subTest(prefix=model):
                agent, sampler, _ = self.runtime(
                    prefix_model=model, horizon=5, prefix_rank=2, prefix_flow_steps=2
                )
                agent, metrics = agent.update(sampler.sample(2), full_metrics=False)
                self.assertTrue(np.isfinite(np.asarray(metrics['loss/total'])))
                prefixes = agent.sample_prefixes(
                    self.observations,
                    self.endpoints,
                    self.key,
                    num_samples=2,
                    include_deterministic=True,
                )
                repeated = agent.sample_prefixes(
                    self.observations,
                    self.endpoints,
                    self.key,
                    num_samples=2,
                    include_deterministic=True,
                )
                self.assertEqual(prefixes.shape, (2, 3, 6, 4))
                np.testing.assert_array_equal(np.asarray(prefixes), np.asarray(repeated))
                np.testing.assert_array_equal(
                    np.asarray(prefixes[:, :, -1]),
                    np.broadcast_to(np.asarray(self.endpoints[:, None, :]), (2, 3, 4)),
                )
                np.testing.assert_array_equal(
                    np.asarray(prefixes[:, 0]),
                    np.asarray(agent.construct_bridge(self.observations, self.endpoints)),
                )

    def test_action_free_agent_plans_states_and_rejects_action_decoding(self):
        agent, sampler, _ = self.runtime(
            offline_action_free=True, horizon=5, training_objective_protocol='manuscript-v1'
        )
        self.assertNotIn('modules_idm', agent.network.params)
        batch = sampler.sample(2)
        self.assertNotIn('actions', batch)
        agent, metrics = agent.update(batch, full_metrics=False)
        self.assertTrue(np.isfinite(np.asarray(metrics['loss/total'])))
        states = agent.sample_state_prefix(
            self.observations, self.endpoints, self.key, num_candidates=2, temperature=0.5
        )
        self.assertEqual(states.shape, (2, 6, 4))
        np.testing.assert_array_equal(np.asarray(states[:, 0]), np.asarray(self.observations))
        with self.assertRaises(ValueError):
            agent.sample_action_chunks(self.observations, self.endpoints, self.key)

    def test_temporal_weights_are_bounded_after_warmup(self):
        agent, sampler, _ = self.runtime(
            path_weight_beta=0.5, path_weight_warmup=0, path_weight_ramp=0
        )
        updated, metrics = agent.update(sampler.sample(2), full_metrics=True)
        self.assertTrue(np.isfinite(np.asarray(metrics['loss/total'])))
        self.assertGreaterEqual(float(metrics['bridge/path_weight_min']), 0.1 - 1e-6)
        self.assertLessEqual(float(metrics['bridge/path_weight_mean']), 1.0 + 1e-6)
        self.assertEqual(float(metrics['bridge/path_weight_strength']), 0.5)
        self.assertEqual(int(updated.network.step), int(agent.network.step) + 1)
