"""Batch deployment inference using the same kernel as evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import jax
import numpy as np
from absl import app, flags
from ml_collections import config_flags

from agents import PathBridgerAgent
from envs.env_utils import make_env_and_datasets
from utils.config import (
    config_sha256,
    normalize_packaged_file_flag,
    resolve_config,
)
from utils.deployment import (
    export_inference_bundle,
    inference_bundle_spec,
    restore_inference_bundle,
)
from utils.flax_utils import initialize_from_legacy_pathbridger, restore_agent
from utils.runtime import build_state_runtime

FLAGS = flags.FLAGS
_DEFAULT_CONFIG = str(
    Path(__file__).resolve().parent / 'configs' / 'pbf' / 'antmaze_medium.py'
)

flags.DEFINE_string('checkpoint_dir', '', 'Training checkpoint directory or exact file.')
flags.DEFINE_integer('checkpoint_step', 0, 'Training checkpoint step.')
flags.DEFINE_string('inference_bundle', '', 'Self-contained inference bundle.')
flags.DEFINE_string(
    'legacy_dynamics_checkpoint',
    '',
    'Exact split legacy dynamics checkpoint; requires legacy_critic_checkpoint.',
)
flags.DEFINE_string(
    'legacy_critic_checkpoint',
    '',
    'Exact split legacy critic checkpoint; requires legacy_dynamics_checkpoint.',
)
flags.DEFINE_string('export_path', '', 'Optional path for an exported inference bundle.')
flags.DEFINE_string('input_path', '', 'NPZ containing observations and goals arrays.')
flags.DEFINE_string('output_path', '', 'Output NPZ path for action_chunks.')
flags.DEFINE_string('dataset_dir', '', 'OGBench data; required only for a training checkpoint.')
flags.DEFINE_integer('seed', 0, 'Inference PRNG seed.')
flags.DEFINE_integer('num_candidates', 0, 'Override Best-of-N; zero uses resolved config.')
flags.DEFINE_float('temperature', -1.0, 'Override temperature; negative uses resolved config.')
flags.DEFINE_string(
    'config_yaml', '', 'Optional strict YAML overlay for a training checkpoint.'
)
flags.DEFINE_multi_string(
    'config_override', [], 'Highest-priority training-checkpoint config key=value override; repeatable.'
)

config_flags.DEFINE_config_file(
    'agent',
    _DEFAULT_CONFIG,
    'Training-checkpoint environment/method preset; ignored for a bundle.',
    lock_config=False,
)


def _from_bundle():
    spec = inference_bundle_spec(FLAGS.inference_bundle)
    config = resolve_config(spec['config'])
    observations = np.zeros((1, spec['observation_dim']), dtype=np.float32)
    actions = np.zeros((1, spec['action_dim']), dtype=np.float32)
    agent = PathBridgerAgent.create(
        FLAGS.seed,
        observations,
        actions,
        config,
        state_scale=spec['state_scale'],
    )
    migration = (spec.get('provenance') or {}).get('legacy_migration')
    return (
        restore_inference_bundle(agent, FLAGS.inference_bundle),
        config,
        spec,
        migration,
    )


def _from_training_checkpoint():
    config = resolve_config(
        FLAGS.agent,
        yaml_path=FLAGS.config_yaml,
        cli_overrides=FLAGS.config_override,
    )
    _, train_data, _ = make_env_and_datasets(
        str(config.env_name), dataset_dir=FLAGS.dataset_dir or None
    )
    agent, _, _ = build_state_runtime(train_data, config, seed=FLAGS.seed)
    migration = None
    checkpoint_restore = None
    if FLAGS.legacy_dynamics_checkpoint:
        agent, migration = initialize_from_legacy_pathbridger(
            agent,
            FLAGS.legacy_dynamics_checkpoint,
            FLAGS.legacy_critic_checkpoint,
            return_info=True,
        )
    else:
        agent, checkpoint_restore = restore_agent(
            agent,
            FLAGS.checkpoint_dir,
            FLAGS.checkpoint_step,
            return_info=True,
            inference_only=True,
        )
    action_dim = int(np.asarray(train_data['actions']).shape[-1])
    if FLAGS.export_path:
        provenance = {}
        if migration is not None:
            provenance['legacy_migration'] = migration
        if checkpoint_restore is not None:
            provenance['checkpoint_restore'] = checkpoint_restore
        export_inference_bundle(
            agent,
            FLAGS.export_path,
            action_dim=action_dim,
            provenance=provenance or None,
        )
    spec = {
        'observation_dim': int(np.asarray(train_data['observations']).shape[-1]),
        'action_dim': action_dim,
        'checkpoint_restore': checkpoint_restore,
    }
    return agent, config, spec, migration


def main(_):
    legacy_dynamics = bool(FLAGS.legacy_dynamics_checkpoint)
    legacy_critic = bool(FLAGS.legacy_critic_checkpoint)
    if legacy_dynamics != legacy_critic:
        raise ValueError(
            'legacy_dynamics_checkpoint and legacy_critic_checkpoint must be '
            'provided together.'
        )
    legacy_pair = legacy_dynamics and legacy_critic
    source_count = sum(
        (bool(FLAGS.checkpoint_dir), bool(FLAGS.inference_bundle), legacy_pair)
    )
    if source_count != 1:
        raise ValueError(
            'Provide exactly one source: checkpoint_dir, inference_bundle, or '
            'the complete legacy dynamics/critic checkpoint pair.'
        )
    if FLAGS.inference_bundle and FLAGS.export_path:
        raise ValueError('export_path is only valid with a training checkpoint.')
    if FLAGS.inference_bundle and (FLAGS.config_yaml or FLAGS.config_override):
        raise ValueError(
            'A bundle config is immutable; use num_candidates or temperature '
            'for deployment-time sampling overrides.'
        )
    if FLAGS.checkpoint_step < 0:
        raise ValueError('checkpoint_step cannot be negative.')
    if FLAGS.num_candidates < 0:
        raise ValueError('num_candidates cannot be negative.')

    migration = None
    if FLAGS.inference_bundle:
        agent, config, spec, migration = _from_bundle()
    else:
        agent, config, spec, migration = _from_training_checkpoint()

    if not FLAGS.input_path:
        if FLAGS.output_path:
            raise ValueError('output_path requires input_path.')
        payload = {
            'loaded': True,
            'env_name': str(config.env_name),
            'endpoint_distribution': str(config.endpoint_distribution),
            'config_sha256': config_sha256(config),
            'legacy_migration': migration,
            'checkpoint_restore': spec.get('checkpoint_restore'),
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    if not FLAGS.output_path:
        raise ValueError('output_path is required when input_path is provided.')

    with np.load(FLAGS.input_path, allow_pickle=False) as inputs:
        missing = {'observations', 'goals'} - set(inputs.files)
        if missing:
            raise ValueError(f'Inference input is missing arrays: {sorted(missing)}.')
        observations = np.asarray(inputs['observations'], dtype=np.float32)
        goals = np.asarray(inputs['goals'], dtype=np.float32)
    if observations.ndim != 2 or goals.ndim != 2:
        raise ValueError('observations and goals must both have shape [B, D].')
    if observations.shape != goals.shape:
        raise ValueError(
            f'observations/goals shape mismatch: {observations.shape} vs {goals.shape}.'
        )
    if observations.shape[0] < 1:
        raise ValueError('Inference batch B must be at least 1.')
    if observations.shape[-1] != spec['observation_dim']:
        raise ValueError(
            f'Input D={observations.shape[-1]} does not match bundle/model '
            f'D={spec["observation_dim"]}.'
        )
    if not np.all(np.isfinite(observations)) or not np.all(np.isfinite(goals)):
        raise ValueError('observations and goals must contain only finite values.')

    num_candidates = FLAGS.num_candidates or int(config.eval_num_candidates)
    temperature = (
        float(config.eval_temperature)
        if FLAGS.temperature < 0.0
        else float(FLAGS.temperature)
    )
    action_chunks = agent.sample_action_chunks(
        observations=observations,
        goals=goals,
        seed=jax.random.PRNGKey(FLAGS.seed),
        num_candidates=num_candidates,
        temperature=temperature,
    )
    action_chunks = np.asarray(jax.device_get(action_chunks), dtype=np.float32)
    output = Path(FLAGS.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, action_chunks=action_chunks)
    print(f'wrote {output} shape={action_chunks.shape}')


def run():
    normalize_packaged_file_flag(
        'agent', package_root=Path(__file__).resolve().parent
    )
    app.run(main)


if __name__ == '__main__':
    run()
