"""Trusted, hash-pinned decoder imports and source-bound action-program checkpoints."""

from __future__ import annotations

import hashlib
import pickle
from pathlib import Path

import flax
import jax
import numpy as np

from skill_extensions.action_agent import ActionProgramAgent
from skill_extensions.runtime import atomic_json, digest, write_checkpoint

FORMAT = 'pathbridger-action-program-v1'
SOURCE_FILES = (
    'skill_extensions/action_models.py',
    'skill_extensions/action_agent.py',
    'skill_extensions/action_data.py',
    'skill_extensions/action_runtime.py',
    'skill_extensions/action_policy.py',
    'action_skill_main.py',
    'skill_extensions/action_diagnostics.py',
    'skill_extensions/models.py',
    'utils/networks.py',
    'utils/datasets.py',
    'utils/training_protocol.py',
    'utils/skill_data.py',
)
ROOT = Path(__file__).resolve().parents[1]


def source_identity():
    return {name: digest(ROOT / name) for name in SOURCE_FILES}


def tree_digest(tree):
    h = hashlib.sha256()
    for leaf in jax.tree_util.tree_leaves(tree):
        array = np.ascontiguousarray(leaf)
        h.update(str((array.shape, array.dtype.str)).encode())
        h.update(memoryview(array).cast('B'))
    return h.hexdigest()


def import_decoder(path, expected_sha256, *, allow_smoke=False):
    """Load only a user-owned trusted pickle after its exact SHA has been checked."""
    if digest(path) != expected_sha256:
        raise ValueError('decoder donor SHA256 mismatch')
    with Path(path).open('rb') as stream:
        donor = pickle.load(stream)
    if donor.get('format') != 'pathbridger-variable-skill-v1':
        raise ValueError('unsupported donor format; export a legacy skill checkpoint explicitly')
    if donor.get('representation_step', 0) < 1 or (donor.get('smoke') and not allow_smoke):
        raise ValueError('formal decoder donor must be a trained non-smoke representation')
    config = donor['config']
    # Parent encoder, splitter, Q, proposal, and optimizer state are deliberately not copied.
    asset = {
        'params': donor['state']['params']['decoder'],
        'mean': np.asarray(donor['mean'], np.float32),
        'scale': np.asarray(donor['scale'], np.float32),
        'latent_dim': config['latent_dim'],
        'width': config['width'],
        'max_duration': max(config['durations']),
        'observation_dim': donor['observation_dim'],
        'action_dim': donor['action_dim'],
        'donor_sha256': expected_sha256,
        'donor_smoke': bool(donor.get('smoke')),
        'donor_representation_step': donor['representation_step'],
    }
    asset['decoder_parameter_sha256'] = tree_digest(asset['params'])
    return asset


def snapshot(agent, sampler, asset, metadata):
    return {
        'format': FORMAT,
        'sources': source_identity(),
        'asset': jax.device_get(asset),
        'config': dict(agent.config),
        'stage': agent.stage,
        'state': flax.serialization.to_state_dict(jax.device_get(agent.state)),
        'sampler_rng': sampler.get_rng_state(),
        'metadata': dict(metadata),
    }


def save(path, agent, sampler, asset, metadata, *, smoke=False):
    return write_checkpoint(path, snapshot(agent, sampler, asset, metadata), smoke=smoke)


def restore(path, *, expected_sha256=None, require_deployable=False):
    if expected_sha256 is None:
        raise ValueError('checkpoint SHA256 must be provided before loading a trusted pickle')
    if digest(path) != expected_sha256:
        raise ValueError('action-program checkpoint SHA256 mismatch')
    with Path(path).open('rb') as stream:
        payload = pickle.load(stream)
    if payload.get('format') != FORMAT or payload['sources'] != source_identity():
        raise ValueError('checkpoint format/source identity mismatch')
    if tree_digest(payload['asset']['params']) != payload['asset']['decoder_parameter_sha256']:
        raise ValueError('frozen decoder parameter identity mismatch')
    agent = ActionProgramAgent.create(
        payload['asset'], seed=payload['metadata']['seed'], config=payload['config']
    )
    agent = agent.replace(
        state=flax.serialization.from_state_dict(agent.state, payload['state']),
        stage=payload['stage'],
    )
    if require_deployable and (
        agent.stage != 'value'
        or int(agent.state.value_step) < 1
        or payload['metadata'].get('smoke')
        or payload['asset']['donor_smoke']
    ):
        raise ValueError('deployment requires a trained non-smoke representation/proposal/Q')
    return agent, payload


__all__ = ['atomic_json', 'digest', 'import_decoder', 'restore', 'save', 'tree_digest']
