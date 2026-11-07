"""Hash-bound skill checkpoints and a frozen official PB inference bundle."""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import shutil
from pathlib import Path

import flax
import jax
import numpy as np

from skill_extensions.planner import FORMAT, SkillPlanner

ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = (
    'skill_extensions/models.py',
    'skill_extensions/planner.py',
    'skill_extensions/runtime.py',
    'utils/skill_data.py',
    'utils/datasets.py',
    'utils/networks.py',
    'utils/goal_representation.py',
    'agents/pathbridger.py',
    'agents/models.py',
    'agents/extensions.py',
    'agents/prefix_generators.py',
    'utils/deployment.py',
    'utils/flax_utils.py',
    'utils/evaluation.py',
    'skill_main.py',
)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1024**2), b''):
            h.update(part)
    return h.hexdigest()


def sources():
    return {p: digest(ROOT / p) for p in SOURCE_FILES}


def verify_sources(expected):
    if expected != sources():
        raise ValueError('skill/PB implementation changed since checkpoint creation')


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    with temp.open('w') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def write_checkpoint(path, payload, *, smoke=False):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    required = max(3 * len(content), 0 if smoke else 1024**3)
    free = shutil.disk_usage(path.parent).free
    if free < required:
        raise OSError(f'checkpoint space {free} < required {required}')
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    with temporary.open('wb') as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    return {'path': str(path.resolve()), 'sha256': digest(path), 'required_bytes': required}


def load_pb(path, expected_sha256):
    if digest(path) != expected_sha256:
        raise ValueError('PB inference bundle SHA256 mismatch')
    # Kept lazy: pure numerical extension tests do not import optional PB CLI deps.
    from agents import PathBridgerAgent
    from utils.deployment import inference_bundle_spec, restore_inference_bundle

    spec = inference_bundle_spec(path)
    pb = PathBridgerAgent.create(
        0,
        np.zeros((1, spec['observation_dim']), np.float32),
        np.zeros((1, spec['action_dim']), np.float32),
        spec['config'],
        state_scale=spec['state_scale'],
    )
    pb = restore_inference_bundle(pb, path)
    if int(pb.network.step) < 1:
        raise ValueError(
            'PB terminal teacher must be a trained inference bundle, not initialization'
        )
    return pb, spec


def restore(path, pb, pb_sha256, *, mode='global', require_trained=True):
    with Path(path).open('rb') as stream:
        payload = pickle.load(stream)
    if payload.get('format') != FORMAT or payload['pb_sha256'] != pb_sha256:
        raise ValueError('skill format/PB binding mismatch')
    verify_sources(payload['sources'])
    if require_trained and (
        payload['smoke'] or payload['representation_step'] < 1 or payload['value_step'] < 1
    ):
        raise ValueError('evaluation requires a non-smoke representation+value checkpoint')
    agent = SkillPlanner.create(
        pb,
        payload['observation_dim'],
        payload['action_dim'],
        payload['mean'],
        payload['scale'],
        seed=payload['seed'],
        config=payload['config'],
        mode=mode,
    )
    state = flax.serialization.from_state_dict(agent.state, payload['state'])
    agent = agent.replace(state=state, rng=np.asarray(payload['rng']))
    return agent, payload


def checkpoint_payload(agent, sampler, metadata):
    return {
        **metadata,
        'format': FORMAT,
        'config': dict(agent.config),
        'state': flax.serialization.to_state_dict(jax.device_get(agent.state)),
        'rng': np.asarray(agent.rng),
        'mean': np.asarray(agent.mean),
        'scale': np.asarray(agent.scale),
        'representation_step': int(agent.state.representation_step),
        'value_step': int(agent.state.value_step),
        'sampler_rng': sampler.get_rng_state(),
        'sources': sources(),
    }
