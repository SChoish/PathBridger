"""Self-contained, optimizer-free PathBridger inference bundle support."""

from __future__ import annotations

import os
import pickle
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import flax
import numpy as np

from utils.flax_utils import _checkpoint_metadata, _validate_checkpoint_metadata

_INFERENCE_FORMAT_V1 = 'pathbridger-inference-v1'
_INFERENCE_FORMAT = 'pathbridger-inference-v2'


def _read_bundle(
    bundle_path: str | os.PathLike[str],
) -> tuple[Path, Mapping[str, Any]]:
    path = Path(bundle_path)
    if not path.is_file():
        raise FileNotFoundError(f'Inference bundle not found: {path}')
    with path.open('rb') as file:
        payload = pickle.load(file)
    if not isinstance(payload, Mapping) or payload.get('format') not in {
        _INFERENCE_FORMAT_V1,
        _INFERENCE_FORMAT,
    }:
        raise ValueError(f'Invalid PathBridger inference bundle: {path}')
    return path, payload


def export_inference_bundle(
    agent: Any,
    export_path: str | os.PathLike[str],
    *,
    action_dim: int,
    provenance: Mapping[str, Any] | None = None,
) -> str:
    """Export params and runtime spec without optimizer or training RNG state."""

    path = Path(export_path)
    if path.suffix != '.pkl':
        raise ValueError('Inference bundle path must end in .pkl.')
    action_dim = int(action_dim)
    if action_dim < 1:
        raise ValueError(f'action_dim must be positive, got {action_dim}.')
    state_scale = np.asarray(agent.state_scale, dtype=np.float32)
    if state_scale.ndim != 1 or not np.all(np.isfinite(state_scale)):
        raise ValueError('agent.state_scale must be a finite 1D array.')
    config = (
        agent.config.to_dict()
        if hasattr(agent.config, 'to_dict')
        else dict(agent.config)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'format': _INFERENCE_FORMAT,
        'params': flax.serialization.to_state_dict(agent.network.params),
        'network_step': int(agent.network.step),
        'metadata': _checkpoint_metadata(agent),
        'config': config,
        'state_scale': state_scale,
        'observation_dim': int(state_scale.shape[0]),
        'action_dim': action_dim,
        'provenance': dict(provenance or {}),
    }
    temporary = path.with_suffix('.pkl.tmp')
    with temporary.open('wb') as file:
        pickle.dump(payload, file, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, path)
    return str(path)


def inference_bundle_spec(
    bundle_path: str | os.PathLike[str],
) -> dict[str, Any]:
    """Read and validate the self-contained runtime specification."""

    path, payload = _read_bundle(bundle_path)
    if payload.get('format') != _INFERENCE_FORMAT:
        raise ValueError(
            f'Legacy inference bundle {path} has no self-contained runtime spec; '
            're-export it from its training checkpoint.'
        )
    required = {'config', 'state_scale', 'observation_dim', 'action_dim'}
    missing = required - set(payload)
    if missing:
        raise ValueError(f'Inference bundle runtime spec is missing: {sorted(missing)}.')
    config = payload['config']
    if not isinstance(config, Mapping):
        raise TypeError('Inference bundle config must be a mapping.')
    state_scale = np.asarray(payload['state_scale'], dtype=np.float32)
    observation_dim = int(payload['observation_dim'])
    action_dim = int(payload['action_dim'])
    if (
        state_scale.shape != (observation_dim,)
        or observation_dim < 1
        or action_dim < 1
        or not np.all(np.isfinite(state_scale))
    ):
        raise ValueError(f'Invalid inference bundle runtime spec: {path}')
    metadata = payload.get('metadata')
    if not isinstance(metadata, Mapping):
        raise TypeError('Inference bundle metadata must be a mapping.')
    metadata_scale = np.asarray(metadata.get('state_scale'), dtype=np.float32)
    if not np.array_equal(state_scale, metadata_scale):
        raise ValueError('Inference bundle state_scale disagrees with metadata.')
    return {
        'config': dict(config),
        'state_scale': state_scale,
        'observation_dim': observation_dim,
        'action_dim': action_dim,
        'provenance': dict(payload.get('provenance') or {}),
    }


def restore_inference_bundle(
    agent: Any,
    bundle_path: str | os.PathLike[str],
) -> Any:
    """Load a fail-closed inference bundle into an initialized agent template."""

    path, payload = _read_bundle(bundle_path)
    if 'params' not in payload:
        raise ValueError(f'Inference bundle has no parameter tree: {path}')
    _validate_checkpoint_metadata(agent, payload.get('metadata'))
    params = flax.serialization.from_state_dict(
        agent.network.params,
        payload['params'],
    )
    network = agent.network.replace(
        step=int(payload.get('network_step', 0)),
        params=params,
        tx=None,
        opt_state=None,
    )
    return agent.replace(network=network)


__all__ = [
    'export_inference_bundle',
    'inference_bundle_spec',
    'restore_inference_bundle',
]
