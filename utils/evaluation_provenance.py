"""Evaluation result provenance loading and contract binding."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PROVENANCE_SCHEMA_VERSION = 'pathbridger-evaluation-provenance/v1'


def load_execution_provenance(path: str) -> dict[str, Any] | None:
    if not path:
        return None
    provenance_path = Path(path)
    try:
        payload = json.loads(provenance_path.read_text(encoding='utf-8'))
    except (json.JSONDecodeError, OSError) as error:
        raise ValueError(
            f'Invalid execution provenance input: {provenance_path}'
        ) from error
    if not isinstance(payload, dict):
        raise TypeError('Execution provenance input must be one JSON object.')
    if payload.get('schema_version') != PROVENANCE_SCHEMA_VERSION:
        raise ValueError('Unsupported execution provenance schema.')
    if not isinstance(payload.get('evaluation_contract'), dict):
        raise TypeError('Execution provenance lacks evaluation_contract.')
    if 'effective_evaluation_sha256' in payload:
        raise ValueError(
            'Execution provenance input must not predeclare the runtime '
            'effective_evaluation_sha256.'
        )
    return payload


def bind_execution_provenance(
    result: dict[str, Any], provenance: dict[str, Any] | None
) -> dict[str, Any] | None:
    if provenance is None:
        return None
    settings = result['effective_evaluation']
    expected_contract = {
        'checkpoint_step': result['checkpoint_step'],
        'env_name': result['env_name'],
        'endpoint_distribution': result['endpoint_distribution'],
        'config_sha256': result['config_sha256'],
        'evaluation_protocol_version': result['evaluation_protocol_version'],
        'task_ids': settings['task_ids'],
        'episodes_per_task': settings['episodes_per_task'],
        'evaluation_seed': settings['policy_jax_seed'],
        'eval_num_candidates': result['eval_num_candidates'],
        'eval_temperature': result['eval_temperature'],
    }
    if provenance['evaluation_contract'] != expected_contract:
        raise ValueError(
            'Execution provenance evaluation_contract does not match the '
            f'effective evaluation: expected {expected_contract!r}.'
        )
    return {
        **provenance,
        'effective_evaluation_sha256': settings['settings_sha256'],
    }


__all__ = [
    'PROVENANCE_SCHEMA_VERSION',
    'bind_execution_provenance',
    'load_execution_provenance',
]
