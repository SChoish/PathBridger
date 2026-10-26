#!/usr/bin/env python3
"""Aggregate state evaluation JSON without crossing protocol boundaries."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


_SHA256_RE = re.compile(r'^[0-9a-f]{64}$')


def _required_digest(record: dict, key: str, *, source: Path) -> str:
    value = record.get(key)
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f'{source} lacks valid required digest {key!r}.')
    return value


def _checkpoint_key(record: dict, *, source: Path) -> tuple:
    identity = record.get('checkpoint_identity')
    if not isinstance(identity, dict) or identity.get('kind') not in {
        'unified',
        'legacy-split',
    }:
        raise ValueError(f'{source} lacks valid checkpoint_identity.')
    files = identity.get('files')
    if not isinstance(files, list) or not files:
        raise ValueError(f'{source} checkpoint_identity has no files.')
    resolved = []
    for item in files:
        if not isinstance(item, dict):
            raise ValueError(f'{source} has malformed checkpoint file identity.')
        role = item.get('role')
        size = item.get('size_bytes')
        digest = item.get('sha256')
        if (
            not isinstance(role, str)
            or not role
            or not isinstance(size, int)
            or size < 0
            or not isinstance(digest, str)
            or _SHA256_RE.fullmatch(digest) is None
        ):
            raise ValueError(f'{source} has malformed checkpoint file identity.')
        resolved.append((role, size, digest))
    return (identity['kind'], tuple(resolved))


def _dataset_key(record: dict, *, source: Path) -> tuple:
    identity = record.get('dataset_identity')
    if not isinstance(identity, dict):
        raise ValueError(f'{source} lacks required dataset_identity.')
    if identity.get('schema_version') != 'pathbridger-dataset-identity/v1':
        raise ValueError(f'{source} has unsupported dataset_identity schema.')
    digest = identity.get('sha256')
    size = identity.get('size')
    if (
        not isinstance(digest, str)
        or _SHA256_RE.fullmatch(digest) is None
        or not isinstance(size, int)
        or size < 1
    ):
        raise ValueError(f'{source} has malformed dataset_identity.')
    return (digest, size)


def aggregate(paths: list[Path]) -> dict:
    if not paths:
        raise ValueError('At least one evaluation JSON is required.')
    records = [json.loads(path.read_text(encoding='utf-8')) for path in paths]
    for path, record in zip(paths, records):
        if record.get('result_schema_version') != 'pathbridger-evaluation-result/v2':
            raise ValueError(
                f'{path} is not a pathbridger-evaluation-result/v2 record.'
            )
        _required_digest(record, 'config_sha256', source=path)
        _required_digest(record, 'aggregation_config_sha256', source=path)
        _checkpoint_key(record, source=path)
        _dataset_key(record, source=path)
    missing = [
        str(path)
        for path, record in zip(paths, records)
        if 'evaluation_protocol_version' not in record
        or 'effective_evaluation' not in record
    ]
    if missing:
        raise ValueError(
            'Versionless evaluation results cannot be auto-aggregated: '
            + ', '.join(missing)
        )
    protocols = {
        str(record['evaluation_protocol_version']) for record in records
    }
    if len(protocols) != 1:
        raise ValueError(
            'Refusing to mix evaluation protocols: ' + ', '.join(sorted(protocols))
        )
    task_contracts = {
        (
            tuple(record['effective_evaluation']['task_ids']),
            int(record['effective_evaluation']['episodes_per_task']),
            int(record['effective_evaluation']['max_episode_steps']),
            record['effective_evaluation']['task_episode_order'],
            record['effective_evaluation']['termination'],
            record['effective_evaluation']['success'],
            record['effective_evaluation']['overall_aggregation'],
        )
        for record in records
    }
    if len(task_contracts) != 1:
        raise ValueError('Refusing to mix incompatible effective evaluation settings.')
    summary_contracts = {
        (
            record.get('env_name'),
            record.get('endpoint_distribution'),
            record.get('checkpoint_step'),
            _checkpoint_key(record, source=path),
            _dataset_key(record, source=path),
            _required_digest(record, 'aggregation_config_sha256', source=path),
            record.get('prefix_model'),
            record.get('eval_prefix_selection'),
            record.get('eval_num_prefix_samples'),
            record.get('eval_prefix_temperature'),
        )
        for path, record in zip(paths, records)
    }
    if len(summary_contracts) != 1:
        raise ValueError(
            'Refusing to mix environments, methods, checkpoints, datasets, '
            'model configs, or prefix settings in one N/T summary.'
        )
    overalls = [float(record['evaluation/overall_success']) for record in records]
    return {
        'evaluation_protocol_version': protocols.pop(),
        'checkpoint_identity': records[0]['checkpoint_identity'],
        'dataset_identity': records[0]['dataset_identity'],
        'aggregation_config_sha256': records[0]['aggregation_config_sha256'],
        'inputs_in_order': [str(path.resolve()) for path in paths],
        'records': len(records),
        'cells_in_order': [
            {
                'input': str(path.resolve()),
                'num_candidates': record.get('eval_num_candidates'),
                'temperature': record.get('eval_temperature'),
                'overall_success': float(
                    record['evaluation/overall_success']
                ),
            }
            for path, record in zip(paths, records)
        ],
        'aggregation_rule': (
            'mean-and-max of overall_success in caller-provided N/T order'
        ),
        'overall_success_mean': sum(overalls) / len(overalls),
        'overall_success_max': max(overalls),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('results', nargs='+', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = aggregate(args.results)
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
