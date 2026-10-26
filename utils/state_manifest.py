"""Strict state-experiment manifest loading and command planning."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from utils.config import config_sha256, resolve_config
from utils.evaluation import normalize_evaluation_protocol


SCHEMA_VERSION = 'pathbridger-state-experiment/v2'
FULL_NT_GRID = (
    (1, 0.0),
    *((n, t) for n in (1, 2, 4, 8, 16, 32) for t in (0.25, 0.5, 1.0)),
)

_ROOT_KEYS = {
    'schema_version',
    'manifest_id',
    'source',
    'data',
    'training_defaults',
    'aggregation_contract',
    'environments',
    'experiments',
    'evaluation_plans',
    'unresolved',
}
_ENVIRONMENT_KEYS = {
    'env_name',
    'pbf_preset',
    'pbg_preset',
    'pbf_source',
    'pbg_source',
    'pbf_run_group',
    'pbg_run_group',
    'pbf_best_nt',
    'pbg_best_nt',
}
_EXPERIMENT_KEYS = {
    'kind',
    'method',
    'source_commit',
    'preset_family',
    'environments',
    'run_group',
    'seed',
    'training',
    'config_overrides',
    'source_paths',
    'required_runtime_inputs',
}
_EVALUATION_KEYS = {
    'source_commit',
    'experiments',
    'checkpoints',
    'episodes_per_task',
    'task_ids',
    'nt_grid',
    'prefix_samples',
    'prefix_temperatures',
    'prefix_selection',
    'include_deterministic_prefix',
    'protocol',
    'aggregation_order',
    'source_paths',
    'status',
}
_TRAINING_KEYS = {
    'train_steps',
    'batch_size',
    'eval_interval',
    'save_interval',
    'extra_save_steps',
    'async_prefetch',
}
_AGGREGATION_CONTRACT_KEYS = {
    'episode_success',
    'task_success',
    'overall_success',
    'nt_grid_summary',
    'checkpoint_summary',
    'source_paths',
}


def _plain(value: Any) -> Any:
    if hasattr(value, 'to_dict'):
        value = value.to_dict()
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    if isinstance(value, list):
        return [_plain(item) for item in value]
    return value


def _require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f'{label} must be a mapping.')
    return {str(key): item for key, item in value.items()}


def _reject_unknown(mapping: Mapping[str, Any], allowed: set[str], label: str) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise KeyError(f'Unsupported {label} keys: {unknown}.')


def _load_preset(repo_root: Path, relative_path: str):
    path = repo_root / relative_path
    if not path.is_file():
        raise FileNotFoundError(f'Manifest preset not found: {path}')
    module_name = 'pathbridger_manifest_' + hashlib.sha256(
        str(path).encode('utf-8')
    ).hexdigest()
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot load manifest preset: {path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(module_name, None)
    if not hasattr(module, 'get_config'):
        raise AttributeError(f'Manifest preset has no get_config(): {path}')
    return module.get_config()


def _override_text(key: str, value: Any) -> str:
    rendered = json.dumps(value, separators=(',', ':'), ensure_ascii=True)
    return f'{key}={rendered}'


def load_state_manifest(path: str | Path) -> dict[str, Any]:
    """Load and strictly validate one state experiment manifest."""

    path = Path(path)
    with path.open(encoding='utf-8') as file:
        payload = yaml.safe_load(file)
    manifest = _require_mapping(payload, 'manifest root')
    _reject_unknown(manifest, _ROOT_KEYS, 'manifest')
    missing = sorted(_ROOT_KEYS - set(manifest))
    if missing:
        raise KeyError(f'Manifest is missing required keys: {missing}.')
    if manifest['schema_version'] != SCHEMA_VERSION:
        raise ValueError(
            f'Unsupported manifest schema {manifest["schema_version"]!r}; '
            f'expected {SCHEMA_VERSION!r}.'
        )
    source = _require_mapping(manifest['source'], 'source')
    if set(source) != {'commit', 'paths', 'notes'}:
        raise KeyError('source must contain exactly commit, paths, and notes.')
    if not isinstance(source['paths'], list) or not source['paths']:
        raise ValueError('source.paths must be a non-empty list.')
    training_defaults = _require_mapping(
        manifest['training_defaults'], 'training_defaults'
    )
    _reject_unknown(training_defaults, _TRAINING_KEYS, 'training_defaults')
    if int(training_defaults.get('batch_size', 0)) != 1024:
        raise ValueError(
            'training_defaults.batch_size must be 1024 because main.py does '
            'not expose a batch-size override.'
        )
    aggregation_contract = _require_mapping(
        manifest['aggregation_contract'], 'aggregation_contract'
    )
    if set(aggregation_contract) != _AGGREGATION_CONTRACT_KEYS:
        raise KeyError(
            'aggregation_contract must contain exactly '
            f'{sorted(_AGGREGATION_CONTRACT_KEYS)}.'
        )

    environments = _require_mapping(manifest['environments'], 'environments')
    if not environments:
        raise ValueError('Manifest must define at least one environment.')
    for name, raw in environments.items():
        environment = _require_mapping(raw, f'environments.{name}')
        _reject_unknown(environment, _ENVIRONMENT_KEYS, f'environments.{name}')
        missing_env = (_ENVIRONMENT_KEYS - {'pbg_best_nt'}) - set(environment)
        if missing_env:
            raise KeyError(f'environments.{name} is missing {sorted(missing_env)}.')
        families_with_nt = ('pbf', 'pbg') if 'pbg_best_nt' in environment else ('pbf',)
        for family in families_with_nt:
            snapshot = _require_mapping(
                environment[f'{family}_source'],
                f'environments.{name}.{family}_source',
            )
            if set(snapshot) != {'commit', 'path', 'sha256'}:
                raise KeyError(
                    f'environments.{name}.{family}_source must contain exactly '
                    'commit, path, and sha256.'
                )
            if len(str(snapshot['commit'])) != 40 or len(str(snapshot['sha256'])) != 64:
                raise ValueError(
                    f'environments.{name}.{family}_source has an invalid identity.'
                )
        for family in ('pbf', 'pbg'):
            nt = tuple(environment[f'{family}_best_nt'])
            if len(nt) != 2 or (int(nt[0]), float(nt[1])) not in FULL_NT_GRID:
                raise ValueError(
                    f'environments.{name}.{family}_best_nt is not in FULL_NT_GRID.'
                )

    experiments = _require_mapping(manifest['experiments'], 'experiments')
    for name, raw in experiments.items():
        experiment = _require_mapping(raw, f'experiments.{name}')
        _reject_unknown(experiment, _EXPERIMENT_KEYS, f'experiments.{name}')
        missing_experiment = {
            'kind',
            'method',
            'source_commit',
            'preset_family',
            'environments',
            'run_group',
            'seed',
            'training',
            'config_overrides',
            'source_paths',
            'required_runtime_inputs',
        } - set(experiment)
        if missing_experiment:
            raise KeyError(
                f'experiments.{name} is missing {sorted(missing_experiment)}.'
            )
        if experiment['preset_family'] not in {'pbf', 'pbg'}:
            raise ValueError(
                f'experiments.{name}.preset_family must be pbf or pbg.'
            )
        unknown_envs = sorted(set(experiment['environments']) - set(environments))
        if unknown_envs:
            raise KeyError(f'experiments.{name} has unknown environments {unknown_envs}.')
        training = _require_mapping(
            experiment['training'], f'experiments.{name}.training'
        )
        _reject_unknown(training, _TRAINING_KEYS, f'experiments.{name}.training')
        _require_mapping(
            experiment['config_overrides'],
            f'experiments.{name}.config_overrides',
        )

    evaluation_plans = _require_mapping(
        manifest['evaluation_plans'], 'evaluation_plans'
    )
    for name, raw in evaluation_plans.items():
        plan = _require_mapping(raw, f'evaluation_plans.{name}')
        _reject_unknown(plan, _EVALUATION_KEYS, f'evaluation_plans.{name}')
        missing_plan = _EVALUATION_KEYS - set(plan)
        if missing_plan:
            raise KeyError(f'evaluation_plans.{name} is missing {sorted(missing_plan)}.')
        unknown_experiments = sorted(set(plan['experiments']) - set(experiments))
        if unknown_experiments:
            raise KeyError(
                f'evaluation_plans.{name} has unknown experiments '
                f'{unknown_experiments}.'
            )
        normalize_evaluation_protocol(plan['protocol'])
        if plan['nt_grid'] != 'best' and plan['nt_grid'] != 'full':
            raise ValueError(
                f'evaluation_plans.{name}.nt_grid must be best or full.'
            )
        if int(plan['episodes_per_task']) < 1:
            raise ValueError(
                f'evaluation_plans.{name}.episodes_per_task must be positive.'
            )
        if list(plan['aggregation_order']) != [
            'checkpoint',
            'prefix_samples',
            'prefix_temperature',
            'n_t_cell',
            'task_id',
            'episode',
        ]:
            raise ValueError(
                f'evaluation_plans.{name}.aggregation_order is unsupported.'
            )
    if not isinstance(manifest['unresolved'], list):
        raise TypeError('unresolved must be a list.')
    return manifest


def resolve_experiment_jobs(
    manifest: Mapping[str, Any],
    *,
    repo_root: str | Path,
) -> list[dict[str, Any]]:
    """Resolve every manifest training job through the production config loader."""

    repo_root = Path(repo_root).resolve()
    jobs: list[dict[str, Any]] = []
    defaults = dict(manifest['training_defaults'])
    for experiment_id, experiment in manifest['experiments'].items():
        overrides = dict(experiment['config_overrides'])
        override_args = tuple(
            _override_text(str(key), value) for key, value in overrides.items()
        )
        training = {**defaults, **dict(experiment['training'])}
        for environment_id in experiment['environments']:
            environment = manifest['environments'][environment_id]
            preset = str(environment[f'{experiment["preset_family"]}_preset'])
            base = _load_preset(repo_root, preset)
            config = resolve_config(base, cli_overrides=override_args)
            if str(config.env_name) != str(environment['env_name']):
                raise ValueError(
                    f'{experiment_id}/{environment_id} resolved env_name '
                    f'{config.env_name!r}, expected {environment["env_name"]!r}.'
                )
            jobs.append(
                {
                    'job_id': f'{experiment_id}/{environment_id}',
                    'experiment_id': experiment_id,
                    'environment_id': environment_id,
                    'kind': experiment['kind'],
                    'method': experiment['method'],
                    'preset': preset,
                    'preset_family': str(experiment['preset_family']),
                    'preset_source': _plain(
                        environment[f'{experiment["preset_family"]}_source']
                    ),
                    'seed': int(experiment['seed']),
                    'run_group': str(
                        environment[f'{experiment["preset_family"]}_run_group']
                        if experiment['kind'] == 'main'
                        else experiment['run_group']
                    ),
                    'training': _plain(training),
                    'config_overrides': _plain(overrides),
                    'resolved_config': _plain(config),
                    'resolved_config_sha256': config_sha256(config),
                    'source_commit': str(experiment['source_commit']),
                    'source_paths': list(experiment['source_paths']),
                    'required_runtime_inputs': list(
                        experiment['required_runtime_inputs']
                    ),
                }
            )
    return jobs


def resolve_evaluation_jobs(
    manifest: Mapping[str, Any],
    training_jobs: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Expand checkpoint/prefix/N-T evaluation cells in canonical order."""

    by_experiment: dict[str, list[Mapping[str, Any]]] = {}
    for job in training_jobs:
        by_experiment.setdefault(str(job['experiment_id']), []).append(job)
    jobs: list[dict[str, Any]] = []
    for plan_id, plan in manifest['evaluation_plans'].items():
        prefix_samples = plan['prefix_samples']
        if prefix_samples is None:
            prefix_samples = [None]
        prefix_temperatures = plan['prefix_temperatures']
        if prefix_temperatures is None:
            prefix_temperatures = [None]
        for experiment_id in plan['experiments']:
            for training_job in by_experiment[experiment_id]:
                environment = manifest['environments'][
                    training_job['environment_id']
                ]
                best_nt_key = f'{training_job["preset_family"]}_best_nt'
                best_nt = environment.get(
                    best_nt_key,
                    (
                        training_job['resolved_config']['eval_num_candidates'],
                        training_job['resolved_config']['eval_temperature'],
                    ),
                )
                nt_grid = (
                    (tuple(best_nt),)
                    if plan['nt_grid'] == 'best'
                    else FULL_NT_GRID
                )
                for checkpoint in plan['checkpoints']:
                    for prefix_count in prefix_samples:
                        for prefix_temperature in prefix_temperatures:
                            for n, temperature in nt_grid:
                                overrides = dict(training_job['config_overrides'])
                                overrides.update(
                                    eval_num_candidates=int(n),
                                    eval_temperature=float(temperature),
                                )
                                if prefix_count is not None:
                                    overrides.update(
                                        eval_prefix_selection=plan[
                                            'prefix_selection'
                                        ],
                                        eval_num_prefix_samples=int(prefix_count),
                                        eval_include_deterministic_prefix=bool(
                                            plan['include_deterministic_prefix']
                                        ),
                                    )
                                if prefix_temperature is not None:
                                    overrides['eval_prefix_temperature'] = float(
                                        prefix_temperature
                                    )
                                resolved = resolve_config(
                                    training_job['resolved_config'],
                                    cli_overrides=tuple(
                                        _override_text(str(key), value)
                                        for key, value in overrides.items()
                                    ),
                                )
                                jobs.append(
                                    {
                                        'job_id': (
                                            f'{plan_id}/{training_job["job_id"]}/'
                                            f'step{int(checkpoint)}/m{prefix_count}/'
                                            f'pt{prefix_temperature}/n{int(n)}/'
                                            f't{float(temperature):g}'
                                        ),
                                        'plan_id': plan_id,
                                        'experiment_id': experiment_id,
                                        'environment_id': training_job[
                                            'environment_id'
                                        ],
                                        'preset': training_job['preset'],
                                        'preset_family': training_job[
                                            'preset_family'
                                        ],
                                        'checkpoint_step': int(checkpoint),
                                        'episodes_per_task': int(
                                            plan['episodes_per_task']
                                        ),
                                        'task_ids': [
                                            int(value) for value in plan['task_ids']
                                        ],
                                        'protocol': normalize_evaluation_protocol(
                                            plan['protocol']
                                        ),
                                        'n': int(n),
                                        'temperature': float(temperature),
                                        'prefix_samples': (
                                            None
                                            if prefix_count is None
                                            else int(prefix_count)
                                        ),
                                        'prefix_temperature': (
                                            None
                                            if prefix_temperature is None
                                            else float(prefix_temperature)
                                        ),
                                        'config_overrides': _plain(overrides),
                                        'resolved_config': _plain(resolved),
                                        'resolved_config_sha256': config_sha256(
                                            resolved
                                        ),
                                        'aggregation_order': list(
                                            plan['aggregation_order']
                                        ),
                                        'source_commit': str(plan['source_commit']),
                                        'source_paths': list(plan['source_paths']),
                                        'status': plan['status'],
                                    }
                                )
    return jobs


def resolve_aggregation_jobs(
    manifest: Mapping[str, Any],
    evaluation_jobs: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Group evaluation cells exactly as the deleted evaluator summarized them."""

    groups: OrderedDict[tuple[Any, ...], list[Mapping[str, Any]]] = OrderedDict()
    for job in evaluation_jobs:
        key = (
            job['plan_id'],
            job['experiment_id'],
            job['environment_id'],
            int(job['checkpoint_step']),
            job['prefix_samples'],
            job['prefix_temperature'],
        )
        groups.setdefault(key, []).append(job)

    contract = _plain(manifest['aggregation_contract'])
    jobs = []
    for key, cells in groups.items():
        (
            plan_id,
            experiment_id,
            environment_id,
            checkpoint,
            prefix_samples,
            prefix_temperature,
        ) = key
        jobs.append(
            {
                'job_id': (
                    f'{plan_id}/{experiment_id}/{environment_id}/'
                    f'step{checkpoint}/m{prefix_samples}/pt{prefix_temperature}'
                ),
                'plan_id': plan_id,
                'experiment_id': experiment_id,
                'environment_id': environment_id,
                'checkpoint_step': checkpoint,
                'prefix_samples': prefix_samples,
                'prefix_temperature': prefix_temperature,
                'evaluation_job_ids_in_order': [
                    cell['job_id'] for cell in cells
                ],
                'nt_cells_in_order': [
                    {'n': int(cell['n']), 'temperature': float(cell['temperature'])}
                    for cell in cells
                ],
                'aggregation_order': list(cells[0]['aggregation_order']),
                'aggregation_contract': contract,
                'protocol': cells[0]['protocol'],
                'source_commit': cells[0]['source_commit'],
                'source_paths': list(contract['source_paths']),
            }
        )
    return jobs


def manifest_plan(
    path: str | Path,
    *,
    repo_root: str | Path,
) -> dict[str, Any]:
    manifest = load_state_manifest(path)
    training_jobs = resolve_experiment_jobs(manifest, repo_root=repo_root)
    evaluation_jobs = resolve_evaluation_jobs(manifest, training_jobs)
    aggregation_jobs = resolve_aggregation_jobs(manifest, evaluation_jobs)
    return {
        'schema_version': manifest['schema_version'],
        'manifest_id': manifest['manifest_id'],
        'manifest_path': str(Path(path).resolve()),
        'source': _plain(manifest['source']),
        'data': _plain(manifest['data']),
        'aggregation_contract': _plain(manifest['aggregation_contract']),
        'training_jobs': training_jobs,
        'evaluation_jobs': evaluation_jobs,
        'aggregation_jobs': aggregation_jobs,
        'unresolved': _plain(manifest['unresolved']),
    }


__all__ = [
    'FULL_NT_GRID',
    'SCHEMA_VERSION',
    'load_state_manifest',
    'manifest_plan',
    'resolve_evaluation_jobs',
    'resolve_aggregation_jobs',
    'resolve_experiment_jobs',
]
