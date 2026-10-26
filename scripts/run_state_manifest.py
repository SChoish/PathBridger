#!/usr/bin/env python3
"""Resolve, dry-run, or execute one declarative state experiment manifest."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from utils.config import normalize_packaged_file_flag  # noqa: E402
from utils.state_manifest import manifest_plan  # noqa: E402


def _runtime_path(value: str) -> str:
    if value.startswith('<') and value.endswith('>'):
        return value
    return str(Path(value).expanduser().resolve())


def _override_args(overrides: dict[str, Any]) -> list[str]:
    return [
        '--config_override='
        + str(key)
        + '='
        + json.dumps(value, separators=(',', ':'), ensure_ascii=True)
        for key, value in overrides.items()
    ]


def _training_command(
    job: dict[str, Any],
    *,
    dataset_dir: str,
    save_dir: str,
) -> list[str]:
    training = job['training']
    command = [
        sys.executable,
        str(_REPO / 'main.py'),
        f'--agent={job["preset"]}',
        f'--seed={job["seed"]}',
        f'--run_group={job["run_group"]}',
        f'--save_dir={_runtime_path(save_dir)}',
        f'--dataset_dir={_runtime_path(dataset_dir)}',
        f'--train_steps={training["train_steps"]}',
        f'--eval_interval={training["eval_interval"]}',
        f'--save_interval={training["save_interval"]}',
        '--extra_save_steps='
        + ','.join(str(value) for value in training['extra_save_steps']),
        '--async_prefetch=' + str(bool(training['async_prefetch'])),
        '--use_tqdm=False',
        '--use_wandb=False',
    ]
    command.extend(_override_args(job['config_overrides']))
    return command


def _token(value: Any) -> str:
    if value is None:
        return 'none'
    value = float(value)
    return str(int(value)) if value == int(value) else str(value).replace('.', 'p')


def _evaluation_output(job: dict[str, Any], output_root: Path) -> Path:
    return (
        output_root
        / job['plan_id']
        / job['experiment_id']
        / job['environment_id']
        / f'step_{job["checkpoint_step"]}'
        / (
            f'm{_token(job["prefix_samples"])}_'
            f'pt{_token(job["prefix_temperature"])}_'
            f'n{job["n"]}_t{_token(job["temperature"])}.json'
        )
    )


def _evaluation_command(
    job: dict[str, Any],
    *,
    checkpoint: str,
    dataset_dir: str,
    output_root: Path,
) -> list[str]:
    return [
        sys.executable,
        str(_REPO / 'evaluate.py'),
        f'--agent={job["preset"]}',
        f'--checkpoint_dir={_runtime_path(checkpoint)}',
        f'--checkpoint_step={job["checkpoint_step"]}',
        f'--dataset_dir={_runtime_path(dataset_dir)}',
        f'--episodes={job["episodes_per_task"]}',
        '--task_ids=' + ','.join(str(value) for value in job['task_ids']),
        '--seed=0',
        f'--eval_protocol={job["protocol"]}',
        f'--output_path={_evaluation_output(job, output_root).resolve()}',
        *_override_args(job['config_overrides']),
    ]


def _aggregation_output(job: dict[str, Any], output_root: Path) -> Path:
    return (
        output_root
        / job['plan_id']
        / job['experiment_id']
        / job['environment_id']
        / f'step_{job["checkpoint_step"]}'
        / (
            f'm{_token(job["prefix_samples"])}_'
            f'pt{_token(job["prefix_temperature"])}_summary.json'
        )
    )


def _aggregation_command(
    job: dict[str, Any],
    *,
    evaluation_by_id: dict[str, dict[str, Any]],
    output_root: Path,
) -> list[str]:
    inputs = [
        str(_evaluation_output(evaluation_by_id[job_id], output_root).resolve())
        for job_id in job['evaluation_job_ids_in_order']
    ]
    return [
        sys.executable,
        str(_REPO / 'scripts' / 'aggregate_state_evaluations.py'),
        *inputs,
        f'--output={_aggregation_output(job, output_root).resolve()}',
    ]


def _selected(
    jobs: list[dict[str, Any]],
    args,
    *,
    include_evaluation_plan: bool,
    include_cell_selectors: bool = False,
) -> list[dict[str, Any]]:
    selected = jobs
    selectors = [
        ('experiment_id', args.experiment),
        ('environment_id', args.environment),
    ]
    if include_evaluation_plan:
        selectors.append(('plan_id', args.evaluation_plan))
        selectors.extend(
            [
                ('checkpoint_step', args.checkpoint_step),
                ('prefix_samples', args.prefix_samples),
                ('prefix_temperature', args.prefix_temperature),
            ]
        )
    if include_cell_selectors:
        selectors.extend(
            [
                ('n', args.num_candidates),
                ('temperature', args.temperature),
            ]
        )
    for key, value in selectors:
        if value is not None:
            selected = [job for job in selected if job.get(key) == value]
    if not selected:
        raise ValueError('Selection matched no manifest jobs.')
    return selected


def main() -> None:
    normalize_packaged_file_flag('manifest', package_root=_REPO)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--manifest',
        type=Path,
        default=_REPO / 'manifests' / 'state' / 'paper.yaml',
    )
    parser.add_argument(
        '--phase',
        choices=('all', 'train', 'evaluate', 'aggregate'),
        default='all',
    )
    parser.add_argument('--experiment')
    parser.add_argument('--environment')
    parser.add_argument('--evaluation-plan')
    parser.add_argument('--checkpoint-step', type=int)
    parser.add_argument('--num-candidates', type=int)
    parser.add_argument('--temperature', type=float)
    parser.add_argument('--prefix-samples', type=int)
    parser.add_argument('--prefix-temperature', type=float)
    parser.add_argument('--dataset-dir', default='<DATASET_DIR>')
    parser.add_argument('--save-dir', default='<SAVE_DIR>')
    parser.add_argument('--checkpoint', default='<CHECKPOINT_FILE>')
    parser.add_argument(
        '--output-root',
        type=Path,
        default=Path('validation/manifest_evaluation'),
    )
    parser.add_argument('--output-plan', type=Path)
    parser.add_argument('--quiet', action='store_true')
    parser.add_argument(
        '--execute',
        action='store_true',
        help='Run the selected command(s); default is a complete JSON dry-run.',
    )
    args = parser.parse_args()
    evaluation_selectors = (
        args.evaluation_plan,
        args.checkpoint_step,
        args.num_candidates,
        args.temperature,
        args.prefix_samples,
        args.prefix_temperature,
    )
    if args.phase == 'train' and any(
        value is not None for value in evaluation_selectors
    ):
        parser.error('Evaluation selectors are unsupported for --phase=train.')
    if args.phase == 'aggregate' and (
        args.num_candidates is not None or args.temperature is not None
    ):
        parser.error(
            'N/T cell selectors are unsupported for --phase=aggregate; '
            'aggregation consumes the complete ordered grid.'
        )

    plan = manifest_plan(args.manifest, repo_root=_REPO)
    training_jobs = (
        _selected(plan['training_jobs'], args, include_evaluation_plan=False)
        if args.phase in {'all', 'train'}
        else []
    )
    evaluation_jobs = (
        _selected(
            plan['evaluation_jobs'],
            args,
            include_evaluation_plan=True,
            include_cell_selectors=True,
        )
        if args.phase in {'all', 'evaluate', 'aggregate'}
        else []
    )
    aggregation_jobs = (
        _selected(
            plan['aggregation_jobs'],
            args,
            include_evaluation_plan=True,
        )
        if args.phase in {'all', 'aggregate'}
        else []
    )
    rendered = {
        **plan,
        'training_jobs': [],
        'evaluation_jobs': [],
        'aggregation_jobs': [],
    }
    if args.phase in {'all', 'train'}:
        for job in training_jobs:
            rendered['training_jobs'].append(
                {
                    **job,
                    'command': _training_command(
                        job,
                        dataset_dir=args.dataset_dir,
                        save_dir=args.save_dir,
                    ),
                }
            )
    if args.phase in {'all', 'evaluate'}:
        for job in evaluation_jobs:
            rendered['evaluation_jobs'].append(
                {
                    **job,
                    'output_path': str(
                        _evaluation_output(job, args.output_root).resolve()
                    ),
                    'command': _evaluation_command(
                        job,
                        checkpoint=args.checkpoint,
                        dataset_dir=args.dataset_dir,
                        output_root=args.output_root,
                    ),
                }
            )
    if args.phase in {'all', 'aggregate'}:
        evaluation_by_id = {
            job['job_id']: job for job in evaluation_jobs
        }
        for job in aggregation_jobs:
            output_path = _aggregation_output(job, args.output_root)
            rendered['aggregation_jobs'].append(
                {
                    **job,
                    'input_paths_in_order': [
                        str(
                            _evaluation_output(
                                evaluation_by_id[job_id], args.output_root
                            ).resolve()
                        )
                        for job_id in job['evaluation_job_ids_in_order']
                    ],
                    'output_path': str(output_path.resolve()),
                    'command': _aggregation_command(
                        job,
                        evaluation_by_id=evaluation_by_id,
                        output_root=args.output_root,
                    ),
                }
            )

    if args.execute:
        placeholders = {
            '<DATASET_DIR>',
            '<SAVE_DIR>',
            '<LEGACY_PBF_ROOT>',
            '<CHECKPOINT_FILE>',
        }
        commands = [
            job['command']
            for key in ('training_jobs', 'evaluation_jobs', 'aggregation_jobs')
            for job in rendered[key]
        ]
        unresolved = sorted(
            placeholder
            for placeholder in placeholders
            if any(placeholder in argument for command in commands for argument in command)
        )
        if unresolved:
            parser.error(
                'Execution requires concrete runtime inputs; unresolved '
                + ', '.join(unresolved)
            )
        if args.phase not in {'train', 'evaluate', 'aggregate'} or len(commands) != 1:
            parser.error(
                '--execute requires a single concrete phase and '
                'selectors resolving to exactly one job.'
            )
        if args.phase in {'evaluate', 'aggregate'}:
            key = (
                'evaluation_jobs'
                if args.phase == 'evaluate'
                else 'aggregation_jobs'
            )
            output_path = Path(rendered[key][0]['output_path'])
            output_path.parent.mkdir(parents=True, exist_ok=True)
        for command in commands:
            print('+ ' + shlex.join(command), flush=True)
            subprocess.run(command, cwd=_REPO, check=True)
        return

    text = json.dumps(rendered, indent=2, sort_keys=True)
    if not args.quiet:
        print(text)
    if args.output_plan:
        args.output_plan.parent.mkdir(parents=True, exist_ok=True)
        args.output_plan.write_text(text + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
