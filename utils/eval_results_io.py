"""Save checkpoint / in-training env-eval results under ``runs/<run_dir>/eval_results/``."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def eval_score_type_tag(score_type: str | None) -> str:
    """Filename suffix for non-default eval score types (empty keeps legacy paths)."""
    st = str(score_type or 'transitive_ratio').lower()
    if st in ('transitive_ratio', 'transitive', 'v_v_v', 'ratio', ''):
        return ''
    if st in ('goal_value', 'v_z_g', 'vzg'):
        return '_score_goal_value'
    return f'_score_{st}'


def eval_result_path(
    run_dir: Path | str,
    *,
    epoch: int,
    eval_n: int,
    score_type: str | None = None,
    suffix: str = '',
) -> Path:
    tag = eval_score_type_tag(score_type)
    extra = f'_{suffix.strip()}' if str(suffix).strip() else ''
    return Path(run_dir) / 'eval_results' / f'epoch{int(epoch)}_n{int(eval_n)}{tag}{extra}.json'


def save_eval_results(
    run_dir: Path | str,
    *,
    epoch: int,
    subgoal_eval_num_samples: int,
    task_ids: tuple[int, ...] | list[int],
    episodes_per_task: int,
    metrics: dict[str, Any],
    fg: dict[str, Any],
    root: dict[str, Any],
    result_suffix: str = '',
) -> Path:
    run_dir = Path(run_dir)
    eval_n = int(subgoal_eval_num_samples)
    idm_tasks = {
        str(tid): float(metrics[f'eval_idm/task_{tid}/success_rate'])
        for tid in task_ids
        if f'eval_idm/task_{tid}/success_rate' in metrics
    }
    actor_tasks = {
        str(tid): float(metrics[f'eval/task_{tid}/success_rate'])
        for tid in task_ids
        if f'eval/task_{tid}/success_rate' in metrics
    }
    dyn = root.get('dynamics', {})
    record: dict[str, Any] = {
        'timestamp': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        'run_dir': str(run_dir.resolve()),
        'run_group': str(fg.get('run_group', '')),
        'env_name': str(fg.get('env_name', '')),
        'epoch': int(epoch),
        'subgoal_eval_num_samples': eval_n,
        'subgoal_eval_score_type': str(dyn.get('subgoal_eval_score_type', 'transitive_ratio')),
        'subgoal_num_samples_train': int(dyn.get('subgoal_num_samples', 0)),
        'subgoal_value_gap_scale': float(dyn.get('subgoal_value_gap_scale', 0.0)),
        'subgoal_value_weight_max': float(dyn.get('subgoal_value_weight_max', 0.0)),
        'subgoal_flow_steps': int(dyn.get('subgoal_flow_steps', 0)),
        'subgoal_temperature': float(dyn.get('subgoal_temperature', 1.0)),
        'eval_episodes_per_task': int(episodes_per_task),
        'eval_task_ids': [int(t) for t in task_ids],
        'idm_success_rate_mean': float(metrics.get('eval_idm/success_rate_mean', float('nan'))),
        'actor_success_rate_mean': float(metrics.get('eval/success_rate_mean', float('nan'))),
        'idm_task_success_rates': idm_tasks,
        'actor_task_success_rates': actor_tasks,
    }
    out_dir = run_dir / 'eval_results'
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = eval_result_path(
        run_dir,
        epoch=epoch,
        eval_n=eval_n,
        score_type=str(dyn.get('subgoal_eval_score_type', 'transitive_ratio')),
        suffix=result_suffix,
    )
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(record, f, indent=2)
        f.write('\n')

    csv_path = out_dir / 'all.csv'
    row = {
        'timestamp': record['timestamp'],
        'epoch': record['epoch'],
        'eval_n': eval_n,
        'idm_mean': record['idm_success_rate_mean'],
        'actor_mean': record['actor_success_rate_mean'],
        'idm_tasks': ','.join(f'{k}:{v:.4f}' for k, v in sorted(idm_tasks.items())),
        'actor_tasks': ','.join(f'{k}:{v:.4f}' for k, v in sorted(actor_tasks.items())),
    }
    write_header = not csv_path.is_file()
    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    return json_path
