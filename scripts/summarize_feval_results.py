#!/usr/bin/env python3
"""Aggregate runs/*/eval_results/epoch*_n*.json → docs/flow_trl_feval_results.csv + .md."""

from __future__ import annotations

import csv
import glob
import json
import os
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, '..')
RUNS_DIR = os.path.join(PROJECT_ROOT, 'runs')
OUT_CSV = os.path.join(PROJECT_ROOT, 'docs', 'flow_trl_feval_results.csv')
OUT_MD = os.path.join(PROJECT_ROOT, 'docs', 'flow_trl_feval_results.md')

CSV_COLUMNS = [
    'run_dir', 'run_group', 'env', 'gap', 'maxgap', 'train_n',
    'epoch', 'eval_n', 'IDM', 'ACTOR',
    'idm_tasks', 'actor_tasks', 'timestamp',
]


def collect_records(runs_dir: str) -> list[dict]:
    rows: list[dict] = []
    pattern = os.path.join(runs_dir, '*', 'eval_results', 'epoch*_n*.json')
    for path in sorted(glob.glob(pattern)):
        with open(path, encoding='utf-8') as f:
            rec = json.load(f)
        idm_tasks = rec.get('idm_task_success_rates', {})
        actor_tasks = rec.get('actor_task_success_rates', {})
        rows.append({
            'run_dir': os.path.basename(str(rec.get('run_dir', ''))),
            'run_group': rec.get('run_group', ''),
            'env': rec.get('env_name', ''),
            'gap': rec.get('subgoal_value_gap_scale', ''),
            'maxgap': rec.get('subgoal_value_weight_max', ''),
            'train_n': rec.get('subgoal_num_samples_train', ''),
            'epoch': rec.get('epoch', ''),
            'eval_n': rec.get('subgoal_eval_num_samples', ''),
            'IDM': rec.get('idm_success_rate_mean', ''),
            'ACTOR': rec.get('actor_success_rate_mean', ''),
            'idm_tasks': ','.join(
                f'{k}:{v:.4f}' for k, v in sorted(idm_tasks.items(), key=lambda x: int(x[0]))
            ),
            'actor_tasks': ','.join(
                f'{k}:{v:.4f}' for k, v in sorted(actor_tasks.items(), key=lambda x: int(x[0]))
            ),
            'timestamp': rec.get('timestamp', ''),
        })
    return rows


def write_csv(rows: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_markdown(rows: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lines = [
        '# Flow TRL final eval (N sweep) 결과',
        '',
        f'자동 생성: {datetime.now().strftime("%Y-%m-%d %H:%M")} · `scripts/summarize_feval_results.py`',
        f'소스: `{RUNS_DIR}/*/eval_results/epoch*_n*.json`',
        '',
        f'총 **{len(rows)}** eval records.',
        f'CSV: [`flow_trl_feval_results.csv`](flow_trl_feval_results.csv)',
        '',
        '| run_group | env | gap | maxgap | epoch | eval_n | IDM | ACTOR |',
        '| --- | --- | --- | --- | --- | --- | --- | --- |',
    ]
    for r in rows:
        lines.append(
            '| '
            + ' | '.join(
                str(r.get(k, ''))
                for k in ('run_group', 'env', 'gap', 'maxgap', 'epoch', 'eval_n', 'IDM', 'ACTOR')
            )
            + ' |'
        )
    lines.append('')
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')


def main() -> None:
    rows = collect_records(RUNS_DIR)
    write_csv(rows, OUT_CSV)
    write_markdown(rows, OUT_MD)
    print(f'Wrote {OUT_CSV} + {OUT_MD} ({len(rows)} records)')


if __name__ == '__main__':
    main()
