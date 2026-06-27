#!/usr/bin/env python3
"""Aggregate eval_results JSON → docs/flow_trl_feval_results_choi.csv + .md.

Sources:
  - runs/*/eval_results/epoch*_n*.json
  - checkpoints/flow_trl_best_epoch600/*/eval_results/epoch*_n*.json
"""

from __future__ import annotations

import csv
import glob
import json
import os
import re
from datetime import datetime

from docs_output_paths import docs_output_path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, '..')
RUNS_DIR = os.path.join(PROJECT_ROOT, 'runs')
BEST_CKPT_DIR = os.path.join(PROJECT_ROOT, 'checkpoints', 'flow_trl_best_epoch600')
OUT_CSV = docs_output_path(PROJECT_ROOT, 'flow_trl_feval_results', 'csv')
OUT_MD = docs_output_path(PROJECT_ROOT, 'flow_trl_feval_results', 'md')

EVAL_JSON_RE = re.compile(r'^epoch(\d+)_n(\d+)(?:_(.+))?\.json$')

CSV_COLUMNS = [
    'run_dir', 'run_group', 'env', 'gap', 'maxgap', 'train_n',
    'epoch', 'eval_n', 'temperature', 'eval_suffix',
    'IDM', 'ACTOR',
    'idm_tasks', 'actor_tasks', 'timestamp',
]

SOURCE_ROOTS = (RUNS_DIR, BEST_CKPT_DIR)


def _eval_suffix_from_filename(path: str) -> str:
    m = EVAL_JSON_RE.match(os.path.basename(path))
    return m.group(3) or '' if m else ''


def collect_records(source_roots: tuple[str, ...] = SOURCE_ROOTS) -> list[dict]:
    rows: list[dict] = []
    for root in source_roots:
        pattern = os.path.join(root, '*', 'eval_results', 'epoch*_n*.json')
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
                'temperature': rec.get('subgoal_temperature', 1.0),
                'eval_suffix': _eval_suffix_from_filename(path),
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
    csv_name = os.path.basename(OUT_CSV)
    lines = [
        '# Flow TRL final eval (N sweep) 결과',
        '',
        f'자동 생성: {datetime.now().strftime("%Y-%m-%d %H:%M")} · `scripts/summarize_feval_results.py`',
        f'소스: `runs/*/eval_results/` + `checkpoints/flow_trl_best_epoch600/*/eval_results/`',
        '',
        f'총 **{len(rows)}** eval records.',
        f'CSV: [`{csv_name}`]({csv_name})',
        '',
        '| run_group | env | gap | epoch | eval_n | temp | suffix | IDM | ACTOR |',
        '| --- | --- | --- | --- | --- | --- | --- | --- | --- |',
    ]
    for r in rows:
        lines.append(
            '| '
            + ' | '.join(
                str(r.get(k, ''))
                for k in (
                    'run_group', 'env', 'gap', 'epoch', 'eval_n',
                    'temperature', 'eval_suffix', 'IDM', 'ACTOR',
                )
            )
            + ' |'
        )
    lines.append('')
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')


def main() -> None:
    rows = collect_records()
    write_csv(rows, OUT_CSV)
    write_markdown(rows, OUT_MD)
    print(f'Wrote {OUT_CSV} + {OUT_MD} ({len(rows)} records)')


if __name__ == '__main__':
    main()
