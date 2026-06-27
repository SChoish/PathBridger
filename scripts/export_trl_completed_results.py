#!/usr/bin/env python3
"""Export completed TRL runs → docs/trl_completed_results_choi.csv (one row per eval N)."""

from __future__ import annotations

import csv
import glob
import json
import os
import re
from collections import defaultdict

from docs_output_paths import docs_output_path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, '..')
PATHBRIDGER_ROOT = os.path.normpath(os.path.join(PROJECT_ROOT, '..', 'Pathbridger'))
OUT_CSV = docs_output_path(PROJECT_ROOT, 'trl_completed_results', 'csv')

RUN_TARGETS = [
    {'project': 'Pathbridger_flow', 'runs_dir': os.path.join(PROJECT_ROOT, 'runs')},
    {'project': 'Pathbridger', 'runs_dir': os.path.join(PATHBRIDGER_ROOT, 'runs')},
]

TRL_CRITIC_TYPES = {'trl', 'direct_chunk_trl'}

CSV_COLUMNS = [
    'project',
    'run_dir',
    'run_start_ts',
    'run_group',
    'env',
    'seed',
    'subgoal',
    'critic_type',
    'gap',
    'wmax',
    'horizon_K',
    'h_a',
    'train_N',
    'train_eval_N',
    'lambda',
    'kappa_b',
    'kappa_d',
    'gamma',
    'batch_size',
    'train_epochs',
    'final_eval_episodes',
    'config_key',
    'duplicate_index',
    'duplicate_count',
    'eval_epoch',
    'eval_n',
    'eval_episodes',
    'eval_score_type',
    'IDM',
    'ACTOR',
    'idm_tasks',
    'actor_tasks',
    'eval_timestamp',
    'eval_source',
]


def parse_run_start_ts(run_dir_name: str) -> str:
    m = re.match(r'^(\d{8})_(\d{6})_', run_dir_name)
    if not m:
        return ''
    d, t = m.group(1), m.group(2)
    return f'{d[:4]}-{d[4:6]}-{d[6:8]}T{t[:2]}:{t[2:4]}:{t[4:6]}'


def log_paths(run_dir: str) -> list[str]:
    paths: list[str] = []
    main = os.path.join(run_dir, 'run.log')
    if os.path.isfile(main):
        paths.append(main)
    paths.extend(sorted(glob.glob(os.path.join(run_dir, 'run_resume*.log'))))
    return paths


def is_completed(run_dir: str) -> bool:
    for path in log_paths(run_dir):
        with open(path, errors='replace') as f:
            if 'done run_dir=' in f.read():
                return True
    return False


def discover_run_dirs(runs_root: str) -> list[str]:
    found: set[str] = set()
    if not os.path.isdir(runs_root):
        return []
    for log_path in glob.glob(os.path.join(runs_root, '**', 'run.log'), recursive=True):
        found.add(os.path.dirname(log_path))
    for path in glob.glob(os.path.join(runs_root, '*')):
        if os.path.isdir(path) and os.path.isfile(os.path.join(path, 'run.log')):
            found.add(path)
    return sorted(found)


def load_run_params(run_dir: str, *, project: str, runs_root: str) -> dict | None:
    flags_path = os.path.join(run_dir, 'flags.json')
    if not os.path.isfile(flags_path):
        return None
    with open(flags_path, encoding='utf-8') as f:
        data = json.load(f)
    fg = data.get('flags', {})
    dyn = data.get('dynamics', {})
    critic = data.get('critic_agent', {})
    critic_type = str(critic.get('critic_type', '') or critic.get('algorithm', '')).lower()
    if critic_type not in TRL_CRITIC_TYPES:
        return None

    horizon = fg.get('horizon', dyn.get('dynamics_N', ''))
    gap = dyn.get('subgoal_value_gap_scale', '')
    wmax = dyn.get('subgoal_value_weight_max', '')
    train_n = dyn.get('subgoal_num_samples', '')
    lam = critic.get('value_distance_weight_power', '')
    kappa_b = critic.get('kappa_b', '')
    kappa_d = critic.get('kappa_d', '')
    run_group = str(fg.get('run_group', ''))

    config_key = '|'.join([
        f'proj={project}',
        f'group={run_group}',
        f'env={fg.get("env_name", "")}',
        f'gap={gap}',
        f'wmax={wmax}',
        f'h={horizon}',
        f'train_n={train_n}',
        f'lam={lam}',
        f'kb={kappa_b}',
        f'kd={kappa_d}',
        f'subgoal={dyn.get("subgoal_distribution", "")}',
        f'critic={critic_type}',
    ])

    rel_run_dir = os.path.relpath(run_dir, runs_root)
    ts_name = os.path.basename(run_dir)

    return {
        'project': project,
        'run_dir': rel_run_dir,
        'run_start_ts': parse_run_start_ts(ts_name),
        'run_group': run_group,
        'env': str(fg.get('env_name', '')),
        'seed': fg.get('seed', ''),
        'subgoal': str(dyn.get('subgoal_distribution', '')),
        'critic_type': critic_type,
        'gap': gap,
        'wmax': wmax,
        'horizon_K': horizon,
        'h_a': critic.get('action_chunk_horizon', ''),
        'train_N': train_n,
        'train_eval_N': dyn.get('subgoal_eval_num_samples', ''),
        'lambda': lam,
        'kappa_b': kappa_b,
        'kappa_d': kappa_d,
        'gamma': critic.get('discount', dyn.get('discount', '')),
        'batch_size': fg.get('batch_size', ''),
        'train_epochs': fg.get('train_epochs', ''),
        'final_eval_episodes': fg.get('final_eval_episodes_per_task', ''),
        'config_key': config_key,
    }


def parse_log_evals(run_dir: str) -> list[dict]:
    eval_blocks: list[dict] = []
    cur: dict | None = None
    stat_eps = ''
    for path in log_paths(run_dir):
        with open(path, errors='replace') as f:
            for line in f:
                if '=== EVAL START' in line:
                    m_ep = re.search(r'epoch=(\d+)', line)
                    m_stat = re.search(r'stat_episodes_per_task=(\d+)', line)
                    stat_eps = m_stat.group(1) if m_stat else ''
                    cur = {
                        'epoch': m_ep.group(1) if m_ep else '',
                        'eval_episodes': stat_eps,
                        'idm': '',
                        'actor': '',
                        'idm_tasks': [],
                        'actor_tasks': [],
                    }
                elif cur is not None:
                    if 'idm env_success_rate_mean=' in line:
                        cur['idm'] = line.split('=')[-1].strip()
                    elif 'actor env_success_rate_mean=' in line:
                        cur['actor'] = line.split('=')[-1].strip()
                    elif re.search(r'idm task_\d+ env=', line):
                        cur['idm_tasks'].append(line.split('=')[-1].strip())
                    elif re.search(r'actor task_\d+ env=', line):
                        cur['actor_tasks'].append(line.split('=')[-1].strip())
                    elif '=== EVAL END' in line:
                        eval_blocks.append(cur)
                        cur = None
    return eval_blocks


def eval_rows_from_json(run_dir: str, params: dict) -> list[dict]:
    rows: list[dict] = []
    pattern = os.path.join(run_dir, 'eval_results', 'epoch*_n*.json')
    for path in sorted(glob.glob(pattern)):
        with open(path, encoding='utf-8') as f:
            rec = json.load(f)
        base = os.path.basename(path)
        score_type = 'transitive_ratio'
        if '_score_goal_value' in base:
            score_type = 'goal_value'
        elif '_score_' in base:
            m = re.search(r'_score_([^.]+)', base)
            if m:
                score_type = m.group(1)
        idm_tasks = rec.get('idm_task_success_rates', {})
        actor_tasks = rec.get('actor_task_success_rates', {})
        rows.append({
            **params,
            'eval_epoch': rec.get('epoch', ''),
            'eval_n': rec.get('subgoal_eval_num_samples', ''),
            'eval_episodes': rec.get('eval_episodes_per_task', params.get('final_eval_episodes', '')),
            'eval_score_type': rec.get('subgoal_eval_score_type', score_type),
            'IDM': rec.get('idm_success_rate_mean', ''),
            'ACTOR': rec.get('actor_success_rate_mean', ''),
            'idm_tasks': ','.join(
                f'{k}:{float(v):.4f}' for k, v in sorted(idm_tasks.items(), key=lambda x: int(x[0]))
            ),
            'actor_tasks': ','.join(
                f'{k}:{float(v):.4f}' for k, v in sorted(actor_tasks.items(), key=lambda x: int(x[0]))
            ),
            'eval_timestamp': rec.get('timestamp', ''),
            'eval_source': 'eval_results_json',
        })
    return rows


def eval_rows_from_log(run_dir: str, params: dict) -> list[dict]:
    blocks = parse_log_evals(run_dir)
    if not blocks:
        return []
    train_epochs = str(params.get('train_epochs', ''))
    final_eps = str(params.get('final_eval_episodes', ''))
    # Prefer final-epoch eval with final episode count.
    candidates = [
        b for b in blocks
        if b['epoch'] == train_epochs and (not final_eps or b.get('eval_episodes') == final_eps)
    ]
    if not candidates:
        candidates = [b for b in blocks if b['epoch'] == train_epochs]
    if not candidates:
        candidates = [blocks[-1]]
    block = candidates[-1]
    eval_n = params.get('train_eval_N') or params.get('train_N') or ''
    idm_tasks = ','.join(f'{i+1}:{float(v):.4f}' for i, v in enumerate(block['idm_tasks']))
    actor_tasks = ','.join(f'{i+1}:{float(v):.4f}' for i, v in enumerate(block['actor_tasks']))
    return [{
        **params,
        'eval_epoch': block['epoch'],
        'eval_n': eval_n,
        'eval_episodes': block.get('eval_episodes', final_eps),
        'eval_score_type': 'transitive_ratio',
        'IDM': block['idm'],
        'ACTOR': block['actor'],
        'idm_tasks': idm_tasks,
        'actor_tasks': actor_tasks,
        'eval_timestamp': '',
        'eval_source': 'run_log_final',
    }]


def collect_rows(runs_dir: str, *, project: str) -> list[dict]:
    run_params: list[dict] = []
    for path in discover_run_dirs(runs_dir):
        if not is_completed(path):
            continue
        params = load_run_params(path, project=project, runs_root=runs_dir)
        if params is None:
            continue
        run_params.append((path, params))

    by_key: dict[str, list[dict]] = defaultdict(list)
    for _, p in run_params:
        by_key[p['config_key']].append(p)
    for key in by_key:
        by_key[key].sort(key=lambda x: x['run_start_ts'] or x['run_dir'])
        for i, p in enumerate(by_key[key], start=1):
            p['duplicate_index'] = i
            p['duplicate_count'] = len(by_key[key])

    rows: list[dict] = []
    for run_dir, p in run_params:
        eval_rows = eval_rows_from_json(run_dir, p)
        if not eval_rows:
            eval_rows = eval_rows_from_log(run_dir, p)
        rows.extend(eval_rows)
    rows.sort(key=lambda r: (
        r.get('project', ''),
        r['run_start_ts'],
        r['run_dir'],
        int(r.get('eval_epoch', 0) or 0),
        int(r.get('eval_n', 0) or 0),
    ))
    return rows


def write_csv(rows: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    all_rows: list[dict] = []
    per_project: list[str] = []
    for target in RUN_TARGETS:
        rows = collect_rows(target['runs_dir'], project=target['project'])
        all_rows.extend(rows)
        n_runs = len({r['run_dir'] for r in rows})
        per_project.append(f"{target['project']}={n_runs}")
    write_csv(all_rows, OUT_CSV)
    n_runs = len({(r['project'], r['run_dir']) for r in all_rows})
    n_keys = len({(r['project'], r['config_key']) for r in all_rows})
    print(
        f'Wrote {OUT_CSV}: {len(all_rows)} eval rows from {n_runs} completed TRL runs '
        f'({n_keys} unique config keys; {", ".join(per_project)})'
    )


if __name__ == '__main__':
    main()
