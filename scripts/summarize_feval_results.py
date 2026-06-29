#!/usr/bin/env python3
"""Aggregate available feval eval JSONs → docs/flow_trl_feval_results_7ch.csv (+ .md).

All available eval_results/epoch600_n*.json files are included, so follow-up
evals such as N=32 and duplicate reruns are reflected.
Duplicate restarts for the same run_group are kept as separate rows (run_start_ts disambiguates).
"""

from __future__ import annotations

import csv
import glob
import json
import os
import re
import sys
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
PROJECT_ROOT = os.path.join(SCRIPT_DIR, '..')
RUNS_DIR = os.path.join(PROJECT_ROOT, 'runs')

from docs_output_paths import docs_7ch_name, docs_7ch_path

OUT_CSV = str(docs_7ch_path(PROJECT_ROOT, 'flow_trl_feval_results.csv'))
OUT_MD = str(docs_7ch_path(PROJECT_ROOT, 'flow_trl_feval_results.md'))

CSV_COLUMNS = [
    'run_uid',
    'run_dir',
    'run_start_ts',
    'eval_timestamp',
    'run_group',
    'config',
    'env',
    'seed',
    'horizon',
    'sweep_tag',
    'gap',
    'maxgap',
    'train_n',
    'gamma',
    'vdist_pow',
    'subgoal_eval_selection',
    'temperature',
    'eval_episodes_per_task',
    'final_eval_episodes_per_task',
    'epoch',
    'eval_n',
    'IDM',
    'ACTOR',
    'FLOW_IDM',
    'FLOW_ACTOR',
    'SPI_SUBGOAL_IDM',
    'SPI_SUBGOAL_ACTOR',
    'idm_tasks',
    'actor_tasks',
    'flow_idm_tasks',
    'flow_actor_tasks',
    'spi_subgoal_idm_tasks',
    'spi_subgoal_actor_tasks',
    'source',
]

_RUN_DIR_TS_RE = re.compile(r'^(\d{8})_(\d{6})_')


def _parse_run_start_ts(run_dir_name: str) -> str:
    m = _RUN_DIR_TS_RE.match(run_dir_name)
    if not m:
        return ''
    return datetime.strptime(m.group(1) + m.group(2), '%Y%m%d%H%M%S').strftime('%Y-%m-%dT%H:%M:%S')


def _load_run_params(run_dir: str) -> dict:
    flags_path = os.path.join(run_dir, 'flags.json')
    if not os.path.isfile(flags_path):
        return {}
    with open(flags_path, encoding='utf-8') as f:
        root = json.load(f)
    fg = root.get('flags', {})
    dyn = root.get('dynamics', {})
    critic = root.get('critic_agent', {})
    run_config = str(fg.get('run_config', ''))
    config = os.path.basename(run_config).replace('.yaml', '') if run_config else ''
    return {
        'seed': fg.get('seed', ''),
        'gamma': critic.get('discount', critic.get('gamma', '')),
        'vdist_pow': critic.get('value_distance_weight_power', critic.get('vdist_pow', '')),
        'subgoal_eval_selection': dyn.get('subgoal_eval_selection', ''),
        'eval_episodes_per_task': fg.get('eval_episodes_per_task', ''),
        'final_eval_episodes_per_task': fg.get('final_eval_episodes_per_task', ''),
        'config': config,
        'horizon': fg.get('horizon', ''),
    }


def _eval_n_from_path(path: str) -> int:
    m = re.search(r'_n(\d+)\.json$', os.path.basename(path))
    return int(m.group(1)) if m else -1


def _parse_eval_json_name(path: str) -> tuple[int, int, float]:
    """Return (epoch, eval_n, temperature)."""
    bn = os.path.basename(path)
    m = re.match(r'epoch(\d+)(?:_t([\dp]+))?(?:_m(\d+))?_n(\d+)\.json$', bn)
    if not m:
        return 600, _eval_n_from_path(path), 1.0
    epoch = int(m.group(1))
    temp = 1.0
    if m.group(2):
        temp = float(m.group(2).replace('p', '.'))
    eval_n = int(m.group(4))
    return epoch, eval_n, temp


def _row_key(row: dict) -> tuple:
    return (
        row['run_dir'],
        int(row.get('epoch', 600)),
        int(row['eval_n']),
        float(row.get('temperature', 1.0) or 1.0),
    )


def _sweep_tag(*, run_group: str, config: str, horizon: str | int | float) -> str:
    rg = str(run_group)
    cfg = str(config)
    h = str(horizon)
    if 'p456g999' in rg or 'p456' in rg or 'puzzle_45_46' in rg:
        return 'p456_feval'
    if 'k40' in cfg or 'k40_best' in rg:
        return 'k40_best'
    if 'gamma_rerun' in rg or rg.endswith('_gamma_rerun'):
        return 'gamma_rerun'
    if h == '40':
        return 'k40'
    if 'feval' in rg:
        return 'k25_feval'
    return 'other'


def _append_row(
    rows: list[dict],
    *,
    run_dir: str,
    run_dir_name: str,
    params: dict,
    rec: dict,
    json_path: str,
    source: str,
    final_epoch: int,
) -> None:
    run_start_ts = _parse_run_start_ts(run_dir_name)
    run_group = str(rec.get('run_group', params.get('run_group', '')))
    run_uid = f'{run_group}|{run_start_ts}' if run_start_ts else run_group
    idm_tasks = rec.get('idm_task_success_rates', {}) or {}
    actor_tasks = rec.get('actor_task_success_rates', {}) or {}
    four_way_means = rec.get('four_way_success_rate_means', {}) or {}
    four_way_tasks = rec.get('four_way_task_success_rates', {}) or {}
    _, eval_n, temp = _parse_eval_json_name(json_path) if json_path else (
        int(rec.get('epoch', final_epoch)),
        int(rec.get('subgoal_eval_num_samples', rec.get('eval_n', 0))),
        float(rec.get('subgoal_temperature', 1.0)),
    )
    if 'subgoal_eval_num_samples' in rec:
        eval_n = int(rec['subgoal_eval_num_samples'])
    if 'subgoal_temperature' in rec:
        temp = float(rec['subgoal_temperature'])
    config_name = params.get('config', run_group.replace('flow_trl_feval_', ''))
    horizon_val = params.get('horizon', rec.get('horizon', ''))
    rows.append({
        'run_uid': run_uid,
        'run_dir': run_dir_name,
        'run_start_ts': run_start_ts,
        'eval_timestamp': rec.get('timestamp', ''),
        'run_group': run_group,
        'config': config_name,
        'env': rec.get('env_name', params.get('env_name', '')),
        'seed': params.get('seed', ''),
        'horizon': horizon_val,
        'sweep_tag': _sweep_tag(run_group=run_group, config=config_name, horizon=horizon_val),
        'gap': rec.get('subgoal_value_gap_scale', ''),
        'maxgap': rec.get('subgoal_value_weight_max', ''),
        'train_n': rec.get('subgoal_num_samples_train', ''),
        'gamma': params.get('gamma', ''),
        'vdist_pow': params.get('vdist_pow', ''),
        'subgoal_eval_selection': params.get('subgoal_eval_selection', ''),
        'temperature': temp,
        'eval_episodes_per_task': params.get('eval_episodes_per_task', rec.get('eval_episodes_per_task', '')),
        'final_eval_episodes_per_task': params.get(
            'final_eval_episodes_per_task', rec.get('eval_episodes_per_task', ''),
        ),
        'epoch': rec.get('epoch', final_epoch),
        'eval_n': eval_n,
        'IDM': rec.get('idm_success_rate_mean', rec.get('IDM', '')),
        'ACTOR': rec.get('actor_success_rate_mean', rec.get('ACTOR', '')),
        'FLOW_IDM': four_way_means.get('eval_flow_idm', ''),
        'FLOW_ACTOR': four_way_means.get('eval_flow_actor', ''),
        'SPI_SUBGOAL_IDM': four_way_means.get('eval_spi_subgoal_idm', ''),
        'SPI_SUBGOAL_ACTOR': four_way_means.get('eval_spi_subgoal_actor', ''),
        'idm_tasks': rec.get('idm_tasks', '') or ','.join(
            f'{k}:{float(v):.4f}' for k, v in sorted(idm_tasks.items(), key=lambda x: int(x[0]))
        ),
        'actor_tasks': rec.get('actor_tasks', '') or ','.join(
            f'{k}:{float(v):.4f}' for k, v in sorted(actor_tasks.items(), key=lambda x: int(x[0]))
        ),
        'flow_idm_tasks': ','.join(
            f'{k}:{float(v):.4f}'
            for k, v in sorted((four_way_tasks.get('eval_flow_idm', {}) or {}).items(), key=lambda x: int(x[0]))
        ),
        'flow_actor_tasks': ','.join(
            f'{k}:{float(v):.4f}'
            for k, v in sorted((four_way_tasks.get('eval_flow_actor', {}) or {}).items(), key=lambda x: int(x[0]))
        ),
        'spi_subgoal_idm_tasks': ','.join(
            f'{k}:{float(v):.4f}'
            for k, v in sorted((four_way_tasks.get('eval_spi_subgoal_idm', {}) or {}).items(), key=lambda x: int(x[0]))
        ),
        'spi_subgoal_actor_tasks': ','.join(
            f'{k}:{float(v):.4f}'
            for k, v in sorted((four_way_tasks.get('eval_spi_subgoal_actor', {}) or {}).items(), key=lambda x: int(x[0]))
        ),
        'source': source,
    })


def _collect_log_records(runs_dir: str, *, final_epoch: int = 600) -> list[dict]:
    """Recover eval rows from nohup logs when JSON is missing."""
    log_dir = os.path.join(PROJECT_ROOT, 'nohup_logs')
    rows: list[dict] = []
    for log_path in sorted(glob.glob(os.path.join(log_dir, '*.log'))):
        with open(log_path, encoding='utf-8', errors='replace') as f:
            text = f.read()
        if 'eval_idm/success_rate_mean=' not in text:
            continue
        m_idm = re.search(r'eval_idm/success_rate_mean=([0-9.]+)', text)
        m_act = re.search(r'eval/success_rate_mean=([0-9.]+)', text)
        if not m_idm or not m_act:
            continue
        saved = re.search(r'Saved eval results: (.+\.json)', text)
        if saved and os.path.isfile(saved.group(1).strip()):
            continue
        run_m = re.search(r'Loaded epoch=\d+ from (.+)$', text, re.M)
        if not run_m:
            continue
        run_dir = run_m.group(1).strip()
        run_dir_name = os.path.basename(run_dir)
        params = _load_run_params(run_dir)
        temp_m = re.search(r'subgoal_temperature=([0-9.]+)', text)
        eval_n_m = re.search(r'subgoal_eval_num_samples=(\d+)', text)
        fn_n = re.search(r'\.n(\d+)\.log$', os.path.basename(log_path))
        eval_n = int(fn_n.group(1)) if fn_n else int(eval_n_m.group(1))
        temp = float(temp_m.group(1)) if temp_m else 1.0
        idm_tasks = {
            k: float(v) for k, v in re.findall(r'eval_idm/task_(\d+)/success_rate=([0-9.]+)', text)
        }
        actor_tasks = {
            k: float(v) for k, v in re.findall(r'eval/task_(\d+)/success_rate=([0-9.]+)', text)
        }
        with open(os.path.join(run_dir, 'flags.json'), encoding='utf-8') as f:
            root = json.load(f)
        fg = root.get('flags', {})
        dyn = root.get('dynamics', {})
        rec = {
            'timestamp': '',
            'run_group': fg.get('run_group', ''),
            'env_name': fg.get('env_name', ''),
            'epoch': final_epoch,
            'subgoal_eval_num_samples': eval_n,
            'subgoal_temperature': temp,
            'subgoal_value_gap_scale': dyn.get('subgoal_value_gap_scale', ''),
            'subgoal_value_weight_max': dyn.get('subgoal_value_weight_max', ''),
            'subgoal_num_samples_train': dyn.get('subgoal_num_samples', ''),
            'idm_success_rate_mean': float(m_idm.group(1)),
            'actor_success_rate_mean': float(m_act.group(1)),
            'idm_task_success_rates': idm_tasks,
            'actor_task_success_rates': actor_tasks,
            'eval_episodes_per_task': fg.get('eval_episodes_per_task', ''),
        }
        pseudo_json = f'epoch{final_epoch}_t{format(temp, "g").replace(".", "p")}_n{eval_n}.json' if temp != 1.0 else f'epoch{final_epoch}_n{eval_n}.json'
        _append_row(
            rows,
            run_dir=run_dir,
            run_dir_name=run_dir_name,
            params=params,
            rec=rec,
            json_path=pseudo_json,
            source=f'nohup:{os.path.basename(log_path)}',
            final_epoch=final_epoch,
        )
    return rows


def collect_records(runs_dir: str, *, final_epoch: int = 600) -> list[dict]:
    rows: list[dict] = []
    seen: set[tuple] = set()
    for run_dir in sorted(glob.glob(os.path.join(runs_dir, '*'))):
        if not os.path.isdir(run_dir):
            continue
        json_paths = sorted(
            set(glob.glob(os.path.join(run_dir, 'eval_results', f'epoch{int(final_epoch)}*.json'))),
            key=_eval_n_from_path,
        )
        if not json_paths:
            continue

        run_dir_name = os.path.basename(run_dir)
        params = _load_run_params(run_dir)

        for json_path in json_paths:
            with open(json_path, encoding='utf-8') as f:
                rec = json.load(f)
            _append_row(
                rows,
                run_dir=run_dir,
                run_dir_name=run_dir_name,
                params=params,
                rec=rec,
                json_path=json_path,
                source='json',
                final_epoch=final_epoch,
            )
            seen.add(_row_key(rows[-1]))

    for log_row in _collect_log_records(runs_dir, final_epoch=final_epoch):
        if _row_key(log_row) not in seen:
            rows.append(log_row)
            seen.add(_row_key(log_row))

    rows.sort(key=lambda r: (r['env'], float(r['gap'] or 0), r['run_start_ts'], float(r.get('temperature', 1) or 1), int(r['eval_n'])))
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
    n_runs = len({r['run_uid'] for r in rows})
    eval_ns = sorted({int(r['eval_n']) for r in rows if str(r.get('eval_n', '')).strip()})
    tags = sorted({r.get('sweep_tag', '') for r in rows if r.get('sweep_tag')})
    tag_counts = {t: sum(1 for r in rows if r.get('sweep_tag') == t) for t in tags}
    lines = [
        '# Flow TRL final eval (N sweep) 결과',
        '',
        f'자동 생성: {datetime.now().strftime("%Y-%m-%d %H:%M")} · `scripts/summarize_feval_results.py`',
        f'소스: `{RUNS_DIR}/*/eval_results/epoch600_n*.json`, `epoch600_t*_n*.json`, nohup log fallback',
        '',
        '**Master CSV** — K=25 gap sweep, gamma rerun, K=40 best follow-up, temp sweep 등 모든 epoch-600 env eval row 포함.',
        '',
        f'run **{n_runs}**개 · eval record **{len(rows)}**개 · eval_N={eval_ns}',
        f'sweep_tag counts: {tag_counts}',
        f'CSV: [`{docs_7ch_name("flow_trl_feval_results.csv")}`]({docs_7ch_name("flow_trl_feval_results.csv")})',
        '',
        '별도 pivot CSV ('
        f'`{docs_7ch_name("flow_trl_k40_multi_n_results.csv")}`, '
        f'`{docs_7ch_name("flow_trl_temp0.5_k25_k40_results.csv")}`'
        ')는 같은 데이터의 요약 view입니다.',
        '',
        '중복 run_group은 `run_start_ts` / `run_uid`로 구분합니다.',
        '',
        '| sweep_tag | config | env | horizon | gap | eval_n | temp | IDM | ACTOR |',
        '| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    for r in rows:
        lines.append(
            '| '
            + ' | '.join(
                str(r.get(k, ''))
                for k in ('sweep_tag', 'config', 'env', 'horizon', 'gap', 'eval_n', 'temperature', 'IDM', 'ACTOR')
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
    n_runs = len({r['run_uid'] for r in rows})
    print(f'Wrote {OUT_CSV} + {OUT_MD} ({n_runs} runs, {len(rows)} records)')


if __name__ == '__main__':
    main()
