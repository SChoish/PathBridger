#!/usr/bin/env python3
"""Summarize K=40 best runs across eval_N in {4,8,16,32}."""

from __future__ import annotations

import csv
import glob
import json
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
OUT_CSV = REPO / 'docs' / 'flow_trl_k40_multi_n_results.csv'
OUT_MD = REPO / 'docs' / 'flow_trl_k40_multi_n_results.md'
EVAL_NS = [4, 8, 16, 32]
ENV_ORDER = [
    'puzzle-3x3-play-v0',
    'puzzle-4x4-play-v0',
    'cube-single-play-v0',
    'cube-double-play-v0',
    'cube-triple-play-v0',
    'antmaze-medium-navigate-v0',
    'antmaze-large-navigate-v0',
    'antmaze-giant-navigate-v0',
]


def _env_label(env: str) -> str:
    return (
        env.replace('-navigate-v0', '')
        .replace('-play-v0', '')
        .replace('-', '_')
    )


def _resolve_k40_runs() -> list[dict]:
    rows: list[dict] = []
    for cfg in sorted(glob.glob(str(REPO / 'config' / 'sweep_flow_trl_k40_best' / '*.yaml'))):
        name = Path(cfg).name
        if name.startswith('_'):
            continue
        c = yaml.safe_load(open(cfg, encoding='utf-8'))
        env = str(c['env_name'])
        rg = str(c['run_group'])
        matches = []
        for rd in (REPO / 'runs').glob(f'*_seed0_{env}'):
            fp = rd / 'flags.json'
            if not fp.is_file():
                continue
            flags = json.load(open(fp, encoding='utf-8'))
            if flags.get('flags', {}).get('run_group') != rg:
                continue
            matches.append(rd)
        if not matches:
            continue
        run_dir = max(matches, key=lambda p: p.stat().st_mtime)
        row = {
            'config': name.removesuffix('.yaml'),
            'env': env,
            'env_label': _env_label(env),
            'gap': float(c['dynamics']['subgoal_value_gap_scale']),
            'horizon': int(c.get('horizon', 40)),
            'run_dir': str(run_dir.relative_to(REPO)),
        }
        for n in EVAL_NS:
            p = run_dir / 'eval_results' / f'epoch600_n{n}.json'
            if p.is_file():
                m = json.load(open(p, encoding='utf-8'))
                row[f'idm_n{n}'] = float(m['idm_success_rate_mean'])
                row[f'actor_n{n}'] = float(m['actor_success_rate_mean'])
            else:
                row[f'idm_n{n}'] = ''
                row[f'actor_n{n}'] = ''
        rows.append(row)
    rows.sort(key=lambda r: ENV_ORDER.index(r['env']) if r['env'] in ENV_ORDER else 99)
    return rows


def _best_n(row: dict, *, metric: str) -> tuple[int | None, float | None]:
    best_n: int | None = None
    best_v: float | None = None
    for n in EVAL_NS:
        v = row.get(f'{metric}_n{n}')
        if v == '' or v is None:
            continue
        fv = float(v)
        if best_v is None or fv > best_v:
            best_v = fv
            best_n = n
    return best_n, best_v


def write_outputs(rows: list[dict]) -> None:
    fieldnames = [
        'config', 'env', 'env_label', 'gap', 'horizon', 'run_dir',
        *(f'idm_n{n}' for n in EVAL_NS),
        *(f'actor_n{n}' for n in EVAL_NS),
        'best_idm_n', 'best_idm', 'best_actor_n', 'best_actor',
    ]
    for row in rows:
        best_idm_n, best_idm = _best_n(row, metric='idm')
        best_actor_n, best_actor = _best_n(row, metric='actor')
        row['best_idm_n'] = '' if best_idm_n is None else best_idm_n
        row['best_idm'] = '' if best_idm is None else f'{best_idm:.4f}'
        row['best_actor_n'] = '' if best_actor_n is None else best_actor_n
        row['best_actor'] = '' if best_actor is None else f'{best_actor:.4f}'

    with open(OUT_CSV, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)

    lines = [
        '# Flow TRL K=40 Multi-N Eval Results',
        '',
        f'Source runs: `config/sweep_flow_trl_k40_best/` ({len(rows)} envs)',
        f'Eval N values: {", ".join(str(n) for n in EVAL_NS)}',
        '',
        '## IDM Success Rate',
        '',
        '| env | gap | ' + ' | '.join(f'N={n}' for n in EVAL_NS) + ' | best N | best |',
        '| --- | ---: | ' + ' | '.join('---:' for _ in EVAL_NS) + ' | ---: | ---: |',
    ]
    for row in rows:
        idm_cells = []
        for n in EVAL_NS:
            v = row.get(f'idm_n{n}')
            idm_cells.append('' if v == '' else f'{float(v):.3f}')
        lines.append(
            f"| {row['env_label']} | {row['gap']:.1f} | "
            + ' | '.join(idm_cells)
            + f" | {row['best_idm_n']} | {row['best_idm']} |"
        )

    lines.extend([
        '',
        '## ACTOR Success Rate',
        '',
        '| env | gap | ' + ' | '.join(f'N={n}' for n in EVAL_NS) + ' | best N | best |',
        '| --- | ---: | ' + ' | '.join('---:' for _ in EVAL_NS) + ' | ---: | ---: |',
    ])
    for row in rows:
        actor_cells = []
        for n in EVAL_NS:
            v = row.get(f'actor_n{n}')
            actor_cells.append('' if v == '' else f'{float(v):.3f}')
        lines.append(
            f"| {row['env_label']} | {row['gap']:.1f} | "
            + ' | '.join(actor_cells)
            + f" | {row['best_actor_n']} | {row['best_actor']} |"
        )

    OUT_MD.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'Wrote {OUT_CSV}')
    print(f'Wrote {OUT_MD}')


def main() -> None:
    rows = _resolve_k40_runs()
    write_outputs(rows)


if __name__ == '__main__':
    main()
