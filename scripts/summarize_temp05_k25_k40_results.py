#!/usr/bin/env python3
"""Summarize K=25/K=40 evals at subgoal_temperature=0.5 across N=2,4,8,16,32."""

from __future__ import annotations

import csv
import glob
import json
import os
import re
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
TEMP = float(os.environ.get('TEMP', os.environ.get('SUBGOAL_TEMPERATURE', '0.5')))
EVAL_NS = [2, 4, 8, 16, 32]
OUT_CSV = REPO / 'docs' / f'flow_trl_temp{TEMP:g}_k25_k40_results.csv'
OUT_MD = REPO / f'docs/flow_trl_temp{TEMP:g}_k25_k40_results.md'

K25_SPECS = [
    ('puzzle-3x3-play-v0', 'flow_trl_feval_p3_g1_w5_n1', 'p3_g1_w5_n1', 25),
    ('puzzle-4x4-play-v0', 'flow_trl_feval_p4_g5_w5_n1', 'p4_g5_w5_n1', 25),
    ('cube-single-play-v0', 'flow_trl_feval_cs_g10_w5_n1', 'cs_g10_w5_n1', 25),
    ('cube-double-play-v0', 'flow_trl_feval_cd_g10_w5_n1', 'cd_g10_w5_n1', 25),
    ('cube-triple-play-v0', 'flow_trl_feval_ct_g10_w5_n1', 'ct_g10_w5_n1', 25),
    ('antmaze-medium-navigate-v0', 'flow_trl_feval_amm_g3_w5_n1', 'amm_g3_w5_n1', 25),
    ('antmaze-large-navigate-v0', 'flow_trl_feval_aml_g10_w5_n1', 'aml_g10_w5_n1', 25),
    ('antmaze-giant-navigate-v0', 'flow_trl_feval_amg_g3_w5_n1', 'amg_g3_w5_n1', 25),
]


def _temp_tag(temperature: float) -> str:
    if abs(temperature - round(temperature)) < 1e-9:
        return str(int(round(temperature)))
    return format(temperature, 'g').replace('.', 'p')


def _env_label(env: str) -> str:
    return env.replace('-navigate-v0', '').replace('-play-v0', '').replace('-', '_')


def _resolve_run(env: str, run_group: str) -> Path | None:
    matches = []
    for rd in (REPO / 'runs').glob(f'*_seed0_{env}'):
        fp = rd / 'flags.json'
        if not fp.is_file():
            continue
        flags = json.load(open(fp, encoding='utf-8'))
        if flags.get('flags', {}).get('run_group') != run_group:
            continue
        matches.append(rd)
    if not matches:
        return None
    return max(matches, key=lambda p: p.stat().st_mtime)


def _load_metric(run_dir: Path, eval_n: int) -> tuple[str, str]:
    p = run_dir / 'eval_results' / f'epoch600_t{_temp_tag(TEMP)}_n{eval_n}.json'
    if not p.is_file():
        return '', ''
    m = json.load(open(p, encoding='utf-8'))
    return f"{float(m['idm_success_rate_mean']):.4f}", f"{float(m['actor_success_rate_mean']):.4f}"


def _best_n(row: dict, prefix: str) -> tuple[str, str]:
    best_n = ''
    best_v = None
    for n in EVAL_NS:
        v = row.get(f'{prefix}_n{n}', '')
        if not v:
            continue
        fv = float(v)
        if best_v is None or fv > best_v:
            best_v = fv
            best_n = str(n)
    return best_n, '' if best_v is None else f'{best_v:.4f}'


def collect_rows() -> list[dict]:
    rows: list[dict] = []
    for env, rg, cfg, horizon in K25_SPECS:
        rd = _resolve_run(env, rg)
        if rd is None:
            continue
        row = {
            'horizon': horizon,
            'config': cfg,
            'env': env,
            'env_label': _env_label(env),
            'run_dir': str(rd.relative_to(REPO)),
            'temperature': TEMP,
        }
        for n in EVAL_NS:
            idm, actor = _load_metric(rd, n)
            row[f'idm_n{n}'] = idm
            row[f'actor_n{n}'] = actor
        rows.append(row)

    for cfg_path in sorted(glob.glob(str(REPO / 'config' / 'sweep_flow_trl_k40_best' / '*.yaml'))):
        name = Path(cfg_path).name
        if name.startswith('_'):
            continue
        c = yaml.safe_load(open(cfg_path, encoding='utf-8'))
        env = str(c['env_name'])
        rg = str(c['run_group'])
        rd = _resolve_run(env, rg)
        if rd is None:
            continue
        row = {
            'horizon': 40,
            'config': name.removesuffix('.yaml'),
            'env': env,
            'env_label': _env_label(env),
            'run_dir': str(rd.relative_to(REPO)),
            'temperature': TEMP,
        }
        for n in EVAL_NS:
            idm, actor = _load_metric(rd, n)
            row[f'idm_n{n}'] = idm
            row[f'actor_n{n}'] = actor
        rows.append(row)
    return rows


def write_outputs(rows: list[dict]) -> None:
    fieldnames = [
        'horizon', 'config', 'env', 'env_label', 'temperature', 'run_dir',
        *(f'idm_n{n}' for n in EVAL_NS),
        *(f'actor_n{n}' for n in EVAL_NS),
        'best_idm_n', 'best_idm', 'best_actor_n', 'best_actor',
    ]
    for row in rows:
        row['best_idm_n'], row['best_idm'] = _best_n(row, 'idm')
        row['best_actor_n'], row['best_actor'] = _best_n(row, 'actor')

    with open(OUT_CSV, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)

    lines = [
        f'# Flow TRL temp={TEMP:g} K=25 vs K=40 Eval Results',
        '',
        f'Eval N values: {", ".join(str(n) for n in EVAL_NS)}',
        f'Files: `runs/<run>/eval_results/epoch600_t{_temp_tag(TEMP)}_n<N>.json`',
        '',
        '## IDM Success Rate',
        '',
        '| horizon | env | ' + ' | '.join(f'N={n}' for n in EVAL_NS) + ' | best N | best |',
        '| ---: | --- | ' + ' | '.join('---:' for _ in EVAL_NS) + ' | ---: | ---: |',
    ]
    for row in rows:
        cells = [row.get(f'idm_n{n}', '') for n in EVAL_NS]
        lines.append(
            f"| {row['horizon']} | {row['env_label']} | "
            + ' | '.join(cells)
            + f" | {row['best_idm_n']} | {row['best_idm']} |"
        )

    lines.extend([
        '',
        '## ACTOR Success Rate',
        '',
        '| horizon | env | ' + ' | '.join(f'N={n}' for n in EVAL_NS) + ' | best N | best |',
        '| ---: | --- | ' + ' | '.join('---:' for _ in EVAL_NS) + ' | ---: | ---: |',
    ])
    for row in rows:
        cells = [row.get(f'actor_n{n}', '') for n in EVAL_NS]
        lines.append(
            f"| {row['horizon']} | {row['env_label']} | "
            + ' | '.join(cells)
            + f" | {row['best_actor_n']} | {row['best_actor']} |"
        )

    OUT_MD.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'Wrote {OUT_CSV}')
    print(f'Wrote {OUT_MD}')


def main() -> None:
    write_outputs(collect_rows())


if __name__ == '__main__':
    main()
