#!/usr/bin/env python3
"""Export per-environment best Flow+TRL runs → docs/env_best_runs bundle + zip.

Selection (per env, from all feval JSON records under runs/ and
checkpoints/flow_trl_best_epoch600/):
  - Excludes eval_suffix in EXCLUDED_EVAL_SUFFIXES (e.g. fs10 flow-steps ablation)
  1. Highest ACTOR success rate
  2. Tie-break: highest IDM success rate
  3. Tie-break: latest timestamp

Outputs under docs/env_best_runs_<suffix>/ and docs/env_best_runs_<suffix>.zip
"""

from __future__ import annotations

import csv
import json
import os
import re
import shutil
import subprocess
import zipfile
from datetime import datetime
from pathlib import Path

import yaml

from docs_output_paths import DOCS_SUFFIX, docs_output_path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
RUNS_DIR = PROJECT_ROOT / 'runs'
BEST_CKPT_DIR = PROJECT_ROOT / 'checkpoints' / 'flow_trl_best_epoch600'
FEval_CSV = Path(docs_output_path(PROJECT_ROOT, 'flow_trl_feval_results', 'csv'))
BUNDLE_DIR = PROJECT_ROOT / 'docs' / f'env_best_runs_{DOCS_SUFFIX}'
ZIP_PATH = PROJECT_ROOT / 'docs' / f'env_best_runs_{DOCS_SUFFIX}.zip'

EVAL_JSON_RE = re.compile(r'^epoch(\d+)_n(\d+)(?:_(.+))?\.json$')
EXCLUDED_EVAL_SUFFIXES = frozenset({'fs10'})
CHECKPOINT_EPOCH = 600
CHECKPOINT_AGENTS = ('dynamics', 'critic', 'actor')

TRAIN_PARAM_KEYS = [
    'project', 'env', 'run_dir', 'run_group', 'seed', 'subgoal', 'critic_type',
    'gap', 'wmax', 'horizon_K', 'h_a', 'train_N', 'train_eval_N',
    'lambda', 'kappa_b', 'kappa_d', 'gamma', 'batch_size', 'train_epochs',
    'final_eval_episodes', 'subgoal_flow_steps', 'subgoal_temperature',
    'subgoal_goal_representation', 'subgoal_target_mode', 'planner_type',
    'value_hidden_dims', 'actor_hidden_dims',
]

EVAL_PARAM_KEYS = [
    'eval_epoch', 'eval_n', 'eval_episodes', 'eval_temperature', 'eval_suffix',
    'eval_score_type', 'eval_max_chunks', 'eval_flow_steps',
    'IDM', 'ACTOR', 'idm_tasks', 'actor_tasks', 'eval_timestamp',
]

CSV_COLUMNS = [
    'env', 'run_dir_abs', 'run_group',
    'project', 'run_dir', 'seed', 'subgoal', 'critic_type',
    'gap', 'wmax', 'horizon_K', 'h_a', 'train_N', 'train_eval_N',
    'lambda', 'kappa_b', 'kappa_d', 'gamma', 'batch_size', 'train_epochs',
    'final_eval_episodes', 'subgoal_flow_steps', 'subgoal_temperature',
    'subgoal_goal_representation', 'subgoal_target_mode', 'planner_type',
    'value_hidden_dims', 'actor_hidden_dims',
    *EVAL_PARAM_KEYS,
]


def _refresh_source_csvs() -> None:
    for script in ('summarize_feval_results.py', 'export_trl_completed_results.py'):
        path = SCRIPT_DIR / script
        if path.is_file():
            subprocess.run(
                [os.environ.get('PYTHON_BIN', 'python3'), str(path)],
                cwd=PROJECT_ROOT,
                check=False,
            )


def _run_roots() -> list[Path]:
    roots = []
    if RUNS_DIR.is_dir():
        roots.append(RUNS_DIR)
    if BEST_CKPT_DIR.is_dir():
        roots.append(BEST_CKPT_DIR)
    return roots


def _resolve_run_root(run_dir_name: str, *, env: str = '', run_group: str = '') -> Path | None:
    if run_dir_name:
        for root in _run_roots():
            candidate = root / run_dir_name
            if (candidate / 'flags.json').is_file() or (candidate / 'config_used.yaml').is_file():
                return candidate
    # manifest fallback: archive names → checkpoint dirs
    manifest = BEST_CKPT_DIR / 'manifest.json'
    if manifest.is_file():
        data = json.loads(manifest.read_text())
        for entry in data.get('runs', []):
            if env and entry.get('env') == env:
                run_rel = entry.get('run_dir', '')
                if run_rel:
                    p = PROJECT_ROOT / run_rel
                    if p.is_dir():
                        return p
                prefix = entry.get('archive_prefix', '')
                if prefix:
                    ck = BEST_CKPT_DIR / os.path.basename(prefix)
                    if ck.is_dir():
                        return ck
            run_rel = entry.get('run_dir', '')
            if run_dir_name and (
                run_rel.endswith(run_dir_name) or os.path.basename(run_rel) == run_dir_name
            ):
                p = PROJECT_ROOT / run_rel
                if p.is_dir():
                    return p
            prefix = entry.get('archive_prefix', '')
            if prefix and run_dir_name and run_dir_name in prefix:
                ck = BEST_CKPT_DIR / os.path.basename(prefix)
                if ck.is_dir():
                    return ck
    # flags.json scan: match run_group (+ env) when CSV run_dir is stale
    if run_group or env:
        matches: list[Path] = []
        for root in _run_roots():
            for flags_path in root.glob('*/flags.json'):
                data = json.loads(flags_path.read_text())
                fg = data.get('flags', {})
                rg = str(fg.get('run_group', ''))
                ev = str(fg.get('env_name', ''))
                if run_group and rg != run_group:
                    continue
                if env and ev != env:
                    continue
                matches.append(flags_path.parent)
        if matches:
            return sorted(matches, key=lambda p: p.name)[-1]
    return None


def _find_eval_json(run_root: Path, epoch: str | int, eval_n: str | int, suffix: str) -> Path | None:
    eval_dir = run_root / 'eval_results'
    if not eval_dir.is_dir():
        return None
    epoch_s, n_s = str(epoch), str(eval_n)
    if suffix:
        exact = eval_dir / f'epoch{epoch_s}_n{n_s}_{suffix}.json'
        if exact.is_file():
            return exact
    exact_plain = eval_dir / f'epoch{epoch_s}_n{n_s}.json'
    if exact_plain.is_file() and not suffix:
        return exact_plain
    # fuzzy: match epoch + n, prefer suffix if given
    matches = []
    for p in eval_dir.glob(f'epoch{epoch_s}_n{n_s}*.json'):
        m = EVAL_JSON_RE.match(p.name)
        if m:
            matches.append(p)
    if not matches:
        return None
    if suffix:
        for p in matches:
            if p.name.endswith(f'_{suffix}.json'):
                return p
    return sorted(matches)[-1]


def _load_feval_rows() -> list[dict]:
    if not FEval_CSV.is_file():
        raise FileNotFoundError(f'Missing {FEval_CSV}; run summarize_feval_results.py first')
    with FEval_CSV.open(encoding='utf-8') as f:
        return list(csv.DictReader(f))


def _pick_best_per_env(rows: list[dict]) -> dict[str, dict]:
    by_env: dict[str, list[dict]] = {}
    for row in rows:
        env = row.get('env', '')
        suffix = (row.get('eval_suffix') or '').strip()
        if suffix in EXCLUDED_EVAL_SUFFIXES:
            continue
        if env:
            by_env.setdefault(env, []).append(row)

    best: dict[str, dict] = {}
    for env, env_rows in by_env.items():
        def key(r: dict) -> tuple:
            try:
                actor = float(r.get('ACTOR', -1))
            except (TypeError, ValueError):
                actor = -1.0
            try:
                idm = float(r.get('IDM', -1))
            except (TypeError, ValueError):
                idm = -1.0
            ts = r.get('timestamp', '') or ''
            return (actor, idm, ts)

        best[env] = max(env_rows, key=key)
    return dict(sorted(best.items()))


def _yaml_get(text: str, key: str) -> str:
    m = re.search(rf'^\s*{re.escape(key)}:\s*(\S+)', text, re.MULTILINE)
    return m.group(1).rstrip(',') if m else ''


def _load_train_params(run_root: Path, feval_row: dict) -> dict:
    flags_path = run_root / 'flags.json'
    cfg_path = run_root / 'config_used.yaml'
    params: dict = {
        'project': 'Pathbridger_flow',
        'env': feval_row.get('env', ''),
        'run_dir': feval_row.get('run_dir', ''),
        'run_dir_abs': str(run_root),
        'run_group': feval_row.get('run_group', ''),
    }
    if flags_path.is_file():
        data = json.loads(flags_path.read_text())
        fg = data.get('flags', {})
        dyn = data.get('dynamics', {})
        critic = data.get('critic_agent', {})
        actor = data.get('actor', {})
        params.update({
            'seed': fg.get('seed', ''),
            'subgoal': dyn.get('subgoal_distribution', ''),
            'critic_type': critic.get('critic_type', critic.get('algorithm', '')),
            'gap': dyn.get('subgoal_value_gap_scale', feval_row.get('gap', '')),
            'wmax': dyn.get('subgoal_value_weight_max', feval_row.get('maxgap', '')),
            'horizon_K': fg.get('horizon', dyn.get('dynamics_N', '')),
            'h_a': critic.get('action_chunk_horizon', ''),
            'train_N': dyn.get('subgoal_num_samples', feval_row.get('train_n', '')),
            'train_eval_N': dyn.get('subgoal_eval_num_samples', ''),
            'lambda': critic.get('value_distance_weight_power', ''),
            'kappa_b': critic.get('kappa_b', ''),
            'kappa_d': critic.get('kappa_d', ''),
            'gamma': critic.get('discount', dyn.get('discount', '')),
            'batch_size': fg.get('batch_size', ''),
            'train_epochs': fg.get('train_epochs', ''),
            'final_eval_episodes': fg.get('final_eval_episodes_per_task', ''),
            'subgoal_flow_steps': dyn.get('subgoal_flow_steps', ''),
            'subgoal_temperature': dyn.get('subgoal_temperature', ''),
            'subgoal_goal_representation': dyn.get('subgoal_goal_representation', ''),
            'subgoal_target_mode': dyn.get('subgoal_target_mode', ''),
            'planner_type': dyn.get('planner_type', ''),
            'value_hidden_dims': critic.get('value_hidden_dims', ''),
            'actor_hidden_dims': actor.get('hidden_dims', ''),
        })
    if cfg_path.is_file():
        text = cfg_path.read_text()
        params.setdefault('eval_max_chunks', _yaml_get(text, 'eval_max_chunks'))
        if not params.get('subgoal_flow_steps'):
            params['subgoal_flow_steps'] = _yaml_get(text, 'subgoal_flow_steps')
    return params


def _load_eval_params(run_root: Path, feval_row: dict) -> dict:
    epoch = feval_row.get('epoch', '')
    eval_n = feval_row.get('eval_n', '')
    suffix = feval_row.get('eval_suffix', '') or ''
    temperature = feval_row.get('temperature', '')
    eval_path = _find_eval_json(run_root, epoch, eval_n, suffix)
    out = {
        'eval_epoch': epoch,
        'eval_n': eval_n,
        'eval_temperature': temperature,
        'eval_suffix': suffix,
        'IDM': feval_row.get('IDM', ''),
        'ACTOR': feval_row.get('ACTOR', ''),
        'idm_tasks': feval_row.get('idm_tasks', ''),
        'actor_tasks': feval_row.get('actor_tasks', ''),
        'eval_timestamp': feval_row.get('timestamp', ''),
        'eval_episodes': '',
        'eval_score_type': '',
        'eval_flow_steps': '',
        'eval_max_chunks': '',
    }
    if eval_path and eval_path.is_file():
        rec = json.loads(eval_path.read_text())
        out.update({
            'eval_episodes': rec.get('eval_episodes_per_task', ''),
            'eval_score_type': rec.get('subgoal_eval_score_type', ''),
            'eval_flow_steps': rec.get('subgoal_flow_steps', ''),
            'eval_json_path': str(eval_path.relative_to(PROJECT_ROOT)),
        })
    return out


def _build_records(refresh: bool = True) -> list[dict]:
    if refresh:
        _refresh_source_csvs()
    feval_rows = _load_feval_rows()
    best_by_env = _pick_best_per_env(feval_rows)
    records: list[dict] = []
    for env, feval_row in best_by_env.items():
        run_root = _resolve_run_root(
            feval_row.get('run_dir', ''),
            env=env,
            run_group=feval_row.get('run_group', ''),
        )
        if run_root is None:
            records.append({
                'env': env,
                'run_dir_abs': '',
                'run_group': feval_row.get('run_group', ''),
                'note': f"run_dir not found: {feval_row.get('run_dir')}",
                **_load_eval_params(Path('.'), feval_row),
            })
            continue
        train = _load_train_params(run_root, feval_row)
        eval_p = _load_eval_params(run_root, feval_row)
        if not eval_p.get('eval_max_chunks') and train.get('eval_max_chunks'):
            eval_p['eval_max_chunks'] = train['eval_max_chunks']
        records.append({**train, **eval_p})
    return records


def _write_csv(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction='ignore')
        writer.writeheader()
        for rec in records:
            writer.writerow(rec)


def _write_markdown(records: list[dict], path: Path) -> None:
    lines = [
        '# 환경별 Best Run 파라미터',
        '',
        f'생성: {datetime.now().strftime("%Y-%m-%d %H:%M")} · `scripts/export_env_best_runs.py`',
        '',
        '**선정 규칙:** feval JSON 전체(`runs/` + `checkpoints/flow_trl_best_epoch600/`)에서',
        '**`fs10` suffix 제외** 후 환경별 **ACTOR** 최대 → 동률 시 **IDM** → **timestamp**.',
        '',
        f'총 **{len(records)}** environments.',
        '',
        '## 요약表',
        '',
        '| env | ACTOR | IDM | gap | wmax | train_N | λ | eval_n | temp | suffix | run_group |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |',
    ]
    for r in records:
        def pct(v):
            try:
                return f'{float(v)*100:.1f}%'
            except (TypeError, ValueError):
                return '-'
        lines.append(
            '| '
            + ' | '.join([
                r.get('env', ''),
                pct(r.get('ACTOR')),
                pct(r.get('IDM')),
                str(r.get('gap', '')),
                str(r.get('wmax', '')),
                str(r.get('train_N', '')),
                str(r.get('lambda', '')),
                str(r.get('eval_n', '')),
                str(r.get('eval_temperature', '')),
                str(r.get('eval_suffix', '') or '-'),
                str(r.get('run_group', ''))[:40],
            ])
            + ' |'
        )
    lines.extend(['', '## 환경별 상세', ''])
    for r in records:
        lines.append(f'### {r.get("env", "?")}')
        lines.append('')
        lines.append(f'- run: `{r.get("run_dir_abs", r.get("run_dir", "?"))}`')
        lines.append(f'- run_group: `{r.get("run_group", "")}`')
        lines.append('')
        lines.append('**학습 파라미터**')
        lines.append('')
        for k in TRAIN_PARAM_KEYS:
            if k in ('project', 'env', 'run_dir', 'run_group'):
                continue
            v = r.get(k, '')
            if v != '' and v is not None:
                lines.append(f'- `{k}`: {v}')
        lines.append('')
        lines.append('**Eval 파라미터 (best row)**')
        lines.append('')
        for k in EVAL_PARAM_KEYS:
            v = r.get(k, '')
            if v != '' and v is not None:
                lines.append(f'- `{k}`: {v}')
        lines.append('')
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def _write_readme(records: list[dict], path: Path) -> None:
    path.write_text(
        '\n'.join([
            '# env_best_runs bundle',
            '',
            'Pathbridger_flow 프로젝트에서 지금까지 수집된 Flow+TRL feval 결과 기준,',
            '환경별 best run의 **학습 하이퍼파라미터**, **eval 설정**, **epoch 600 checkpoint**를 한곳에 모았습니다.',
            '',
            '## 파일',
            '',
            '| 파일 | 설명 |',
            '| --- | --- |',
            '| `env_best_runs_choi.csv` | 환경당 1행 flat table |',
            '| `env_best_runs_choi.md` | 요약表 + 환경별 상세 |',
            '| `env_best_params.json` | programmatic JSON |',
            '| `checkpoints/<env>/` | `flags.json` + `checkpoints/{dynamics,critic,actor}/params_600.pkl` |',
            '| `configs/<env>.yaml` | best run의 `config_used.yaml` 복사 |',
            '| `eval/<env>_best.json` | best eval JSON 스냅샷 |',
            '',
            '## 재생성',
            '',
            '```bash',
            'PYTHONPATH=.:scripts python scripts/export_env_best_runs.py',
            '```',
            '',
            f'생성 시각: {datetime.now().isoformat(timespec="seconds")}',
            f'환경 수: {len(records)}',
            '',
        ]) + '\n',
        encoding='utf-8',
    )


def _copy_artifacts(records: list[dict], bundle_dir: Path) -> None:
    cfg_dir = bundle_dir / 'configs'
    eval_dir = bundle_dir / 'eval'
    ckpt_root = bundle_dir / 'checkpoints'
    cfg_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)
    ckpt_root.mkdir(parents=True, exist_ok=True)
    for rec in records:
        env = rec.get('env', 'unknown')
        safe = env.replace('/', '_')
        run_root_s = rec.get('run_dir_abs', '')
        if not run_root_s:
            rec['checkpoint_bundle'] = ''
            rec['checkpoint_missing'] = 'run_dir_abs'
            continue
        run_root = Path(run_root_s)
        cfg = run_root / 'config_used.yaml'
        if cfg.is_file():
            shutil.copy2(cfg, cfg_dir / f'{safe}.yaml')
        eval_path = rec.get('eval_json_path')
        if eval_path:
            src = PROJECT_ROOT / eval_path
            if src.is_file():
                shutil.copy2(src, eval_dir / f'{safe}_best.json')

        dest = ckpt_root / safe
        dest.mkdir(parents=True, exist_ok=True)
        flags = run_root / 'flags.json'
        if flags.is_file():
            shutil.copy2(flags, dest / 'flags.json')
        if cfg.is_file():
            shutil.copy2(cfg, dest / 'config_used.yaml')

        missing: list[str] = []
        copied: list[str] = []
        for agent in CHECKPOINT_AGENTS:
            src_ckpt = run_root / 'checkpoints' / agent / f'params_{CHECKPOINT_EPOCH}.pkl'
            if not src_ckpt.is_file():
                missing.append(agent)
                continue
            out_dir = dest / 'checkpoints' / agent
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f'params_{CHECKPOINT_EPOCH}.pkl'
            shutil.copy2(src_ckpt, out_path)
            copied.append(str(out_path.relative_to(bundle_dir)))

        rec['checkpoint_epoch'] = CHECKPOINT_EPOCH
        rec['checkpoint_bundle'] = str(dest.relative_to(bundle_dir))
        rec['checkpoint_files'] = copied
        rec['checkpoint_missing'] = ','.join(missing) if missing else ''


def _write_json(records: list[dict], path: Path) -> None:
    payload = {
        'generated_at': datetime.now().isoformat(timespec='seconds'),
        'selection_rule': 'exclude eval_suffix=fs10; then max ACTOR → max IDM → latest timestamp',
        'source_csv': str(FEval_CSV.relative_to(PROJECT_ROOT)),
        'environments': records,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def _make_zip(bundle_dir: Path, zip_path: Path) -> None:
    if zip_path.is_file():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(bundle_dir.rglob('*')):
            if path.is_file():
                zf.write(path, path.relative_to(bundle_dir.parent))


def export_bundle(*, refresh: bool = True) -> tuple[Path, Path]:
    records = _build_records(refresh=refresh)
    if bundle_dir_exists := BUNDLE_DIR.is_dir():
        shutil.rmtree(BUNDLE_DIR)
    BUNDLE_DIR.mkdir(parents=True)
    stem = f'env_best_runs_{DOCS_SUFFIX}'
    _write_csv(records, BUNDLE_DIR / f'{stem}.csv')
    _write_markdown(records, BUNDLE_DIR / f'{stem}.md')
    _copy_artifacts(records, BUNDLE_DIR)
    _write_json(records, BUNDLE_DIR / 'env_best_params.json')
    _write_readme(records, BUNDLE_DIR / 'README.md')
    _make_zip(BUNDLE_DIR, ZIP_PATH)
    return BUNDLE_DIR, ZIP_PATH


def main() -> None:
    bundle, zpath = export_bundle(refresh=True)
    n_env = sum(1 for _ in open(bundle / f'env_best_runs_{DOCS_SUFFIX}.csv')) - 1
    dir_mb = sum(f.stat().st_size for f in bundle.rglob('*') if f.is_file()) / (1024 * 1024)
    zip_mb = zpath.stat().st_size / (1024 * 1024)
    missing = sum(1 for line in open(bundle / 'env_best_params.json') if 'checkpoint_missing' in line and '""' not in line)
    print(f'Wrote {bundle}/ ({n_env} envs, {dir_mb:.0f} MB) → {zpath} ({zip_mb:.0f} MB)')


if __name__ == '__main__':
    main()
