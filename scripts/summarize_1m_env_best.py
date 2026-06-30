#!/usr/bin/env python3
"""Regenerate checkpoints/1m_env_best summary with best SPI sweep columns.

Scans actor_spi / subgoal_spi 1M finetune runs (epoch_1000000*) and picks
best τ per env.  Per τ run, score = max over eval checkpoints during the
100K finetune (@50K and @100K), not final-only:
  - actor SPI: max actor_success_rate_mean in eval_results/
  - subgoal SPI: max four_way eval_spi_subgoal_actor in eval_results/
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
CKPT_ROOT = ROOT / 'checkpoints' / '1m_env_best'
ACTOR_SPI_EXTERNAL = CKPT_ROOT / 'actor_spi_external.yaml'
LABEL_ORDER = ['amm', 'aml', 'amg', 'cs', 'cd', 'ct', 'p3', 'p4']


def _load_env_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for meta_path in sorted(CKPT_ROOT.glob('*/best_eval_meta.yaml')):
        meta = yaml.safe_load(meta_path.read_text(encoding='utf-8'))
        be = meta.get('best_eval') or {}
        bae = meta.get('best_actor_eval') or {}
        bundle = meta_path.parent.name
        row: dict[str, Any] = {
            'label': meta.get('label', bundle.split('_')[0]),
            'env': meta.get('env', ''),
            'env_short': (meta.get('env', '')
                          .replace('-navigate-v0', '')
                          .replace('-play-v0', '')),
            'config': meta.get('config', ''),
            'gap': be.get('gap'),
            'checkpoint_bundle': f'checkpoints/1m_env_best/{bundle}',
            'source_run_dir': meta.get('source_run_dir', ''),
            'eval_n': be.get('eval_N'),
            'eval_temperature': be.get('temp'),
            'IDM': be.get('IDM'),
            'ACTOR': be.get('ACTOR'),
            'eval_json_path': be.get('eval_json', ''),
        }
        if bae:
            row['best_actor_eval_n'] = bae.get('eval_N')
            row['best_actor_eval_temperature'] = bae.get('temp')
            row['best_actor_IDM'] = bae.get('IDM')
            row['best_actor_ACTOR'] = bae.get('ACTOR')
        rows.append(row)
    rows.sort(key=lambda r: LABEL_ORDER.index(r['label']) if r['label'] in LABEL_ORDER else 99)
    return rows


def _parse_tau_tag(tau_dir_name: str) -> float:
    tag = tau_dir_name.removeprefix('tau_')
    return float(tag.replace('p', '.'))


def _best_over_100k_eval(
    run_dir: Path,
    *,
    actor_metric: str = 'actor_success_rate_mean',
    subgoal_metric: str = 'eval_spi_subgoal_actor',
    mode: str = 'actor',
) -> dict[str, Any] | None:
    eval_dir = run_dir / 'eval_results'
    if not eval_dir.is_dir():
        return None

    best: dict[str, Any] | None = None
    for eval_path in sorted(eval_dir.glob('epoch*.json')):
        record = json.loads(eval_path.read_text(encoding='utf-8'))
        if mode == 'subgoal':
            four_way = record.get('four_way_success_rate_means') or {}
            score = four_way.get(subgoal_metric)
            idm = four_way.get('eval_spi_subgoal_idm')
        else:
            score = record.get(actor_metric)
            idm = record.get('idm_success_rate_mean')
        if score is None or score != score:
            continue
        score = float(score)
        rec = {
            'score': score,
            'idm': float(idm) if idm is not None and idm == idm else None,
            'step': int(record.get('epoch', 0) or 0),
            'eval_json': str(eval_path.relative_to(ROOT)),
        }
        if best is None or score > best['score']:
            best = rec
    return best


def _scan_spi_best(
    spi_root: Path,
    *,
    mode: str,
    epoch_prefix: str = 'epoch_1000000',
) -> dict[str, dict[str, Any]]:
    best_by_env: dict[str, dict[str, Any]] = {}
    if not spi_root.is_dir():
        return best_by_env

    for run_dir in sorted(spi_root.glob(f'**/{epoch_prefix}*/tau_*/seed_*')):
        if not run_dir.is_dir():
            continue
        source_tag = run_dir.parent.parent.name
        tau = _parse_tau_tag(run_dir.parent.name)
        run_best = _best_over_100k_eval(run_dir, mode=mode)
        if run_best is None:
            continue

        env = None
        meta_path = run_dir / ('actor_spi_meta.yaml' if mode == 'actor' else 'subgoal_spi_meta.yaml')
        if meta_path.is_file():
            meta = yaml.safe_load(meta_path.read_text(encoding='utf-8'))
            env = meta.get('env')
        if not env:
            for eval_path in sorted((run_dir / 'eval_results').glob('epoch*.json')):
                env = json.loads(eval_path.read_text(encoding='utf-8')).get('env_name')
                if env:
                    break
        if not env:
            continue

        rec = {
            'tau': tau,
            'score': run_best['score'],
            'idm': run_best.get('idm'),
            'best_step': run_best['step'],
            'eval_json_path': run_best['eval_json'],
            'run_dir': str(run_dir.relative_to(ROOT)),
            'source_tag': source_tag,
        }
        if env not in best_by_env or rec['score'] > best_by_env[env]['score']:
            best_by_env[env] = rec
    return best_by_env


def _pct(v: float | None) -> str:
    if v is None or v != v:
        return '—'
    return f'{100 * v:.1f}%'


def _signed_pp(v: float | None) -> str:
    if v is None or v != v:
        return '—'
    return f'{v:+.1f}%'


def _format_tau(row: dict[str, Any]) -> str:
    if row.get('actor_spi_best_tau_display'):
        return str(row['actor_spi_best_tau_display'])
    tau = row.get('actor_spi_best_tau')
    if tau is None:
        return '—'
    if isinstance(tau, list):
        return ' / '.join(f'{t:g}' for t in tau)
    return f'{tau:g}'


def _load_actor_spi_external() -> dict[str, dict[str, Any]]:
    if not ACTOR_SPI_EXTERNAL.is_file():
        return {}
    data = yaml.safe_load(ACTOR_SPI_EXTERNAL.read_text(encoding='utf-8')) or {}
    by_env: dict[str, dict[str, Any]] = {}
    for rec in data.get('runs') or []:
        env = rec.get('env')
        if env:
            by_env[str(env)] = rec
    return by_env


def _apply_actor_spi(row: dict[str, Any], rec: dict[str, Any], *, external: bool) -> None:
    row['actor_spi_best_actor'] = float(rec['actor_spi_best_actor'])
    row['actor_spi_best_idm'] = float(rec.get('actor_spi_best_idm', float('nan')))
    tau = rec.get('best_tau')
    row['actor_spi_best_tau'] = tau
    row['actor_spi_best_tau_display'] = rec.get('best_tau_display') or (
        ' / '.join(f'{t:g}' for t in tau) if isinstance(tau, list) else f'{tau:g}'
    )
    if 'vs_idm_pp' in rec:
        row['actor_spi_vs_idm_pp'] = float(rec['vs_idm_pp'])
    if 'vs_baseline_actor_pp' in rec:
        row['actor_spi_vs_baseline_actor_pp'] = float(rec['vs_baseline_actor_pp'])
    if external:
        row['actor_spi_source'] = 'external'
        row['actor_spi_external_note'] = rec.get('source_note') or 'actor_spi_external.yaml'
    else:
        row['actor_spi_source'] = 'local'
        if rec.get('best_step') is not None:
            row['actor_spi_best_step'] = rec['best_step']
        if rec.get('eval_json_path'):
            row['actor_spi_eval_json_path'] = rec['eval_json_path']
        if rec.get('run_dir'):
            row['actor_spi_run_dir'] = rec['run_dir']


def _attach_spi(rows: list[dict[str, Any]]) -> None:
    actor_best = _scan_spi_best(ROOT / 'checkpoints' / 'actor_spi', mode='actor')
    actor_external = _load_actor_spi_external()
    subgoal_best = _scan_spi_best(ROOT / 'checkpoints' / 'subgoal_spi', mode='subgoal')
    for row in rows:
        env = row['env']
        ext = actor_external.get(env)
        if ext:
            _apply_actor_spi(row, ext, external=True)
        elif env in actor_best:
            ab = actor_best[env]
            _apply_actor_spi(
                row,
                {
                    'actor_spi_best_actor': ab['score'],
                    'actor_spi_best_idm': ab.get('idm'),
                    'best_tau': ab['tau'],
                    'best_step': ab.get('best_step'),
                    'eval_json_path': ab.get('eval_json_path'),
                    'run_dir': ab.get('run_dir'),
                },
                external=False,
            )
            baseline = row.get('ACTOR')
            if baseline is not None and baseline == baseline:
                row['actor_spi_vs_baseline_actor_pp'] = 100 * (
                    row['actor_spi_best_actor'] - float(baseline)
                )
            idm = row.get('actor_spi_best_idm')
            if idm is not None and idm == idm:
                row['actor_spi_vs_idm_pp'] = 100 * (row['actor_spi_best_actor'] - float(idm))

        sb = subgoal_best.get(env)
        if sb:
            row['subgoal_spi_best_actor'] = sb['score']
            row['subgoal_spi_best_idm'] = sb.get('idm')
            row['subgoal_spi_best_tau'] = sb['tau']
            row['subgoal_spi_best_step'] = sb['best_step']
            row['subgoal_spi_eval_json_path'] = sb['eval_json_path']
            row['subgoal_spi_run_dir'] = sb['run_dir']


def _write_json(rows: list[dict[str, Any]]) -> None:
    payload = {
        'generated_at': str(date.today()),
        'checkpoint_epoch': 1000000,
        'selection_rule': 'best IDM over eval temp/N grid (see per-env best_eval_meta.yaml)',
        'spi_sweep_rule': (
            'per tau: max eval metric over 100K finetune checkpoints (@50K, @100K); '
            'per env: best over tau in {0.5,1,3,10}'
        ),
        'checkpoint_root': 'checkpoints/1m_env_best',
        'environments': rows,
    }
    out = CKPT_ROOT / 'env_best_params.json'
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def _write_readme(rows: list[dict[str, Any]]) -> None:
    lines = [
        '# 1M Env Best (IDM 기준)',
        '',
        'Flow+TRL 1M checkpoint + eval grid에서 **best IDM** 설정을 env별로 모은 bundle.',
        '',
        '- **epoch**: 1,000,000',
        '- **선정**: temp × eval_N grid에서 IDM 최대 (동률 시 ACTOR)',
        '- **SPI finetune** (actor/subgoal SPI): 아래 `eval_N`, `temp`를 proposal/eval N/T로 사용',
        '',
        '## Best eval (IDM 선정)',
        '',
        '| label | env | IDM | ACTOR | actor SPI best | τ | subgoal SPI best | τ | gap | eval_N | temp | config |',
        '|-------|-----|----:|------:|---------------:|--:|-----------------:|--:|----:|-------:|-----:|--------|',
    ]
    for r in rows:
        actor_spi = _pct(r.get('actor_spi_best_actor'))
        actor_tau_s = _format_tau(r)
        subgoal_spi = _pct(r.get('subgoal_spi_best_actor'))
        subgoal_tau = r.get('subgoal_spi_best_tau')
        subgoal_tau_s = f'{subgoal_tau:g}' if subgoal_tau is not None else '—'
        lines.append(
            f"| {r['label']} | {r['env_short']} | **{_pct(r['IDM'])}** | {_pct(r['ACTOR'])} "
            f"| {actor_spi} | {actor_tau_s} | {subgoal_spi} | {subgoal_tau_s} "
            f"| {r['gap']} | {r['eval_n']} | {r['eval_temperature']} | `{r['config']}` |"
        )
    lines.extend([
        '',
        '> **Bold** = IDM (선정 기준). ACTOR는 같은 (N, temp) 설정에서의 actor success rate.',
        '> **actor SPI best** = τ sweep {0.5,1,3,10} 각 run의 100K 중 eval actor SR 최대, 그중 env best.',
        '> **subgoal SPI best** = τ sweep {0.5,1,3,10} 각 run의 100K 중 eval_spi_subgoal_actor 최대, 그중 env best (미완료 시 —).',
        '',
        '## Actor SPI best (100K finetune)',
        '',
        'τ ∈ {0.5, 1, 3, 10} sweep · per-τ score = 100K 중 eval actor SR max (@50K/@100K) · env best = τ 중 max.',
        'antmaze(amm/aml/amg)는 다른 machine 결과 → [actor_spi_external.yaml](actor_spi_external.yaml).',
        '',
        '| env | best τ | 100K actor | 100K IDM | vs IDM | vs 1M ckpt actor | step | src |',
        '|-----|-------:|-----------:|---------:|-------:|-----------------:|-----:|-----|',
    ])
    for r in rows:
        if r.get('actor_spi_best_actor') is None:
            lines.append(f"| {r['label']} | — | — | — | — | — | — | — |")
            continue
        baseline = _pct(r.get('ACTOR'))
        step = r.get('actor_spi_best_step')
        step_s = f'{int(step) // 1000}K' if step else '—'
        src = 'ext' if r.get('actor_spi_source') == 'external' else 'local'
        lines.append(
            f"| {r['label']} | {_format_tau(r)} | {_pct(r.get('actor_spi_best_actor'))} "
            f"| {_pct(r.get('actor_spi_best_idm'))} | {_signed_pp(r.get('actor_spi_vs_idm_pp'))} "
            f"| {_signed_pp(r.get('actor_spi_vs_baseline_actor_pp'))} ({baseline}) | {step_s} | {src} |"
        )
    lines.extend([
        '',
        '## Best ACTOR (참고, IDM 선정과 다른 N/T)',
        '',
        '| label | ACTOR | IDM | eval_N | temp |',
        '|-------|------:|----:|-------:|-----:|',
    ])
    for r in rows:
        if 'best_actor_ACTOR' not in r:
            continue
        lines.append(
            f"| {r['label']} | **{_pct(r['best_actor_ACTOR'])}** | {_pct(r['best_actor_IDM'])} "
            f"| {r['best_actor_eval_n']} | {r['best_actor_eval_temperature']} |"
        )
    lines.extend([
        '',
        '## Eval grid',
        '',
        '| 그룹 | temp | eval_N |',
        '|------|------|--------|',
        '| antmaze / cube / p4 (local) | {0, 0.25, 0.5} | {1, 2, 8, 16, 32} |',
        '| p3 (remote) | {0, 0.25, 0.5, 1.0} | {1, 2, 8, 16, 32} |',
        '',
        '## Bundle 구조',
        '',
        '```',
        '<label>_<env>/',
        '  checkpoints/{dynamics,critic,actor}/params_1000000.pkl',
        '  config_used.yaml',
        '  flags.json',
        '  best_eval_meta.yaml',
        '  eval_results/epoch1000000_t*_n*.json',
        '```',
        '',
        '## Machine-readable manifest',
        '',
        '- [manifest.yaml](manifest.yaml) — 요약 + source run dir',
        '- [env_best_params.json](env_best_params.json) — SPI finetune / 스크립트용',
        '- [1m_env_best.md](1m_env_best.md) — 이 README와 동일 (md export)',
        '- [actor_spi_external.yaml](actor_spi_external.yaml) — antmaze actor SPI (external run)',
        '',
        '재생성: `python scripts/summarize_1m_env_best.py`',
        '',
        '## SPI finetune sweep (actor / subgoal)',
        '',
        '공통 설정: τ ∈ {0.5, 1, 3, 10}, 100K steps, lr=3e-4, eval/save @50K/100K',
        '',
        '```bash',
        '# actor SPI (전 env)',
        'bash scripts/run_actor_spi_1m_sweep.sh 0',
        '',
        '# subgoal SPI (전 env)',
        'bash scripts/run_subgoal_spi_1m_sweep.sh 0',
        '```',
        '',
        '각 env의 N/T는 위 best IDM table과 동일.',
        '',
        '## Eval 예시',
        '',
        '```bash',
        'python eval_checkpoint.py \\',
        '  --run_dir checkpoints/1m_env_best/p3_puzzle-3x3 \\',
        '  --epoch 1000000 \\',
        '  --subgoal_temperature 1.0 \\',
        '  --subgoal_eval_num_samples 32',
        '```',
        '',
        '## Source runs',
        '',
        '| label | source_run_dir |',
        '|-------|----------------|',
    ])
    for r in rows:
        lines.append(f"| {r['label']} | `{r['source_run_dir']}` |")
    lines.append('')
    readme_text = '\n'.join(lines)
    (CKPT_ROOT / 'README.md').write_text(readme_text, encoding='utf-8')
    (CKPT_ROOT / '1m_env_best.md').write_text(readme_text, encoding='utf-8')


def main() -> None:
    rows = _load_env_rows()
    _attach_spi(rows)
    _write_json(rows)
    _write_readme(rows)
    print(f'Wrote {CKPT_ROOT / "env_best_params.json"}')
    print(f'Wrote {CKPT_ROOT / "README.md"}')
    print(f'Wrote {CKPT_ROOT / "1m_env_best.md"}')
    for r in rows:
        ab = _pct(r.get('actor_spi_best_actor'))
        sb = _pct(r.get('subgoal_spi_best_actor'))
        print(f"  {r['label']:4s} actor_spi={ab:>6s}  subgoal_spi={sb:>6s}")


if __name__ == '__main__':
    main()
