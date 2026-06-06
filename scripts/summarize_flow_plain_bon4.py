#!/usr/bin/env python3
"""Summarize plain Flow-BC + eval BoN4 runs under runs/."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_ROOT = REPO_ROOT / 'runs'
OUT_CSV = REPO_ROOT / 'results' / 'flow_plain_bon4_summary.csv'

MATCH_KEYS = {
    'subgoal_distribution': 'flow',
    'subgoal_flow_energy_weighted': False,
    'subgoal_eval_selection': 'best_of_n_value',
    'subgoal_eval_num_samples': 4,
}

METRIC_SUFFIXES = (
    ('best_actor_success', 'eval/success_rate_mean'),
    ('final_actor_success', 'eval/success_rate_mean'),
    ('best_idm_success', 'eval_idm/success_rate_mean'),
    ('final_idm_success', 'eval_idm/success_rate_mean'),
    ('final_loss_subgoal', 'train/dynamics/phase1/loss_subgoal_epoch_mean'),
    ('final_flow_fm_raw', 'train/dynamics/phase1/subgoal_flow_fm_raw_epoch_mean'),
    ('final_flow_energy_weighted', 'train/dynamics/phase1/subgoal_flow_energy_weighted_epoch_mean'),
    ('final_proposal_goal_std', 'train/coupling/proposal_goal_std_mean_epoch_mean'),
    ('final_proposal_count', 'train/coupling/proposal_count_epoch_mean'),
)


def _load_yaml(path: Path) -> dict:
    with open(path, encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def _load_run_config(run_dir: Path) -> dict[str, Any]:
    flags_path = run_dir / 'flags.json'
    if flags_path.is_file():
        with open(flags_path, encoding='utf-8') as f:
            data = json.load(f)
        dyn = (data.get('dynamics') or {})
        return dyn
    cfg_path = run_dir / 'config_used.yaml'
    if cfg_path.is_file():
        cfg = _load_yaml(cfg_path)
        return cfg.get('dynamics') or {}
    return {}


def _matches_plain_bon4(dyn: dict[str, Any]) -> bool:
    for key, expected in MATCH_KEYS.items():
        if dyn.get(key) != expected:
            return False
    return True


def _read_train_csv(run_dir: Path) -> tuple[list[str], list[dict[str, str]]]:
    csv_path = run_dir / 'train.csv'
    if not csv_path.is_file():
        return [], []
    with open(csv_path, encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    return fieldnames, rows


def _metric_at(rows: list[dict[str, str]], col: str, epoch: int) -> str:
    for row in reversed(rows):
        if row.get('train/epoch') == str(epoch):
            val = row.get(col, '')
            if val not in ('', None):
                return val
    return ''


def _best_and_final(rows: list[dict[str, str]], col: str) -> tuple[str, str, str, str]:
    best_val = ''
    best_epoch = ''
    final_val = ''
    final_epoch = ''
    eval_rows = [r for r in rows if r.get(col, '') not in ('', None)]
    if not eval_rows:
        return best_val, best_epoch, final_val, final_epoch
    final_row = eval_rows[-1]
    final_epoch = final_row.get('train/epoch', '')
    final_val = final_row.get(col, '')
    best_row = max(eval_rows, key=lambda r: float(r[col]))
    best_val = best_row.get(col, '')
    best_epoch = best_row.get('train/epoch', '')
    return best_val, best_epoch, final_val, final_epoch


def _final_metric(rows: list[dict[str, str]], col: str) -> str:
    for row in reversed(rows):
        val = row.get(col, '')
        if val not in ('', None):
            return val
    return ''


def _env_name_for_run(run_dir: Path, dyn: dict[str, Any]) -> str:
    flags_path = run_dir / 'flags.json'
    if flags_path.is_file():
        with open(flags_path, encoding='utf-8') as f:
            data = json.load(f)
        env = (data.get('flags') or {}).get('env_name')
        if env:
            return str(env)
    cfg_path = run_dir / 'config_used.yaml'
    if cfg_path.is_file():
        cfg = _load_yaml(cfg_path)
        if cfg.get('env_name'):
            return str(cfg['env_name'])
    return str(dyn.get('env_name', run_dir.name))


def _summarize_run(run_dir: Path) -> dict[str, str] | None:
    dyn = _load_run_config(run_dir)
    if not _matches_plain_bon4(dyn):
        return None
    _, rows = _read_train_csv(run_dir)
    if not rows:
        return None

    out: dict[str, str] = {
        'env_name': _env_name_for_run(run_dir, dyn),
        'run_dir': str(run_dir.relative_to(REPO_ROOT)),
        'residual_target_mode': str(dyn.get('residual_target_mode', '')),
        'subgoal_target_mode': str(dyn.get('subgoal_target_mode', '')),
    }

    best_actor, best_actor_ep, final_actor, final_actor_ep = _best_and_final(
        rows, 'eval/success_rate_mean',
    )
    best_idm, best_idm_ep, final_idm, final_idm_ep = _best_and_final(
        rows, 'eval_idm/success_rate_mean',
    )
    out.update(
        {
            'best_actor_success': best_actor,
            'best_actor_epoch': best_actor_ep,
            'final_actor_success': final_actor,
            'final_actor_epoch': final_actor_ep,
            'best_idm_success': best_idm,
            'best_idm_epoch': best_idm_ep,
            'final_idm_success': final_idm,
            'final_idm_epoch': final_idm_ep,
            'final_loss_subgoal': _final_metric(rows, 'train/dynamics/phase1/loss_subgoal_epoch_mean'),
            'final_flow_fm_raw': _final_metric(rows, 'train/dynamics/phase1/subgoal_flow_fm_raw_epoch_mean'),
            'final_flow_energy_weighted': _final_metric(
                rows, 'train/dynamics/phase1/subgoal_flow_energy_weighted_epoch_mean',
            ),
            'final_proposal_goal_std': _final_metric(
                rows, 'train/coupling/proposal_goal_std_mean_epoch_mean',
            ),
            'final_proposal_count': _final_metric(rows, 'train/coupling/proposal_count_epoch_mean'),
        }
    )
    return out


def main() -> int:
    if not RUNS_ROOT.is_dir():
        print(f'No runs directory: {RUNS_ROOT}', file=sys.stderr)
        return 1

    summaries: list[dict[str, str]] = []
    for run_dir in sorted(RUNS_ROOT.iterdir()):
        if not run_dir.is_dir():
            continue
        row = _summarize_run(run_dir)
        if row is not None:
            summaries.append(row)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        'env_name',
        'residual_target_mode',
        'subgoal_target_mode',
        'run_dir',
        'best_actor_success',
        'best_actor_epoch',
        'final_actor_success',
        'final_actor_epoch',
        'best_idm_success',
        'best_idm_epoch',
        'final_idm_success',
        'final_idm_epoch',
        'final_loss_subgoal',
        'final_flow_fm_raw',
        'final_flow_energy_weighted',
        'final_proposal_goal_std',
        'final_proposal_count',
    ]
    with open(OUT_CSV, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)

    print(f'Wrote {len(summaries)} rows to {OUT_CSV.relative_to(REPO_ROOT)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
