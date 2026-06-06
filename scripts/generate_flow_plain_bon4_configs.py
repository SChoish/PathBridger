#!/usr/bin/env python3
"""Emit plain Flow-BC + eval BoN4 YAMLs from per-env best diag_gaussian baselines.

Only ``dynamics`` subgoal keys change; critic/actor/IDM hyperparameters are preserved.

Usage (repo root):
  python scripts/generate_flow_plain_bon4_configs.py
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / 'config' / 'flow_plain_bon4_by_env'

RUNS_SEARCH = (
    REPO_ROOT / 'runs',
    REPO_ROOT / 'runs(diag_gaus)',
    REPO_ROOT / 'prev_runs2',
)

RES_SHORT = {'absolute': 'ra', 'displacement': 'rd'}
SG_SHORT = {'absolute': 'sa', 'displacement': 'sd'}

# (residual_target_mode, subgoal_target_mode)
TARGET_MODE_GRID: tuple[tuple[str, str], ...] = (
    ('displacement', 'displacement'),
    ('displacement', 'absolute'),
    ('absolute', 'displacement'),
    ('absolute', 'absolute'),
)

FLOW_DYNAMICS_PATCH: dict[str, Any] = {
    'subgoal_distribution': 'flow',
    'subgoal_flow_energy_weighted': False,
    'subgoal_flow_use_value_bonus': False,
    'subgoal_value_alpha': 0.0,
    'subgoal_value_gap_scale': 0.0,
    'subgoal_flow_steps': 8,
    'subgoal_flow_t_min': 1.0e-4,
    'subgoal_flow_noise_scale': 1.0,
    'subgoal_temperature': 1.0,
    'subgoal_num_samples': 4,
    'subgoal_eval_selection': 'best_of_n_value',
    'subgoal_eval_num_samples': 4,
    'subgoal_eval_include_zero_candidate': False,
    'subgoal_eval_seed': 0,
    'subgoal_stochastic_loss': 'mse',
}

TOP_LEVEL_KEYS = (
    'env_name',
    'batch_size',
    'horizon',
    'plan_candidates',
    'plan_noise_scale',
    'eval_freq',
    'eval_episodes_per_task',
    'final_eval_episodes_per_task',
    'save_every_n_epochs',
    'log_every_n_epochs',
)

ENV_SPECS: list[dict[str, Any]] = [
    {
        'env_stem': 'puzzle_3x3',
        'env_name': 'puzzle-3x3-play-v0',
        'ref_run': '20260517_154217',
        'train_epochs': 700,
        'grid_yaml': REPO_ROOT / 'config/grid_fbr_displacement_puzzle/puzzle_3x3_a0p0_gap20p0_k0p6.yaml',
    },
    {
        'env_stem': 'antmaze_medium',
        'env_name': 'antmaze-medium-navigate-v0',
        'ref_run': '20260521_051154',
        'train_epochs': 1000,
        'table_yaml': REPO_ROOT / 'config/antmaze_medium_navigate_table_phi_disp.yaml',
        'alpha': 0.0,
        'gap': 1.0,
        'kappa': 0.7,
        'discount': 0.99,
        'batch_size': 1024,
    },
    {
        'env_stem': 'antmaze_large',
        'env_name': 'antmaze-large-navigate-v0',
        'ref_run': '20260520_073104',
        'train_epochs': 1000,
        'table_yaml': REPO_ROOT / 'config/antmaze_large_navigate_table_phi_disp.yaml',
        'alpha': 0.0,
        'gap': 10.0,
        'kappa': 0.9,
        'discount': 0.99,
        'batch_size': 1024,
    },
    {
        'env_stem': 'antmaze_giant',
        'env_name': 'antmaze-giant-navigate-v0',
        'ref_run': '20260518_103317',
        'train_epochs': 1000,
        'table_yaml': REPO_ROOT / 'config/antmaze_medium_navigate_table_phi_disp.yaml',
        'alpha': 0.3,
        'gap': 5.0,
        'kappa': 0.6,
        'discount': 0.99,
        'batch_size': 1024,
    },
    {
        'env_stem': 'cube_single',
        'env_name': 'cube-single-play-v0',
        'ref_run': '20260523_172916',
        'train_epochs': 700,
        'table_yaml': REPO_ROOT / 'config/puzzle_3x3_play_table_full_disp_diag_gaussian.yaml',
        'alpha': 0.0,
        'gap': 1.0,
        'kappa': 0.9,
        'discount': 0.99,
        'batch_size': 1024,
        'goal_representation': 'full',
    },
    {
        'env_stem': 'cube_double',
        'env_name': 'cube-double-play-v0',
        'ref_run': '20260521_134814',
        'train_epochs': 700,
        'table_yaml': REPO_ROOT / 'config/puzzle_3x3_play_table_full_disp_diag_gaussian.yaml',
        'alpha': 0.0,
        'gap': 5.0,
        'kappa': 0.6,
        'discount': 0.99,
        'batch_size': 1024,
        'goal_representation': 'full',
    },
    {
        'env_stem': 'cube_triple',
        'env_name': 'cube-triple-play-v0',
        'ref_run': '20260518_022224',
        'train_epochs': 700,
        'table_yaml': REPO_ROOT / 'config/puzzle_3x3_play_table_full_disp_diag_gaussian.yaml',
        'alpha': 0.0,
        'gap': 5.0,
        'kappa': 0.8,
        'discount': 0.995,
        'batch_size': 4096,
        'goal_representation': 'full',
    },
]


def _load_yaml(path: Path) -> dict:
    with open(path, encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def _deep_copy_cfg(cfg: dict) -> dict:
    return copy.deepcopy(cfg)


def _run_dir(spec: dict[str, Any]) -> Path | None:
    env_name = str(spec['env_name'])
    ref_run = str(spec['ref_run'])
    for root in RUNS_SEARCH:
        p = root / f'{ref_run}_seed0_{env_name}'
        if p.is_dir():
            return p
    return None


def _load_from_flags(run_dir: Path) -> tuple[dict, str]:
    flags_path = run_dir / 'flags.json'
    with open(flags_path, encoding='utf-8') as f:
        data = json.load(f)
    flags = data.get('flags') or {}
    cfg: dict[str, Any] = {
        'dynamics': _deep_copy_cfg(data.get('dynamics') or {}),
        'critic_agent': _deep_copy_cfg(data.get('critic_agent') or {}),
        'actor': _deep_copy_cfg(data.get('actor') or {}),
    }
    for key in TOP_LEVEL_KEYS:
        if key in flags:
            cfg[key] = flags[key]
    if 'env_name' not in cfg:
        cfg['env_name'] = flags.get('env_name', '')
    if 'train_epochs' in flags:
        cfg['_baseline_train_epochs'] = int(flags['train_epochs'])
    return cfg, f'flags:{flags_path}'


def _load_from_config_used(run_dir: Path) -> tuple[dict, str]:
    cfg_path = run_dir / 'config_used.yaml'
    return _deep_copy_cfg(_load_yaml(cfg_path)), f'config_used:{cfg_path}'


def _synth_from_table(spec: dict[str, Any], table_yaml: Path) -> dict:
    cfg = _deep_copy_cfg(_load_yaml(table_yaml))
    env_name = str(spec['env_name'])
    alpha = float(spec['alpha'])
    gap = float(spec['gap'])
    kappa = float(spec['kappa'])
    discount = float(spec['discount'])
    batch_size = int(spec['batch_size'])
    goal_rep = str(spec.get('goal_representation', 'phi'))

    cfg['env_name'] = env_name
    cfg['batch_size'] = batch_size
    cfg['horizon'] = int(cfg.get('horizon', 25))
    cfg['plan_candidates'] = int(cfg.get('plan_candidates', 1))
    cfg['eval_freq'] = int(cfg.get('eval_freq', 100))
    cfg['eval_episodes_per_task'] = int(cfg.get('eval_episodes_per_task', 10))
    cfg['final_eval_episodes_per_task'] = int(cfg.get('final_eval_episodes_per_task', 50))
    cfg['log_every_n_epochs'] = int(cfg.get('log_every_n_epochs', 10))
    cfg['save_every_n_epochs'] = int(cfg.get('save_every_n_epochs', 100))

    dyn = dict(cfg.get('dynamics') or {})
    dyn['planner_type'] = 'forward_bridge_residual'
    dyn['forward_bridge_path_loss_horizon'] = int(dyn.get('forward_bridge_path_loss_horizon', 5))
    dyn['max_goal_steps_from_env'] = True
    dyn['clip_path_to_goal'] = True
    dyn['subgoal_distribution'] = 'diag_gaussian'
    dyn['subgoal_stochastic_loss'] = 'nll'
    dyn['subgoal_num_samples'] = 4
    dyn['subgoal_value_alpha'] = alpha
    dyn['subgoal_value_gap_scale'] = gap
    dyn['subgoal_goal_representation'] = goal_rep
    dyn['subgoal_target_mode'] = 'displacement'
    cfg['dynamics'] = dyn

    cri = dict(cfg.get('critic_agent') or {})
    cri['action_chunk_horizon'] = int(cri.get('action_chunk_horizon', 5))
    cri['kappa_b'] = kappa
    cri['kappa_d'] = kappa
    cri['discount'] = discount
    cri['max_goal_steps_from_env'] = True
    cri['clip_chunk_to_goal'] = True
    cri['goal_representation'] = goal_rep
    cfg['critic_agent'] = cri

    cfg['actor'] = dict(cfg.get('actor') or {'spi_beta': 1.0, 'spi_tau': 5.0})
    return cfg


def _load_baseline(spec: dict[str, Any]) -> tuple[dict, str]:
    run_dir = _run_dir(spec)
    if run_dir is not None:
        flags_path = run_dir / 'flags.json'
        if flags_path.is_file():
            return _load_from_flags(run_dir)
        cfg_used = run_dir / 'config_used.yaml'
        if cfg_used.is_file():
            return _load_from_config_used(run_dir)

    grid_yaml = spec.get('grid_yaml')
    if grid_yaml is not None and Path(grid_yaml).is_file():
        return _deep_copy_cfg(_load_yaml(Path(grid_yaml))), f'grid:{grid_yaml}'

    table_yaml = spec.get('table_yaml')
    if table_yaml is None or not Path(table_yaml).is_file():
        raise FileNotFoundError(
            f'No baseline source for {spec["env_stem"]}: '
            f'grid={grid_yaml!r}, run missing, table={table_yaml!r}'
        )
    return _synth_from_table(spec, Path(table_yaml)), f'synth:{table_yaml}'


def _train_epochs_for(spec: dict[str, Any], baseline: dict) -> int:
    if spec['env_name'].startswith('antmaze-'):
        old = baseline.get('_baseline_train_epochs')
        if old is not None:
            return int(old)
        return int(spec['train_epochs'])
    return int(spec['train_epochs'])


def _mode_tag(residual_mode: str, subgoal_mode: str) -> str:
    return f'{RES_SHORT[residual_mode]}_{SG_SHORT[subgoal_mode]}'


def _yaml_name(env_stem: str, mode_tag: str) -> str:
    return f'flow_plain_bon4_{env_stem}_{mode_tag}_from_diag_best.yaml'


def _apply_flow_plain_bon4(
    cfg: dict,
    spec: dict[str, Any],
    *,
    residual_mode: str,
    subgoal_mode: str,
) -> dict:
    out = _deep_copy_cfg(cfg)
    out.pop('_baseline_train_epochs', None)
    out['train_epochs'] = _train_epochs_for(spec, cfg)
    tag = _mode_tag(residual_mode, subgoal_mode)
    out['run_group'] = f'FlowPlainBoN4_{spec["env_stem"]}_{tag}'
    out['eval_freq'] = int(out.get('eval_freq', 100))
    out['eval_episodes_per_task'] = int(out.get('eval_episodes_per_task', 10))
    out['final_eval_episodes_per_task'] = int(out.get('final_eval_episodes_per_task', 50))
    out['save_every_n_epochs'] = int(out.get('save_every_n_epochs', 100))
    out['log_every_n_epochs'] = int(out.get('log_every_n_epochs', 10))

    dyn = dict(out.get('dynamics') or {})
    dyn.update(FLOW_DYNAMICS_PATCH)
    dyn['residual_target_mode'] = residual_mode
    dyn['subgoal_target_mode'] = subgoal_mode
    out['dynamics'] = dyn
    return out


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    stale = {p for p in OUT_DIR.glob('*.yaml')}

    for spec in ENV_SPECS:
        base, source = _load_baseline(spec)
        for residual_mode, subgoal_mode in TARGET_MODE_GRID:
            tag = _mode_tag(residual_mode, subgoal_mode)
            flow_cfg = _apply_flow_plain_bon4(
                base,
                spec,
                residual_mode=residual_mode,
                subgoal_mode=subgoal_mode,
            )
            out_path = OUT_DIR / _yaml_name(str(spec['env_stem']), tag)
            stale.discard(out_path)
            with open(out_path, 'w', encoding='utf-8') as f:
                f.write(
                    '# Auto-generated by scripts/generate_flow_plain_bon4_configs.py\n'
                    f'# env={spec["env_name"]} ref_run={spec["ref_run"]} '
                    f'train_epochs={flow_cfg["train_epochs"]}\n'
                    f'# residual_target_mode={residual_mode} '
                    f'subgoal_target_mode={subgoal_mode} tag={tag}\n'
                    f'# baseline_source={source}\n'
                    f'# plain Flow-BC + eval BoN4 (no value tilting in flow loss)\n'
                )
                yaml.safe_dump(flow_cfg, f, sort_keys=False, default_flow_style=False)
            written.append(out_path)
            print(
                f'wrote {out_path.relative_to(REPO_ROOT)}  '
                f'{tag} epochs={flow_cfg["train_epochs"]} '
                f'kappa={flow_cfg["critic_agent"]["kappa_b"]}'
            )

    for old_path in sorted(stale):
        old_path.unlink()
        print(f'removed stale {old_path.relative_to(REPO_ROOT)}')

    print(f'\nWrote {len(written)} configs under {OUT_DIR.relative_to(REPO_ROOT)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
