"""Shared builders for Flow subgoal + TRL critic gap / wmax / N sweeps."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from utils.trl_critic_config import TRL_ENV_SPECS, trl_critic_agent_config
from yaml_run_config import build_trl_run_config

GAP_VALUES = [1.0, 3.0, 5.0, 10.0]
TRAIN_WMAX = 5.0
TRAIN_N = 1
FINAL_EVAL_N_VALUES = [1, 2, 4, 8, 16]

# Run order: puzzle -> cube -> antmaze.
ENV_ORDER = ['p3', 'p4', 'cs', 'cd', 'ct', 'amm', 'aml', 'amg']

FLOW_DYNAMICS_BASE: dict[str, Any] = {
    'subgoal_distribution': 'flow',
    'subgoal_stochastic_loss': 'mse',
    'subgoal_value_alpha': 0.0,
    'subgoal_flow_steps': 8,
    'subgoal_flow_t_min': 1.0e-4,
    'subgoal_flow_noise_scale': 1.0,
    'subgoal_temperature': 1.0,
    'subgoal_eval_selection': 'best_of_n_value',
    'subgoal_eval_include_zero_candidate': False,
    'subgoal_eval_seed': 0,
}


def _format_float_tag(x: float) -> str:
    if abs(x - round(x)) < 1e-9:
        return str(int(round(x)))
    return format(x, 'g').replace('.', 'p')


def variant_suffix(*, gap: float, wmax: float = TRAIN_WMAX, n: int = TRAIN_N) -> str:
    return f'g{_format_float_tag(gap)}_w{_format_float_tag(wmax)}_n{n}'


def parse_variant_suffix(suffix: str) -> tuple[float, float, int]:
    gap_part, wmax_part, n_part = suffix.split('_')
    gap = float(gap_part[1:].replace('p', '.'))
    wmax = float(wmax_part[1:].replace('p', '.'))
    n = int(n_part[1:])
    return gap, wmax, n


def build_flow_trl_variants(*, include_notes: bool = True) -> dict[str, dict[str, Any]]:
    variants: dict[str, dict[str, Any]] = {}
    for gap in GAP_VALUES:
        suffix = variant_suffix(gap=gap)
        note = ''
        if include_notes and gap == 10.0:
            note = 'Config B training anchor (gap10/wmax5/N1)'
        variants[suffix] = {'gap': gap, 'wmax': TRAIN_WMAX, 'n': TRAIN_N, 'note': note}
    return variants


FLOW_TRL_VARIANTS = build_flow_trl_variants()


def build_flow_trl_sweep_config(
    *,
    env_prefix: str,
    variant_suffix_key: str,
    variant: dict[str, Any],
    run_group_prefix: str = 'flow_trl_',
) -> dict[str, Any]:
    env_spec = TRL_ENV_SPECS[env_prefix]
    n = int(variant['n'])
    dynamics_overrides = deepcopy(FLOW_DYNAMICS_BASE)
    dynamics_overrides.update(
        {
            'subgoal_value_gap_scale': float(variant['gap']),
            'subgoal_value_weight_max': float(variant['wmax']),
            'subgoal_num_samples': n,
            'subgoal_eval_num_samples': n,
        }
    )
    critic_cfg = trl_critic_agent_config(env_prefix)
    cfg = build_trl_run_config(
        env_name=str(env_spec['env_name']),
        run_group=f'{run_group_prefix}{env_prefix}_{variant_suffix_key}',
        gap_scale=float(variant['gap']),
        weight_max=float(variant['wmax']),
        discount=float(critic_cfg['discount']),
        value_distance_weight_power=float(critic_cfg['value_distance_weight_power']),
        batch_size=int(env_spec['batch_size']),
        train_epochs=600,
        dynamics_overrides=dynamics_overrides,
        critic_overrides=critic_cfg,
        value_goal_sampling={},
        actor_goal_sampling=deepcopy(env_spec['actor_goal_sampling']),
    )
    if run_group_prefix.startswith('flow_trl_feval'):
        cfg['final_eval_subgoal_eval_num_samples'] = final_eval_n_values_csv()
    return cfg


def config_sort_key(path_name: str) -> tuple[int, float, float, int]:
    stem = path_name.removesuffix('.yaml')
    env_prefix, variant_key = stem.split('_', 1) if '_' in stem else (stem, '')
    env_idx = ENV_ORDER.index(env_prefix) if env_prefix in ENV_ORDER else 99
    gap, wmax, n = parse_variant_suffix(variant_key) if variant_key else (0.0, 0.0, 0)
    return env_idx, gap, wmax, n


def resolve_run_dir(
    *,
    config_path: str,
    runs_root: str,
    seed: int,
    final_epoch: int = 600,
    require_checkpoint: bool = True,
) -> str:
    import glob
    import json
    import os

    import yaml

    with open(config_path, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    run_group = str(cfg.get('run_group', ''))
    env_name = str(cfg.get('env_name', ''))
    if not run_group or not env_name:
        return ''

    candidates: list[str] = []
    pattern = os.path.join(runs_root, f'*_seed{seed}_{env_name}')
    for run_dir in glob.glob(pattern):
        flags_path = os.path.join(run_dir, 'flags.json')
        if not os.path.isfile(flags_path):
            continue
        with open(flags_path, encoding='utf-8') as f:
            flags = json.load(f).get('flags', {})
        if flags.get('run_group') != run_group:
            continue
        if require_checkpoint and final_epoch > 0:
            ckpt = os.path.join(
                run_dir, 'checkpoints', 'dynamics', f'params_{final_epoch}.pkl'
            )
            if not os.path.isfile(ckpt):
                continue
        candidates.append(run_dir)
    if not candidates:
        return ''
    return max(candidates, key=os.path.getmtime)


def eval_results_complete(
    run_dir: str,
    *,
    epoch: int = 600,
    n_values: list[int] | None = None,
) -> bool:
    import os

    if not run_dir:
        return False
    n_values = n_values or FINAL_EVAL_N_VALUES
    for n in n_values:
        path = os.path.join(run_dir, 'eval_results', f'epoch{int(epoch)}_n{int(n)}.json')
        if not os.path.isfile(path):
            return False
    return True


def final_eval_n_values_csv() -> str:
    return ','.join(str(n) for n in FINAL_EVAL_N_VALUES)
