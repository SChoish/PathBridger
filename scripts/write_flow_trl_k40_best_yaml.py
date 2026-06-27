#!/usr/bin/env python3
"""Emit K=40 Flow+TRL configs using per-env best params from feval CSV."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts'))

from utils.trl_critic_config import TRL_ENV_SPECS, trl_critic_agent_config
from yaml_run_config import build_trl_run_config, dump_run_config_yaml

OUT = REPO / 'config' / 'sweep_flow_trl_k40_best'
HORIZON = 40
TRAIN_N = 1
WMAX = 5.0

# IDM-best params from docs/flow_trl_feval_results_7ch.csv (2026-06-22).
BEST_SPECS: list[dict[str, Any]] = [
    {'env_prefix': 'p3', 'gap': 1.0, 'eval_n': 32, 'source_idm': 0.672, 'source_actor': 0.392},
    {'env_prefix': 'p4', 'gap': 5.0, 'eval_n': 16, 'source_idm': 0.800, 'source_actor': 0.688},
    {'env_prefix': 'cs', 'gap': 10.0, 'eval_n': 2, 'source_idm': 0.768, 'source_actor': 0.656},
    {'env_prefix': 'cd', 'gap': 10.0, 'eval_n': 8, 'source_idm': 0.680, 'source_actor': 0.648},
    {'env_prefix': 'ct', 'gap': 10.0, 'eval_n': 8, 'source_idm': 0.328, 'source_actor': 0.208},
    {'env_prefix': 'amm', 'gap': 3.0, 'eval_n': 8, 'source_idm': 0.960, 'source_actor': 0.928},
    {'env_prefix': 'aml', 'gap': 10.0, 'eval_n': 16, 'source_idm': 0.768, 'source_actor': 0.728},
    {'env_prefix': 'amg', 'gap': 3.0, 'eval_n': 8, 'source_idm': 0.232, 'source_actor': 0.216},
]

FLOW_DYNAMICS_PATCH: dict[str, Any] = {
    'subgoal_distribution': 'flow',
    'subgoal_stochastic_loss': 'mse',
    'subgoal_num_samples': TRAIN_N,
    'subgoal_value_alpha': 0.0,
    'subgoal_flow_steps': 8,
    'subgoal_flow_t_min': 1.0e-4,
    'subgoal_flow_noise_scale': 1.0,
    'subgoal_temperature': 1.0,
    'subgoal_eval_selection': 'best_of_n_value',
    'subgoal_eval_include_zero_candidate': False,
    'subgoal_eval_seed': 0,
}


def _tag_float(x: float) -> str:
    return str(int(x)) if abs(x - round(x)) < 1e-9 else str(x).replace('.', 'p')


def build_config(spec: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    env_prefix = str(spec['env_prefix'])
    env_spec = TRL_ENV_SPECS[env_prefix]
    critic_cfg = trl_critic_agent_config(env_prefix)
    critic_cfg['full_chunk_horizon'] = HORIZON
    gap = float(spec['gap'])
    eval_n = int(spec['eval_n'])
    dynamics_overrides = deepcopy(FLOW_DYNAMICS_PATCH)
    dynamics_overrides.update(
        {
            'dynamics_N': HORIZON,
            'subgoal_steps': HORIZON,
            'subgoal_value_gap_scale': gap,
            'subgoal_value_weight_max': WMAX,
            'subgoal_eval_num_samples': TRAIN_N,
        }
    )
    cfg = build_trl_run_config(
        env_name=str(env_spec['env_name']),
        run_group=f'flow_trl_k40_best_{env_prefix}_g{_tag_float(gap)}_en{eval_n}',
        gap_scale=gap,
        weight_max=WMAX,
        discount=float(critic_cfg['discount']),
        value_distance_weight_power=float(critic_cfg['value_distance_weight_power']),
        batch_size=int(env_spec['batch_size']),
        train_epochs=600,
        horizon=HORIZON,
        dynamics_overrides=dynamics_overrides,
        critic_overrides=critic_cfg,
        value_goal_sampling={},
        actor_goal_sampling=deepcopy(env_spec['actor_goal_sampling']),
    )
    cfg['final_eval_subgoal_eval_num_samples'] = str(eval_n)
    fname = f'{env_prefix}_k40_g{_tag_float(gap)}_w5_n1_en{eval_n}.yaml'
    return fname, cfg


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest: list[str] = []
    written: set[Path] = set()
    for spec in BEST_SPECS:
        fname, cfg = build_config(spec)
        out_path = OUT / fname
        header = (
            '# K=40 Flow+TRL best-param follow-up\n'
            f'# source IDM={spec["source_idm"]} ACTOR={spec["source_actor"]}; '
            f'horizon={HORIZON}; final_eval_N={spec["eval_n"]}\n'
        )
        dump_run_config_yaml(out_path, cfg, header=header)
        written.add(out_path)
        manifest.append(str(out_path.relative_to(REPO)))
        print(out_path)

    for stale in OUT.glob('*.yaml'):
        if stale not in written:
            stale.unlink()

    manifest_path = OUT / '_manifest.txt'
    with open(manifest_path, 'w', encoding='utf-8') as f:
        f.write(f'# {len(manifest)} K=40 best-param Flow+TRL configs\n')
        f.write('# Source params: docs/flow_trl_best_params.md\n')
        for item in manifest:
            f.write(item + '\n')
    print(f'Manifest: {manifest_path}')


if __name__ == '__main__':
    main()
