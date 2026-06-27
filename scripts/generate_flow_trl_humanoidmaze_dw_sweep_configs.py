#!/usr/bin/env python3
"""Flow+TRL humanoidmaze medium/large sweep: gap × distance_weight (paper hyperparams)."""

from __future__ import annotations

import json
from pathlib import Path

from flow_trl_sweep_common import build_flow_trl_sweep_config, variant_suffix
from yaml_run_config import dump_run_config_yaml

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / 'config' / 'sweep_flow_trl_humanoidmaze_dw'
FINAL_EVAL_N_VALUES = [2, 8, 16, 32]

# humanoidmaze-medium / large; gamma=0.999; policy & TRL value (0,0,1,0) via TRL_ENV_SPECS.
ENVS = ('hmm', 'hml')
GAPS = (5.0, 10.0)
DISTANCE_WEIGHT_POWERS = (0.0, 0.1)


def _dwp_tag(dwp: float) -> str:
    if abs(dwp - round(dwp)) < 1e-9:
        return str(int(round(dwp)))
    return format(dwp, 'g').replace('.', 'p')


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    meta: list[dict] = []

    for env_prefix in ENVS:
        for gap in GAPS:
            for dwp in DISTANCE_WEIGHT_POWERS:
                base_suffix = variant_suffix(gap=gap)
                stem = f'{env_prefix}_{base_suffix}_dwp{_dwp_tag(dwp)}'
                variant = {'gap': gap, 'wmax': 5.0, 'n': 1, 'note': ''}
                cfg = build_flow_trl_sweep_config(
                    env_prefix=env_prefix,
                    variant_suffix_key=f'{base_suffix}_dwp{_dwp_tag(dwp)}',
                    variant=variant,
                    run_group_prefix='flow_trl_humanoidmaze_dw_',
                )
                cfg['critic_agent']['value_distance_weight_power'] = float(dwp)
                cfg['final_eval_subgoal_eval_num_samples'] = ','.join(
                    str(n) for n in FINAL_EVAL_N_VALUES
                )
                env_name = str(cfg['env_name'])
                cri = cfg['critic_agent']
                dyn = cfg['dynamics']
                header = (
                    '# Flow+TRL humanoidmaze sweep (gap × distance_weight)\n'
                    f'# env={env_name} gap={gap} wmax=5 train_N=1 gamma={cri["discount"]} '
                    f'distance_weight={dwp}\n'
                    '# policy (p_cur,p_geom,p_traj,p_rand)=(0,0,1,0); '
                    'TRL value (0,0,1,0)\n'
                    f'# actor: cur={dyn["actor_p_curgoal"]} traj={dyn["actor_p_trajgoal"]} '
                    f'rand={dyn["actor_p_randomgoal"]} geom={dyn["actor_geom_sample"]}\n'
                    f'# value: cur={cri["value_p_curgoal"]} traj={cri["value_p_trajgoal"]} '
                    f'rand={cri["value_p_randomgoal"]} geom={cri["value_geom_sample"]}\n'
                    f'# final_eval_subgoal_eval_num_samples={FINAL_EVAL_N_VALUES}; '
                    'episodes_per_task=25; eval_max_chunks=400 (env cap 2000 / h_a=5)\n'
                )
                out_path = OUT / f'{stem}.yaml'
                dump_run_config_yaml(out_path, cfg, header=header)
                print(out_path)
                meta.append(
                    {
                        'config': stem,
                        'env_prefix': env_prefix,
                        'env': env_name,
                        'gap': gap,
                        'distance_weight': dwp,
                        'gamma': float(cri['discount']),
                    }
                )

    manifest = OUT / 'manifest.json'
    manifest.write_text(
        json.dumps(
            {
                'final_eval_n_values': FINAL_EVAL_N_VALUES,
                'gaps': list(GAPS),
                'distance_weights': list(DISTANCE_WEIGHT_POWERS),
                'gamma': 0.999,
                'policy_goal_ratio': [0, 0, 1, 0],
                'value_goal_ratio_trl': [0, 0, 1, 0],
                'runs': meta,
            },
            indent=2,
        )
        + '\n',
        encoding='utf-8',
    )
    print(f'Manifest: {manifest} ({len(meta)} configs)')


if __name__ == '__main__':
    main()
