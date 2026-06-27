#!/usr/bin/env python3
"""Flow+TRL antmaze (humanoid) medium/large sweep: gap × distance_weight."""

from __future__ import annotations

import json
from pathlib import Path

from flow_trl_sweep_common import build_flow_trl_sweep_config, variant_suffix
from yaml_run_config import dump_run_config_yaml

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / 'config' / 'sweep_flow_trl_antmaze_dw'
FINAL_EVAL_N_VALUES = [2, 8, 16, 32]

ENVS = ('amm', 'aml')
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
                    run_group_prefix='flow_trl_antmaze_dw_',
                )
                cfg['critic_agent']['value_distance_weight_power'] = float(dwp)
                cfg['final_eval_subgoal_eval_num_samples'] = ','.join(
                    str(n) for n in FINAL_EVAL_N_VALUES
                )
                env_name = str(cfg['env_name'])
                cri = cfg['critic_agent']
                header = (
                    '# Flow+TRL antmaze humanoid sweep (gap × distance_weight)\n'
                    f'# env={env_name} gap={gap} wmax=5 train_N=1 '
                    f'gamma={cri["discount"]} distance_weight={dwp}\n'
                    f'# final_eval_subgoal_eval_num_samples={FINAL_EVAL_N_VALUES}; '
                    'episodes_per_task=25\n'
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
                    }
                )

    manifest = OUT / 'manifest.json'
    manifest.write_text(
        json.dumps(
            {
                'final_eval_n_values': FINAL_EVAL_N_VALUES,
                'gaps': list(GAPS),
                'distance_weights': list(DISTANCE_WEIGHT_POWERS),
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
