#!/usr/bin/env python3
"""Generate Flow+TRL gap-tune configs from best-params manifest.

Rule: best gap=1 -> try 0.5; best gap=10 -> try 20; others skipped.
"""

from __future__ import annotations

import json
from pathlib import Path

from flow_trl_sweep_common import build_flow_trl_sweep_config, variant_suffix
from yaml_run_config import dump_run_config_yaml

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / 'config' / 'sweep_flow_trl_gap_tune'
MANIFEST = REPO / 'checkpoints' / 'flow_trl_best_epoch600' / 'manifest.json'
FINAL_EVAL_N_VALUES = [2, 8, 16, 32]

ENV_PREFIX = {
    'puzzle-3x3-play-v0': 'p3',
    'puzzle-4x4-play-v0': 'p4',
    'cube-single-play-v0': 'cs',
    'cube-double-play-v0': 'cd',
    'cube-triple-play-v0': 'ct',
    'antmaze-medium-navigate-v0': 'amm',
    'antmaze-large-navigate-v0': 'aml',
    'antmaze-giant-navigate-v0': 'amg',
}


def _target_gap(best_gap: float) -> float | None:
    if abs(best_gap - 1.0) < 1e-9:
        return 0.5
    if abs(best_gap - 10.0) < 1e-9:
        return 20.0
    return None


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    OUT.mkdir(parents=True, exist_ok=True)
    meta: list[dict] = []

    for run in manifest['runs']:
        best_gap = float(run['gap'])
        new_gap = _target_gap(best_gap)
        if new_gap is None:
            continue

        env_name = str(run['env'])
        env_prefix = ENV_PREFIX[env_name]
        suffix = variant_suffix(gap=new_gap)
        variant = {'gap': new_gap, 'wmax': 5.0, 'n': 1, 'note': ''}
        cfg = build_flow_trl_sweep_config(
            env_prefix=env_prefix,
            variant_suffix_key=suffix,
            variant=variant,
            run_group_prefix='flow_trl_gap_tune_',
        )
        cfg['final_eval_subgoal_eval_num_samples'] = ','.join(
            str(n) for n in FINAL_EVAL_N_VALUES
        )
        cri = cfg['critic_agent']
        header = (
            '# Flow+TRL gap tune from best params\n'
            f'# env={env_name} best_gap={best_gap} tune_gap={new_gap} '
            f'wmax=5 train_N=1 gamma={cri["discount"]} '
            f'vdist_pow={cri["value_distance_weight_power"]}\n'
            f'# baseline config={run["config"]} baseline_idm={run["idm"]:.3f} '
            f'baseline_actor={run["actor"]:.3f} best_eval_n={run["eval_n"]}\n'
            f'# final_eval_subgoal_eval_num_samples={FINAL_EVAL_N_VALUES}; episodes_per_task=25\n'
        )
        out_path = OUT / f'{env_prefix}_{suffix}.yaml'
        dump_run_config_yaml(out_path, cfg, header=header)
        print(out_path)
        meta.append(
            {
                'env': env_name,
                'env_prefix': env_prefix,
                'config': f'{env_prefix}_{suffix}',
                'best_gap': best_gap,
                'tune_gap': new_gap,
                'best_eval_n': int(run['eval_n']),
                'baseline_config': run['config'],
                'baseline_idm': run['idm'],
                'baseline_actor': run['actor'],
            }
        )

    meta_path = OUT / 'manifest.json'
    meta_path.write_text(
        json.dumps(
            {
                'final_eval_n_values': FINAL_EVAL_N_VALUES,
                'runs': meta,
            },
            indent=2,
        )
        + '\n',
        encoding='utf-8',
    )
    print(f'Manifest: {meta_path} ({len(meta)} configs)')


if __name__ == '__main__':
    main()
