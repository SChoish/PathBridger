"""Print or execute the released state-based PBF/PBG experiment commands."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

ENVIRONMENTS = (
    'antmaze_medium',
    'antmaze_large',
    'cube_single',
    'cube_double',
    'cube_triple',
    'puzzle_3x3',
    'puzzle_4x4',
    'scene',
)


def parse_seeds(value: str) -> tuple[int, ...]:
    try:
        seeds = tuple(int(token.strip()) for token in value.split(','))
    except ValueError as error:
        raise argparse.ArgumentTypeError('seeds must be comma-separated integers') from error
    if not seeds or any(seed < 0 for seed in seeds) or len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError('seeds must be distinct non-negative integers')
    return seeds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--algorithm', choices=('pbf', 'pbg', 'all'), default='all')
    parser.add_argument('--env', choices=(*ENVIRONMENTS, 'all'), default='all')
    parser.add_argument('--seeds', type=parse_seeds, default=(0,))
    parser.add_argument('--phase', choices=('train', 'eval', 'all'), default='all')
    parser.add_argument('--run-root', type=Path, default=Path('exp/reproduction'))
    parser.add_argument('--dataset-dir', type=Path)
    parser.add_argument('--train-steps', type=int, default=1_000_000)
    parser.add_argument(
        '--checkpoint-step', type=int, help='Evaluation step; defaults to train-steps.'
    )
    parser.add_argument('--episodes', type=int, default=50, help='Episodes per task, tasks 1–5.')
    parser.add_argument(
        '--eval-protocol',
        choices=('historical-v1', 'deterministic-v1'),
        default='historical-v1',
        help='Historical compatibility default; select deterministic-v1 for paired comparisons.',
    )
    parser.add_argument(
        '--execute', action='store_true', help='Execute; otherwise only print commands.'
    )
    args = parser.parse_args()
    step = args.train_steps if args.checkpoint_step is None else args.checkpoint_step
    if min(args.train_steps, step, args.episodes) < 1:
        parser.error('train-steps, checkpoint-step, and episodes must be positive')
    if args.phase == 'all' and step > args.train_steps:
        parser.error('checkpoint-step cannot exceed train-steps when phase=all')
    if args.phase == 'all' and step != args.train_steps and step % 100_000:
        parser.error('intermediate checkpoint-step must be a multiple of the 100000 save interval')

    root = Path(__file__).resolve().parents[1]
    run_root = args.run_root.expanduser().resolve()
    algorithms = ('pbf', 'pbg') if args.algorithm == 'all' else (args.algorithm,)
    environments = ENVIRONMENTS if args.env == 'all' else (args.env,)
    jobs: list[tuple[str, list[str]]] = []
    # Finish seed 0 across methods/environments before starting seed 1.
    for seed in args.seeds:
        for algorithm in algorithms:
            for environment in environments:
                preset = root / 'configs' / f'{algorithm}_{environment}.yaml'
                run = run_root / f'seed_{seed}' / f'{algorithm}_{environment}'
                common = [f'--agent={preset}', f'--seed={seed}']
                if args.dataset_dir is not None:
                    common.append(f'--dataset_dir={args.dataset_dir.expanduser().resolve()}')
                if args.phase in ('train', 'all'):
                    if args.execute and (run / 'flags.json').exists():
                        parser.error(
                            f'run already exists: {run}; use a fresh run-root or resume main.py explicitly'
                        )
                    jobs.append(
                        (
                            'train',
                            [
                                sys.executable,
                                str(root / 'main.py'),
                                *common,
                                f'--run_dir={run}',
                                f'--train_steps={args.train_steps}',
                                '--eval_interval=0',
                                '--save_interval=100000',
                            ],
                        )
                    )
                if args.phase in ('eval', 'all'):
                    checkpoint = run / 'checkpoints' / f'params_{step}.pkl'
                    if args.execute and args.phase == 'eval' and not checkpoint.is_file():
                        parser.error(f'checkpoint not found: {checkpoint}')
                    result = run / 'evaluation' / args.eval_protocol / f'step_{step}'
                    if args.execute and result.with_suffix('.json').exists():
                        parser.error(f'evaluation result already exists: {result}.json')
                    jobs.append(
                        (
                            'eval',
                            [
                                sys.executable,
                                str(root / 'evaluate.py'),
                                *common,
                                f'--checkpoint_dir={run / "checkpoints"}',
                                f'--checkpoint_step={step}',
                                f'--episodes={args.episodes}',
                                '--task_ids=1,2,3,4,5',
                                f'--eval_protocol={args.eval_protocol}',
                                f'--output_path={result}.json',
                                f'--trace_path={result}_episodes.json',
                            ],
                        )
                    )
    for phase, command in jobs:
        print(f'# {phase}', flush=True)
        print(shlex.join(command), flush=True)
        if args.execute:
            subprocess.run(command, cwd=root, check=True)


if __name__ == '__main__':
    main()
