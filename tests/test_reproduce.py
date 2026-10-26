"""Exercise the actual shell/CLI plan without launching training."""

import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class ReproductionTests(unittest.TestCase):
    def plan(self, *arguments):
        result = subprocess.run(
            [sys.executable, str(ROOT / 'scripts/reproduce.py'), *arguments],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=True,
        )
        return [
            shlex.split(line) for line in result.stdout.splitlines() if not line.startswith('#')
        ]

    def test_complete_plan_is_seed_major_and_has_unique_outputs(self):
        commands = self.plan('--seeds=0,1')
        self.assertEqual(len(commands), 64)
        self.assertTrue(all('--seed=0' in command for command in commands[:32]))
        self.assertTrue(all('--seed=1' in command for command in commands[32:]))
        outputs = [
            argument
            for command in commands
            for argument in command
            if argument.startswith('--output_path=')
        ]
        self.assertEqual(len(outputs), len(set(outputs)))

    def test_selected_plan_preserves_protocol_and_quoted_paths(self):
        with tempfile.TemporaryDirectory(prefix='pb runs ') as directory:
            commands = self.plan(
                '--algorithm=pbg',
                '--env=cube_double',
                '--phase=eval',
                '--eval-protocol=deterministic-v1',
                f'--run-root={directory}',
            )
        self.assertEqual(len(commands), 1)
        self.assertIn('--task_ids=1,2,3,4,5', commands[0])
        self.assertIn('--eval_protocol=deterministic-v1', commands[0])
        self.assertTrue(any(directory in argument for argument in commands[0]))

    def test_shell_wrapper_and_invalid_seeds(self):
        result = subprocess.run(
            ['bash', str(ROOT / 'scripts/reproduce.sh'), '--help'], text=True, capture_output=True
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--execute', result.stdout)
        result = subprocess.run(
            [sys.executable, str(ROOT / 'scripts/reproduce.py'), '--seeds=0,0'],
            text=True,
            capture_output=True,
        )
        self.assertNotEqual(result.returncode, 0)

    def test_execute_refuses_existing_run_before_starting_jobs(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / 'seed_0/pbf_cube_single'
            run.mkdir(parents=True)
            (run / 'flags.json').write_text('{}')
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / 'scripts/reproduce.py'),
                    '--execute',
                    '--phase=train',
                    '--algorithm=pbf',
                    '--env=cube_single',
                    f'--run-root={directory}',
                ],
                text=True,
                capture_output=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('run already exists', result.stderr)
