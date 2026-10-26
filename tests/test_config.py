"""Protect the published presets and public imports during refactors."""

import json
import unittest
from pathlib import Path

import agents.models as models
import agents.pathbridger as legacy_imports
from agents import PathBridgerAgent, get_config
from utils.config import config_sha256, load_config

ROOT = Path(__file__).resolve().parents[1]


class PresetTests(unittest.TestCase):
    def test_resolved_presets_match_release_baseline(self):
        expected = json.loads((Path(__file__).parent / 'preset_hashes.json').read_text())
        presets = sorted((ROOT / 'configs').glob('*.yaml'))
        self.assertEqual({path.name for path in presets}, set(expected))
        for path in presets:
            with self.subTest(preset=path.name):
                self.assertEqual(config_sha256(load_config(path)), expected[path.name])

    def test_public_imports_remain_compatible(self):
        self.assertIs(PathBridgerAgent, legacy_imports.PathBridgerAgent)
        self.assertIs(get_config, legacy_imports.get_config)
        for name in models.__all__:
            self.assertIs(getattr(models, name), getattr(legacy_imports, name))

    def test_overrides_are_strict_and_preserve_goal_mix_values(self):
        preset = ROOT / 'configs/pbf_cube_double.yaml'
        with self.assertRaises(KeyError):
            load_config(preset, cli_overrides=['eval_num_candiates=8'])
        config = load_config(preset, cli_overrides=['actor_p=[0, 0.5, 0, 0.5]'])
        self.assertEqual(list(config.actor_p), [0, 0.5, 0, 0.5])
