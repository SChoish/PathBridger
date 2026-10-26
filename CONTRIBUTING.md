# Development

Install the development tools after installing the JAX build for your machine:

```bash
pip install -e ".[dev]"
ruff check .
ruff format --check .
python -m compileall -q agents envs utils main.py evaluate.py inference.py
python -m unittest discover -s tests -v
python -m build
```

Use `ruff check --fix .` to organize imports and `ruff format .` to format code.

## Source layout

| Path | Responsibility |
| --- | --- |
| `main.py` | Training, logging, checkpoint scheduling, and graceful interruption |
| `evaluate.py` | Checkpoint evaluation and result provenance |
| `inference.py` | Inference bundle export and batch action prediction |
| `agents/pathbridger.py` | Agent state, objectives, updates, bridge construction, and sampling |
| `agents/models.py` | Value, endpoint proposer, bridge residual, and inverse-dynamics networks |
| `agents/defaults.py` | Complete configuration schema and default values |
| `agents/extensions.py` | Stateless path-weighting, stochastic-prefix, state-only, and staged-inference methods |
| `agents/_constants.py` | Constants shared by the core and extensions |
| `agents/prefix_generators.py` | Optional stochastic prefix networks |
| `configs/` | Sixteen PBF/PBG environment presets |
| `envs/env_utils.py` | OGBench environment and dataset loading |
| `utils/config.py` | Strict configuration loading, overrides, and hashes |
| `utils/datasets.py` | Hindsight relabeling, trajectory sampling, and normalization |
| `utils/evaluation.py` | Rollouts and historical/deterministic evaluation protocols |
| `utils/flax_utils.py` | Training state and checkpoint serialization |
| `utils/deployment.py` | Optimizer-free inference bundles |
| `scripts/reproduce.sh` | Print or execute seed-ordered training/evaluation commands |
| `docs/reproduction.md` | Preset table, experiment recipe, protocols, and evidence limits |
| `tests/` | Preset hashes, CLI plans, training, extensions, and serialization checks |

The public `from agents import PathBridgerAgent, get_config` API is retained.
Network classes and `get_config` also remain importable from
`agents.pathbridger` for existing callers.

The extension mixin has no fields; the public agent remains the same serialized
PyTree type. Keep the core execution order visible: value, endpoint proposer,
bridge, IDM, then joint updates and policy sampling.

CI uses Python 3.12 and the CPU constraints in `requirements/ci-cpu.txt`.
These identify a tested integration environment, not the original paper GPU
stack. The same checks can run locally with
`pip install -c requirements/ci-cpu.txt -e ".[dev]"`.

## Refactoring research code

Keep parameter-tree names, optimizer state, PRNG split order, sampler state,
and checkpoint metadata compatible. Preserve the sixteen presets and their
resolved configuration hashes. Keep `historical-v1`, `deterministic-v1`, and
the two training-objective protocols distinct.

When moving a neural module, compare initialization and outputs with the
previous revision under the same seed, and check checkpoint and inference
bundle restoration. Formatting or moving definitions does not establish
historical score reproduction; that requires the recorded evaluation protocol
and checkpoints described in the README.

Datasets, run outputs, checkpoints, and inference bundles belong outside the
tracked source tree. Temporary build and development files are ignored.
