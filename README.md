# PathBridger

Official implementation of **PathBridger: Subgoal Bridges for Offline
Goal-Conditioned Reinforcement Learning** for state-based OGBench tasks.

**Paper:** [arXiv:2608.29061](https://arxiv.org/abs/2608.29061)

![PathBridger architecture](assets/architecture.jpg)

PathBridger predicts a subgoal, builds a state-space bridge to that subgoal,
and decodes the first bridge transitions with an inverse-dynamics model. The
repository provides two endpoint proposers through one training and inference
runtime:

- **PBF:** conditional rectified flow
- **PBG:** conditional diagonal Gaussian

## Installation

Python 3.10 or newer is required. First install the JAX build for your
accelerator using the [official JAX installation guide](https://docs.jax.dev/en/latest/installation.html),
then install the repository:

```bash
git clone https://github.com/SChoish/PathBridger.git
cd PathBridger
pip install -e .
```

Optional Weights & Biases logging is available with:

```bash
pip install -e ".[tracking]"
```

## Training

Train PBF on Cube Double:

```bash
python main.py \
  --agent=configs/pbf/cube_double.py \
  --dataset_dir=/path/to/ogbench \
  --seed=0
```

Train PBG with the manuscript objective:

```bash
python main.py \
  --agent=configs/manuscript_pbg/cube_double.py \
  --dataset_dir=/path/to/ogbench \
  --seed=0
```

`dataset_dir` may contain local `<env_name>.npz` and
`<env_name>-val.npz` OGBench files; both are loaded with OGBench's compact
dataset loader. If no matching NPZ files are present, OGBench performs its
normal resolution or download into that directory. Supported presets cover
AntMaze Medium/Large, Cube Single/Double/Triple, Puzzle 3x3/4x4, and Scene.

For new PBG experiments, use `configs/manuscript_pbg/`. Use
`configs/paper_pbg/` only when comparing with reconstructed reported runs.

The configuration families are separated intentionally:

- `configs/pbf/`: PBF presets for all supported state environments
- `configs/pbg/`: PBG compatibility presets for all supported environments
- `configs/manuscript_pbg/`: manuscript-objective PBG presets for Cube Double,
  Puzzle 3x3, and Scene
- `configs/paper_pbg/`: best-supported reconstruction of the reported PBG
  settings for those three environments

The general and reported-run presets use `reported-run-legacy-v1`; the
manuscript presets use `manuscript-v1`. Checkpoints record the protocol, and a
training resume across protocols is rejected.

Configuration is resolved from the Python preset, an optional strict YAML
overlay, and repeatable `--config_override=key=value` flags, in that order.
Unknown keys fail. Each run writes its resolved configuration and hash.

## Evaluation

Evaluate a checkpoint on the five OGBench tasks:

```bash
python evaluate.py \
  --agent=configs/pbf/cube_double.py \
  --checkpoint_dir=/path/to/checkpoints \
  --checkpoint_step=1000000 \
  --dataset_dir=/path/to/ogbench \
  --eval_protocol=historical-v1 \
  --episodes=50
```

`checkpoint_dir` accepts either a checkpoint directory together with
`checkpoint_step`, or an exact `params_<step>.pkl` file whose step is inferred.
The default task order is OGBench task IDs 1 through 5. The JSON result is
printed to stdout; pass `--output_path=/path/to/result.json` to save it.

`episodes` is the number of episodes per task. `historical-v1` preserves the
reported evaluator contract and is the compatibility default.
`deterministic-v1` additionally seeds each task and episode and must be chosen
explicitly. Result files record the effective protocol and evaluation config.

## Paper experiment plan

`manifests/state/paper.yaml` contains the Main and completed-ablation settings,
including environment presets, seeds, Best-(N,T), the 19-cell N/T grid,
800k/900k/1M checkpoint order, stochastic-prefix settings, path weighting,
episode budgets, and aggregation order.

Resolve the full plan without starting a job:

```bash
python scripts/run_state_manifest.py \
  --manifest=manifests/state/paper.yaml \
  --phase=all \
  --output-plan=paper-plan.json \
  --quiet
```

Execute one selected training job:

```bash
python scripts/run_state_manifest.py \
  --manifest=manifests/state/paper.yaml \
  --phase=train \
  --experiment=main_pbf \
  --environment=cube_double \
  --dataset-dir=/path/to/ogbench \
  --save-dir=/path/to/runs \
  --execute
```

The manifest runner is a dry-run by default. Execution requires `--execute`
and selectors that identify exactly one job.

## Inference

Export an optimizer-free inference bundle:

```bash
python inference.py \
  --agent=configs/pbf/cube_double.py \
  --dataset_dir=/path/to/ogbench \
  --checkpoint_dir=/path/to/checkpoints/params_1000000.pkl \
  --export_path=/path/to/pathbridger.pkl
```

As in evaluation, `checkpoint_dir` may be a directory plus
`--checkpoint_step`, or an exact checkpoint file.

Run batch inference from the bundle:

```bash
python inference.py \
  --inference_bundle=/path/to/pathbridger.pkl \
  --input_path=/path/to/state_goals.npz \
  --output_path=/path/to/action_chunks.npz
```

The input NPZ contains `observations` and `goals` arrays with shape `[B, D]`.
The output contains `action_chunks` with shape `[B, 5, A]`. Clip actions to the
environment bounds before stepping.

## Notes

- Pretrained checkpoints and datasets are not bundled.
- The exact Git SHA of the original reported runs was not recorded. The
  reported-run protocol is the closest source-backed reconstruction, not a
  claim of byte-identical historical training.
- The original PBG Scene configuration is unavailable; its paper preset is a
  best-supported reconstruction.
- `manuscript-v1` and `reported-run-legacy-v1` are distinct training
  objectives and are not claimed to produce identical optimizer states.

## Repository layout

```text
agents/             unified PBF/PBG agent
configs/            environment and protocol presets
envs/               OGBench state-data loading
manifests/state/    paper experiment plan
scripts/            manifest execution and aggregation
utils/              data, checkpoint, evaluation, and runtime utilities
main.py             training
evaluate.py         checkpoint evaluation
inference.py        checkpoint export and batch inference
```

## Citation

See [`CITATION.cff`](CITATION.cff).

## License

[MIT](LICENSE)
