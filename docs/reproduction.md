# Reproducing the released state-based experiments

## Scope and provenance

The release supports PBF (flow endpoints) and PBG (Gaussian endpoints) on eight
state-based OGBench environments. The sixteen YAML presets retain exactly the
resolved settings present at commit `7da9fc0b92b815c8d466134c2270dbd7c58aa155`.
Their hashes are recorded in `tests/preset_hashes.json`.

These are settings reconstructed from surviving experiment records. The
original result-producing Git SHA was not recorded, and the PBG Scene preset
was reconstructed from its surviving run record. They are not a claim of
byte-identical historical training or of newly reproduced paper scores.

Three kinds of evidence have different meanings:

| Evidence | What it establishes |
| --- | --- |
| Preset and configuration hash | Which settings were requested |
| CPU update/inference/checkpoint checks | Whether the implementation and serialization work under the tested stack |
| Recorded environment evaluations | Success rates for the identified dataset, checkpoint, tasks, and evaluation protocol |

Pretrained checkpoints, original run logs, and datasets are not distributed.
Use the paper for its reported scores. The commands below provide the execution
recipe; running them produces new results with their own provenance.

## Installation

Install the appropriate JAX build for your accelerator, then run `pip install -e .`.
The top-level README gives the normal training, evaluation, and inference commands.
Weights & Biases is optional (`pip install -e ".[tracking]"`); CSV/JSON output
works without a W&B account.

For the CPU verification environment used during this cleanup, use Python 3.12:

```bash
pip install -c requirements/ci-cpu.txt -e ".[dev]"
python -m unittest discover -s tests -v
```

`requirements/ci-cpu.txt` records a tested CPU smoke-test stack. It is not the
original paper's GPU software environment, and it is not a CUDA installation recipe.

## Generate and run commands

The wrapper prints commands by default. It resolves the configuration, run,
checkpoint, and result paths so the printed commands can also be run directly.

```bash
# One method/environment, training followed by evaluation. Prints commands only.
bash scripts/reproduce.sh \
  --algorithm pbf --env cube_double --seeds 0 --phase all \
  --dataset-dir /path/to/ogbench --run-root exp/reproduction \
  --eval-protocol historical-v1

# Run the same selected experiment.
bash scripts/reproduce.sh \
  --algorithm pbf --env cube_double --seeds 0 --phase all \
  --dataset-dir /path/to/ogbench --run-root exp/reproduction \
  --eval-protocol historical-v1 --execute

# Print the complete PBF/PBG × eight-environment plan for four chosen seeds.
bash scripts/reproduce.sh --algorithm all --env all --seeds 0,1,2,3
```

The seeds in the last example are a user-selected run plan, not an assertion
that the original paper used those exact seed identifiers. All methods and
environments for the first requested seed precede the next seed. Execution is
sequential; the wrapper does not allocate GPUs or create parallel workers.
It uses the current Python interpreter; set `PYTHON_BIN` for the shell wrapper
if needed. `--dataset-dir` can be omitted to use OGBench's normal resolution.

Defaults are 1,000,000 updates, checkpoints every 100,000 updates, and a final
checkpoint. In-process evaluation is disabled in generated training commands;
the standalone evaluator runs afterward. Existing run metadata and evaluation
results are rejected on execution to prevent accidental replacement. For
training resume, use the existing `main.py --restore_path=...` interface.

Evaluate an existing generated run:

```bash
bash scripts/reproduce.sh \
  --algorithm pbf --env cube_double --seeds 0 --phase eval \
  --run-root exp/reproduction --checkpoint-step 1000000 \
  --eval-protocol deterministic-v1 --execute
```

For shorter integration runs, set `--train-steps` explicitly. A short run is a
smoke test and must not be compared as a 1M-update paper result.

## Released settings

The table is transcribed from the YAML files. `K` is the endpoint/bridge horizon;
the action chunk is always five steps. `actor_p` uses the order
`current/geometric/trajectory/random`. The value-distance power and endpoint
weight scale are distinct settings.

| Preset | K | Discount | actor_p | Endpoint scale | Value-distance power | N | Temperature |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| `pbf_antmaze_large` | 25 | 0.995 | 0/0/1/0 | 10 | 0 | 32 | 0.5 |
| `pbf_antmaze_medium` | 25 | 0.99 | 0/0/1/0 | 10 | 0 | 8 | 0.25 |
| `pbf_cube_double` | 40 | 0.99 | 0/0/1/0 | 10 | 1 | 8 | 0.25 |
| `pbf_cube_single` | 40 | 0.99 | 0/0/1/0 | 5 | 0.7 | 1 | 0 |
| `pbf_cube_triple` | 40 | 0.995 | 0/0/1/0 | 10 | 1 | 1 | 0 |
| `pbf_puzzle_3x3` | 25 | 0.99 | 0/0/1/0 | 10 | 0.5 | 32 | 1 |
| `pbf_puzzle_4x4` | 25 | 0.99 | 0/0/1/0 | 10 | 2 | 16 | 1 |
| `pbf_scene` | 25 | 0.99 | 0/0/1/0 | 5 | 1 | 8 | 0.5 |
| `pbg_antmaze_large` | 25 | 0.995 | 0/0/1/0 | 10 | 0 | 1 | 0 |
| `pbg_antmaze_medium` | 25 | 0.99 | 0/0/1/0 | 10 | 0 | 1 | 0 |
| `pbg_cube_double` | 25 | 0.99 | 0/0/1/0 | 10 | 1 | 1 | 0 |
| `pbg_cube_single` | 25 | 0.99 | 0/0/1/0 | 10 | 0.7 | 1 | 0 |
| `pbg_cube_triple` | 25 | 0.995 | 0/0/1/0 | 10 | 1 | 1 | 0 |
| `pbg_puzzle_3x3` | 40 | 0.99 | 0/0.5/0/0.5 | 10 | 0.5 | 2 | 0.25 |
| `pbg_puzzle_4x4` | 40 | 0.995 | 0/0.5/0/0.5 | 10 | 2 | 32 | 0.5 |
| `pbg_scene` | 40 | 0.99 | 0/0/1/0 | 5 | 1 | 16 | 0.5 |

All presets use `training_objective_protocol=reported-run-legacy-v1`,
`critic_p=(0,1,0,0)`, `prefix_model=deterministic`, `path_weight_beta=0`, and
`offline_action_free=false`. The optimizer rate is `3e-4`, batch size is 1024,
networks use three 512-unit layers with LayerNorm, and the endpoint flow uses
eight Euler steps. The checked-in YAML files remain explicit and complete;
the table is an overview, not a second source of configuration defaults.

Additional features and their defaults are documented in [extensions.md](extensions.md).

## Evaluation and reporting

The wrapper explicitly selects tasks 1–5 and 50 episodes per task (250 total)
unless `--episodes` is changed. Success means any `info['success']` within the
episode. Overall success is the unweighted mean of task success rates.
Different evaluation contracts must remain separately labeled:

| Protocol | Contract and use |
| --- | --- |
| `historical-v1` | Compatibility default. Environment resets receive task options without explicit episode/reset/action-space seeds. |
| `deterministic-v1` | Explicit task/episode reset and action-space seeding, including temporary NumPy reset seeding. Use for paired comparisons. |

The wrapper preserves `historical-v1` as its default and always emits the
chosen protocol in the command. Do not pool results from both protocols as
if their reset panels were identical. Training-objective protocol and
evaluation protocol are separate concepts; selecting deterministic evaluation
does not switch the training objective to `manuscript-v1`.

Each run is stored under `RUN_ROOT/seed_S/ALGORITHM_ENV/`. A generated evaluation
writes `evaluation/PROTOCOL/step_STEP.json` and a separate `_episodes.json`
trace. The evaluator records checkpoint file hashes, dataset identity,
configuration hashes, checkpoint step, task counts, sampling settings, and the
effective evaluation protocol. Keep those records with any published table.

For multiple training seeds, first compute each seed's task and overall
success rates; then report the mean and standard deviation across training
seeds. Episodes within one trained policy are not independent training runs.
Always state the checkpoint selection rule (for example, final 1M checkpoint)
and distinguish final scores from best-checkpoint scores.

## Implementation checks

The automated suite checks the released preset hashes, public imports,
one-update training and target updates, exact checkpoint resume including
sampler/optimizer state, inference bundle restoration, and the optional
prefix/state-only/path-weighting paths. Reproduction CLI checks inspect the
actual generated commands and overwrite guards without launching training.
These checks do not measure benchmark success rates.

The cleanup also compares full-width CPU runs against the preceding release
under identical inputs and seeds. Numerical comparisons are scoped to that
runtime; they do not establish equivalence across all JAX/CUDA versions.

## Release design references

The structure follows the research-code conventions inspected in the official
[OGBench implementations](https://github.com/seohongpark/ogbench/tree/a977253ce5b88434f4e4f531c50047cc95993a51/impls),
[FQL](https://github.com/seohongpark/fql/tree/e8cd16eb490332924dfa2492097219f181765933),
[IQL](https://github.com/ikostrikov/implicit_q_learning/tree/09d700248117881a75cb21f0adb95c6c8a694cb2),
[TD3+BC](https://github.com/sfujim/TD3_BC/tree/8791ad7d7656cb8396f1b3ac8c37f170b2a2dd5f),
and [CORL](https://github.com/corl-team/CORL/tree/eeeeef2671235fcce3a3eb255576a207d78840e6).
The choices concern source organization, explicit experiment commands, and
result provenance; they do not imply independent validation by those projects.
