# Optional extensions and compatibility

The sixteen released state-based presets use a deterministic bridge followed
by inverse-dynamics decoding. Their core implementation is in
`agents/pathbridger.py`; neural modules are in `agents/models.py`.

`agents/extensions.py` contains optional methods inherited by the same public
`PathBridgerAgent` class. The mixin adds no state. Agent state remains `rng`,
`network`, `state_scale`, and `config`, and parameter subtree names and the
checkpoint/bundle formats remain unchanged.

| Feature | Relevant settings | Released preset behavior |
| --- | --- | --- |
| Temporal path weighting | `path_weight_beta`, `path_weight_min`, warmup/ramp, distance cap | `beta=0`; diagnostic geometry may still be computed during full-metric updates |
| Stochastic prefixes | `prefix_model`, loss weight, rank, sigma floor, flow steps | `prefix_model=deterministic`; no learned stochastic prefix module |
| Prefix selection | `eval_prefix_selection`, sample count, temperature, deterministic inclusion | One deterministic prefix |
| State-only planning | `offline_action_free`, `sample_state_prefix` | `offline_action_free=false`; IDM is trained and actions are decoded |
| Historical staged inference | Programmatic `legacy_inference_staging` compatibility flag | Compiled kernel; the explicit legacy shim remains callable |

`prefix_scale_floor` also controls the shared dataset-derived state scale in
`utils/runtime.py`. It remains active infrastructure even when stochastic
prefix models are disabled.

## Existing callers

Public imports remain:

```python
from agents import PathBridgerAgent, get_config
```

Network classes and `get_config` remain importable from `agents.pathbridger`.
Existing prefix/planning/staged-inference methods remain available on the
public agent. Configuration keys and their defaults are retained so resolved
configuration hashes and old checkpoint metadata keep their meaning.

## Explicit protocol versions

`reported-run-legacy-v1` and `manuscript-v1` describe different training
semantics. Both remain in the core implementation, and training resume rejects
a mismatch. Inference-only restoration follows its existing explicit contract.

`historical-v1` and `deterministic-v1` describe evaluation resets and sampling
panels. They are independent of the training-objective version. Neither type
of protocol is renamed or silently selected during this cleanup.

The staged inference shim is an implementation compatibility path, not a third
evaluation protocol. It remains distinct from both protocol namespaces.
