"""Minimal Flax training-state and checkpoint utilities."""

from __future__ import annotations

import functools
import hashlib
import os
import pickle
import random
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import flax
import flax.linen as nn
import jax
import numpy as np
import optax

from utils.training_protocol import (
    REPORTED_RUN_LEGACY_TRAINING_PROTOCOL,
    normalize_training_protocol,
    training_protocol_from_config,
)

nonpytree_field = functools.partial(flax.struct.field, pytree_node=False)


class ModuleDict(nn.Module):
    """Expose a dictionary of Flax modules through one shared parameter tree."""

    modules: dict[str, nn.Module]

    @nn.compact
    def __call__(self, *args, name: str | None = None, **kwargs):
        if name is not None:
            return self.modules[name](*args, **kwargs)

        if kwargs.keys() != self.modules.keys():
            raise ValueError(
                'Initialization arguments must exactly match the module names: '
                f'expected {tuple(self.modules)}, got {tuple(kwargs)}.'
            )

        outputs = {}
        for module_name, module_args in kwargs.items():
            module = self.modules[module_name]
            if isinstance(module_args, Mapping):
                outputs[module_name] = module(**module_args)
            elif isinstance(module_args, Sequence):
                outputs[module_name] = module(*module_args)
            else:
                outputs[module_name] = module(module_args)
        return outputs


class TrainState(flax.struct.PyTreeNode):
    """Parameters, optimizer state, and the callable Flax module definition."""

    step: int
    apply_fn: Any = nonpytree_field()
    model_def: Any = nonpytree_field()
    params: Any
    tx: Any = nonpytree_field()
    opt_state: Any

    @classmethod
    def create(
        cls,
        model_def: nn.Module,
        params: Any,
        tx: optax.GradientTransformation | Mapping[str, optax.GradientTransformation] | None = None,
        **kwargs,
    ):
        if isinstance(tx, Mapping):
            opt_state = {}
            for module_name, transform in tx.items():
                key = f'modules_{module_name}'
                if key not in params:
                    raise ValueError(
                        f'Optimizer module {module_name!r} has no parameter subtree {key!r}.'
                    )
                opt_state[module_name] = transform.init(params[key])
        else:
            opt_state = None if tx is None else tx.init(params)
        return cls(
            step=1,
            apply_fn=model_def.apply,
            model_def=model_def,
            params=params,
            tx=tx,
            opt_state=opt_state,
            **kwargs,
        )

    def __call__(self, *args, params=None, method: str | None = None, **kwargs):
        variables = {'params': self.params if params is None else params}
        method_fn = None if method is None else getattr(self.model_def, method)
        return self.apply_fn(variables, *args, method=method_fn, **kwargs)

    def select(self, name: str):
        """Return a callable bound to one module in a :class:`ModuleDict`."""
        return functools.partial(self, name=name)

    def apply_gradients(self, grads, **kwargs):
        if self.tx is None:
            raise ValueError('Cannot apply gradients without an optimizer.')
        if isinstance(self.tx, Mapping):
            raise ValueError(
                'Module optimizers require apply_loss_fn(..., trainable_modules=...).'
            )
        updates, opt_state = self.tx.update(grads, self.opt_state, self.params)
        params = optax.apply_updates(self.params, updates)
        return self.replace(step=self.step + 1, params=params, opt_state=opt_state, **kwargs)

    def apply_loss_fn(self, loss_fn, *, trainable_modules: Sequence[str] | None = None):
        """Differentiate ``loss_fn`` and update only explicitly selected modules.

        A mapping of module name to optimizer gives every trainable subtree its
        own Adam counter and moments.  Optimizers for omitted modules are not
        called, so both their parameters and optimizer states remain bitwise
        unchanged across phase boundaries and delayed-policy steps.
        """
        if not isinstance(self.tx, Mapping):
            if trainable_modules is not None:
                raise ValueError('trainable_modules requires module optimizers.')
            grads, info = jax.grad(loss_fn, has_aux=True)(self.params)
            return self.apply_gradients(grads), info
        if not trainable_modules:
            raise ValueError('At least one trainable module must be selected.')

        names = tuple(str(name) for name in trainable_modules)
        if len(set(names)) != len(names):
            raise ValueError(f'Duplicate trainable modules: {names}.')
        keys = tuple(f'modules_{name}' for name in names)
        for module_name, key in zip(names, keys):
            if module_name not in self.tx:
                raise ValueError(f'No optimizer configured for module {module_name!r}.')
            if key not in self.params:
                raise ValueError(
                    f'Trainable module {module_name!r} has no parameter subtree {key!r}.'
                )

        # Differentiate only the requested subtrees.  Building a gradient tree
        # for frozen image encoders/targets creates large zero buffers and makes
        # phase-specific updates compile and dispatch substantially more work.
        trainable_params = {key: self.params[key] for key in keys}

        def selected_loss(selected_params):
            if isinstance(self.params, flax.core.FrozenDict):
                full_params = self.params.copy(selected_params)
            else:
                full_params = {**self.params, **selected_params}
            return loss_fn(full_params)

        grads, info = jax.grad(selected_loss, has_aux=True)(trainable_params)
        frozen_params = isinstance(self.params, flax.core.FrozenDict)
        params = flax.core.unfreeze(self.params) if frozen_params else dict(self.params)
        opt_state = dict(self.opt_state)
        for module_name, key in zip(names, keys):
            transform = self.tx[module_name]
            updates, module_opt_state = transform.update(
                grads[key], self.opt_state[module_name], params[key]
            )
            params[key] = optax.apply_updates(params[key], updates)
            opt_state[module_name] = module_opt_state
        if frozen_params:
            params = flax.core.freeze(params)
        return self.replace(
            step=self.step + 1,
            params=params,
            opt_state=opt_state,
        ), info


_CHECKPOINT_NAME = re.compile(r'^params_(\d+)\.pkl$')


def resolve_checkpoint(
    path: str | os.PathLike[str],
    step: int = 0,
) -> tuple[Path, int]:
    """Resolve a checkpoint path and its unambiguous training step.

    A zero step means "infer from an exact ``params_<step>.pkl`` filename".
    Checkpoint directories always require an explicit positive step.
    """

    path = Path(path)
    if path.suffix == '.pkl':
        match = _CHECKPOINT_NAME.fullmatch(path.name)
        inferred_step = int(match.group(1)) if match is not None else None
        requested_step = int(step)
        if requested_step < 0:
            raise ValueError('Checkpoint step cannot be negative.')
        if requested_step == 0:
            if inferred_step is None:
                raise ValueError(
                    'An exact checkpoint with a nonstandard filename requires '
                    'an explicit positive step.'
                )
            requested_step = inferred_step
        elif inferred_step is not None and requested_step != inferred_step:
            raise ValueError(
                f'Checkpoint filename implies step {inferred_step}, but '
                f'step {requested_step} was requested.'
            )
        return path, requested_step

    step = int(step)
    if step < 1:
        raise ValueError(
            'A checkpoint directory requires an explicit positive step.'
        )
    return path / f'params_{step}.pkl', step


def _checkpoint_metadata(agent: Any) -> dict[str, Any]:
    """Serialize architecture settings and residual normalization metadata."""

    config = agent.config
    endpoint_distribution = str(config['endpoint_distribution'])
    prefix_model = str(config['prefix_model'])
    metadata = {
        'endpoint_distribution': endpoint_distribution,
        'prefix_model': prefix_model,
        'horizon': int(config['horizon']),
        'state_scale': np.asarray(agent.state_scale, dtype=np.float32),
        'offline_action_free': bool(config.get('offline_action_free', False)),
        'training_objective_protocol': training_protocol_from_config(config),
    }
    if prefix_model == 'low_rank_gaussian':
        metadata.update({
            'prefix_rank': int(config['prefix_rank']),
            'prefix_sigma_floor': float(config['prefix_sigma_floor']),
        })
    elif prefix_model == 'joint_flow':
        metadata['prefix_flow_steps'] = int(config['prefix_flow_steps'])
    return metadata


def _validate_checkpoint_metadata(
    agent: Any,
    metadata: Any,
    *,
    inference_only: bool = False,
) -> None:
    """Reject stochastic checkpoint/config mismatches before deserialization."""

    prefix_model = str(agent.config['prefix_model'])
    if metadata is None:
        if prefix_model != 'deterministic':
            raise ValueError(
                'A stochastic prefix checkpoint requires architecture metadata.'
            )
        if (
            not inference_only
            and training_protocol_from_config(agent.config)
            != REPORTED_RUN_LEGACY_TRAINING_PROTOCOL
        ):
            raise ValueError(
                'Checkpoint has no training_objective_protocol metadata; '
                'pre-versioned checkpoints require '
                'reported-run-legacy-v1.'
            )
        return
    if not isinstance(metadata, Mapping):
        raise ValueError('Invalid PathBridger checkpoint metadata.')

    expected = _checkpoint_metadata(agent)
    metadata = dict(metadata)
    # Checkpoints written before the protocol field existed used the behavior
    # now named reported-run-legacy-v1. This compatibility default is
    # intentionally one-way: a pre-versioned checkpoint cannot be resumed as
    # manuscript-v1.
    metadata['training_objective_protocol'] = normalize_training_protocol(
        metadata.get(
            'training_objective_protocol',
            REPORTED_RUN_LEGACY_TRAINING_PROTOCOL,
        )
    )
    keys = [
        'endpoint_distribution',
        'prefix_model',
        'horizon',
        'offline_action_free',
    ]
    if not inference_only:
        keys.append('training_objective_protocol')
    if prefix_model == 'low_rank_gaussian':
        keys.extend(['prefix_rank', 'prefix_sigma_floor'])
    elif prefix_model == 'joint_flow':
        keys.append('prefix_flow_steps')
    mismatches = [
        key
        for key in keys
        if key not in metadata or metadata[key] != expected[key]
    ]
    if mismatches:
        details = ', '.join(
            f'{key}: checkpoint={metadata.get(key)!r}, config={expected[key]!r}'
            for key in mismatches
        )
        raise ValueError(f'Checkpoint architecture mismatch ({details}).')

    checkpoint_scale = np.asarray(metadata.get('state_scale'), dtype=np.float32)
    expected_scale = np.asarray(expected['state_scale'], dtype=np.float32)
    if checkpoint_scale.shape != expected_scale.shape or not np.allclose(
        checkpoint_scale,
        expected_scale,
        rtol=1e-6,
        atol=1e-6,
    ):
        raise ValueError(
            'Checkpoint state_scale does not match the current training dataset.'
        )



def _file_sha256(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _legacy_checkpoint_params(
    path: str | os.PathLike[str],
) -> Mapping[str, Any]:
    checkpoint_path = Path(path)
    with checkpoint_path.open('rb') as file:
        payload = pickle.load(file)
    try:
        params = payload['agent']['network']['params']
    except (KeyError, TypeError) as error:
        raise ValueError(
            f'Invalid legacy PathBridger checkpoint: {checkpoint_path}'
        ) from error
    if not isinstance(params, Mapping):
        raise ValueError(
            f'Invalid legacy parameter tree: {checkpoint_path}'
        )
    return params


def _validated_legacy_subtree(
    template: Any,
    candidate: Any,
    label: str,
) -> Any:
    template_flat = flax.traverse_util.flatten_dict(
        flax.core.unfreeze(template)
    )
    candidate_flat = flax.traverse_util.flatten_dict(
        flax.core.unfreeze(candidate)
    )
    if template_flat.keys() != candidate_flat.keys():
        missing = sorted(template_flat.keys() - candidate_flat.keys())
        extra = sorted(candidate_flat.keys() - template_flat.keys())
        raise ValueError(
            f'Legacy subtree {label} has incompatible keys; '
            f'missing={missing}, extra={extra}.'
        )
    shape_mismatches = [
        (key, np.shape(candidate_flat[key]), np.shape(template_flat[key]))
        for key in template_flat
        if np.shape(candidate_flat[key]) != np.shape(template_flat[key])
    ]
    if shape_mismatches:
        raise ValueError(
            f'Legacy subtree {label} has incompatible shapes: '
            f'{shape_mismatches}.'
        )
    return flax.serialization.from_state_dict(template, candidate)


def validate_initialization_paths(
    *,
    restore_path: str,
    restore_step: int,
    legacy_dynamics_checkpoint: str,
    legacy_critic_checkpoint: str,
) -> bool:
    """Validate mutually exclusive resume and parameter-only warm-start inputs."""

    legacy_dynamics = bool(legacy_dynamics_checkpoint)
    legacy_critic = bool(legacy_critic_checkpoint)
    if legacy_dynamics != legacy_critic:
        raise ValueError(
            'legacy_dynamics_checkpoint and legacy_critic_checkpoint must be '
            'provided together.'
        )
    legacy_pair = legacy_dynamics and legacy_critic
    if restore_path and legacy_pair:
        raise ValueError(
            'restore_path and the legacy dynamics/critic checkpoint pair are '
            'mutually exclusive.'
        )
    if restore_step and not restore_path:
        raise ValueError('restore_step requires restore_path.')
    if legacy_pair:
        for label, raw_path in (
            ('legacy_dynamics_checkpoint', legacy_dynamics_checkpoint),
            ('legacy_critic_checkpoint', legacy_critic_checkpoint),
        ):
            path = Path(raw_path)
            if not path.is_file():
                raise FileNotFoundError(f'{label} is not a file: {path}')
    return legacy_pair


def initialize_from_legacy_pathbridger(
    agent: Any,
    dynamics_path: str | os.PathLike[str],
    critic_path: str | os.PathLike[str],
    *,
    legacy_inference_staging: bool = True,
    return_info: bool = False,
) -> Any:
    """Map legacy Flow/PBG parameters for inference or a fresh warm start.

    The split legacy checkpoints do not contain Python, NumPy, or sampler RNG
    state.  Their two optimizer states also do not have the unified schema, so
    this adapter deliberately migrates parameters only and never claims exact
    training continuation.  Staged inference is the default for exact legacy
    checkpoint comparison; fresh unified training explicitly disables it.
    """

    distribution = str(agent.config['endpoint_distribution']).lower()
    if distribution not in {'flow', 'gaussian'}:
        raise ValueError(
            'Legacy Main migration requires endpoint_distribution=flow or '
            f'gaussian, got {distribution!r}.'
        )
    if str(agent.config['prefix_model']).lower() != 'deterministic':
        raise ValueError('Legacy Main migration cannot initialize a new prefix model.')

    dynamics = _legacy_checkpoint_params(dynamics_path)
    critic = _legacy_checkpoint_params(critic_path)
    endpoint = dynamics.get('modules_subgoal_net')
    if endpoint is None:
        raise ValueError('Legacy dynamics checkpoint is missing modules_subgoal_net.')
    if distribution == 'gaussian':
        endpoint = flax.core.unfreeze(endpoint)
        if 'mu_head' in endpoint:
            endpoint['mean'] = endpoint.pop('mu_head')
        if 'log_std_head' in endpoint:
            endpoint['log_std'] = endpoint.pop('log_std_head')

    sources = {
        'modules_endpoint': endpoint,
        'modules_bridge': dynamics.get('modules_path_residual_net'),
        'modules_value': critic.get('modules_value'),
        'modules_target_value': critic.get('modules_target_value'),
    }
    if 'modules_idm' in agent.network.params:
        sources['modules_idm'] = dynamics.get('modules_idm_net')

    frozen = isinstance(agent.network.params, flax.core.FrozenDict)
    params = flax.core.unfreeze(agent.network.params) if frozen else dict(agent.network.params)
    for destination, source in sources.items():
        if source is None:
            raise ValueError(
                f'Legacy checkpoint is missing source for {destination!r}.'
            )
        params[destination] = _validated_legacy_subtree(
            params[destination], source, destination
        )
    if frozen:
        params = flax.core.freeze(params)
    migrated_config = agent.config.copy(
        add_or_replace={
            'legacy_inference_staging': bool(legacy_inference_staging)
        }
    )
    migrated = agent.replace(
        network=agent.network.replace(params=params),
        config=migrated_config,
    )
    info = {
        'migration': 'legacy-split-parameters-only',
        'inference_execution': (
            'legacy-staged-jit-boundaries'
            if legacy_inference_staging
            else 'unified-compiled-kernel'
        ),
        'inference_parameters_mapped': sorted(sources),
        'optimizer_state': 'not-migrated',
        'python_random_state': 'unavailable',
        'numpy_random_state': 'unavailable',
        'sampler_random_state': 'unavailable',
        'exact_training_resume_guaranteed': False,
        'sources': {
            'dynamics': {
                'path': str(Path(dynamics_path).resolve()),
                'sha256': _file_sha256(dynamics_path),
            },
            'critic': {
                'path': str(Path(critic_path).resolve()),
                'sha256': _file_sha256(critic_path),
            },
        },
    }
    return (migrated, info) if return_info else migrated


def save_agent(
    agent: Any,
    save_dir: str | os.PathLike[str],
    step: int,
    *,
    sampler: Any | None = None,
) -> str:
    """Save the unified agent and explicit host/sampler RNG states."""
    save_path, _ = resolve_checkpoint(save_dir, step)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'checkpoint_schema_version': 2,
        'agent': flax.serialization.to_state_dict(agent),
        'metadata': _checkpoint_metadata(agent),
        'numpy_random_state': np.random.get_state(),
        'python_random_state': random.getstate(),
    }
    if sampler is not None:
        if not hasattr(sampler, 'get_rng_state'):
            raise TypeError('sampler must expose get_rng_state().')
        payload['sampler_random_state'] = sampler.get_rng_state()
    temporary_path = save_path.with_suffix('.pkl.tmp')
    with temporary_path.open('wb') as file:
        pickle.dump(payload, file, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary_path, save_path)
    return str(save_path)


def restore_agent(
    agent: Any,
    restore_path: str | os.PathLike[str],
    step: int = 0,
    *,
    sampler: Any | None = None,
    return_info: bool = False,
    inference_only: bool = False,
) -> Any:
    """Restore an agent and report sampler/protocol compatibility.

    Training resume is fail-closed on objective-protocol mismatch. Explicit
    inference-only loading may use an inference-equivalent config, and records
    both protocol identities in the returned restore information.
    """
    checkpoint_path, _ = resolve_checkpoint(restore_path, step)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path}')
    with checkpoint_path.open('rb') as file:
        payload = pickle.load(file)
    if not isinstance(payload, dict) or 'agent' not in payload:
        raise ValueError(f'Invalid PathBridger checkpoint: {checkpoint_path}')
    metadata = payload.get('metadata')
    _validate_checkpoint_metadata(
        agent, metadata, inference_only=inference_only
    )
    checkpoint_protocol = (
        normalize_training_protocol(
            metadata.get(
                'training_objective_protocol',
                REPORTED_RUN_LEGACY_TRAINING_PROTOCOL,
            )
        )
        if isinstance(metadata, Mapping)
        else REPORTED_RUN_LEGACY_TRAINING_PROTOCOL
    )
    requested_protocol = training_protocol_from_config(agent.config)
    agent_state = payload['agent']
    if isinstance(agent_state, Mapping) and 'state_scale' not in agent_state:
        if str(agent.config['prefix_model']) != 'deterministic':
            raise ValueError('Legacy checkpoints cannot restore a stochastic prefix model.')
        agent_state = dict(agent_state)
        agent_state['state_scale'] = flax.serialization.to_state_dict(
            agent.state_scale
        )
    restored_agent = flax.serialization.from_state_dict(agent, agent_state)
    if 'numpy_random_state' in payload:
        np.random.set_state(payload['numpy_random_state'])
    if 'python_random_state' in payload:
        random.setstate(payload['python_random_state'])
    if sampler is None:
        sampler_resume = 'not-requested'
    elif not hasattr(sampler, 'set_rng_state'):
        raise TypeError('sampler must expose set_rng_state().')
    elif 'sampler_random_state' in payload:
        sampler.set_rng_state(payload['sampler_random_state'])
        sampler_resume = 'schema-v2-exact'
    elif 'numpy_random_state' in payload:
        sampler.set_rng_state(payload['numpy_random_state'])
        sampler_resume = 'legacy-global-numpy-migration'
    else:
        sampler_resume = 'unavailable'
    restore_info = {
        'checkpoint_schema_version': int(
            payload.get('checkpoint_schema_version', 0)
        ),
        'numpy_random_state': (
            'restored' if 'numpy_random_state' in payload else 'unavailable'
        ),
        'python_random_state': (
            'restored' if 'python_random_state' in payload else 'unavailable'
        ),
        'sampler_resume': sampler_resume,
        'exact_sampler_resume_guaranteed': sampler_resume == 'schema-v2-exact',
        'checkpoint_training_objective_protocol': checkpoint_protocol,
        'requested_training_objective_protocol': requested_protocol,
        'training_objective_protocol_match': (
            checkpoint_protocol == requested_protocol
        ),
        'inference_only': bool(inference_only),
    }
    if return_info:
        return restored_agent, restore_info
    return restored_agent


__all__ = [
    'ModuleDict',
    'TrainState',
    'nonpytree_field',
    'resolve_checkpoint',
    'validate_initialization_paths',
    'initialize_from_legacy_pathbridger',
    'restore_agent',
    'save_agent',
]
