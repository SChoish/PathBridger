#!/usr/bin/env python3
"""Subgoal SPI finetuning from an existing run checkpoint.

Loads a flow-trained dynamics/critic/actor checkpoint, adds a random-init
``subgoal_spi_net``, freezes everything else, and finetunes only the
deterministic subgoal SPI net against the frozen critic.
"""

from __future__ import annotations

import glob
import json
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Any

import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
import yaml
from absl import app, flags

import main as M  # noqa: F401
from eval_checkpoint import _build_configs
from main import (
    FLAGS,
    _create_actor_agent,
    _create_critic_agent,
    _evaluate_env_tasks,
    _extract_critic_value_params,
    _intersect_valid_starts,
    _make_critic_dataset,
    _sample_shared_idxs,
)
from agents.dynamics import DynamicsAgent
from utils.datasets import Dataset, PathHGCDataset
from utils.env_utils import make_env_and_datasets
from utils.eval_results_io import save_eval_results
from utils.flax_utils import TrainState, restore_agent, save_agent
from utils.freeze_check import assert_frozen, assert_trained, summarize_param_diff
from utils.goal_representation import infer_phi_goal_obs_indices, normalize_phi_goal_obs_indices
from utils.log_utils import CsvLogger
from utils.run_io import (
    list_checkpoint_suffixes,
    parse_int_list,
    pick_epoch,
    resolve_actor_checkpoint_dir,
    resolve_critic_checkpoint_dir,
    resolve_dynamics_checkpoint_dir,
)

_REPO = Path(__file__).resolve().parent

flags.DEFINE_string('pretrained_ckpt_dir', '', 'Pretrained run dir. Empty -> select runs/ best-IDM eval.')
flags.DEFINE_integer('pretrained_epoch', -1, 'Checkpoint suffix to load; -1 = best eval epoch or latest.')
flags.DEFINE_integer('subgoal_spi_steps', 100000, 'Number of subgoal SPI gradient steps.')
flags.DEFINE_float('subgoal_spi_lr', 3e-4, 'Adam learning rate for subgoal SPI finetuning.')
flags.DEFINE_float('subgoal_spi_tau', -1.0, 'Override subgoal_spi_tau; < 0 keeps the default (5.0).')
flags.DEFINE_integer('subgoal_spi_batch_size', 0, 'Batch size; 0 = use checkpoint batch_size.')
flags.DEFINE_integer('eval_interval', 50000, 'Eval every N steps; 0 disables intermediate eval.')
flags.DEFINE_integer('save_interval', 50000, 'Save dynamics every N steps; 0 = final only.')
flags.DEFINE_boolean(
    'subgoal_spi_select_best',
    True,
    'Best-of-N: score N flow candidates with the critic and use only the single '
    'highest-energy candidate as the proximal target (one-hot rho), matching actor SPI.',
)
flags.DEFINE_boolean('freeze_non_subgoal_spi', True, 'Assert non-SPI params do not change.')
flags.DEFINE_boolean('reset_dynamics_optimizer', True, 'Reinitialize dynamics optimizer before finetuning.')
flags.DEFINE_integer('debug_num_steps', 0, 'If > 0, override total steps for a smoke test.')
flags.DEFINE_string('subgoal_spi_out_root', 'checkpoints/subgoal_spi', 'Root dir for subgoal SPI outputs.')
flags.DEFINE_integer('subgoal_spi_eval_episodes', 25, 'Episodes per task for eval.')
flags.DEFINE_integer('subgoal_spi_log_interval', 500, 'Log every N steps.')
flags.DEFINE_string('mujoco_gl', '', 'Optional MuJoCo GL backend, e.g. egl.')
flags.DEFINE_string('subgoal_spi_config', '', 'Optional YAML setting subgoal SPI flags; CLI args win.')
flags.DEFINE_string('best_runs_root', 'runs', 'Root containing training runs for best-IDM auto selection.')
flags.DEFINE_string(
    'best_params_json',
    'docs/env_best_runs_choi/env_best_params.json',
    'JSON manifest for env best params. Empty disables.',
)
flags.DEFINE_string(
    'checkpoint_bundle_root',
    'docs/env_best_runs_choi/checkpoints',
    'Root containing per-env checkpoint bundles with flags.json.',
)
flags.DEFINE_boolean(
    'use_best_params_bundle',
    True,
    'Load from checkpoint_bundle_root/<env> and N/T from best_params_json.',
)
flags.DEFINE_integer(
    'subgoal_spi_num_samples',
    0,
    'Proposal candidates N for SPI update; 0 = use eval_n from best params (same as actor SPI).',
)


def _argv_sets_flag(name: str) -> bool:
    dashed = name.replace('_', '-')
    for arg in sys.argv[1:]:
        if arg.startswith(f'--{name}=') or arg.startswith(f'--{dashed}='):
            return True
        if arg in (f'--{name}', f'--{dashed}', f'--no{name}', f'--no{dashed}'):
            return True
    return False


def _apply_config_yaml(path: str) -> None:
    if not path:
        return
    with open(path, encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}
    for key, value in data.items():
        if not hasattr(FLAGS, key):
            raise ValueError(f'Unknown subgoal_spi_config key: {key!r}')
        if not _argv_sets_flag(key):
            setattr(FLAGS, key, value)


def _env_short(env_name: str) -> str:
    out = str(env_name)
    for suffix in ('-play-v0', '-navigate-v0'):
        if out.endswith(suffix):
            out = out[: -len(suffix)]
    return out


def _matches_env(requested: str, candidate: str) -> bool:
    return requested == candidate or requested == _env_short(candidate) or _env_short(requested) == _env_short(candidate)


def _has_checkpoint(run_dir: Path, epoch: int) -> bool:
    for sub in ('dynamics', 'critic', 'actor'):
        if not (run_dir / 'checkpoints' / sub / f'params_{int(epoch)}.pkl').is_file():
            return False
    return (run_dir / 'flags.json').is_file()


def _best_idm_eval_from_runs(env_name: str, runs_root: Path) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    best_key: tuple[float, float, int, int] | None = None
    for eval_path in sorted(runs_root.glob('*/eval_results/*.json')):
        try:
            with open(eval_path, encoding='utf-8') as f:
                rec = json.load(f)
        except Exception:
            continue
        candidate_env = str(rec.get('env_name', ''))
        if not candidate_env or not _matches_env(env_name, candidate_env):
            continue
        try:
            idm = float(rec['idm_success_rate_mean'])
        except Exception:
            continue
        actor = float(rec.get('actor_success_rate_mean', -1.0))
        epoch = int(rec.get('epoch', 0) or 0)
        eval_n = int(rec.get('subgoal_eval_num_samples', 0) or 0)
        run_dir = Path(rec.get('run_dir') or eval_path.parents[1])
        if not run_dir.is_absolute():
            run_dir = (_REPO / run_dir).resolve()
        if epoch <= 0 or not _has_checkpoint(run_dir, epoch):
            continue
        key = (idm, actor, epoch, eval_n)
        if best_key is None or key > best_key:
            best_key = key
            best = {
                'env_name': candidate_env,
                'run_dir': run_dir,
                'epoch': epoch,
                'eval_n': eval_n,
                'subgoal_temperature': rec.get('subgoal_temperature', None),
                'eval_json': eval_path,
                'idm': idm,
                'actor': actor,
            }
    if best is None:
        raise FileNotFoundError(
            f'No best-IDM eval checkpoint for env_name={env_name!r} under {runs_root}. '
            'Pass --pretrained_ckpt_dir explicitly.'
        )
    return best


def _best_idm_eval_from_checkpoint_meta(env_name: str, run_dir: Path) -> dict[str, Any] | None:
    meta_path = run_dir / 'best_eval_meta.yaml'
    if not meta_path.is_file():
        return None
    with open(meta_path, encoding='utf-8') as f:
        meta = yaml.safe_load(f) or {}
    candidate_env = str(meta.get('env') or env_name)
    if not _matches_env(env_name, candidate_env):
        raise ValueError(f'best_eval_meta env={candidate_env!r} does not match requested env={env_name!r}')
    best_eval = meta.get('best_eval') or {}
    if not best_eval:
        return None
    return {
        'env_name': candidate_env,
        'run_dir': run_dir,
        'epoch': int(meta.get('checkpoint_epoch', 0) or 0),
        'eval_n': int(best_eval.get('eval_N', 0) or 0),
        'subgoal_temperature': best_eval.get('temp', None),
        'eval_json': run_dir / str(best_eval.get('eval_json', '')),
        'idm': float(best_eval.get('IDM', float('nan'))),
        'actor': float(best_eval.get('ACTOR', float('nan'))),
    }


def _load_best_params_entry(env_name: str, json_path: Path) -> dict[str, Any]:
    with open(json_path, encoding='utf-8') as f:
        data = json.load(f)
    for entry in data.get('environments', []):
        candidate = str(entry.get('env', ''))
        if candidate and _matches_env(env_name, candidate):
            return entry
    raise FileNotFoundError(f'No env={env_name!r} in {json_path}')


def _resolve_best_params_bundle(
    env_name: str,
    bundle_root: Path,
    params_entry: dict[str, Any],
) -> tuple[str, Path, dict[str, Any]]:
    resolved_env = str(params_entry.get('env', env_name))
    ckpt_dir = bundle_root / resolved_env
    if not ckpt_dir.is_dir():
        raise FileNotFoundError(f'Checkpoint bundle not found: {ckpt_dir}')
    if not (ckpt_dir / 'flags.json').is_file():
        raise FileNotFoundError(f'Missing flags.json in checkpoint bundle: {ckpt_dir}')
    eval_n = int(params_entry.get('eval_n', 1) or 1)
    eval_temp = float(params_entry.get('eval_temperature', params_entry.get('subgoal_temperature', 1.0)))
    epoch = int(params_entry.get('checkpoint_epoch', 600) or 600)
    best_eval = {
        'env_name': resolved_env,
        'run_dir': ckpt_dir,
        'epoch': epoch,
        'eval_n': eval_n,
        'subgoal_temperature': eval_temp,
        'eval_json': params_entry.get('eval_json_path', ''),
        'idm': float(params_entry.get('IDM', float('nan'))),
        'actor': float(params_entry.get('ACTOR', float('nan'))),
    }
    return resolved_env, ckpt_dir, best_eval


def _resolve_pretrained(env_name: str, explicit: str) -> tuple[str, Path, dict[str, Any] | None]:
    if explicit:
        run_dir = Path(explicit)
        if not run_dir.is_absolute():
            run_dir = (_REPO / run_dir).resolve()
        if not run_dir.is_dir():
            raise FileNotFoundError(f'pretrained_ckpt_dir not found: {run_dir}')
        flags_path = run_dir / 'flags.json'
        resolved_env = env_name
        if flags_path.is_file():
            with open(flags_path, encoding='utf-8') as f:
                resolved_env = json.load(f).get('flags', {}).get('env_name', env_name)
        best = _best_idm_eval_from_checkpoint_meta(resolved_env, run_dir)
        return resolved_env, run_dir, best

    runs_root = Path(FLAGS.best_runs_root)
    if not runs_root.is_absolute():
        runs_root = (_REPO / runs_root).resolve()
    best = _best_idm_eval_from_runs(env_name, runs_root)
    return str(best['env_name']), Path(best['run_dir']), best


def _format_tau_tag(tau: float) -> str:
    return ('%g' % float(tau)).replace('.', 'p')


def _to_floats(info: dict[str, Any]) -> dict[str, float]:
    out = {}
    for key, value in info.items():
        try:
            out[key] = float(np.asarray(value))
        except (TypeError, ValueError):
            continue
    return out


def _deep_merge_params(dst: dict[str, Any], src: dict[str, Any]) -> dict[str, Any]:
    out = dict(dst)
    for key, value in src.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_merge_params(out[key], value)
        else:
            out[key] = value
    return out


def _restore_dynamics_with_spi_net(agent: DynamicsAgent, restore_path: str, restore_epoch: int) -> DynamicsAgent:
    """Restore flow checkpoint into an agent that also has random-init ``subgoal_spi_net``."""
    candidates = glob.glob(restore_path)
    assert len(candidates) == 1, f'Found {len(candidates)} candidates: {candidates}'
    pkl_path = candidates[0] + f'/params_{restore_epoch}.pkl'
    with open(pkl_path, 'rb') as f:
        load_dict = pickle.load(f)
    ckpt_agent = load_dict['agent']
    merged = flax.serialization.to_state_dict(agent)
    merged['network']['params'] = _deep_merge_params(
        merged['network']['params'],
        ckpt_agent['network']['params'],
    )
    if 'schedule' in ckpt_agent:
        merged['schedule'] = ckpt_agent['schedule']
    restored = flax.serialization.from_state_dict(agent, merged)
    print(f'[subgoal_spi] partial dynamics restore from {pkl_path}', flush=True)
    return restored


def _module_params(params: dict[str, Any], module_key: str) -> dict[str, Any]:
    return params.get(module_key, {})


def _write_effective_metadata(
    *,
    root: dict[str, Any],
    cfg_used: Path,
    out_dir: Path,
    dynamics_config: Any,
    critic_config: Any,
    actor_config: Any,
    spi_tau: float,
    batch_size: int,
    best_eval: dict[str, Any] | None,
    spi_num_samples: int,
) -> None:
    effective_root = json.loads(json.dumps(root))
    effective_root.setdefault('dynamics', {})['subgoal_eval_num_samples'] = int(
        dynamics_config.get('subgoal_eval_num_samples', 1)
    )
    effective_root['dynamics']['subgoal_temperature'] = float(dynamics_config.get('subgoal_temperature', 1.0))
    effective_root['dynamics']['subgoal_spi_enabled'] = True
    effective_root['dynamics']['subgoal_spi_tau'] = float(spi_tau)
    effective_root['dynamics']['subgoal_spi_num_samples'] = int(spi_num_samples)
    effective_root['dynamics']['batch_size'] = int(batch_size)
    effective_root.setdefault('critic_agent', {})['batch_size'] = int(batch_size)
    effective_root.setdefault('actor', {})['batch_size'] = int(batch_size)
    effective_root.setdefault('flags', {})['batch_size'] = int(batch_size)
    effective_root['flags']['subgoal_spi_source_best_eval_json'] = (
        str(best_eval['eval_json']) if best_eval is not None else ''
    )
    effective_root['flags']['subgoal_spi_spi_tau'] = float(spi_tau)
    effective_root['flags']['subgoal_spi_lr'] = float(dynamics_config.get('lr', 0.0))
    (out_dir / 'flags.json').write_text(json.dumps(effective_root, indent=2, sort_keys=True), encoding='utf-8')

    if cfg_used.is_file():
        with open(cfg_used, encoding='utf-8') as f:
            cfg = yaml.safe_load(f) or {}
    else:
        cfg = {}
    cfg.setdefault('dynamics', {})['subgoal_eval_num_samples'] = int(
        dynamics_config.get('subgoal_eval_num_samples', 1)
    )
    cfg['dynamics']['subgoal_temperature'] = float(dynamics_config.get('subgoal_temperature', 1.0))
    cfg['dynamics']['subgoal_spi_enabled'] = True
    cfg['dynamics']['subgoal_spi_tau'] = float(spi_tau)
    cfg['dynamics']['subgoal_spi_num_samples'] = int(spi_num_samples)
    cfg['dynamics']['lr'] = float(dynamics_config.get('lr', 0.0))
    cfg['batch_size'] = int(batch_size)
    cfg['subgoal_spi'] = {
        'source_best_eval_json': str(best_eval['eval_json']) if best_eval is not None else '',
        'source_best_eval_idm': float(best_eval['idm']) if best_eval is not None else None,
        'source_best_eval_actor': float(best_eval['actor']) if best_eval is not None else None,
        'subgoal_spi_tau': float(spi_tau),
        'subgoal_spi_lr': float(dynamics_config.get('lr', 0.0)),
        'proposal_num_samples': int(spi_num_samples),
        'proposal_temperature': float(dynamics_config.get('subgoal_temperature', 1.0)),
        'subgoal_eval_num_samples': int(dynamics_config.get('subgoal_eval_num_samples', 1)),
        'subgoal_temperature': float(dynamics_config.get('subgoal_temperature', 1.0)),
    }
    (out_dir / 'config_used.yaml').write_text(
        yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False), encoding='utf-8'
    )


def main(_):
    _apply_config_yaml(FLAGS.subgoal_spi_config)

    if str(FLAGS.mujoco_gl).strip():
        from rollout.env import configure_mujoco_gl

        configure_mujoco_gl(str(FLAGS.mujoco_gl))

    seed = int(FLAGS.seed)
    requested_env = str(FLAGS.env_name)
    if bool(FLAGS.use_best_params_bundle) and str(FLAGS.best_params_json).strip():
        params_json = Path(FLAGS.best_params_json)
        if not params_json.is_absolute():
            params_json = (_REPO / params_json).resolve()
        params_entry = _load_best_params_entry(requested_env, params_json)
        bundle_root = Path(FLAGS.checkpoint_bundle_root)
        if not bundle_root.is_absolute():
            bundle_root = (_REPO / bundle_root).resolve()
        resolved_env, ckpt_dir, best_eval = _resolve_best_params_bundle(
            requested_env, bundle_root, params_entry
        )
    else:
        resolved_env, ckpt_dir, best_eval = _resolve_pretrained(
            requested_env, str(FLAGS.pretrained_ckpt_dir).strip()
        )
    FLAGS.env_name = resolved_env

    flags_path = ckpt_dir / 'flags.json'
    with open(flags_path, encoding='utf-8') as f:
        root = json.load(f)
    fg = root['flags']

    FLAGS.plan_candidates = int(fg.get('plan_candidates', 1))
    FLAGS.plan_noise_scale = float(fg.get('plan_noise_scale', 1.0))
    FLAGS.measure_timing = False

    dynamics_config, critic_config, actor_config = _build_configs(root, fg)
    if best_eval is not None:
        dynamics_config['subgoal_eval_num_samples'] = int(best_eval['eval_n'])
        if best_eval.get('subgoal_temperature') is not None:
            dynamics_config['subgoal_temperature'] = float(best_eval['subgoal_temperature'])

    dynamics_config['subgoal_spi_enabled'] = True
    dynamics_config['subgoal_spi_proposal_distribution'] = 'flow'
    dynamics_config['subgoal_spi_proposal_loss_weight'] = 0.0
    dynamics_config['subgoal_spi_select_best'] = bool(FLAGS.subgoal_spi_select_best)
    dynamics_config['subgoal_spi_beta'] = float(dynamics_config.get('subgoal_spi_beta', 1.0))
    dynamics_config['subgoal_spi_energy_norm_eps'] = float(
        dynamics_config.get('subgoal_spi_energy_norm_eps', 1e-6)
    )
    dynamics_config['lr'] = float(FLAGS.subgoal_spi_lr)
    if float(FLAGS.subgoal_spi_tau) >= 0.0:
        dynamics_config['subgoal_spi_tau'] = float(FLAGS.subgoal_spi_tau)
    spi_tau = float(dynamics_config.get('subgoal_spi_tau', 5.0))
    eval_n = int(dynamics_config.get('subgoal_eval_num_samples', 1))
    eval_temperature = float(dynamics_config.get('subgoal_temperature', 1.0))
    # Match actor SPI: proposal sampling N/T from best_eval_meta (eval_n, temp).
    spi_num_samples = int(FLAGS.subgoal_spi_num_samples)
    if spi_num_samples <= 0:
        spi_num_samples = eval_n
    dynamics_config['subgoal_spi_num_samples'] = max(1, spi_num_samples)
    dynamics_config['subgoal_temperature'] = float(eval_temperature)

    batch_size = int(FLAGS.subgoal_spi_batch_size) if int(FLAGS.subgoal_spi_batch_size) > 0 else int(fg['batch_size'])
    dynamics_config['batch_size'] = batch_size
    critic_config['batch_size'] = batch_size
    actor_config['batch_size'] = batch_size

    dataset_dir = fg.get('dataset_dir', '') or str(FLAGS.dataset_dir)
    env, train_plain, _ = make_env_and_datasets(
        resolved_env,
        frame_stack=critic_config['frame_stack'],
        dataset_dir=dataset_dir,
    )
    obs_dim_env = int(np.prod(env.observation_space.shape))
    phi_idxs = normalize_phi_goal_obs_indices(critic_config.get('phi_goal_obs_indices', ()))
    if not phi_idxs:
        phi_idxs = infer_phi_goal_obs_indices(str(resolved_env), obs_dim_env)
    critic_config['phi_goal_obs_indices'] = phi_idxs
    dynamics_config['phi_goal_obs_indices'] = phi_idxs
    action_dim = int(np.asarray(env.action_space.shape).prod())
    critic_config['action_dim'] = action_dim
    actor_config['action_dim'] = action_dim

    dynamics_dataset = PathHGCDataset(Dataset.create(**train_plain), dynamics_config)
    critic_dataset = _make_critic_dataset(train_plain, critic_config)
    common_valid_starts = _intersect_valid_starts(dynamics_dataset, critic_dataset)
    np.random.seed(seed)
    ex_idxs = _sample_shared_idxs(common_valid_starts, batch_size)
    ex_dynamics = dynamics_dataset.sample(len(ex_idxs), idxs=ex_idxs)
    ex_critic = critic_dataset.sample(len(ex_idxs), idxs=ex_idxs)

    dynamics_agent = DynamicsAgent.create(
        seed, ex_dynamics['observations'], dynamics_config, ex_actions=ex_dynamics['actions']
    )
    critic_agent = _create_critic_agent(seed, ex_critic, critic_config)
    actor_agent = _create_actor_agent(seed, ex_dynamics, actor_config)

    dyn_dir = resolve_dynamics_checkpoint_dir(ckpt_dir)
    if int(FLAGS.pretrained_epoch) >= 0:
        epoch = pick_epoch(int(FLAGS.pretrained_epoch), list_checkpoint_suffixes(dyn_dir))
    elif best_eval is not None:
        epoch = int(best_eval['epoch'])
    else:
        epoch = pick_epoch(-1, list_checkpoint_suffixes(dyn_dir))

    dynamics_agent = _restore_dynamics_with_spi_net(dynamics_agent, str(dyn_dir), epoch)
    critic_agent = restore_agent(critic_agent, str(resolve_critic_checkpoint_dir(ckpt_dir)), epoch)
    actor_dir = resolve_actor_checkpoint_dir(ckpt_dir, required=True)
    actor_agent = restore_agent(actor_agent, str(actor_dir), epoch)

    dynamics_agent = dynamics_agent.replace(
        config=flax.core.FrozenDict(
            {
                **dict(dynamics_agent.config),
                'subgoal_spi_enabled': True,
                'subgoal_spi_proposal_loss_weight': 0.0,
                'subgoal_spi_tau': float(spi_tau),
                'subgoal_spi_num_samples': int(spi_num_samples),
                'subgoal_eval_num_samples': int(eval_n),
                'subgoal_temperature': float(eval_temperature),
                'lr': float(FLAGS.subgoal_spi_lr),
            }
        )
    )

    if bool(FLAGS.reset_dynamics_optimizer):
        dynamics_agent = dynamics_agent.replace(
            network=TrainState.create(
                dynamics_agent.network.model_def,
                dynamics_agent.network.params,
                tx=optax.adam(float(FLAGS.subgoal_spi_lr)),
            )
        )

    tau_tag = _format_tau_tag(spi_tau)
    source_tag = f'epoch_{epoch}_n{eval_n}_t{_format_tau_tag(eval_temperature)}'
    if bool(FLAGS.use_best_params_bundle):
        out_root = Path(FLAGS.subgoal_spi_out_root) / 'best_params'
    elif int(epoch) >= 999999 or '1m_env_best' in str(ckpt_dir):
        out_root = Path(FLAGS.subgoal_spi_out_root) / '1m'
    else:
        out_root = Path(FLAGS.subgoal_spi_out_root)
    out_dir = out_root / resolved_env / source_tag / f'tau_{tau_tag}' / f'seed_{seed}'
    if not out_dir.is_absolute():
        out_dir = _REPO / out_dir
    dynamics_out_dir = out_dir / 'checkpoints' / 'dynamics'
    dynamics_out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'eval_results').mkdir(parents=True, exist_ok=True)
    cfg_used = ckpt_dir / 'config_used.yaml'
    _write_effective_metadata(
        root=root,
        cfg_used=cfg_used,
        out_dir=out_dir,
        dynamics_config=dynamics_config,
        critic_config=critic_config,
        actor_config=actor_config,
        spi_tau=spi_tau,
        batch_size=batch_size,
        best_eval=best_eval,
        spi_num_samples=spi_num_samples,
    )

    total_steps = int(FLAGS.debug_num_steps) if int(FLAGS.debug_num_steps) > 0 else int(FLAGS.subgoal_spi_steps)
    log_interval = max(1, int(FLAGS.subgoal_spi_log_interval))
    eval_task_ids = parse_int_list(FLAGS.eval_task_ids)
    eval_episodes = max(1, int(FLAGS.subgoal_spi_eval_episodes))

    print(
        f'[subgoal_spi] env={resolved_env} ckpt_dir={ckpt_dir} epoch={epoch} '
        f'best_eval={best_eval and best_eval.get("eval_json", "")} '
        f'idm={best_eval and best_eval.get("idm")} actor={best_eval and best_eval.get("actor")} '
        f'eval_n={eval_n} eval_T={eval_temperature} proposal_N={spi_num_samples} '
        f'tau={spi_tau} lr={float(FLAGS.subgoal_spi_lr)} batch={batch_size} steps={total_steps}',
        flush=True,
    )
    print(f'[subgoal_spi] out_dir={out_dir}', flush=True)

    params_before = jax.device_get(dynamics_agent.network.params)
    spi_before = _module_params(params_before, 'modules_subgoal_spi_net')
    subgoal_before = _module_params(params_before, 'modules_subgoal_net')
    bridge_before = _module_params(params_before, 'modules_path_residual_net')
    idm_before = _module_params(params_before, 'modules_idm_net')
    critic_before = jax.device_get(critic_agent.network.params)
    actor_before = jax.device_get(actor_agent.actor.params)
    csv_logger = CsvLogger(os.path.join(out_dir, 'train.csv'), flush_every_n=1)

    def _run_eval(step: int) -> dict[str, Any]:
        metrics = _evaluate_env_tasks(
            env,
            dynamics_agent,
            actor_agent,
            actor_config,
            critic_config,
            critic_agent=critic_agent,
            task_ids=eval_task_ids,
            episodes_per_task=eval_episodes,
            wandb_enabled=False,
            subgoal_override_goal=bool(FLAGS.subgoal_override_goal),
        )
        save_eval_results(
            out_dir,
            epoch=step,
            subgoal_eval_num_samples=eval_n,
            task_ids=eval_task_ids,
            episodes_per_task=eval_episodes,
            metrics=metrics,
            fg=fg,
            root=root,
            subgoal_temperature=eval_temperature,
        )
        return metrics

    first_time = time.time()
    final_metrics: dict[str, Any] | None = None
    for step in range(1, total_steps + 1):
        idxs = _sample_shared_idxs(common_valid_starts, batch_size)
        dynamics_batch = dynamics_dataset.sample(batch_size, idxs=idxs)
        dynamics_agent, dyn_info = dynamics_agent.update_subgoal_spi(
            dynamics_batch,
            critic_value_params=_extract_critic_value_params(critic_agent),
        )

        if int(FLAGS.save_interval) > 0 and step % int(FLAGS.save_interval) == 0:
            save_agent(dynamics_agent, str(dynamics_out_dir), step)

        do_eval = int(FLAGS.eval_interval) > 0 and step % int(FLAGS.eval_interval) == 0
        do_log = step % log_interval == 0 or step == total_steps or do_eval or int(FLAGS.debug_num_steps) > 0
        metrics = None
        if do_eval:
            metrics = _run_eval(step)
            if step == total_steps:
                final_metrics = metrics

        if do_log:
            info = _to_floats(dyn_info)
            row = {
                'subgoal_spi/loss': info.get('phase1/subgoal_spi_loss', float('nan')),
                'subgoal_spi/proposal_loss': info.get('phase1/subgoal_spi_proposal_loss', float('nan')),
                'subgoal_spi/energy_mean': info.get('phase1/subgoal_spi_energy_mean', float('nan')),
                'subgoal_spi/prox_mean': info.get('phase1/subgoal_spi_prox_mean', float('nan')),
                'subgoal_spi/rho_entropy': info.get('phase1/subgoal_spi_rho_entropy', float('nan')),
                'subgoal_spi/tau': spi_tau,
                'subgoal_spi/eval_n': float(eval_n),
                'subgoal_spi/eval_temperature': float(eval_temperature),
                'subgoal_spi/proposal_n': float(spi_num_samples),
                'subgoal_spi/step': float(step),
                'time/total_sec': time.time() - first_time,
            }
            if metrics is not None:
                row['subgoal_spi/eval_spi_subgoal_actor'] = float(
                    metrics.get('eval_spi_subgoal_actor/success_rate_mean', float('nan'))
                )
                row['subgoal_spi/eval_spi_subgoal_idm'] = float(
                    metrics.get('eval_spi_subgoal_idm/success_rate_mean', float('nan'))
                )
                row['subgoal_spi/eval_flow_actor'] = float(
                    metrics.get('eval_flow_actor/success_rate_mean', float('nan'))
                )
                row['subgoal_spi/eval_flow_idm'] = float(
                    metrics.get('eval_flow_idm/success_rate_mean', float('nan'))
                )
            csv_logger.log(row, step=step)
            print(
                f'[subgoal_spi] step={step}/{total_steps} loss={row["subgoal_spi/loss"]:.4f} '
                f'energy={row["subgoal_spi/energy_mean"]:.4f} prox={row["subgoal_spi/prox_mean"]:.4f}',
                flush=True,
            )

    save_agent(dynamics_agent, str(dynamics_out_dir), total_steps)

    params_after = jax.device_get(dynamics_agent.network.params)
    spi_after = _module_params(params_after, 'modules_subgoal_spi_net')
    subgoal_after = _module_params(params_after, 'modules_subgoal_net')
    bridge_after = _module_params(params_after, 'modules_path_residual_net')
    idm_after = _module_params(params_after, 'modules_idm_net')
    critic_after = jax.device_get(critic_agent.network.params)
    actor_after = jax.device_get(actor_agent.actor.params)

    if bool(FLAGS.freeze_non_subgoal_spi):
        assert_frozen(subgoal_before, subgoal_after, name='subgoal_net', tol=1e-6)
        assert_frozen(bridge_before, bridge_after, name='path_residual_net', tol=1e-6)
        assert_frozen(idm_before, idm_after, name='idm_net', tol=1e-6)
        assert_frozen(critic_before, critic_after, name='critic', tol=1e-6)
        assert_frozen(actor_before, actor_after, name='actor', tol=1e-6)
        assert_trained(spi_before, spi_after, name='subgoal_spi_net', min_abs=0.0)
        print('[subgoal_spi] freeze check PASSED', flush=True)

    if final_metrics is None:
        final_metrics = _run_eval(total_steps)

    spi_diff = summarize_param_diff(spi_before, spi_after)
    meta = {
        'env': resolved_env,
        'requested_env': requested_env,
        'pretrained_ckpt_dir': str(ckpt_dir),
        'pretrained_epoch': int(epoch),
        'best_eval_json': str(best_eval['eval_json']) if best_eval is not None else '',
        'best_eval_idm': float(best_eval['idm']) if best_eval is not None else None,
        'best_eval_actor': float(best_eval['actor']) if best_eval is not None else None,
        'subgoal_eval_num_samples': eval_n,
        'subgoal_temperature': eval_temperature,
        'subgoal_spi_num_samples': spi_num_samples,
        'subgoal_spi_select_best': bool(FLAGS.subgoal_spi_select_best),
        'use_best_params_bundle': bool(FLAGS.use_best_params_bundle),
        'checkpoint_bundle': str(ckpt_dir),
        'subgoal_spi_tau': spi_tau,
        'subgoal_spi_lr': float(FLAGS.subgoal_spi_lr),
        'subgoal_spi_steps': int(total_steps),
        'subgoal_spi_batch_size': int(batch_size),
        'seed': seed,
        'dynamics_checkpoint': f'checkpoints/dynamics/params_{total_steps}.pkl',
        'param_diff': {'subgoal_spi_max_abs': spi_diff['max_abs']},
        'final_eval': {
            'spi_subgoal_actor': float(
                final_metrics.get('eval_spi_subgoal_actor/success_rate_mean', float('nan'))
            ),
            'spi_subgoal_idm': float(
                final_metrics.get('eval_spi_subgoal_idm/success_rate_mean', float('nan'))
            ),
            'flow_actor': float(final_metrics.get('eval_flow_actor/success_rate_mean', float('nan'))),
            'flow_idm': float(final_metrics.get('eval_flow_idm/success_rate_mean', float('nan'))),
        },
    }
    (out_dir / 'subgoal_spi_meta.yaml').write_text(
        yaml.safe_dump(meta, sort_keys=False, default_flow_style=False), encoding='utf-8'
    )
    csv_logger.close() if hasattr(csv_logger, 'close') else None
    print(
        f"[subgoal_spi] FINAL spi_actor={meta['final_eval']['spi_subgoal_actor']:.4f} "
        f"spi_idm={meta['final_eval']['spi_subgoal_idm']:.4f} "
        f"flow_actor={meta['final_eval']['flow_actor']:.4f} "
        f"flow_idm={meta['final_eval']['flow_idm']:.4f}",
        flush=True,
    )
    print(f'[subgoal_spi] DONE outputs={out_dir}', flush=True)


if __name__ == '__main__':
    app.run(main)
