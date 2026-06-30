#!/usr/bin/env python3
"""Actor-only SPI finetuning from an existing run checkpoint.

By default this script scans ``runs/*/eval_results/*.json`` for the requested
environment's best IDM eval, loads the matching checkpoint suffix, freezes
dynamics/IDM/critic, and finetunes only the deterministic SPI actor.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
import yaml
from absl import app, flags

# Import main first so its flags and helper functions are registered.
import main as M  # noqa: F401
from eval_checkpoint import _build_configs
from main import (
    FLAGS,
    _build_actor_batch_from_dynamics,
    _create_actor_agent,
    _create_critic_agent,
    _evaluate_env_tasks,
    _intersect_valid_starts,
    _make_critic_dataset,
    _rescore_actor_batch_for_update,
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
flags.DEFINE_integer('actor_spi_steps', 100000, 'Number of actor SPI gradient steps.')
flags.DEFINE_float('actor_spi_lr', 3e-4, 'Adam learning rate for actor SPI finetuning.')
flags.DEFINE_float('spi_tau', -1.0, 'Override actor SPI tau; < 0 keeps the checkpoint value.')
flags.DEFINE_integer('actor_spi_batch_size', 0, 'Batch size; 0 = use checkpoint batch_size.')
flags.DEFINE_integer('eval_interval', 50000, 'Eval every N actor SPI steps; 0 disables intermediate eval.')
flags.DEFINE_integer('save_interval', 50000, 'Save actor every N steps; 0 = final only.')
flags.DEFINE_boolean('freeze_non_actor', True, 'Assert dynamics/critic params do not change.')
flags.DEFINE_boolean('reset_actor_optimizer', True, 'Reinitialize actor optimizer before finetuning.')
flags.DEFINE_integer('debug_num_steps', 0, 'If > 0, override total steps for a smoke test.')
flags.DEFINE_string('actor_spi_out_root', 'checkpoints/actor_spi', 'Root dir for actor SPI outputs.')
flags.DEFINE_integer('actor_spi_eval_episodes', 25, 'Episodes per task for actor SPI eval.')
flags.DEFINE_integer('actor_spi_log_interval', 500, 'Log every N actor SPI steps.')
flags.DEFINE_string('mujoco_gl', '', 'Optional MuJoCo GL backend, e.g. egl.')
flags.DEFINE_string('actor_spi_config', '', 'Optional YAML setting actor SPI flags; CLI args win.')
flags.DEFINE_string('best_runs_root', 'runs', 'Root containing training runs for best-IDM auto selection.')


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
            raise ValueError(f'Unknown actor_spi_config key: {key!r}')
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
        return resolved_env, run_dir, None

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


def main(_):
    _apply_config_yaml(FLAGS.actor_spi_config)

    if str(FLAGS.mujoco_gl).strip():
        from rollout.env import configure_mujoco_gl

        configure_mujoco_gl(str(FLAGS.mujoco_gl))

    seed = int(FLAGS.seed)
    requested_env = str(FLAGS.env_name)
    resolved_env, ckpt_dir, best_eval = _resolve_pretrained(requested_env, str(FLAGS.pretrained_ckpt_dir).strip())
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

    actor_config['lr'] = float(FLAGS.actor_spi_lr)
    if float(FLAGS.spi_tau) >= 0.0:
        actor_config['spi_tau'] = float(FLAGS.spi_tau)
    spi_tau = float(actor_config['spi_tau'])

    batch_size = int(FLAGS.actor_spi_batch_size) if int(FLAGS.actor_spi_batch_size) > 0 else int(fg['batch_size'])
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
    dynamics_agent = restore_agent(dynamics_agent, str(dyn_dir), epoch)
    critic_agent = restore_agent(critic_agent, str(resolve_critic_checkpoint_dir(ckpt_dir)), epoch)
    actor_agent = restore_agent(actor_agent, str(resolve_actor_checkpoint_dir(ckpt_dir, required=True)), epoch)

    if bool(FLAGS.reset_actor_optimizer):
        actor_agent = actor_agent.replace(
            actor=TrainState.create(
                actor_agent.actor.model_def,
                actor_agent.actor.params,
                tx=optax.adam(float(FLAGS.actor_spi_lr)),
            )
        )

    eval_n = int(dynamics_config.get('subgoal_eval_num_samples', 1))
    tau_tag = _format_tau_tag(spi_tau)
    source_tag = f'epoch_{epoch}_n{eval_n}'
    out_dir = Path(FLAGS.actor_spi_out_root) / resolved_env / source_tag / f'tau_{tau_tag}' / f'seed_{seed}'
    if not out_dir.is_absolute():
        out_dir = _REPO / out_dir
    actor_out_dir = out_dir / 'checkpoints' / 'actor'
    actor_out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'eval_results').mkdir(parents=True, exist_ok=True)
    shutil.copy2(flags_path, out_dir / 'flags.json')
    cfg_used = ckpt_dir / 'config_used.yaml'
    if cfg_used.is_file():
        shutil.copy2(cfg_used, out_dir / 'config_used.yaml')

    total_steps = int(FLAGS.debug_num_steps) if int(FLAGS.debug_num_steps) > 0 else int(FLAGS.actor_spi_steps)
    log_interval = max(1, int(FLAGS.actor_spi_log_interval))
    eval_task_ids = parse_int_list(FLAGS.eval_task_ids)
    eval_episodes = max(1, int(FLAGS.actor_spi_eval_episodes))

    print(
        f'[actor_spi] env={resolved_env} ckpt_dir={ckpt_dir} epoch={epoch} best_eval={best_eval and best_eval["eval_json"]} '
        f'idm={best_eval and best_eval["idm"]} actor={best_eval and best_eval["actor"]} eval_n={eval_n} '
        f'tau={spi_tau} lr={float(FLAGS.actor_spi_lr)} batch={batch_size} steps={total_steps}',
        flush=True,
    )
    print(f'[actor_spi] out_dir={out_dir}', flush=True)

    actor_before = jax.device_get(actor_agent.actor.params)
    critic_before = jax.device_get(critic_agent.network.params)
    dynamics_before = jax.device_get(dynamics_agent.network.params)
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
            subgoal_temperature=dynamics_config.get('subgoal_temperature'),
        )
        return metrics

    first_time = time.time()
    final_metrics: dict[str, Any] | None = None
    for step in range(1, total_steps + 1):
        idxs = _sample_shared_idxs(common_valid_starts, batch_size)
        dynamics_batch = dynamics_dataset.sample(batch_size, idxs=idxs)
        dynamics_agent, actor_batch, _, _ = _build_actor_batch_from_dynamics(
            dynamics_agent, critic_agent, dynamics_batch, actor_config
        )
        actor_batch_for_update, _ = _rescore_actor_batch_for_update(actor_batch, critic_agent, actor_config)
        actor_agent, actor_info = actor_agent.update(actor_batch_for_update, critic_agent)

        if int(FLAGS.save_interval) > 0 and step % int(FLAGS.save_interval) == 0:
            save_agent(actor_agent, str(actor_out_dir), step)

        do_eval = int(FLAGS.eval_interval) > 0 and step % int(FLAGS.eval_interval) == 0
        do_log = step % log_interval == 0 or step == total_steps or do_eval or int(FLAGS.debug_num_steps) > 0
        metrics = None
        if do_eval:
            metrics = _run_eval(step)
            if step == total_steps:
                final_metrics = metrics

        if do_log:
            info = _to_floats(actor_info)
            row = {
                'actor_spi/loss': info.get('spi_actor/actor_loss', float('nan')),
                'actor_spi/q_term': info.get('spi_actor/q_term', float('nan')),
                'actor_spi/prox_term': info.get('spi_actor/prox_term', float('nan')),
                'actor_spi/spi_tau': spi_tau,
                'actor_spi/actor_action_norm': info.get('spi_actor/actor_action_norm', float('nan')),
                'actor_spi/idm_action_norm': info.get('spi_actor/idm_action_norm', float('nan')),
                'actor_spi/action_l2_to_idm': info.get('spi_actor/action_l2_to_idm', float('nan')),
                'actor_spi/step': float(step),
                'time/total_sec': time.time() - first_time,
            }
            if metrics is not None:
                row['actor_spi/eval_actor_success_rate'] = float(metrics.get('eval/success_rate_mean', float('nan')))
                row['actor_spi/eval_idm_success_rate'] = float(metrics.get('eval_idm/success_rate_mean', float('nan')))
            csv_logger.log(row, step=step)
            print(
                f'[actor_spi] step={step}/{total_steps} loss={row["actor_spi/loss"]:.4f} '
                f'q={row["actor_spi/q_term"]:.4f} prox={row["actor_spi/prox_term"]:.4f} '
                f'l2={row["actor_spi/action_l2_to_idm"]:.4f}',
                flush=True,
            )

    save_agent(actor_agent, str(actor_out_dir), total_steps)

    actor_after = jax.device_get(actor_agent.actor.params)
    critic_after = jax.device_get(critic_agent.network.params)
    dynamics_after = jax.device_get(dynamics_agent.network.params)
    actor_diff = summarize_param_diff(actor_before, actor_after)
    critic_diff = summarize_param_diff(critic_before, critic_after)
    dynamics_diff = summarize_param_diff(dynamics_before, dynamics_after)
    if bool(FLAGS.freeze_non_actor):
        assert_frozen(critic_before, critic_after, name='critic', tol=1e-6)
        assert_frozen(dynamics_before, dynamics_after, name='dynamics', tol=1e-6)
        assert_trained(actor_before, actor_after, name='actor', min_abs=0.0)
        print('[actor_spi] freeze check PASSED', flush=True)

    if final_metrics is None:
        final_metrics = _run_eval(total_steps)

    meta = {
        'env': resolved_env,
        'requested_env': requested_env,
        'pretrained_ckpt_dir': str(ckpt_dir),
        'pretrained_epoch': int(epoch),
        'best_eval_json': str(best_eval['eval_json']) if best_eval is not None else '',
        'best_eval_idm': float(best_eval['idm']) if best_eval is not None else None,
        'best_eval_actor': float(best_eval['actor']) if best_eval is not None else None,
        'subgoal_eval_num_samples': eval_n,
        'subgoal_temperature': float(dynamics_config.get('subgoal_temperature', 1.0)),
        'spi_tau': spi_tau,
        'actor_spi_lr': float(FLAGS.actor_spi_lr),
        'actor_spi_steps': int(total_steps),
        'actor_spi_batch_size': int(batch_size),
        'seed': seed,
        'actor_checkpoint': f'checkpoints/actor/params_{total_steps}.pkl',
        'param_diff': {
            'actor_max_abs': actor_diff['max_abs'],
            'critic_max_abs': critic_diff['max_abs'],
            'dynamics_max_abs': dynamics_diff['max_abs'],
        },
        'final_eval': {
            'actor_success_rate_mean': float(final_metrics.get('eval/success_rate_mean', float('nan'))),
            'idm_success_rate_mean': float(final_metrics.get('eval_idm/success_rate_mean', float('nan'))),
        },
    }
    (out_dir / 'actor_spi_meta.yaml').write_text(
        yaml.safe_dump(meta, sort_keys=False, default_flow_style=False), encoding='utf-8'
    )
    csv_logger.close() if hasattr(csv_logger, 'close') else None
    print(
        f"[actor_spi] FINAL actor={meta['final_eval']['actor_success_rate_mean']:.4f} "
        f"idm={meta['final_eval']['idm_success_rate_mean']:.4f}",
        flush=True,
    )
    print(f'[actor_spi] DONE outputs={out_dir}', flush=True)


if __name__ == '__main__':
    app.run(main)
