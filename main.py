"""Train PathBridger on a fixed state-based OGBench dataset."""

from __future__ import annotations

import json
import os
import random
import signal
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import jax
import numpy as np
import tqdm
from absl import app, flags

from envs.env_utils import make_env_and_datasets
from utils.config import (
    load_config,
    normalize_packaged_file_flag,
    write_resolved_config,
)
from utils.datasets import action_free_view
from utils.evaluation import (
    DEFAULT_TASK_IDS,
    HISTORICAL_EVALUATION_PROTOCOL,
    evaluate,
    evaluation_protocol_metadata,
    normalize_evaluation_protocol,
)
from utils.flax_utils import resolve_checkpoint, restore_agent, save_agent
from utils.log_utils import CsvLogger, get_exp_name, get_flag_dict, setup_wandb
from utils.prefetch import resolve_stop_boundary, submit_prefetched_batch
from utils.runtime import build_state_runtime

FLAGS = flags.FLAGS

_BATCH_SIZE = 1024
_DEFAULT_CONFIG = str(
    Path(__file__).resolve().parent / 'configs' / 'pbf_antmaze_medium.yaml'
)

flags.DEFINE_string('run_group', 'Debug', 'Experiment group.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('save_dir', 'exp/', 'Root output directory.')
flags.DEFINE_string('dataset_dir', '', 'Optional OGBench dataset directory.')
flags.DEFINE_string('restore_path', '', 'Checkpoint directory or exact .pkl file.')
flags.DEFINE_integer(
    'restore_step',
    0,
    'Checkpoint step; required for a directory and inferred from an exact params_<step>.pkl file.',
)
flags.DEFINE_string(
    'run_dir',
    '',
    'Optional existing experiment directory; when set, continue logging/checkpoints there.',
)
flags.DEFINE_integer('train_steps', 1_000_000, 'Number of gradient updates.')
flags.DEFINE_integer('log_interval', 5_000, 'Training CSV/W&B logging interval.')
flags.DEFINE_integer(
    'eval_interval', 0, 'In-process evaluation interval; zero disables it.'
)
flags.DEFINE_integer('save_interval', 100_000, 'Checkpoint interval; 0 saves only the final checkpoint.')
flags.DEFINE_string(
    'extra_save_steps',
    '',
    'Comma-separated extra checkpoint steps (e.g. 800000,900000,1000000).',
)
flags.DEFINE_integer(
    'eval_episodes', 50, 'Episodes per task for in-process evaluation.'
)
flags.DEFINE_string(
    'eval_protocol',
    HISTORICAL_EVALUATION_PROTOCOL,
    'Evaluation contract: historical-v1 (default) or deterministic-v1.',
)
flags.DEFINE_boolean('use_wandb', False, 'Enable optional Weights & Biases logging.')
flags.DEFINE_boolean('use_tqdm', True, 'Show a training progress bar.')
flags.DEFINE_boolean('async_prefetch', True, 'Overlap host batch sampling with accelerator work.')
flags.DEFINE_multi_string(
    'config_override', [], 'Highest-priority config key=value override; repeatable.'
)

flags.DEFINE_string(
    'agent',
    _DEFAULT_CONFIG,
    'PathBridger agent and paper-reproduction environment config.',
)


def _host_metrics(info) -> dict[str, float]:
    metrics = {}
    for key, value in info.items():
        array = np.asarray(jax.device_get(value))
        if array.size != 1:
            raise ValueError(f'Training metric {key!r} must be scalar, got shape {array.shape}.')
        metrics[str(key)] = float(array.reshape(()))
    return metrics


def _write_run_config(
    run_dir: str,
    config,
    state_scale: np.ndarray,
    initialization: dict[str, object],
) -> None:
    config_path = os.path.join(run_dir, 'flags.json')
    payload = {
        'flags': get_flag_dict(),
        'agent': config.to_dict() if hasattr(config, 'to_dict') else dict(config),
        'state_scale': np.asarray(state_scale, dtype=np.float32).tolist(),
        'initialization': initialization,
    }
    if initialization.get('mode') == 'unified_resume' and os.path.isfile(
        config_path
    ):
        with open(config_path, encoding='utf-8') as file:
            previous = json.load(file)
        origin = previous.get('initialization')
        payload['initialization'] = origin
        payload['resume'] = initialization
    with open(config_path, 'w', encoding='utf-8') as file:
        json.dump(payload, file, indent=2, sort_keys=True)
        file.write('\n')


def _parse_extra_save_steps(raw: str) -> frozenset[int]:
    text = (raw or '').strip()
    if not text:
        return frozenset()
    steps = []
    for part in text.split(','):
        token = part.strip()
        if not token:
            continue
        steps.append(int(token))
    return frozenset(steps)


def _validate_runtime_flags() -> None:
    if FLAGS.train_steps < 1:
        raise ValueError('train_steps must be at least 1.')
    if FLAGS.restore_step < 0:
        raise ValueError('restore_step cannot be negative.')
    if FLAGS.restore_step and not FLAGS.restore_path:
        raise ValueError('restore_step requires restore_path.')
    if FLAGS.log_interval < 1:
        raise ValueError('log_interval must be at least 1.')
    if FLAGS.eval_interval < 0:
        raise ValueError('eval_interval cannot be negative.')
    if FLAGS.save_interval < 0:
        raise ValueError('save_interval cannot be negative.')
    extra_save_steps = _parse_extra_save_steps(FLAGS.extra_save_steps)
    if any(step < 1 or step > FLAGS.train_steps for step in extra_save_steps):
        raise ValueError('extra_save_steps must be in [1, train_steps].')
    if FLAGS.eval_episodes < 1:
        raise ValueError('eval_episodes must be at least 1.')


def main(_):
    _validate_runtime_flags()
    eval_protocol = normalize_evaluation_protocol(FLAGS.eval_protocol)
    extra_save_steps = _parse_extra_save_steps(FLAGS.extra_save_steps)
    config = load_config(
        FLAGS.agent,
        cli_overrides=FLAGS.config_override,
    )

    restore_step = 0
    if FLAGS.restore_path:
        _, restore_step = resolve_checkpoint(
            FLAGS.restore_path,
            FLAGS.restore_step,
        )
        if restore_step >= FLAGS.train_steps:
            raise ValueError('The restored step must be smaller than train_steps.')

    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    env, train_data, _ = make_env_and_datasets(
        str(config.env_name),
        dataset_dir=FLAGS.dataset_dir or None,
    )
    action_free = bool(config.get('offline_action_free', False))
    if action_free:
        train_data = action_free_view(train_data)
    agent, dataset, state_scale = build_state_runtime(
        train_data, config, seed=FLAGS.seed
    )
    if FLAGS.restore_path:
        agent, restore_info = restore_agent(
            agent,
            FLAGS.restore_path,
            FLAGS.restore_step,
            sampler=dataset,
            return_info=True,
        )
        initialization = {
            'mode': 'unified_resume',
            'checkpoint': str(
                resolve_checkpoint(FLAGS.restore_path, FLAGS.restore_step)[0]
            ),
            'step': restore_step,
            'restore_contract': restore_info,
        }
    else:
        initialization = {'mode': 'random'}

    if FLAGS.run_dir:
        run_dir = os.path.abspath(FLAGS.run_dir)
        exp_name = os.path.basename(run_dir)
    else:
        exp_name = get_exp_name(
            FLAGS.seed,
            env_name=str(config.env_name),
            agent_name=(
            f'pathbridger_{config.endpoint_distribution}'
        ),
        )
        run_dir = os.path.abspath(
            os.path.join(FLAGS.save_dir, 'pathbridger', FLAGS.run_group, exp_name)
        )
    checkpoint_dir = os.path.join(run_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    _write_run_config(run_dir, config, state_scale, initialization)
    write_resolved_config(
        os.path.join(run_dir, 'resolved_config.json'),
        config,
        sources={
            'preset_yaml': FLAGS.agent,
            'cli': list(FLAGS.config_override),
        },
    )
    if FLAGS.eval_interval > 0:
        protocol_path = os.path.join(run_dir, 'evaluation_protocol.json')
        with open(protocol_path, 'w', encoding='utf-8') as file:
            json.dump(
                evaluation_protocol_metadata(
                    env,
                    protocol=eval_protocol,
                    task_ids=DEFAULT_TASK_IDS,
                    episodes_per_task=FLAGS.eval_episodes,
                    num_candidates=int(config.eval_num_candidates),
                    temperature=float(config.eval_temperature),
                    seed=FLAGS.seed,
                ),
                file,
                indent=2,
                sort_keys=True,
            )
            file.write('\n')

    wandb_run = None
    if FLAGS.use_wandb:
        wandb_run = setup_wandb(
            project='PathBridger',
            group=FLAGS.run_group,
            name=exp_name,
            config={'flags': get_flag_dict(), 'agent': config.to_dict()},
            directory=run_dir,
        )

    start_step = restore_step + 1
    steps = range(start_step, FLAGS.train_steps + 1)
    if FLAGS.use_tqdm:
        steps = tqdm.tqdm(steps, smoothing=0.1, dynamic_ncols=True)

    train_logger = CsvLogger(
        os.path.join(run_dir, 'train.csv'),
        resume=bool(FLAGS.run_dir or FLAGS.restore_path),
    )
    eval_logger = (
        CsvLogger(
            os.path.join(run_dir, 'eval.csv'),
            resume=bool(FLAGS.run_dir or FLAGS.restore_path),
        )
        if FLAGS.eval_interval > 0
        else None
    )
    pool = (
        ThreadPoolExecutor(max_workers=1, thread_name_prefix='pathbridger-prefetch')
        if FLAGS.async_prefetch
        else None
    )

    def submit_batch():
        return submit_prefetched_batch(pool, dataset, _BATCH_SIZE)

    next_batch = None
    start_time = time.time()
    interval_start = start_time
    last_completed_step = restore_step

    # Soft-stop: finish current step, save checkpoint, then exit cleanly.
    stop_requested = {'flag': False, 'signum': None}

    def _request_stop(signum, _frame):
        if stop_requested['flag']:
            return
        stop_requested['flag'] = True
        stop_requested['signum'] = int(signum)
        print(
            f'[signal] signum={signum} — will emergency-save after current step '
            f'(checkpoint_dir={checkpoint_dir})',
            flush=True,
        )

    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)
    try:
        signal.signal(signal.SIGHUP, _request_stop)
    except (ValueError, OSError):
        pass

    def _record_emergency_save(step: int) -> None:
        marker = os.path.join(run_dir, f'EMERGENCY_SAVE_step{step}')
        try:
            with open(marker, 'w', encoding='utf-8') as file:
                file.write(
                    f'signal={stop_requested["signum"]} step={step}\n'
                )
        except OSError:
            pass
        print(
            f'[signal] emergency_save_done step={step} '
            f'signum={stop_requested["signum"]} checkpoint_dir={checkpoint_dir}',
            flush=True,
        )

    try:
        if pool is not None:
            next_batch = submit_batch()
        for step in steps:
            # A signal observed between completed updates must not consume or
            # apply the already-prefetched next batch.  Step zero cannot be
            # represented by the checkpoint API, so a fresh run still
            # completes its first update before emergency-saving.
            stop_before_update, next_batch = resolve_stop_boundary(
                next_batch,
                dataset,
                stop_requested=bool(stop_requested['flag']),
                last_completed_step=last_completed_step,
            )
            if stop_before_update:
                save_agent(
                    agent,
                    checkpoint_dir,
                    last_completed_step,
                    sampler=dataset,
                )
                _record_emergency_save(last_completed_step)
                break

            if next_batch is None:
                batch = dataset.sample(_BATCH_SIZE)
            else:
                batch = next_batch.result()
                next_batch = None

            is_final = step == FLAGS.train_steps
            emergency_stop = bool(stop_requested['flag'])
            do_log = step % FLAGS.log_interval == 0 or is_final or emergency_stop
            do_save = (
                is_final
                or emergency_stop
                or (FLAGS.save_interval > 0 and step % FLAGS.save_interval == 0)
                or step in extra_save_steps
            )
            do_eval = (
                FLAGS.eval_interval > 0
                and not emergency_stop
                and (step % FLAGS.eval_interval == 0 or is_final)
            )
            if pool is not None and not do_save:
                next_batch = submit_batch()

            agent, update_info = agent.update(batch, full_metrics=do_log)
            last_completed_step = step

            emergency_stop = bool(stop_requested['flag'])
            if emergency_stop and next_batch is not None:
                # A signal may arrive after prefetch was submitted but before
                # this update completed. The unused batch must remain the next
                # batch after resume, so roll the sampler back before saving.
                next_batch.rollback(dataset)
                next_batch = None
            do_log = do_log or emergency_stop
            do_eval = do_eval and not emergency_stop
            if do_log:
                train_metrics = _host_metrics(update_info)
                train_metrics['time/interval_seconds'] = time.time() - interval_start
                train_metrics['time/total_seconds'] = time.time() - start_time
                interval_start = time.time()
                train_logger.log(train_metrics, step=step)
                if wandb_run is not None:
                    wandb_run.log({f'training/{key}': value for key, value in train_metrics.items()}, step=step)

            if do_eval:
                eval_metrics = evaluate(
                    agent,
                    env,
                    task_ids=DEFAULT_TASK_IDS,
                    episodes_per_task=FLAGS.eval_episodes,
                    num_candidates=int(config.eval_num_candidates),
                    temperature=float(config.eval_temperature),
                    seed=FLAGS.seed,
                    protocol=eval_protocol,
                )
                eval_metrics['protocol_version'] = eval_protocol
                eval_logger.log(eval_metrics, step=step)
                if wandb_run is not None:
                    wandb_run.log(
                        {f'evaluation/{key}': value for key, value in eval_metrics.items()},
                        step=step,
                    )

            do_save = (
                is_final
                or emergency_stop
                or (FLAGS.save_interval > 0 and step % FLAGS.save_interval == 0)
                or step in extra_save_steps
            )
            if do_save:
                save_agent(agent, checkpoint_dir, step, sampler=dataset)
                if pool is not None and not is_final and not emergency_stop:
                    next_batch = submit_batch()
                if emergency_stop:
                    _record_emergency_save(step)
                    break
    finally:
        if pool is not None:
            pool.shutdown(wait=True)
        train_logger.close()
        if eval_logger is not None:
            eval_logger.close()
        if wandb_run is not None:
            wandb_run.finish()

    print(f'Run saved to {run_dir}')


def run():
    """Run the command-line training entry point."""

    normalize_packaged_file_flag(
        'agent', package_root=Path(__file__).resolve().parent
    )
    app.run(main)


if __name__ == '__main__':
    run()
