"""Evaluate one unified PathBridger checkpoint on the five OGBench tasks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from absl import app, flags

from envs.env_utils import make_env_and_datasets
from utils.config import (
    config_sha256,
    load_config,
    normalize_packaged_file_flag,
)
from utils.datasets import dataset_identity
from utils.evaluation import (
    DEFAULT_TASK_IDS,
    HISTORICAL_EVALUATION_PROTOCOL,
    evaluate,
    evaluation_protocol_metadata,
    normalize_evaluation_protocol,
)
from utils.flax_utils import resolve_checkpoint, restore_agent
from utils.runtime import build_state_runtime

FLAGS = flags.FLAGS
_DEFAULT_CONFIG = str(
    Path(__file__).resolve().parent / 'configs' / 'pbf_antmaze_medium.yaml'
)

flags.DEFINE_string('checkpoint_dir', '', 'Checkpoint directory or exact .pkl file.')
flags.DEFINE_integer(
    'checkpoint_step',
    0,
    'Checkpoint step; required for a directory and inferred from an exact params_<step>.pkl file.',
)
flags.DEFINE_string('dataset_dir', '', 'Optional OGBench dataset directory.')
flags.DEFINE_integer('episodes', 50, 'Episodes for each of the five predefined tasks.')
flags.DEFINE_integer('seed', 0, 'Evaluation seed.')
flags.DEFINE_string(
    'task_ids',
    ','.join(str(task_id) for task_id in DEFAULT_TASK_IDS),
    'Comma-separated OGBench task ids in evaluation order.',
)
flags.DEFINE_string(
    'eval_protocol',
    HISTORICAL_EVALUATION_PROTOCOL,
    'Evaluation contract: historical-v1 (default) or deterministic-v1.',
)
flags.DEFINE_string('output_path', '', 'Optional JSON result path.')
flags.DEFINE_string(
    'trace_path',
    '',
    'Optional JSON path for per-episode reset/result hashes.',
)
flags.DEFINE_multi_string(
    'config_override', [], 'Highest-priority config key=value override; repeatable.'
)

flags.DEFINE_string(
    'agent',
    _DEFAULT_CONFIG,
    'PathBridger paper-reproduction config used by the checkpoint.',
)


def _file_identity(path: Path, *, role: str) -> dict:
    digest = hashlib.sha256()
    with path.open('rb') as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(chunk)
    return {
        'role': role,
        'path': str(path.resolve()),
        'size_bytes': path.stat().st_size,
        'sha256': digest.hexdigest(),
    }


def _aggregation_config_sha256(config) -> str:
    values = config.to_dict() if hasattr(config, 'to_dict') else dict(config)
    values.pop('eval_num_candidates', None)
    values.pop('eval_temperature', None)
    return config_sha256(values)


def main(_):
    if not FLAGS.checkpoint_dir:
        raise ValueError('checkpoint_dir is required.')
    if FLAGS.checkpoint_step < 0:
        raise ValueError('checkpoint_step cannot be negative.')
    if FLAGS.episodes < 1:
        raise ValueError('episodes must be at least 1.')
    task_ids = tuple(
        int(token.strip())
        for token in FLAGS.task_ids.split(',')
        if token.strip()
    )
    if not task_ids:
        raise ValueError('task_ids must contain at least one task id.')
    checkpoint_path, checkpoint_step = resolve_checkpoint(
        FLAGS.checkpoint_dir,
        FLAGS.checkpoint_step,
    )
    checkpoint_files = [_file_identity(checkpoint_path, role='unified')]
    checkpoint_kind = 'unified'

    config = load_config(
        FLAGS.agent,
        cli_overrides=FLAGS.config_override,
    )
    eval_protocol = normalize_evaluation_protocol(FLAGS.eval_protocol)
    env, train_data, _ = make_env_and_datasets(
        str(config.env_name),
        dataset_dir=FLAGS.dataset_dir or None,
    )
    agent, _, _ = build_state_runtime(train_data, config, seed=FLAGS.seed)
    agent, checkpoint_restore = restore_agent(
        agent,
        FLAGS.checkpoint_dir,
        FLAGS.checkpoint_step,
        return_info=True,
        inference_only=True,
    )

    episode_records = [] if FLAGS.trace_path else None
    metrics = evaluate(
        agent,
        env,
        task_ids=task_ids,
        episodes_per_task=FLAGS.episodes,
        num_candidates=int(config.eval_num_candidates),
        temperature=float(config.eval_temperature),
        seed=FLAGS.seed,
        protocol=eval_protocol,
        episode_records=episode_records,
    )
    effective_evaluation = evaluation_protocol_metadata(
        env,
        protocol=eval_protocol,
        task_ids=task_ids,
        episodes_per_task=FLAGS.episodes,
        num_candidates=int(config.eval_num_candidates),
        temperature=float(config.eval_temperature),
        seed=FLAGS.seed,
    )
    result = {
        'result_schema_version': 'pathbridger-evaluation-result/v2',
        'checkpoint_step': checkpoint_step,
        'checkpoint_identity': {
            'kind': checkpoint_kind,
            'files': checkpoint_files,
        },
        'dataset_identity': dataset_identity(train_data),
        'env_name': str(config.env_name),
        'endpoint_distribution': str(config.endpoint_distribution),
        'prefix_model': str(config.prefix_model),
        'eval_prefix_selection': str(config.eval_prefix_selection),
        'eval_num_prefix_samples': int(config.eval_num_prefix_samples),
        'eval_prefix_temperature': float(config.eval_prefix_temperature),
        'eval_include_deterministic_prefix': bool(
            config.eval_include_deterministic_prefix
        ),
        'prefix_rank': int(config.prefix_rank),
        'prefix_flow_steps': int(config.prefix_flow_steps),
        'eval_num_candidates': int(config.eval_num_candidates),
        'eval_temperature': float(config.eval_temperature),
        'config_sha256': config_sha256(config),
        'aggregation_config_sha256': _aggregation_config_sha256(config),
        'checkpoint_restore': checkpoint_restore,
        'evaluation_protocol_version': eval_protocol,
        'effective_evaluation': effective_evaluation,
        **{f'evaluation/{key}': value for key, value in metrics.items()},
    }
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if FLAGS.output_path:
        output_path = Path(FLAGS.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open('w', encoding='utf-8') as file:
            file.write(text)
            file.write('\n')
    if FLAGS.trace_path:
        trace_path = Path(FLAGS.trace_path)
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_payload = {
            'evaluation_protocol_version': eval_protocol,
            'effective_evaluation': effective_evaluation,
            'episodes': episode_records,
        }
        with trace_path.open('w', encoding='utf-8') as file:
            json.dump(trace_payload, file, indent=2, sort_keys=True)
            file.write('\n')


def run():
    """Run the command-line evaluation entry point."""

    normalize_packaged_file_flag(
        'agent', package_root=Path(__file__).resolve().parent
    )
    app.run(main)


if __name__ == '__main__':
    run()
