"""Experimental PB-backed variable skills; official PBF/PBG paths are unchanged.

Use `python skill_main.py train --help` and `evaluate --help`. Training is
offline. Evaluation fixes learned weights and compares subgoal/global modes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import numpy as np

from skill_extensions.planner import SkillPlanner, validate_config
from skill_extensions.runtime import (
    atomic_json,
    checkpoint_payload,
    digest,
    load_pb,
    restore,
    write_checkpoint,
)
from utils.datasets import dataset_identity
from utils.skill_data import SkillDataset, episode_fingerprints, fit_skill_normalization


def train(
    pb,
    train_data,
    val_data,
    *,
    config,
    seed,
    output,
    checkpoint_dir,
    pb_sha256,
    representation_updates,
    value_updates,
    batch_size=256,
    checkpoint_every=1000,
    smoke=False,
    resume=None,
):
    """Two explicit optimizer scopes. No environment use, best-epoch selection, or online updates."""
    output, checkpoint_dir = Path(output), Path(checkpoint_dir)
    config = validate_config(config)
    if min(representation_updates, value_updates, batch_size, checkpoint_every) < 1:
        raise ValueError('positive fixed budgets required')
    if smoke and max(representation_updates, value_updates) > 3:
        raise ValueError('smoke permits at most three updates per stage')
    if not smoke and jax.default_backend() != 'gpu':
        raise RuntimeError('formal training requires GPU; use --smoke for bounded numerical checks')
    train_id, val_id = dataset_identity(train_data), dataset_identity(val_data)
    if train_id['sha256'] == val_id['sha256']:
        raise ValueError('training and validation datasets are identical')
    if episode_fingerprints(train_data) & episode_fingerprints(val_data):
        raise ValueError('training and validation contain identical complete episodes')
    sampler = SkillDataset(train_data, pb.config, config, seed=seed)
    validation = SkillDataset(val_data, pb.config, config, seed=seed + 100000)
    mean, scale = fit_skill_normalization(train_data)
    action_dim, observation_dim = (
        train_data['actions'].shape[1],
        train_data['observations'].shape[1],
    )
    metadata = dict(
        seed=seed,
        pb_sha256=pb_sha256,
        smoke=smoke,
        observation_dim=observation_dim,
        action_dim=action_dim,
        train_identity=train_id,
        validation_identity=val_id,
        environment=str(pb.config['env_name']),
        environment_steps=0,
        label_contract='posterior-code factual-window approximation; PB terminal value frozen',
        inference_modes=['subgoal', 'global'],
        shared_training=True,
        representation_updates=representation_updates,
        value_updates=value_updates,
        batch_size=batch_size,
    )
    if resume:
        agent, payload = restore(resume, pb, pb_sha256, require_trained=False)
        for key in (
            'seed',
            'smoke',
            'train_identity',
            'validation_identity',
            'batch_size',
            'representation_updates',
            'value_updates',
        ):
            if payload[key] != metadata[key]:
                raise ValueError(f'resume contract changed: {key}')
        if (
            dict(agent.config) != config
            or not np.array_equal(agent.mean, mean)
            or not np.array_equal(agent.scale, scale)
        ):
            raise ValueError('resume model/fit normalization changed')
        sampler.set_rng_state(payload['sampler_rng'])
    else:
        if output.exists() or checkpoint_dir.exists():
            raise FileExistsError('new skill run needs fresh directories')
        agent = SkillPlanner.create(
            pb, observation_dim, action_dim, mean, scale, seed=seed, config=config
        )
    if (
        int(agent.state.representation_step) > representation_updates
        or int(agent.state.value_step) > value_updates
    ):
        raise ValueError('checkpoint exceeds requested budget')
    if (
        int(agent.state.value_step)
        and int(agent.state.representation_step) != representation_updates
    ):
        raise ValueError('cannot modify the encoder after value/proposal learning has started')
    output.mkdir(parents=True, exist_ok=True)
    report = {**metadata, 'config': config, 'status': 'running', 'history': [], 'checkpoints': []}
    atomic_json(output / 'manifest.json', report)
    try:
        for stage, budget in (('representation', representation_updates), ('value', value_updates)):
            current = int(
                agent.state.representation_step
                if stage == 'representation'
                else agent.state.value_step
            )
            for index in range(current, budget):
                batch = sampler.sample(batch_size)
                agent, metrics = agent.update(batch, stage=stage)
                values = {k: float(v) for k, v in metrics.items()}
                if not all(np.isfinite(v) for v in values.values()):
                    raise FloatingPointError('nonfinite skill training metrics')
                if index == 0 or (index + 1) % 100 == 0 or index + 1 == budget:
                    row = {'stage': stage, 'step': index + 1, **values}
                    report['history'].append(row)
                    print(json.dumps(row), flush=True)
                if (index + 1) % checkpoint_every == 0 or index + 1 == budget:
                    name = f'repr{int(agent.state.representation_step):07d}_value{int(agent.state.value_step):07d}.pkl'
                    receipt = write_checkpoint(
                        checkpoint_dir / name,
                        checkpoint_payload(agent, sampler, metadata),
                        smoke=smoke,
                    )
                    report['checkpoints'].append(receipt)
                    atomic_json(output / 'manifest.json', report)
        # A fixed final diagnostic batch; not a model/hyperparameter selection set.
        batch = validation.sample(batch_size)
        _, repr_metrics = agent.representation_loss(
            agent.state.params, batch, jax.random.PRNGKey(seed)
        )
        _, value_metrics = agent.value_loss(agent.state.params, batch, jax.random.PRNGKey(seed))
        report.update(
            status='completed',
            validation_representation={k: float(v) for k, v in repr_metrics.items()},
            validation_value={k: float(v) for k, v in value_metrics.items()},
            validation_use='fixed final diagnostic; no checkpoint selection',
        )
    except BaseException as error:
        report.update(
            status='failed',
            error_type=type(error).__name__,
            representation_step=int(agent.state.representation_step),
            value_step=int(agent.state.value_step),
        )
        raise
    finally:
        atomic_json(output / 'manifest.json', report)
    return agent, report


class TraceEnv:
    """Read-only-policy evaluation wrapper with an explicit all-arm step cap."""

    def __init__(self, env, output, step_cap):
        self.env, self.output, self.step_cap = env, Path(output), step_cap
        self.steps, self.index = 0, 0
        self.observations, self.actions, self.successes = [], [], []
        self.goals = None

    def __getattr__(self, name):
        return getattr(self.env, name)

    def flush(self):
        if self.observations:
            path = self.output / f'episode_{self.index:06d}.npz'
            np.savez_compressed(
                path,
                observations=np.asarray(self.observations),
                actions=np.asarray(self.actions),
                successes=np.asarray(self.successes),
                goal=self.goals,
            )
            self.index += 1
            self.observations, self.actions, self.successes = [], [], []

    def reset(self, **kwargs):
        self.flush()
        obs, info = self.env.reset(**kwargs)
        self.observations = [np.asarray(obs).copy()]
        self.goals = np.asarray(info['goal']).copy()
        return obs, info

    def step(self, action):
        if self.steps >= self.step_cap:
            raise RuntimeError('evaluation policy step budget exhausted before env.step')
        self.steps += 1
        obs, reward, term, trunc, info = self.env.step(action)
        self.observations.append(np.asarray(obs).copy())
        self.actions.append(np.asarray(action).copy())
        self.successes.append(bool(info.get('success', False)))
        return obs, reward, term, trunc, info


class DiagnosticPolicy:
    def __init__(self, agent):
        self.agent, self.history = agent, []

    def sample_action_chunks(self, observations, goals, seed, num_candidates, temperature):
        actions, info = self.agent.plan(
            observations, goals, seed, num_candidates=num_candidates, temperature=temperature
        )
        scores = np.asarray(info['scores'])
        if not np.isfinite(scores).all() or not np.isfinite(np.asarray(actions)).all():
            raise FloatingPointError('nonfinite skill score/action')
        self.history.append(
            {
                'duration': np.asarray(info['selected_duration']).tolist(),
                'index': np.asarray(info['selected']).tolist(),
                'score': scores.max(axis=1).tolist(),
                'skill_goal': np.asarray(info['selected_goal']).tolist(),
            }
        )
        return actions


def evaluate(
    agent,
    env,
    output,
    *,
    seed,
    episodes,
    task_ids,
    step_cap,
    include_pb=True,
    modes=('subgoal', 'global'),
):
    from utils.evaluation import _max_episode_steps, evaluate as pb_evaluate

    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    if (
        episodes < 1
        or not task_ids
        or not modes
        or any(m not in ('subgoal', 'global') for m in modes)
    ):
        raise ValueError('positive episode/task counts and explicit subgoal/global modes required')
    modes = list(modes) + (['pb'] if include_pb else [])
    required = len(modes) * len(task_ids) * episodes * _max_episode_steps(env)
    if step_cap < required:
        raise ValueError(f'all-arm evaluation requires a cap of at least {required}')
    output.mkdir(parents=True)
    wrapper = TraceEnv(env, output, step_cap)
    result = {
        'status': 'running',
        'seed': seed,
        'required_policy_step_upper_bound': required,
        'arms': {},
        'training_updates': 0,
        'planning_horizon_is_not_execution_horizon': True,
    }
    atomic_json(output / 'result.json', result)
    try:
        for mode in modes:
            records = []
            policy = agent.pb if mode == 'pb' else DiagnosticPolicy(agent.replace(mode=mode))
            metrics = pb_evaluate(
                policy,
                wrapper,
                task_ids=task_ids,
                episodes_per_task=episodes,
                num_candidates=agent.config['candidate_count'],
                temperature=agent.config['temperature'],
                seed=seed,
                protocol='deterministic-v1',
                episode_records=records,
            )
            wrapper.flush()
            result['arms'][mode] = {
                'metrics': metrics,
                'episodes': records,
                'plans': getattr(policy, 'history', []),
            }
            atomic_json(output / 'result.json', result)
        result.update(
            status='completed',
            policy_steps=wrapper.steps,
            budget_excludes='environment-internal reset/render work',
            comparison='same skill candidate budget; subgoal has extra PB endpoint selection work',
        )
    except BaseException as error:
        result.update(status='failed', error_type=type(error).__name__, policy_steps=wrapper.steps)
        raise
    finally:
        wrapper.flush()
        atomic_json(output / 'result.json', result)
    return result


def run():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('train')
    p.add_argument(
        '--config', type=Path, default=Path(__file__).parent / 'skill_configs/common.json'
    )
    p.add_argument('--pb-bundle', type=Path, required=True)
    p.add_argument('--pb-sha256', required=True)
    p.add_argument('--dataset-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--checkpoint-dir', type=Path, required=True)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--representation-updates', type=int, default=20000)
    p.add_argument('--value-updates', type=int, default=20000)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--checkpoint-every', type=int, default=1000)
    p.add_argument('--resume', type=Path)
    p.add_argument('--smoke', action='store_true')
    p = commands.add_parser('evaluate')
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--pb-bundle', type=Path, required=True)
    p.add_argument('--pb-sha256', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--episodes', type=int, default=50)
    p.add_argument('--task-ids', default='1,2,3,4,5')
    p.add_argument('--step-cap', type=int, required=True)
    p.add_argument('--skip-pb', action='store_true')
    p.add_argument('--mode', choices=('both', 'subgoal', 'global'), default='both')
    args = parser.parse_args()
    jax.config.update('jax_default_matmul_precision', 'highest')
    pb, spec = load_pb(args.pb_bundle, args.pb_sha256)
    if args.command == 'train':
        # No implicit downloads: formal users supply their existing local archive.
        from envs.env_utils import _local_npz_paths, make_env_and_datasets

        if not _local_npz_paths(spec['config']['env_name'], args.dataset_dir):
            raise ValueError(
                'local train/validation NPZ archive required; this command will not download data'
            )
        env, train_data, val_data = make_env_and_datasets(
            spec['config']['env_name'], args.dataset_dir
        )
        env.close()
        for data in (train_data, val_data):
            if (
                data['observations'].shape[-1] != spec['observation_dim']
                or data['actions'].shape[-1] != spec['action_dim']
            ):
                raise ValueError('local archive dimensions do not match the pinned PB bundle')
        _, report = train(
            pb,
            train_data,
            val_data,
            config=json.loads(args.config.read_text()),
            seed=args.seed,
            output=args.output,
            checkpoint_dir=args.checkpoint_dir,
            pb_sha256=args.pb_sha256,
            representation_updates=3 if args.smoke else args.representation_updates,
            value_updates=3 if args.smoke else args.value_updates,
            batch_size=args.batch_size,
            checkpoint_every=args.checkpoint_every,
            smoke=args.smoke,
            resume=args.resume,
        )
    else:
        import ogbench

        agent, payload = restore(args.checkpoint, pb, args.pb_sha256)
        env = ogbench.make_env_and_datasets(payload['environment'], env_only=True)
        try:
            report = evaluate(
                agent,
                env,
                args.output,
                seed=args.seed,
                episodes=args.episodes,
                task_ids=tuple(map(int, args.task_ids.split(','))),
                step_cap=args.step_cap,
                include_pb=not args.skip_pb,
                modes=('subgoal', 'global') if args.mode == 'both' else (args.mode,),
            )
            report.update(checkpoint_sha256=digest(args.checkpoint), pb_sha256=args.pb_sha256)
            atomic_json(args.output / 'result.json', report)
        finally:
            env.close()
    print(json.dumps({'status': report['status']}))


if __name__ == '__main__':
    run()
