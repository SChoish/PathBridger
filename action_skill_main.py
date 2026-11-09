"""Action-only skill pipeline; every stage is explicit, offline, and separately frozen.

This entry point never imports or creates an environment. It cannot launch a
server job. Use a fixed, trusted donor and explicit local train/validation NPZs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from skill_extensions.action_agent import ActionProgramAgent
from skill_extensions.action_data import (
    ActionWindowSampler,
    add_value_labels,
    dataset_contract,
    load_archive,
    load_goal_contract,
)
from skill_extensions.action_runtime import (
    atomic_json,
    digest,
    import_decoder,
    restore,
    save,
    tree_digest,
)


def root_statistics(agent, sampler, *, batch_size):
    """Streaming float64 training-only moments; no archive-sized root cache."""
    count = 0
    mean = np.zeros(agent.config['latent_dim'], np.float64)
    m2 = np.zeros_like(mean)
    for first in range(0, len(sampler.starts), batch_size):
        starts = sampler.starts[first : first + batch_size]
        # No goal draws: fitting preprocessing must not consume the training RNG.
        actions = np.asarray(
            sampler.dataset['actions'][starts[:, None] + np.arange(20)], np.float32
        )
        root = np.asarray(agent.encode_actions(actions), np.float64)
        if not np.isfinite(root).all():
            raise FloatingPointError('nonfinite root statistics')
        batch_count = len(root)
        batch_mean = root.mean(0)
        delta = batch_mean - mean
        total = count + batch_count
        m2 += ((root - batch_mean) ** 2).sum(0) + delta**2 * count * batch_count / total
        mean += delta * batch_count / total
        count = total
    return mean.astype(np.float32), np.maximum(np.sqrt(m2 / count), 1e-3).astype(np.float32)


def check_gate(path, expected_sha256, agent, checkpoint_sha256):
    if not path or not expected_sha256 or digest(path) != expected_sha256:
        raise ValueError('formal proposal learning requires an exact SHA-pinned G1 report')
    report = json.loads(Path(path).read_text())
    expected = tree_digest({name: agent.state.params[name] for name in ('encoder', 'reader')})
    if (
        report.get('status') != 'pass'
        or report.get('representation_parameter_sha256') != expected
        or report.get('representation_checkpoint_sha256') != checkpoint_sha256
        or not report.get('evidence')
    ):
        raise ValueError('G1 report does not validate this representation checkpoint')
    return {'sha256': expected_sha256, 'report': report}


def run_train(args):
    if args.updates < 1 or args.batch_size < 1 or args.checkpoint_every < 1:
        raise ValueError('positive training budgets required')
    if args.smoke and args.updates > 3:
        raise ValueError('smoke permits at most three updates per stage')
    if not args.smoke and jax.default_backend() != 'gpu':
        raise RuntimeError('formal training requires GPU; CPU is for bounded --smoke checks')
    if args.stage == 'decoder':
        from skill_extensions.action_decoder import run_decoder_train

        return run_decoder_train(args)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError('use a fresh output directory, including for resumed runs')
    train, validation = load_archive(args.train_npz), load_archive(args.validation_npz)
    identity = dataset_contract(train, validation)
    goal_config = json.loads(Path(args.goal_config).read_text()) if args.goal_config else None
    if args.stage != 'representation' and goal_config is None:
        raise ValueError('proposal/value require a pinned goal sampler configuration')
    if args.stage == 'value' and (not args.goal_contract or not args.boundary_kind):
        raise ValueError('Q requires both goal contract and explicit true-terminal semantics')
    sampler = ActionWindowSampler(
        train,
        seed=args.seed,
        goal_config=goal_config,
        boundary_kind=args.boundary_kind,
    )
    validation_sampler = ActionWindowSampler(
        validation,
        seed=args.seed + 100000,
        goal_config=goal_config,
        boundary_kind=args.boundary_kind,
    )
    if bool(args.input_checkpoint) != bool(args.input_sha256):
        raise ValueError('input checkpoint and its SHA256 must be supplied together')
    previous = None
    if args.input_checkpoint:
        agent, previous = restore(args.input_checkpoint, expected_sha256=args.input_sha256)
        asset = previous['asset']
        if args.config:
            raise ValueError('input checkpoints carry their immutable model configuration')
        for key, expected in (
            ('seed', args.seed),
            ('smoke', args.smoke),
            ('datasets', identity),
            ('batch_size', args.batch_size),
        ):
            if previous['metadata'].get(key) != expected:
                raise ValueError(f'checkpoint/data contract changed: {key}')
    else:
        if args.stage != 'representation' or not args.donor or not args.donor_sha256:
            raise ValueError('new representation requires an explicit decoder donor and SHA256')
        asset = import_decoder(args.donor, args.donor_sha256, allow_smoke=args.smoke)
        config = json.loads(Path(args.config).read_text()) if args.config else {}
        agent = ActionProgramAgent.create(asset, seed=args.seed, config=config)
    if (
        train['observations'].shape[1] != agent.observation_dim
        or validation['observations'].shape[1] != agent.observation_dim
        or train['actions'].shape[1] != agent.action_dim
        or validation['actions'].shape[1] != agent.action_dim
    ):
        raise ValueError('archive dimensions differ from frozen decoder contract')
    gate = previous['metadata'].get('gate') if previous else None
    if args.stage == 'proposal' and agent.stage == 'representation':
        gate = (
            {'smoke_only_bypass': True}
            if args.smoke
            else check_gate(args.gate_report, args.gate_sha256, agent, args.input_sha256)
        )
        mean, scale = root_statistics(agent, sampler, batch_size=args.batch_size)
        agent = agent.seal_representation(mean, scale)
    if args.stage == 'value' and agent.stage == 'proposal':
        agent = agent.seal_proposal()
    if agent.stage != args.stage:
        raise ValueError(f'cannot transition {agent.stage} -> {args.stage}')
    goal_predicate, goal_contract = None, None
    if args.stage == 'value':
        goal_predicate, goal_contract = load_goal_contract(args.goal_contract, smoke=args.smoke)
        if goal_contract['environment'] != goal_config['env_name']:
            raise ValueError('goal predicate and sampler environment differ')
    same_stage = previous is not None and previous['stage'] == args.stage
    if same_stage:
        for key, expected in (
            ('goal_config', goal_config),
            ('goal_contract', goal_contract),
            ('boundary_kind', args.boundary_kind),
        ):
            if previous['metadata'].get(key) != expected:
                raise ValueError(f'resume contract changed: {key}')
        sampler.set_rng_state(previous['sampler_rng'])
    metadata = dict(
        seed=args.seed,
        smoke=args.smoke,
        datasets=identity,
        batch_size=args.batch_size,
        goal_config=goal_config,
        goal_contract=goal_contract,
        boundary_kind=args.boundary_kind,
        gate=gate,
        environment_steps=0,
        parent_initialization='fresh action-only encoder; no donor parent weights/optimizer',
        decoder_parameter_sha256=asset['decoder_parameter_sha256'],
        label_contract='posterior-action-code factual-window approximation',
    )
    current = int(getattr(agent.state, f'{args.stage}_step'))
    if current >= args.updates:
        raise ValueError('checkpoint has already reached or exceeded the requested stage budget')
    output.mkdir(parents=True)
    manifest = {**metadata, 'stage': args.stage, 'status': 'running', 'history': []}
    atomic_json(output / 'manifest.json', manifest)
    try:
        for index in range(current, args.updates):
            batch = sampler.sample(args.batch_size)
            if args.stage == 'value':
                batch = add_value_labels(
                    batch, goal_predicate, goal_contract['parameters'], goal_config['discount']
                )
            batch = {name: jnp.asarray(value) for name, value in batch.items()}
            agent, metrics = agent.update(batch, stage=args.stage)
            row = {'step': index + 1, **{key: float(value) for key, value in metrics.items()}}
            if not all(np.isfinite(value) for value in row.values()):
                raise FloatingPointError('nonfinite action-program training metrics')
            if index == current or (index + 1) % 100 == 0 or index + 1 == args.updates:
                manifest['history'].append(row)
                print(json.dumps(row), flush=True)
            if (index + 1) % args.checkpoint_every == 0 or index + 1 == args.updates:
                receipt = save(
                    output / f'{args.stage}_{index + 1:07d}.pkl',
                    agent,
                    sampler,
                    asset,
                    metadata,
                    smoke=args.smoke,
                )
                manifest['last_checkpoint'] = receipt
                atomic_json(output / 'manifest.json', manifest)
        diagnostics = sampler_diagnostic(agent, validation_sampler, args.batch_size)
        manifest.update(
            status='completed',
            fixed_validation=diagnostics,
            validation_use='fixed diagnostic only; no checkpoint/model selection',
        )
    except BaseException as error:
        manifest.update(status='failed', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        atomic_json(output / 'manifest.json', manifest)
    return manifest


def sampler_diagnostic(agent, sampler, batch_size):
    batch = {name: jnp.asarray(value) for name, value in sampler.sample(batch_size).items()}
    _, metrics = agent.representation_loss(agent.state.params, batch)
    return {key: float(value) for key, value in metrics.items()}


def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--stage', required=True, choices=('decoder', 'representation', 'proposal', 'value')
    )
    parser.add_argument('--train-npz', required=True, type=Path)
    parser.add_argument('--validation-npz', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--donor', type=Path)
    parser.add_argument('--donor-sha256')
    parser.add_argument('--input-checkpoint', type=Path)
    parser.add_argument('--input-sha256')
    parser.add_argument('--config', type=Path)
    parser.add_argument('--goal-config', type=Path)
    parser.add_argument('--goal-contract', type=Path)
    parser.add_argument('--gate-report', type=Path)
    parser.add_argument('--gate-sha256')
    parser.add_argument('--boundary-kind', choices=('timeout', 'terminal', 'field'))
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--updates', type=int, default=20000)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--checkpoint-every', type=int, default=1000)
    parser.add_argument('--smoke', action='store_true')
    report = run_train(parser.parse_args())
    print(json.dumps({'status': report['status'], 'environment_steps': 0}))


if __name__ == '__main__':
    run()
