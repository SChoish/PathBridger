"""Micro-profile build_actor_proposals to split trajectory sampling vs IDM decode.

Usage:
    PYTHONPATH=. python scripts/profile_proposal_build.py \
        --env_name=antmaze-giant-navigate-v0 --plan_candidates=8 --batch_size=1024 \
        --num_warmup=2 --num_iters=50
"""

from __future__ import annotations

import argparse
import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import yaml

from agents.dynamics import DynamicsAgent, get_dynamics_config
from utils.env_utils import make_env_and_datasets


def _block(x):
    return jax.tree_util.tree_map(
        lambda v: v.block_until_ready() if hasattr(v, 'block_until_ready') else v, x
    )


def _time_loop(name: str, fn, num_warmup: int, num_iters: int):
    for _ in range(num_warmup):
        out = fn()
        _block(out)
    t0 = time.perf_counter()
    for _ in range(num_iters):
        out = fn()
        _block(out)
    dt = (time.perf_counter() - t0) / max(1, num_iters)
    print(f"  {name:35s} {dt*1000:8.3f} ms / call")
    return dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--env_name', default='antmaze-giant-navigate-v0')
    ap.add_argument('--config_yaml',
                    default='runs/20260427_135228_joint_dqc_seed0_antmaze-giant-navigate-v0_resume_pc8_from101011/config_used.yaml',
                    help='YAML providing dynamics: overrides (optional).')
    ap.add_argument('--plan_candidates', type=int, default=8)
    ap.add_argument('--batch_size', type=int, default=1024)
    ap.add_argument('--actor_chunk_horizon', type=int, default=5)
    ap.add_argument('--num_warmup', type=int, default=2)
    ap.add_argument('--num_iters', type=int, default=50)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--noise_scale', type=float, default=0.01)
    args = ap.parse_args()

    print(f"jax.devices() = {jax.devices()}")

    print(f"Loading env/dataset for {args.env_name} (just for example shapes)...")
    env, train_plain, _ = make_env_and_datasets(args.env_name, frame_stack=None)

    rng_np = np.random.default_rng(args.seed)
    obs_arr = np.asarray(train_plain['observations'], dtype=np.float32)
    state_dim = obs_arr.shape[-1]
    obs_idx = rng_np.integers(0, obs_arr.shape[0], size=args.batch_size)
    goal_idx = rng_np.integers(0, obs_arr.shape[0], size=args.batch_size)
    obs_batch = jnp.asarray(obs_arr[obs_idx])
    goals_batch = jnp.asarray(obs_arr[goal_idx])
    print(f"  obs_batch shape = {obs_batch.shape}")

    cfg = get_dynamics_config()
    if args.config_yaml:
        try:
            with open(args.config_yaml) as f:
                yml = yaml.safe_load(f) or {}
            for k in ('dynamics',):
                if isinstance(yml.get(k), dict):
                    for kk, vv in yml[k].items():
                        cfg[kk] = vv
            print(f"Applied dynamics overrides from {args.config_yaml}")
        except FileNotFoundError:
            pass

    cfg['action_dim'] = int(np.asarray(env.action_space.shape).prod())
    cfg['batch_size'] = int(args.batch_size)
    ex_actions = jnp.zeros((1, cfg['action_dim']), dtype=jnp.float32)

    print('Creating agent...')
    dynamics_agent = DynamicsAgent.create(
        seed=args.seed,
        ex_observations=obs_batch[:1],
        config=cfg,
        ex_actions=ex_actions,
    )

    pc = int(args.plan_candidates)
    K = int(args.actor_chunk_horizon)
    rng_key = jax.random.PRNGKey(args.seed)
    print(f"plan_candidates={pc}, actor_chunk_horizon(K)={K}, batch={args.batch_size}")

    # 1) Subgoal forward only
    @jax.jit
    def _subgoal_only(obs, goals):
        return dynamics_agent._subgoal_forward(obs, goals)

    # 2) Plan sampling for pc=1 (single trajectory, K steps)
    @jax.jit
    def _sample_one(obs, mu, rng):
        return dynamics_agent._sample_plan_trajectory(obs, mu, rng, args.noise_scale, num_steps=K)

    # 3) Plan sampling for pc>1 (vmap over candidates, each K steps)
    @partial(jax.jit, static_argnames=('num_candidates',))
    def _sample_pc(obs, mu, rng, num_candidates):
        return dynamics_agent.sample_plan_candidates(
            obs, mu, rng,
            num_candidates=num_candidates,
            noise_scale=args.noise_scale,
            include_mean=True,
            num_steps=K,
        )

    # 4) IDM decode only (input = pre-rolled trajectories shaped [B*N, K+1, D])
    @partial(jax.jit, static_argnames=('horizon',))
    def _idm_only(traj_flat, horizon):
        return dynamics_agent._idm_actions_from_trajectories(traj_flat, horizon)

    # 5) Full build_actor_proposals (= what main.py actually times as t_prop)
    @partial(jax.jit, static_argnames=('proposal_horizon', 'plan_candidates', 'sample_noise_scale'))
    def _full_build(obs, goals, rng, proposal_horizon, plan_candidates, sample_noise_scale):
        return dynamics_agent.build_actor_proposals(
            obs, goals, rng,
            proposal_horizon=proposal_horizon,
            plan_candidates=plan_candidates,
            sample_noise_scale=sample_noise_scale,
        )

    print("\n=== Per-call timing (medians, iters={}) ===".format(args.num_iters))

    print("\n[1] subgoal forward (predict subgoal point)")
    _time_loop('subgoal_net forward', lambda: _subgoal_only(obs_batch, goals_batch),
               args.num_warmup, args.num_iters)

    mu = _block(_subgoal_only(obs_batch, goals_batch))

    print("\n[2] state-trajectory sampling: pc=1 (single trajectory)")
    _time_loop(
        'sample_plan_trajectory(pc=1, K)',
        lambda: _sample_one(obs_batch, mu, rng_key),
        args.num_warmup, args.num_iters,
    )

    print(f"\n[3] state-trajectory sampling: pc={pc} (vmap'd over candidates)")
    sample_pc = _time_loop(
        f'sample_plan_candidates(pc={pc}, K)',
        lambda: _sample_pc(obs_batch, mu, rng_key, pc),
        args.num_warmup, args.num_iters,
    )

    # Pre-roll trajectories for IDM-only timing.
    cand_traj = _block(_sample_pc(obs_batch, mu, rng_key, pc))
    flat_traj = cand_traj.reshape(-1, cand_traj.shape[2], cand_traj.shape[3])
    print(f"\n[4] IDM-only decode (flat shape={tuple(flat_traj.shape)}, B*N={flat_traj.shape[0]})")
    idm_t = _time_loop('idm_actions_from_trajectories', lambda: _idm_only(flat_traj, K),
                       args.num_warmup, args.num_iters)

    print("\n[5] full build_actor_proposals (= main.py t_prop equivalent)")
    full_t = _time_loop(
        f'build_actor_proposals(pc={pc}, K)',
        lambda: _full_build(obs_batch, goals_batch, rng_key, K, pc, args.noise_scale),
        args.num_warmup, args.num_iters,
    )

    spe = 965  # giant valid_starts/batch=1024 => 965 steps/epoch
    print(f"\n=== Aggregated to per-epoch (steps_per_epoch={spe}) ===")
    print(f"  full build_actor_proposals : {full_t * spe:7.3f} s/epoch")
    print(f"  └─ state traj sampling pc={pc}: {sample_pc * spe:7.3f} s/epoch")
    print(f"  └─ IDM decode               : {idm_t * spe:7.3f} s/epoch")
    other = full_t - sample_pc - idm_t
    print(f"  └─ remainder (subgoal+misc) : {other * spe:7.3f} s/epoch")


if __name__ == '__main__':
    main()
