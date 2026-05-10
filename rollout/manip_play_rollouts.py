#!/usr/bin/env python3
"""OGBench **ManipSpace ``*-play-v0``** (cube / puzzle 등) IDM/actor rollout + 영상.

`flags.json` 의 `env_name` 만 보고 cube/puzzle을 자동 인식한다. 한 task마다
IDM rollout과 actor rollout을 모두 실행하며, 각각 success/fail MP4 +
`rollout_task_summary.csv` 를 남긴다 (state-space open-loop plot은 그리지 않음).

성공 판정은 항상 env의 ``info['success']`` (any step) 만으로 결정된다 — 사용자가
정의한 거리 임계치 같은 임의 기준은 사용하지 않는다.

기본 출력: ``<run_dir>/rollouts_manip_<env_slug>_ep<EPOCH>/``

같은 ``out_dir`` 에는 파일 락 (``.manip_play_rollouts.lock``) 으로 한 번에 하나의
프로세스만 실행되게 한다.

실행: ``python -m rollout.manip_play_rollouts --run_dir=...``

JAX: 오프라인 롤아웃은 기본 CPU (``JAX_PLATFORMS`` 미설정 시 여기서 ``cpu`` 로 고정).
GPU로 돌리려면 실행 전 ``export JAX_PLATFORMS=cuda``.
"""

from __future__ import annotations

import csv
import os
from typing import Any

if 'JAX_PLATFORMS' not in os.environ:
    os.environ['JAX_PLATFORMS'] = 'cpu'

import argparse
import contextlib
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from agents.actor import ActorAgent, get_actor_config
from agents.dynamics import DynamicsAgent
from rollout.common import manip_play_family, slug_from_env
from rollout.env import configure_mujoco_gl, max_episode_steps_from_wrappers
from rollout.episode_runner import (
    make_actor_chunk_fn,
    make_idm_chunk_fn,
    run_chunked_episode,
)
from rollout.plot import write_rgb_array_mp4
from rollout.value_field import load_critic_for_run
from utils.env_utils import make_env_and_datasets
from utils.run_io import (
    list_checkpoint_suffixes,
    load_checkpoint_pkl,
    load_run_flags,
    parse_int_list,
    pick_epoch,
    resolve_actor_checkpoint_dir,
    resolve_dynamics_checkpoint_dir,
)
from utils.subgoal_filter import make_value_subgoal_filter_from_critic_agent


def _chunk_budget_for_full_episode(env, chunk_h: int, flags_max_chunks: int) -> int:
    ms = max_episode_steps_from_wrappers(env)
    ch = max(1, int(chunk_h))
    fm = max(1, int(flags_max_chunks))
    if ms is None:
        return fm
    need = (int(ms) + ch - 1) // ch
    return max(fm, int(need) + 2)


def _load_eval_rollout_limits(run_dir: Path) -> tuple[int, int, int]:
    flags_path = run_dir / 'flags.json'
    with open(flags_path, 'r', encoding='utf-8') as f:
        root = json.load(f)
    fg = root.get('flags') if isinstance(root.get('flags'), dict) else root
    max_chunks = max(1, int(fg.get('eval_max_chunks', 200)))
    critic = root.get('critic_agent') if isinstance(root.get('critic_agent'), dict) else {}
    actor = root.get('actor') if isinstance(root.get('actor'), dict) else {}
    idm_h = max(1, int(critic.get('action_chunk_horizon', 5)))
    act_h = max(1, int(actor.get('actor_chunk_horizon', idm_h)))
    return max_chunks, idm_h, act_h


def _load_actor_cfg(flags_path: Path) -> dict:
    with open(flags_path, 'r', encoding='utf-8') as f:
        root = json.load(f)
    act = root.get('actor')
    if not isinstance(act, dict):
        raise KeyError(f'{flags_path} must contain an "actor" object.')
    base = get_actor_config()
    for k, v in act.items():
        base[k] = v
    return dict(base)


def _pad_rgb_frames_min_duration(frames: np.ndarray, fps: float, min_seconds: float) -> np.ndarray:
    if frames is None or frames.size == 0 or min_seconds <= 0 or fps <= 0:
        return frames
    n = int(frames.shape[0])
    need = int(np.ceil(float(min_seconds) * float(fps))) - n
    if need <= 0:
        return frames
    tail = np.repeat(np.asarray(frames[-1:], dtype=np.uint8), need, axis=0)
    return np.concatenate([frames, tail], axis=0)


@contextlib.contextmanager
def _exclusive_out_dir_lock(out_dir: Path):
    """같은 ``out_dir`` 에 롤아웃이 동시에 두 개 뜨지 않게 비차단 배타 락 (POSIX)."""
    try:
        import fcntl as _fcntl
    except ImportError:
        yield
        return
    lock_path = out_dir / '.manip_play_rollouts.lock'
    out_dir.mkdir(parents=True, exist_ok=True)
    fp = open(lock_path, 'a', encoding='utf-8')
    try:
        _fcntl.flock(fp.fileno(), _fcntl.LOCK_EX | _fcntl.LOCK_NB)
    except BlockingIOError as e:
        fp.close()
        raise SystemExit(
            f'이미 다른 manip_play_rollouts 가 {out_dir} 를 쓰는 중입니다 ({lock_path}). '
            f'끝날 때까지 기다리거나 ``--out_dir`` 로 다른 디렉터리를 쓰세요.'
        ) from e
    try:
        yield
    finally:
        try:
            _fcntl.flock(fp.fileno(), _fcntl.LOCK_UN)
        except OSError:
            pass
        try:
            fp.close()
        except OSError:
            pass


def _path_rel_to(root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


_ROLLOUT_SUMMARY_FIELDS: tuple[str, ...] = (
    'task_id',
    'env_name',
    'family',
    'checkpoint_epoch',
    'actor_checkpoint_epoch',
    'eval_max_chunks',
    'idm_horizon',
    'actor_horizon',
    'idm_chunks',
    'actor_chunks',
    'idm_env_success',
    'actor_env_success',
    'idm_mp4',
    'actor_mp4',
    'idm_mp4_frames',
    'actor_mp4_frames',
)


def _write_rollout_task_summary_csv(out_dir: Path, rows: list[dict[str, Any]]) -> Path:
    """태스크별 성공/지표를 ``rollout_task_summary.csv`` 로 저장 (``out_dir`` 기준 상대 경로 열)."""
    path = out_dir / 'rollout_task_summary.csv'
    if not rows:
        return path
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(_ROLLOUT_SUMMARY_FIELDS), extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)
    return path


def _run_one_task(
    run_dir: Path,
    task_id: int,
    ckpt_epoch: int,
    out_task_dir: Path,
    out_dir: Path,
    *,
    mujoco_gl: str,
    seed: int,
    idm_max_chunks: int,
    actor_max_chunks: int,
    fps: float,
    min_mp4_seconds: float,
    subgoal_filter: bool,
    subgoal_filter_threshold: float,
) -> dict[str, Any]:
    configure_mujoco_gl(mujoco_gl)
    cfg, env_name = load_run_flags(run_dir)
    family = manip_play_family(env_name)
    em, idm_h, act_h = _load_eval_rollout_limits(run_dir)

    env, train_raw, _ = make_env_and_datasets(
        env_name,
        frame_stack=cfg.get('frame_stack'),
        render_mode='rgb_array',
    )
    u = env.unwrapped
    n_tasks = int(getattr(u, 'num_tasks', 5))
    if not (1 <= int(task_id) <= n_tasks):
        raise ValueError(f'task_id must be in [1, {n_tasks}]')

    ob, info = env.reset(options=dict(task_id=int(task_id), render_goal=False))
    if 'goal' not in info:
        raise RuntimeError('reset did not set info["goal"]')
    s0 = np.asarray(ob, dtype=np.float32).reshape(-1)
    s_g = np.asarray(info['goal'], dtype=np.float32).reshape(-1)

    dyn_dir = resolve_dynamics_checkpoint_dir(run_dir)
    ckpt_epoch = pick_epoch(int(ckpt_epoch), list_checkpoint_suffixes(dyn_dir))
    act_dir = resolve_actor_checkpoint_dir(run_dir, required=True)
    act_ep = pick_epoch(int(ckpt_epoch), list_checkpoint_suffixes(act_dir))

    ex = jnp.zeros((1, s0.shape[-1]), dtype=jnp.float32)
    act_dim = int(np.prod(env.action_space.shape))
    ex_act = jnp.zeros((1, act_dim), dtype=jnp.float32)
    agent = DynamicsAgent.create(int(seed), ex, cfg, ex_actions=ex_act)
    dyn_pkl = dyn_dir / f'params_{ckpt_epoch}.pkl'
    agent = load_checkpoint_pkl(agent, dyn_pkl)
    value_subgoal_filter = None
    if bool(subgoal_filter):
        critic_agent = load_critic_for_run(
            run_dir,
            int(ckpt_epoch),
            env,
            train_raw,
            seed=int(seed),
        )
        value_subgoal_filter = make_value_subgoal_filter_from_critic_agent(
            critic_agent,
            reachability_threshold=float(subgoal_filter_threshold),
        )
        print(
            '[subgoal_filter] enabled: replace phi(predicted_subgoal) with phi(goal) when '
            f'V(subgoal,g) <= V(s,g) and V(s,subgoal) > {float(subgoal_filter_threshold):.4f}.'
        )

    low = np.asarray(env.action_space.low, dtype=np.float32).reshape(-1)
    high = np.asarray(env.action_space.high, dtype=np.float32).reshape(-1)

    idm_chunks = (
        int(idm_max_chunks)
        if int(idm_max_chunks) >= 0
        else _chunk_budget_for_full_episode(env, int(idm_h), int(em))
    )
    ob, info = env.reset(options=dict(task_id=int(task_id), render_goal=False))
    s0 = np.asarray(ob, dtype=np.float32).reshape(-1)
    s_g = np.asarray(info['goal'], dtype=np.float32).reshape(-1)

    idm_outcome = run_chunked_episode(
        env,
        s0,
        s_g,
        low=low,
        high=high,
        max_chunks=int(idm_chunks),
        sample_action_chunk=make_idm_chunk_fn(agent, int(idm_h), subgoal_filter=value_subgoal_filter),
        record_rgb=True,
    )
    idm_out_tag = 'success' if idm_outcome.ok_env else 'fail'
    mp4_idm = out_task_dir / f'idm_env_rgb_{idm_out_tag}.mp4'
    idm_mp4_rel = ''
    idm_mp4_frames = 0
    if idm_outcome.rgb_frames is not None and idm_outcome.rgb_frames.size > 0:
        frames = _pad_rgb_frames_min_duration(idm_outcome.rgb_frames, float(fps), float(min_mp4_seconds))
        write_rgb_array_mp4(frames, mp4_idm, float(fps))
        idm_mp4_rel = _path_rel_to(out_dir, mp4_idm)
        idm_mp4_frames = int(frames.shape[0])
        print(
            f'[task {task_id}] wrote {mp4_idm}  raw_frames={idm_outcome.rgb_frames.shape[0]} '
            f'mp4_frames={frames.shape[0]} chunks={idm_outcome.n_chunks} idm_horizon={idm_h} '
            f'eval_max_chunks={em} env_info_success={idm_outcome.ok_env}'
        )
    else:
        print(f'[task {task_id}] IDM: no RGB frames states={idm_outcome.states.shape[0]}')

    flags_path = run_dir / 'flags.json'
    actor_cfg = _load_actor_cfg(flags_path)
    actor_cfg['action_dim'] = act_dim
    ex_goal = jnp.asarray(s_g.reshape(1, -1), dtype=jnp.float32)
    actor_agent = ActorAgent.create(int(seed), ex, actor_cfg, ex_goals=ex_goal)
    actor_pkl = act_dir / f'params_{act_ep}.pkl'
    actor_agent = load_checkpoint_pkl(actor_agent, actor_pkl)

    actor_chunks = (
        int(actor_max_chunks)
        if int(actor_max_chunks) >= 0
        else _chunk_budget_for_full_episode(env, int(act_h), int(em))
    )
    ob, info = env.reset(options=dict(task_id=int(task_id), render_goal=False))
    s0 = np.asarray(ob, dtype=np.float32).reshape(-1)
    s_g = np.asarray(info['goal'], dtype=np.float32).reshape(-1)

    actor_outcome = run_chunked_episode(
        env,
        s0,
        s_g,
        low=low,
        high=high,
        max_chunks=int(actor_chunks),
        sample_action_chunk=make_actor_chunk_fn(
            agent,
            actor_agent,
            int(act_h),
            int(act_dim),
            subgoal_filter=value_subgoal_filter,
        ),
        record_rgb=True,
    )
    ac_out_tag = 'success' if actor_outcome.ok_env else 'fail'
    mp4_ac = out_task_dir / f'actor_env_rgb_{ac_out_tag}.mp4'
    actor_mp4_rel = ''
    actor_mp4_frames = 0
    if actor_outcome.rgb_frames is not None and actor_outcome.rgb_frames.size > 0:
        frames = _pad_rgb_frames_min_duration(actor_outcome.rgb_frames, float(fps), float(min_mp4_seconds))
        write_rgb_array_mp4(frames, mp4_ac, float(fps))
        actor_mp4_rel = _path_rel_to(out_dir, mp4_ac)
        actor_mp4_frames = int(frames.shape[0])
        print(
            f'[task {task_id}] wrote {mp4_ac}  raw_frames={actor_outcome.rgb_frames.shape[0]} '
            f'mp4_frames={frames.shape[0]} chunks={actor_outcome.n_chunks} actor_horizon={act_h} '
            f'eval_max_chunks={em} env_info_success={actor_outcome.ok_env}'
        )
    else:
        print(f'[task {task_id}] actor: no RGB frames states={actor_outcome.states.shape[0]}')

    return {
        'task_id': int(task_id),
        'env_name': str(env_name),
        'family': str(family),
        'checkpoint_epoch': int(ckpt_epoch),
        'actor_checkpoint_epoch': int(act_ep),
        'eval_max_chunks': int(em),
        'idm_horizon': int(idm_h),
        'actor_horizon': int(act_h),
        'idm_chunks': int(idm_outcome.n_chunks),
        'actor_chunks': int(actor_outcome.n_chunks),
        'idm_env_success': int(bool(idm_outcome.ok_env)),
        'actor_env_success': int(bool(actor_outcome.ok_env)),
        'idm_mp4': idm_mp4_rel,
        'actor_mp4': actor_mp4_rel,
        'idm_mp4_frames': idm_mp4_frames,
        'actor_mp4_frames': actor_mp4_frames,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run_dir', type=str, required=True)
    p.add_argument('--checkpoint_epoch', type=int, default=1000)
    p.add_argument(
        '--out_dir',
        type=str,
        default='',
        help='Default: <run_dir>/rollouts_manip_<env_slug>_ep<EPOCH>.',
    )
    p.add_argument('--task_ids', type=str, default='1,2,3,4,5', help='Comma-separated; e.g. "1" for single task.')
    p.add_argument('--mujoco_gl', type=str, default='osmesa')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--fps', type=float, default=30.0, help='MP4 playback FPS (default 30).')
    p.add_argument(
        '--idm_max_chunks',
        type=int,
        default=-1,
        help='Outer replans; -1 → max(flags.eval_max_chunks, ceil(TimeLimit/chunk)).',
    )
    p.add_argument(
        '--actor_max_chunks',
        type=int,
        default=-1,
        help='Outer replans; -1 → max(flags.eval_max_chunks, ceil(TimeLimit/chunk)).',
    )
    p.add_argument(
        '--min_mp4_seconds',
        type=float,
        default=0.0,
        help='If >0, pad MP4 by repeating the last frame until at least this duration (0 = actual length only).',
    )
    p.add_argument(
        '--subgoal_filter',
        action='store_true',
        help='If V(predicted_subgoal, goal) <= V(current_state, goal) and V(current_state, predicted_subgoal) > R, replace goal-representation channels.',
    )
    p.add_argument(
        '--subgoal_filter_threshold',
        type=float,
        default=0.5,
        help='Reachability threshold R for value filtering, applied to sigmoid V(current_state, predicted_subgoal).',
    )
    p.add_argument(
        '--no_exclusive_lock',
        action='store_true',
        help='기본 배타 락(out_dir/.manip_play_rollouts.lock)을 쓰지 않음 (디버그용; 중복 실행 주의).',
    )
    args = p.parse_args()

    run_dir = Path(args.run_dir).resolve()
    ckpt = int(args.checkpoint_epoch)
    out_arg = str(args.out_dir).strip()
    _, env_nm = load_run_flags(run_dir)
    slug = slug_from_env(env_nm)
    out_dir = (
        Path(out_arg).resolve()
        if out_arg
        else run_dir / f'rollouts_manip_{slug}_ep{ckpt}'
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    tids = parse_int_list(str(args.task_ids))
    if not tids:
        raise SystemExit('empty --task_ids')
    rows: list[dict[str, Any]] = []
    lock_cm = contextlib.nullcontext() if bool(args.no_exclusive_lock) else _exclusive_out_dir_lock(out_dir)
    with lock_cm:
        for tid in tids:
            sub = out_dir / f'task{tid}'
            sub.mkdir(parents=True, exist_ok=True)
            rows.append(
                _run_one_task(
                    run_dir,
                    int(tid),
                    ckpt,
                    sub,
                    out_dir,
                    mujoco_gl=str(args.mujoco_gl),
                    seed=int(args.seed),
                    idm_max_chunks=int(args.idm_max_chunks),
                    actor_max_chunks=int(args.actor_max_chunks),
                    fps=float(args.fps),
                    min_mp4_seconds=float(args.min_mp4_seconds),
                    subgoal_filter=bool(args.subgoal_filter),
                    subgoal_filter_threshold=float(args.subgoal_filter_threshold),
                )
            )
        summary_path = _write_rollout_task_summary_csv(out_dir, rows)
    print(f'done out_dir={out_dir} wrote {summary_path}')


__all__ = ['main']


if __name__ == '__main__':
    main()
