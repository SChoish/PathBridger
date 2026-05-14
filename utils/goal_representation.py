"""Goal representation helpers shared by goal-conditioned networks.

The ``phi`` mode mirrors OGBench's ``compute_oracle_observation`` for ManipSpace:
  - ManipSpace cube envs: ``(scaled) xyz`` per cube
    (``ogbench/manipspace/envs/cube_env.py::CubeEnv.compute_oracle_observation``).
  - ManipSpace puzzle envs: ``binary state`` per button
    (``ogbench/manipspace/envs/puzzle_env.py::PuzzleEnv.compute_oracle_observation``,
    returns ``self._cur_button_states.astype(np.float64)``).

Because cube-quadruple (``obs_dim=55``) and puzzle-3x3 (``obs_dim=55``) collide
on observation dimensionality, ``obs_dim`` alone cannot distinguish the two
layouts. Callers should pass ``env_name`` so we can route to the correct
oracle layout. For maze-style goals, ``critic_agent.phi_goal_obs_indices`` is
still required when the observation does not match either ManipSpace layout.
"""

from __future__ import annotations

from collections.abc import Sequence

import jax.numpy as jnp

_MANIP_ARM_JOINT_DIM = 6
_MANIP_HEAD_DIM = 2 * _MANIP_ARM_JOINT_DIM + 3 + 1 + 1 + 1 + 1
_MANIP_CUBE_STRIDE = 3 + 4 + 1 + 1
_MANIP_BUTTON_STATE_DIM = 2
_MANIP_BUTTON_STRIDE = _MANIP_BUTTON_STATE_DIM + 1 + 1


def _env_kind_from_name(env_name: str | None) -> str | None:
    """Return ``'cube'``, ``'puzzle'``, or ``None`` based on the env name."""

    if not env_name:
        return None
    name = str(env_name).lower()
    if 'puzzle' in name:
        return 'puzzle'
    if 'cube' in name or 'scene' in name:
        return 'cube'
    return None


def manip_cube_pos_indices(obs_dim: int) -> tuple[int, ...]:
    """Return compact ManipSpace cube-position channels for one observation frame.

    Cube obs layout (see ``ogbench.manipspace.envs.cube_env``):
    ``head(19) + n_cubes * (pos[3] + quat[4] + cos_yaw + sin_yaw)``. This helper
    returns the ``pos[3]`` channels for every cube, matching the cube oracle
    representation up to the ``xyz_center`` / ``xyz_scaler`` affine transform
    that the dataset's compact obs already bakes in.
    """

    dim = int(obs_dim)
    rem = dim - _MANIP_HEAD_DIM
    if rem < _MANIP_CUBE_STRIDE or rem % _MANIP_CUBE_STRIDE != 0:
        return ()
    idxs: list[int] = []
    for start in range(_MANIP_HEAD_DIM, dim, _MANIP_CUBE_STRIDE):
        idxs.extend((start, start + 1, start + 2))
    return tuple(idxs)


def manip_button_state_indices(obs_dim: int) -> tuple[int, ...]:
    """Return per-button binary-state channels for ManipSpace puzzle obs.

    Puzzle obs layout (see ``ogbench.manipspace.envs.puzzle_env``):
    ``head(19) + n_buttons * (one_hot_state[2] + button_pos[1] + button_vel[1])``.
    The puzzle oracle is ``self._cur_button_states.astype(np.float64)`` (binary
    scalar per button), which corresponds to the ``state=1`` one-hot channel in
    the compact obs (i.e. ``obs[start + 1]`` for each button block).
    """

    dim = int(obs_dim)
    rem = dim - _MANIP_HEAD_DIM
    if rem <= 0 or rem % _MANIP_BUTTON_STRIDE != 0:
        return ()
    n_buttons = rem // _MANIP_BUTTON_STRIDE
    idxs: list[int] = []
    for i in range(n_buttons):
        start = _MANIP_HEAD_DIM + i * _MANIP_BUTTON_STRIDE
        idxs.append(start + 1)
    return tuple(idxs)


def normalize_phi_goal_obs_indices(raw: object) -> tuple[int, ...]:
    """Parse YAML / CLI values into a tuple of non-negative ints (may be empty)."""

    if raw is None:
        return ()
    if isinstance(raw, (list, tuple)):
        return tuple(int(x) for x in raw)
    raise TypeError(f'phi_goal_obs_indices must be a list/tuple of ints, got {type(raw).__name__}')


def assert_phi_goal_obs_indices(
    obs_dim: int,
    mode: str,
    phi_goal_obs_indices: Sequence[int] | tuple[int, ...],
    *,
    where: str,
    env_name: str | None = None,
) -> None:
    """Validate phi goal channels for ``obs_dim`` when mode uses phi."""

    mode_l = str(mode).lower()
    if mode_l in ('full', 'raw', 'none', ''):
        return
    if mode_l not in ('phi', 'auto', 'goal_phi'):
        return
    dim = int(obs_dim)
    kind = _env_kind_from_name(env_name)
    if kind == 'puzzle':
        if not manip_button_state_indices(dim):
            raise ValueError(
                f'{where}: goal_representation={mode_l!r} for env={env_name!r}: '
                f'obs_dim={dim} is not compatible with the puzzle layout '
                f'(head={_MANIP_HEAD_DIM}, button_stride={_MANIP_BUTTON_STRIDE}).'
            )
        return
    if kind == 'cube':
        if not manip_cube_pos_indices(dim):
            raise ValueError(
                f'{where}: goal_representation={mode_l!r} for env={env_name!r}: '
                f'obs_dim={dim} is not compatible with the cube layout '
                f'(head={_MANIP_HEAD_DIM}, cube_stride={_MANIP_CUBE_STRIDE}).'
            )
        return

    cube_idxs = manip_cube_pos_indices(dim)
    button_idxs = manip_button_state_indices(dim)
    if cube_idxs and button_idxs:
        raise ValueError(
            f'{where}: goal_representation={mode_l!r} with obs_dim={dim} matches both '
            'cube and puzzle compact layouts; set env_name (e.g. puzzle-3x3-play-v0) '
            'or use critic_agent.phi_goal_obs_indices explicitly.'
        )
    if cube_idxs or button_idxs:
        return
    idxs = tuple(int(x) for x in phi_goal_obs_indices)
    if not idxs:
        raise ValueError(
            f'{where}: goal_representation={mode_l!r} with obs_dim={dim} requires '
            'critic_agent.phi_goal_obs_indices (e.g. [0, 1] for planar x,y in the '
            'goal observation). Implicit [:2] slicing is disabled.'
        )
    for i in idxs:
        if i < 0 or i >= dim:
            raise ValueError(
                f'{where}: phi_goal_obs_indices={idxs!r} out of range for obs_dim={dim}.'
            )


def phi_subgoal_filter_replace_indices(
    obs_dim: int,
    phi_goal_obs_indices: Sequence[int] | tuple[int, ...],
    *,
    env_name: str | None = None,
) -> tuple[int, ...]:
    """Observation indices to copy from goal onto subgoal when value-filtering."""

    dim = int(obs_dim)
    kind = _env_kind_from_name(env_name)
    if kind == 'puzzle':
        idxs = manip_button_state_indices(dim)
        if not idxs:
            raise ValueError(
                f'Subgoal filter: puzzle env={env_name!r} but obs_dim={dim} does not match puzzle layout.'
            )
        return idxs
    if kind == 'cube':
        idxs = manip_cube_pos_indices(dim)
        if not idxs:
            raise ValueError(
                f'Subgoal filter: cube env={env_name!r} but obs_dim={dim} does not match cube layout.'
            )
        return idxs

    cube_idxs = manip_cube_pos_indices(dim)
    button_idxs = manip_button_state_indices(dim)
    if cube_idxs and button_idxs:
        raise ValueError(
            f'Subgoal filter: obs_dim={dim} matches cube and puzzle layouts; set env_name on critic config.'
        )
    if cube_idxs:
        return cube_idxs
    if button_idxs:
        return button_idxs
    idxs = tuple(int(x) for x in phi_goal_obs_indices)
    if not idxs:
        raise ValueError(
            'Subgoal filter replacement needs ManipSpace cube/puzzle layout or '
            'critic_agent.phi_goal_obs_indices for non-ManipSpace observations.'
        )
    return idxs


def goal_representation(
    goals: jnp.ndarray | None,
    mode: str,
    phi_goal_obs_indices: Sequence[int] | tuple[int, ...] = (),
    *,
    env_name: str | None = None,
) -> jnp.ndarray | None:
    """Map a full goal state to the configured goal representation.

    ``full`` keeps historical behavior. ``phi`` / ``auto`` / ``goal_phi``:
    When ``env_name`` indicates a ManipSpace puzzle, use per-button binary
    channels; when it indicates a cube env, use inferred xyz per cube.
    Otherwise, if ``obs_dim`` matches only one ManipSpace layout, use that;
    if both match (ambiguous), ``env_name`` is required. For other observations,
    ``phi_goal_obs_indices`` must list goal indices (e.g. ``(0, 1)`` for maze x,y).
    """

    if goals is None:
        return None
    mode_l = str(mode).lower()
    if mode_l in ('full', 'raw', 'none', ''):
        return goals
    if mode_l not in ('phi', 'auto', 'goal_phi'):
        raise ValueError(
            f"Unknown goal_representation={mode!r}; expected 'full' or 'phi'."
        )

    obs_dim = int(goals.shape[-1])
    kind = _env_kind_from_name(env_name)

    if kind == 'puzzle':
        idxs = manip_button_state_indices(obs_dim)
        if not idxs:
            raise ValueError(
                f"goal_representation='phi' for env={env_name!r}: obs_dim={obs_dim} "
                f"is not compatible with the puzzle layout "
                f"(head={_MANIP_HEAD_DIM}, button_stride={_MANIP_BUTTON_STRIDE})."
            )
        return jnp.take(goals, jnp.asarray(idxs, dtype=jnp.int32), axis=-1)

    if kind == 'cube':
        idxs = manip_cube_pos_indices(obs_dim)
        if not idxs:
            raise ValueError(
                f"goal_representation='phi' for env={env_name!r}: obs_dim={obs_dim} "
                f"is not compatible with the cube layout "
                f"(head={_MANIP_HEAD_DIM}, cube_stride={_MANIP_CUBE_STRIDE})."
            )
        return jnp.take(goals, jnp.asarray(idxs, dtype=jnp.int32), axis=-1)

    cube_idxs = manip_cube_pos_indices(obs_dim)
    button_idxs = manip_button_state_indices(obs_dim)
    if cube_idxs and button_idxs:
        raise ValueError(
            f'goal_representation={mode_l!r}: obs_dim={obs_dim} matches both cube and puzzle '
            'compact layouts; set env_name (e.g. puzzle-3x3-play-v0) or phi_goal_obs_indices.'
        )
    if cube_idxs:
        take = jnp.asarray(cube_idxs, dtype=jnp.int32)
        return jnp.take(goals, take, axis=-1)
    if button_idxs:
        take = jnp.asarray(button_idxs, dtype=jnp.int32)
        return jnp.take(goals, take, axis=-1)

    idxs = tuple(int(x) for x in phi_goal_obs_indices)
    if not idxs:
        raise ValueError(
            f'goal_representation={mode_l!r} requires critic_agent.phi_goal_obs_indices for '
            f'obs_dim={obs_dim} (non-ManipSpace).'
        )
    for i in idxs:
        if i < 0 or i >= obs_dim:
            raise ValueError(f'phi_goal_obs_indices={idxs!r} out of range for obs_dim={obs_dim}.')
    take = jnp.asarray(idxs, dtype=jnp.int32)
    return jnp.take(goals, take, axis=-1)
