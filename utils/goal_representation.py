"""Goal representation helpers shared by goal-conditioned networks."""

from __future__ import annotations

import jax.numpy as jnp

_MANIP_ARM_JOINT_DIM = 6
_MANIP_HEAD_DIM = 2 * _MANIP_ARM_JOINT_DIM + 3 + 1 + 1 + 1 + 1
_MANIP_CUBE_STRIDE = 3 + 4 + 1 + 1
_DEFAULT_GOAL_REP_DIM = 2


def manip_cube_pos_indices(obs_dim: int) -> tuple[int, ...]:
    """Return compact ManipSpace cube-position channels for one observation frame."""

    dim = int(obs_dim)
    rem = dim - _MANIP_HEAD_DIM
    if rem < _MANIP_CUBE_STRIDE or rem % _MANIP_CUBE_STRIDE != 0:
        return ()
    idxs: list[int] = []
    for start in range(_MANIP_HEAD_DIM, dim, _MANIP_CUBE_STRIDE):
        idxs.extend((start, start + 1, start + 2))
    return tuple(idxs)


def goal_representation(goals: jnp.ndarray | None, mode: str = 'full') -> jnp.ndarray | None:
    """Map a full goal state to the configured goal representation.

    ``full`` keeps historical behavior. ``phi``/``auto`` uses compact
    ManipSpace cube positions when the observation shape matches, otherwise it
    falls back to the first two channels used by maze-style xy goals.
    """

    if goals is None:
        return None
    mode = str(mode).lower()
    if mode in ('full', 'raw', 'none', ''):
        return goals
    if mode not in ('phi', 'auto', 'goal_phi'):
        raise ValueError(
            f"Unknown goal_representation={mode!r}; expected 'full' or 'phi'."
        )
    idxs = manip_cube_pos_indices(int(goals.shape[-1]))
    if not idxs:
        n = min(_DEFAULT_GOAL_REP_DIM, int(goals.shape[-1]))
        idxs = tuple(range(n))
    return jnp.take(goals, jnp.asarray(idxs, dtype=jnp.int32), axis=-1)
