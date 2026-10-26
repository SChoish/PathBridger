"""Shared construction of the state PathBridger training/inference runtime."""

from __future__ import annotations

from typing import Any

from agents import PathBridgerAgent
from utils.datasets import PathBridgerDataset, observation_state_scale


def build_state_runtime(
    train_data: Any,
    config: Any,
    *,
    seed: int,
) -> tuple[PathBridgerAgent, PathBridgerDataset, Any]:
    """Create the common sampler, normalization, and agent template."""

    action_free = bool(config.get('offline_action_free', False))
    dataset = PathBridgerDataset(
        train_data,
        config,
        require_actions=not action_free,
        seed=int(seed),
    )
    state_scale = observation_state_scale(
        train_data,
        floor=float(config.prefix_scale_floor),
    )
    example_batch = dataset.sample(1)
    agent = PathBridgerAgent.create(
        int(seed),
        example_batch['observations'],
        example_batch.get('actions'),
        config,
        state_scale=state_scale,
    )
    return agent, dataset, state_scale


__all__ = ['build_state_runtime']
