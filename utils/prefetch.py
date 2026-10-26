"""Deterministic host-side batch prefetch lifecycle."""

from __future__ import annotations

import copy
from concurrent.futures import Executor, Future
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PrefetchedBatch:
    """A future paired with the sampler state from before its submission."""

    future: Future
    sampler_rng_state: Any

    def result(self) -> Any:
        return self.future.result()

    def rollback(self, sampler: Any) -> None:
        """Discard the batch without advancing the next checkpoint's RNG.

        The caller must own the sampler exclusively, and ``get_rng_state`` /
        ``set_rng_state`` must cover every mutable state changed by ``sample``.
        Waiting before restoration prevents the worker from advancing the
        sampler again after its pre-sample state has been restored.
        """

        try:
            self.future.result()
        finally:
            sampler.set_rng_state(copy.deepcopy(self.sampler_rng_state))


def submit_prefetched_batch(
    executor: Executor,
    sampler: Any,
    batch_size: int,
) -> PrefetchedBatch:
    """Submit one sample while retaining its complete pre-sample state.

    Submission requires exclusive sampler ownership until ``result`` or
    ``rollback`` completes.  PathBridgerDataset satisfies the state contract:
    sampling mutates only the RNG exposed by get/set_rng_state.
    """

    if not hasattr(sampler, 'get_rng_state') or not hasattr(sampler, 'set_rng_state'):
        raise TypeError(
            'An asynchronously prefetched sampler must expose get_rng_state() and set_rng_state().'
        )
    sampler_rng_state = copy.deepcopy(sampler.get_rng_state())
    return PrefetchedBatch(
        future=executor.submit(sampler.sample, batch_size),
        sampler_rng_state=sampler_rng_state,
    )


def resolve_stop_boundary(
    prefetched: PrefetchedBatch | None,
    sampler: Any,
    *,
    stop_requested: bool,
    last_completed_step: int,
) -> tuple[bool, PrefetchedBatch | None]:
    """Roll back pending work when a stop precedes the next update.

    A fresh run has no representable step-zero checkpoint, so its first update
    remains the current update even if a signal is already pending.
    """

    should_stop = bool(stop_requested) and int(last_completed_step) > 0
    if not should_stop:
        return False, prefetched
    if prefetched is not None:
        prefetched.rollback(sampler)
    return True, None
