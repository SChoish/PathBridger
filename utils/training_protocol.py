"""Explicit training-semantics versions for state PathBridger.

The protocol is deliberately narrower than a whole-runtime revision.  It
versions the hindsight sampler plus scalar-value and endpoint-weighting
objectives whose meanings differ between the manuscript implementation and the
July result-producing split implementation.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

MANUSCRIPT_TRAINING_PROTOCOL = 'manuscript-v1'
REPORTED_RUN_LEGACY_TRAINING_PROTOCOL = 'reported-run-legacy-v1'
SUPPORTED_TRAINING_PROTOCOLS = frozenset(
    {
        MANUSCRIPT_TRAINING_PROTOCOL,
        REPORTED_RUN_LEGACY_TRAINING_PROTOCOL,
    }
)


def normalize_training_protocol(value: Any) -> str:
    """Validate one explicit protocol without accepting silent aliases."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            'training_objective_protocol must be a non-empty string; '
            f'expected one of {sorted(SUPPORTED_TRAINING_PROTOCOLS)}.'
        )
    protocol = value.strip().lower()
    if protocol not in SUPPORTED_TRAINING_PROTOCOLS:
        raise ValueError(
            f'Unknown training_objective_protocol {value!r}; expected one of '
            f'{sorted(SUPPORTED_TRAINING_PROTOCOLS)}.'
        )
    return protocol


def training_protocol_from_config(config: Any) -> str:
    """Resolve the protocol from a mapping/config object.

    Old programmatic configs predate the field. They retain the previously
    shipped unified behavior, which is the reported-run legacy protocol. All
    file-backed presets declare the field through the agent default, and
    manuscript-target presets override it explicitly.
    """

    if isinstance(config, Mapping):
        value = config.get(
            'training_objective_protocol',
            REPORTED_RUN_LEGACY_TRAINING_PROTOCOL,
        )
    else:
        value = getattr(
            config,
            'training_objective_protocol',
            REPORTED_RUN_LEGACY_TRAINING_PROTOCOL,
        )
    return normalize_training_protocol(value)


__all__ = [
    'MANUSCRIPT_TRAINING_PROTOCOL',
    'REPORTED_RUN_LEGACY_TRAINING_PROTOCOL',
    'SUPPORTED_TRAINING_PROTOCOLS',
    'normalize_training_protocol',
    'training_protocol_from_config',
]
