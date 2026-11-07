"""Bounded mathematical checks; no learned checkpoint, optimizer, or environment.

This audit is a self-review of ede2b8a, not a policy-performance experiment.
Run from the repository root with the installed JAX/Flax/Optax dependencies.
"""

from __future__ import annotations

import argparse
import json
from fractions import Fraction
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from skill_extensions.planner import first_hit, return_target, validate_config

SOURCE_REVISION = 'ede2b8a5b39f367f8ecf8f14003aa0ace880b18b'


def factual_score(hit, duration, continuation, gamma):
    """Independent exact scalar definition, with None denoting no hit."""
    return gamma**hit if hit is not None else gamma**duration * continuation


def check_partition():
    config = validate_config({})
    durations = config['durations']
    observations, goals, lengths, expected_hits = [], [], [], []
    exact_cases, float_errors = 0, []
    for d in durations:
        for hit in [*range(d + 1), None]:
            observations.append(np.arange(41, dtype=np.float32)[:, None])
            goals.append([100.0 if hit is None else float(hit)])
            lengths.append(d)
            expected_hits.append(-1 if hit is None else hit)
            for gamma in (Fraction(9, 10), Fraction(99, 100), Fraction(1)):
                for terminal in (Fraction(0), Fraction(3, 10), Fraction(1)):
                    direct = factual_score(hit, d, terminal, gamma)
                    if hit is not None and hit <= 5:
                        partitioned = gamma**hit
                    else:
                        suffix_hit = None if hit is None else hit - 5
                        partitioned = gamma**5 * factual_score(
                            suffix_hit, d - 5, terminal, gamma
                        )
                    assert direct == partitioned
                    exact_cases += 1

    found = first_hit(
        jnp.array(np.stack(observations)),
        jnp.array(lengths),
        jnp.array(goals),
        jnp.ones(1),
        config['goal_tolerance'],
    )
    np.testing.assert_array_equal(found, expected_hits)
    for gamma in (Fraction(9, 10), Fraction(99, 100), Fraction(1)):
        for terminal in (Fraction(0), Fraction(3, 10), Fraction(1)):
            predicted = np.asarray(
                return_target(found, jnp.array(lengths), float(terminal), float(gamma))
            )
            expected = np.array(
                [
                    float(factual_score(None if h < 0 else h, d, terminal, gamma))
                    for h, d in zip(expected_hits, lengths, strict=True)
                ]
            )
            error = float(np.max(np.abs(predicted - expected)))
            assert error < 5e-7, error
            float_errors.append(error)
    return {
        'exact_partition_cases': exact_cases,
        'first_hit_cases': len(lengths),
        'return_target_comparisons': len(lengths) * 9,
        'maximum_float32_return_error': max(float_errors),
        'lengths': list(durations),
        'cut': 5,
        'scope': 'factual suffix score with the same terminal potential; not learned Q accuracy',
    }


def check_parent_loss_weight():
    # A balanced batch with 32 examples per duration, including d=5.
    # Reconstruct normalization from actual anchor/split populations; both Q
    # heads introduce the same factor and cancel in this relative weight.
    batch, long = 256, 224
    anchor_count = 2 * batch + long
    anchor_per_parent = Fraction(1, anchor_count)
    split_per_parent = Fraction(1, long)
    ratio = split_per_parent / anchor_per_parent
    assert ratio == Fraction(23, 7)
    # For one independently variable parent prediction q with direct target y
    # and a suffix backup of zero, the BCE minimizer is a*y/(a+b).
    direct = Fraction(99, 100) ** 20
    minimized = anchor_per_parent * direct / (anchor_per_parent + split_per_parent)
    assert minimized / direct == Fraction(7, 30)
    gradient_in_q = (
        anchor_per_parent * (minimized - direct) + split_per_parent * minimized
    ) / (minimized * (1 - minimized))
    assert gradient_in_q == 0
    return {
        'batch': batch,
        'long_examples': long,
        'anchor_query_count': anchor_count,
        'split_to_full_anchor_relative_weight': str(ratio),
        'direct_target': float(direct),
        'hypothetical_zero_suffix_backup': 0.0,
        'independent_parent_bce_minimizer': float(minimized),
        'scope': 'illustrative normalization calculation; not an observed model collapse',
    }


def check_missing_execution_assumption():
    # A deterministic scalar accumulator, evaluated algebraically rather than
    # through an environment. A decoder error in the prefix invalidates using
    # the recorded boundary as that decoder's actual next state.
    reference = np.r_[0.0, np.cumsum(np.ones(10))].astype(np.float32)
    decoded = np.r_[0.0, np.cumsum(np.r_[np.zeros(5), np.ones(5)])].astype(np.float32)
    states = jnp.array(np.stack((reference, decoded)))[:, :, None]
    hits = first_hit(states, jnp.array([10, 10]), jnp.full((2, 1), 10.0), jnp.ones(1), 1e-5)
    np.testing.assert_array_equal(hits, [10, -1])
    scores = np.asarray(return_target(hits, jnp.array([10, 10]), jnp.zeros(2), 0.99))
    assert scores[0] > 0.9 and scores[1] == 0
    return {
        'factual_boundary': float(reference[5]),
        'decoded_boundary': float(decoded[5]),
        'factual_score': float(scores[0]),
        'decoded_score': float(scores[1]),
        'scope': 'counterexample to an unconditional exact-policy interpretation, not to approximate learning',
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result', type=Path, required=True)
    args = parser.parse_args()
    result = {
        'source_revision': SOURCE_REVISION,
        'review_provenance': 'self-review',
        'status': 'completed',
        'formal_training_updates': 0,
        'environment_steps': 0,
        'partition': check_partition(),
        'normalization': check_parent_loss_weight(),
        'missing_execution_assumption': check_missing_execution_assumption(),
    }
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
