"""Gated track-to-object assignment.

Each track and each object may stay unmatched at NO_MATCH_COST, so the solver
never forces a poor pair just to match everyone (KIWA 2026-11-02 00:54: a
12-scan track was paired with a 14 km2 sliver so that a far-away 1-scan track
could take the main squall line).
"""
import itertools

import numpy as np
from hypothesis import given, settings, strategies as st
from hypothesis.extra.numpy import arrays

from src.tracking.association import NO_MATCH_COST, UNMATCHED_COST, gated_assignment


def _objective(cost: np.ndarray, pairs: list[tuple[int, int]]) -> float:
    rows, cols = cost.shape
    return float(sum(cost[r, c] for r, c in pairs)) + NO_MATCH_COST * ((rows - len(pairs)) + (cols - len(pairs)))


def _brute_force_best(cost: np.ndarray) -> float:
    """The lowest objective over every partial matching of real candidates."""
    rows, cols = cost.shape
    best = _objective(cost, [])
    for size in range(1, min(rows, cols) + 1):
        for chosen_rows in itertools.combinations(range(rows), size):
            for chosen_cols in itertools.permutations(range(cols), size):
                pairs = list(zip(chosen_rows, chosen_cols))
                if any(cost[r, c] >= UNMATCHED_COST for r, c in pairs):
                    continue
                best = min(best, _objective(cost, pairs))
    return best


# Real candidate costs, or UNMATCHED_COST where a pair is not a candidate.
_cost_entry = st.one_of(
    st.floats(min_value=0.0, max_value=5.0, allow_nan=False, width=32),
    st.just(UNMATCHED_COST),
)
_cost_matrices = st.tuples(st.integers(0, 4), st.integers(0, 4)).flatmap(
    lambda shape: arrays(np.float64, shape, elements=_cost_entry)
)


def test_the_kiwa_squall_line_keeps_its_own_track():
    # rows: the 12-scan line track, a 1-scan track 99 km away
    # cols: the 3291 km2 squall line, a 14 km2 sliver
    cost = np.array([[0.48, 1.01], [1.84, UNMATCHED_COST]])
    assert gated_assignment(cost) == [(0, 0)]


@given(_cost_matrices)
@settings(max_examples=300, deadline=None)
def test_pairs_are_real_candidates_with_each_row_and_column_used_once(cost):
    pairs = gated_assignment(cost)
    assert all(cost[r, c] < UNMATCHED_COST for r, c in pairs)
    assert len({r for r, _ in pairs}) == len(pairs)
    assert len({c for _, c in pairs}) == len(pairs)


@given(_cost_matrices)
@settings(max_examples=300, deadline=None)
def test_the_assignment_is_the_cheapest_partial_matching(cost):
    assert _objective(cost, gated_assignment(cost)) <= _brute_force_best(cost) + 1e-6


@given(_cost_matrices, st.lists(st.floats(2.0 * NO_MATCH_COST + 0.01, 5.0), min_size=4, max_size=4))
@settings(max_examples=300, deadline=None)
def test_a_track_whose_every_candidate_is_poor_is_never_matched(cost, poor):
    rows, cols = cost.shape
    poor_row = np.asarray(poor[:cols], dtype=float).reshape(1, cols)
    pairs = gated_assignment(np.vstack([cost, poor_row]))
    assert all(r != rows for r, _ in pairs)
