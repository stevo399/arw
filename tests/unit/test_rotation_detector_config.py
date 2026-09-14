"""Rotation candidate rules (plan docs/superpowers/plans/2026-09-14-rotation-detection-truth.md, Task 6).

Synthetic sweeps from tests/unit/test_velocity.py: 360 rays one degree apart,
gates from 2 to 230 km (gate 155 is 72.8 km out, where one degree spans
1.27 km), Nyquist velocity 26.2 m/s.
"""
from dataclasses import replace

import numpy as np

from src.velocity import RotationDetectorConfig, detect_rotation_signatures
from tests.unit.test_velocity import _make_sweep, _make_velocity_data

# Each rule is tested by switching it on from the baseline, so these tests do not
# depend on which rules the corpus selected as defaults.
BASELINE = RotationDetectorConfig.baseline()
AZIMUTHAL = replace(BASELINE, azimuthal_pairs_only=True)


def _radial_convergence():
    """Opposite signs along the beam (same rays, next gate): convergence, not rotation."""
    grid = np.full((360, 500), np.nan)
    grid[50:60, 150:155] = 20.0
    grid[50:60, 155:160] = -20.0
    return grid


def _azimuthal_couplet(first_row=50):
    grid = np.full((360, 500), np.nan)
    grid[first_row:first_row + 5, 150:160] = -20.0
    grid[first_row + 5:first_row + 10, 150:160] = 20.0
    return grid


def test_the_baseline_counts_pairs_along_the_beam():
    velocity = _make_velocity_data([_make_sweep(_radial_convergence())])
    assert detect_rotation_signatures(velocity, RotationDetectorConfig.baseline())


def test_azimuthal_pairs_only_rejects_convergence_and_keeps_rotation():
    assert detect_rotation_signatures(_make_velocity_data([_make_sweep(_radial_convergence())]), AZIMUTHAL) == []
    assert detect_rotation_signatures(_make_velocity_data([_make_sweep(_azimuthal_couplet())]), AZIMUTHAL)


def test_min_side_requires_both_inbound_and_outbound_strength():
    grid = np.full((360, 500), np.nan)
    grid[50:55, 150:160] = -4.0
    grid[55:60, 150:160] = 30.0
    velocity = _make_velocity_data([_make_sweep(grid)])
    assert detect_rotation_signatures(velocity, AZIMUTHAL)
    assert detect_rotation_signatures(velocity, replace(AZIMUTHAL, min_side_ms=10.0)) == []


def test_fold_rejection_drops_a_jump_near_twice_nyquist():
    grid = np.full((360, 500), np.nan)
    grid[50:55, 150:160] = 24.0      # +24 beside -24 with Nyquist 26.2: a fold, not 48 m/s of shear
    grid[55:60, 150:160] = -24.0
    velocity = _make_velocity_data([_make_sweep(grid)])
    assert detect_rotation_signatures(velocity, AZIMUTHAL)
    assert detect_rotation_signatures(velocity, replace(AZIMUTHAL, fold_rejection=True)) == []


def test_ground_overlap_merge_needs_candidates_to_overlap_not_just_lie_within_10_km():
    # Shear boundaries at rays 54/55 and 61/62: 7 degrees, 8.9 km apart at 72.8 km.
    # Each candidate is about 4 km across, so they do not overlap.
    velocity = _make_velocity_data([
        _make_sweep(_azimuthal_couplet(50), elevation=0.48),
        _make_sweep(_azimuthal_couplet(57), elevation=0.88),
    ])
    by_distance = detect_rotation_signatures(velocity, AZIMUTHAL)
    by_overlap = detect_rotation_signatures(velocity, replace(AZIMUTHAL, ground_overlap_merge=True))
    assert max(s.sweep_count for s in by_distance) == 2
    assert max(s.sweep_count for s in by_overlap) == 1


def test_ground_overlap_merge_still_confirms_a_circulation_seen_on_two_tilts():
    velocity = _make_velocity_data([
        _make_sweep(_azimuthal_couplet(50), elevation=0.48),
        _make_sweep(_azimuthal_couplet(50), elevation=0.88),
    ])
    signatures = detect_rotation_signatures(velocity, replace(AZIMUTHAL, ground_overlap_merge=True))
    assert [s.sweep_count for s in signatures] == [2]


def test_max_diameter_drops_candidates_larger_than_a_mesocyclone():
    grid = np.full((360, 500), np.nan)
    grid[50:55, 100:140] = -20.0     # 40 gates, about 18 km along the beam
    grid[55:60, 100:140] = 20.0
    velocity = _make_velocity_data([_make_sweep(grid)])
    assert detect_rotation_signatures(velocity, AZIMUTHAL)
    assert detect_rotation_signatures(velocity, replace(AZIMUTHAL, max_diameter_km=10.0)) == []


def test_defaults_are_the_configuration_selected_on_the_corpus():
    assert RotationDetectorConfig() == RotationDetectorConfig(
        min_shear_ms=15.0, min_side_ms=0.0, max_diameter_km=None,
        fold_rejection=True, azimuthal_pairs_only=True, ground_overlap_merge=True,
    )
    assert RotationDetectorConfig() != BASELINE
