"""Tracking compares scans by true azimuth, not by beam index.

Each Level II volume starts at whatever azimuth the antenna happens to be
pointing, so beam 0 differs between scans (measured 336, 343, 341 and 8
degrees on consecutive KEYX volumes; 177 degrees apart on a KTLX pair).
"""
from dataclasses import replace
from datetime import datetime, timedelta

import numpy as np

from src.tracker import StormTracker
from src.tracking.alignment import align_scan_to
from src.tracking.association import associate_tracks
from tests.unit.test_tracking_association import _make_object, _make_scan

T0 = datetime(2026, 9, 12, 18, 0)
SHAPE = (360, 500)


def _block(row: int, col: int, rows: int = 12, cols: int = 12) -> np.ndarray:
    mask = np.zeros(SHAPE, dtype=bool)
    mask[row:row + rows, col:col + cols] = True
    return mask


def _rotated(scan, rays: int):
    """The same geography, recorded by a volume whose beam 0 starts `rays` beams later."""
    sweep = scan.reflectivity_data
    rotated_sweep = replace(
        sweep,
        reflectivity=np.roll(sweep.reflectivity, -rays, axis=0),
        azimuths=np.roll(sweep.azimuths, -rays),
        elevations=np.roll(sweep.elevations, -rays),
    )
    labels = np.roll(scan.labeled_grid, -rays, axis=0)
    masks = {object_id: np.roll(mask, -rays, axis=0) for object_id, mask in scan.object_masks.items()}
    return replace(scan, reflectivity_data=rotated_sweep, labeled_grid=labels, object_masks=masks)


def test_alignment_is_exactly_the_identity_for_identical_beam_order():
    scan = _make_scan("KTLX", T0, [_make_object(1, 35.3, -97.3)], {1: _block(100, 200)})
    scan.reflectivity_data.reflectivity[100:112, 200:212] = 45.0
    aligned = align_scan_to(scan, scan)
    assert np.array_equal(aligned.reflectivity_data.reflectivity, scan.reflectivity_data.reflectivity, equal_nan=True)
    assert np.array_equal(aligned.labeled_grid, scan.labeled_grid)
    assert np.array_equal(aligned.object_masks[1], scan.object_masks[1])
    assert np.array_equal(aligned.reflectivity_data.azimuths, scan.reflectivity_data.azimuths)


def test_alignment_puts_a_rotated_scan_back_on_the_current_beam_order():
    previous = _make_scan("KTLX", T0, [_make_object(1, 35.3, -97.3)], {1: _block(100, 200)})
    previous.reflectivity_data.reflectivity[100:112, 200:212] = 45.0
    current = _rotated(_make_scan("KTLX", T0 + timedelta(minutes=5), [_make_object(1, 35.3, -97.3)], {1: _block(100, 200)}), 54)

    aligned = align_scan_to(previous, current)

    assert np.array_equal(aligned.object_masks[1], current.object_masks[1])
    assert np.array_equal(aligned.reflectivity_data.azimuths, current.reflectivity_data.azimuths)
    assert np.array_equal(
        aligned.reflectivity_data.reflectivity[:, 200:212] >= 45.0,
        current.object_masks[1][:, 200:212],
    )


def test_stationary_storm_matches_with_full_overlap_across_a_rotated_volume():
    previous = _make_scan("KTLX", T0, [_make_object(1, 35.3, -97.3)], {1: _block(100, 200)})
    current = _rotated(_make_scan("KTLX", T0 + timedelta(minutes=5), [_make_object(1, 35.3, -97.3)], {1: _block(100, 200)}), 54)
    tracker = StormTracker()
    tracker.update(previous)

    association = associate_tracks(previous, current, tracker.all_tracks, tracker._obj_to_track)

    assert association.primary_matches == {1: 1}
    score = next(s for s in association.candidate_scores if s.track_id == 1 and s.object_id == 1)
    assert score.overlap_score == 1.0
    assert score.advected_overlap_score == 1.0


def test_storm_missing_for_a_scan_is_reacquired_across_rotated_volumes():
    stored = {}
    tracker = StormTracker(scan_loader=lambda site_id, timestamp: stored.get(timestamp), reacquire=True)
    storm_b = (_make_object(2, 35.0, -97.0, 45.0), _block(250, 300))

    def scan(minute, storms, rays):
        objects = [obj for obj, _ in storms]
        masks = {obj.object_id: mask for obj, mask in storms}
        return _rotated(_make_scan("KTLX", T0 + timedelta(minutes=minute), objects, masks), rays)

    storm_a = (_make_object(1, 35.30, -97.50, 55.0), _block(100, 100))
    b_only = (_make_object(1, 35.0, -97.0, 45.0), _block(250, 300))
    for minute, storms, rays in ((0, [storm_a, storm_b], 0), (5, [b_only], 14), (10, [storm_a, storm_b], 90)):
        current = scan(minute, storms, rays)
        tracker.update(current)
        stored[current.timestamp] = current

    track_a = tracker.get_track(1)
    assert track_a.status == "active"
    assert track_a.identity_diagnostics.event_context == "reacquired"
