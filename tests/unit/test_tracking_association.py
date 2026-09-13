from datetime import datetime, timedelta

import numpy as np

from src.buffer import BufferedScan
from src.detection import DetectedObject
from src.parser import SweepData
from src.tracker import StormTracker
from src.tracking.association import AssociationResult, associate_tracks, compute_advected_iou


def _make_object(obj_id: int, lat: float, lon: float, peak_dbz: float = 45.0, area_km2: float = 100.0) -> DetectedObject:
    return DetectedObject(
        object_id=obj_id,
        centroid_lat=lat,
        centroid_lon=lon,
        distance_km=40.0,
        bearing_deg=270.0,
        peak_dbz=peak_dbz,
        peak_label="heavy precipitation",
        area_km2=area_km2,
        layers=[],
    )


def _make_scan(
    site_id: str,
    timestamp: datetime,
    objects: list[DetectedObject],
    masks: dict[int, np.ndarray],
    grid_shape: tuple[int, int] = (360, 500),
) -> BufferedScan:
    ref_data = SweepData(
        reflectivity=np.full(grid_shape, np.nan),
        azimuths=np.linspace(0, 359, grid_shape[0]),
        ranges_m=np.linspace(2000, 250000, grid_shape[1]),
        radar_lat=35.3331,
        radar_lon=-97.2778,
        elevation_angle=0.5,
        elevations=np.full(grid_shape[0], 0.5),
        elevation_angles=[0.5],
        radar_alt_m=390.0,
        timestamp=timestamp.isoformat(),
    )
    labeled_grid = np.zeros(grid_shape, dtype=int)
    for obj_id, mask in masks.items():
        labeled_grid[mask] = obj_id
    return BufferedScan(
        timestamp=timestamp,
        site_id=site_id,
        reflectivity_data=ref_data,
        detected_objects=objects,
        labeled_grid=labeled_grid,
        object_masks=masks,
    )


def test_associate_tracks_simple_global_match():
    tracker = StormTracker()
    t1 = datetime(2026, 4, 8, 18, 30)
    t2 = t1 + timedelta(minutes=5)
    prev_obj = _make_object(1, 35.5, -97.3)
    new_obj = _make_object(1, 35.51, -97.29)
    prev_mask = np.zeros((360, 500), dtype=bool)
    prev_mask[85:95, 195:205] = True
    new_mask = np.zeros((360, 500), dtype=bool)
    new_mask[86:96, 196:206] = True
    scan1 = _make_scan("KTLX", t1, [prev_obj], {1: prev_mask})
    scan2 = _make_scan("KTLX", t2, [new_obj], {1: new_mask})
    tracker.update(scan1)

    result = associate_tracks(scan1, scan2, tracker.all_tracks, tracker._obj_to_track)
    assert isinstance(result, AssociationResult)
    assert result.primary_matches == {1: 1}
    assert not result.merge_candidates
    assert not result.split_candidates


def test_associate_tracks_merge_candidate_has_single_survivor():
    tracker = StormTracker()
    t1 = datetime(2026, 4, 8, 18, 30)
    t2 = t1 + timedelta(minutes=5)
    obj_a = _make_object(1, 35.5, -97.3, peak_dbz=50.0)
    obj_b = _make_object(2, 35.5, -97.28, peak_dbz=40.0)
    mask_a = np.zeros((360, 500), dtype=bool)
    mask_a[85:95, 195:205] = True
    mask_b = np.zeros((360, 500), dtype=bool)
    mask_b[85:95, 205:215] = True
    merged_obj = _make_object(1, 35.5, -97.29, peak_dbz=52.0)
    merged_mask = np.zeros((360, 500), dtype=bool)
    merged_mask[85:95, 195:215] = True
    scan1 = _make_scan("KTLX", t1, [obj_a, obj_b], {1: mask_a, 2: mask_b})
    scan2 = _make_scan("KTLX", t2, [merged_obj], {1: merged_mask})
    tracker.update(scan1)

    result = associate_tracks(scan1, scan2, tracker.all_tracks, tracker._obj_to_track)
    assert 1 in result.merge_candidates
    related_tracks = result.merge_candidates[1]
    assert len(related_tracks) == len(set(related_tracks))
    assert len(related_tracks) >= 2


def test_associate_tracks_split_candidate_is_deduped():
    tracker = StormTracker()
    t1 = datetime(2026, 4, 8, 18, 30)
    t2 = t1 + timedelta(minutes=5)
    obj1 = _make_object(1, 35.5, -97.3, peak_dbz=50.0)
    mask1 = np.zeros((360, 500), dtype=bool)
    mask1[85:95, 195:215] = True
    obj_a = _make_object(1, 35.5, -97.31, peak_dbz=48.0, area_km2=60.0)
    obj_b = _make_object(2, 35.5, -97.28, peak_dbz=35.0, area_km2=40.0)
    mask_a = np.zeros((360, 500), dtype=bool)
    mask_a[85:95, 195:205] = True
    mask_b = np.zeros((360, 500), dtype=bool)
    mask_b[85:95, 205:215] = True
    scan1 = _make_scan("KTLX", t1, [obj1], {1: mask1})
    scan2 = _make_scan("KTLX", t2, [obj_a, obj_b], {1: mask_a, 2: mask_b})
    tracker.update(scan1)

    result = associate_tracks(scan1, scan2, tracker.all_tracks, tracker._obj_to_track)
    assert 1 in result.split_candidates
    related_new_ids = result.split_candidates[1]
    assert len(related_new_ids) == len(set(related_new_ids))
    assert len(related_new_ids) == 2


def test_compute_advected_iou_rewards_motion_aligned_overlap():
    prev_mask = np.zeros((20, 20), dtype=bool)
    prev_mask[5:8, 5:8] = True
    new_mask = np.zeros((20, 20), dtype=bool)
    new_mask[7:10, 8:11] = True
    assert compute_advected_iou(prev_mask, new_mask, shift_rows=2, shift_cols=3) > 0.9
    assert compute_advected_iou(prev_mask, new_mask, shift_rows=0, shift_cols=0) == 0.0


def test_associate_tracks_moves_the_previous_outline_by_the_tracks_measured_motion():
    from src.geometry import gate_ground_xy_m
    from src.tracking.motion import report_motion
    from src.tracking.types import VelocitySample

    tracker = StormTracker()
    t1 = datetime(2026, 4, 8, 18, 30)
    t2 = t1 + timedelta(minutes=5)

    prev_obj = _make_object(1, 35.5, -97.3)
    new_obj = _make_object(1, 35.52, -97.26)
    prev_mask = np.zeros((360, 500), dtype=bool)
    prev_mask[85:95, 195:205] = True
    new_mask = np.zeros((360, 500), dtype=bool)
    new_mask[89:99, 201:211] = True

    scan1 = _make_scan("KTLX", t1, [prev_obj], {1: prev_mask})
    scan2 = _make_scan("KTLX", t2, [new_obj], {1: new_mask})
    tracker.update(scan1)

    # The storm's measured velocity is the ground displacement between the outlines.
    sweep = scan1.reflectivity_data
    def centre(mask):
        rays, gates = np.nonzero(mask)
        x, y = gate_ground_xy_m(rays, gates, sweep.azimuths, sweep.ranges_m, sweep.elevations)
        return x.mean() / 1000.0, y.mean() / 1000.0
    (x0, y0), (x1, y1) = centre(prev_mask), centre(new_mask)
    hours = 5.0 / 60.0
    track = tracker.get_track(1)
    track.positions.append(track.positions[0])
    track.measured_velocities = [VelocitySample(timestamp=t1, east_kmh=(x1 - x0) / hours, north_kmh=(y1 - y0) / hours)]
    track.last_motion = report_motion(position_count=2, measured=track.measured_velocities, nearby=None, now=t1)

    result = associate_tracks(scan1, scan2, tracker.all_tracks, tracker._obj_to_track)

    assert result.primary_matches == {1: 1}
    score = result.candidate_scores[0]
    assert score.overlap_score < 0.5
    assert score.advected_overlap_score > 0.85


def test_a_track_with_unknown_motion_is_not_moved():
    tracker = StormTracker()
    t1 = datetime(2026, 4, 8, 18, 30)
    mask = np.zeros((360, 500), dtype=bool)
    mask[85:95, 195:205] = True
    scan1 = _make_scan("KTLX", t1, [_make_object(1, 35.5, -97.3)], {1: mask})
    scan2 = _make_scan("KTLX", t1 + timedelta(minutes=5), [_make_object(1, 35.5, -97.3)], {1: mask.copy()})
    tracker.update(scan1)
    assert tracker.get_track(1).get_motion().heading_label == "unknown"

    result = associate_tracks(scan1, scan2, tracker.all_tracks, tracker._obj_to_track)

    assert result.candidate_scores[0].advected_overlap_score == 1.0


def test_shifting_an_outline_wraps_around_ray_zero():
    from src.tracking.association import _shift_mask

    mask = np.zeros((360, 500), dtype=bool)
    mask[355:360, 100:110] = True
    shifted = _shift_mask(mask, 3, 0)
    assert shifted[358:360, 100:110].all() and shifted[0:3, 100:110].all()
    assert np.count_nonzero(shifted) == np.count_nonzero(mask)


def _block(rows: tuple[int, int], cols: tuple[int, int], shape=(360, 500)) -> np.ndarray:
    grid = np.zeros(shape, dtype=bool)
    grid[rows[0]:rows[1], cols[0]:cols[1]] = True
    return grid


def _storm_continuing_while_a_neighbour_absorbs_its_edge():
    """Storm A continues as Y; its edge overlaps X, which continues storm C."""
    t1 = datetime(2026, 4, 8, 18, 30)
    previous = _make_scan(
        "KTLX", t1,
        [_make_object(1, 35.30, -97.50, 50.0, 100.0), _make_object(2, 35.30, -97.45, 50.0, 100.0)],
        {1: _block((50, 70), (100, 120)), 2: _block((50, 70), (130, 150))},
    )
    current = _make_scan(
        "KTLX", t1 + timedelta(minutes=5),
        [_make_object(1, 35.30, -97.51, 50.0, 75.0), _make_object(2, 35.30, -97.47, 50.0, 170.0)],
        {1: _block((50, 70), (100, 115)), 2: _block((50, 70), (116, 150))},
    )
    return previous, current


def test_a_track_matched_to_its_own_object_is_never_a_merge_candidate():
    previous, current = _storm_continuing_while_a_neighbour_absorbs_its_edge()
    tracker = StormTracker()
    tracker.update(previous)

    association = associate_tracks(previous, current, tracker.all_tracks, tracker._obj_to_track)

    assert association.primary_matches == {1: 1, 2: 2}
    matched = set(association.primary_matches.values())
    for new_id, track_ids in association.merge_candidates.items():
        survivor, merged = track_ids[0], track_ids[1:]
        assert not (set(merged) & matched), f"object {new_id} lists matched tracks {merged} as merged"


def test_the_same_storm_keeps_its_track_when_a_neighbour_touches_it():
    previous, current = _storm_continuing_while_a_neighbour_absorbs_its_edge()
    tracker = StormTracker()
    tracker.update(previous)
    tracker.update(current)

    storm_a = tracker.get_track(1)
    assert storm_a.status == "active"
    assert storm_a.current_object.centroid_lon == -97.51
    assert not any(1 in event["involved_track_ids"] for event in tracker.recent_events if event["event_type"] == "merge")
    assert len(tracker.active_tracks) == len(current.detected_objects)


def test_an_object_matched_to_another_track_is_never_a_split_child(monkeypatch):
    """Global assignment gives U to Q although P scores U best.

    P -> U 0.10, P -> V 0.20, Q -> U 0.15.  The optimum is P -> V, Q -> U, so
    U continues storm Q and must not also become a split child of P.
    """
    import src.tracking.association as association_module
    from src.tracking.types import AssociationScore

    costs = {(1, 1): 0.10, (1, 2): 0.20, (2, 1): 0.15}

    def scripted_score(track, new_object, **_kwargs):
        cost = costs.get((track.track_id, new_object.object_id))
        if cost is None:
            return None
        return AssociationScore(track.track_id, new_object.object_id, 0.9, 0.9, 0.0, 0.0, 0.0, 0.0, cost)

    monkeypatch.setattr(association_module, "_candidate_score", scripted_score)
    t1 = datetime(2026, 4, 8, 18, 30)
    previous = _make_scan(
        "KTLX", t1,
        [_make_object(1, 35.30, -97.50), _make_object(2, 35.32, -97.52)],
        {1: _block((50, 70), (100, 120)), 2: _block((100, 120), (100, 120))},
    )
    current = _make_scan(
        "KTLX", t1 + timedelta(minutes=5),
        [_make_object(1, 35.31, -97.51), _make_object(2, 35.30, -97.49)],
        {1: _block((50, 60), (100, 120)), 2: _block((60, 70), (100, 120))},
    )
    tracker = StormTracker()
    tracker.update(previous)

    association = associate_tracks(previous, current, tracker.all_tracks, tracker._obj_to_track)

    assert association.primary_matches == {2: 1, 1: 2}
    for track_id, object_ids in association.split_candidates.items():
        children = object_ids[1:]
        assert all(association.primary_matches.get(child) in (None, track_id) for child in children), (
            f"track {track_id} lists objects matched to other tracks as split children: {children}"
        )


def _random_blob(rng, shape) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    if rng.random() < 0.05:
        return mask  # empty
    row, col = int(rng.integers(0, shape[0])), int(rng.integers(0, shape[1]))
    height, width = int(rng.integers(1, 40)), int(rng.integers(1, 40))
    mask[row:row + height, col:col + width] = rng.random((min(height, shape[0] - row), min(width, shape[1] - col))) < 0.8
    return mask


def test_windowed_overlap_scores_equal_the_full_grid_functions_exactly():
    from src.tracking.association import (
        _iou_from_extents,
        _overlap_from_extents,
        _shift_mask,
        compute_iou,
        compute_overlap,
        mask_extent,
    )

    rng = np.random.default_rng(12)
    shape = (120, 160)
    for _ in range(600):
        a, b = _random_blob(rng, shape), _random_blob(rng, shape)
        if rng.random() < 0.3:  # force overlap
            b = a.copy() if rng.random() < 0.5 else np.roll(a, (int(rng.integers(-5, 6)), int(rng.integers(-5, 6))), axis=(0, 1))
        shift_rows, shift_cols = float(rng.uniform(-60, 60)), float(rng.uniform(-80, 80))
        shifted = _shift_mask(a, shift_rows, shift_cols)
        assert _overlap_from_extents(mask_extent(a), mask_extent(b)) == compute_overlap(a, b)
        assert _iou_from_extents(mask_extent(shifted), mask_extent(b)) == compute_iou(shifted, b)


def test_each_track_mask_is_shifted_once_not_once_per_candidate(monkeypatch):
    import src.tracking.association as association_module

    t1 = datetime(2026, 4, 8, 18, 30)
    storms = 12
    masks = {i + 1: _block((10 + i * 28, 38 + i * 28), (100, 110)) for i in range(storms)}
    objects = [_make_object(i + 1, 35.0 + i * 0.3, -97.0) for i in range(storms)]
    previous = _make_scan("KTLX", t1, objects, masks)
    current = _make_scan("KTLX", t1 + timedelta(minutes=5), [_make_object(i + 1, 35.0 + i * 0.3, -97.0) for i in range(storms)], dict(masks))
    tracker = StormTracker()
    tracker.update(previous)

    shifts = []
    real_shift = association_module._shift_mask
    monkeypatch.setattr(association_module, "_shift_mask", lambda *args: shifts.append(args) or real_shift(*args))
    association = associate_tracks(previous, current, tracker.all_tracks, tracker._obj_to_track)

    assert association.primary_matches == {i + 1: i + 1 for i in range(storms)}
    assert len(shifts) == storms
