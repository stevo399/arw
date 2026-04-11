from datetime import datetime, timedelta

import numpy as np

from src.buffer import BufferedScan
from src.detection import DetectedObject
from src.parser import ReflectivityData
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
        peak_label="heavy rain",
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
    ref_data = ReflectivityData(
        reflectivity=np.full(grid_shape, np.nan),
        azimuths=np.linspace(0, 359, grid_shape[0]),
        ranges_m=np.linspace(2000, 250000, grid_shape[1]),
        radar_lat=35.3331,
        radar_lon=-97.2778,
        elevation_angle=0.5,
        elevation_angles=[0.5],
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


def test_associate_tracks_uses_advected_geometry_to_keep_match():
    tracker = StormTracker()
    t1 = datetime(2026, 4, 8, 18, 30)
    t2 = t1 + timedelta(minutes=5)

    reflectivity1 = np.full((360, 500), np.nan)
    reflectivity2 = np.full((360, 500), np.nan)
    reflectivity1[85:95, 195:205] = 45.0
    reflectivity2[89:99, 201:211] = 45.0

    prev_obj = _make_object(1, 35.5, -97.3)
    new_obj = _make_object(1, 35.52, -97.26)
    prev_mask = np.zeros((360, 500), dtype=bool)
    prev_mask[85:95, 195:205] = True
    new_mask = np.zeros((360, 500), dtype=bool)
    new_mask[89:99, 201:211] = True

    scan1 = _make_scan("KTLX", t1, [prev_obj], {1: prev_mask})
    scan2 = _make_scan("KTLX", t2, [new_obj], {1: new_mask})
    scan1.reflectivity_data.reflectivity = reflectivity1
    scan2.reflectivity_data.reflectivity = reflectivity2
    tracker.update(scan1)

    from src.tracking.motion_field import MotionFieldEstimate
    from unittest.mock import patch

    with patch(
        "src.tracking.association.estimate_motion_field",
        return_value=MotionFieldEstimate(
            shift_rows=4.0,
            shift_cols=6.0,
            quality=0.9,
            source="phase_correlation",
            downsample=4,
        ),
    ):
        result = associate_tracks(scan1, scan2, tracker.all_tracks, tracker._obj_to_track)
    assert result.primary_matches == {1: 1}
    score = result.candidate_scores[0]
    assert score.advected_overlap_score > score.overlap_score
