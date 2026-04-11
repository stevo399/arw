from datetime import datetime

import numpy as np

from src.buffer import BufferedScan
from src.detection import DetectedObject
from src.parser import ReflectivityData
from src.tracker import StormTracker
from src.tracking.types import IdentityConfidence


def _make_object(obj_id: int, lat: float, lon: float, peak_dbz: float = 45.0) -> DetectedObject:
    return DetectedObject(
        object_id=obj_id,
        centroid_lat=lat,
        centroid_lon=lon,
        distance_km=40.0,
        bearing_deg=270.0,
        peak_dbz=peak_dbz,
        peak_label="heavy rain",
        area_km2=100.0,
        layers=[],
    )


def _make_scan(
    site_id: str,
    timestamp: datetime,
    objects: list[DetectedObject],
    masks: dict[int, np.ndarray],
) -> BufferedScan:
    grid_shape = (360, 500)
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


def test_tracker_uses_motion_field_for_weak_identity_track_motion():
    tracker = StormTracker()
    t1 = datetime(2026, 4, 8, 18, 0)
    t2 = datetime(2026, 4, 8, 18, 5)

    obj1 = _make_object(1, 35.0, -97.0)
    obj2 = _make_object(1, 36.8, -94.0)
    mask1 = np.zeros((360, 500), dtype=bool)
    mask1[85:95, 195:205] = True
    mask2 = np.zeros((360, 500), dtype=bool)
    mask2[86:96, 196:206] = True
    scan1 = _make_scan("KTLX", t1, [obj1], masks={1: mask1})
    scan2 = _make_scan("KTLX", t2, [obj2], masks={1: mask2})

    tracker.update(scan1)
    tracker.update(scan2)

    track = tracker.active_tracks[0]
    track.identity_confidence = 0.2
    tracker._refresh_track_motions(
        field_estimates={
            track.track_id: type(
                "Field",
                (),
                {"delta_lat": 0.02, "delta_lon": 0.0, "quality": 0.8, "source": "object_weighted_centroid"},
            )()
        },
        field_dt_hours=1.0,
    )
    motion = track.get_motion()
    assert motion.source == "motion_field"
    assert track.diagnostic_motion is not None
    assert track.diagnostic_motion.source == "track_history"


def test_tracker_suppresses_history_motion_for_fragile_merge_survivor():
    tracker = StormTracker()
    t1 = datetime(2026, 4, 8, 18, 0)
    t2 = datetime(2026, 4, 8, 18, 5)
    t3 = datetime(2026, 4, 8, 18, 10)

    obj1 = _make_object(1, 35.0, -97.0)
    obj2 = _make_object(1, 35.5, -96.6)
    obj3 = _make_object(1, 35.8, -96.2)
    mask1 = np.zeros((360, 500), dtype=bool)
    mask1[85:95, 195:205] = True
    mask2 = np.zeros((360, 500), dtype=bool)
    mask2[86:96, 196:206] = True
    mask3 = np.zeros((360, 500), dtype=bool)
    mask3[87:97, 197:207] = True
    scan1 = _make_scan("KTLX", t1, [obj1], masks={1: mask1})
    scan2 = _make_scan("KTLX", t2, [obj2], masks={1: mask2})
    scan3 = _make_scan("KTLX", t3, [obj3], masks={1: mask3})

    tracker.update(scan1)
    tracker.update(scan2)
    tracker.update(scan3)

    track = tracker.active_tracks[0]
    track.identity_confidence = 0.7
    track.identity_diagnostics = IdentityConfidence(
        label="medium",
        score=0.7,
        reason="fragile continuity",
        ambiguity_margin=0.2,
        event_context="merge_survivor",
    )
    tracker._recent_events = [{"event_type": "merge"} for _ in range(6)]
    tracker._refresh_track_motions(field_estimates=None, field_dt_hours=1.0)

    motion = track.get_motion()
    assert motion.source == "suppressed"
    assert motion.heading_label == "uncertain"

