# tests/unit/test_tracker.py
from datetime import datetime, timedelta
import numpy as np
from src.tracker import StormTracker, Track
from src.detection import DetectedObject
from src.buffer import BufferedScan
from src.parser import ReflectivityData


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
    grid_shape: tuple[int, int] = (360, 500),
    masks: dict[int, np.ndarray] | None = None,
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
    if masks is None:
        masks = {}
        for obj in objects:
            mask = np.zeros(grid_shape, dtype=bool)
            row = min(85 + (obj.object_id - 1) * 40, grid_shape[0] - 15)
            mask[row:row + 10, 195:205] = True
            labeled_grid[mask] = obj.object_id
            masks[obj.object_id] = mask
    else:
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


def test_tracker_first_scan_creates_tracks():
    tracker = StormTracker()
    t = datetime(2026, 4, 8, 18, 30)
    objects = [_make_object(1, 35.5, -97.3), _make_object(2, 35.8, -97.1)]
    scan = _make_scan("KTLX", t, objects)
    tracker.update(scan)
    assert len(tracker.active_tracks) == 2
    for track in tracker.active_tracks:
        assert track.status == "active"
        assert len(track.positions) == 1


def test_tracker_second_scan_updates_tracks():
    tracker = StormTracker()
    t1 = datetime(2026, 4, 8, 18, 30)
    t2 = datetime(2026, 4, 8, 18, 35)
    obj1 = _make_object(1, 35.5, -97.3)
    obj2 = _make_object(1, 35.51, -97.29)  # Slightly moved
    mask1 = np.zeros((360, 500), dtype=bool)
    mask1[85:95, 195:205] = True
    mask2 = np.zeros((360, 500), dtype=bool)
    mask2[86:96, 196:206] = True  # Overlaps with mask1
    scan1 = _make_scan("KTLX", t1, [obj1], masks={1: mask1})
    scan2 = _make_scan("KTLX", t2, [obj2], masks={1: mask2})
    tracker.update(scan1)
    tracker.update(scan2)
    assert len(tracker.active_tracks) == 1
    track = tracker.active_tracks[0]
    assert len(track.positions) == 2


def test_tracker_unmatched_object_creates_new_track():
    tracker = StormTracker()
    t1 = datetime(2026, 4, 8, 18, 30)
    t2 = datetime(2026, 4, 8, 18, 35)
    obj1 = _make_object(1, 35.5, -97.3)
    obj2 = _make_object(1, 36.5, -96.0)  # Far away, no overlap
    mask1 = np.zeros((360, 500), dtype=bool)
    mask1[85:95, 195:205] = True
    mask2 = np.zeros((360, 500), dtype=bool)
    mask2[300:310, 400:410] = True  # No overlap with mask1
    scan1 = _make_scan("KTLX", t1, [obj1], masks={1: mask1})
    scan2 = _make_scan("KTLX", t2, [obj2], masks={1: mask2})
    tracker.update(scan1)
    tracker.update(scan2)
    # Original track should be active (1 missed scan), new track created
    all_tracks = tracker.all_tracks
    assert len(all_tracks) >= 2


def test_tracker_lost_after_missed_scans():
    tracker = StormTracker()
    t1 = datetime(2026, 4, 8, 18, 30)
    obj1 = _make_object(1, 35.5, -97.3)
    mask1 = np.zeros((360, 500), dtype=bool)
    mask1[85:95, 195:205] = True
    scan1 = _make_scan("KTLX", t1, [obj1], masks={1: mask1})
    tracker.update(scan1)
    # Two empty scans = lost
    for i in range(2):
        t = t1 + timedelta(minutes=(i + 1) * 5)
        empty_scan = _make_scan("KTLX", t, [])
        tracker.update(empty_scan)
    lost = [t for t in tracker.all_tracks if t.status == "lost"]
    assert len(lost) == 1


def test_tracker_merge_detection():
    tracker = StormTracker()
    t1 = datetime(2026, 4, 8, 18, 30)
    t2 = datetime(2026, 4, 8, 18, 35)
    # Two separate objects in scan 1
    obj_a = _make_object(1, 35.5, -97.3, peak_dbz=50.0)
    obj_b = _make_object(2, 35.5, -97.28, peak_dbz=40.0)
    mask_a = np.zeros((360, 500), dtype=bool)
    mask_a[85:95, 195:205] = True
    mask_b = np.zeros((360, 500), dtype=bool)
    mask_b[85:95, 205:215] = True
    scan1 = _make_scan("KTLX", t1, [obj_a, obj_b], masks={1: mask_a, 2: mask_b})
    # One merged object in scan 2, overlapping both mask_a and mask_b
    obj_merged = _make_object(1, 35.5, -97.29, peak_dbz=52.0)
    mask_merged = np.zeros((360, 500), dtype=bool)
    mask_merged[85:95, 195:215] = True  # Covers both previous masks
    scan2 = _make_scan("KTLX", t2, [obj_merged], masks={1: mask_merged})
    tracker.update(scan1)
    tracker.update(scan2)
    merged = [t for t in tracker.all_tracks if t.status == "merged"]
    assert len(merged) >= 1
    events = tracker.recent_events
    merge_events = [e for e in events if e["event_type"] == "merge"]
    assert len(merge_events) >= 1


def test_tracker_split_detection():
    tracker = StormTracker()
    t1 = datetime(2026, 4, 8, 18, 30)
    t2 = datetime(2026, 4, 8, 18, 35)
    # One object in scan 1
    obj1 = _make_object(1, 35.5, -97.3, peak_dbz=50.0)
    mask1 = np.zeros((360, 500), dtype=bool)
    mask1[85:95, 195:215] = True
    scan1 = _make_scan("KTLX", t1, [obj1], masks={1: mask1})
    # Two objects in scan 2, each overlapping the original
    obj_a = _make_object(1, 35.5, -97.31, peak_dbz=48.0)
    obj_b = _make_object(2, 35.5, -97.28, peak_dbz=35.0)
    mask_a = np.zeros((360, 500), dtype=bool)
    mask_a[85:95, 195:205] = True  # Left half
    mask_b = np.zeros((360, 500), dtype=bool)
    mask_b[85:95, 205:215] = True  # Right half
    scan2 = _make_scan("KTLX", t2, [obj_a, obj_b], masks={1: mask_a, 2: mask_b})
    tracker.update(scan1)
    tracker.update(scan2)
    active = tracker.active_tracks
    assert len(active) >= 2
    events = tracker.recent_events
    split_events = [e for e in events if e["event_type"] == "split"]
    assert len(split_events) >= 1


def test_tracker_get_track_by_id():
    tracker = StormTracker()
    t = datetime(2026, 4, 8, 18, 30)
    scan = _make_scan("KTLX", t, [_make_object(1, 35.5, -97.3)])
    tracker.update(scan)
    track = tracker.get_track(1)
    assert track is not None
    assert track.track_id == 1
    assert tracker.get_track(999) is None
