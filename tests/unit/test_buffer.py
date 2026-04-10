# tests/unit/test_buffer.py
from datetime import datetime, timedelta
import numpy as np
from src.buffer import ReplayBuffer, BufferedScan
from src.parser import ReflectivityData
from src.detection import DetectedObject


def _make_buffered_scan(site_id: str, timestamp: datetime, num_objects: int = 1) -> BufferedScan:
    """Create a minimal BufferedScan for testing."""
    ref_data = ReflectivityData(
        reflectivity=np.full((360, 500), np.nan),
        azimuths=np.linspace(0, 359, 360),
        ranges_m=np.linspace(2000, 250000, 500),
        radar_lat=35.3331,
        radar_lon=-97.2778,
        elevation_angle=0.5,
        elevation_angles=[0.5],
        timestamp=timestamp.isoformat(),
    )
    objects = [
        DetectedObject(
            object_id=i + 1,
            centroid_lat=35.5 + i * 0.1,
            centroid_lon=-97.3,
            distance_km=40.0 + i * 10,
            bearing_deg=270.0,
            peak_dbz=45.0,
            peak_label="heavy rain",
            area_km2=100.0,
            layers=[],
        )
        for i in range(num_objects)
    ]
    labeled_grid = np.zeros((360, 500), dtype=int)
    object_masks = {}
    for i in range(num_objects):
        mask = np.zeros((360, 500), dtype=bool)
        mask[85 + i * 20:95 + i * 20, 195:205] = True
        labeled_grid[mask] = i + 1
        object_masks[i + 1] = mask
    return BufferedScan(
        timestamp=timestamp,
        site_id=site_id,
        reflectivity_data=ref_data,
        detected_objects=objects,
        labeled_grid=labeled_grid,
        object_masks=object_masks,
    )


def test_buffer_initially_empty():
    buf = ReplayBuffer()
    assert buf.scan_count == 0
    assert buf.current_scan is None
    assert buf.previous_scan is None


def test_buffer_add_scan():
    buf = ReplayBuffer()
    scan = _make_buffered_scan("KTLX", datetime(2026, 4, 8, 18, 30))
    buf.add_scan(scan)
    assert buf.scan_count == 1
    assert buf.current_scan is scan
    assert buf.previous_scan is None


def test_buffer_two_scans():
    buf = ReplayBuffer()
    scan1 = _make_buffered_scan("KTLX", datetime(2026, 4, 8, 18, 30))
    scan2 = _make_buffered_scan("KTLX", datetime(2026, 4, 8, 18, 35))
    buf.add_scan(scan1)
    buf.add_scan(scan2)
    assert buf.scan_count == 2
    assert buf.current_scan is scan2
    assert buf.previous_scan is scan1


def test_buffer_evicts_old_scans():
    buf = ReplayBuffer(max_age_minutes=120)
    base_time = datetime(2026, 4, 8, 16, 0)
    # Add scans spanning 3 hours
    for i in range(36):  # every 5 min for 3 hours
        scan = _make_buffered_scan("KTLX", base_time + timedelta(minutes=i * 5))
        buf.add_scan(scan)
    # Should have evicted scans older than 2 hours from the latest
    latest = buf.current_scan.timestamp
    for scan in buf.all_scans:
        age = latest - scan.timestamp
        assert age <= timedelta(minutes=120)


def test_buffer_resets_on_site_change():
    buf = ReplayBuffer()
    scan1 = _make_buffered_scan("KTLX", datetime(2026, 4, 8, 18, 30))
    buf.add_scan(scan1)
    assert buf.scan_count == 1
    scan2 = _make_buffered_scan("KFWS", datetime(2026, 4, 8, 18, 35))
    buf.add_scan(scan2)
    assert buf.scan_count == 1  # Reset, only new scan
    assert buf.current_scan.site_id == "KFWS"


def test_buffer_all_scans_ordered():
    buf = ReplayBuffer()
    times = [datetime(2026, 4, 8, 18, i * 5) for i in range(5)]
    for t in times:
        buf.add_scan(_make_buffered_scan("KTLX", t))
    scans = buf.all_scans
    for i in range(len(scans) - 1):
        assert scans[i].timestamp <= scans[i + 1].timestamp


def test_buffer_time_range():
    buf = ReplayBuffer()
    scan1 = _make_buffered_scan("KTLX", datetime(2026, 4, 8, 18, 0))
    scan2 = _make_buffered_scan("KTLX", datetime(2026, 4, 8, 18, 30))
    buf.add_scan(scan1)
    buf.add_scan(scan2)
    start, end = buf.time_range
    assert start == datetime(2026, 4, 8, 18, 0)
    assert end == datetime(2026, 4, 8, 18, 30)
