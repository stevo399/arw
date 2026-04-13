from datetime import datetime
from pathlib import Path

import numpy as np

from src.buffer import BufferedScan
from src.detection import DetectedObject
from src.motion import MotionVector
from src.parser import ReflectivityData
from src.preprocess import ScanQuality
from src.tracker import StormTracker, Track
from scripts.live_replay import _cached_scans, _local_only_scans, _select_scan_count, summarize_scan


def _make_buffered_scan() -> BufferedScan:
    reflectivity = ReflectivityData(
        reflectivity=np.full((360, 500), np.nan),
        azimuths=np.linspace(0, 359, 360),
        ranges_m=np.linspace(2000, 250000, 500),
        radar_lat=35.0,
        radar_lon=-97.0,
        elevation_angle=0.5,
        elevation_angles=[0.5],
        timestamp="2026-04-10T20:00:00Z",
    )
    detected = DetectedObject(
        object_id=1,
        centroid_lat=35.5,
        centroid_lon=-97.3,
        distance_km=40.0,
        bearing_deg=270.0,
        peak_dbz=45.0,
        peak_label="heavy rain",
        area_km2=100.0,
        layers=[],
    )
    mask = np.zeros((360, 500), dtype=bool)
    mask[85:95, 195:205] = True
    labeled = np.zeros((360, 500), dtype=int)
    labeled[mask] = 1
    return BufferedScan(
        timestamp=datetime(2026, 4, 10, 20, 0),
        site_id="KTLX",
        reflectivity_data=reflectivity,
        detected_objects=[detected],
        labeled_grid=labeled,
        object_masks={1: mask},
        scan_quality=ScanQuality(
            score=0.9,
            finite_fraction=1.0,
            removed_speckle_pixels=0,
            removed_speckle_fraction=0.0,
            flags=[],
        ),
    )


def test_summarize_scan_reports_motion_sanity_fields():
    buffered = _make_buffered_scan()
    tracker = StormTracker()
    tracker.update(buffered)
    track = tracker.active_tracks[0]
    track._motion_override = MotionVector(
        speed_kmh=220.0,
        speed_mph=137,
        heading_deg=None,
        heading_label="uncertain",
    )
    track.get_motion = lambda: track._motion_override
    diagnostics = summarize_scan("Oklahoma City", buffered, tracker)
    assert diagnostics.object_count == 1
    assert diagnostics.active_count == 1
    assert diagnostics.uncertain_tracks == 1
    assert diagnostics.max_speed_mph == 137
    assert diagnostics.focus_track_id == 1
    assert diagnostics.focus_identity_label is not None
    assert diagnostics.focus_continuity_label is not None
    assert diagnostics.scan_quality_score == 0.9
    assert diagnostics.scan_quality_flags == []
    assert "tracking uncertain" in diagnostics.summary


def test_select_scan_count_defaults_to_quick_window():
    class Args:
        scans = None
        quick = True

    assert _select_scan_count(Args()) == 3


def test_cached_scans_filters_missing_entries(monkeypatch):
    class Scan:
        def __init__(self, filename: str):
            self.filename = filename

    scans = [Scan("A"), Scan("B"), Scan("C")]
    monkeypatch.setattr("scripts.live_replay.scan_is_cached", lambda site_id, filename: filename in {"A", "C"})
    cached = _cached_scans("KTLX", scans)
    assert [scan.filename for scan in cached] == ["A", "C"]


def test_local_only_scans_falls_back_to_most_recent_cached_for_date(monkeypatch, tmp_path):
    site_cache_dir = tmp_path / "KTLX"
    site_cache_dir.mkdir()
    for filename in [
        "KTLX20260410_170000_V06",
        "KTLX20260410_171000_V06",
        "KTLX20260410_172000_V06",
    ]:
        (site_cache_dir / filename).write_text("")

    class Scan:
        def __init__(self, filename: str):
            self.filename = filename

    monkeypatch.setattr("scripts.live_replay._cached_scans", lambda site_id, scans: [])
    monkeypatch.setattr(
        "scripts.live_replay.get_cache_path",
        lambda site_id, filename: str(Path(tmp_path) / site_id / filename),
    )

    selected = _local_only_scans("KTLX", "2026-04-10", [Scan("missing")], 2)
    assert [scan.filename for scan in selected] == [
        "KTLX20260410_171000_V06",
        "KTLX20260410_172000_V06",
    ]
