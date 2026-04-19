# tests/unit/test_summary.py
from datetime import datetime, timedelta

from src.summary import generate_summary, km_to_miles
from src.detection import DetectedObject
from src.motion import MotionVector
from src.tracker import Track
from src.velocity import RotationSignature


def _make_object(obj_id=1, distance_km=40.2, bearing_deg=270.0, peak_dbz=45.0,
                 peak_label="heavy rain", area_km2=120.5) -> DetectedObject:
    return DetectedObject(
        object_id=obj_id,
        centroid_lat=35.5,
        centroid_lon=-97.3,
        distance_km=distance_km,
        bearing_deg=bearing_deg,
        peak_dbz=peak_dbz,
        peak_label=peak_label,
        area_km2=area_km2,
        layers=[],
    )


def _make_track(track_id=1, obj: DetectedObject | None = None,
                speed_kmh=56.3, heading_label="NE") -> Track:
    track = Track(track_id=track_id, status="active")
    if obj is not None:
        track.add_position(datetime(2026, 4, 8, 18, 0), obj)
    return track


def test_km_to_miles():
    assert km_to_miles(1.60934) == 1
    assert km_to_miles(0.0) == 0
    assert km_to_miles(100.0) == 62


def test_generate_summary_no_objects():
    text = generate_summary(
        site_id="KTLX",
        site_name="Oklahoma City",
        timestamp="2026-04-08T18:30:00Z",
        objects=[],
    )
    assert text == "Oklahoma City: No significant precipitation detected."


def test_generate_summary_single_object_no_tracks():
    """Phase 1 behavior: no tracks passed, no motion info."""
    obj = _make_object()
    text = generate_summary(
        site_id="KTLX",
        site_name="Oklahoma City",
        timestamp="2026-04-08T18:30:00Z",
        objects=[obj],
    )
    assert "Oklahoma City" in text
    assert "1 rain object" in text
    assert "heavy rain" in text
    assert "25 miles" in text
    assert "W" in text
    assert "47 square miles" in text
    assert "moving" not in text


def test_generate_summary_with_motion():
    """Phase 2: tracks passed, motion info included."""
    obj = _make_object()
    track = _make_track(obj=obj)
    track._motion_override = MotionVector(speed_kmh=56.3, speed_mph=35, heading_deg=45.0, heading_label="NE")
    text = generate_summary(
        site_id="KTLX",
        site_name="Oklahoma City",
        timestamp="2026-04-08T18:30:00Z",
        objects=[obj],
        tracks=[track],
    )
    assert "moving NE at 35 mph" in text


def test_generate_summary_stationary():
    """Stationary track shows 'stationary'."""
    obj = _make_object()
    track = _make_track(obj=obj)
    track._motion_override = MotionVector(speed_kmh=0.0, speed_mph=0, heading_deg=None, heading_label="stationary")
    text = generate_summary(
        site_id="KTLX",
        site_name="Oklahoma City",
        timestamp="2026-04-08T18:30:00Z",
        objects=[obj],
        tracks=[track],
    )
    assert "stationary" in text


def test_generate_summary_with_merge_event():
    obj = _make_object()
    track = _make_track(obj=obj)
    track._motion_override = MotionVector(speed_kmh=56.3, speed_mph=35, heading_deg=45.0, heading_label="NE")
    events = [{"event_type": "merge", "description": "Tracks 2, 3 merged into track 1"}]
    text = generate_summary(
        site_id="KTLX",
        site_name="Oklahoma City",
        timestamp="2026-04-08T18:30:00Z",
        objects=[obj],
        tracks=[track],
        events=events,
    )
    assert "merged" in text.lower()


def test_generate_summary_uncertain_motion():
    obj = _make_object()
    track = _make_track(obj=obj)
    track._motion_override = MotionVector(
        speed_kmh=220.0,
        speed_mph=137,
        heading_deg=None,
        heading_label="uncertain",
    )
    text = generate_summary(
        site_id="KTLX",
        site_name="Oklahoma City",
        timestamp="2026-04-08T18:30:00Z",
        objects=[obj],
        tracks=[track],
    )
    assert "tracking uncertain" in text
    assert "moving" not in text


def test_generate_summary_downgrades_low_identity_primary_focus_motion():
    obj = _make_object()
    track = _make_track(obj=obj)
    track.is_primary_focus = True
    track.identity_confidence = 0.4
    track.focus_continuity = type("Focus", (), {"score": 0.4, "recent_structural_event_count": 1})()
    track._motion_override = MotionVector(speed_kmh=56.3, speed_mph=35, heading_deg=45.0, heading_label="NE")
    text = generate_summary(
        site_id="KTLX",
        site_name="Oklahoma City",
        timestamp="2026-04-08T18:30:00Z",
        objects=[obj],
        tracks=[track],
        events=[{"event_type": "merge", "description": "Tracks 2, 3 merged into track 1"}],
    )
    assert "tracking uncertain" in text
    assert "moving NE" not in text


def test_generate_summary_downgrades_primary_focus_motion_under_high_event_pressure():
    obj = _make_object()
    track = _make_track(obj=obj)
    track.is_primary_focus = True
    track.identity_confidence = 0.7
    track.focus_continuity = type("Focus", (), {"score": 0.6, "recent_structural_event_count": 6})()
    track._motion_override = MotionVector(speed_kmh=32.0, speed_mph=20, heading_deg=292.0, heading_label="WNW")
    events = [{"event_type": "merge", "description": "merge"} for _ in range(4)] + [
        {"event_type": "split", "description": "split"} for _ in range(2)
    ]
    text = generate_summary(
        site_id="KTLX",
        site_name="Oklahoma City",
        timestamp="2026-04-08T18:30:00Z",
        objects=[obj],
        tracks=[track],
        events=events,
    )
    assert "tracking uncertain" in text
    assert "moving WNW" not in text


def test_generate_summary_downgrades_primary_focus_motion_under_repeated_heading_reversals():
    obj = _make_object()
    track = _make_track(obj=obj)
    track.is_primary_focus = True
    track.identity_confidence = 0.85
    track.focus_continuity = type(
        "Focus",
        (),
        {"score": 0.7, "recent_structural_event_count": 6, "recent_heading_flip_count": 2},
    )()
    track._motion_override = MotionVector(speed_kmh=32.0, speed_mph=20, heading_deg=135.0, heading_label="SE")
    events = [{"event_type": "merge", "description": "merge"} for _ in range(4)] + [
        {"event_type": "split", "description": "split"} for _ in range(2)
    ]
    text = generate_summary(
        site_id="KTLX",
        site_name="Oklahoma City",
        timestamp="2026-04-08T18:30:00Z",
        objects=[obj],
        tracks=[track],
        events=events,
    )
    assert "tracking uncertain" in text
    assert "moving SE" not in text


def test_generate_summary_multiple_objects():
    obj1 = _make_object(obj_id=1, peak_dbz=55.0, peak_label="intense rain", area_km2=200.0)
    obj2 = _make_object(obj_id=2, peak_dbz=30.0, peak_label="moderate rain", area_km2=50.0)
    text = generate_summary(
        site_id="KTLX",
        site_name="Oklahoma City",
        timestamp="2026-04-08T18:30:00Z",
        objects=[obj1, obj2],
    )
    assert "2 rain objects" in text
    assert "intense rain" in text


def test_generate_summary_uses_total_coverage_area():
    obj1 = _make_object(obj_id=1, peak_dbz=55.0, peak_label="intense rain", area_km2=200.0)
    obj2 = _make_object(obj_id=2, peak_dbz=30.0, peak_label="moderate rain", area_km2=50.0)
    text = generate_summary(
        site_id="KTLX",
        site_name="Oklahoma City",
        timestamp="2026-04-08T18:30:00Z",
        objects=[obj1, obj2],
    )
    assert "97 square miles" in text


def test_generate_summary_prefers_stable_larger_tracked_object_over_tiny_peak_spike():
    stable = _make_object(
        obj_id=1,
        distance_km=30.0,
        bearing_deg=45.0,
        peak_dbz=58.0,
        peak_label="intense rain",
        area_km2=800.0,
    )
    spiky = _make_object(
        obj_id=2,
        distance_km=120.0,
        bearing_deg=90.0,
        peak_dbz=61.0,
        peak_label="severe core",
        area_km2=8.0,
    )
    stable_track = _make_track(track_id=1, obj=stable)
    for minutes in range(5, 25, 5):
        stable_track.add_position(datetime(2026, 4, 8, 18, 0) + timedelta(minutes=minutes), stable)

    text = generate_summary(
        site_id="KTLX",
        site_name="Oklahoma City",
        timestamp="2026-04-08T18:30:00Z",
        objects=[spiky, stable],
        tracks=[stable_track],
    )
    assert "intense rain" in text
    assert "19 miles NE of the radar" in text


def test_generate_summary_prefers_primary_focus_track():
    focused = _make_object(
        obj_id=1,
        distance_km=40.0,
        bearing_deg=90.0,
        peak_dbz=55.0,
        peak_label="intense rain",
        area_km2=250.0,
    )
    challenger = _make_object(
        obj_id=2,
        distance_km=20.0,
        bearing_deg=45.0,
        peak_dbz=58.0,
        peak_label="severe core",
        area_km2=120.0,
    )
    focus_track = _make_track(track_id=1, obj=focused)
    for minutes in range(5, 25, 5):
        focus_track.add_position(datetime(2026, 4, 8, 18, 0) + timedelta(minutes=minutes), focused)
    focus_track.identity_confidence = 0.95
    focus_track.is_primary_focus = True

    challenger_track = _make_track(track_id=2, obj=challenger)
    challenger_track.identity_confidence = 0.8

    text = generate_summary(
        site_id="KTLX",
        site_name="Oklahoma City",
        timestamp="2026-04-08T18:30:00Z",
        objects=[challenger, focused],
        tracks=[focus_track, challenger_track],
    )
    assert "intense rain" in text
    assert "25 miles E of the radar" in text


def test_generate_summary_uses_primary_focus_object_directly_when_available():
    focused = _make_object(
        obj_id=1,
        distance_km=60.0,
        bearing_deg=0.0,
        peak_dbz=50.0,
        peak_label="intense rain",
        area_km2=180.0,
    )
    challenger = _make_object(
        obj_id=2,
        distance_km=20.0,
        bearing_deg=90.0,
        peak_dbz=60.0,
        peak_label="severe core",
        area_km2=300.0,
    )
    focus_track = _make_track(track_id=1, obj=focused)
    focus_track.is_primary_focus = True

    challenger_track = _make_track(track_id=2, obj=challenger)

    text = generate_summary(
        site_id="KTLX",
        site_name="Oklahoma City",
        timestamp="2026-04-08T18:30:00Z",
        objects=[challenger, focused],
        tracks=[focus_track, challenger_track],
    )
    assert "intense rain" in text
    assert "37 miles N of the radar" in text


def _make_object_with_rotation(strength="moderate"):
    return DetectedObject(
        object_id=1, centroid_lat=35.5, centroid_lon=-97.0,
        distance_km=50.0, bearing_deg=90.0,
        peak_dbz=55.0, peak_label="intense rain", area_km2=100.0,
        rotation=RotationSignature(
            centroid_lat=35.5, centroid_lon=-97.0,
            distance_km=50.0, bearing_deg=90.0,
            max_shear_ms=30.0, max_inbound_ms=-20.0, max_outbound_ms=15.0,
            diameter_km=3.0, sweep_count=2, elevation_angles=[0.5, 1.5],
            strength=strength,
        ),
    )


def test_summary_includes_rotation_for_strongest_object():
    obj = _make_object_with_rotation("moderate")
    text = generate_summary(
        site_id="KTLX", site_name="Oklahoma City",
        timestamp="2026-04-10T21:00:00Z", objects=[obj],
    )
    assert "rotation" in text.lower()


def test_summary_includes_rotation_strength():
    obj = _make_object_with_rotation("strong")
    text = generate_summary(
        site_id="KTLX", site_name="Oklahoma City",
        timestamp="2026-04-10T21:00:00Z", objects=[obj],
    )
    assert "strong rotation" in text.lower()


def test_summary_no_rotation_when_none():
    obj = DetectedObject(
        object_id=1, centroid_lat=35.5, centroid_lon=-97.0,
        distance_km=50.0, bearing_deg=90.0,
        peak_dbz=55.0, peak_label="intense rain", area_km2=100.0,
    )
    text = generate_summary(
        site_id="KTLX", site_name="Oklahoma City",
        timestamp="2026-04-10T21:00:00Z", objects=[obj],
    )
    assert "rotation" not in text.lower()
