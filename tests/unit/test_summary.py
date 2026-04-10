# tests/unit/test_summary.py
from src.summary import generate_summary, km_to_miles
from src.detection import DetectedObject, IntensityLayerData, degrees_to_bearing
from src.motion import MotionVector
from src.tracker import Track


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
        track.current_object = obj
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
