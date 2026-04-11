from datetime import datetime, timedelta

from src.detection import DetectedObject
from src.tracker import StormTracker


def _make_object(
    obj_id: int,
    distance_km: float,
    bearing_deg: float,
    peak_dbz: float,
    peak_label: str,
    area_km2: float,
) -> DetectedObject:
    return DetectedObject(
        object_id=obj_id,
        centroid_lat=35.0,
        centroid_lon=-97.0,
        distance_km=distance_km,
        bearing_deg=bearing_deg,
        peak_dbz=peak_dbz,
        peak_label=peak_label,
        area_km2=area_km2,
        layers=[],
    )


def test_update_primary_focus_prefers_nearer_relevant_track():
    tracker = StormTracker()
    distant = tracker._create_track(
        datetime(2026, 4, 8, 18, 0),
        _make_object(1, 250.0, 90.0, 45.0, "heavy rain", 140.0),
    )
    nearer = tracker._create_track(
        datetime(2026, 4, 8, 18, 0),
        _make_object(2, 30.0, 90.0, 58.0, "severe core", 220.0),
    )
    distant.identity_confidence = 0.95
    distant.is_primary_focus = True
    nearer.identity_confidence = 0.95
    for minutes in range(5, 20, 5):
        distant.add_position(datetime(2026, 4, 8, 18, 0) + timedelta(minutes=minutes), distant.current_object)
        nearer.add_position(datetime(2026, 4, 8, 18, 0) + timedelta(minutes=minutes), nearer.current_object)

    tracker._update_primary_focus()

    assert nearer.is_primary_focus is True
    assert distant.is_primary_focus is False


def test_update_primary_focus_keeps_existing_focus_without_clear_winner():
    tracker = StormTracker()
    current_focus = tracker._create_track(
        datetime(2026, 4, 8, 18, 0),
        _make_object(1, 60.0, 0.0, 52.0, "intense rain", 220.0),
    )
    challenger = tracker._create_track(
        datetime(2026, 4, 8, 18, 0),
        _make_object(2, 55.0, 10.0, 53.0, "intense rain", 215.0),
    )
    current_focus.identity_confidence = 0.9
    challenger.identity_confidence = 0.9
    current_focus.is_primary_focus = True
    for minutes in range(5, 20, 5):
        current_focus.add_position(datetime(2026, 4, 8, 18, 0) + timedelta(minutes=minutes), current_focus.current_object)
        challenger.add_position(datetime(2026, 4, 8, 18, 0) + timedelta(minutes=minutes), challenger.current_object)

    tracker._update_primary_focus()

    assert current_focus.is_primary_focus is True
    assert challenger.is_primary_focus is False
