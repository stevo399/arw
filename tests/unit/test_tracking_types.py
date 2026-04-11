from datetime import datetime

from src.tracking.events import normalize_merge_event, normalize_split_event
from src.tracking.types import AssociationScore, MotionConfidence, Track
from src.detection import DetectedObject


def _make_object() -> DetectedObject:
    return DetectedObject(
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


def test_track_shared_type_add_position():
    track = Track(track_id=1, status="active")
    now = datetime(2026, 4, 10, 18, 0)
    track.add_position(now, _make_object())
    assert len(track.positions) == 1
    assert len(track.peak_history) == 1
    assert track.first_seen == now
    assert track.last_seen == now


def test_association_score_type():
    score = AssociationScore(
        track_id=1,
        object_id=2,
        overlap_score=0.7,
        advected_overlap_score=0.8,
        distance_score=0.2,
        predicted_position_score=0.1,
        area_change_score=0.05,
        intensity_change_score=0.03,
        total_cost=0.18,
    )
    assert score.track_id == 1
    assert score.object_id == 2
    assert score.total_cost == 0.18


def test_motion_confidence_type():
    confidence = MotionConfidence(label="uncertain", score=0.2, reason="short history")
    assert confidence.label == "uncertain"
    assert confidence.reason == "short history"


def test_normalize_merge_event_dedupes_and_removes_survivor():
    now = datetime(2026, 4, 10, 18, 0)
    event = normalize_merge_event(now, surviving_track_id=5, merged_track_ids=[5, 7, 7, 8])
    assert event is not None
    assert event["involved_track_ids"] == [5, 7, 8]
    assert "5" not in event["description"].split("merged into")[0]


def test_normalize_split_event_dedupes_and_removes_parent():
    now = datetime(2026, 4, 10, 18, 0)
    event = normalize_split_event(now, parent_track_id=3, child_track_ids=[3, 4, 4, 5])
    assert event is not None
    assert event["involved_track_ids"] == [3, 4, 5]
    assert event["event_type"] == "split"


def test_normalize_merge_event_returns_none_without_children():
    now = datetime(2026, 4, 10, 18, 0)
    assert normalize_merge_event(now, surviving_track_id=5, merged_track_ids=[5, 5]) is None
