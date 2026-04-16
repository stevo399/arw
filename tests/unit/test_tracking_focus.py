from datetime import datetime, timedelta

from src.detection import DetectedObject
from src.motion import MotionVector
from src.tracker import StormTracker
from src.tracking.types import IdentityConfidence, MotionConfidence, MotionSample


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


def test_build_focus_continuity_penalizes_repeated_recent_heading_reversals():
    tracker = StormTracker()
    track = tracker._create_track(
        datetime(2026, 4, 8, 18, 0),
        _make_object(1, 40.0, 90.0, 55.0, "intense rain", 180.0),
    )
    positions = [
        (35.00, -97.00),
        (35.05, -96.95),
        (35.10, -96.90),
        (35.05, -96.98),
        (35.12, -96.92),
    ]
    base = datetime(2026, 4, 8, 18, 0)
    track.positions.clear()
    for index, (lat, lon) in enumerate(positions):
        obj = DetectedObject(
            object_id=index + 1,
            centroid_lat=lat,
            centroid_lon=lon,
            distance_km=40.0,
            bearing_deg=90.0,
            peak_dbz=55.0,
            peak_label="intense rain",
            area_km2=180.0,
            layers=[],
        )
        track.add_position(base + timedelta(minutes=index * 5), obj)
    track.identity_confidence = 0.85
    track.identity_diagnostics = IdentityConfidence(label="high", score=0.85)
    track.last_motion = MotionVector(
        speed_kmh=40.0,
        speed_mph=25,
        heading_deg=90.0,
        heading_label="E",
        confidence=MotionConfidence(label="high", score=0.9),
    )

    continuity = tracker._build_focus_continuity(track, previous_focus_track_id=track.track_id, structural_event_count=6)

    assert continuity.recent_heading_flip_count >= 2
    assert continuity.score <= 0.2
    assert continuity.label == "low"


def test_build_focus_continuity_ignores_heading_flip_penalty_for_stationaryish_motion():
    tracker = StormTracker()
    track = tracker._create_track(
        datetime(2026, 4, 8, 18, 0),
        _make_object(1, 40.0, 90.0, 55.0, "intense rain", 180.0),
    )
    positions = [
        (35.00, -97.00),
        (35.05, -96.95),
        (35.10, -96.90),
        (35.05, -96.98),
        (35.12, -96.92),
    ]
    base = datetime(2026, 4, 8, 18, 0)
    track.positions.clear()
    for index, (lat, lon) in enumerate(positions):
        obj = DetectedObject(
            object_id=index + 1,
            centroid_lat=lat,
            centroid_lon=lon,
            distance_km=40.0,
            bearing_deg=90.0,
            peak_dbz=55.0,
            peak_label="intense rain",
            area_km2=180.0,
            layers=[],
        )
        track.add_position(base + timedelta(minutes=index * 5), obj)
    track.identity_confidence = 0.8
    track.identity_diagnostics = IdentityConfidence(label="high", score=0.8)
    track.last_motion = MotionVector(
        speed_kmh=0.8,
        speed_mph=0,
        heading_deg=None,
        heading_label="nearly stationary",
        confidence=MotionConfidence(label="medium", score=1.0),
    )

    continuity = tracker._build_focus_continuity(track, previous_focus_track_id=track.track_id, structural_event_count=1)

    assert continuity.recent_heading_flip_count == 0
    assert continuity.score == 1.0
    assert continuity.label == "high"


def test_build_focus_continuity_penalizes_low_motion_confidence_under_high_structural_pressure():
    tracker = StormTracker()
    track = tracker._create_track(
        datetime(2026, 4, 8, 18, 0),
        _make_object(1, 40.0, 90.0, 55.0, "intense rain", 180.0),
    )
    track.identity_confidence = 0.85
    track.identity_diagnostics = IdentityConfidence(label="high", score=0.85)
    track.last_motion = MotionVector(
        speed_kmh=0.0,
        speed_mph=0,
        heading_deg=None,
        heading_label="uncertain",
        confidence=MotionConfidence(label="low", score=0.0),
    )

    continuity = tracker._build_focus_continuity(track, previous_focus_track_id=track.track_id, structural_event_count=6)

    assert continuity.score == 0.3
    assert continuity.label == "low"
    assert continuity.reason == "high structural pressure with low motion confidence"


def test_build_focus_continuity_records_crowded_challenger_pressure_diagnostics():
    tracker = StormTracker()
    track = tracker._create_track(
        datetime(2026, 4, 8, 18, 0),
        _make_object(1, 40.0, 90.0, 55.0, "intense rain", 180.0),
    )
    track.identity_confidence = 0.85
    track.identity_diagnostics = IdentityConfidence(label="high", score=0.85)
    track.last_motion = MotionVector(
        speed_kmh=20.0,
        speed_mph=12,
        heading_deg=90.0,
        heading_label="E",
        confidence=MotionConfidence(label="high", score=0.9),
    )

    continuity = tracker._build_focus_continuity(
        track,
        previous_focus_track_id=track.track_id,
        structural_event_count=6,
        selection_margin=2.4,
        runner_up_track_id=9,
    )

    assert continuity.selection_margin == 2.4
    assert continuity.runner_up_track_id == 9
    assert continuity.score == 0.7
    assert continuity.label == "medium"
    assert continuity.reason == "high structural event pressure around focus"


def test_build_focus_continuity_penalizes_reported_motion_reversal_under_structural_pressure():
    tracker = StormTracker()
    track = tracker._create_track(
        datetime(2026, 4, 8, 18, 0),
        _make_object(1, 40.0, 90.0, 55.0, "intense rain", 180.0),
    )
    track.identity_confidence = 0.85
    track.identity_diagnostics = IdentityConfidence(label="high", score=0.85)
    track.last_motion = MotionVector(
        speed_kmh=20.0,
        speed_mph=12,
        heading_deg=290.0,
        heading_label="WNW",
        confidence=MotionConfidence(label="high", score=0.98),
        source="motion_field",
    )
    track.motion_history.extend([
        MotionSample(
            timestamp=datetime(2026, 4, 8, 18, 0),
            heading_deg=140.0,
            heading_label="SE",
            source="motion_field",
            confidence_score=0.98,
        ),
        MotionSample(
            timestamp=datetime(2026, 4, 8, 18, 5),
            heading_deg=138.0,
            heading_label="SE",
            source="motion_field",
            confidence_score=0.98,
        ),
        MotionSample(
            timestamp=datetime(2026, 4, 8, 18, 10),
            heading_deg=290.0,
            heading_label="WNW",
            source="motion_field",
            confidence_score=0.98,
        ),
    ])

    continuity = tracker._build_focus_continuity(track, previous_focus_track_id=track.track_id, structural_event_count=6)

    assert continuity.recent_reported_heading_flip_count == 1
    assert continuity.reported_heading_stability_label == "mixed"
    assert continuity.reported_heading_stability_score == 0.45
    assert continuity.score == 0.5
    assert continuity.label == "medium"
    assert continuity.reason in {
        "mixed reported focus heading sequence under structural pressure",
        "high structural event pressure around focus",
    }


def test_build_focus_continuity_relaxes_raw_heading_flip_penalty_for_clear_focus_winner():
    tracker = StormTracker()
    track = tracker._create_track(
        datetime(2026, 4, 8, 18, 0),
        _make_object(1, 40.0, 90.0, 55.0, "intense rain", 180.0),
    )
    positions = [
        (35.00, -97.00),
        (35.05, -96.95),
        (35.10, -96.90),
        (35.03, -96.99),
        (35.08, -96.92),
    ]
    base = datetime(2026, 4, 8, 18, 0)
    track.positions.clear()
    for index, (lat, lon) in enumerate(positions):
        obj = DetectedObject(
            object_id=index + 1,
            centroid_lat=lat,
            centroid_lon=lon,
            distance_km=40.0,
            bearing_deg=90.0,
            peak_dbz=55.0,
            peak_label="intense rain",
            area_km2=180.0,
            layers=[],
        )
        track.add_position(base + timedelta(minutes=index * 5), obj)
    track.identity_confidence = 0.78
    track.identity_diagnostics = IdentityConfidence(label="high", score=0.78)
    track.last_motion = MotionVector(
        speed_kmh=20.0,
        speed_mph=12,
        heading_deg=140.0,
        heading_label="SE",
        confidence=MotionConfidence(label="high", score=0.98),
        source="motion_field",
    )
    track.motion_history.extend([
        MotionSample(
            timestamp=datetime(2026, 4, 8, 18, 0),
            heading_deg=138.0,
            heading_label="SE",
            source="motion_field",
            confidence_score=0.98,
        ),
        MotionSample(
            timestamp=datetime(2026, 4, 8, 18, 5),
            heading_deg=141.0,
            heading_label="SE",
            source="motion_field",
            confidence_score=0.98,
        ),
        MotionSample(
            timestamp=datetime(2026, 4, 8, 18, 10),
            heading_deg=143.0,
            heading_label="SE",
            source="motion_field",
            confidence_score=0.98,
        ),
    ])

    continuity = tracker._build_focus_continuity(
        track,
        previous_focus_track_id=track.track_id,
        structural_event_count=6,
        selection_margin=4.8,
        runner_up_track_id=11,
    )

    assert continuity.recent_heading_flip_count == 0
    assert continuity.recent_reported_heading_flip_count == 0
    assert continuity.reported_heading_stability_label == "stable"
    assert continuity.reported_heading_stability_score == 0.95
    assert continuity.score == 0.85
    assert continuity.label == "high"
    assert continuity.reason == "stable focus winner despite structural pressure"


def test_build_focus_continuity_keeps_reported_reversal_penalty_even_with_clear_focus_margin():
    tracker = StormTracker()
    track = tracker._create_track(
        datetime(2026, 4, 8, 18, 0),
        _make_object(1, 40.0, 90.0, 55.0, "intense rain", 180.0),
    )
    track.identity_confidence = 0.85
    track.identity_diagnostics = IdentityConfidence(label="high", score=0.85)
    track.last_motion = MotionVector(
        speed_kmh=20.0,
        speed_mph=12,
        heading_deg=290.0,
        heading_label="WNW",
        confidence=MotionConfidence(label="high", score=0.98),
        source="motion_field",
    )
    track.motion_history.extend([
        MotionSample(
            timestamp=datetime(2026, 4, 8, 18, 0),
            heading_deg=140.0,
            heading_label="SE",
            source="motion_field",
            confidence_score=0.98,
        ),
        MotionSample(
            timestamp=datetime(2026, 4, 8, 18, 5),
            heading_deg=138.0,
            heading_label="SE",
            source="motion_field",
            confidence_score=0.98,
        ),
        MotionSample(
            timestamp=datetime(2026, 4, 8, 18, 10),
            heading_deg=290.0,
            heading_label="WNW",
            source="motion_field",
            confidence_score=0.98,
        ),
    ])

    continuity = tracker._build_focus_continuity(
        track,
        previous_focus_track_id=track.track_id,
        structural_event_count=6,
        selection_margin=5.0,
        runner_up_track_id=9,
    )

    assert continuity.recent_reported_heading_flip_count == 1
    assert continuity.reported_heading_stability_label == "mixed"
    assert continuity.reported_heading_stability_score == 0.45
    assert continuity.score == 0.65
    assert continuity.label == "medium"
    assert continuity.reason == "high structural event pressure around focus"


def test_build_focus_continuity_treats_one_direction_turn_sequence_as_coherent():
    tracker = StormTracker()
    track = tracker._create_track(
        datetime(2026, 4, 8, 18, 0),
        _make_object(1, 40.0, 90.0, 55.0, "intense rain", 180.0),
    )
    track.identity_confidence = 0.85
    track.identity_diagnostics = IdentityConfidence(label="high", score=0.85)
    track.last_motion = MotionVector(
        speed_kmh=35.0,
        speed_mph=22,
        heading_deg=182.0,
        heading_label="S",
        confidence=MotionConfidence(label="high", score=0.98),
        source="motion_field",
    )
    track.motion_history.extend([
        MotionSample(
            timestamp=datetime(2026, 4, 8, 18, 0),
            heading_deg=48.0,
            heading_label="NE",
            source="motion_field",
            confidence_score=0.98,
        ),
        MotionSample(
            timestamp=datetime(2026, 4, 8, 18, 5),
            heading_deg=51.0,
            heading_label="NE",
            source="motion_field",
            confidence_score=0.98,
        ),
        MotionSample(
            timestamp=datetime(2026, 4, 8, 18, 10),
            heading_deg=126.0,
            heading_label="SE",
            source="motion_field",
            confidence_score=0.98,
        ),
        MotionSample(
            timestamp=datetime(2026, 4, 8, 18, 15),
            heading_deg=182.0,
            heading_label="S",
            source="motion_field",
            confidence_score=0.98,
        ),
    ])

    continuity = tracker._build_focus_continuity(
        track,
        previous_focus_track_id=track.track_id,
        structural_event_count=6,
        selection_margin=4.8,
        runner_up_track_id=11,
    )

    assert continuity.reported_heading_stability_label == "coherent_turn"
    assert continuity.reported_heading_stability_score == 0.85
    assert continuity.score == 0.85
    assert continuity.label == "high"
    assert continuity.reason == "stable focus winner despite structural pressure"
