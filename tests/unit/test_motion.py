# tests/unit/test_motion.py
import math
from datetime import datetime, timedelta
from src.motion import compute_motion, MotionVector, motion_from_field, resolve_reported_motion
from src.tracking.motion import _recent_consensus_heading_deg
from src.tracking.motion_field import GeographicMotionFieldEstimate

NEARLY_STATIONARY_KMH = 2.0


def test_motion_single_point():
    """Single position = stationary."""
    positions = [
        (datetime(2026, 4, 8, 18, 30), 35.0, -97.0),
    ]
    motion = compute_motion(positions)
    assert isinstance(motion, MotionVector)
    assert motion.speed_kmh == 0.0
    assert motion.speed_mph == 0
    assert motion.heading_deg is None
    assert motion.heading_label == "stationary"


def test_motion_two_points_moving_north():
    """Two points moving north should give ~0 deg heading."""
    positions = [
        (datetime(2026, 4, 8, 18, 0), 35.0, -97.0),
        (datetime(2026, 4, 8, 18, 30), 35.5, -97.0),  # ~55 km north in 30 min
    ]
    motion = compute_motion(positions)
    assert motion.speed_kmh > 50
    assert motion.speed_kmh < 150
    assert motion.heading_deg is not None
    assert abs(motion.heading_deg - 0) < 20 or abs(motion.heading_deg - 360) < 20
    assert motion.heading_label in ("N", "NNE", "NNW")


def test_motion_two_points_moving_east():
    """Two points moving east should give ~90 deg heading."""
    positions = [
        (datetime(2026, 4, 8, 18, 0), 35.0, -97.0),
        (datetime(2026, 4, 8, 18, 30), 35.0, -96.5),  # ~45 km east in 30 min
    ]
    motion = compute_motion(positions)
    assert motion.speed_kmh > 40
    assert motion.heading_deg is not None
    assert 60 < motion.heading_deg < 120
    assert motion.heading_label in ("E", "ENE", "ESE")


def test_motion_nearly_stationary():
    """Barely moving should be labeled nearly stationary."""
    positions = [
        (datetime(2026, 4, 8, 18, 0), 35.0, -97.0),
        (datetime(2026, 4, 8, 18, 30), 35.0001, -97.0),  # ~11 meters in 30 min
    ]
    motion = compute_motion(positions)
    assert motion.speed_kmh < NEARLY_STATIONARY_KMH
    assert motion.heading_label == "nearly stationary"


def test_motion_three_points_smooths():
    """Three points should produce a smoother estimate than two."""
    positions = [
        (datetime(2026, 4, 8, 18, 0), 35.0, -97.0),
        (datetime(2026, 4, 8, 18, 5), 35.05, -97.0),
        (datetime(2026, 4, 8, 18, 10), 35.1, -97.0),
    ]
    motion = compute_motion(positions)
    assert motion.speed_kmh > 0
    assert motion.heading_deg is not None
    assert abs(motion.heading_deg - 0) < 20 or abs(motion.heading_deg - 360) < 20


def test_motion_mph_conversion():
    """Speed in mph should be km/h divided by 1.60934, rounded."""
    positions = [
        (datetime(2026, 4, 8, 18, 0), 35.0, -97.0),
        (datetime(2026, 4, 8, 19, 0), 35.0, -96.0),  # ~90 km east in 1 hour
    ]
    motion = compute_motion(positions)
    expected_mph = round(motion.speed_kmh / 1.60934)
    assert motion.speed_mph == expected_mph


def test_motion_absurd_speed_becomes_uncertain():
    positions = [
        (datetime(2026, 4, 8, 18, 0), 35.0, -97.0),
        (datetime(2026, 4, 8, 18, 5), 38.0, -92.0),
    ]
    motion = compute_motion(positions)
    assert motion.heading_label == "uncertain"
    assert motion.confidence is not None
    assert motion.confidence.label == "low"


def test_motion_inconsistent_steps_becomes_uncertain():
    positions = [
        (datetime(2026, 4, 8, 18, 0), 35.0, -97.0),
        (datetime(2026, 4, 8, 18, 5), 35.02, -97.0),
        (datetime(2026, 4, 8, 18, 10), 35.4, -97.0),
    ]
    motion = compute_motion(positions)
    assert motion.heading_label == "uncertain"
    assert motion.confidence is not None
    assert motion.confidence.label == "low"


def test_motion_inconsistent_step_headings_becomes_uncertain():
    positions = [
        (datetime(2026, 4, 8, 18, 0), 35.00, -97.00),
        (datetime(2026, 4, 8, 18, 5), 35.00, -96.94),
        (datetime(2026, 4, 8, 18, 10), 35.03, -96.98),
    ]
    motion = compute_motion(positions)
    assert motion.heading_label == "uncertain"
    assert motion.confidence is not None
    assert motion.confidence.reason == "step headings are inconsistent"


def test_motion_from_field_returns_publishable_vector():
    estimate = GeographicMotionFieldEstimate(
        delta_lat=0.1,
        delta_lon=0.0,
        quality=0.8,
        source="object_weighted_centroid",
    )
    motion = motion_from_field(estimate, dt_hours=1.0)
    assert motion is not None
    assert motion.source == "motion_field"
    assert motion.heading_label in ("N", "NNE", "NNW")
    assert motion.confidence is not None
    assert motion.confidence.score >= 0.8


def test_motion_from_field_suppresses_implausible_speed():
    estimate = GeographicMotionFieldEstimate(
        delta_lat=5.0,
        delta_lon=0.0,
        quality=0.9,
        source="phase_correlation",
    )
    motion = motion_from_field(estimate, dt_hours=1.0)
    assert motion is not None
    assert motion.source == "suppressed"
    assert motion.heading_label == "uncertain"
    assert motion.speed_mph == 0


def test_resolve_reported_motion_uses_field_when_identity_is_weak():
    positions = [
        (datetime(2026, 4, 8, 18, 0), 35.0, -97.0),
        (datetime(2026, 4, 8, 18, 5), 38.0, -92.0),
    ]
    field = GeographicMotionFieldEstimate(
        delta_lat=0.02,
        delta_lon=0.0,
        quality=0.7,
        source="object_weighted_centroid",
    )
    reported, diagnostic = resolve_reported_motion(
        positions,
        identity_confidence=0.2,
        field_estimate=field,
        field_dt_hours=1.0,
    )
    assert diagnostic.source == "track_history"
    assert diagnostic.heading_label == "uncertain"
    assert reported.source == "motion_field"
    assert reported.heading_label in ("N", "NNE", "NNW")


def test_resolve_reported_motion_suppresses_when_identity_and_field_are_weak():
    positions = [
        (datetime(2026, 4, 8, 18, 0), 35.0, -97.0),
        (datetime(2026, 4, 8, 18, 5), 38.0, -92.0),
    ]
    field = GeographicMotionFieldEstimate(
        delta_lat=0.02,
        delta_lon=0.0,
        quality=0.1,
        source="object_weighted_centroid",
    )
    reported, diagnostic = resolve_reported_motion(
        positions,
        identity_confidence=0.2,
        field_estimate=field,
        field_dt_hours=1.0,
    )
    assert diagnostic.heading_label == "uncertain"
    assert reported.source == "suppressed"
    assert reported.heading_label == "uncertain"
    assert reported.speed_mph == 0


def test_resolve_reported_motion_uses_field_on_strong_history_field_disagreement():
    positions = [
        (datetime(2026, 4, 8, 18, 0), 35.0, -97.0),
        (datetime(2026, 4, 8, 18, 5), 35.5, -96.7),
        (datetime(2026, 4, 8, 18, 10), 36.0, -96.4),
    ]
    field = GeographicMotionFieldEstimate(
        delta_lat=0.015,
        delta_lon=0.0,
        quality=0.8,
        source="object_weighted_centroid",
    )
    reported, diagnostic = resolve_reported_motion(
        positions,
        identity_confidence=0.95,
        field_estimate=field,
        field_dt_hours=1.0,
    )
    assert diagnostic.source == "track_history"
    assert diagnostic.speed_kmh > 50
    assert reported.source == "motion_field"


def test_recent_consensus_heading_ignores_wide_heading_spread():
    positions = [
        (datetime(2026, 4, 8, 18, 0), 35.00, -97.00),
        (datetime(2026, 4, 8, 18, 5), 35.03, -96.97),
        (datetime(2026, 4, 8, 18, 10), 35.01, -96.91),
        (datetime(2026, 4, 8, 18, 15), 34.97, -96.95),
    ]
    assert _recent_consensus_heading_deg(positions) is None


def test_resolve_reported_motion_suppresses_field_when_recent_steps_disagree():
    positions = [
        (datetime(2026, 4, 8, 18, 0), 35.00, -97.00),
        (datetime(2026, 4, 8, 18, 5), 34.99, -96.97),
        (datetime(2026, 4, 8, 18, 10), 34.98, -96.94),
        (datetime(2026, 4, 8, 18, 15), 34.95, -96.97),
        (datetime(2026, 4, 8, 18, 20), 34.91, -97.02),
        (datetime(2026, 4, 8, 18, 25), 34.87, -97.08),
    ]
    field = GeographicMotionFieldEstimate(
        delta_lat=0.03,
        delta_lon=0.03,
        quality=0.8,
        source="phase_correlation",
    )
    reported, diagnostic = resolve_reported_motion(
        positions,
        identity_confidence=0.95,
        field_estimate=field,
        field_dt_hours=1.0,
    )
    assert _recent_consensus_heading_deg(positions) is not None
    assert reported.source == "suppressed"
    assert reported.heading_label == "uncertain"
    assert reported.confidence is not None
    assert reported.confidence.reason == "recent track steps disagree with field heading"
