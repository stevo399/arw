# tests/unit/test_motion.py
import math
from datetime import datetime, timedelta
from src.motion import compute_motion, MotionVector

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
