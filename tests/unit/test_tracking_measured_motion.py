"""Which motion a storm reports, from measured pattern velocities.

Rules and evidence: docs/superpowers/specs/2026-09-13-measured-motion-design.md.
"""
from datetime import datetime, timedelta

import pytest

from src.history.records import dumps, loads
from src.tracking.motion import NEARLY_STATIONARY_KMH, report_motion
from src.tracking.types import Track, VelocitySample

T0 = datetime(2026, 9, 13, 20, 0)
NOW = T0 + timedelta(minutes=15)


def _samples(*velocities):
    return [VelocitySample(timestamp=T0 + timedelta(minutes=5 * i), east_kmh=e, north_kmh=n) for i, (e, n) in enumerate(velocities)]


def test_an_established_storm_reports_the_median_of_its_last_three_measured_velocities():
    motion = report_motion(position_count=4, measured=_samples((10.0, 0.0), (30.0, 30.0), (20.0, 20.0), (22.0, 18.0)), nearby=None, now=NOW)
    # Last three: (30, 30), (20, 20), (22, 18) -> median (22, 20).
    assert motion.source == "pattern_match"
    assert motion.speed_kmh == pytest.approx(29.7, abs=0.1)
    assert motion.heading_deg == pytest.approx(47.7, abs=0.1)
    assert motion.heading_label == "NE"
    assert motion.speed_mph == 18
    assert motion.confidence.label == "high"
    assert motion.confidence.score == 0.9


def test_fewer_than_three_measured_velocities_are_reported_with_medium_confidence():
    motion = report_motion(position_count=2, measured=_samples((0.0, -40.0)), nearby=(50.0, 0.0), now=NOW)
    assert motion.source == "pattern_match"
    assert motion.heading_label == "S"
    assert motion.confidence.label == "medium"
    assert motion.confidence.score < 0.9


def test_a_storm_with_one_position_reports_nearby_storms_motion_even_with_its_own_measurement():
    motion = report_motion(position_count=1, measured=_samples((0.0, -40.0)), nearby=(30.0, 0.0), now=NOW)
    assert motion.source == "nearby_storms"
    assert motion.heading_label == "E"
    assert motion.speed_kmh == pytest.approx(30.0)
    # Never trusted enough for storm-relative velocity (requires 0.9).
    assert motion.confidence.score < 0.9


def test_a_storm_without_its_own_measurement_reports_nearby_storms_motion():
    motion = report_motion(position_count=5, measured=[], nearby=(-20.0, 20.0), now=NOW)
    assert motion.source == "nearby_storms"
    assert motion.heading_label == "NW"


def test_a_storm_with_nothing_measured_reports_motion_unknown():
    motion = report_motion(position_count=1, measured=[], nearby=None, now=NOW)
    assert motion.source == "not_measured"
    assert motion.heading_label == "unknown"
    assert motion.heading_deg is None
    assert motion.speed_kmh is None
    assert motion.speed_mph is None
    assert motion.confidence.score == 0.0


def test_speeds_indistinguishable_from_no_motion_are_nearly_stationary_never_stationary():
    below = report_motion(position_count=4, measured=_samples((3.0, 4.0), (3.0, 4.0), (3.0, 4.0)), nearby=None, now=NOW)
    assert below.speed_kmh == pytest.approx(5.0)
    assert below.heading_label == "nearly stationary"
    assert below.heading_deg is None
    at = report_motion(position_count=4, measured=_samples((NEARLY_STATIONARY_KMH, 0.0)), nearby=None, now=NOW)
    assert at.heading_label == "E"
    still = report_motion(position_count=4, measured=_samples((0.0, 0.0)), nearby=None, now=NOW)
    assert still.heading_label == "nearly stationary"


def test_measured_velocities_survive_the_tracker_state_codec():
    track = Track(track_id=7, status="active", measured_velocities=_samples((12.5, -3.25)))
    restored = loads(dumps(track))
    assert restored.measured_velocities == track.measured_velocities


def test_velocities_older_than_the_continuity_limit_are_not_reported():
    old = [VelocitySample(timestamp=T0, east_kmh=40.0, north_kmh=0.0)]
    stale = report_motion(position_count=3, measured=old, nearby=None, now=T0 + timedelta(minutes=21))
    assert stale.heading_label == "unknown"
    recent = report_motion(position_count=3, measured=old, nearby=None, now=T0 + timedelta(minutes=19))
    assert recent.source == "pattern_match"
