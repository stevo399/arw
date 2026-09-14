"""The tracker reports motion measured from the storms, never inferred.

Synthetic volumes are run through the real detector, then the tracker.
"""
from datetime import datetime, timedelta
import math

import pytest

from src.buffer import BufferedScan
from src.detection import detect_objects_with_grid
from src.tracker import StormTracker
from src.summary import generate_summary
from tests.unit.test_tracking_pattern_motion import _sweep

T0 = datetime(2026, 9, 13, 20, 0)
STEP = timedelta(minutes=5)
EAST_KMH, NORTH_KMH = 25.0, 10.0
STORMS = [(60000.0, 80000.0), (80000.0, 95000.0), (95000.0, 70000.0), (70000.0, 60000.0)]


def _scan(minutes: int, storms, first_azimuth: float = 0.25) -> BufferedScan:
    hours = minutes / 60.0
    cells = [(x + EAST_KMH * hours * 1000.0, y + NORTH_KMH * hours * 1000.0, 52.0, 4000.0, 5000.0) for x, y in storms]
    sweep = _sweep(cells, first_azimuth=first_azimuth)
    detection = detect_objects_with_grid(
        reflectivity=sweep.reflectivity,
        azimuths=sweep.azimuths,
        ranges_m=sweep.ranges_m,
        radar_lat=sweep.radar_lat,
        radar_lon=sweep.radar_lon,
        elevation_deg=sweep.elevation_angle,
        elevations=sweep.elevations,
    )
    return BufferedScan(
        timestamp=T0 + timedelta(minutes=minutes),
        site_id="KTLX",
        reflectivity_data=sweep,
        detected_objects=detection.objects,
        labeled_grid=detection.labeled_grid,
        object_masks=detection.object_masks,
    )


def _heading(motion) -> float:
    return motion.heading_deg


def test_first_scan_storms_have_unknown_motion_and_are_never_called_stationary():
    tracker = StormTracker()
    scan = _scan(0, STORMS)
    tracker.update(scan)

    assert len(tracker.active_tracks) == len(STORMS)
    for track in tracker.active_tracks:
        motion = track.get_motion()
        assert motion.heading_label == "unknown"
        assert motion.speed_kmh is None
    text = generate_summary("KTLX", "Oklahoma City", scan.timestamp.isoformat(), scan.detected_objects, tracker.active_tracks)
    assert "motion not yet known" in text
    assert "stationary" not in text


def test_established_storms_report_their_measured_motion():
    tracker = StormTracker()
    for index in range(4):
        tracker.update(_scan(5 * index, STORMS, first_azimuth=(index * 67.5) % 360.0 + 0.25))

    expected_speed = math.hypot(EAST_KMH, NORTH_KMH)
    expected_heading = math.degrees(math.atan2(EAST_KMH, NORTH_KMH))
    for track in tracker.active_tracks:
        motion = track.get_motion()
        assert motion.source == "pattern_match"
        assert motion.confidence.label == "high"
        assert motion.speed_kmh == pytest.approx(expected_speed, abs=3.0)
        assert _heading(motion) == pytest.approx(expected_heading, abs=8.0)
        assert len(track.measured_velocities) == 3


def test_second_scan_storms_report_measured_motion_with_medium_confidence():
    tracker = StormTracker()
    tracker.update(_scan(0, STORMS))
    tracker.update(_scan(5, STORMS))

    for track in tracker.active_tracks:
        motion = track.get_motion()
        assert motion.source == "pattern_match"
        assert motion.confidence.label == "medium"


def test_a_new_storm_among_moving_storms_reports_their_motion_as_nearby_storms():
    tracker = StormTracker()
    tracker.update(_scan(0, STORMS))
    tracker.update(_scan(5, STORMS))
    newcomer = (75000.0, 75000.0)
    scan = _scan(10, STORMS + [newcomer])
    tracker.update(scan)

    new_tracks = [track for track in tracker.active_tracks if len(track.positions) == 1]
    assert len(new_tracks) == 1
    motion = new_tracks[0].get_motion()
    assert motion.source == "nearby_storms"
    assert motion.speed_kmh == pytest.approx(math.hypot(EAST_KMH, NORTH_KMH), abs=3.0)

    track = new_tracks[0]
    track.is_primary_focus = True
    text = generate_summary(
        "KTLX", "Oklahoma City", scan.timestamp.isoformat(),
        [track.current_object], [track],
    )
    assert "with nearby storms" in text
    assert "likely moving" in text


def test_a_lone_new_storm_has_unknown_motion():
    tracker = StormTracker()
    tracker.update(_scan(0, []))
    tracker.update(_scan(5, [STORMS[0]]))

    (track,) = tracker.active_tracks
    assert track.get_motion().heading_label == "unknown"


def test_a_stationary_storm_among_moving_storms_reports_its_own_motion_once_corroborated():
    """Neighbours' vector median rejects its first measurement; the second agrees with the first.

    The stationary storm is 35 km or more from the moving ones, so the echo
    matched around it is its own, and within 60 km of three of them, so they
    judge it.
    """
    still = (115000.0, 100000.0)

    def scan(minutes):
        hours = minutes / 60.0
        cells = [(x + EAST_KMH * hours * 1000.0, y + NORTH_KMH * hours * 1000.0, 52.0, 4000.0, 5000.0) for x, y in STORMS]
        cells.append((still[0], still[1], 50.0, 3000.0, 3000.0))
        sweep = _sweep(cells)
        detection = detect_objects_with_grid(
            reflectivity=sweep.reflectivity, azimuths=sweep.azimuths, ranges_m=sweep.ranges_m,
            radar_lat=sweep.radar_lat, radar_lon=sweep.radar_lon, elevation_deg=sweep.elevation_angle,
            elevations=sweep.elevations,
        )
        return BufferedScan(
            timestamp=T0 + timedelta(minutes=minutes), site_id="KTLX", reflectivity_data=sweep,
            detected_objects=detection.objects, labeled_grid=detection.labeled_grid, object_masks=detection.object_masks,
        )

    def still_track(tracker):
        # The stationary storm is the smallest.
        return min(tracker.active_tracks, key=lambda track: track.current_object.area_km2)

    tracker = StormTracker()
    tracker.update(scan(0))
    tracker.update(scan(5))
    assert still_track(tracker).get_motion().source == "nearby_storms"
    tracker.update(scan(10))
    motion = still_track(tracker).get_motion()
    assert motion.source == "pattern_match"
    assert motion.heading_label == "nearly stationary"
