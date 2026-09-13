"""Storm motion measured by matching each storm's reflectivity pattern between scans.

Guards and their evidence: docs/test_reports/2026-09-13-pattern-motion-evaluation.md.
"""
import numpy as np
import pytest

from src.geometry import gate_ground_xy_m
from src.parser import SweepData
from src.tracking.pattern_motion import (
    GroundSampler,
    PatternMatch,
    guard_matches,
    match_storm,
    nearby_velocity,
)

RAYS = 720
RANGES_M = np.arange(2125.0, 300000.0, 250.0)
HOURS = 5.0 / 60.0


def _sweep(cells, first_azimuth=0.25):
    """A sweep containing Gaussian cells (east m, north m, peak dBZ, east radius m, north radius m)."""
    azimuths = (np.arange(RAYS) * 0.5 + first_azimuth) % 360.0
    elevations = np.full(RAYS, 0.5)
    rays, gates = np.meshgrid(np.arange(RAYS), np.arange(len(RANGES_M)), indexing="ij")
    x, y = gate_ground_xy_m(rays.ravel(), gates.ravel(), azimuths, RANGES_M, elevations)
    x = x.reshape(rays.shape)
    y = y.reshape(rays.shape)
    field = np.zeros(rays.shape)
    for cx, cy, peak, rx, ry in cells:
        d2 = ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2
        texture = 6.0 * np.sin((x - cx) / 1700.0) * np.cos((y - cy) / 2300.0)
        field = np.maximum(field, (peak + texture) * np.exp(-d2))
    return SweepData(
        reflectivity=np.where(field >= 20.0, field, np.nan),
        azimuths=azimuths,
        ranges_m=RANGES_M,
        elevation_angle=0.5,
        elevations=elevations,
        elevation_angles=[0.5],
        radar_lat=35.0,
        radar_lon=-97.0,
        radar_alt_m=300.0,
        timestamp="2026-09-13T20:00:00Z",
    )


def _mask_near(sweep, cx, cy, radius_m):
    rays, gates = np.meshgrid(np.arange(RAYS), np.arange(len(RANGES_M)), indexing="ij")
    x, y = gate_ground_xy_m(rays.ravel(), gates.ravel(), sweep.azimuths, sweep.ranges_m, sweep.elevations)
    near = np.hypot(x - cx, y - cy).reshape(rays.shape) <= radius_m
    return near & np.isfinite(sweep.reflectivity)


def test_ground_sampler_reads_the_gate_under_a_ground_point_whatever_the_first_azimuth():
    sweep = _sweep([], first_azimuth=137.25)
    sweep.reflectivity[:] = np.nan
    ray = int(np.argmin(np.abs((sweep.azimuths - 45.0 + 180.0) % 360.0 - 180.0)))
    gate = 400
    sweep.reflectivity[ray, gate] = 42.0
    x, y = gate_ground_xy_m(np.array([ray]), np.array([gate]), sweep.azimuths, sweep.ranges_m, sweep.elevations)
    sampler = GroundSampler.from_sweep(sweep)
    assert sampler.sample(sweep.reflectivity, x, y, np.nan)[0] == 42.0


def test_a_storm_moving_east_north_east_is_measured_from_its_pattern():
    # 25 km/h east and 10 km/h north over 5 minutes.
    dx, dy = 25.0 * HOURS * 1000.0, 10.0 * HOURS * 1000.0
    previous = _sweep([(60000.0, 80000.0, 52.0, 5000.0, 7000.0), (90000.0, 70000.0, 45.0, 4000.0, 4000.0)])
    current = _sweep([(60000.0 + dx, 80000.0 + dy, 52.0, 5000.0, 7000.0), (90000.0 + dx, 70000.0 + dy, 45.0, 4000.0, 4000.0)], first_azimuth=88.75)

    found = match_storm(current, _mask_near(current, 60000.0 + dx, 80000.0 + dy, 15000.0), previous, HOURS)

    assert isinstance(found, PatternMatch)
    assert found.east_kmh == pytest.approx(25.0, abs=3.0)
    assert found.north_kmh == pytest.approx(10.0, abs=3.0)
    assert not found.at_edge


def test_a_storm_straddling_due_north_is_measured():
    dx, dy = -20.0 * HOURS * 1000.0, 30.0 * HOURS * 1000.0
    previous = _sweep([(-1500.0, 95000.0, 50.0, 5000.0, 6000.0)], first_azimuth=200.25)
    current = _sweep([(-1500.0 + dx, 95000.0 + dy, 50.0, 5000.0, 6000.0)], first_azimuth=10.25)

    found = match_storm(current, _mask_near(current, -1500.0 + dx, 95000.0 + dy, 15000.0), previous, HOURS)

    assert found.east_kmh == pytest.approx(-20.0, abs=3.0)
    assert found.north_kmh == pytest.approx(30.0, abs=3.0)


def test_a_storm_absent_from_the_previous_scan_gives_no_match():
    previous = _sweep([])
    current = _sweep([(60000.0, 80000.0, 52.0, 5000.0, 7000.0)])
    assert match_storm(current, _mask_near(current, 60000.0, 80000.0, 15000.0), previous, HOURS) is None


def _match(east, north, at_edge=False):
    return PatternMatch(east_kmh=east, north_kmh=north, correlation=0.9, at_edge=at_edge)


def test_guard_drops_a_match_on_the_edge_of_the_search_window():
    accepted = guard_matches({1: _match(20.0, 5.0, at_edge=True), 2: _match(20.0, 5.0)}, {1: (35.5, -97.0), 2: (35.5, -96.9)})
    assert set(accepted) == {2}


def test_guard_drops_a_velocity_far_from_its_neighbours():
    centres = {track: (35.5 + 0.05 * track, -97.0) for track in range(1, 6)}
    matches = {1: _match(20.0, 5.0), 2: _match(22.0, 4.0), 3: _match(19.0, 6.0), 4: _match(21.0, 5.0), 5: _match(-110.0, 60.0)}
    accepted = guard_matches(matches, centres)
    assert set(accepted) == {1, 2, 3, 4}


def test_guard_keeps_a_storm_without_enough_neighbours_to_judge():
    centres = {1: (35.5, -97.0), 2: (35.55, -97.0), 3: (37.5, -97.0)}
    matches = {1: _match(20.0, 5.0), 2: _match(-60.0, 40.0), 3: _match(10.0, 10.0)}
    assert set(guard_matches(matches, centres)) == {1, 2, 3}


def test_nearby_velocity_is_the_vector_median_of_other_storms_within_60_km():
    centres = {1: (35.5, -97.0), 2: (35.6, -97.0), 3: (35.7, -97.0), 4: (35.55, -96.9), 5: (38.0, -97.0)}
    accepted = {2: (20.0, 4.0), 3: (24.0, 8.0), 4: (22.0, 6.0), 5: (-50.0, -50.0)}
    assert nearby_velocity(1, accepted, centres) == (22.0, 6.0)
    assert nearby_velocity(5, {1: (1.0, 1.0)}, {1: (35.5, -97.0), 5: (38.0, -97.0)}) is None


def test_guard_drops_a_match_faster_than_the_search_limit_in_any_direction():
    # 144 km/h east and 123 km/h north each fit a square search window; together 189 km/h does not.
    accepted = guard_matches({1: _match(144.0, 123.0), 2: _match(100.0, 0.0)}, {1: (35.5, -97.0), 2: (38.0, -97.0)})
    assert set(accepted) == {2}
