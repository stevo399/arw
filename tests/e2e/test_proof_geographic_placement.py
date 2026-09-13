"""Proof (2026-09-13): every detected storm is placed at its true geographic centre.

Storm centres were computed by averaging ray and gate indices.  Ray 0 and the
last ray are adjacent, so a storm straddling them averaged to the far side of
the radar: a survey of 49 cached volumes found such storms in 10, misplaced by
12 to 451 km.  These volumes each contain at least one such storm.

The reference centre here is computed a different way from the pipeline's:
reflectivity times ground area per gate, averaged as unit vectors on the
sphere from Py-ART-derived gate latitudes and longitudes.
"""

import math

import numpy as np
import pytest

import src.server as server
from src.geometry import gate_coordinates
from src.sites import haversine_distance_km

VOLUMES = [
    ("KEYX", "cache/KEYX/KEYX20260410_182921_V06"),
    ("KTLX", "cache/KTLX/KTLX20260410_224737_V06"),
    ("KJAX", "cache/KJAX/KJAX20260912_175903_V06"),
]


def _spherical_centre(lats, lons, weights):
    lat_r, lon_r = np.radians(lats), np.radians(lons)
    x = np.dot(weights, np.cos(lat_r) * np.cos(lon_r))
    y = np.dot(weights, np.cos(lat_r) * np.sin(lon_r))
    z = np.dot(weights, np.sin(lat_r))
    return math.degrees(math.atan2(z, math.hypot(x, y))), math.degrees(math.atan2(y, x))


def _initial_bearing(lat1, lon1, lat2, lon2):
    p1, p2, dl = math.radians(lat1), math.radians(lat2), math.radians(lon2 - lon1)
    return math.degrees(math.atan2(math.sin(dl) * math.cos(p2), math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl))) % 360.0


@pytest.mark.parametrize("site_id,path", VOLUMES)
def test_every_storm_is_placed_at_its_geographic_centre(site_id, path):
    scan = server._process_scan_file(site_id, path)
    sweep = scan.reflectivity_data
    lat_grid, lon_grid = gate_coordinates(sweep.azimuths, sweep.ranges_m, sweep.elevations, sweep.radar_lat, sweep.radar_lon)
    az_spacing = math.radians(float(np.median(np.abs(np.diff(np.unwrap(sweep.azimuths, period=360.0))))))
    gate_spacing_km = float(np.median(np.diff(sweep.ranges_m))) / 1000.0

    straddling = 0
    worst = (0.0, None)
    for obj in scan.detected_objects:
        mask = np.asarray(scan.object_masks[obj.object_id])
        straddling += bool(mask[0].any() and mask[-1].any())
        rays, gates = np.nonzero(mask)
        lats, lons = lat_grid[rays, gates], lon_grid[rays, gates]
        ground_km = np.array([haversine_distance_km(sweep.radar_lat, sweep.radar_lon, a, b) for a, b in zip(lats, lons)])
        dbz = np.nan_to_num(sweep.reflectivity[rays, gates], nan=0.0)
        expected = _spherical_centre(lats, lons, dbz * ground_km * az_spacing * gate_spacing_km)

        error_km = haversine_distance_km(obj.centroid_lat, obj.centroid_lon, *expected)
        worst = max(worst, (error_km, obj.object_id))
        distance = haversine_distance_km(sweep.radar_lat, sweep.radar_lon, obj.centroid_lat, obj.centroid_lon)
        bearing = _initial_bearing(sweep.radar_lat, sweep.radar_lon, obj.centroid_lat, obj.centroid_lon)
        assert obj.distance_km == pytest.approx(distance, abs=0.15), obj.object_id
        assert min(abs(obj.bearing_deg - bearing), 360.0 - abs(obj.bearing_deg - bearing)) < 0.2 or distance < 1.0, obj.object_id

    assert straddling >= 1, "volume no longer exercises a storm straddling the first and last ray"
    assert worst[0] < 0.25, f"object {worst[1]} placed {worst[0]:.2f} km from its geographic centre"


VELOCITY_VOLUMES = [
    "cache/KTLX/KTLX20260410_224737_V06",
    "cache/KMLB/KMLB20260908_011857_V06",
]


@pytest.mark.parametrize("path", VELOCITY_VOLUMES)
def test_every_velocity_region_is_placed_at_its_geographic_centre(path):
    from src.parser import extract_velocity, parse_radar_file
    from src.velocity import _detect_regions_single_sweep

    vel = extract_velocity(parse_radar_file(path))
    straddling = 0
    worst = (0.0, None)
    for sweep in vel.sweeps:
        lat_grid, lon_grid = gate_coordinates(sweep.azimuths, sweep.ranges_m, sweep.elevations, vel.radar_lat, vel.radar_lon)
        az_spacing = math.radians(float(np.median(np.abs(np.diff(np.unwrap(sweep.azimuths, period=360.0))))))
        gate_spacing_km = float(np.median(np.diff(sweep.ranges_m))) / 1000.0
        results = _detect_regions_single_sweep(
            sweep.velocity, sweep.azimuths, sweep.ranges_m, vel.radar_lat, vel.radar_lon, sweep.elevation_angle, sweep.elevations,
        )
        for index, (region, mask) in enumerate(results):
            straddling += bool(mask[0].any() and mask[-1].any())
            rays, gates = np.nonzero(mask)
            lats, lons = lat_grid[rays, gates], lon_grid[rays, gates]
            ground_km = np.array([haversine_distance_km(vel.radar_lat, vel.radar_lon, a, b) for a, b in zip(lats, lons)])
            speed = np.nan_to_num(np.abs(sweep.velocity[rays, gates]), nan=0.0)
            expected = _spherical_centre(lats, lons, speed * ground_km * az_spacing * gate_spacing_km)
            worst = max(worst, (haversine_distance_km(region.centroid_lat, region.centroid_lon, *expected), (sweep.elevation_angle, index)))

    assert straddling >= 1, "volume no longer exercises a region straddling the first and last ray"
    assert worst[0] < 0.25, f"region {worst[1]} placed {worst[0]:.2f} km from its geographic centre"


def test_velocity_regions_merge_across_tilts_that_start_at_different_azimuths():
    """KTLX 22:47:37Z's three Doppler cuts start at 84, 128 and 176 degrees.
    Compared by ray index, no region ever matched across tilts (every region
    had sweep_count 1); compared by azimuth, 12 of 122 do."""
    from src.parser import extract_velocity, parse_radar_file
    from src.velocity import detect_velocity_regions

    vel = extract_velocity(parse_radar_file("cache/KTLX/KTLX20260410_224737_V06"))
    starts = {round(float(sweep.azimuths[0])) for sweep in vel.sweeps}
    assert len(starts) == len(vel.sweeps), starts

    regions = detect_velocity_regions(vel)
    assert sum(region.sweep_count >= 2 for region in regions) >= 1
