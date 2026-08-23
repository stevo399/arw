import math

import numpy as np
import pyart
import pytest

from src.detection import polar_to_latlon
from src.geometry import gate_coordinates, gate_areas_km2, ground_range_m

REFERENCE_VOLUME = "cache/KEMX/KEMX20260712_022646_V06"


@pytest.fixture(scope="module")
def radar():
    return pyart.io.read_nexrad_archive(REFERENCE_VOLUME)


def test_gate_coordinates_match_pyart_reference(radar):
    """Our georeferencing must agree with Py-ART's Doviak & Zrnic implementation."""
    sweep_start, sweep_end = radar.get_start_end(0)
    azimuths = radar.azimuth["data"][sweep_start:sweep_end + 1]
    ranges_m = radar.range["data"]
    # Per-ray elevation, not the nominal fixed_angle. Py-ART georeferences with
    # the real pointing angle of each ray, and they differ enough within one
    # sweep to move far gates by tens of metres.
    elevations = radar.elevation["data"][sweep_start:sweep_end + 1]

    lat, lon = gate_coordinates(
        azimuths=azimuths,
        ranges_m=ranges_m,
        elevation_deg=elevations,
        radar_lat=float(radar.latitude["data"][0]),
        radar_lon=float(radar.longitude["data"][0]),
    )

    expected_lat = radar.gate_latitude["data"][sweep_start:sweep_end + 1]
    expected_lon = radar.gate_longitude["data"][sweep_start:sweep_end + 1]

    assert lat.shape == expected_lat.shape
    np.testing.assert_allclose(lat, expected_lat, atol=1e-6)
    np.testing.assert_allclose(lon, expected_lon, atol=1e-6)


def test_legacy_polar_to_latlon_displacement_is_documented(radar):
    """Records how far the pre-correction implementation was wrong.

    This test exists to document the magnitude of the error that motivated
    the correction. Delete it only when polar_to_latlon is removed.
    """
    sweep_start, sweep_end = radar.get_start_end(0)
    azimuths = radar.azimuth["data"][sweep_start:sweep_end + 1]
    ranges_m = radar.range["data"]
    gate_lat = radar.gate_latitude["data"][sweep_start:sweep_end + 1]
    gate_lon = radar.gate_longitude["data"][sweep_start:sweep_end + 1]

    expected_displacement_m = {200: 8, 400: 26, 800: 107, 1600: 546}

    for range_index, expected_m in expected_displacement_m.items():
        legacy_lat, legacy_lon = polar_to_latlon(
            float(radar.latitude["data"][0]),
            float(radar.longitude["data"][0]),
            float(azimuths[0]),
            float(ranges_m[range_index]),
        )
        reference_lat = float(gate_lat[0, range_index])
        reference_lon = float(gate_lon[0, range_index])
        delta_north_m = (legacy_lat - reference_lat) * 111195.0
        delta_east_m = (
            (legacy_lon - reference_lon) * 111195.0 * math.cos(math.radians(reference_lat))
        )
        displacement_m = math.hypot(delta_north_m, delta_east_m)
        assert displacement_m == pytest.approx(expected_m, abs=5)


def test_beam_height_uses_effective_earth_radius():
    """Beam height must use the 4/3 effective radius, not the true radius.

    Expected values computed from the Doviak and Zrnic exact form at 0.5
    degrees elevation with the antenna at sea level.
    """
    from src.geometry import beam_height_m

    expected_m = {
        50_000: 583.5,
        100_000: 1461.1,
        200_000: 4098.7,
        300_000: 7911.7,
    }
    for range_m, expected in expected_m.items():
        assert beam_height_m(range_m, elevation_deg=0.5) == pytest.approx(expected, abs=1.0)


def test_beam_height_ten_km_cutoff_reaches_345km():
    """The 10 km rejection threshold must be crossed near 345 km, not 306 km."""
    from src.geometry import beam_height_m

    assert beam_height_m(340_000, elevation_deg=0.5) < 10_000.0
    assert beam_height_m(350_000, elevation_deg=0.5) > 10_000.0


def test_beam_height_adds_site_altitude():
    """Site altitude must be added directly to beam height."""
    from src.geometry import beam_height_m

    at_sea_level = beam_height_m(100_000, elevation_deg=0.5)
    at_altitude = beam_height_m(100_000, elevation_deg=0.5, site_alt_m=400.0)
    assert at_altitude == pytest.approx(at_sea_level + 400.0, abs=0.1)


def test_ground_range_is_shorter_than_slant_range():
    slant = np.array([100_000.0, 300_000.0])
    ground = ground_range_m(slant, elevation_deg=0.5)
    assert np.all(ground < slant)
    # At low elevation the difference is small but non-zero.
    assert ground[0] == pytest.approx(100_000.0, rel=1e-3)


def test_gate_areas_grow_with_range():
    azimuths = np.linspace(0.0, 359.5, 720)
    ranges_m = np.arange(2125.0, 460_000.0, 250.0)
    areas = gate_areas_km2(azimuths, ranges_m, elevation_deg=0.5)
    assert areas.shape == ranges_m.shape
    assert np.all(np.diff(areas) > 0)


def test_gate_areas_use_ground_range_not_slant():
    """Areas must be computed from ground range, so they are smaller than the
    slant-range calculation the legacy code used."""
    azimuths = np.linspace(0.0, 359.5, 720)
    ranges_m = np.array([300_000.0, 300_250.0])
    areas = gate_areas_km2(azimuths, ranges_m, elevation_deg=0.5)
    az_spacing_rad = np.radians(0.5)
    slant_area_km2 = (300_000.0 * az_spacing_rad * 250.0) / 1e6
    assert areas[0] < slant_area_km2
