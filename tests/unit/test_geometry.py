import math

import numpy as np
import pyart
import pytest
from scipy.ndimage import label

from src.detection import polar_to_latlon
from src.geometry import (
    align_field_by_azimuth,
    gate_coordinates,
    gate_areas_km2,
    ground_range_m,
    label_periodic_azimuth,
)
from src.parser import extract_sweep_data, extract_velocity

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


def test_gate_areas_use_median_azimuth_spacing_not_first_gap():
    """One anomalous gap must not set the spacing for the whole sweep.

    The legacy implementation used az[1] - az[0]. Here that first gap is ten
    times the true spacing, so first-gap logic would inflate every area 10x.
    """
    azimuths = np.concatenate(([0.0, 5.0], np.arange(5.5, 359.5, 0.5)))
    ranges_m = np.array([100_000.0, 100_250.0])
    areas = gate_areas_km2(azimuths, ranges_m, elevation_deg=0.5)
    ground = ground_range_m(ranges_m, 0.5)
    expected = (ground * np.radians(0.5) * 250.0) / 1e6
    np.testing.assert_allclose(areas, expected, rtol=1e-6)


def test_gate_areas_handle_azimuth_wraparound():
    """Rays straddling 0/360 must not be read as a 359.5 degree gap."""
    azimuths = np.array([359.5, 0.0])
    ranges_m = np.array([100_000.0, 100_250.0])
    areas = gate_areas_km2(azimuths, ranges_m, elevation_deg=0.5)
    ground = ground_range_m(ranges_m, 0.5)
    expected = (ground * np.radians(0.5) * 250.0) / 1e6
    np.testing.assert_allclose(areas, expected, rtol=1e-6)


def test_label_periodic_azimuth_merges_across_seam():
    """Ray 0 and ray N-1 are physically adjacent. A storm occupying the same
    gate index in both must be one component, not two.

    Plain scipy.ndimage.label treats row 0 and row N-1 as unconnected array
    edges and yields count == 2 for this mask; that is the bug this function
    corrects.
    """
    mask = np.zeros((10, 10), dtype=bool)
    mask[0, 5] = True
    mask[-1, 5] = True

    plain_labeled, plain_count = label(mask)
    assert plain_count == 2, "test setup assumption: plain label sees two components here"

    labeled, count = label_periodic_azimuth(mask)
    assert count == 1
    assert labeled[0, 5] == labeled[-1, 5]
    assert labeled[0, 5] != 0


def test_label_periodic_azimuth_leaves_separate_components_alone():
    """Two blobs that do not touch the seam must remain two components with
    contiguous ids -- guards against a fix that merges everything."""
    mask = np.zeros((10, 10), dtype=bool)
    mask[3:5, 3:5] = True
    mask[6:8, 6:8] = True

    labeled, count = label_periodic_azimuth(mask)
    assert count == 2
    ids = sorted(np.unique(labeled[labeled > 0]).tolist())
    assert ids == [1, 2]
    assert labeled[3, 3] != labeled[6, 6]


def test_label_periodic_azimuth_merges_diagonally_across_seam():
    """A component crossing the seam at a shifted gate index must merge under
    8-connectivity, and must NOT merge under 4-connectivity.

    Both production call sites that pass an explicit structure rely on this
    branch: real storm echo rarely lands on exactly the same gate index either
    side of due north.
    """
    mask = np.zeros((36, 40), dtype=bool)
    mask[-1, 10] = True
    mask[0, 11] = True

    _four_conn, four_count = label_periodic_azimuth(mask)
    _eight_conn, eight_count = label_periodic_azimuth(
        mask, structure=np.ones((3, 3), dtype=int)
    )

    assert four_count == 2
    assert eight_count == 1


def test_label_periodic_azimuth_does_not_wrap_the_range_axis():
    """Only azimuth is periodic. Gate 0 and gate N-1 are far apart in space."""
    mask = np.zeros((36, 40), dtype=bool)
    mask[0, 0] = True
    mask[-1, 39] = True
    _labeled, count = label_periodic_azimuth(mask, structure=np.ones((3, 3), dtype=int))
    assert count == 2


def test_align_field_by_azimuth_corrects_split_cut_offset():
    """Split-cut VCPs scan reflectivity and velocity on separate antenna
    revolutions, so the same array index names a different compass bearing
    in each. Pairing by raw index (what src/server.py did before this fix)
    silently misattributes the data. This proves both halves: the band lands
    at the correct rays with alignment, and at the wrong rays without it.
    """
    n = 360
    target_azimuths = np.arange(n, dtype=float)  # e.g. the reflectivity/surveillance cut
    offset_deg = 20.0
    # e.g. the velocity/Doppler cut: same ray count, started 20 degrees later
    source_azimuths = (target_azimuths + offset_deg) % 360.0

    source_field = np.zeros(n)
    source_field[100:105] = 99.0  # a distinct band at source rays 100-104 (azimuth 120-124)

    aligned = align_field_by_azimuth(source_field, source_azimuths, target_azimuths)

    # Correct physical location: azimuth 120-124 is target rays 120-124.
    assert np.array_equal(aligned[120:125], np.full(5, 99.0))
    assert not np.any(aligned[100:105] == 99.0)

    # Prove the pre-fix bug directly: pairing the same two arrays by raw
    # index (no alignment at all) puts the band at the wrong rays -- the
    # source's own indices, 20 degrees off from where it physically is.
    unaligned = source_field
    assert np.array_equal(unaligned[100:105], np.full(5, 99.0))
    assert not np.any(unaligned[120:125] == 99.0)


REFLECTIVITY_VELOCITY_VOLUMES = [
    # (path, minimum same-index offset the defect produces, maximum residual
    # after alignment). Minimums are set safely below the values measured
    # directly against these volumes (10.99 deg and 21.04 deg) so the test
    # keeps demonstrating the defect even if a future re-extraction shifts
    # the azimuth sampling slightly; the residual ceiling is half a ray
    # (0.5 deg for a 720-ray sweep).
    ("cache/KTLX/KTLX20260410_000100_V06", 5.0, 0.5),
    ("cache/KEMX/KEMX20260712_022646_V06", 5.0, 0.5),
]


@pytest.mark.parametrize("path,min_offset_deg,max_residual_deg", REFLECTIVITY_VELOCITY_VOLUMES)
def test_align_field_by_azimuth_fixes_real_split_cut_volumes(path, min_offset_deg, max_residual_deg):
    """Live regression for the classify_gates velocity-misattribution defect.

    Also the "live assertion" that the reflectivity and velocity cuts share
    ranges_m: alignment only remaps the azimuth axis, so if a future VCP ever
    produces cuts with different ranges_m, this must fail loudly rather than
    let alignment silently misattribute range too.
    """
    radar = pyart.io.read_nexrad_archive(path)
    ref = extract_sweep_data(radar)
    vel = extract_velocity(radar, max_sweeps=3)
    velocity_sweep = vel.sweeps[0]

    assert np.array_equal(ref.ranges_m, velocity_sweep.ranges_m), (
        f"{path}: reflectivity and velocity ranges_m differ -- azimuth-only "
        "alignment is unsafe for this volume"
    )

    same_index_offset = (velocity_sweep.azimuths - ref.azimuths + 180.0) % 360.0 - 180.0
    assert np.max(np.abs(same_index_offset)) > min_offset_deg, (
        f"{path}: expected a large same-index azimuth offset demonstrating "
        "the split-cut defect this test guards against"
    )

    aligned = align_field_by_azimuth(velocity_sweep.velocity, velocity_sweep.azimuths, ref.azimuths)
    assert aligned.shape == ref.reflectivity.shape

    source = velocity_sweep.azimuths[None, :]
    target = ref.azimuths[:, None]
    nearest_offsets = (source - target + 180.0) % 360.0 - 180.0
    residual_deg = np.min(np.abs(nearest_offsets), axis=1)
    assert np.max(residual_deg) < max_residual_deg
