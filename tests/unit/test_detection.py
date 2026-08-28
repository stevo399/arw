import math
import numpy as np
import pyart
import pytest
from src.detection import (
    detect_objects,
    detect_objects_with_grid,
    classify_intensity,
    compute_object_properties,
    degrees_to_bearing,
    polar_to_latlon,
    MIN_OBJECT_AREA_KM2,
    INTENSITY_THRESHOLDS,
    DetectedObject,
    DetectionResult,
    ThresholdHierarchyNode,
)
from src.sweeps import select_reflectivity_sweep

REFERENCE_VOLUME = "cache/KEMX/KEMX20260712_022646_V06"


@pytest.fixture(scope="module")
def radar():
    return pyart.io.read_nexrad_archive(REFERENCE_VOLUME)


def test_intensity_thresholds_defined():
    assert len(INTENSITY_THRESHOLDS) == 5
    assert INTENSITY_THRESHOLDS[0] == (20, 30, "light precipitation")
    assert INTENSITY_THRESHOLDS[-1] == (60, float("inf"), "severe core")


def test_classify_intensity():
    assert classify_intensity(15.0) == "drizzle"
    assert classify_intensity(25.0) == "light precipitation"
    assert classify_intensity(35.0) == "moderate precipitation"
    assert classify_intensity(45.0) == "heavy precipitation"
    assert classify_intensity(55.0) == "intense precipitation"
    assert classify_intensity(65.0) == "severe core"


def test_degrees_to_bearing():
    assert degrees_to_bearing(0) == "N"
    assert degrees_to_bearing(90) == "E"
    assert degrees_to_bearing(180) == "S"
    assert degrees_to_bearing(270) == "W"
    assert degrees_to_bearing(45) == "NE"
    assert degrees_to_bearing(22.5) == "NNE"


def test_polar_to_latlon():
    # From a radar at 35.0, -97.0, a point 100km due north should be ~35.9, -97.0
    lat, lon = polar_to_latlon(
        radar_lat=35.0, radar_lon=-97.0,
        azimuth_deg=0.0, range_m=100000.0,
    )
    assert abs(lat - 35.9) < 0.1
    assert abs(lon - (-97.0)) < 0.1


def test_detect_objects_empty_grid():
    """No reflectivity above threshold should return no objects."""
    reflectivity = np.full((360, 500), 10.0)  # All below 20 dBZ
    azimuths = np.linspace(0, 359, 360)
    ranges_m = np.linspace(0, 250000, 500)
    objects = detect_objects(
        reflectivity=reflectivity,
        azimuths=azimuths,
        ranges_m=ranges_m,
        radar_lat=35.0,
        radar_lon=-97.0,
    )
    assert len(objects) == 0


def test_detect_objects_single_blob():
    """A single blob of high reflectivity should produce one object."""
    reflectivity = np.full((360, 500), np.nan)
    # Place a 10x10 blob of 45 dBZ at azimuth 90, range bin 200
    reflectivity[85:95, 195:205] = 45.0
    azimuths = np.linspace(0, 359, 360)
    ranges_m = np.linspace(2000, 250000, 500)  # Start at 2km to avoid zero range
    objects = detect_objects(
        reflectivity=reflectivity,
        azimuths=azimuths,
        ranges_m=ranges_m,
        radar_lat=35.0,
        radar_lon=-97.0,
    )
    assert len(objects) == 1
    assert objects[0].peak_dbz == 45.0
    assert objects[0].peak_label == "heavy precipitation"


def test_detect_objects_two_separate_blobs():
    """Two separated blobs should produce two objects."""
    reflectivity = np.full((360, 500), np.nan)
    reflectivity[10:20, 50:60] = 35.0  # Blob 1
    reflectivity[200:210, 300:310] = 55.0  # Blob 2
    azimuths = np.linspace(0, 359, 360)
    ranges_m = np.linspace(2000, 250000, 500)
    objects = detect_objects(
        reflectivity=reflectivity,
        azimuths=azimuths,
        ranges_m=ranges_m,
        radar_lat=35.0,
        radar_lon=-97.0,
    )
    assert len(objects) == 2


def test_detect_objects_nested_layers():
    """An object with varying intensities should have nested layers."""
    reflectivity = np.full((360, 500), np.nan)
    # Outer ring: light precipitation (25 dBZ)
    reflectivity[80:100, 190:210] = 25.0
    # Inner ring: moderate precipitation (35 dBZ)
    reflectivity[85:95, 195:205] = 35.0
    # Core: heavy precipitation (48 dBZ)
    reflectivity[88:92, 198:202] = 48.0
    azimuths = np.linspace(0, 359, 360)
    ranges_m = np.linspace(2000, 250000, 500)
    objects = detect_objects(
        reflectivity=reflectivity,
        azimuths=azimuths,
        ranges_m=ranges_m,
        radar_lat=35.0,
        radar_lon=-97.0,
    )
    assert len(objects) == 1
    obj = objects[0]
    assert obj.peak_dbz == 48.0
    layer_labels = [layer.label for layer in obj.layers]
    assert "light precipitation" in layer_labels
    assert "moderate precipitation" in layer_labels
    assert "heavy precipitation" in layer_labels


def test_detect_objects_filters_small_objects():
    """Very small objects below MIN_OBJECT_AREA_KM2 should be filtered out."""
    reflectivity = np.full((360, 500), np.nan)
    # Place a tiny 2x2 blob — should be smaller than 4 km²
    reflectivity[90:92, 100:102] = 40.0
    azimuths = np.linspace(0, 359, 360)
    # Use tight range spacing so 2x2 pixels are < 4 km²
    ranges_m = np.linspace(2000, 50000, 500)  # ~96m per bin
    objects = detect_objects(
        reflectivity=reflectivity,
        azimuths=azimuths,
        ranges_m=ranges_m,
        radar_lat=35.0,
        radar_lon=-97.0,
    )
    assert len(objects) == 0


def test_detect_objects_filters_small_weak_objects():
    reflectivity = np.full((360, 500), np.nan)
    reflectivity[90:95, 100:105] = 35.0
    azimuths = np.linspace(0, 359, 360)
    ranges_m = np.linspace(2000, 120000, 500)
    objects = detect_objects(
        reflectivity=reflectivity,
        azimuths=azimuths,
        ranges_m=ranges_m,
        radar_lat=35.0,
        radar_lon=-97.0,
    )
    assert len(objects) == 0


def test_detect_objects_keeps_small_intense_objects():
    reflectivity = np.full((360, 500), np.nan)
    reflectivity[90:97, 100:107] = 50.0
    azimuths = np.linspace(0, 359, 360)
    ranges_m = np.linspace(2000, 120000, 500)
    objects = detect_objects(
        reflectivity=reflectivity,
        azimuths=azimuths,
        ranges_m=ranges_m,
        radar_lat=35.0,
        radar_lon=-97.0,
    )
    assert len(objects) == 1
    assert objects[0].peak_label == "intense precipitation"


def test_detect_objects_with_grid_returns_result():
    reflectivity = np.full((360, 500), np.nan)
    reflectivity[85:95, 195:205] = 45.0
    azimuths = np.linspace(0, 359, 360)
    ranges_m = np.linspace(2000, 250000, 500)
    result = detect_objects_with_grid(
        reflectivity=reflectivity,
        azimuths=azimuths,
        ranges_m=ranges_m,
        radar_lat=35.0,
        radar_lon=-97.0,
    )
    assert isinstance(result, DetectionResult)
    assert len(result.objects) == 1
    assert result.labeled_grid.shape == reflectivity.shape
    assert len(result.object_masks) == 1
    assert result.object_masks[1].shape == reflectivity.shape
    assert result.object_masks[1].dtype == bool
    assert 1 in result.object_hierarchy
    assert all(isinstance(node, ThresholdHierarchyNode) for node in result.object_hierarchy[1])


def test_detected_object_class_fractions_absent_without_classification():
    """No gate_classification supplied -> class_fractions is the empty dict,
    not a misleading all-zero map. Guards legacy/synthetic callers (most of
    this test file, and every caller predating the 2026-08-26 amendment)
    that never ran quality control."""
    reflectivity = np.full((360, 500), np.nan)
    reflectivity[85:95, 195:205] = 45.0
    azimuths = np.linspace(0, 359, 360)
    ranges_m = np.linspace(2000, 250000, 500)
    result = detect_objects_with_grid(
        reflectivity=reflectivity,
        azimuths=azimuths,
        ranges_m=ranges_m,
        radar_lat=35.0,
        radar_lon=-97.0,
    )
    assert len(result.objects) == 1
    assert result.objects[0].class_fractions == {}


def test_detected_object_carries_class_composition():
    """THE POINT OF THE 2026-08-26 amendment: an object made mostly of
    ground-clutter-classified gates must be distinguishable from one made
    mostly of precipitation-classified gates, by reading the object alone.

    A 10x10 block of reflectivity forms one connected object (all gates
    >= MIN_DBZ_THRESHOLD). 70 of its 100 gates are classified ground_clutter,
    30 precipitation -- class_fractions must reflect that exact composition,
    not just note that the object is non-empty.
    """
    from src.qc.classifier import CLASS_CODES
    from src.qc.parameters import GateClass

    reflectivity = np.full((360, 500), np.nan)
    reflectivity[100:110, 200:210] = 45.0
    gate_classification = np.zeros((360, 500), dtype=np.int8)
    gate_classification[100:107, 200:210] = CLASS_CODES[GateClass.GROUND_CLUTTER]  # 70 gates
    gate_classification[107:110, 200:210] = CLASS_CODES[GateClass.PRECIPITATION]  # 30 gates
    azimuths = np.linspace(0, 359, 360)
    ranges_m = np.linspace(2000, 250000, 500)
    result = detect_objects_with_grid(
        reflectivity=reflectivity,
        azimuths=azimuths,
        ranges_m=ranges_m,
        radar_lat=35.0,
        radar_lon=-97.0,
        gate_classification=gate_classification,
    )
    assert len(result.objects) == 1
    fractions = result.objects[0].class_fractions
    assert fractions[GateClass.GROUND_CLUTTER] == pytest.approx(0.7)
    assert fractions[GateClass.PRECIPITATION] == pytest.approx(0.3)
    assert fractions[GateClass.BIOLOGICAL] == 0.0
    assert sum(fractions.values()) == pytest.approx(1.0)


def test_detect_objects_with_grid_masks_match_objects():
    reflectivity = np.full((360, 500), np.nan)
    reflectivity[10:20, 50:60] = 35.0
    reflectivity[200:210, 300:310] = 55.0
    azimuths = np.linspace(0, 359, 360)
    ranges_m = np.linspace(2000, 250000, 500)
    result = detect_objects_with_grid(
        reflectivity=reflectivity,
        azimuths=azimuths,
        ranges_m=ranges_m,
        radar_lat=35.0,
        radar_lon=-97.0,
    )
    assert len(result.objects) == 2
    assert len(result.object_masks) == 2
    for obj in result.objects:
        assert obj.object_id in result.object_masks


def test_detect_objects_splits_broad_blob_with_multiple_strong_cores():
    reflectivity = np.full((360, 500), np.nan)
    reflectivity[80:110, 190:230] = 25.0
    reflectivity[85:100, 195:208] = 40.0
    reflectivity[85:100, 212:225] = 40.0
    reflectivity[88:96, 198:204] = 55.0
    reflectivity[88:96, 216:222] = 55.0
    azimuths = np.linspace(0, 359, 360)
    ranges_m = np.linspace(2000, 250000, 500)
    result = detect_objects_with_grid(
        reflectivity=reflectivity,
        azimuths=azimuths,
        ranges_m=ranges_m,
        radar_lat=35.0,
        radar_lon=-97.0,
    )
    assert len(result.objects) == 2
    assert len(result.object_masks) == 2
    sorted_objects = sorted(result.objects, key=lambda obj: obj.centroid_lon)
    assert sorted_objects[0].peak_label == "intense precipitation"
    assert sorted_objects[1].peak_label == "intense precipitation"


def test_detect_objects_does_not_split_single_core_blob():
    reflectivity = np.full((360, 500), np.nan)
    reflectivity[80:110, 190:230] = 25.0
    reflectivity[85:105, 198:222] = 40.0
    reflectivity[90:100, 204:216] = 55.0
    azimuths = np.linspace(0, 359, 360)
    ranges_m = np.linspace(2000, 250000, 500)
    result = detect_objects_with_grid(
        reflectivity=reflectivity,
        azimuths=azimuths,
        ranges_m=ranges_m,
        radar_lat=35.0,
        radar_lon=-97.0,
    )
    assert len(result.objects) == 1


def test_detect_objects_records_multilevel_threshold_hierarchy():
    reflectivity = np.full((360, 500), np.nan)
    reflectivity[80:110, 190:230] = 25.0
    reflectivity[85:105, 195:225] = 38.0
    reflectivity[90:100, 200:220] = 48.0
    reflectivity[93:97, 206:214] = 58.0
    azimuths = np.linspace(0, 359, 360)
    ranges_m = np.linspace(2000, 250000, 500)
    result = detect_objects_with_grid(
        reflectivity=reflectivity,
        azimuths=azimuths,
        ranges_m=ranges_m,
        radar_lat=35.0,
        radar_lon=-97.0,
    )
    assert len(result.objects) == 1
    hierarchy = result.object_hierarchy[result.objects[0].object_id]
    thresholds = {node.threshold for node in hierarchy}
    assert 20.0 in thresholds
    assert 30.0 in thresholds
    assert 40.0 in thresholds
    assert 50.0 in thresholds


def test_detect_objects_seam_storm_bearing_is_near_zero_not_180():
    """A storm straddling the 359.x/0.x azimuth wrap must report a bearing
    near 0/360 degrees, not 180.

    Regression test for the seam bug: interpolating a centroid ray index
    against the RAW (non-unwrapped) azimuth array averages e.g. 359.7 and
    0.2 into ~180 -- an exact reversal. A storm due north would be reported
    due south. Rotate a synthetic azimuth array so its wrap seam sits in the
    middle (index 99/100) instead of at the array boundary (359/0), so a
    storm can straddle it the same way a real storm straddles true north on
    the reference volume (seam observed at ray index 261 there).
    """
    n_az, n_rng = 360, 500
    azimuths = (np.arange(n_az, dtype=float) + 260.0) % 360.0
    assert azimuths[99] == pytest.approx(359.0)
    assert azimuths[100] == pytest.approx(0.0)
    ranges_m = np.linspace(2000, 250000, n_rng)
    reflectivity = np.full((n_az, n_rng), np.nan)
    # Storm spans ray indices 95-104, straddling the seam at 99/100 so the
    # dBZ-weighted centroid index lands exactly on the seam (99.5).
    reflectivity[95:105, 195:205] = 45.0

    objects = detect_objects(
        reflectivity=reflectivity,
        azimuths=azimuths,
        ranges_m=ranges_m,
        radar_lat=35.0,
        radar_lon=-97.0,
    )
    assert len(objects) == 1
    bearing = objects[0].bearing_deg
    distance_from_zero = min(bearing, 360.0 - bearing)
    assert distance_from_zero < 5.0, (
        f"expected bearing within 5 degrees of 0/360 (true azimuth at the "
        f"seam), got {bearing} -- looks like the 180-degree seam reversal"
    )


def test_detect_objects_seam_straddling_storm_is_one_object_not_two():
    """Ray 0 and ray N-1 are physically adjacent. A storm occupying the same
    gate indices at both the top and bottom row of the array is a single
    storm, and must be detected as one object with one centroid -- not two.

    Pre-fix, scipy.ndimage.label treats row 0 and row 359 as unconnected
    array edges and splits this storm into two objects.
    """
    reflectivity = np.full((360, 500), np.nan)
    reflectivity[0:6, 195:205] = 45.0
    reflectivity[354:360, 195:205] = 45.0
    azimuths = np.linspace(0, 359, 360)
    ranges_m = np.linspace(2000, 250000, 500)
    objects = detect_objects(
        reflectivity=reflectivity,
        azimuths=azimuths,
        ranges_m=ranges_m,
        radar_lat=35.0,
        radar_lon=-97.0,
    )
    assert len(objects) == 1, (
        f"expected one storm straddling the azimuth seam, got {len(objects)} -- "
        "looks like the array-seam component-labelling bug"
    )


def test_compute_object_properties_centroid_matches_pyart_georeferencing(radar):
    """A compact, uniform storm's reported centroid must match Py-ART's own
    gate_latitude/gate_longitude at that gate, not just be roughly close.

    Uses the real reference volume so azimuth spacing, range spacing and
    per-ray elevation are all realistic (not a synthetic linspace), and
    compares against Py-ART's own georeferencing -- the authority this
    project's math is supposed to agree with.
    """
    sweep_index = select_reflectivity_sweep(radar)
    sweep_start, sweep_end = radar.get_start_end(sweep_index)
    azimuths = np.asarray(radar.azimuth["data"][sweep_start:sweep_end + 1], dtype=float)
    ranges_m = np.asarray(radar.range["data"], dtype=float)
    elevations = np.asarray(radar.elevation["data"][sweep_start:sweep_end + 1], dtype=float)
    radar_lat = float(radar.latitude["data"][0])
    radar_lon = float(radar.longitude["data"][0])

    n_az = len(azimuths)
    n_rng = len(ranges_m)
    # Interior point, far from the 0/360 seam (observed at ray index 261 on
    # this volume) and from the array edges.
    center_ray, center_gate, half = 400, 300, 5
    assert center_ray - half > 261 + 20 or center_ray + half < 261 - 20

    reflectivity = np.full((n_az, n_rng), np.nan)
    reflectivity[center_ray - half:center_ray + half + 1, center_gate - half:center_gate + half + 1] = 45.0
    mask = ~np.isnan(reflectivity)

    obj = compute_object_properties(
        obj_mask=mask,
        reflectivity=reflectivity,
        azimuths=azimuths,
        ranges_m=ranges_m,
        radar_lat=radar_lat,
        radar_lon=radar_lon,
        object_id=1,
        elevation_deg=float(radar.fixed_angle["data"][sweep_index]),
        elevations=elevations,
    )
    assert obj is not None

    gate_lat = radar.gate_latitude["data"][sweep_start:sweep_end + 1]
    gate_lon = radar.gate_longitude["data"][sweep_start:sweep_end + 1]
    expected_lat = float(gate_lat[center_ray, center_gate])
    expected_lon = float(gate_lon[center_ray, center_gate])

    # Symmetric uniform block => weighted centroid index is exactly
    # (center_ray, center_gate), so the only slack allowed is the function's
    # own 4-decimal-place rounding (worst case ~5.5m) -- far tighter than
    # the 180-degree/164km seam bug or the 8-107m legacy polar_to_latlon
    # displacement this whole project is correcting.
    assert obj.centroid_lat == pytest.approx(expected_lat, abs=1e-3)
    assert obj.centroid_lon == pytest.approx(expected_lon, abs=1e-3)


def test_intensity_labels_do_not_assert_precipitation_phase():
    """ARW cannot distinguish rain from snow without melting-layer height, so
    it must not claim either."""
    for dbz in (25.0, 35.0, 45.0, 55.0, 65.0):
        assert "rain" not in classify_intensity(dbz)


def test_intensity_thresholds_are_unchanged():
    assert classify_intensity(25.0) == "light precipitation"
    assert classify_intensity(35.0) == "moderate precipitation"
    assert classify_intensity(45.0) == "heavy precipitation"
    assert classify_intensity(55.0) == "intense precipitation"
    assert classify_intensity(65.0) == "severe core"
    assert classify_intensity(10.0) == "drizzle"
