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


def _many_layered_storms(storm_count: int = 40) -> np.ndarray:
    """Separate storms on a full-resolution grid, each with 30/40/50 dBZ cores."""
    reflectivity = np.full((720, 1832), np.nan)
    for index in range(storm_count):
        row, col = 20 + (index % 8) * 85, 200 + (index // 8) * 300
        reflectivity[row:row + 40, col:col + 60] = 25.0
        reflectivity[row + 8:row + 32, col + 10:col + 50] = 35.0
        reflectivity[row + 14:row + 26, col + 20:col + 40] = 45.0
        reflectivity[row + 17:row + 23, col + 26:col + 34] = 55.0
    return reflectivity


def _detect_full_grid(reflectivity):
    return detect_objects_with_grid(
        reflectivity=reflectivity,
        azimuths=np.linspace(0, 359.5, reflectivity.shape[0]),
        ranges_m=np.linspace(2000, 460000, reflectivity.shape[1]),
        radar_lat=35.0,
        radar_lon=-97.0,
    )


def test_detection_result_retains_no_hierarchy_masks():
    result = _detect_full_grid(_many_layered_storms(8))
    nodes = [node for hierarchy in result.object_hierarchy.values() for node in hierarchy]
    assert nodes, "fixture must produce threshold hierarchies"
    assert {node.threshold for node in nodes} >= {20.0, 30.0, 40.0, 50.0}
    assert all(not isinstance(value, np.ndarray) for node in nodes for value in vars(node).values())


def test_detection_retains_only_object_masks_and_label_grid():
    """Detection used to keep a full-grid mask for every hierarchy node of
    every object: about 2.2 GB for 63 real KTLX objects (2026-09-13)."""
    import tracemalloc

    reflectivity = _many_layered_storms(40)
    tracemalloc.start()
    try:
        before, _ = tracemalloc.get_traced_memory()
        result = _detect_full_grid(reflectivity)
        retained = tracemalloc.get_traced_memory()[0] - before
    finally:
        tracemalloc.stop()

    assert len(result.objects) == 40
    expected = sum(mask.nbytes for mask in result.object_masks.values()) + result.labeled_grid.nbytes
    assert retained < expected + 20_000_000, (
        f"detection retained {retained / 1e6:.1f} MB; masks and labels are {expected / 1e6:.1f} MB"
    )


def test_one_storm_complex_with_many_cores_does_not_allocate_a_grid_per_core():
    """A 40,000-gate complex with 100 separate 55 dBZ cores.

    Each core is a hierarchy component at 30, 40 and 50 dBZ.  Building the
    hierarchy used to allocate one full-grid mask per component: on real KEMX
    data, one complex of 36,222 gates produced 1,329 nodes and a 1.77 GB
    peak (2026-09-13).
    """
    import tracemalloc

    reflectivity = np.full((720, 1832), np.nan)
    reflectivity[100:300, 400:600] = 25.0
    reflectivity[105:300:20, 405:600:20] = 55.0
    tracemalloc.start()
    try:
        start, _ = tracemalloc.get_traced_memory()
        result = _detect_full_grid(reflectivity)
        peak = tracemalloc.get_traced_memory()[1] - start
    finally:
        tracemalloc.stop()

    nodes = [node for hierarchy in result.object_hierarchy.values() for node in hierarchy]
    assert len(nodes) > 300, "fixture must produce a large hierarchy"
    assert peak < 100_000_000, f"detection peaked at {peak / 1e6:.1f} MB for {len(nodes)} hierarchy nodes"



def _single_object(reflectivity, azimuths, ranges_m):
    objects = detect_objects(
        reflectivity=reflectivity,
        azimuths=azimuths,
        ranges_m=ranges_m,
        radar_lat=35.0,
        radar_lon=-97.0,
    )
    assert len(objects) == 1
    return objects[0]


def test_storm_straddling_the_first_and_last_ray_is_placed_at_the_storm():
    """Ray 0 and ray N-1 are adjacent, so a storm occupying both is one storm
    due north here.  Averaging its ray *indices* (about 3 and 357) gives ray
    180 -- due south, on the far side of the radar.  Real volumes misplaced
    such storms by 12 to 451 km (2026-09-13 survey).
    """
    reflectivity = np.full((360, 500), np.nan)
    reflectivity[0:6, 195:205] = 45.0
    reflectivity[354:360, 195:205] = 45.0
    obj = _single_object(reflectivity, np.linspace(0, 359, 360), np.linspace(2000, 250000, 500))

    assert min(obj.bearing_deg, 360.0 - obj.bearing_deg) < 1.0, obj.bearing_deg
    assert obj.centroid_lat > 35.0
    assert obj.centroid_lon == pytest.approx(-97.0, abs=0.02)


def test_storm_centroid_weights_each_gate_by_the_ground_it_covers():
    """A uniform storm from 50 to 150 km covers more ground per gate at far
    range.  Weighting by area, its centroid lies at (150^3 - 50^3) / 3 divided
    by (150^2 - 50^2) / 2 = 108.3 km, not at the 100 km mid-range."""
    ranges_m = np.arange(250.0, 250000.0, 250.0)
    reflectivity = np.full((360, len(ranges_m)), np.nan)
    in_storm = (ranges_m >= 50000.0) & (ranges_m <= 150000.0)
    reflectivity[88:93, in_storm] = 40.0
    obj = _single_object(reflectivity, np.arange(360, dtype=float), ranges_m)

    assert obj.distance_km == pytest.approx(108.3, abs=0.3)
    assert obj.bearing_deg == pytest.approx(90.0, abs=0.5)


def test_arc_shaped_storm_centroid_is_its_geographic_centre():
    """A band of storm spanning 90 degrees of azimuth at 100 km range has its
    geographic centre inside the arc, at 100 * sin(45) / (pi / 4) = 90.0 km.
    Averaging in azimuth and range places it on the arc at 100 km."""
    ranges_m = np.arange(250.0, 250000.0, 250.0)
    reflectivity = np.full((360, len(ranges_m)), np.nan)
    band = (ranges_m >= 99000.0) & (ranges_m <= 101000.0)
    reflectivity[45:136, band] = 40.0
    obj = _single_object(reflectivity, np.arange(360, dtype=float), ranges_m)

    assert obj.distance_km == pytest.approx(90.0, abs=0.3)
    assert obj.bearing_deg == pytest.approx(90.0, abs=0.5)


def test_single_core_straddling_the_first_and_last_ray_is_not_split_in_two():
    """One storm whose 40 and 55 dBZ cores straddle ray 0 has one core.  The
    threshold hierarchy labelled without wrap-around saw two cores and split
    it into two storms at the array edge."""
    reflectivity = np.full((360, 500), np.nan)
    for rows in (slice(0, 15), slice(345, 360)):
        reflectivity[rows, 190:230] = 25.0
    for rows in (slice(0, 8), slice(352, 360)):
        reflectivity[rows, 198:222] = 40.0
    for rows in (slice(0, 4), slice(356, 360)):
        reflectivity[rows, 204:216] = 55.0
    objects = detect_objects(
        reflectivity=reflectivity,
        azimuths=np.linspace(0, 359, 360),
        ranges_m=np.linspace(2000, 250000, 500),
        radar_lat=35.0,
        radar_lon=-97.0,
    )
    assert len(objects) == 1


def test_split_gives_each_gate_to_the_core_nearest_on_the_ground():
    """At about 150 km one ray (1 degree here) spans 2.6 km but one gate only
    0.5 km.  The gate at ray 100, gate 300 is 5 km from core A (ray 100, gate
    290) and 21 km from core B (ray 108, gate 300), yet nearer B in ray and
    gate numbers (8 against 10)."""
    reflectivity = np.full((360, 500), np.nan)
    reflectivity[94:114, 282:310] = 25.0
    reflectivity[97:104, 286:295] = 40.0
    reflectivity[99:102, 288:293] = 55.0
    reflectivity[106:111, 296:305] = 40.0
    reflectivity[107:110, 298:303] = 55.0
    result = detect_objects_with_grid(
        reflectivity=reflectivity,
        azimuths=np.arange(360, dtype=float),
        ranges_m=np.linspace(2000, 250000, 500),
        radar_lat=35.0,
        radar_lon=-97.0,
    )
    assert len(result.objects) == 2
    core_a = next(mask for mask in result.object_masks.values() if mask[100, 290])
    assert core_a[100, 300]
