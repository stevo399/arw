import math
import numpy as np
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
)


def test_intensity_thresholds_defined():
    assert len(INTENSITY_THRESHOLDS) == 5
    assert INTENSITY_THRESHOLDS[0] == (20, 30, "light rain")
    assert INTENSITY_THRESHOLDS[-1] == (60, float("inf"), "severe core")


def test_classify_intensity():
    assert classify_intensity(15.0) == "drizzle"
    assert classify_intensity(25.0) == "light rain"
    assert classify_intensity(35.0) == "moderate rain"
    assert classify_intensity(45.0) == "heavy rain"
    assert classify_intensity(55.0) == "intense rain"
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
    assert objects[0].peak_label == "heavy rain"


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
    # Outer ring: light rain (25 dBZ)
    reflectivity[80:100, 190:210] = 25.0
    # Inner ring: moderate rain (35 dBZ)
    reflectivity[85:95, 195:205] = 35.0
    # Core: heavy rain (48 dBZ)
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
    assert "light rain" in layer_labels
    assert "moderate rain" in layer_labels
    assert "heavy rain" in layer_labels


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
    assert objects[0].peak_label == "intense rain"


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
    assert sorted_objects[0].peak_label == "intense rain"
    assert sorted_objects[1].peak_label == "intense rain"


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
