import numpy as np

from src.tracking.segmentation import (
    SegmentationResult,
    adapt_detection_result,
    compute_bbox,
    segment_storm_objects,
)
from src.detection import DetectionResult, DetectedObject


def _make_detection_result() -> DetectionResult:
    reflectivity_shape = (360, 500)
    labeled_grid = np.zeros(reflectivity_shape, dtype=int)
    mask1 = np.zeros(reflectivity_shape, dtype=bool)
    mask1[10:20, 50:60] = True
    mask2 = np.zeros(reflectivity_shape, dtype=bool)
    mask2[200:210, 300:315] = True
    labeled_grid[mask1] = 1
    labeled_grid[mask2] = 2
    objects = [
        DetectedObject(
            object_id=1,
            centroid_lat=35.1,
            centroid_lon=-97.1,
            distance_km=25.0,
            bearing_deg=90.0,
            peak_dbz=35.0,
            peak_label="moderate rain",
            area_km2=10.0,
            layers=[],
        ),
        DetectedObject(
            object_id=2,
            centroid_lat=35.2,
            centroid_lon=-97.2,
            distance_km=50.0,
            bearing_deg=180.0,
            peak_dbz=55.0,
            peak_label="intense rain",
            area_km2=20.0,
            layers=[],
        ),
    ]
    return DetectionResult(
        objects=objects,
        labeled_grid=labeled_grid,
        object_masks={1: mask1, 2: mask2},
    )


def test_compute_bbox():
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:5, 6:9] = True
    assert compute_bbox(mask) == (2, 6, 4, 8)


def test_adapt_detection_result_returns_segmented_objects():
    result = adapt_detection_result(_make_detection_result())
    assert isinstance(result, SegmentationResult)
    assert len(result.objects) == 2
    assert result.objects[0].detected_object.object_id == 1
    assert result.objects[0].bbox == (10, 50, 19, 59)
    assert result.objects[0].pixel_count == 100


def test_adapt_detection_result_preserves_object_order():
    result = adapt_detection_result(_make_detection_result())
    assert [obj.object_id for obj in result.objects] == [1, 2]


def test_segment_storm_objects_wraps_existing_detection():
    reflectivity = np.full((360, 500), np.nan)
    reflectivity[85:95, 195:205] = 45.0
    azimuths = np.linspace(0, 359, 360)
    ranges_m = np.linspace(2000, 250000, 500)
    result = segment_storm_objects(
        reflectivity=reflectivity,
        azimuths=azimuths,
        ranges_m=ranges_m,
        radar_lat=35.0,
        radar_lon=-97.0,
    )
    assert isinstance(result, SegmentationResult)
    assert len(result.objects) == 1
    obj = result.objects[0]
    assert obj.detected_object.peak_dbz == 45.0
    assert obj.pixel_count == 100
    assert obj.bbox == (85, 195, 94, 204)
