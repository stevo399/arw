import numpy as np

from src.tracking.motion_field import (
    GeographicMotionFieldEstimate,
    MotionFieldEstimate,
    estimate_geographic_motion_field,
    estimate_motion_field,
    predict_bbox,
    predict_latlon_position,
    predict_pixel_position,
)
from src.tracking.types import SegmentedStormObject
from src.detection import DetectedObject


def test_estimate_motion_field_empty_grids():
    prev_grid = np.full((32, 32), np.nan)
    curr_grid = np.full((32, 32), np.nan)
    estimate = estimate_motion_field(prev_grid, curr_grid, downsample=1)
    assert estimate.shift_rows == 0.0
    assert estimate.shift_cols == 0.0
    assert estimate.quality == 0.0


def test_estimate_motion_field_detects_simple_shift():
    prev_grid = np.full((64, 64), np.nan)
    curr_grid = np.full((64, 64), np.nan)
    prev_grid[20:28, 30:38] = 45.0
    curr_grid[23:31, 34:42] = 45.0
    estimate = estimate_motion_field(prev_grid, curr_grid, downsample=1)
    assert isinstance(estimate, MotionFieldEstimate)
    # The bulk-motion estimate returns the shift needed to align current back to previous.
    assert estimate.shift_rows == -3.0
    assert estimate.shift_cols == -4.0
    assert estimate.quality > 0.0


def test_estimate_motion_field_ignores_subthreshold_noise():
    prev_grid = np.full((64, 64), np.nan)
    curr_grid = np.full((64, 64), np.nan)
    prev_grid[20:28, 30:38] = 15.0
    curr_grid[23:31, 34:42] = 15.0
    estimate = estimate_motion_field(prev_grid, curr_grid, downsample=1)
    assert estimate.shift_rows == 0.0
    assert estimate.shift_cols == 0.0
    assert estimate.quality == 0.0


def test_predict_pixel_position():
    estimate = MotionFieldEstimate(
        shift_rows=-2.0,
        shift_cols=5.0,
        quality=3.0,
        source="phase_correlation",
        downsample=1,
    )
    assert predict_pixel_position(10, 20, estimate) == (8.0, 25.0)


def test_predict_bbox():
    estimate = MotionFieldEstimate(
        shift_rows=-2.0,
        shift_cols=5.0,
        quality=3.0,
        source="phase_correlation",
        downsample=1,
    )
    assert predict_bbox((10, 20, 15, 30), estimate) == (8.0, 25.0, 13.0, 35.0)


def _make_segmented_object(
    object_id: int,
    lat: float,
    lon: float,
    area_km2: float,
    peak_dbz: float,
) -> SegmentedStormObject:
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:5, 2:5] = True
    return SegmentedStormObject(
        object_id=object_id,
        detected_object=DetectedObject(
            object_id=object_id,
            centroid_lat=lat,
            centroid_lon=lon,
            distance_km=40.0,
            bearing_deg=270.0,
            peak_dbz=peak_dbz,
            peak_label="heavy rain",
            area_km2=area_km2,
            layers=[],
        ),
        mask=mask,
        bbox=(2, 2, 4, 4),
        pixel_count=9,
    )


def test_estimate_geographic_motion_field():
    previous = [
        _make_segmented_object(1, 35.0, -97.0, area_km2=10.0, peak_dbz=40.0),
        _make_segmented_object(2, 35.2, -96.8, area_km2=20.0, peak_dbz=50.0),
    ]
    current = [
        _make_segmented_object(1, 35.1, -96.9, area_km2=10.0, peak_dbz=40.0),
        _make_segmented_object(2, 35.3, -96.7, area_km2=20.0, peak_dbz=50.0),
    ]
    estimate = estimate_geographic_motion_field(previous, current)
    assert estimate.delta_lat == 0.1
    assert estimate.delta_lon == 0.1
    assert estimate.quality > 0.0


def test_predict_latlon_position():
    estimate = GeographicMotionFieldEstimate(
        delta_lat=0.15,
        delta_lon=-0.2,
        quality=0.8,
        source="object_weighted_centroid",
    )
    assert predict_latlon_position(35.0, -97.0, estimate) == (35.15, -97.2)
