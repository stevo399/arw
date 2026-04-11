from dataclasses import dataclass

import numpy as np

from src.buffer import BufferedScan
from src.detection import polar_to_latlon
from src.tracking.types import SegmentedStormObject


DEFAULT_DOWNSAMPLE = 4
DEFAULT_MIN_DBZ = 20.0


@dataclass
class MotionFieldEstimate:
    shift_rows: float
    shift_cols: float
    quality: float
    source: str
    downsample: int


@dataclass
class GeographicMotionFieldEstimate:
    delta_lat: float
    delta_lon: float
    quality: float
    source: str


def _preprocess_reflectivity(
    reflectivity: np.ndarray,
    min_dbz: float = DEFAULT_MIN_DBZ,
    downsample: int = DEFAULT_DOWNSAMPLE,
) -> np.ndarray:
    """Prepare a reflectivity grid for bulk motion estimation."""
    grid = np.nan_to_num(reflectivity.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
    grid = np.where(grid >= min_dbz, grid, 0.0)
    if downsample > 1:
        grid = grid[::downsample, ::downsample]
    return grid


def _wrap_shift(index: int, size: int) -> int:
    """Convert FFT peak index into a signed shift."""
    return index - size if index > size // 2 else index


def _weighted_centroid(grid: np.ndarray) -> tuple[float, float]:
    rows, cols = np.indices(grid.shape)
    weights = np.clip(grid, a_min=0.0, a_max=None)
    total = float(weights.sum())
    if total <= 0.0:
        raise ValueError("cannot compute centroid without positive signal")
    centroid_row = float((rows * weights).sum() / total)
    centroid_col = float((cols * weights).sum() / total)
    return centroid_row, centroid_col


def _weighted_object_centroid(objects: list[SegmentedStormObject]) -> tuple[float, float, float]:
    total_weight = 0.0
    lat_sum = 0.0
    lon_sum = 0.0
    for obj in objects:
        detected = obj.detected_object
        weight = max(detected.area_km2, 0.1) * max(detected.peak_dbz, DEFAULT_MIN_DBZ)
        lat_sum += detected.centroid_lat * weight
        lon_sum += detected.centroid_lon * weight
        total_weight += weight
    if total_weight <= 0.0:
        raise ValueError("cannot compute object centroid without positive weight")
    return lat_sum / total_weight, lon_sum / total_weight, total_weight


def estimate_motion_field(
    previous_reflectivity: np.ndarray,
    current_reflectivity: np.ndarray,
    *,
    min_dbz: float = DEFAULT_MIN_DBZ,
    downsample: int = DEFAULT_DOWNSAMPLE,
) -> MotionFieldEstimate:
    """Estimate bulk scene displacement between two reflectivity grids."""
    prev_grid = _preprocess_reflectivity(previous_reflectivity, min_dbz=min_dbz, downsample=downsample)
    curr_grid = _preprocess_reflectivity(current_reflectivity, min_dbz=min_dbz, downsample=downsample)

    if not np.any(prev_grid) or not np.any(curr_grid):
        return MotionFieldEstimate(
            shift_rows=0.0,
            shift_cols=0.0,
            quality=0.0,
            source="phase_correlation",
            downsample=downsample,
        )

    prev_fft = np.fft.fft2(prev_grid)
    curr_fft = np.fft.fft2(curr_grid)
    cross_power = prev_fft * np.conj(curr_fft)
    magnitude = np.abs(cross_power)
    cross_power /= np.where(magnitude == 0.0, 1.0, magnitude)
    correlation = np.fft.ifft2(cross_power)
    correlation_abs = np.abs(correlation)
    peak_index = np.unravel_index(np.argmax(correlation_abs), correlation_abs.shape)
    raw_row_shift = _wrap_shift(int(peak_index[0]), correlation_abs.shape[0])
    raw_col_shift = _wrap_shift(int(peak_index[1]), correlation_abs.shape[1])

    peak_value = float(correlation_abs[peak_index])
    mean_value = float(np.mean(correlation_abs))
    quality = 0.0 if peak_value <= 0.0 else max(0.0, min(1.0, (peak_value - mean_value) / peak_value))

    return MotionFieldEstimate(
        shift_rows=round(float(raw_row_shift * downsample), 2),
        shift_cols=round(float(raw_col_shift * downsample), 2),
        quality=round(quality, 3),
        source="phase_correlation",
        downsample=downsample,
    )


def estimate_geographic_motion_field(
    previous_objects: list[SegmentedStormObject],
    current_objects: list[SegmentedStormObject],
) -> GeographicMotionFieldEstimate:
    """Estimate bulk geographic displacement from segmented storm objects."""
    if not previous_objects or not current_objects:
        return GeographicMotionFieldEstimate(
            delta_lat=0.0,
            delta_lon=0.0,
            quality=0.0,
            source="object_weighted_centroid",
        )

    prev_lat, prev_lon, prev_weight = _weighted_object_centroid(previous_objects)
    curr_lat, curr_lon, curr_weight = _weighted_object_centroid(current_objects)
    weight_ratio = min(prev_weight, curr_weight) / max(prev_weight, curr_weight) if max(prev_weight, curr_weight) > 0 else 0.0
    return GeographicMotionFieldEstimate(
        delta_lat=round(curr_lat - prev_lat, 4),
        delta_lon=round(curr_lon - prev_lon, 4),
        quality=round(weight_ratio, 3),
        source="object_weighted_centroid",
    )


def estimate_scan_geographic_motion_field(
    previous_scan: BufferedScan,
    current_scan: BufferedScan,
    *,
    min_dbz: float = DEFAULT_MIN_DBZ,
    downsample: int = DEFAULT_DOWNSAMPLE,
) -> GeographicMotionFieldEstimate:
    """Estimate geographic displacement from full-scan phase correlation.

    This is closer to the published first-guess motion methodology than an
    object-centroid-only estimate and is therefore preferred for guidance.
    """
    previous_ref = previous_scan.reflectivity_data
    current_ref = current_scan.reflectivity_data
    estimate = estimate_motion_field(
        previous_ref.reflectivity,
        current_ref.reflectivity,
        min_dbz=min_dbz,
        downsample=downsample,
    )

    prev_grid = _preprocess_reflectivity(previous_ref.reflectivity, min_dbz=min_dbz, downsample=1)
    curr_grid = _preprocess_reflectivity(current_ref.reflectivity, min_dbz=min_dbz, downsample=1)
    if estimate.quality <= 0.0 or not np.any(prev_grid) or not np.any(curr_grid):
        return GeographicMotionFieldEstimate(
            delta_lat=0.0,
            delta_lon=0.0,
            quality=0.0,
            source="phase_correlation",
        )

    prev_row, prev_col = _weighted_centroid(prev_grid)
    curr_row = prev_row - estimate.shift_rows
    curr_col = prev_col - estimate.shift_cols

    row_coords = np.arange(len(previous_ref.azimuths))
    col_coords = np.arange(len(previous_ref.ranges_m))
    prev_azimuth = float(np.interp(prev_row, row_coords, previous_ref.azimuths))
    prev_range_m = float(np.interp(prev_col, col_coords, previous_ref.ranges_m))
    curr_azimuth = float(np.interp(curr_row, row_coords, current_ref.azimuths))
    curr_range_m = float(np.interp(curr_col, col_coords, current_ref.ranges_m))

    prev_lat, prev_lon = polar_to_latlon(
        previous_ref.radar_lat,
        previous_ref.radar_lon,
        prev_azimuth,
        prev_range_m,
    )
    curr_lat, curr_lon = polar_to_latlon(
        current_ref.radar_lat,
        current_ref.radar_lon,
        curr_azimuth,
        curr_range_m,
    )
    return GeographicMotionFieldEstimate(
        delta_lat=round(curr_lat - prev_lat, 4),
        delta_lon=round(curr_lon - prev_lon, 4),
        quality=estimate.quality,
        source="phase_correlation",
    )


def predict_pixel_position(row: float, col: float, estimate: MotionFieldEstimate) -> tuple[float, float]:
    """Project a pixel location forward using a bulk motion-field estimate."""
    return row + estimate.shift_rows, col + estimate.shift_cols


def predict_bbox(
    bbox: tuple[int, int, int, int],
    estimate: MotionFieldEstimate,
) -> tuple[float, float, float, float]:
    """Project a bounding box forward using a bulk motion-field estimate."""
    min_row, min_col, max_row, max_col = bbox
    predicted_min_row, predicted_min_col = predict_pixel_position(min_row, min_col, estimate)
    predicted_max_row, predicted_max_col = predict_pixel_position(max_row, max_col, estimate)
    return (
        predicted_min_row,
        predicted_min_col,
        predicted_max_row,
        predicted_max_col,
    )


def predict_latlon_position(
    latitude: float,
    longitude: float,
    estimate: GeographicMotionFieldEstimate,
) -> tuple[float, float]:
    """Project a lat/lon position forward using a geographic motion-field estimate."""
    return latitude + estimate.delta_lat, longitude + estimate.delta_lon
