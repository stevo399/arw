from dataclasses import dataclass, replace

import numpy as np
from scipy.ndimage import label

from src.parser import ReflectivityData

MIN_DBZ_FOR_OBJECTS = 20.0
MAX_SPECKLE_PIXELS = 3
MIN_SPECKLE_PEAK_DBZ_TO_KEEP = 35.0
HIGH_MISSING_FRACTION = 0.35
NOTICEABLE_SPECKLE_FRACTION = 0.005


@dataclass
class ScanQuality:
    score: float
    finite_fraction: float
    removed_speckle_pixels: int
    removed_speckle_fraction: float
    flags: list[str]


def _remove_weak_speckle(reflectivity: np.ndarray) -> tuple[np.ndarray, int]:
    processed = np.array(reflectivity, copy=True)
    storm_like = ~np.isnan(processed) & (processed >= MIN_DBZ_FOR_OBJECTS)
    labeled, count = label(storm_like, structure=np.ones((3, 3), dtype=int))
    removed_pixels = 0
    for component_id in range(1, count + 1):
        component_mask = labeled == component_id
        component_pixels = int(np.count_nonzero(component_mask))
        if component_pixels > MAX_SPECKLE_PIXELS:
            continue
        peak_dbz = float(np.nanmax(processed[component_mask]))
        if peak_dbz >= MIN_SPECKLE_PEAK_DBZ_TO_KEEP:
            continue
        processed[component_mask] = np.nan
        removed_pixels += component_pixels
    return processed, removed_pixels


def assess_scan_quality(
    original_reflectivity: np.ndarray,
    processed_reflectivity: np.ndarray,
    removed_speckle_pixels: int,
) -> ScanQuality:
    total_pixels = int(original_reflectivity.size)
    finite_fraction = float(np.count_nonzero(~np.isnan(original_reflectivity))) / float(total_pixels)
    removed_fraction = float(removed_speckle_pixels) / float(total_pixels)
    score = max(0.0, min(1.0, 1.0 - ((1.0 - finite_fraction) * 0.7) - min(removed_fraction * 20.0, 0.3)))
    flags: list[str] = []
    if (1.0 - finite_fraction) >= HIGH_MISSING_FRACTION:
        flags.append("high_missing_fraction")
    if removed_speckle_pixels > 0:
        flags.append("speckle_filtered")
    if not np.any(~np.isnan(processed_reflectivity) & (processed_reflectivity >= MIN_DBZ_FOR_OBJECTS)):
        flags.append("no_object_scale_echo")
    return ScanQuality(
        score=round(score, 3),
        finite_fraction=round(finite_fraction, 3),
        removed_speckle_pixels=removed_speckle_pixels,
        removed_speckle_fraction=round(removed_fraction, 5),
        flags=flags,
    )


def preprocess_reflectivity_data(reflectivity_data: ReflectivityData) -> tuple[ReflectivityData, ScanQuality]:
    processed_reflectivity, removed_speckle_pixels = _remove_weak_speckle(reflectivity_data.reflectivity)
    quality = assess_scan_quality(
        original_reflectivity=reflectivity_data.reflectivity,
        processed_reflectivity=processed_reflectivity,
        removed_speckle_pixels=removed_speckle_pixels,
    )
    return replace(reflectivity_data, reflectivity=processed_reflectivity), quality
