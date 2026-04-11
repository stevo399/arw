import numpy as np

from src.parser import ReflectivityData
from src.preprocess import preprocess_reflectivity_data


def _make_reflectivity_data(grid: np.ndarray) -> ReflectivityData:
    return ReflectivityData(
        reflectivity=grid,
        azimuths=np.linspace(0, 359, grid.shape[0]),
        ranges_m=np.linspace(2000, 250000, grid.shape[1]),
        radar_lat=35.3331,
        radar_lon=-97.2778,
        elevation_angle=0.5,
        elevation_angles=[0.5],
        timestamp="2026-04-11T00:00:00Z",
    )


def test_preprocess_removes_tiny_weak_speckle():
    grid = np.full((32, 32), np.nan)
    grid[10, 10] = 28.0
    reflectivity, quality = preprocess_reflectivity_data(_make_reflectivity_data(grid))
    assert np.isnan(reflectivity.reflectivity[10, 10])
    assert quality.removed_speckle_pixels == 1
    assert "speckle_filtered" in quality.flags


def test_preprocess_keeps_compact_strong_core():
    grid = np.full((32, 32), np.nan)
    grid[10, 10] = 45.0
    reflectivity, quality = preprocess_reflectivity_data(_make_reflectivity_data(grid))
    assert reflectivity.reflectivity[10, 10] == 45.0
    assert quality.removed_speckle_pixels == 0


def test_preprocess_flags_high_missing_fraction():
    grid = np.full((32, 32), np.nan)
    grid[0:4, 0:4] = 25.0
    _, quality = preprocess_reflectivity_data(_make_reflectivity_data(grid))
    assert "high_missing_fraction" in quality.flags
    assert quality.score < 0.8
