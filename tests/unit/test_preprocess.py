import numpy as np

from src.parser import SweepData
from src.preprocess import preprocess_reflectivity_data


def _make_reflectivity_data(grid: np.ndarray) -> SweepData:
    return SweepData(
        reflectivity=grid,
        azimuths=np.linspace(0, 359, grid.shape[0]),
        ranges_m=np.linspace(2000, 250000, grid.shape[1]),
        radar_lat=35.3331,
        radar_lon=-97.2778,
        elevation_angle=0.5,
        elevations=np.full(grid.shape[0], 0.5),
        elevation_angles=[0.5],
        radar_alt_m=390.0,
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


def test_preprocess_blanks_polarimetric_fields_at_removed_speckle_gates():
    """Speckle removal must not break co-registration.

    A gate whose weak, isolated reflectivity gets NaN'd out must also lose
    its RhoHV/ZDR reading -- otherwise a consumer reading `sweep.rhohv` at
    that gate sees a value with no corresponding reflectivity, violating
    SweepData's co-registered-fields promise.
    """
    grid = np.full((32, 32), np.nan)
    grid[10, 10] = 28.0  # isolated weak speckle pixel -- removed
    grid[20, 20] = 45.0  # compact strong core -- kept
    sweep = _make_reflectivity_data(grid)
    sweep.rhohv = np.full((32, 32), 0.95)
    sweep.zdr = np.full((32, 32), 1.5)

    processed, quality = preprocess_reflectivity_data(sweep)

    assert quality.removed_speckle_pixels == 1
    assert np.isnan(processed.reflectivity[10, 10])
    assert np.isnan(processed.rhohv[10, 10]), "rhohv must be blanked where reflectivity was removed"
    assert np.isnan(processed.zdr[10, 10]), "zdr must be blanked where reflectivity was removed"

    # The kept gate's polarimetric readings must survive untouched.
    assert processed.reflectivity[20, 20] == 45.0
    assert processed.rhohv[20, 20] == 0.95
    assert processed.zdr[20, 20] == 1.5

    # A gate that was already NaN in the original reflectivity (not
    # "removed" by speckle filtering) must not have its rhohv/zdr touched.
    assert processed.rhohv[0, 0] == 0.95
    assert processed.zdr[0, 0] == 1.5
