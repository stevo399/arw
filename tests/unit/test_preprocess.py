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


from src.preprocess import preprocess_sweep


def _sweep(reflectivity, **overrides):
    from src.parser import SweepData

    n_az, n_rng = reflectivity.shape
    base = dict(
        reflectivity=reflectivity,
        rhohv=np.full(reflectivity.shape, 0.99),
        zdr=np.full(reflectivity.shape, 0.5),
        azimuths=np.linspace(0.0, 355.0, n_az),
        ranges_m=np.arange(2125.0, 2125.0 + 250.0 * n_rng, 250.0),
        elevation_angle=0.5,
        elevations=np.full(n_az, 0.5),
        elevation_angles=[0.5],
        radar_lat=35.3331,
        radar_lon=-97.2778,
        radar_alt_m=390.0,
        timestamp="2026-08-23T00:00:00Z",
    )
    base.update(overrides)
    return SweepData(**base)


def test_preprocess_returns_quality_with_class_fractions():
    sweep = _sweep(np.full((36, 40), 30.0))
    _processed, quality, _rejected = preprocess_sweep(sweep, [])
    assert quality.class_fractions
    assert 0.0 <= quality.rejected_fraction <= 1.0


def test_quality_control_runs_before_speckle_removal():
    """A small intense core must survive. If speckle removal ran first it would
    delete the core before QC could protect it."""
    reflectivity = np.full((36, 40), np.nan)
    reflectivity[10, 10] = 58.0
    reflectivity[10, 11] = 56.0
    sweep = _sweep(reflectivity)
    processed, _quality, _rejected = preprocess_sweep(sweep, [])
    assert np.isfinite(processed.reflectivity[10, 10])


def test_preprocess_attaches_gate_classification():
    sweep = _sweep(np.full((36, 40), 30.0))
    processed, _quality, _rejected = preprocess_sweep(sweep, [])
    assert processed.gate_classification is not None


def test_speckle_removal_runs_after_classification_not_before():
    """Wiring proof for the QC-then-speckle order inside preprocess_sweep.

    A 34 dBZ single-gate core sits inside a 3x3 block of weak, ambiguous
    22 dBZ echo (no rhohv/zdr, so the classifier falls back to reflectivity
    and texture alone). The 3x3 block classifies as: the 8 surrounding
    gates as biological (weak, high-texture, unprotected -- below the 50 dBZ
    hard floor and with no rotation signature) and the center as
    precipitation.

    The two orderings genuinely diverge here, in the opposite direction from
    what a "protect small intense cores" intuition suggests:

    - QC-then-speckle (the order preprocess_sweep is wired for): QC rejects
      the 8 surrounding gates first, leaving the center as a 1-gate island.
      Speckle removal then sees a component of size 1 (<= MAX_SPECKLE_PIXELS)
      whose peak (34) is below MIN_SPECKLE_PEAK_DBZ_TO_KEEP (35), and deletes
      it. The core is lost.
    - speckle-then-QC (the forbidden order): speckle removal runs on the raw
      3x3 block first. The whole block is one connected 9-gate component,
      too large for speckle's size filter, so nothing is removed. QC then
      runs on that untouched data, rejects the 8 clutter gates, and the
      center gate -- already past the one and only speckle pass -- survives
      at its original value.

    This test locks down the mandated ordering's actual, verified behavior
    (the core is removed) so that if the two calls inside preprocess_sweep
    are ever swapped, this test fails: the swapped order keeps the core
    (finite, == 34.0) instead of removing it.

    This is called out explicitly in the Task 13 report: the ordering
    constraint's stated rationale ("QC protects cores speckle would delete")
    does not hold for this scenario -- here QC-then-speckle is what causes
    the loss. Not fixed here per instructions to wire the pipeline exactly as
    specified without adjusting thresholds or protection to compensate.
    """
    reflectivity = np.full((36, 40), np.nan)
    reflectivity[9:12, 9:12] = 22.0
    reflectivity[10, 10] = 34.0
    sweep = _sweep(reflectivity, rhohv=None, zdr=None)

    processed, _quality, _rejected = preprocess_sweep(sweep, [])

    assert np.isnan(processed.reflectivity[10, 10]), (
        "expected the mandated QC-then-speckle order to remove this core; "
        "if this now fails, the two steps inside preprocess_sweep were "
        "likely swapped"
    )
