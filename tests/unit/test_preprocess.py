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
    _processed, quality, _advisory = preprocess_sweep(sweep, [])
    assert quality.class_fractions
    assert 0.0 <= quality.advisory_fraction <= 1.0


def test_small_intense_core_survives_speckle_removal():
    """A small intense core must survive despeckling.

    Formerly titled `test_quality_control_runs_before_speckle_removal` and
    framed as proving QC's position ahead of speckle removal protects this
    core. That framing no longer holds: since the 2026-08-26 amendment QC
    never deletes anything, so it cannot "protect" a core from speckle
    removal by running first -- ordering QC before or after speckle removal
    now has no effect on reflectivity at all (see
    test_qc_ordering_no_longer_affects_what_speckle_removal_sees below). What
    actually keeps this core is `_remove_weak_speckle`'s own
    MIN_SPECKLE_PEAK_DBZ_TO_KEEP=35 exception: this 2-gate component peaks at
    58 dBZ, comfortably above that floor, so speckle removal keeps it
    regardless of QC.
    """
    reflectivity = np.full((36, 40), np.nan)
    reflectivity[10, 10] = 58.0
    reflectivity[10, 11] = 56.0
    sweep = _sweep(reflectivity)
    processed, _quality, _advisory = preprocess_sweep(sweep, [])
    assert np.isfinite(processed.reflectivity[10, 10])


def test_preprocess_attaches_gate_classification():
    sweep = _sweep(np.full((36, 40), 30.0))
    processed, _quality, _advisory = preprocess_sweep(sweep, [])
    assert processed.gate_classification is not None


def test_qc_ordering_no_longer_affects_what_speckle_removal_sees():
    """Replaces `test_speckle_removal_runs_after_classification_not_before`.

    That test locked down a QC-then-speckle behavior from before the
    2026-08-26 amendment: QC used to NaN out flagged gates, so running it
    before speckle removal could strip a component down to an isolated
    island that speckle removal then deleted. It asserted the resulting core
    (reflectivity[10, 10]) came out NaN, and said so would fail if QC and
    speckle removal were ever swapped.

    That premise is now false. QC no longer modifies reflectivity at all
    (`apply_quality_control` classifies gates but returns the field
    byte-for-byte unchanged -- see
    test_qc_apply.test_reflectivity_is_never_modified_by_quality_control), so
    the ordering of QC vs. speckle removal inside `preprocess_sweep` can no
    longer change what speckle removal sees, and therefore can no longer
    change the output reflectivity either. This test asserts that new
    invariant directly, using the same fixture as the old test: a 34 dBZ
    single-gate core inside a 3x3 block of weak, ambiguous 22 dBZ echo (no
    rhohv/zdr, so the classifier falls back to reflectivity and texture
    alone -- the 8 surrounding gates classify as biological, the center as
    precipitation). Both orderings now see the SAME thing: a single connected
    9-gate component, too large for speckle's MAX_SPECKLE_PIXELS=3 filter, so
    nothing is removed and the center survives at its original value -- the
    opposite of what the pre-amendment ordering produced.
    """
    reflectivity = np.full((36, 40), np.nan)
    reflectivity[9:12, 9:12] = 22.0
    reflectivity[10, 10] = 34.0
    sweep = _sweep(reflectivity, rhohv=None, zdr=None)

    processed, _quality, _advisory = preprocess_sweep(sweep, [])

    assert np.isfinite(processed.reflectivity[10, 10]), (
        "expected the center gate to survive: QC no longer removes the "
        "surrounding gates first, so speckle removal sees the whole 9-gate "
        "component (too large to be filtered) regardless of ordering"
    )
    assert processed.reflectivity[10, 10] == 34.0
    # The surrounding gates are classified biological (advisory-flaggable)
    # but their reflectivity is untouched too -- QC classifies without
    # deleting, everywhere, not just at the center gate.
    assert processed.reflectivity[9, 9] == 22.0
    from src.qc.classifier import CLASS_CODES
    from src.qc.parameters import GateClass

    assert processed.gate_classification[9, 9] == CLASS_CODES[GateClass.BIOLOGICAL]
