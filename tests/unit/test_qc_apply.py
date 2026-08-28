import numpy as np
import pytest

from src.parser import SweepData
from src.qc.apply import (
    DEGRADED_LOW_CONFIDENCE,
    DEGRADED_NO_DUAL_POL,
    apply_quality_control,
)
from src.qc.parameters import GateClass


def _sweep(**overrides) -> SweepData:
    shape = (36, 40)
    base = dict(
        reflectivity=np.full(shape, 30.0),
        rhohv=np.full(shape, 0.99),
        zdr=np.full(shape, 0.5),
        azimuths=np.linspace(0.0, 350.0, shape[0]),
        ranges_m=np.arange(2125.0, 2125.0 + 250.0 * shape[1], 250.0),
        elevation_angle=0.5,
        elevations=np.full(shape[0], 0.5),
        elevation_angles=[0.5],
        radar_lat=35.3331,
        radar_lon=-97.2778,
        radar_alt_m=390.0,
        timestamp="2026-08-23T00:00:00Z",
    )
    base.update(overrides)
    return SweepData(**base)


def test_clean_precipitation_survives_quality_control():
    filtered, advisory, report = apply_quality_control(_sweep(), [])
    assert np.isfinite(filtered.reflectivity).all()
    assert not advisory.mask.any()
    assert report.advisory_fraction == 0.0


def test_reflectivity_is_never_modified_by_quality_control():
    """2026-08-26 amendment: quality control classifies but never deletes.

    Formerly `test_clutter_is_removed_from_the_filtered_field`, which
    asserted the opposite of the current contract (that flagged gates became
    NaN in the filtered field). That behavior no longer exists by design --
    Proof 4 found protection was the only thing standing between the
    classifier and silently deleting real tornado debris, so nothing is
    removed from reflectivity any more, ever. This test asserts the new
    invariant directly: `apply_quality_control`'s output reflectivity is
    byte-for-byte identical to the input, even on a field with heavy,
    confidently-classified clutter.
    """
    rng = np.random.default_rng(3)
    shape = (36, 40)
    original_reflectivity = np.clip(rng.normal(35.0, 12.0, shape), None, 45.0)
    sweep = _sweep(
        # Clipped below PROTECTED_MIN_DBZ (50.0): this fixture has no NaN
        # gaps anywhere, so protected_mask's connected-component rule 3
        # treats the whole 36x40 grid as a single storm body. An unclipped
        # sample above 50 dBZ (near-certain across 1440 draws) would
        # therefore protect the entire grid rather than just clutter-like
        # gates, defeating the point of this test. See task-12-report.md.
        reflectivity=original_reflectivity.copy(),
        rhohv=np.full(shape, 0.6),
        zdr=rng.normal(0.0, 3.0, shape),
    )
    filtered, advisory, report = apply_quality_control(
        sweep, [], velocity=np.zeros(shape)
    )
    # The advisory mechanism still works -- gates are still classified and
    # would still be flagged -- but nothing is actually removed.
    assert advisory.mask.any()
    assert report.advisory_fraction > 0.0
    assert np.array_equal(filtered.reflectivity, original_reflectivity)
    assert np.isfinite(filtered.reflectivity).all()


def test_flagged_gates_retain_reasons():
    rng = np.random.default_rng(4)
    shape = (36, 40)
    sweep = _sweep(
        # See clipping note in test_reflectivity_is_never_modified_by_quality_control.
        reflectivity=np.clip(rng.normal(35.0, 12.0, shape), None, 45.0),
        rhohv=np.full(shape, 0.6),
        zdr=rng.normal(0.0, 3.0, shape),
    )
    _filtered, advisory, _report = apply_quality_control(
        sweep, [], velocity=np.zeros(shape)
    )
    reasons = set(advisory.reasons[advisory.mask].tolist())
    assert reasons
    assert GateClass.PRECIPITATION not in reasons


def test_protected_gates_are_never_flagged():
    """Protection no longer prevents deletion (nothing is ever deleted), but
    it still exempts gates from the advisory `flagged` mask -- this is what
    the clutter-persistence and severe-case proofs rely on to compute a
    meaningful advisory signal. The finite-reflectivity assertion is now a
    trivial invariant (true for every gate regardless of protection, per
    test_reflectivity_is_never_modified_by_quality_control) and is kept only
    to document that fact explicitly at the protected gate too.
    """
    shape = (36, 40)
    reflectivity = np.full(shape, 20.0)
    reflectivity[10, 10] = 62.0
    sweep = _sweep(reflectivity=reflectivity, rhohv=np.full(shape, 0.5))
    filtered, advisory, _report = apply_quality_control(
        sweep, [], velocity=np.zeros(shape)
    )
    assert not advisory.mask[10, 10]
    assert np.isfinite(filtered.reflectivity[10, 10])


def test_hail_gate_is_never_flagged():
    """Mutation-gap test: with NON_METEOROLOGICAL_CLASSES = (ground_clutter,
    biological) this must hold. If hail were ever added to that tuple, this
    is the only test in the suite that would catch it -- none of the other
    fixtures produce a classifier-confirmed hail gate that is not already
    shielded by protected_mask's own >=50 dBZ rule, so they cannot
    distinguish "hail is protected via rule 1" from "hail is simply never a
    flagging candidate." This gate sits at 49 dBZ, deliberately below
    PROTECTED_MIN_DBZ (50.0), so protected_mask contributes nothing here and
    only NON_METEOROLOGICAL_CLASSES decides the outcome. (Since the
    2026-08-26 amendment, no gate of any class is ever actually removed --
    this test is about the advisory mask, not deletion.)
    """
    shape = (36, 40)
    reflectivity = np.full(shape, 20.0)
    reflectivity[4:8, 4:8] = 49.0
    zdr = np.full(shape, 0.5)
    zdr[4:8, 4:8] = 0.0
    rhohv = np.full(shape, 0.99)
    rhohv[4:8, 4:8] = 0.9
    sweep = _sweep(reflectivity=reflectivity, zdr=zdr, rhohv=rhohv)

    filtered, advisory, _report = apply_quality_control(
        sweep, [], velocity=np.zeros(shape)
    )

    from src.qc.classifier import CLASS_CODES

    assert filtered.gate_classification[5, 5] == CLASS_CODES[GateClass.HAIL]
    assert not advisory.mask[5, 5]
    assert np.isfinite(filtered.reflectivity[5, 5])


def test_debris_gate_is_never_flagged():
    """Mirrors test_hail_gate_is_never_flagged. Debris is the tornado
    signature -- the single most important class that must never be
    condemned by quality control. A grid search over classify_gates confirmed
    a 4x4 block at reflectivity 49.0 dBZ / zdr 0.0 / rhohv 0.7 is classified
    'debris' with confidence 0.85, and is NOT protected (max reflectivity in
    the whole fixture is 49 dBZ, deliberately below PROTECTED_MIN_DBZ =
    50.0, and there is no rotation signature), so protected_mask contributes
    nothing here and only NON_METEOROLOGICAL_CLASSES decides the outcome.
    (Since the 2026-08-26 amendment, no gate of any class is ever actually
    removed -- this test is about the advisory mask, not deletion.)
    """
    shape = (36, 40)
    reflectivity = np.full(shape, 20.0)
    reflectivity[4:8, 4:8] = 49.0
    zdr = np.full(shape, 0.5)
    zdr[4:8, 4:8] = 0.0
    rhohv = np.full(shape, 0.99)
    rhohv[4:8, 4:8] = 0.7
    sweep = _sweep(reflectivity=reflectivity, zdr=zdr, rhohv=rhohv)

    filtered, advisory, _report = apply_quality_control(
        sweep, [], velocity=np.zeros(shape)
    )

    from src.qc.classifier import CLASS_CODES

    assert filtered.gate_classification[5, 5] == CLASS_CODES[GateClass.DEBRIS]
    assert not advisory.mask[5, 5]
    assert np.isfinite(filtered.reflectivity[5, 5])


def test_missing_dual_pol_reports_degraded_mode():
    _filtered, _rejected, report = apply_quality_control(
        _sweep(rhohv=None, zdr=None), []
    )
    assert DEGRADED_NO_DUAL_POL in report.degraded_modes


def test_low_confidence_reports_degraded_mode():
    """DEGRADED_LOW_CONFIDENCE must be reachable, not dead code.

    Found by a real end-to-end search (not a synthetic freeform texture
    value): a checkerboard reflectivity field with dual-pol AND velocity
    both present drove classify_gates's mean confidence down to 0.3125,
    below LOW_CONFIDENCE_THRESHOLD (0.35). Every class's memberships are
    weak here at once -- reflectivity is negative and noisy (texture from
    the -15/-7 dBZ checkerboard), rhohv (0.89) sits in the ambiguous band
    between clutter and precipitation, zdr (-7.0) is outside every class's
    membership window, and velocity (5.0 m/s) doesn't strongly indicate
    clutter either. See task-12-report.md for the search that found this.
    """
    shape = (36, 40)
    reflectivity = np.empty(shape)
    reflectivity[:, 0::2] = -15.0
    reflectivity[:, 1::2] = -7.0
    sweep = _sweep(
        reflectivity=reflectivity,
        rhohv=np.full(shape, 0.89),
        zdr=np.full(shape, -7.0),
    )
    _filtered, _rejected, report = apply_quality_control(
        sweep, [], velocity=np.full(shape, 5.0)
    )
    assert report.mean_confidence < 0.35
    assert DEGRADED_LOW_CONFIDENCE in report.degraded_modes


def test_gates_beyond_velocity_range_report_degraded_mode():
    """The surveillance cut reaches 460 km but Doppler cuts only 300 km, so
    debris protection is unavailable in the outer ring. That limit is a
    property of the radar and must be reported, not silently absorbed."""
    from src.qc.apply import DEGRADED_BEYOND_VELOCITY_RANGE

    long_ranges = np.arange(2125.0, 460_000.0, 250.0)
    shape = (36, len(long_ranges))
    sweep = _sweep(
        reflectivity=np.full(shape, 30.0),
        rhohv=np.full(shape, 0.99),
        zdr=np.full(shape, 0.5),
        ranges_m=long_ranges,
    )
    _filtered, _rejected, report = apply_quality_control(
        sweep, [], velocity=np.zeros(shape)
    )
    assert DEGRADED_BEYOND_VELOCITY_RANGE in report.degraded_modes


def test_short_range_sweep_does_not_report_range_degradation():
    from src.qc.apply import DEGRADED_BEYOND_VELOCITY_RANGE

    _filtered, _rejected, report = apply_quality_control(
        _sweep(), [], velocity=np.zeros((36, 40))
    )
    assert DEGRADED_BEYOND_VELOCITY_RANGE not in report.degraded_modes


def test_class_fractions_sum_to_one():
    _filtered, _rejected, report = apply_quality_control(_sweep(), [])
    assert sum(report.class_fractions.values()) == pytest.approx(1.0, abs=1e-6)


def test_gate_classification_is_attached_to_filtered_sweep():
    filtered, _rejected, _report = apply_quality_control(_sweep(), [])
    assert filtered.gate_classification is not None
    assert filtered.gate_classification.dtype == np.int8
