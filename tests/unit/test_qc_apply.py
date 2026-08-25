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
    filtered, rejected, report = apply_quality_control(_sweep(), [])
    assert np.isfinite(filtered.reflectivity).all()
    assert not rejected.mask.any()
    assert report.rejected_fraction == 0.0


def test_clutter_is_removed_from_the_filtered_field():
    rng = np.random.default_rng(3)
    shape = (36, 40)
    sweep = _sweep(
        # Clipped below PROTECTED_MIN_DBZ (50.0): this fixture has no NaN
        # gaps anywhere, so protected_mask's connected-component rule 3
        # treats the whole 36x40 grid as a single storm body. An unclipped
        # sample above 50 dBZ (near-certain across 1440 draws) would
        # therefore protect the entire grid rather than just clutter-like
        # gates, defeating the point of this test. See task-12-report.md.
        reflectivity=np.clip(rng.normal(35.0, 12.0, shape), None, 45.0),
        rhohv=np.full(shape, 0.6),
        zdr=rng.normal(0.0, 3.0, shape),
    )
    filtered, rejected, report = apply_quality_control(
        sweep, [], velocity=np.zeros(shape)
    )
    assert rejected.mask.any()
    assert report.rejected_fraction > 0.0
    assert np.isnan(filtered.reflectivity[rejected.mask]).all()


def test_rejected_gates_retain_reasons():
    rng = np.random.default_rng(4)
    shape = (36, 40)
    sweep = _sweep(
        # See clipping note in test_clutter_is_removed_from_the_filtered_field.
        reflectivity=np.clip(rng.normal(35.0, 12.0, shape), None, 45.0),
        rhohv=np.full(shape, 0.6),
        zdr=rng.normal(0.0, 3.0, shape),
    )
    _filtered, rejected, _report = apply_quality_control(
        sweep, [], velocity=np.zeros(shape)
    )
    reasons = set(rejected.reasons[rejected.mask].tolist())
    assert reasons
    assert GateClass.PRECIPITATION not in reasons


def test_protected_gates_are_never_rejected():
    shape = (36, 40)
    reflectivity = np.full(shape, 20.0)
    reflectivity[10, 10] = 62.0
    sweep = _sweep(reflectivity=reflectivity, rhohv=np.full(shape, 0.5))
    filtered, rejected, _report = apply_quality_control(
        sweep, [], velocity=np.zeros(shape)
    )
    assert not rejected.mask[10, 10]
    assert np.isfinite(filtered.reflectivity[10, 10])


def test_hail_gate_is_never_rejected():
    """Mutation-gap test: with NON_METEOROLOGICAL_CLASSES = (ground_clutter,
    biological) this must hold. If hail were ever added to that tuple, this
    is the only test in the suite that would catch it -- none of the other
    fixtures produce a classifier-confirmed hail gate that is not already
    shielded by protected_mask's own >=50 dBZ rule, so they cannot
    distinguish "hail is protected via rule 1" from "hail is simply never a
    rejection candidate." This gate sits at 49 dBZ, deliberately below
    PROTECTED_MIN_DBZ (50.0), so protected_mask contributes nothing here and
    only NON_METEOROLOGICAL_CLASSES decides the outcome.
    """
    shape = (36, 40)
    reflectivity = np.full(shape, 20.0)
    reflectivity[4:8, 4:8] = 49.0
    zdr = np.full(shape, 0.5)
    zdr[4:8, 4:8] = 0.0
    rhohv = np.full(shape, 0.99)
    rhohv[4:8, 4:8] = 0.9
    sweep = _sweep(reflectivity=reflectivity, zdr=zdr, rhohv=rhohv)

    filtered, rejected, _report = apply_quality_control(
        sweep, [], velocity=np.zeros(shape)
    )

    from src.qc.classifier import CLASS_CODES

    assert filtered.gate_classification[5, 5] == CLASS_CODES[GateClass.HAIL]
    assert not rejected.mask[5, 5]
    assert np.isfinite(filtered.reflectivity[5, 5])


def test_missing_dual_pol_reports_degraded_mode():
    _filtered, _rejected, report = apply_quality_control(
        _sweep(rhohv=None, zdr=None), []
    )
    assert DEGRADED_NO_DUAL_POL in report.degraded_modes


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
