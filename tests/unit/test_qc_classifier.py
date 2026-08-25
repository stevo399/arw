import numpy as np
import pytest

from src.parser import SweepData
from src.qc.classifier import CLASS_CODES, classify_gates
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


def test_uniform_rain_classifies_as_precipitation():
    result = classify_gates(_sweep())
    assert result.classes[10, 10] == CLASS_CODES[GateClass.PRECIPITATION]


def test_low_rhohv_stationary_noisy_classifies_as_ground_clutter():
    rng = np.random.default_rng(1)
    shape = (36, 40)
    sweep = _sweep(
        reflectivity=rng.normal(35.0, 12.0, shape),
        rhohv=np.full(shape, 0.6),
        zdr=rng.normal(0.0, 3.0, shape),
    )
    result = classify_gates(sweep, velocity=np.zeros(shape))
    clutter = CLASS_CODES[GateClass.GROUND_CLUTTER]
    assert (result.classes == clutter).mean() > 0.5


def test_high_zdr_weak_echo_classifies_as_biological():
    rng = np.random.default_rng(2)
    shape = (36, 40)
    sweep = _sweep(
        reflectivity=rng.normal(15.0, 8.0, shape),
        rhohv=np.full(shape, 0.6),
        zdr=np.full(shape, 6.0),
    )
    result = classify_gates(sweep, velocity=np.full(shape, 12.0))
    biological = CLASS_CODES[GateClass.BIOLOGICAL]
    assert (result.classes == biological).mean() > 0.4


def test_intense_low_zdr_core_classifies_as_hail():
    shape = (36, 40)
    sweep = _sweep(
        reflectivity=np.full(shape, 62.0),
        rhohv=np.full(shape, 0.91),
        zdr=np.full(shape, 0.0),
    )
    result = classify_gates(sweep)
    assert result.classes[10, 10] == CLASS_CODES[GateClass.HAIL]


def test_nan_gates_classify_as_unknown():
    shape = (36, 40)
    sweep = _sweep(reflectivity=np.full(shape, np.nan))
    result = classify_gates(sweep)
    assert (result.classes == CLASS_CODES[GateClass.UNKNOWN]).all()


def test_absent_polarimetric_fields_are_omitted_not_substituted():
    """Absent variables must be dropped from scoring, not read as zero.

    A uniform 45 dBZ field with no dual-pol available is precipitation. If the
    missing RhoHV and ZDR were substituted with 0.0 instead of omitted, the same
    field classifies as 100% tornado debris - a low RhoHV and a zero ZDR are
    exactly the debris signature. Pre-2013 volumes have no dual-pol at all, so
    this is the guard that stops an entire historical scan being mislabelled.
    """
    shape = (36, 40)
    reflectivity = np.full(shape, 45.0)

    omitted = classify_gates(_sweep(reflectivity=reflectivity, rhohv=None, zdr=None))
    substituted = classify_gates(
        _sweep(reflectivity=reflectivity, rhohv=np.zeros(shape), zdr=np.zeros(shape))
    )

    precipitation = CLASS_CODES[GateClass.PRECIPITATION]
    assert (omitted.classes == precipitation).mean() > 0.9
    assert (substituted.classes == precipitation).mean() < 0.1


def test_classes_array_is_int8():
    result = classify_gates(_sweep())
    assert result.classes.dtype == np.int8


def test_confidence_between_zero_and_one():
    result = classify_gates(_sweep())
    finite = result.confidence[np.isfinite(result.confidence)]
    assert finite.min() >= 0.0
    assert finite.max() <= 1.0
