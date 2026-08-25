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


def test_missing_polarimetric_fields_do_not_classify_everything_as_clutter():
    """Global constraint: absent is not low. A pre-2013 scan must not become
    wall-to-wall ground clutter."""
    shape = (36, 40)
    sweep = _sweep(rhohv=None, zdr=None)
    result = classify_gates(sweep)
    clutter = CLASS_CODES[GateClass.GROUND_CLUTTER]
    assert (result.classes == clutter).mean() < 0.2


def test_classes_array_is_int8():
    result = classify_gates(_sweep())
    assert result.classes.dtype == np.int8


def test_confidence_between_zero_and_one():
    result = classify_gates(_sweep())
    finite = result.confidence[np.isfinite(result.confidence)]
    assert finite.min() >= 0.0
    assert finite.max() <= 1.0
