import numpy as np
import pytest

from src.qc.membership import trapezoid
from src.qc.parameters import CLASS_PARAMETERS, GateClass


def test_trapezoid_plateau_is_one():
    assert trapezoid(np.array([5.0]), 0.0, 2.0, 8.0, 10.0)[0] == pytest.approx(1.0)


def test_trapezoid_outside_support_is_zero():
    values = np.array([-1.0, 11.0])
    assert np.allclose(trapezoid(values, 0.0, 2.0, 8.0, 10.0), 0.0)


def test_trapezoid_ramps_linearly():
    assert trapezoid(np.array([1.0]), 0.0, 2.0, 8.0, 10.0)[0] == pytest.approx(0.5)
    assert trapezoid(np.array([9.0]), 0.0, 2.0, 8.0, 10.0)[0] == pytest.approx(0.5)


def test_trapezoid_handles_nan():
    assert np.isnan(trapezoid(np.array([np.nan]), 0.0, 2.0, 8.0, 10.0)[0])


def test_trapezoid_supports_open_left_edge():
    """x1 == x2 gives a hard left edge rather than a ramp."""
    assert trapezoid(np.array([0.0]), 0.0, 0.0, 8.0, 10.0)[0] == pytest.approx(1.0)


def test_every_parameter_has_provenance():
    """Global constraint: no membership parameter may lack a provenance marker."""
    for class_name, variables in CLASS_PARAMETERS.items():
        for variable_name, parameter in variables.items():
            assert parameter.provenance, f"{class_name}.{variable_name} lacks provenance"


def test_all_expected_classes_present():
    assert set(CLASS_PARAMETERS) == {
        GateClass.PRECIPITATION,
        GateClass.GROUND_CLUTTER,
        GateClass.BIOLOGICAL,
        GateClass.HAIL,
        GateClass.DEBRIS,
    }


def test_melting_layer_classes_are_absent():
    """Classes needing melting-layer height must not be present — spec 8."""
    forbidden = {"wet_snow", "dry_snow", "ice_crystals", "graupel", "big_drops"}
    assert forbidden.isdisjoint(CLASS_PARAMETERS)
