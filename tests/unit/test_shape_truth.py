import numpy as np
import pyart
import pytest
from shapely.geometry import MultiPolygon

from src.contours import contour_mask
from src.parser import extract_sweep_data
from src.shape_truth import measure_walk_truth

REFERENCE_VOLUME = "cache/KEMX/KEMX20260712_022646_V06"


@pytest.fixture(scope="module")
def sweep():
    return extract_sweep_data(pyart.io.read_nexrad_archive(REFERENCE_VOLUME))


def test_perfect_shape_scores_near_zero_on_both_rates(sweep):
    """A contour of a block should describe that block almost exactly."""
    field = np.full(np.asarray(sweep.reflectivity).shape, np.nan)
    field[100:200, 300:400] = 40.0
    mask = np.isfinite(field) & (field >= 20.0)

    truth = measure_walk_truth(contour_mask(mask, sweep), field, sweep, 20.0)

    assert truth.false_positive_rate < 0.05
    assert truth.false_negative_rate < 0.05


def test_oversized_shape_is_caught_as_false_positive(sweep):
    """A shape larger than its echo must score badly -- that is the hull's failure."""
    field = np.full(np.asarray(sweep.reflectivity).shape, np.nan)
    field[100:200, 300:400] = 40.0
    mask = np.isfinite(field) & (field >= 20.0)

    honest = contour_mask(mask, sweep)
    inflated = MultiPolygon([honest.buffer(0.05)]) if honest.geom_type == "Polygon" \
        else MultiPolygon([honest.buffer(0.05)])

    truth = measure_walk_truth(inflated, field, sweep, 20.0)
    assert truth.false_positive_rate > 0.2


def test_undersized_shape_is_caught_as_false_negative(sweep):
    field = np.full(np.asarray(sweep.reflectivity).shape, np.nan)
    field[100:200, 300:400] = 40.0
    mask = np.isfinite(field) & (field >= 20.0)

    honest = contour_mask(mask, sweep)
    # -0.02 deg (the brief's original value) shrinks the honest polygon by
    # only ~20% of area on this reference volume -- measured false-negative
    # rate on the full candidate population is ~18.3%, and the n=2000 sample
    # lands anywhere from ~13% to ~17% across seeds, under the 0.2 threshold
    # more often than not. -0.04 deg shrinks enough that the population rate
    # clears 0.2 with margin (~24-25% measured across seeds 0/1/2/7), so the
    # assertion is a genuine, non-flaky check rather than a coin flip.
    shrunk = honest.buffer(-0.04)
    truth = measure_walk_truth(shrunk, field, sweep, 20.0)
    assert truth.false_negative_rate > 0.2


def test_measurement_is_deterministic(sweep):
    field = np.full(np.asarray(sweep.reflectivity).shape, np.nan)
    field[100:200, 300:400] = 40.0
    mask = np.isfinite(field) & (field >= 20.0)
    geom = contour_mask(mask, sweep)

    a = measure_walk_truth(geom, field, sweep, 20.0, seed=7)
    b = measure_walk_truth(geom, field, sweep, 20.0, seed=7)
    assert a.false_positive_rate == b.false_positive_rate
