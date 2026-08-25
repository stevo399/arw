import numpy as np
import pytest

from src.qc.texture import local_standard_deviation


def test_smooth_field_has_low_texture():
    field = np.full((20, 20), 30.0)
    texture = local_standard_deviation(field)
    assert np.nanmax(texture) < 1e-6


def test_noisy_field_has_high_texture():
    rng = np.random.default_rng(0)
    field = rng.normal(loc=30.0, scale=15.0, size=(20, 20))
    texture = local_standard_deviation(field)
    assert np.nanmean(texture) > 5.0


def test_gradient_field_has_moderate_texture():
    field = np.tile(np.arange(20, dtype=float), (20, 1))
    texture = local_standard_deviation(field)
    assert 0.5 < np.nanmean(texture) < 2.0


def test_nan_gates_stay_nan():
    field = np.full((10, 10), 30.0)
    field[5, 5] = np.nan
    texture = local_standard_deviation(field)
    assert np.isnan(texture[5, 5])


def test_all_nan_field_returns_all_nan():
    field = np.full((10, 10), np.nan)
    texture = local_standard_deviation(field)
    assert np.all(np.isnan(texture))


def test_nan_neighbour_is_not_biased_by_the_nan():
    """A neighbour of a lone NaN must not see it as a zero.

    In a constant field, excluding the NaN leaves every value in the window
    identical, so texture stays ~0. If the NaN were folded in as 0.0 the
    window would span 0 to 30 and the deviation would be large.
    """
    field = np.full((7, 7), 30.0)
    field[3, 3] = np.nan
    texture = local_standard_deviation(field)
    assert texture[2, 3] == pytest.approx(0.0, abs=1e-9)
    assert texture[3, 2] == pytest.approx(0.0, abs=1e-9)
    assert texture[4, 4] == pytest.approx(0.0, abs=1e-9)
