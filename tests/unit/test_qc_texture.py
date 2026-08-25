import numpy as np

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
