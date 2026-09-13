import logging

import numpy as np
import pytest

from src.history.codec import (
    REFLECTIVITY_OFFSET_DBZ,
    REFLECTIVITY_STEP_DBZ,
    decode,
    encode_integer,
    encode_raw,
    encode_stepped,
)


def _reflectivity_like(shape=(360, 500), seed=7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    codes = rng.integers(1, 256, size=shape)
    values = codes * REFLECTIVITY_STEP_DBZ + REFLECTIVITY_OFFSET_DBZ
    values = values.astype(np.float64)
    values[rng.random(shape) < 0.7] = np.nan
    return values


def _encode_reflectivity(values):
    return encode_stepped(
        values, step=REFLECTIVITY_STEP_DBZ, offset=REFLECTIVITY_OFFSET_DBZ, label="test reflectivity"
    )


def test_stepped_field_round_trips_bit_identically():
    values = _reflectivity_like()
    grid = _encode_reflectivity(values)
    decoded = decode(grid)
    assert grid.kind == "stepped"
    assert decoded.dtype == values.dtype
    assert np.array_equal(decoded, values, equal_nan=True)


def test_stepped_field_preserves_float32_dtype():
    values = _reflectivity_like().astype(np.float32)
    decoded = decode(_encode_reflectivity(values))
    assert decoded.dtype == np.float32
    assert np.array_equal(decoded, values, equal_nan=True)


def test_stepped_extremes_are_exact():
    values = np.array([[-32.0, 94.5, np.nan, 0.0]])
    assert np.array_equal(decode(_encode_reflectivity(values)), values, equal_nan=True)


@pytest.mark.parametrize(
    "bad_value",
    [20.25, -32.5, 95.5, np.inf, -np.inf],
    ids=["off-step", "below-range", "above-range", "inf", "-inf"],
)
def test_unrepresentable_values_fall_back_to_original_dtype(caplog, bad_value):
    values = _reflectivity_like()
    values[0, 0] = bad_value
    with caplog.at_level(logging.WARNING, logger="src.history.codec"):
        grid = _encode_reflectivity(values)
    assert grid.kind == "raw"
    assert "test reflectivity" in caplog.text
    assert np.array_equal(decode(grid), values, equal_nan=True)


def test_all_nan_field_is_stepped_and_exact():
    values = np.full((10, 10), np.nan)
    grid = _encode_reflectivity(values)
    assert grid.kind == "stepped"
    assert np.array_equal(decode(grid), values, equal_nan=True)


def test_stepped_rejects_integer_grids():
    with pytest.raises(TypeError):
        encode_stepped(np.zeros((2, 2), dtype=np.int32), step=0.5, offset=0.0, label="x")


def test_integer_grid_uses_smallest_unsigned_storage_and_restores_dtype():
    labels = np.zeros((720, 1832), dtype=np.int32)
    labels[100:110, 200:210] = 63
    grid = encode_integer(labels, label="labels")
    assert np.dtype(grid.stored_dtype) == np.uint8
    decoded = decode(grid)
    assert decoded.dtype == np.int32
    assert np.array_equal(decoded, labels)


def test_integer_grid_widens_for_large_labels():
    labels = np.zeros((4, 4), dtype=np.int64)
    labels[0, 0] = 70_000
    grid = encode_integer(labels, label="labels")
    assert np.dtype(grid.stored_dtype) == np.uint32
    assert np.array_equal(decode(grid), labels)


def test_integer_grid_rejects_negative_labels():
    with pytest.raises(ValueError):
        encode_integer(np.array([[-1, 0]], dtype=np.int32), label="labels")


def test_raw_grid_round_trips():
    azimuths = np.linspace(0.0, 359.5, 720)
    grid = encode_raw(azimuths)
    assert np.array_equal(decode(grid), azimuths)


def test_sparse_reflectivity_compresses_far_below_float64():
    values = np.full((720, 1832), np.nan)
    values[100:140, 300:400] = 45.0
    grid = _encode_reflectivity(values)
    assert grid.nbytes < values.nbytes / 100
