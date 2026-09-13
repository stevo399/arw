"""Lossless, compact storage for radar grids.

Level II moments are transmitted as integer codes with a fixed scale and
offset, so a decoded field sits exactly on a quantization step.  Storing the
code instead of the float is lossless -- but only if every value really is on
the step.  That is checked on every encode, never assumed: a grid that fails
the check is stored in its original dtype instead, and the fallback is
logged.  A surprise in a future Level II build can cost size, never accuracy.
"""

from dataclasses import dataclass
import logging
import zlib

import numpy as np

_logger = logging.getLogger(__name__)

ZLIB_LEVEL = 6
# NEXRAD Level II reflectivity: 0.5 dBZ steps from -32.0 dBZ.  Code 0 is
# reserved for no data (NaN), so code 1 is -32.0 dBZ and code 255 is 95.0 dBZ,
# which covers Level II's own range of -32.0 to 94.5 dBZ.
REFLECTIVITY_STEP_DBZ = 0.5
REFLECTIVITY_OFFSET_DBZ = -32.5


@dataclass(frozen=True)
class EncodedGrid:
    kind: str  # "stepped", "integer" or "raw"
    stored_dtype: str
    original_dtype: str
    shape: tuple[int, ...]
    payload: bytes
    step: float | None = None
    offset: float | None = None

    @property
    def nbytes(self) -> int:
        return len(self.payload)


def _compress(array: np.ndarray) -> bytes:
    return zlib.compress(np.ascontiguousarray(array).tobytes(), ZLIB_LEVEL)


def encode_raw(values: np.ndarray) -> EncodedGrid:
    """Store a grid exactly as it is, compressed."""
    array = np.asarray(values)
    return EncodedGrid(
        kind="raw",
        stored_dtype=array.dtype.str,
        original_dtype=array.dtype.str,
        shape=tuple(array.shape),
        payload=_compress(array),
    )


def encode_stepped(
    values: np.ndarray, *, step: float, offset: float, label: str
) -> EncodedGrid:
    """Store a float grid as uint8 codes when that is provably lossless."""
    array = np.asarray(values)
    if array.dtype.kind != "f":
        raise TypeError(f"{label}: stepped encoding needs a float grid, got {array.dtype}")
    finite = np.isfinite(array)
    finite_values = array[finite]
    codes = np.rint((finite_values - offset) / step)
    exact = (
        not np.any(np.isinf(array))
        and bool(np.all((codes >= 1) & (codes <= 255)))
        and np.array_equal((codes * step + offset).astype(array.dtype), finite_values)
    )
    if not exact:
        _logger.warning(
            "%s is not exactly representable on its %s step from %s; "
            "storing it in its original dtype %s",
            label, step, offset, array.dtype,
        )
        return encode_raw(array)
    stored = np.zeros(array.shape, dtype=np.uint8)
    stored[finite] = codes.astype(np.uint8)
    return EncodedGrid(
        kind="stepped",
        stored_dtype=stored.dtype.str,
        original_dtype=array.dtype.str,
        shape=tuple(array.shape),
        payload=_compress(stored),
        step=float(step),
        offset=float(offset),
    )


def encode_integer(values: np.ndarray, *, label: str) -> EncodedGrid:
    """Store a non-negative integer grid in the smallest unsigned dtype."""
    array = np.asarray(values)
    if array.dtype.kind not in "iu":
        raise TypeError(f"{label}: integer encoding needs an integer grid, got {array.dtype}")
    high = int(array.max()) if array.size else 0
    if array.size and int(array.min()) < 0:
        raise ValueError(f"{label}: negative values cannot be stored")
    stored_type = next(
        candidate
        for candidate in (np.uint8, np.uint16, np.uint32, np.uint64)
        if high <= np.iinfo(candidate).max
    )
    stored = array.astype(stored_type)
    return EncodedGrid(
        kind="integer",
        stored_dtype=stored.dtype.str,
        original_dtype=array.dtype.str,
        shape=tuple(array.shape),
        payload=_compress(stored),
    )


def decode(grid: EncodedGrid) -> np.ndarray:
    """Rebuild the exact original grid."""
    stored = np.frombuffer(
        zlib.decompress(grid.payload), dtype=np.dtype(grid.stored_dtype)
    ).reshape(grid.shape)
    original = np.dtype(grid.original_dtype)
    if grid.kind in ("raw", "integer"):
        return stored.astype(original, copy=True)
    if grid.kind == "stepped":
        decoded = np.full(grid.shape, np.nan, dtype=original)
        present = stored != 0
        decoded[present] = stored[present].astype(np.float64) * grid.step + grid.offset
        return decoded
    raise ValueError(f"unknown encoded grid kind: {grid.kind!r}")
