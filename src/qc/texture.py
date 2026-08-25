"""Local texture measures.

Meteorological echo varies smoothly from gate to gate. Ground clutter,
biological scatterers and interference do not. Local standard deviation
captures that difference and requires no polarimetric data, so it remains
available on pre-2013 volumes.
"""

import numpy as np
from scipy.ndimage import uniform_filter


def local_standard_deviation(field: np.ndarray, window: int = 3) -> np.ndarray:
    """Standard deviation of a field within a square window around each gate.

    NaN gates are excluded from their neighbours' statistics and stay NaN in
    the output.
    """
    values = np.asarray(field, dtype=float)
    valid = np.isfinite(values)
    filled = np.where(valid, values, 0.0)

    count = uniform_filter(valid.astype(float), size=window, mode="nearest")
    mean = uniform_filter(filled, size=window, mode="nearest")
    mean_square = uniform_filter(filled**2, size=window, mode="nearest")

    with np.errstate(invalid="ignore", divide="ignore"):
        true_mean = mean / count
        true_mean_square = mean_square / count
        variance = np.clip(true_mean_square - true_mean**2, 0.0, None)
        deviation = np.sqrt(variance)

    return np.where(valid & (count > 0), deviation, np.nan)
