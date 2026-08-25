"""Trapezoidal fuzzy membership functions.

Each function rises from 0 at x1 to 1 at x2, holds 1 through x3, and falls to
0 at x4. Setting x1 == x2 or x3 == x4 produces a hard edge instead of a ramp.
"""

import numpy as np


def trapezoid(
    values: np.ndarray,
    x1: float,
    x2: float,
    x3: float,
    x4: float,
) -> np.ndarray:
    """Membership of each value in a trapezoidal set. NaN in, NaN out."""
    v = np.asarray(values, dtype=float)
    result = np.zeros_like(v, dtype=float)

    rising = (v > x1) & (v < x2)
    if x2 > x1:
        result[rising] = (v[rising] - x1) / (x2 - x1)

    result[(v >= x2) & (v <= x3)] = 1.0

    falling = (v > x3) & (v < x4)
    if x4 > x3:
        result[falling] = (x4 - v[falling]) / (x4 - x3)

    return np.where(np.isfinite(v), result, np.nan)
