"""Measure whether a shape lies when walked on.

The project owner explores Audiom by walking across a feature's whole surface
at a step size of his choosing. So the question that matters is not whether a
polygon looks right, but whether standing at a point inside it means there is
genuinely weather at that point.

  false positive: inside the shape, no echo present   -- a lie walked into
  false negative: echo present, outside the shape     -- weather not reported
"""

from dataclasses import dataclass

import numpy as np
from shapely.geometry import Point
from shapely.prepared import prep

from src.geometry import gate_coordinates


@dataclass
class WalkTruth:
    false_positive_rate: float
    false_negative_rate: float
    n_inside: int
    n_outside: int


def measure_walk_truth(geom, field, sweep, level: float, n_samples: int = 2000, seed: int = 0):
    """Sample gates and compare shape membership against real echo.

    Sampling gates rather than uniform points keeps the measurement in the
    radar's own frame: every sample is somewhere the radar actually looked.
    """
    field = np.asarray(field, dtype=float)
    has_echo = np.isfinite(field) & (field >= level)

    lat, lon = gate_coordinates(
        sweep.azimuths, sweep.ranges_m, sweep.elevations,
        sweep.radar_lat, sweep.radar_lon,
    )

    rng = np.random.default_rng(seed)
    minx, miny, maxx, maxy = geom.bounds if not geom.is_empty else (0, 0, 0, 0)
    within_bbox = (lon >= minx) & (lon <= maxx) & (lat >= miny) & (lat <= maxy)
    candidates = np.flatnonzero(within_bbox.ravel())
    if candidates.size == 0:
        return WalkTruth(0.0, 0.0, 0, 0)

    chosen = rng.choice(candidates, size=min(n_samples, candidates.size), replace=False)
    flat_lon = lon.ravel()[chosen]
    flat_lat = lat.ravel()[chosen]
    flat_echo = has_echo.ravel()[chosen]

    prepared = prep(geom)
    inside = np.array([prepared.contains(Point(x, y)) for x, y in zip(flat_lon, flat_lat)])

    n_inside = int(inside.sum())
    n_outside = int((~inside).sum())
    false_positive = int((inside & ~flat_echo).sum())
    false_negative = int((~inside & flat_echo).sum())

    return WalkTruth(
        false_positive_rate=false_positive / n_inside if n_inside else 0.0,
        false_negative_rate=false_negative / n_outside if n_outside else 0.0,
        n_inside=n_inside,
        n_outside=n_outside,
    )
