"""Measure whether a shape lies when walked on.

The project owner explores Audiom by walking across a feature's whole surface
at a step size of his choosing. So the question that matters is not whether a
polygon looks right, but whether standing at a point inside it means there is
genuinely weather at that point.

  false positive: inside the shape, no echo present   -- a lie walked into
  false negative: echo present, outside the shape     -- weather not reported

Rates are AREA-WEIGHTED, not sample-count-weighted. Radar gates are small and
dense near the radar and large and sparse far out, so a plain fraction of
sampled gates under-weights the far field -- precisely where a convex hull's
excess is proportionally largest, since hull excess grows with distance from
the storm's centroid. The person walking measures distance in ground metres,
not in gate-index counts, so each sampled gate is weighted by its own ground
footprint (`src.geometry.gate_areas_km2`) before the rates are computed. Every
sample is still a real gate the radar looked at -- only the weighting of that
sample changes, not the sampling frame itself.
"""

from dataclasses import dataclass

import numpy as np
from shapely.geometry import Point
from shapely.prepared import prep

from src.geometry import gate_areas_km2, gate_coordinates


@dataclass
class WalkTruth:
    false_positive_rate: float
    false_negative_rate: float
    n_inside: int
    n_outside: int
    # Area-weighted totals (sq km) backing the two rates above, so a reader
    # can see the raw sample counts and the ground area they represent side
    # by side. Default 0.0 keeps the dataclass's original four-field
    # constructor call sites (if any exist) working unchanged.
    area_inside_km2: float = 0.0
    area_outside_km2: float = 0.0


def measure_walk_truth(geom, field, sweep, level: float, n_samples: int = 2000, seed: int = 0):
    """Sample gates and compare shape membership against real echo.

    Sampling gates rather than uniform points keeps the measurement in the
    radar's own frame: every sample is somewhere the radar actually looked.
    Each sample is weighted by its own gate's ground area so the resulting
    rates describe the fraction of *ground* that lies, not the fraction of
    *radar samples* that lie -- see the module docstring for why those two
    differ, and by how much, on a convex hull.

    A gate carrying echo that belongs to a DIFFERENT detected object than the
    one `geom` represents is still not a false positive: there genuinely is
    weather at that point, so a person walking there is not being lied to.
    This function deliberately tests echo presence only (`field >= level`),
    never object identity -- keep it that way. Switching this to an
    object-membership test would misclassify real, correctly-reported
    neighbouring-storm echo as a lie.
    """
    field = np.asarray(field, dtype=float)
    has_echo = np.isfinite(field) & (field >= level)

    lat, lon = gate_coordinates(
        sweep.azimuths, sweep.ranges_m, sweep.elevations,
        sweep.radar_lat, sweep.radar_lon,
    )
    # One area per range bin (column); every gate in a given range ring has
    # the same ground footprint regardless of azimuth, so this broadcasts
    # across rays the same way `gate_coordinates` broadcasts range across them.
    range_bin_areas = gate_areas_km2(sweep.azimuths, sweep.ranges_m, sweep.elevation_angle)
    gate_area = np.broadcast_to(range_bin_areas[None, :], lat.shape)

    rng = np.random.default_rng(seed)
    minx, miny, maxx, maxy = geom.bounds if not geom.is_empty else (0, 0, 0, 0)
    within_bbox = (lon >= minx) & (lon <= maxx) & (lat >= miny) & (lat <= maxy)
    candidates = np.flatnonzero(within_bbox.ravel())
    if candidates.size == 0:
        return WalkTruth(0.0, 0.0, 0, 0, 0.0, 0.0)

    chosen = rng.choice(candidates, size=min(n_samples, candidates.size), replace=False)
    flat_lon = lon.ravel()[chosen]
    flat_lat = lat.ravel()[chosen]
    flat_echo = has_echo.ravel()[chosen]
    flat_area = gate_area.ravel()[chosen]

    prepared = prep(geom)
    inside = np.array([prepared.contains(Point(x, y)) for x, y in zip(flat_lon, flat_lat)])

    n_inside = int(inside.sum())
    n_outside = int((~inside).sum())

    area_inside = float(flat_area[inside].sum())
    area_outside = float(flat_area[~inside].sum())
    false_positive_area = float(flat_area[inside & ~flat_echo].sum())
    false_negative_area = float(flat_area[~inside & flat_echo].sum())

    return WalkTruth(
        false_positive_rate=false_positive_area / area_inside if area_inside else 0.0,
        false_negative_rate=false_negative_area / area_outside if area_outside else 0.0,
        n_inside=n_inside,
        n_outside=n_outside,
        area_inside_km2=area_inside,
        area_outside_km2=area_outside,
    )
