"""Storm motion measured by matching each storm's reflectivity pattern between scans.

Following a storm's centre does not measure motion: centres jitter by about
2 km between scans and jump when storms merge or split.  Matching the pattern
of echo around a storm against the previous scan measures its displacement
directly, and a storm first detected in this scan still has one, because the
previous scan's field exists whether or not the storm was detected in it.

Matching alone locks onto the wrong echo in dense scenes, so every match is
guarded (evidence: docs/test_reports/2026-09-13-pattern-motion-evaluation.md):
the template carries every echo within 10 km of the storm, a correlation peak
on the edge of the search window is not a match, a velocity far from its
neighbours' vector median is rejected, and a storm too isolated for neighbours
to judge needs a strong correlation.
"""

from dataclasses import dataclass
import math
import statistics

import numpy as np
from scipy.signal import correlate

from src.geometry import gate_ground_xy_m, ground_range_m
from src.sites import haversine_distance_km

GRID_M = 500.0
CONTEXT_MARGIN_M = 10000.0
MAX_MATCH_SPEED_KMH = 150.0
# Reflectivity below this carries no pattern; values above are offset so a
# weak echo still counts.
MIN_PATTERN_DBZ = 20.0
PATTERN_OFFSET_DBZ = 5.0
MIN_TEMPLATE_CELLS = 8
NEIGHBOUR_RADIUS_KM = 60.0
MIN_NEIGHBOURS_TO_JUDGE = 3
MAX_NEIGHBOUR_DEPARTURE_KMH = 25.0
# A storm too isolated for its neighbours to judge needs a strong match: on
# 868 such matches, requiring this correlation kept the median prediction
# error (1.26 against 1.85 km for no motion) while bringing the worst tenth
# back to no-motion's (5.84 against 5.86 km); accepting all left it at 7.39 km
# (docs/test_reports/2026-09-13-isolated-storm-motion-evaluation.md).
MIN_UNJUDGED_CORRELATION = 0.6
# A match the guards reject is accepted when it agrees this closely with the
# same storm's previous match.  On 5,464 matches this changed no tail and
# slightly improved median error and outline overlap, and it keeps a
# stationary echo among moving storms from taking their motion
# (docs/test_reports/2026-09-13-isolated-storm-motion-evaluation.md).
MAX_SELF_AGREEMENT_KMH = 10.0
# An isolated storm's match needs corroboration by its own previous match when
# it is faster than this.  Of 29 uncorroborated isolated matches above
# 80 km/h, 28 predicted worse than no motion, and no match corroborated by
# neighbours exceeded 80 km/h.
MAX_UNJUDGED_SPEED_KMH = 80.0
# Beyond this range echoes are too weak and sparse to match: judged or not,
# matches there predicted no better than no motion, and noisy neighbours
# corroborated a 116 km/h artifact (KIWA 2026-09-07 23:37Z)
# (docs/test_reports/2026-09-13-isolated-storm-motion-evaluation.md).
MAX_MATCH_RANGE_KM = 350.0


@dataclass(frozen=True)
class PatternMatch:
    east_kmh: float
    north_kmh: float
    correlation: float
    at_edge: bool


class GroundSampler:
    """Nearest-gate lookup, for one sweep, from ground offsets east and north of the radar in metres.

    Uses the sweep's own azimuths, so it does not matter which azimuth the
    sweep began at.
    """

    def __init__(self, azimuths: np.ndarray, ranges_m: np.ndarray, elevation_deg: float):
        azimuths = np.asarray(azimuths, dtype=float) % 360.0
        self._order = np.argsort(azimuths)
        self._sorted_azimuths = azimuths[self._order]
        self._ground_m = np.asarray(ground_range_m(np.asarray(ranges_m, dtype=float), float(elevation_deg)), dtype=float)
        self._gate_spacing_m = float(np.median(np.diff(self._ground_m))) if len(self._ground_m) > 1 else 0.0

    @classmethod
    def from_sweep(cls, sweep) -> "GroundSampler":
        return cls(sweep.azimuths, sweep.ranges_m, sweep.elevation_angle)

    def sample(self, field: np.ndarray, x_m: np.ndarray, y_m: np.ndarray, fill: float) -> np.ndarray:
        x_m = np.asarray(x_m, dtype=float)
        y_m = np.asarray(y_m, dtype=float)
        azimuth = np.degrees(np.arctan2(x_m, y_m)) % 360.0
        ground = np.hypot(x_m, y_m)
        count = len(self._sorted_azimuths)
        after = np.searchsorted(self._sorted_azimuths, azimuth) % count
        before = (after - 1) % count
        gap_after = np.abs((self._sorted_azimuths[after] - azimuth + 180.0) % 360.0 - 180.0)
        gap_before = np.abs((self._sorted_azimuths[before] - azimuth + 180.0) % 360.0 - 180.0)
        rays = self._order[np.where(gap_after <= gap_before, after, before)]
        gates = np.rint(np.interp(ground, self._ground_m, np.arange(len(self._ground_m)))).astype(int)
        half = self._gate_spacing_m / 2.0
        inside = (ground >= self._ground_m[0] - half) & (ground <= self._ground_m[-1] + half)
        values = np.full(x_m.shape, fill, dtype=float)
        values[inside] = np.asarray(field)[rays[inside], gates[inside]]
        return values


def _grid(x0: float, x1: float, y0: float, y1: float) -> tuple[np.ndarray, np.ndarray]:
    """Ground grid; axis 0 runs north, axis 1 east."""
    return np.meshgrid(np.arange(x0, x1 + GRID_M, GRID_M), np.arange(y0, y1 + GRID_M, GRID_M))


def _pattern(sampler: GroundSampler, reflectivity: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    dbz = np.nan_to_num(sampler.sample(reflectivity, x, y, np.nan), nan=0.0)
    return np.clip(dbz - MIN_PATTERN_DBZ + PATTERN_OFFSET_DBZ, 0.0, None)


def _refine(values: np.ndarray, index: int) -> float:
    """Sub-cell peak position from a parabola through the peak and its neighbours."""
    if 0 < index < len(values) - 1:
        left, centre, right = values[index - 1], values[index], values[index + 1]
        curvature = left - 2.0 * centre + right
        if curvature != 0.0:
            return index + 0.5 * (left - right) / curvature
    return float(index)


def match_storm(current_sweep, storm_mask: np.ndarray, previous_sweep, hours: float) -> PatternMatch | None:
    """Velocity that carried the echo around a storm from the previous scan to this one.

    None when the storm or the previous scan has no pattern to match.
    """
    if hours <= 0.0:
        return None
    rays, gates = np.nonzero(storm_mask)
    if rays.size == 0:
        return None
    storm_x, storm_y = gate_ground_xy_m(rays, gates, current_sweep.azimuths, current_sweep.ranges_m, current_sweep.elevations)
    x0 = float(storm_x.min()) - CONTEXT_MARGIN_M
    x1 = float(storm_x.max()) + CONTEXT_MARGIN_M
    y0 = float(storm_y.min()) - CONTEXT_MARGIN_M
    y1 = float(storm_y.max()) + CONTEXT_MARGIN_M

    template_x, template_y = _grid(x0, x1, y0, y1)
    template = _pattern(GroundSampler.from_sweep(current_sweep), current_sweep.reflectivity, template_x, template_y)
    if np.count_nonzero(template) < MIN_TEMPLATE_CELLS:
        return None
    pad = int(math.ceil(MAX_MATCH_SPEED_KMH * hours * 1000.0 / GRID_M))
    search_x, search_y = _grid(x0 - pad * GRID_M, x1 + pad * GRID_M, y0 - pad * GRID_M, y1 + pad * GRID_M)
    search = _pattern(GroundSampler.from_sweep(previous_sweep), previous_sweep.reflectivity, search_x, search_y)
    if not np.any(search):
        return None
    template = template[: search.shape[0] - 2 * pad, : search.shape[1] - 2 * pad]
    template = template - template.mean()
    template_energy = float(np.sum(template ** 2))
    if template_energy <= 0.0:
        return None

    ones = np.ones_like(template)
    covariance = correlate(search, template, mode="valid", method="fft")
    local_sum = correlate(search, ones, mode="valid", method="fft")
    local_square_sum = correlate(search ** 2, ones, mode="valid", method="fft")
    local_variance = np.clip(local_square_sum - local_sum ** 2 / template.size, 1e-9, None)
    correlation = covariance / np.sqrt(local_variance * template_energy)

    row, col = np.unravel_index(int(np.argmax(correlation)), correlation.shape)
    at_edge = row in (0, correlation.shape[0] - 1) or col in (0, correlation.shape[1] - 1)
    north_offset = _refine(correlation[:, col], row) - pad
    east_offset = _refine(correlation[row, :], col) - pad
    # The pattern sat at that offset in the previous scan; it has since moved the opposite way.
    to_kmh = GRID_M / 1000.0 / hours
    return PatternMatch(
        east_kmh=-east_offset * to_kmh,
        north_kmh=-north_offset * to_kmh,
        correlation=float(correlation[row, col]),
        at_edge=bool(at_edge),
    )


def _vector_median(vectors: list[tuple[float, float]]) -> tuple[float, float]:
    return statistics.median(v[0] for v in vectors), statistics.median(v[1] for v in vectors)


def _within_neighbour_radius(centres: dict[int, tuple[float, float]], a: int, b: int) -> bool:
    return haversine_distance_km(*centres[a], *centres[b]) <= NEIGHBOUR_RADIUS_KM


def valid_matches(
    matches: dict[int, PatternMatch],
    ranges_km: dict[int, float] | None = None,
) -> dict[int, tuple[float, float]]:
    """Matches that are matches at all: not on the search edge, within the
    speed limit, and no farther than 350 km from the radar."""
    ranges_km = ranges_km or {}
    return {
        track_id: (match.east_kmh, match.north_kmh)
        for track_id, match in matches.items()
        # The search window is square, so its corners reach beyond the speed limit.
        if not match.at_edge
        and math.hypot(match.east_kmh, match.north_kmh) <= MAX_MATCH_SPEED_KMH
        and ranges_km.get(track_id, 0.0) <= MAX_MATCH_RANGE_KM
    }


def guard_matches(
    matches: dict[int, PatternMatch],
    centres: dict[int, tuple[float, float]],
    previous: dict[int, tuple[float, float]] | None = None,
    ranges_km: dict[int, float] | None = None,
) -> dict[int, tuple[float, float]]:
    """Accepted (east, north) km/h velocities by track id.

    A peak on the search edge, faster than the search speed limit, or farther
    than 350 km from the radar (`ranges_km`) is not a match.  A storm with at
    least three other matches within 60 km is rejected when its velocity
    departs from their vector median by more than 25 km/h; a storm with fewer
    is rejected when its match correlates below 0.6 or is faster than 80 km/h.

    A match either rule rejects is still accepted when it agrees within
    10 km/h with the same storm's previous valid match (`previous`): two scans
    corroborate each other where neighbours cannot, as for a stationary echo
    among moving storms.
    """
    previous = previous or {}
    candidates = valid_matches(matches, ranges_km)
    accepted: dict[int, tuple[float, float]] = {}
    for track_id, velocity in candidates.items():
        neighbours = [
            other_velocity
            for other_id, other_velocity in candidates.items()
            if other_id != track_id and _within_neighbour_radius(centres, track_id, other_id)
        ]
        if len(neighbours) >= MIN_NEIGHBOURS_TO_JUDGE:
            east, north = _vector_median(neighbours)
            passes = math.hypot(velocity[0] - east, velocity[1] - north) <= MAX_NEIGHBOUR_DEPARTURE_KMH
        else:
            passes = (
                matches[track_id].correlation >= MIN_UNJUDGED_CORRELATION
                and math.hypot(*velocity) <= MAX_UNJUDGED_SPEED_KMH
            )
        earlier = previous.get(track_id)
        corroborated = earlier is not None and math.hypot(velocity[0] - earlier[0], velocity[1] - earlier[1]) <= MAX_SELF_AGREEMENT_KMH
        if passes or corroborated:
            accepted[track_id] = velocity
    return accepted


def nearby_velocity(
    track_id: int,
    accepted: dict[int, tuple[float, float]],
    centres: dict[int, tuple[float, float]],
) -> tuple[float, float] | None:
    """Vector median of other storms' accepted velocities within 60 km, or None."""
    neighbours = [
        velocity
        for other_id, velocity in accepted.items()
        if other_id != track_id and other_id in centres and _within_neighbour_radius(centres, track_id, other_id)
    ]
    return _vector_median(neighbours) if neighbours else None
