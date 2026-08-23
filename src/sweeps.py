"""VCP-aware sweep selection.

NEXRAD volume coverage patterns use split cuts at low elevations: each tilt is
scanned twice, once as a surveillance cut (long PRF, reflectivity, unambiguous
range to 460 km, low Nyquist) and once as a Doppler cut (short PRF, velocity,
range to 300 km, high Nyquist). Selecting sweeps by index silently picks
surveillance cuts for velocity, which carry no velocity data.

Selection here inspects what each sweep actually contains, so it is
VCP-independent and survives NWS scan strategy changes.
"""

import numpy as np

ELEVATION_GROUP_TOLERANCE_DEG = 0.05


def sweep_field_coverage(radar, sweep_index: int, field: str) -> float:
    """Fraction of gates in a sweep carrying valid data for a field."""
    if field not in radar.fields:
        return 0.0
    sweep_start, sweep_end = radar.get_start_end(sweep_index)
    data = radar.fields[field]["data"][sweep_start:sweep_end + 1]
    if data.size == 0:
        return 0.0
    valid = ~np.ma.getmaskarray(data)
    if not np.ma.isMaskedArray(data):
        valid = ~np.isnan(np.asarray(data, dtype=float))
    return float(np.count_nonzero(valid)) / float(data.size)


def _elevation_groups(radar) -> list[list[int]]:
    """Group sweep indices by elevation, collapsing split cuts into one group."""
    angles = np.asarray(radar.fixed_angle["data"], dtype=float)
    order = sorted(range(radar.nsweeps), key=lambda i: (angles[i], i))
    groups: list[list[int]] = []
    for sweep_index in order:
        if groups and abs(angles[sweep_index] - angles[groups[-1][0]]) <= ELEVATION_GROUP_TOLERANCE_DEG:
            groups[-1].append(sweep_index)
        else:
            groups.append([sweep_index])
    return groups


def _best_in_group(radar, group: list[int], field: str) -> int | None:
    """Highest-coverage sweep for a field within one elevation group."""
    scored = [(sweep_field_coverage(radar, i, field), -i, i) for i in group]
    coverage, _, best_index = max(scored)
    return best_index if coverage > 0.0 else None


def select_reflectivity_sweep(radar) -> int:
    """Index of the sweep to use for reflectivity: lowest tilt, best coverage."""
    groups = _elevation_groups(radar)
    for group in groups:
        best = _best_in_group(radar, group, "reflectivity")
        if best is not None:
            return best
    return 0


def select_velocity_sweeps(radar, max_sweeps: int = 3) -> list[int]:
    """Indices of the lowest sweeps carrying usable velocity, one per elevation."""
    selected: list[int] = []
    for group in _elevation_groups(radar):
        if len(selected) >= max_sweeps:
            break
        best = _best_in_group(radar, group, "velocity")
        if best is not None:
            selected.append(best)
    return selected
