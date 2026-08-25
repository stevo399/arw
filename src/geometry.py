"""Radar-to-geographic mathematics.

This is the only module that converts between radar antenna coordinates and
geographic coordinates. All beam propagation uses the 4/3 effective earth
radius model (Doviak and Zrnic, equations 2.28b and 2.28c), which is the
standard for WSR-88D data and what Py-ART implements internally.
"""

import numpy as np
from pyart.core.transforms import antenna_to_cartesian, cartesian_to_geographic_aeqd
from scipy.ndimage import label

EARTH_RADIUS_M = 6371000.0
EFFECTIVE_EARTH_RADIUS_M = EARTH_RADIUS_M * 4.0 / 3.0


def gate_coordinates(
    azimuths: np.ndarray,
    ranges_m: np.ndarray,
    elevation_deg: float | np.ndarray,
    radar_lat: float,
    radar_lon: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Geographic coordinates of every gate in a sweep.

    Args:
        azimuths: Azimuth angle for each ray, shaped (n_rays,).
        ranges_m: Range to each gate, shaped (n_gates,).
        elevation_deg: Elevation angle(s). Either a scalar (nominal sweep angle, less preferred)
            or a 1-D array of per-ray elevation angles (preferred). Per-ray elevation is physically
            correct — it reflects where the antenna actually pointed. Pyart.gate_latitude uses
            per-ray elevation; passing nominal fixed_angle instead causes 10-100m disagreement.
        radar_lat: Latitude of radar site.
        radar_lon: Longitude of radar site.

    Returns:
        (latitudes, longitudes), each shaped (n_rays, n_gates).
    """
    ranges_km = np.asarray(ranges_m, dtype=float) / 1000.0
    ranges_2d, azimuths_2d = np.meshgrid(ranges_km, np.asarray(azimuths, dtype=float))

    elevations = np.asarray(elevation_deg, dtype=float)
    # One angle per ray broadcasts down the range axis; a scalar broadcasts everywhere.
    elevations_2d = elevations[:, None] if elevations.ndim == 1 else elevations

    x, y, _z = antenna_to_cartesian(ranges_2d, azimuths_2d, elevations_2d)
    lon, lat = cartesian_to_geographic_aeqd(x, y, radar_lon, radar_lat)
    return lat, lon


def interpolate_azimuth(azimuths: np.ndarray, index: float) -> float:
    """Interpolate a fractional ray index to a compass azimuth, seam-safe.

    NEXRAD azimuth arrays wrap once from ~360 back to ~0. Interpolating the raw
    array across that seam yields a bearing 180 degrees wrong, so a storm due
    north reads as due south. Unwrap first, interpolate, then fold back.
    """
    azimuths = np.asarray(azimuths, dtype=float)
    unwrapped = np.unwrap(azimuths, period=360.0)
    return float(np.interp(index, np.arange(len(unwrapped)), unwrapped)) % 360.0


def gate_latlon(
    azimuth_deg: float,
    range_m: float,
    elevation_deg: float,
    radar_lat: float,
    radar_lon: float,
) -> tuple[float, float]:
    """Geographic coordinates of a single gate.

    Scalar convenience wrapper around `gate_coordinates` for call sites that
    convert one (azimuth, range) pair at a time (e.g. per-point polygon
    vertex construction). Delegates to the same Py-ART-conforming maths
    rather than reimplementing them.
    """
    lat, lon = gate_coordinates(
        azimuths=np.asarray([azimuth_deg], dtype=float),
        ranges_m=np.asarray([range_m], dtype=float),
        elevation_deg=elevation_deg,
        radar_lat=radar_lat,
        radar_lon=radar_lon,
    )
    return float(lat[0, 0]), float(lon[0, 0])


def beam_height_m(
    range_m: float | np.ndarray,
    elevation_deg: float,
    site_alt_m: float = 0.0,
) -> float | np.ndarray:
    """Height of the radar beam centre above sea level at a given slant range.

    Doviak and Zrnic equation 2.28b under the 4/3 effective earth radius model.
    """
    r = np.asarray(range_m, dtype=float)
    theta = np.radians(elevation_deg)
    R = EFFECTIVE_EARTH_RADIUS_M
    height = np.sqrt(r**2 + R**2 + 2.0 * r * R * np.sin(theta)) - R + site_alt_m
    return float(height) if np.isscalar(range_m) or height.ndim == 0 else height


def ground_range_m(
    slant_range_m: float | np.ndarray,
    elevation_deg: float,
) -> np.ndarray:
    """Great-circle distance along the ground beneath each gate.

    Doviak and Zrnic equation 2.28c under the 4/3 effective earth radius model.
    """
    r = np.asarray(slant_range_m, dtype=float)
    theta = np.radians(elevation_deg)
    R = EFFECTIVE_EARTH_RADIUS_M
    height = np.sqrt(r**2 + R**2 + 2.0 * r * R * np.sin(theta)) - R
    return R * np.arcsin(r * np.cos(theta) / (R + height))


def label_periodic_azimuth(
    mask: np.ndarray,
    structure: np.ndarray | None = None,
) -> tuple[np.ndarray, int]:
    """Connected-component labelling that treats the azimuth axis as periodic.

    Ray 0 and ray N-1 of a radar sweep are physically adjacent -- both point just
    either side of due north. scipy.ndimage.label treats them as opposite edges
    of the array, so a storm straddling due north becomes two components: two
    objects with two wrong centroids, and a protected core that fails to protect
    the other half of its own storm.

    Returns (labeled, count) with the same contract as scipy.ndimage.label.
    """
    labeled, count = label(mask, structure=structure)
    if count < 2 or labeled.shape[0] < 2:
        return labeled, count

    first_row = labeled[0]
    last_row = labeled[-1]
    n_gates = labeled.shape[1]

    parent = np.arange(count + 1)

    def find(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(a, b):
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[max(root_a, root_b)] = min(root_a, root_b)

    # A structure of None means 4-connectivity: only the same gate index is
    # adjacent across the seam. An explicit 3x3 structure means 8-connectivity,
    # so neighbouring gate indices touch diagonally across the seam too.
    offsets = (0,) if structure is None else (-1, 0, 1)

    for offset in offsets:
        if offset < 0:
            head, tail = first_row[-offset:], last_row[:n_gates + offset]
        elif offset > 0:
            head, tail = first_row[:n_gates - offset], last_row[offset:]
        else:
            head, tail = first_row, last_row
        both = (head > 0) & (tail > 0)
        for a, b in zip(head[both], tail[both]):
            union(int(a), int(b))

    roots = np.array([find(i) for i in range(count + 1)])
    roots[0] = 0
    surviving = np.unique(roots[1:])
    remap = np.zeros(count + 1, dtype=labeled.dtype)
    for new_id, root in enumerate(surviving, start=1):
        remap[roots == root] = new_id
    remap[0] = 0
    return remap[labeled], int(len(surviving))


def gate_areas_km2(
    azimuths: np.ndarray,
    ranges_m: np.ndarray,
    elevation_deg: float,
) -> np.ndarray:
    """Ground-projected area of one gate at each range bin, in square km.

    Azimuth spacing is the median gap between consecutive rays, which is robust
    to the small irregularities in real NEXRAD azimuth sequences. The legacy
    implementation used the gap between the first two rays only.
    """
    ranges_m = np.asarray(ranges_m, dtype=float)
    if len(ranges_m) < 2 or len(azimuths) < 2:
        return np.zeros_like(ranges_m, dtype=float)

    range_spacing_m = float(np.median(np.abs(np.diff(ranges_m))))
    azimuth_gaps = np.abs(np.diff(np.unwrap(np.asarray(azimuths, dtype=float), period=360.0)))
    az_spacing_rad = np.radians(float(np.median(azimuth_gaps)))

    ground = ground_range_m(ranges_m, elevation_deg)
    return (ground * az_spacing_rad * range_spacing_m) / 1e6
