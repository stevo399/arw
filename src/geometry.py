"""Radar-to-geographic mathematics.

This is the only module that converts between radar antenna coordinates and
geographic coordinates. All beam propagation uses the 4/3 effective earth
radius model (Doviak and Zrnic, equations 2.28b and 2.28c), which is the
standard for WSR-88D data and what Py-ART implements internally.
"""

import numpy as np
from pyart.core.transforms import antenna_to_cartesian, cartesian_to_geographic_aeqd

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
