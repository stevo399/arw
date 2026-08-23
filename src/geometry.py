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
    elevation_deg: float,
    radar_lat: float,
    radar_lon: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Geographic coordinates of every gate in a sweep.

    Returns (latitudes, longitudes), each shaped (n_rays, n_gates).
    """
    ranges_km = np.asarray(ranges_m, dtype=float) / 1000.0
    ranges_2d, azimuths_2d = np.meshgrid(ranges_km, np.asarray(azimuths, dtype=float))
    x, y, _z = antenna_to_cartesian(ranges_2d, azimuths_2d, elevation_deg)
    lon, lat = cartesian_to_geographic_aeqd(x, y, radar_lon, radar_lat)
    return lat, lon
