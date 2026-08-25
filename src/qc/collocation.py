"""Geographic collocation between sweeps.

Rotation signatures are detected on Doppler cuts; classification runs on the
surveillance cut. The two sweeps have different azimuth sampling, so matching
by array index would silently assume a shared grid. Everything here works in
geographic space instead.
"""

import numpy as np

from src.geometry import gate_coordinates

EARTH_RADIUS_KM = 6371.0

# ARW-tuned. A debris field extends beyond the circulation that lofted it, so
# the protection radius is the signature diameter plus this margin.
ROTATION_COLLOCATION_MARGIN_KM = 5.0


def _haversine_km(lat1, lon1, lat2: float, lon2: float) -> np.ndarray:
    """Great-circle distance from every gate to one point, in km."""
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = phi2 - phi1
    delta_lambda = np.radians(lon2 - lon1)
    a = np.sin(delta_phi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def rotation_proximity_mask(
    sweep,
    rotation_signatures,
    margin_km: float = ROTATION_COLLOCATION_MARGIN_KM,
) -> np.ndarray:
    """Gates lying within a rotation signature's radius plus margin."""
    shape = np.asarray(sweep.reflectivity).shape
    mask = np.zeros(shape, dtype=bool)
    if not rotation_signatures:
        return mask

    lat, lon = gate_coordinates(
        azimuths=sweep.azimuths,
        ranges_m=sweep.ranges_m,
        elevation_deg=sweep.elevation_angle,
        radar_lat=sweep.radar_lat,
        radar_lon=sweep.radar_lon,
    )

    for signature in rotation_signatures:
        radius_km = signature.diameter_km / 2.0 + margin_km
        distance_km = _haversine_km(lat, lon, signature.centroid_lat, signature.centroid_lon)
        mask |= distance_km <= radius_km

    return mask
