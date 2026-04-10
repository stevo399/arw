# src/motion.py
import math
from dataclasses import dataclass
from datetime import datetime
import numpy as np
from src.detection import degrees_to_bearing

KM_PER_DEGREE_LAT = 111.32
KM_PER_MILE = 1.60934
NEARLY_STATIONARY_KMH = 2.0


@dataclass
class MotionVector:
    """Computed motion for a storm track."""
    speed_kmh: float
    speed_mph: int
    heading_deg: float | None
    heading_label: str


def compute_motion(positions: list[tuple[datetime, float, float]]) -> MotionVector:
    """Compute motion vector from a list of (timestamp, lat, lon) positions.

    Uses linear regression over position history.
    Single-point: stationary. < 2 km/h: nearly stationary.
    """
    if len(positions) < 2:
        return MotionVector(speed_kmh=0.0, speed_mph=0, heading_deg=None, heading_label="stationary")

    # Convert timestamps to seconds from first position
    t0 = positions[0][0]
    times_s = np.array([(p[0] - t0).total_seconds() for p in positions])
    lats = np.array([p[1] for p in positions])
    lons = np.array([p[2] for p in positions])

    # If all timestamps are identical, no motion can be computed
    if times_s[-1] == 0.0:
        return MotionVector(speed_kmh=0.0, speed_mph=0, heading_deg=None, heading_label="stationary")

    # Linear regression: fit lat and lon vs time
    lat_slope = np.polyfit(times_s, lats, 1)[0]  # degrees per second
    lon_slope = np.polyfit(times_s, lons, 1)[0]  # degrees per second

    # Convert to km/h
    mean_lat = np.mean(lats)
    km_per_degree_lon = KM_PER_DEGREE_LAT * math.cos(math.radians(mean_lat))

    lat_kmh = lat_slope * 3600 * KM_PER_DEGREE_LAT
    lon_kmh = lon_slope * 3600 * km_per_degree_lon

    speed_kmh = math.sqrt(lat_kmh ** 2 + lon_kmh ** 2)
    speed_kmh = round(speed_kmh, 1)

    if speed_kmh < NEARLY_STATIONARY_KMH:
        return MotionVector(
            speed_kmh=speed_kmh,
            speed_mph=round(speed_kmh / KM_PER_MILE),
            heading_deg=None,
            heading_label="nearly stationary",
        )

    # Heading: atan2(east, north) to get compass bearing
    heading_rad = math.atan2(lon_kmh, lat_kmh)
    heading_deg = math.degrees(heading_rad) % 360
    heading_deg = round(heading_deg, 1)
    heading_label = degrees_to_bearing(heading_deg)

    speed_mph = round(speed_kmh / KM_PER_MILE)

    return MotionVector(
        speed_kmh=speed_kmh,
        speed_mph=speed_mph,
        heading_deg=heading_deg,
        heading_label=heading_label,
    )
