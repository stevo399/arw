from dataclasses import dataclass
import numpy as np
import pyart


@dataclass
class ReflectivityData:
    """Parsed reflectivity data from a single radar sweep."""
    reflectivity: np.ndarray
    azimuths: np.ndarray
    ranges_m: np.ndarray
    radar_lat: float
    radar_lon: float
    elevation_angle: float
    elevation_angles: list[float]
    timestamp: str


def extract_reflectivity(filepath: str) -> ReflectivityData:
    """Read a NEXRAD Level II file and extract reflectivity from the lowest sweep."""
    radar = pyart.io.read_nexrad_archive(filepath)
    elevation_angles = sorted(set(np.round(radar.fixed_angle["data"], 1)))
    sweep_start, sweep_end = radar.get_start_end(0)
    reflectivity = radar.fields["reflectivity"]["data"][sweep_start:sweep_end + 1]
    azimuths = radar.azimuth["data"][sweep_start:sweep_end + 1]
    ranges_m = radar.range["data"]
    if hasattr(reflectivity, "filled"):
        reflectivity = reflectivity.filled(np.nan)
    return ReflectivityData(
        reflectivity=reflectivity,
        azimuths=azimuths,
        ranges_m=ranges_m,
        radar_lat=float(radar.latitude["data"][0]),
        radar_lon=float(radar.longitude["data"][0]),
        elevation_angle=float(radar.fixed_angle["data"][0]),
        elevation_angles=[float(a) for a in elevation_angles],
        timestamp=str(radar.time["units"]).replace("seconds since ", ""),
    )
