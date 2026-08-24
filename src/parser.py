from dataclasses import dataclass
import numpy as np
import pyart

from src.sweeps import select_reflectivity_sweep

NEXRAD_FIELD_NAMES = {
    "reflectivity": "reflectivity",
    "velocity": "velocity",
    "rhohv": "cross_correlation_ratio",
    "zdr": "differential_reflectivity",
    "phidp": "differential_phase",
    "spectrum_width": "spectrum_width",
    "clutter_power_removed": "clutter_filter_power_removed",
}


@dataclass
class SweepData:
    """One radar sweep with all co-registered fields available for it."""

    reflectivity: np.ndarray
    azimuths: np.ndarray
    ranges_m: np.ndarray
    elevation_angle: float          # nominal fixed_angle, for reporting
    elevations: np.ndarray          # per-ray actual elevation, for georeferencing
    elevation_angles: list[float]
    radar_lat: float
    radar_lon: float
    radar_alt_m: float
    timestamp: str

    velocity: np.ndarray | None = None
    rhohv: np.ndarray | None = None
    zdr: np.ndarray | None = None
    phidp: np.ndarray | None = None
    spectrum_width: np.ndarray | None = None
    clutter_power_removed: np.ndarray | None = None
    nyquist_velocity: float | None = None
    gate_classification: np.ndarray | None = None


def parse_radar_file(filepath: str):
    """Read a NEXRAD Level II file and return the pyart Radar object."""
    return pyart.io.read_nexrad_archive(filepath)


def _extract_field(
    radar,
    nexrad_name: str,
    sweep_start: int,
    sweep_end: int,
    treat_all_nan_as_absent: bool = True,
) -> np.ndarray | None:
    """Pull one field for one sweep, or None if it carries no usable data.

    A field key can be present in `radar.fields` while every gate for this
    particular sweep is masked/NaN (e.g. `velocity`/`spectrum_width` are
    reflectivity-sweep artifacts of the split-cut scan strategy: they exist
    on the volume but were never populated for the surveillance cut this
    sweep selects). Treating "key present" as "data available" would hand
    downstream code (and Tasks 6-7's dealiasing/QC) an all-NaN array it
    reads as real data. Absent must mean None regardless of *why* there is
    nothing usable -- missing key or all-NaN slice.

    `treat_all_nan_as_absent` is off for `reflectivity`: it is SweepData's
    one required field, and a genuinely clear-air scan (no precipitation
    anywhere) is a legitimate all-NaN reading, not a missing field -- the
    caller already selected the sweep with the best reflectivity coverage
    available, so an empty result there is real data, not an artifact.
    """
    if nexrad_name not in radar.fields:
        return None
    data = radar.fields[nexrad_name]["data"][sweep_start:sweep_end + 1]
    if hasattr(data, "filled"):
        data = data.filled(np.nan)
    data = np.asarray(data, dtype=float)
    if treat_all_nan_as_absent and not np.any(np.isfinite(data)):
        return None
    return data


def extract_sweep_data(radar) -> SweepData:
    """Extract the reflectivity sweep with every co-registered field it carries."""
    sweep_index = select_reflectivity_sweep(radar)
    sweep_start, sweep_end = radar.get_start_end(sweep_index)

    fields = {
        key: _extract_field(
            radar, nexrad_name, sweep_start, sweep_end,
            treat_all_nan_as_absent=(key != "reflectivity"),
        )
        for key, nexrad_name in NEXRAD_FIELD_NAMES.items()
    }

    nyquist = None
    instrument = getattr(radar, "instrument_parameters", None)
    if instrument and "nyquist_velocity" in instrument:
        nyquist = float(instrument["nyquist_velocity"]["data"][sweep_start])

    elevation_angles = sorted(set(np.round(radar.fixed_angle["data"], 1)))

    return SweepData(
        reflectivity=fields["reflectivity"],
        velocity=fields["velocity"],
        rhohv=fields["rhohv"],
        zdr=fields["zdr"],
        phidp=fields["phidp"],
        spectrum_width=fields["spectrum_width"],
        clutter_power_removed=fields["clutter_power_removed"],
        azimuths=np.asarray(radar.azimuth["data"][sweep_start:sweep_end + 1], dtype=float),
        ranges_m=np.asarray(radar.range["data"], dtype=float),
        elevation_angle=float(radar.fixed_angle["data"][sweep_index]),
        elevations=np.asarray(radar.elevation["data"][sweep_start:sweep_end + 1], dtype=float),
        elevation_angles=[float(a) for a in elevation_angles],
        radar_lat=float(radar.latitude["data"][0]),
        radar_lon=float(radar.longitude["data"][0]),
        radar_alt_m=float(radar.altitude["data"][0]),
        timestamp=str(radar.time["units"]).replace("seconds since ", ""),
        nyquist_velocity=nyquist,
    )


@dataclass
class VelocitySweep:
    """Velocity data from a single radar sweep."""
    velocity: np.ndarray
    azimuths: np.ndarray
    ranges_m: np.ndarray
    elevation_angle: float
    nyquist_velocity: float


@dataclass
class VelocityData:
    """Multi-sweep velocity data from a radar volume."""
    sweeps: list[VelocitySweep]
    radar_lat: float
    radar_lon: float


def extract_velocity(radar, max_sweeps: int = 3) -> VelocityData | None:
    """Extract velocity from the lowest N sweeps of a pyart Radar object.

    Returns None if the radar has no velocity field.
    """
    if "velocity" not in radar.fields:
        return None

    # Apply Py-ART region-based dealiasing to unwrap aliased velocities
    try:
        pyart.correct.dealias_region_based(radar, field="velocity")
    except Exception:
        pass  # proceed with raw velocity if dealiasing fails

    sweeps_to_read = min(max_sweeps, radar.nsweeps)
    sweeps: list[VelocitySweep] = []

    for sweep_index in range(sweeps_to_read):
        sweep_start, sweep_end = radar.get_start_end(sweep_index)
        velocity = radar.fields["velocity"]["data"][sweep_start:sweep_end + 1]
        azimuths = radar.azimuth["data"][sweep_start:sweep_end + 1]
        ranges_m = radar.range["data"]

        if hasattr(velocity, "filled"):
            velocity = velocity.filled(np.nan)

        nyquist = float(radar.instrument_parameters["nyquist_velocity"]["data"][sweep_start])

        sweeps.append(VelocitySweep(
            velocity=velocity,
            azimuths=azimuths,
            ranges_m=ranges_m,
            elevation_angle=float(radar.fixed_angle["data"][sweep_index]),
            nyquist_velocity=nyquist,
        ))

    return VelocityData(
        sweeps=sweeps,
        radar_lat=float(radar.latitude["data"][0]),
        radar_lon=float(radar.longitude["data"][0]),
    )
