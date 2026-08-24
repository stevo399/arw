import numpy as np
import pyart
import pytest
from unittest.mock import patch, MagicMock
from src.parser import SweepData, extract_sweep_data, parse_radar_file

REFERENCE_VOLUME = "cache/KEMX/KEMX20260712_022646_V06"


@pytest.fixture(scope="module")
def radar():
    return pyart.io.read_nexrad_archive(REFERENCE_VOLUME)


def test_extract_sweep_data_carries_polarimetric_fields(radar):
    sweep = extract_sweep_data(radar)
    assert isinstance(sweep, SweepData)
    assert sweep.reflectivity.shape == sweep.rhohv.shape
    assert sweep.reflectivity.shape == sweep.zdr.shape
    assert sweep.reflectivity.shape[0] == len(sweep.azimuths)
    assert sweep.reflectivity.shape[1] == len(sweep.ranges_m)


def test_extract_sweep_data_uses_masked_fill(radar):
    sweep = extract_sweep_data(radar)
    assert np.isnan(sweep.reflectivity).any()
    assert not np.ma.isMaskedArray(sweep.reflectivity)


def test_extract_sweep_data_records_site_altitude(radar):
    sweep = extract_sweep_data(radar)
    assert isinstance(sweep.radar_alt_m, float)


def test_extract_sweep_data_records_per_ray_elevations(radar):
    sweep = extract_sweep_data(radar)
    assert sweep.elevations.shape == sweep.azimuths.shape
    assert isinstance(sweep.elevation_angle, float)


def test_extract_sweep_data_treats_all_nan_field_as_absent(radar):
    """velocity/spectrum_width are present-but-all-NaN on the reflectivity
    sweep (split-cut scan strategy artifact) -- must read as None, not as
    a usable all-NaN array, or Tasks 6-7 will treat "key present" as
    "data available"."""
    sweep = extract_sweep_data(radar)
    assert sweep.velocity is None
    assert sweep.spectrum_width is None


def test_extract_sweep_data_all_nan_field_returns_none_not_array():
    """Synthetic, deterministic version of the same rule: a field key that
    exists on the radar but is 100% NaN for this sweep must come back as
    None."""

    class AllNanVelocityRadar:
        nsweeps = 1
        fixed_angle = {"data": np.array([0.5])}
        latitude = {"data": np.array([35.0])}
        longitude = {"data": np.array([-97.0])}
        altitude = {"data": np.array([390.0])}
        azimuth = {"data": np.linspace(0, 359.5, 8)}
        elevation = {"data": np.full(8, 0.5)}
        range = {"data": np.arange(2125.0, 4625.0, 250.0)}
        time = {"units": "seconds since 2026-01-01T00:00:00Z"}
        fields = {
            "reflectivity": {"data": np.ma.masked_invalid(np.full((8, 10), 25.0))},
            "velocity": {"data": np.ma.masked_invalid(np.full((8, 10), np.nan))},
        }
        instrument_parameters = None

        def get_start_end(self, sweep_index):
            return 0, 7

    sweep = extract_sweep_data(AllNanVelocityRadar())
    assert sweep.velocity is None
    assert sweep.reflectivity is not None


def test_extract_sweep_data_all_nan_reflectivity_stays_an_array_not_none():
    """Reflectivity is required and can legitimately be all-NaN on a
    genuinely clear-air scan -- that must NOT be collapsed to None, unlike
    the optional co-registered fields."""

    class ClearAirRadar:
        nsweeps = 1
        fixed_angle = {"data": np.array([0.5])}
        latitude = {"data": np.array([35.0])}
        longitude = {"data": np.array([-97.0])}
        altitude = {"data": np.array([390.0])}
        azimuth = {"data": np.linspace(0, 359.5, 8)}
        elevation = {"data": np.full(8, 0.5)}
        range = {"data": np.arange(2125.0, 4625.0, 250.0)}
        time = {"units": "seconds since 2026-01-01T00:00:00Z"}
        fields = {"reflectivity": {"data": np.ma.masked_invalid(np.full((8, 10), np.nan))}}
        instrument_parameters = None

        def get_start_end(self, sweep_index):
            return 0, 7

    sweep = extract_sweep_data(ClearAirRadar())
    assert sweep.reflectivity is not None
    assert np.isnan(sweep.reflectivity).all()


def test_extract_sweep_data_handles_absent_polarimetric_fields():
    """Pre-2013 volumes have no RhoHV. Absent must mean None, never zeros."""

    class NoDualPolRadar:
        nsweeps = 1
        fixed_angle = {"data": np.array([0.5])}
        latitude = {"data": np.array([35.0])}
        longitude = {"data": np.array([-97.0])}
        altitude = {"data": np.array([390.0])}
        azimuth = {"data": np.linspace(0, 359.5, 8)}
        elevation = {"data": np.full(8, 0.5)}
        range = {"data": np.arange(2125.0, 4625.0, 250.0)}
        time = {"units": "seconds since 2011-05-01T00:00:00Z"}
        fields = {"reflectivity": {"data": np.ma.masked_invalid(np.full((8, 10), 25.0))}}
        instrument_parameters = None

        def get_start_end(self, sweep_index):
            return 0, 7

    sweep = extract_sweep_data(NoDualPolRadar())
    assert sweep.rhohv is None
    assert sweep.zdr is None
    assert sweep.reflectivity is not None


def _make_mock_radar():
    """Create a mock Py-ART radar object with reflectivity data."""
    radar = MagicMock()
    radar.nsweeps = 3
    radar.fixed_angle = {"data": np.array([0.5, 1.5, 2.4])}
    radar.latitude = {"data": np.array([35.3331])}
    radar.longitude = {"data": np.array([-97.2778])}
    radar.altitude = {"data": np.array([365.0])}

    radar.get_start_end.return_value = (0, 359)
    sweep_data = np.random.uniform(-10, 60, (360, 1832)).astype(np.float32)
    radar.fields = {
        "reflectivity": {"data": np.ma.array(sweep_data, mask=False)}
    }
    radar.azimuth = {"data": np.linspace(0, 359, 360)}
    radar.elevation = {"data": np.full(360, 0.5)}
    radar.range = {"data": np.linspace(0, 459750, 1832)}
    radar.time = {"units": "seconds since 2026-04-08T18:30:00Z"}
    radar.instrument_parameters = None
    return radar


def test_extract_sweep_data_returns_dataclass():
    mock_radar = _make_mock_radar()
    result = extract_sweep_data(mock_radar)
    assert isinstance(result, SweepData)
    assert result.reflectivity.shape[0] == 360
    assert result.reflectivity.shape[1] == 1832
    assert result.radar_lat == 35.3331
    assert result.radar_lon == -97.2778


def test_extract_sweep_data_uses_lowest_sweep():
    mock_radar = _make_mock_radar()
    result = extract_sweep_data(mock_radar)
    assert result.elevation_angle == 0.5


def test_extract_sweep_data_returns_azimuth_and_range():
    mock_radar = _make_mock_radar()
    result = extract_sweep_data(mock_radar)
    assert len(result.azimuths) == 360
    assert len(result.ranges_m) == 1832
    assert result.ranges_m[-1] > 400000


def test_extract_sweep_data_elevation_angles():
    mock_radar = _make_mock_radar()
    result = extract_sweep_data(mock_radar)
    assert result.elevation_angles == [0.5, 1.5, 2.4]


def test_parse_radar_file_returns_radar_object():
    mock_radar = _make_mock_radar()
    with patch("src.parser.pyart.io.read_nexrad_archive", return_value=mock_radar):
        radar = parse_radar_file("/fake/path.V06")
    assert radar is mock_radar
