import numpy as np
from unittest.mock import patch, MagicMock
from src.parser import extract_reflectivity, ReflectivityData


def _make_mock_radar():
    """Create a mock Py-ART radar object with reflectivity data."""
    radar = MagicMock()
    radar.nsweeps = 3
    radar.fixed_angle = {"data": np.array([0.5, 1.5, 2.4])}
    radar.latitude = {"data": np.array([35.3331])}
    radar.longitude = {"data": np.array([-97.2778])}

    radar.get_start_end.return_value = (0, 359)
    sweep_data = np.random.uniform(-10, 60, (360, 1832)).astype(np.float32)
    radar.fields = {
        "reflectivity": {"data": np.ma.array(sweep_data, mask=False)}
    }
    radar.azimuth = {"data": np.linspace(0, 359, 360)}
    radar.range = {"data": np.linspace(0, 459750, 1832)}
    radar.time = {"units": "seconds since 2026-04-08T18:30:00Z"}
    return radar


def test_extract_reflectivity_returns_dataclass():
    mock_radar = _make_mock_radar()
    with patch("src.parser.pyart.io.read_nexrad_archive", return_value=mock_radar):
        result = extract_reflectivity("/fake/path.V06")
    assert isinstance(result, ReflectivityData)
    assert result.reflectivity.shape[0] == 360
    assert result.reflectivity.shape[1] == 1832
    assert result.radar_lat == 35.3331
    assert result.radar_lon == -97.2778


def test_extract_reflectivity_uses_lowest_sweep():
    mock_radar = _make_mock_radar()
    with patch("src.parser.pyart.io.read_nexrad_archive", return_value=mock_radar):
        result = extract_reflectivity("/fake/path.V06")
    assert result.elevation_angle == 0.5


def test_extract_reflectivity_returns_azimuth_and_range():
    mock_radar = _make_mock_radar()
    with patch("src.parser.pyart.io.read_nexrad_archive", return_value=mock_radar):
        result = extract_reflectivity("/fake/path.V06")
    assert len(result.azimuths) == 360
    assert len(result.ranges_m) == 1832
    assert result.ranges_m[-1] > 400000


def test_extract_reflectivity_elevation_angles():
    mock_radar = _make_mock_radar()
    with patch("src.parser.pyart.io.read_nexrad_archive", return_value=mock_radar):
        result = extract_reflectivity("/fake/path.V06")
    assert result.elevation_angles == [0.5, 1.5, 2.4]
