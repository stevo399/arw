import numpy as np
from unittest.mock import MagicMock
from src.parser import extract_velocity, VelocityData, VelocitySweep


def _make_mock_radar_with_velocity():
    """Create a mock Py-ART radar with both reflectivity and velocity fields."""
    radar = MagicMock()
    radar.nsweeps = 3
    radar.fixed_angle = {"data": np.array([0.5, 1.5, 2.4])}
    radar.latitude = {"data": np.array([35.3331])}
    radar.longitude = {"data": np.array([-97.2778])}
    radar.instrument_parameters = {
        "nyquist_velocity": {"data": np.full(1080, 26.2)}
    }

    def get_start_end(sweep_index):
        start = sweep_index * 360
        end = start + 359
        return (start, end)

    radar.get_start_end.side_effect = get_start_end

    velocity_data = np.random.uniform(-30, 30, (1080, 1832)).astype(np.float32)
    radar.fields = {
        "reflectivity": {"data": np.ma.array(np.zeros((1080, 1832)), mask=False)},
        "velocity": {"data": np.ma.array(velocity_data, mask=False)},
    }
    radar.azimuth = {"data": np.tile(np.linspace(0, 359, 360), 3)}
    radar.range = {"data": np.linspace(0, 459750, 1832)}
    return radar


def test_extract_velocity_returns_velocity_data():
    radar = _make_mock_radar_with_velocity()
    result = extract_velocity(radar)
    assert isinstance(result, VelocityData)
    assert len(result.sweeps) > 0
    assert result.radar_lat == 35.3331
    assert result.radar_lon == -97.2778


def test_extract_velocity_respects_max_sweeps():
    radar = _make_mock_radar_with_velocity()
    result = extract_velocity(radar, max_sweeps=2)
    assert len(result.sweeps) == 2


def test_extract_velocity_extracts_all_available_when_fewer_than_max():
    radar = _make_mock_radar_with_velocity()
    result = extract_velocity(radar, max_sweeps=5)
    assert len(result.sweeps) == 3


def test_extract_velocity_sweep_has_correct_fields():
    radar = _make_mock_radar_with_velocity()
    result = extract_velocity(radar, max_sweeps=1)
    sweep = result.sweeps[0]
    assert isinstance(sweep, VelocitySweep)
    assert sweep.velocity.shape == (360, 1832)
    assert sweep.elevation_angle == 0.5
    assert sweep.nyquist_velocity > 0
    assert len(sweep.azimuths) == 360
    assert len(sweep.ranges_m) == 1832


def test_extract_velocity_returns_none_when_no_velocity_field():
    radar = _make_mock_radar_with_velocity()
    del radar.fields["velocity"]
    result = extract_velocity(radar)
    assert result is None


def test_extract_velocity_fills_masked_values_with_nan():
    radar = _make_mock_radar_with_velocity()
    velocity_array = np.ma.array(
        np.ones((1080, 1832)) * 10.0,
        mask=np.zeros((1080, 1832), dtype=bool),
    )
    velocity_array.mask[0, 0] = True
    radar.fields["velocity"] = {"data": velocity_array}
    result = extract_velocity(radar, max_sweeps=1)
    assert np.isnan(result.sweeps[0].velocity[0, 0])
    assert not np.isnan(result.sweeps[0].velocity[0, 1])
