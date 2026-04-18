import numpy as np
from src.parser import VelocitySweep, VelocityData
from src.velocity import VelocityRegion, detect_velocity_regions

RADAR_LAT = 35.3331
RADAR_LON = -97.2778


def _make_sweep(velocity_grid: np.ndarray, elevation: float = 0.5) -> VelocitySweep:
    n_az, n_rng = velocity_grid.shape
    return VelocitySweep(
        velocity=velocity_grid,
        azimuths=np.linspace(0, 359, n_az),
        ranges_m=np.linspace(2000, 230000, n_rng),
        elevation_angle=elevation,
        nyquist_velocity=26.2,
    )


def _make_velocity_data(sweeps: list[VelocitySweep]) -> VelocityData:
    return VelocityData(sweeps=sweeps, radar_lat=RADAR_LAT, radar_lon=RADAR_LON)


def test_detect_velocity_regions_finds_inbound():
    grid = np.full((360, 500), np.nan)
    grid[50:70, 100:130] = -20.0  # inbound block
    vel_data = _make_velocity_data([_make_sweep(grid)])
    regions = detect_velocity_regions(vel_data)
    inbound = [r for r in regions if r.region_type == "inbound"]
    assert len(inbound) == 1
    assert inbound[0].peak_velocity_ms <= -20.0


def test_detect_velocity_regions_finds_outbound():
    grid = np.full((360, 500), np.nan)
    grid[100:120, 200:230] = 25.0  # outbound block
    vel_data = _make_velocity_data([_make_sweep(grid)])
    regions = detect_velocity_regions(vel_data)
    outbound = [r for r in regions if r.region_type == "outbound"]
    assert len(outbound) == 1
    assert outbound[0].peak_velocity_ms >= 25.0


def test_detect_velocity_regions_ignores_weak_velocity():
    grid = np.full((360, 500), np.nan)
    grid[50:70, 100:130] = -5.0  # below 10 m/s threshold
    vel_data = _make_velocity_data([_make_sweep(grid)])
    regions = detect_velocity_regions(vel_data)
    assert len(regions) == 0


def test_detect_velocity_regions_filters_small_regions():
    grid = np.full((360, 500), np.nan)
    grid[50, 100] = -20.0  # single pixel — too small
    vel_data = _make_velocity_data([_make_sweep(grid)])
    regions = detect_velocity_regions(vel_data)
    assert len(regions) == 0


def test_detect_velocity_regions_multi_sweep_increases_sweep_count():
    grid1 = np.full((360, 500), np.nan)
    grid1[50:70, 100:130] = -20.0
    grid2 = np.full((360, 500), np.nan)
    grid2[50:70, 100:130] = -22.0  # same location, second sweep
    vel_data = _make_velocity_data([
        _make_sweep(grid1, elevation=0.5),
        _make_sweep(grid2, elevation=1.5),
    ])
    regions = detect_velocity_regions(vel_data)
    inbound = [r for r in regions if r.region_type == "inbound"]
    assert len(inbound) == 1
    assert inbound[0].sweep_count == 2


def test_detect_velocity_regions_returns_empty_for_all_nan():
    grid = np.full((360, 500), np.nan)
    vel_data = _make_velocity_data([_make_sweep(grid)])
    regions = detect_velocity_regions(vel_data)
    assert regions == []
