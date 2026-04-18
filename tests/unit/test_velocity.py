import numpy as np
from src.parser import VelocitySweep, VelocityData
from src.velocity import VelocityRegion, detect_velocity_regions, RotationSignature, detect_rotation_signatures

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


def test_detect_rotation_finds_shear_couplet():
    grid = np.full((360, 500), np.nan)
    # inbound block adjacent to outbound block — classic couplet
    grid[50:60, 150:160] = -20.0  # inbound
    grid[60:70, 150:160] = 20.0   # outbound
    vel_data = _make_velocity_data([_make_sweep(grid)])
    signatures = detect_rotation_signatures(vel_data)
    assert len(signatures) >= 1
    assert signatures[0].max_shear_ms >= 15.0


def test_detect_rotation_classifies_strength_weak():
    grid = np.full((360, 500), np.nan)
    grid[50:60, 150:160] = -10.0
    grid[60:70, 150:160] = 10.0   # 20 m/s shear — weak
    vel_data = _make_velocity_data([_make_sweep(grid)])
    signatures = detect_rotation_signatures(vel_data)
    assert len(signatures) >= 1
    assert signatures[0].strength == "weak"


def test_detect_rotation_classifies_strength_moderate():
    grid = np.full((360, 500), np.nan)
    grid[50:60, 150:160] = -15.0
    grid[60:70, 150:160] = 15.0   # 30 m/s shear — moderate
    vel_data = _make_velocity_data([_make_sweep(grid)])
    signatures = detect_rotation_signatures(vel_data)
    assert len(signatures) >= 1
    assert signatures[0].strength == "moderate"


def test_detect_rotation_classifies_strength_strong():
    grid = np.full((360, 500), np.nan)
    grid[50:60, 150:160] = -20.0
    grid[60:70, 150:160] = 20.0   # 40 m/s shear — strong
    vel_data = _make_velocity_data([_make_sweep(grid)])
    signatures = detect_rotation_signatures(vel_data)
    assert len(signatures) >= 1
    assert signatures[0].strength == "strong"


def test_detect_rotation_ignores_weak_shear():
    grid = np.full((360, 500), np.nan)
    grid[50:60, 150:160] = -5.0
    grid[60:70, 150:160] = 5.0    # 10 m/s — below threshold
    vel_data = _make_velocity_data([_make_sweep(grid)])
    signatures = detect_rotation_signatures(vel_data)
    assert len(signatures) == 0


def test_detect_rotation_returns_empty_for_no_velocity():
    grid = np.full((360, 500), np.nan)
    vel_data = _make_velocity_data([_make_sweep(grid)])
    signatures = detect_rotation_signatures(vel_data)
    assert signatures == []


def test_detect_rotation_multi_sweep_increases_sweep_count():
    grid1 = np.full((360, 500), np.nan)
    grid1[50:60, 150:160] = -20.0
    grid1[60:70, 150:160] = 20.0
    grid2 = np.full((360, 500), np.nan)
    grid2[50:60, 150:160] = -22.0
    grid2[60:70, 150:160] = 22.0
    vel_data = _make_velocity_data([
        _make_sweep(grid1, elevation=0.5),
        _make_sweep(grid2, elevation=1.5),
    ])
    signatures = detect_rotation_signatures(vel_data)
    assert len(signatures) >= 1
    assert signatures[0].sweep_count == 2


from src.velocity import analyze_velocity
from src.detection import DetectedObject


def _make_object(object_id, lat, lon, distance_km, bearing_deg) -> DetectedObject:
    return DetectedObject(
        object_id=object_id,
        centroid_lat=lat,
        centroid_lon=lon,
        distance_km=distance_km,
        bearing_deg=bearing_deg,
        peak_dbz=55.0,
        peak_label="intense rain",
        area_km2=50.0,
    )


def test_analyze_velocity_associates_rotation_with_object():
    grid = np.full((360, 500), np.nan)
    grid[50:60, 150:160] = -20.0
    grid[60:70, 150:160] = 20.0
    vel_data = _make_velocity_data([_make_sweep(grid)])
    # Place an object at roughly the same location as the shear couplet
    azimuths = np.linspace(0, 359, 360)
    ranges_m = np.linspace(2000, 230000, 500)
    centroid_az = float(azimuths[60])
    centroid_range = float(ranges_m[155])
    from src.detection import polar_to_latlon
    lat, lon = polar_to_latlon(RADAR_LAT, RADAR_LON, centroid_az, centroid_range)
    obj = _make_object(1, round(lat, 4), round(lon, 4), round(centroid_range / 1000, 1), round(centroid_az, 1))
    objects = [obj]
    regions, rotations, annotated = analyze_velocity(vel_data, objects)
    assert annotated[0].rotation is not None
    assert annotated[0].rotation.strength in {"weak", "moderate", "strong"}


def test_analyze_velocity_associates_inbound_with_object():
    grid = np.full((360, 500), np.nan)
    grid[50:70, 100:130] = -20.0
    vel_data = _make_velocity_data([_make_sweep(grid)])
    azimuths = np.linspace(0, 359, 360)
    ranges_m = np.linspace(2000, 230000, 500)
    centroid_az = float(azimuths[60])
    centroid_range = float(ranges_m[115])
    from src.detection import polar_to_latlon
    lat, lon = polar_to_latlon(RADAR_LAT, RADAR_LON, centroid_az, centroid_range)
    obj = _make_object(1, round(lat, 4), round(lon, 4), round(centroid_range / 1000, 1), round(centroid_az, 1))
    objects = [obj]
    regions, rotations, annotated = analyze_velocity(vel_data, objects)
    assert annotated[0].max_inbound_ms is not None
    assert annotated[0].max_inbound_ms <= -10.0


def test_analyze_velocity_leaves_distant_objects_unannotated():
    grid = np.full((360, 500), np.nan)
    grid[50:60, 150:160] = -20.0
    grid[60:70, 150:160] = 20.0
    vel_data = _make_velocity_data([_make_sweep(grid)])
    # Place object far from the couplet
    obj = _make_object(1, 40.0, -90.0, 200.0, 180.0)
    objects = [obj]
    regions, rotations, annotated = analyze_velocity(vel_data, objects)
    assert annotated[0].rotation is None
    assert annotated[0].max_inbound_ms is None
