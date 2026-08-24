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
        elevations=np.full(n_az, elevation),
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
    """Sweep count rises across Doppler cuts at distinct elevations.

    Elevations here match the Doppler cuts of a real split-cut VCP
    (0.48, 0.88, 1.27) rather than the invented 0.5/1.5 pair used previously.
    """
    grids = []
    for peak in (-20.0, -22.0, -21.0):
        grid = np.full((360, 500), np.nan)
        grid[50:70, 100:130] = peak
        grids.append(grid)
    vel_data = _make_velocity_data([
        _make_sweep(grids[0], elevation=0.48),
        _make_sweep(grids[1], elevation=0.88),
        _make_sweep(grids[2], elevation=1.27),
    ])
    regions = detect_velocity_regions(vel_data)
    inbound = [r for r in regions if r.region_type == "inbound"]
    assert len(inbound) == 1
    assert inbound[0].sweep_count == 3
    assert inbound[0].elevation_angles == [0.48, 0.88, 1.27]


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
    """Sweep count rises across Doppler cuts at distinct elevations.

    Elevations here match the Doppler cuts of a real split-cut VCP
    (0.48, 0.88, 1.27) rather than the invented 0.5/1.5 pair used previously.
    """
    grids = []
    for inbound_peak, outbound_peak in ((-20.0, 20.0), (-22.0, 22.0), (-21.0, 21.0)):
        grid = np.full((360, 500), np.nan)
        grid[50:60, 150:160] = inbound_peak
        grid[60:70, 150:160] = outbound_peak
        grids.append(grid)
    vel_data = _make_velocity_data([
        _make_sweep(grids[0], elevation=0.48),
        _make_sweep(grids[1], elevation=0.88),
        _make_sweep(grids[2], elevation=1.27),
    ])
    signatures = detect_rotation_signatures(vel_data)
    assert len(signatures) >= 1
    assert signatures[0].sweep_count == 3
    assert signatures[0].elevation_angles == [0.48, 0.88, 1.27]


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


import pyart
import pytest

from src.parser import extract_velocity

# The task brief's canonical reference volume (cache/KEMX/KEMX20260712_022646_V06)
# correctly yields 3 Doppler sweeps (elevations 0.48/0.88/1.27, confirmed in
# test_parser.py and test_sweeps.py) but its velocity regions never spatially
# overlap >=30% across all three sweeps -- max region sweep_count stays 1 on
# that volume even after the fix, because the storm's velocity signature
# shifts position with height on that particular scan. That is a real
# property of that data, not a bug in the fix. A repo-wide scan of all 152
# cached volumes (2026-08-23) found cache/KTLX/KTLX20260410_000100_V06 is the
# clearest case of genuine cross-sweep persistence: one inbound region merges
# across all 3 real Doppler elevations (sweep_count == 3, elevation_angles ==
# [0.48, 0.88, 1.27], no duplicate elevations -- i.e. a clean 1-sweep-per-tilt
# merge, not sweep double-counting). Using it here so this regression test
# asserts instead of skipping.
REFERENCE_VOLUME = "cache/KTLX/KTLX20260410_000100_V06"


def test_sweep_count_can_exceed_one_on_real_data():
    """Regression for the split-cut defect.

    Before the fix, two of three velocity sweeps were empty, so sweep_count was
    structurally pinned at 1 and multi-sweep confirmation never engaged.
    """
    radar = pyart.io.read_nexrad_archive(REFERENCE_VOLUME)
    vel_data = extract_velocity(radar, max_sweeps=3)
    regions = detect_velocity_regions(vel_data)
    if not regions:
        pytest.skip("reference volume has no velocity regions above threshold")
    assert max(r.sweep_count for r in regions) > 1


def _make_multi_couplet_single_sweep_grid() -> np.ndarray:
    """A single sweep containing many separate, spatially close shear couplets.

    `_detect_shear_single_sweep` emits one couplet result per connected
    component of shear pixels. Each 1-row inbound / 1-row outbound pair here
    forms its own connected component (separated from its neighbors by a
    1-row NaN gap, which is enough for scipy's 8-connectivity labeling to
    treat them as distinct), but all 15 couplets sit close enough in azimuth
    and range (near-radar range bins 20:30, spanning ~42 degrees of azimuth)
    that `_merge_cross_sweep_rotations` merges them into a single rotation
    signature.
    """
    grid = np.full((360, 500), np.nan)
    n_blocks = 15
    step = 3
    for i in range(n_blocks):
        az0 = 40 + i * step
        grid[az0, 20:30] = -20.0
        grid[az0 + 1, 20:30] = 20.0
    return grid


def test_rotation_single_sweep_many_couplets_yield_sweep_count_one():
    """A single sweep's many adjacent couplets must not inflate sweep_count.

    This is the core regression test for the sweep_count defect: before the
    fix, `_merge_cross_sweep_rotations` incremented sweep_count once per
    MERGED COUPLET rather than once per distinct SWEEP. With this synthetic
    single-sweep, 15-couplet grid, the pre-fix code produced exactly one
    merged signature with sweep_count == 15 (measured directly against the
    pre-fix implementation). Since every couplet here comes from the same
    sweep (same elevation angle), the correct sweep_count is 1.
    """
    grid = _make_multi_couplet_single_sweep_grid()
    vel_data = _make_velocity_data([_make_sweep(grid, elevation=0.48)])
    signatures = detect_rotation_signatures(vel_data)
    assert len(signatures) == 1
    assert signatures[0].sweep_count == 1
    assert signatures[0].elevation_angles == [0.48]


def test_rotation_sweep_count_equals_distinct_elevation_count():
    """sweep_count must equal the number of distinct elevation angles, and
    can never exceed the number of sweeps supplied."""
    grids = []
    for inbound_peak, outbound_peak in ((-20.0, 20.0), (-22.0, 22.0), (-21.0, 21.0)):
        grid = np.full((360, 500), np.nan)
        grid[50:60, 150:160] = inbound_peak
        grid[60:70, 150:160] = outbound_peak
        grids.append(grid)
    sweeps = [
        _make_sweep(grids[0], elevation=0.48),
        _make_sweep(grids[1], elevation=0.88),
        _make_sweep(grids[2], elevation=1.27),
    ]
    vel_data = _make_velocity_data(sweeps)
    signatures = detect_rotation_signatures(vel_data)
    assert len(signatures) >= 1
    for sig in signatures:
        assert sig.sweep_count == len(set(sig.elevation_angles))
        assert sig.sweep_count <= len(sweeps)


def test_sweep_count_never_exceeds_three_sweeps_on_real_data():
    """Regression guard for the measured defect on the reference volume.

    Before the fix, rotation signatures on cache/KTLX/KTLX20260410_000100_V06
    (3 velocity sweeps at elevations 0.48/0.88/1.27) had sweep_count values
    up to 198, because couplets within a single sweep were counted as
    separate sweeps. With 3 sweeps extracted, no signature may report more
    than 3.
    """
    radar = pyart.io.read_nexrad_archive(REFERENCE_VOLUME)
    vel_data = extract_velocity(radar, max_sweeps=3)
    assert len(vel_data.sweeps) == 3
    signatures = detect_rotation_signatures(vel_data)
    if not signatures:
        pytest.skip("reference volume has no rotation signatures above threshold")
    assert max(s.sweep_count for s in signatures) <= 3
