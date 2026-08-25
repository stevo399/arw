import numpy as np

from src.parser import SweepData
from src.qc.protection import PROTECTED_MIN_DBZ, protected_mask


def _sweep(reflectivity: np.ndarray) -> SweepData:
    n_az, n_rng = reflectivity.shape
    return SweepData(
        reflectivity=reflectivity,
        azimuths=np.linspace(0.0, 355.0, n_az),
        ranges_m=np.arange(2125.0, 2125.0 + 250.0 * n_rng, 250.0),
        elevation_angle=0.5,
        elevations=np.full(n_az, 0.5),
        elevation_angles=[0.5],
        radar_lat=35.3331,
        radar_lon=-97.2778,
        radar_alt_m=390.0,
        timestamp="2026-08-23T00:00:00Z",
    )


def test_intense_gates_are_protected():
    reflectivity = np.full((36, 40), 20.0)
    reflectivity[10, 10] = PROTECTED_MIN_DBZ + 5.0
    mask = protected_mask(_sweep(reflectivity), [])
    assert mask[10, 10]


def test_weak_isolated_gates_are_not_protected():
    reflectivity = np.full((36, 40), 20.0)
    mask = protected_mask(_sweep(reflectivity), [])
    assert not mask.any()


def test_rule_three_protects_the_whole_connected_component():
    """A storm containing an intense core must be protected entirely, so QC
    cannot punch a hole in it or amputate its edge."""
    reflectivity = np.full((36, 40), np.nan)
    reflectivity[8:16, 8:16] = 30.0       # storm body, individually unprotected
    reflectivity[11, 11] = 58.0            # intense core
    mask = protected_mask(_sweep(reflectivity), [])
    assert mask[8:16, 8:16].all(), "protection did not spread to the whole storm"


def test_component_without_a_core_is_not_protected():
    reflectivity = np.full((36, 40), np.nan)
    reflectivity[8:16, 8:16] = 30.0
    mask = protected_mask(_sweep(reflectivity), [])
    assert not mask.any()


def test_separate_components_are_independent():
    reflectivity = np.full((36, 40), np.nan)
    reflectivity[2:6, 2:6] = 30.0          # no core
    reflectivity[20:26, 20:26] = 30.0
    reflectivity[22, 22] = 58.0            # core in the second blob only
    mask = protected_mask(_sweep(reflectivity), [])
    assert not mask[2:6, 2:6].any()
    assert mask[20:26, 20:26].all()


def test_rotation_collocation_protects_weak_debris_gates():
    """The tornado case: debris is weak enough to look like clutter, and is
    saved only by sitting on top of a rotation couplet."""
    from src.geometry import gate_coordinates
    from src.velocity import RotationSignature

    reflectivity = np.full((72, 60), np.nan)
    reflectivity[10:14, 28:32] = 42.0
    sweep = _sweep(reflectivity)
    lat, lon = gate_coordinates(
        sweep.azimuths, sweep.ranges_m, sweep.elevation_angle,
        sweep.radar_lat, sweep.radar_lon,
    )
    signature = RotationSignature(
        centroid_lat=float(lat[12, 30]),
        centroid_lon=float(lon[12, 30]),
        distance_km=10.0,
        bearing_deg=0.0,
        max_shear_ms=40.0,
        max_inbound_ms=-25.0,
        max_outbound_ms=25.0,
        diameter_km=2.0,
        sweep_count=2,
        elevation_angles=[0.48, 0.88],
        strength="strong",
    )
    mask = protected_mask(sweep, [signature])
    assert mask[10:14, 28:32].all()
