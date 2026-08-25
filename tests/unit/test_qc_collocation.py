import numpy as np

from src.qc.collocation import rotation_proximity_mask
from src.velocity import RotationSignature


def _sweep():
    from src.parser import SweepData

    shape = (72, 60)
    return SweepData(
        reflectivity=np.full(shape, 40.0),
        azimuths=np.linspace(0.0, 355.0, shape[0]),
        ranges_m=np.arange(2125.0, 2125.0 + 250.0 * shape[1], 250.0),
        elevation_angle=0.5,
        elevation_angles=[0.5],
        elevations=np.full(shape[0], 0.5),
        radar_lat=35.3331,
        radar_lon=-97.2778,
        radar_alt_m=390.0,
        timestamp="2026-08-23T00:00:00Z",
    )


def _signature(lat: float, lon: float, diameter_km: float = 3.0) -> RotationSignature:
    return RotationSignature(
        centroid_lat=lat,
        centroid_lon=lon,
        distance_km=10.0,
        bearing_deg=0.0,
        max_shear_ms=30.0,
        max_inbound_ms=-20.0,
        max_outbound_ms=20.0,
        diameter_km=diameter_km,
        sweep_count=2,
        elevation_angles=[0.48, 0.88],
        strength="moderate",
    )


def test_no_signatures_gives_empty_mask():
    mask = rotation_proximity_mask(_sweep(), [])
    assert mask.shape == (72, 60)
    assert not mask.any()


def test_gates_near_signature_are_marked():
    sweep = _sweep()
    from src.geometry import gate_coordinates

    lat, lon = gate_coordinates(
        sweep.azimuths, sweep.ranges_m, sweep.elevation_angle,
        sweep.radar_lat, sweep.radar_lon,
    )
    target_lat, target_lon = float(lat[10, 30]), float(lon[10, 30])
    mask = rotation_proximity_mask(sweep, [_signature(target_lat, target_lon)])
    assert mask[10, 30]


def test_distant_gates_are_not_marked():
    sweep = _sweep()
    mask = rotation_proximity_mask(sweep, [_signature(40.0, -100.0)])
    assert not mask.any()


def test_margin_widens_the_mask():
    sweep = _sweep()
    from src.geometry import gate_coordinates

    lat, lon = gate_coordinates(
        sweep.azimuths, sweep.ranges_m, sweep.elevation_angle,
        sweep.radar_lat, sweep.radar_lon,
    )
    signature = _signature(float(lat[10, 30]), float(lon[10, 30]))
    narrow = rotation_proximity_mask(sweep, [signature], margin_km=0.0)
    wide = rotation_proximity_mask(sweep, [signature], margin_km=20.0)
    assert wide.sum() > narrow.sum()
