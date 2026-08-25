import numpy as np

from src.geometry import gate_coordinates
from src.qc.collocation import rotation_proximity_mask, _haversine_km
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


def test_collocation_radius_applied_correctly():
    """Verify that proximity radius (diameter/2 + margin) is applied correctly.

    This test distinguishes between implementations that:
    - Always return False (would fail here)
    - Always return True (would fail here)
    - Correctly apply the distance and radius logic (would pass)
    """
    sweep = _sweep()

    lat, lon = gate_coordinates(
        sweep.azimuths, sweep.ranges_m, sweep.elevation_angle,
        sweep.radar_lat, sweep.radar_lon,
    )

    # Test with a very small margin that shouldn't mark many gates
    target_lat, target_lon = float(lat[10, 30]), float(lon[10, 30])
    signature = _signature(target_lat, target_lon, diameter_km=0.5)

    # With minimal margin, only the gate at/near the centroid should be marked
    tight = rotation_proximity_mask(sweep, [signature], margin_km=0.01)

    # With generous margin, many gates should be marked
    loose = rotation_proximity_mask(sweep, [signature], margin_km=50.0)

    # Both should have the target gate marked
    assert tight[10, 30]
    assert loose[10, 30]

    # Loose should mark significantly more gates than tight
    assert loose.sum() > tight.sum() * 2


def test_collocation_radius_is_half_diameter_plus_margin():
    """Pin the radius arithmetic, not merely its monotonicity.

    A multiplicative radius (diameter * margin) or a margin-only radius
    (diameter ignored) both satisfy the widening inequality tests, so those
    cannot catch a wrong formula. This asserts the actual boundary: every
    marked gate lies within diameter/2 + margin, and every unmarked gate lies
    outside it.
    """
    sweep = _sweep()
    lat, lon = gate_coordinates(
        sweep.azimuths, sweep.ranges_m, sweep.elevation_angle,
        sweep.radar_lat, sweep.radar_lon,
    )
    signature = _signature(float(lat[10, 30]), float(lon[10, 30]), diameter_km=3.0)
    margin_km = 5.0
    expected_radius_km = signature.diameter_km / 2.0 + margin_km

    mask = rotation_proximity_mask(sweep, [signature], margin_km=margin_km)
    distance_km = _haversine_km(lat, lon, signature.centroid_lat, signature.centroid_lon)

    assert mask.any() and not mask.all()
    assert distance_km[mask].max() <= expected_radius_km
    assert distance_km[~mask].min() > expected_radius_km


def test_multi_signature_union_respects_individual_diameters():
    """Multi-signature masks are unions, each using its own diameter.

    When multiple rotation signatures are provided, each contributes its own
    region (based on its diameter and the shared margin), and the result is
    their union. Regions do not bleed into each other.
    """
    sweep = _sweep()
    lat, lon = gate_coordinates(
        sweep.azimuths, sweep.ranges_m, sweep.elevation_angle,
        sweep.radar_lat, sweep.radar_lon,
    )

    # Two signatures at well-separated locations with different diameters
    sig1_lat, sig1_lon = float(lat[10, 20]), float(lon[10, 20])
    sig2_lat, sig2_lon = float(lat[50, 40]), float(lon[50, 40])

    sig1 = _signature(sig1_lat, sig1_lon, diameter_km=2.0)
    sig2 = _signature(sig2_lat, sig2_lon, diameter_km=4.0)

    margin_km = 1.0

    # Get masks for each signature individually
    mask1 = rotation_proximity_mask(sweep, [sig1], margin_km=margin_km)
    mask2 = rotation_proximity_mask(sweep, [sig2], margin_km=margin_km)

    # Get combined mask
    mask_combined = rotation_proximity_mask(sweep, [sig1, sig2], margin_km=margin_km)

    # Combined mask should be the union of individual masks
    assert np.array_equal(mask_combined, mask1 | mask2)

    # Both regions should be marked in the combined mask
    assert mask_combined[10, 20]
    assert mask_combined[50, 40]

    # Verify each signature's radius is independent
    dist1 = _haversine_km(lat, lon, sig1_lat, sig1_lon)
    dist2 = _haversine_km(lat, lon, sig2_lat, sig2_lon)
    radius1 = sig1.diameter_km / 2.0 + margin_km
    radius2 = sig2.diameter_km / 2.0 + margin_km

    # Each region respects its own radius
    assert dist1[mask1].max() <= radius1
    assert dist2[mask2].max() <= radius2
