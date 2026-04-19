from datetime import datetime
from src.tracking.types import Track, RotationHistoryEntry
from src.velocity import RotationSignature


def test_track_rotation_history_starts_empty():
    track = Track(track_id=1, status="active")
    assert track.rotation_history == []


def test_rotation_history_entry_stores_signature():
    entry = RotationHistoryEntry(
        timestamp=datetime(2026, 4, 10, 21, 0),
        rotation=RotationSignature(
            centroid_lat=35.5, centroid_lon=-97.0,
            distance_km=50.0, bearing_deg=90.0,
            max_shear_ms=30.0, max_inbound_ms=-20.0, max_outbound_ms=15.0,
            diameter_km=3.0, sweep_count=2, elevation_angles=[0.5, 1.5],
            strength="moderate",
        ),
    )
    assert entry.rotation.strength == "moderate"


def test_rotation_history_entry_stores_none_for_no_rotation():
    entry = RotationHistoryEntry(
        timestamp=datetime(2026, 4, 10, 21, 0),
        rotation=None,
    )
    assert entry.rotation is None
