from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from src.buffer import TrackingSnapshot
from src.detection import DetectedObject, IntensityLayerData
from src.history.records import dumps, from_jsonable, loads, to_jsonable
from src.preprocess import ScanQuality
from src.tracker import StormTracker
from src.velocity import RotationSignature
from tests.unit.test_tracker import _make_object, _make_scan


def _rotation() -> RotationSignature:
    return RotationSignature(
        centroid_lat=35.1, centroid_lon=-97.4, distance_km=30.0, bearing_deg=200.0,
        max_shear_ms=22.0, max_inbound_ms=-11.0, max_outbound_ms=11.0, diameter_km=2.5,
        sweep_count=2, elevation_angles=[0.5, 0.9], strength="moderate",
        associated_object_id=3,
    )


def test_detected_object_with_open_ended_band_and_rotation_round_trips():
    obj = DetectedObject(
        object_id=3, centroid_lat=35.1, centroid_lon=-97.4, distance_km=30.0,
        bearing_deg=200.0, peak_dbz=62.5, peak_label="severe core", area_km2=120.0,
        layers=[
            IntensityLayerData("heavy precipitation", 40.0, 50.0, 80.0),
            IntensityLayerData("severe core", 60.0, float("inf"), 4.0),
        ],
        rotation=_rotation(),
        class_fractions={"precipitation": 0.9, "hail": 0.1},
        temporal_status="persistent",
    )
    assert loads(dumps(obj)) == obj


def test_scan_quality_and_band_evidence_keys_round_trip():
    quality = ScanQuality(
        score=0.8, finite_fraction=0.3, removed_speckle_pixels=np.int64(12),
        removed_speckle_fraction=0.001, flags=["ok"], class_fractions={"precipitation": 1.0},
    )
    evidence = {(60.0, float("inf")): {"median_rhohv": 0.98, "class_fractions": {}}}
    restored_quality, restored_evidence = loads(dumps([quality, evidence]))
    assert restored_quality == quality
    assert isinstance(restored_quality.removed_speckle_pixels, int)
    assert restored_evidence == evidence


def test_aware_datetimes_and_tuples_round_trip():
    value = {"ref": (datetime(2026, 9, 12, 18, 0, tzinfo=timezone.utc), 4)}
    restored = loads(dumps(value))
    assert restored == value
    assert isinstance(restored["ref"], tuple)


def test_dicts_that_look_like_markers_are_not_misread():
    value = {"__datetime__": "not a date", "__type__": "Track"}
    assert loads(dumps(value)) == value


def test_tracker_tracks_and_snapshot_round_trip_exactly():
    tracker = StormTracker()
    t1 = datetime(2026, 9, 12, 18, 0)
    for minutes in (0, 5, 10):
        tracker.update(_make_scan(
            "KTLX", t1 + timedelta(minutes=minutes),
            [_make_object(1, 35.3, -97.3 + minutes * 0.01), _make_object(2, 35.0, -97.0)],
        ))
    snapshot = tracker.snapshot()
    restored = loads(dumps(snapshot))
    assert isinstance(restored, TrackingSnapshot)
    assert restored == snapshot
    assert dumps(restored) == dumps(snapshot)


def test_unregistered_types_are_rejected_both_ways():
    class NotRegistered:
        pass

    with pytest.raises(TypeError):
        to_jsonable(NotRegistered())
    with pytest.raises(ValueError):
        from_jsonable({"__type__": "os.system", "fields": {}})
    with pytest.raises(ValueError):
        from_jsonable({"unexpected": 1})
