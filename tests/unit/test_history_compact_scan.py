from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from src.detection import IntensityLayerData
from src.history.compact_scan import CompactScan, LabelMaskView, timestamp_key, verify_label_masks
from src.preprocess import ScanQuality
from src.tracker import StormTracker
from tests.unit.test_tracker import _make_object, _make_scan

T0 = datetime(2026, 9, 12, 18, 0)


def _scan_with_content():
    objects = [_make_object(1, 35.3, -97.3, 55.0), _make_object(2, 35.0, -97.0, 40.0)]
    objects[0].layers = [IntensityLayerData("severe core", 60.0, float("inf"), 4.0)]
    scan = _make_scan("KTLX", T0, objects)
    scan.reflectivity_data.reflectivity[85:95, 195:205] = 55.5
    scan.reflectivity_data.reflectivity[125:135, 195:205] = 12.0
    scan.scan_quality = ScanQuality(
        score=0.9, finite_fraction=0.01, removed_speckle_pixels=0,
        removed_speckle_fraction=0.0, flags=[],
    )
    scan.precipitation_band_evidence = {(15.0, 20.0): {"median_rhohv": None, "class_fractions": {}}}
    scan.source_path = "C:/cache/KTLX/KTLX20260912_180000_V06"
    return scan


def test_round_trip_restores_every_field_the_pipeline_reads():
    scan = _scan_with_content()
    rebuilt = CompactScan.from_buffered_scan(scan).to_buffered_scan()

    assert rebuilt.timestamp == scan.timestamp
    assert rebuilt.site_id == scan.site_id
    assert rebuilt.source_path == scan.source_path
    sweep, original = rebuilt.reflectivity_data, scan.reflectivity_data
    for name in ("reflectivity", "azimuths", "ranges_m", "elevations"):
        assert np.array_equal(getattr(sweep, name), getattr(original, name), equal_nan=True), name
        assert getattr(sweep, name).dtype == getattr(original, name).dtype, name
    for name in ("elevation_angle", "elevation_angles", "radar_lat", "radar_lon", "radar_alt_m", "timestamp"):
        assert getattr(sweep, name) == getattr(original, name), name
    assert sweep.rhohv is None and sweep.zdr is None and sweep.gate_classification is None
    assert np.array_equal(rebuilt.labeled_grid, scan.labeled_grid)
    assert rebuilt.labeled_grid.dtype == scan.labeled_grid.dtype
    assert set(rebuilt.object_masks) == set(scan.object_masks)
    for object_id, mask in scan.object_masks.items():
        assert np.array_equal(rebuilt.object_masks[object_id], mask)
    assert rebuilt.detected_objects == scan.detected_objects
    assert rebuilt.scan_quality == scan.scan_quality
    assert rebuilt.precipitation_band_evidence == scan.precipitation_band_evidence
    assert rebuilt.tracking is None


def test_rebuilt_objects_are_fresh_copies():
    compact = CompactScan.from_buffered_scan(_scan_with_content())
    first = compact.to_buffered_scan()
    first.detected_objects[0].temporal_status = "mutated"
    assert compact.to_buffered_scan().detected_objects[0].temporal_status != "mutated"


def test_tracking_snapshot_round_trips_and_sets_tracked_flag():
    tracker = StormTracker()
    scan = _scan_with_content()
    tracker.update(scan)
    scan.tracking = tracker.snapshot()
    compact = CompactScan.from_buffered_scan(scan)
    assert compact.tracked is True
    assert compact.to_buffered_scan().tracking == scan.tracking
    untracked = compact.with_tracking(None)
    assert untracked.tracked is False and untracked.to_buffered_scan().tracking is None


def test_packing_refuses_masks_the_label_grid_cannot_represent():
    scan = _scan_with_content()
    overlap = scan.object_masks[1].copy()
    overlap[125:135, 195:205] = True  # also claims object 2's gates
    scan.object_masks[1] = overlap
    with pytest.raises(ValueError, match="object 1"):
        verify_label_masks(scan)
    with pytest.raises(ValueError, match="object 1"):
        CompactScan.from_buffered_scan(scan)


def test_label_mask_view_behaves_like_the_mask_dict():
    labels = np.array([[0, 1], [2, 2]], dtype=np.int32)
    view = LabelMaskView(labels, [1, 2])
    assert list(view) == [1, 2] and len(view) == 2
    assert 2 in view and np.int64(2) in view and 3 not in view
    assert np.array_equal(view[2], np.array([[False, False], [True, True]]))
    assert view.get(3) is None
    assert dict(view.items()).keys() == {1, 2}
    with pytest.raises(KeyError):
        view[3]


def test_compact_scan_is_small_and_metadata_is_exposed():
    compact = CompactScan.from_buffered_scan(_scan_with_content())
    assert compact.object_count == 2
    assert compact.scan_timestamp_text == T0.isoformat()
    assert compact.nbytes < 50_000
    compact.validate()


def test_timestamp_key_normalizes_to_naive_utc():
    aware = datetime(2026, 9, 12, 13, 0, tzinfo=timezone(timedelta(hours=-5)))
    assert timestamp_key(aware) == T0
    assert timestamp_key(T0) == T0


def test_velocity_regions_round_trip_and_older_records_without_them_load_empty():
    import zlib

    from src.history import records as record_codec
    from src.velocity import VelocityRegion

    scan = _scan_with_content()
    scan.velocity_regions = [VelocityRegion(
        region_type="inbound", peak_velocity_ms=-24.5, mean_velocity_ms=-12.0, area_km2=40.5,
        centroid_lat=35.4, centroid_lon=-97.1, distance_km=22.0, bearing_deg=80.0,
        sweep_count=2, elevation_angles=[0.5, 0.9],
    )]
    compact = CompactScan.from_buffered_scan(scan)
    assert compact.to_buffered_scan().velocity_regions == scan.velocity_regions

    older = compact.read_records()
    del older["velocity_regions"]
    legacy = CompactScan(**{**compact.__dict__, "records": zlib.compress(record_codec.dumps(older).encode("utf-8"))})
    assert legacy.to_buffered_scan().velocity_regions == []
