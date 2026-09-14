from datetime import datetime, timedelta
from threading import Event, Thread

import pytest

import src.history.store as store
from src.history.disk import HISTORY_SUFFIX
from src.history.records import dumps
from src.history.store import TRACKER_STATE_FILENAME, HistoryRegistry, RadarHistory
from tests.unit.test_tracker import _make_object, _make_scan

T0 = datetime(2026, 9, 12, 18, 0)


def _scan(minute: int, lon_shift: float = 0.0, site_id: str = "KTLX"):
    return _make_scan(
        site_id,
        T0 + timedelta(minutes=minute),
        [_make_object(1, 35.3, -97.30 + lon_shift), _make_object(2, 35.0, -97.0)],
    )


def _feed(history: RadarHistory, minutes):
    return [history.add_live_scan(_scan(minute, minute * 0.001)) for minute in minutes]


def test_ring_keeps_only_the_newest_scans_in_memory_and_on_disk(tmp_path):
    history = RadarHistory.open("KTLX", tmp_path, ring_size=5)
    _feed(history, [0, 5, 10, 15, 20, 25])
    assert [scan.timestamp for scan in history.scans()] == [
        T0 + timedelta(minutes=m) for m in (5, 10, 15, 20, 25)
    ]
    assert len(list(tmp_path.glob(f"*{HISTORY_SUFFIX}"))) == 5
    assert (tmp_path / TRACKER_STATE_FILENAME).is_file()


def test_every_retained_scan_is_tracked_and_newest_is_live(tmp_path):
    history = RadarHistory.open("KTLX", tmp_path)
    _feed(history, [0, 5, 10])
    assert all(scan.tracked for scan in history.scans())
    newest = history.newest().to_buffered_scan()
    assert [len(track.positions) for track in newest.tracking.active_tracks] == [3, 3]
    assert history.tracker._prev_scan is None, "full previous scan must not be retained"


def test_scan_not_newer_than_the_newest_is_refused_and_not_tracked(tmp_path):
    history = RadarHistory.open("KTLX", tmp_path)
    _feed(history, [0, 5])
    before = dumps(history.tracker.export_state())
    assert history.add_live_scan(_scan(5)) is None
    assert history.add_live_scan(_scan(3)) is None
    assert dumps(history.tracker.export_state()) == before


def test_find_by_timestamp_and_source(tmp_path):
    history = RadarHistory.open("KTLX", tmp_path)
    scan = _scan(0)
    scan.source_path = "C:/cache/KTLX/KTLX20260912_180000_V06"
    history.add_live_scan(scan)
    assert history.find_by_timestamp(T0).timestamp == T0
    assert history.find_by_timestamp(T0 + timedelta(seconds=1)) is None
    assert history.find_by_source(scan.source_path).timestamp == T0


def test_restart_with_saved_state_continues_uninterrupted(tmp_path):
    uninterrupted = RadarHistory.open("KTLX", tmp_path / "a")
    _feed(uninterrupted, [0, 5, 10, 15])

    first = RadarHistory.open("KTLX", tmp_path / "b")
    _feed(first, [0, 5])
    restarted = RadarHistory.open("KTLX", tmp_path / "b")
    _feed(restarted, [10, 15])

    assert dumps(restarted.tracker.export_state()) == dumps(uninterrupted.tracker.export_state())
    assert not restarted.newest().to_buffered_scan().tracking.continuity_rebuilt


def test_stale_tracker_state_triggers_a_flagged_rebuild(tmp_path):
    history = RadarHistory.open("KTLX", tmp_path / "current")
    _feed(history, [0, 5, 10])
    older = RadarHistory.open("KTLX", tmp_path / "older")
    _feed(older, [0, 5])
    # State saved before the newest scan was written.
    (tmp_path / "current" / TRACKER_STATE_FILENAME).write_bytes(
        (tmp_path / "older" / TRACKER_STATE_FILENAME).read_bytes()
    )

    rebuilt = RadarHistory.open("KTLX", tmp_path / "current")

    snapshots = [scan.to_buffered_scan().tracking for scan in rebuilt.scans()]
    assert all(snapshot.continuity_rebuilt for snapshot in snapshots)
    assert [len(t.positions) for t in snapshots[-1].active_tracks] == [3, 3]
    assert rebuilt.tracker.last_scan_timestamp == T0 + timedelta(minutes=10)


@pytest.mark.parametrize("damage", ["missing", "garbage"])
def test_missing_or_unreadable_tracker_state_rebuilds(tmp_path, damage):
    history = RadarHistory.open("KTLX", tmp_path)
    _feed(history, [0, 5])
    state_path = tmp_path / TRACKER_STATE_FILENAME
    if damage == "missing":
        state_path.unlink()
    else:
        state_path.write_text("{not json", encoding="utf-8")
    rebuilt = RadarHistory.open("KTLX", tmp_path)
    assert rebuilt.newest().to_buffered_scan().tracking.continuity_rebuilt


def test_corrupt_snapshot_is_skipped_and_deleted(tmp_path):
    history = RadarHistory.open("KTLX", tmp_path)
    _feed(history, [0, 5, 10])
    damaged = sorted(tmp_path.glob(f"*{HISTORY_SUFFIX}"))[1]
    damaged.write_bytes(b"not an archive")

    reopened = RadarHistory.open("KTLX", tmp_path)

    assert [scan.timestamp for scan in reopened.scans()] == [T0, T0 + timedelta(minutes=10)]
    assert not damaged.exists()


def test_write_failure_is_reported_and_live_tracking_continues(tmp_path, monkeypatch):
    history = RadarHistory.open("KTLX", tmp_path)

    def failing_save(scan, directory):
        raise OSError("disk full")

    monkeypatch.setattr(store, "save_compact_scan", failing_save)
    _feed(history, [0, 5])
    assert history.last_write_error == "OSError: disk full"
    assert len(history.scans()) == 2
    assert [len(t.positions) for t in history.tracker.active_tracks] == [2, 2]


def test_in_memory_history_never_touches_disk(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    history = RadarHistory.open("KTLX", None)
    _feed(history, [0, 5])
    assert len(history.scans()) == 2
    assert list(tmp_path.iterdir()) == []


def test_registry_evicts_least_recently_used_idle_radar_and_reloads_it(tmp_path):
    registry = HistoryRegistry(tmp_path, max_radars=2)
    with registry.use("KTLX") as history:
        history.add_live_scan(_scan(0))
    with registry.use("KEMX"):
        pass
    with registry.use("KIWA"):
        pass
    assert registry.loaded_sites() == ["KEMX", "KIWA"]
    with registry.use("ktlx") as reloaded:
        assert [scan.timestamp for scan in reloaded.scans()] == [T0]


def test_registry_never_evicts_a_radar_in_use(tmp_path):
    registry = HistoryRegistry(tmp_path, max_radars=1)
    holding, release = Event(), Event()

    def hold_ktlx():
        with registry.use("KTLX"):
            holding.set()
            release.wait(2)

    worker = Thread(target=hold_ktlx)
    worker.start()
    try:
        assert holding.wait(2)
        with registry.use("KEMX"):
            # Both radars are in use: neither may be evicted, even though
            # the registry is over its limit until one is released.
            assert registry.loaded_sites() == ["KTLX", "KEMX"]
        assert "KTLX" in registry.loaded_sites()
    finally:
        release.set()
        worker.join(2)


from tests.unit.test_tracking_reacquisition import _scan as _reacquisition_scan
from tests.unit.test_tracking_reacquisition import _storm_a, _storm_b


def test_radar_history_reacquires_through_its_ring(tmp_path):
    history = RadarHistory.open("KTLX", tmp_path)
    history.add_live_scan(_reacquisition_scan(0, [_storm_a(1), _storm_b(2)]))
    history.add_live_scan(_reacquisition_scan(5, [_storm_b(1)]))
    history.add_live_scan(_reacquisition_scan(10, [_storm_a(1, row=51, lat=35.301), _storm_b(2)]))

    tracking = history.newest().to_buffered_scan().tracking
    assert [event["event_type"] for event in tracking.recent_events] == ["reacquired"]
    assert history.tracker.get_track(1).status == "active"


def test_missing_track_is_lost_when_its_scan_leaves_the_ring(tmp_path):
    history = RadarHistory.open("KTLX", tmp_path, ring_size=2)
    history.add_live_scan(_reacquisition_scan(0, [_storm_a(1), _storm_b(2)]))
    history.add_live_scan(_reacquisition_scan(5, [_storm_b(1)]))
    assert history.tracker.get_track(1).status == "missing"
    history.add_live_scan(_reacquisition_scan(10, [_storm_b(1)]))  # evicts the 0-minute scan

    assert history.tracker.get_track(1).status == "lost"


def _scan_with_rotation(minute: int, evidence_level: str = "unconfirmed"):
    from dataclasses import replace
    from src.velocity import RotationSignature

    scan = _scan(minute)
    storm = scan.detected_objects[0]
    rotation = RotationSignature(
        centroid_lat=storm.centroid_lat, centroid_lon=storm.centroid_lon, distance_km=40.0, bearing_deg=270.0,
        max_shear_ms=30.0, max_inbound_ms=-15.0, max_outbound_ms=15.0, diameter_km=2.0, sweep_count=1,
        elevation_angles=[0.5], strength="moderate", associated_object_id=storm.object_id,
        evidence_level=evidence_level,
    )
    scan.detected_objects[0] = replace(storm, rotation=rotation)
    scan.rotation_signatures = [rotation]
    return scan


def test_rotation_seen_again_on_the_same_tracked_storm_is_promoted_to_persistent(tmp_path):
    history = RadarHistory.open("KTLX", tmp_path)
    history.add_live_scan(_scan_with_rotation(0))
    first = history.newest().to_buffered_scan()
    assert [r.evidence_level for r in first.rotation_signatures] == ["unconfirmed"]

    history.add_live_scan(_scan_with_rotation(5))
    newest = history.newest().to_buffered_scan()
    assert [r.evidence_level for r in newest.rotation_signatures] == ["persistent"]
    assert newest.detected_objects[0].rotation.evidence_level == "persistent"
    # The tracker's entry for this scan carries the final assessment, so
    # speech that reads track history ("rotation weakening") sees it.
    track_id = history.tracker._obj_to_track[newest.detected_objects[0].object_id]
    entry = history.tracker.get_track(track_id).rotation_history[-1]
    assert entry.timestamp == T0 + timedelta(minutes=5)
    assert entry.rotation.evidence_level == "persistent"
    snapshot_track = next(t for t in newest.tracking.active_tracks if t.track_id == track_id)
    assert snapshot_track.rotation_history[-1].rotation.evidence_level == "persistent"


def test_rotation_is_not_promoted_across_a_gap_longer_than_the_persistence_window(tmp_path):
    history = RadarHistory.open("KTLX", tmp_path)
    history.add_live_scan(_scan_with_rotation(0))
    history.add_live_scan(_scan(5))                 # the storm continues without rotation
    history.add_live_scan(_scan_with_rotation(25))  # 25 minutes after the last rotation
    newest = history.newest().to_buffered_scan()
    assert [r.evidence_level for r in newest.rotation_signatures] == ["unconfirmed"]
