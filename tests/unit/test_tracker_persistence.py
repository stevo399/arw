from datetime import datetime, timedelta
import logging

import pytest

from src.history.records import dumps, loads
from src.tracker import StormTracker
from tests.unit.test_tracker import _make_object, _make_scan

T0 = datetime(2026, 9, 12, 18, 0)


def _scans():
    return [
        _make_scan("KTLX", T0, [_make_object(1, 35.3, -97.30), _make_object(2, 35.0, -97.0)]),
        _make_scan("KTLX", T0 + timedelta(minutes=5), [_make_object(1, 35.3, -97.29), _make_object(2, 35.0, -97.0)]),
        _make_scan("KTLX", T0 + timedelta(minutes=10), [_make_object(1, 35.3, -97.28)]),
        _make_scan("KTLX", T0 + timedelta(minutes=15), [_make_object(1, 35.3, -97.27), _make_object(2, 35.0, -97.0)]),
    ]


def _store(scans):
    stored = {(scan.site_id, scan.timestamp): scan for scan in scans}
    return lambda site_id, timestamp: stored.get((site_id, timestamp))


def test_tracker_with_loader_matches_tracker_that_retains_scans():
    retained = StormTracker()
    for scan in _scans():
        retained.update(scan)

    scans = _scans()
    loaded = StormTracker(scan_loader=_store(scans))
    for scan in scans:
        loaded.update(scan)
        loaded.release_previous_scan()

    assert loaded._prev_scan is None
    assert dumps(loaded.export_state()) == dumps(retained.export_state())


def test_restored_tracker_continues_exactly_like_an_uninterrupted_one():
    uninterrupted = StormTracker()
    for scan in _scans():
        uninterrupted.update(scan)

    scans = _scans()
    first = StormTracker(scan_loader=_store(scans))
    for scan in scans[:2]:
        first.update(scan)
        first.release_previous_scan()
    resumed = StormTracker.from_state(loads(dumps(first.export_state())), scan_loader=_store(scans))
    for scan in scans[2:]:
        resumed.update(scan)
        resumed.release_previous_scan()

    assert dumps(resumed.export_state()) == dumps(uninterrupted.export_state())


def test_unavailable_previous_scan_restarts_tracking_with_a_warning(caplog):
    scans = _scans()
    tracker = StormTracker(scan_loader=lambda site_id, timestamp: None)
    tracker.update(scans[0])
    tracker.release_previous_scan()
    with caplog.at_level(logging.WARNING, logger="src.tracker"):
        tracker.update(scans[1])
    assert "no longer available" in caplog.text
    assert [track.track_id for track in tracker.active_tracks] == [1, 2]
    assert all(len(track.positions) == 1 for track in tracker.active_tracks)


def test_release_requires_a_loader():
    tracker = StormTracker()
    tracker.update(_scans()[0])
    with pytest.raises(RuntimeError):
        tracker.release_previous_scan()


def test_state_version_mismatch_is_rejected():
    state = StormTracker().export_state()
    state["version"] = 999
    with pytest.raises(ValueError, match="version"):
        StormTracker.from_state(state, scan_loader=lambda site_id, timestamp: None)
