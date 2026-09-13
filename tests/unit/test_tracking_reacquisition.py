from datetime import datetime, timedelta

import numpy as np
import pytest

from src.summary import _get_track_for_object
from src.tracker import MAX_TEMPORAL_CONTINUITY_MINUTES, MEDIUM_CONFIDENCE, StormTracker
from src.tracking.association import MAX_REACQUISITION_MINUTES
from tests.unit.test_tracking_association import _make_object, _make_scan

SHAPE = (360, 500)
T0 = datetime(2026, 9, 12, 18, 0)


def _mask(row: int, col: int, rows: int = 10, cols: int = 10) -> np.ndarray:
    grid = np.zeros(SHAPE, dtype=bool)
    grid[row:row + rows, col:col + cols] = True
    return grid


def _storm_a(object_id, row=50, lat=35.30):
    return _make_object(object_id, lat, -97.50, 55.0), _mask(row, 100)


def _storm_b(object_id):
    return _make_object(object_id, 35.0, -97.0, 45.0), _mask(200, 300)


def _scan(minute, storms):
    objects = [obj for obj, _ in storms]
    masks = {obj.object_id: mask for obj, mask in storms}
    return _make_scan("KTLX", T0 + timedelta(minutes=minute), objects, masks)


class _Recorder:
    """A tracker whose loader serves every scan it has processed."""

    def __init__(self, reacquire=True):
        self.stored = {}
        self.tracker = StormTracker(
            scan_loader=lambda site_id, timestamp: self.stored.get(timestamp),
            reacquire=reacquire,
        )

    def update(self, scan):
        self.tracker.update(scan)
        self.stored[scan.timestamp] = scan


def _storm_a_hidden(recorder, hidden_minutes=(5,), return_minute=10):
    recorder.update(_scan(0, [_storm_a(1), _storm_b(2)]))
    for minute in hidden_minutes:
        recorder.update(_scan(minute, [_storm_b(1)]))
    recorder.update(_scan(return_minute, [_storm_a(1, row=51, lat=35.301), _storm_b(2)]))


def test_storm_missing_for_one_scan_keeps_its_identity_with_low_confidence():
    recorder = _Recorder()
    _storm_a_hidden(recorder)
    track = recorder.tracker.get_track(1)

    assert track.status == "active"
    assert [position.latitude for position in track.positions] == [35.30, 35.301]
    assert track.identity_diagnostics.event_context == "reacquired"
    assert track.identity_diagnostics.missed_scans == 1
    assert "1 missed scan" in track.identity_diagnostics.reason
    assert track.identity_confidence < MEDIUM_CONFIDENCE
    assert track.identity_diagnostics.label == "low"
    assert [event["event_type"] for event in recorder.tracker.recent_events] == ["reacquired"]
    assert recorder.tracker.temporal_status_for_current_object(1) == "persistent"
    assert len(recorder.tracker.all_tracks) == 2


def test_storm_missing_for_two_scans_is_reacquired():
    recorder = _Recorder()
    _storm_a_hidden(recorder, hidden_minutes=(5, 10), return_minute=15)
    track = recorder.tracker.get_track(1)

    assert track.status == "active"
    assert track.identity_diagnostics.missed_scans == 2
    assert "2 missed scans" in track.identity_diagnostics.reason
    assert len(recorder.tracker.all_tracks) == 2


def test_missing_track_is_not_active_and_not_described():
    recorder = _Recorder()
    recorder.update(_scan(0, [_storm_a(1), _storm_b(2)]))
    recorder.update(_scan(5, [_storm_b(1)]))

    track_a = recorder.tracker.get_track(1)
    assert track_a.status == "missing"
    assert track_a not in recorder.tracker.active_tracks
    storm_b, _ = _storm_b(1)
    described = _get_track_for_object(storm_b, recorder.tracker.active_tracks)
    assert described.track_id == 2


def test_a_different_nearby_storm_is_not_taken_for_the_missing_one():
    recorder = _Recorder()
    recorder.update(_scan(0, [_storm_a(1), _storm_b(2)]))
    recorder.update(_scan(5, [_storm_b(1)]))
    nearby = (_make_object(1, 35.28, -97.50, 55.0), _mask(80, 100))  # no overlap with A
    recorder.update(_scan(10, [nearby, _storm_b(2)]))

    assert recorder.tracker.get_track(1).status == "missing"
    assert len(recorder.tracker.all_tracks) == 3
    assert not any(e["event_type"] == "reacquired" for e in recorder.tracker.recent_events)


def test_storm_unseen_longer_than_the_continuity_limit_is_lost_not_reacquired():
    recorder = _Recorder()
    recorder.update(_scan(0, [_storm_a(1), _storm_b(2)]))
    recorder.update(_scan(15, [_storm_b(1)]))
    assert recorder.tracker.get_track(1).status == "missing"
    recorder.update(_scan(30, [_storm_a(1, row=51, lat=35.301), _storm_b(2)]))

    assert recorder.tracker.get_track(1).status == "lost"
    assert len(recorder.tracker.all_tracks) == 3


def test_track_is_lost_when_its_last_seen_scan_is_expired():
    recorder = _Recorder()
    recorder.update(_scan(0, [_storm_a(1), _storm_b(2)]))
    recorder.update(_scan(5, [_storm_b(1)]))
    del recorder.stored[T0]
    recorder.tracker.expire_scan("KTLX", T0)

    assert recorder.tracker.get_track(1).status == "lost"
    assert recorder.tracker.get_track(2).status == "active"


def test_unloadable_last_seen_scan_makes_the_track_lost():
    recorder = _Recorder()
    recorder.update(_scan(0, [_storm_a(1), _storm_b(2)]))
    recorder.update(_scan(5, [_storm_b(1)]))
    del recorder.stored[T0]  # gone without an expiry notice
    recorder.update(_scan(10, [_storm_a(1, row=51, lat=35.301), _storm_b(2)]))

    assert recorder.tracker.get_track(1).status == "lost"
    assert len(recorder.tracker.all_tracks) == 3


def test_continuously_tracked_storm_always_keeps_a_contested_object():
    recorder = _Recorder()
    # A and D have adjacent masks but centroids 16.7 km apart -- beyond the
    # 10 km plausible distance for a 5-minute scan -- so the second scan does
    # not merge A into D.
    storm_a = (_make_object(1, 35.45, -97.50, 55.0), _mask(50, 100))
    storm_d = (_make_object(2, 35.30, -97.48, 50.0), _mask(50, 112))
    recorder.update(_scan(0, [storm_a, storm_d]))
    recorder.update(_scan(5, [(_make_object(1, 35.30, -97.48, 50.0), _mask(50, 112))]))
    # Covers A's last mask and D's mask, centred where A was last seen, so A
    # alone would pass every reacquisition check.  D was tracked continuously.
    contested = (_make_object(1, 35.45, -97.50, 55.0), _mask(50, 100, cols=22))
    recorder.update(_scan(10, [contested]))

    assert recorder.tracker.get_track(2).current_object.centroid_lat == 35.45
    assert recorder.tracker.get_track(1).status == "missing"
    assert not any(e["event_type"] == "reacquired" for e in recorder.tracker.recent_events)


def test_without_reacquisition_the_returning_storm_gets_a_new_identity():
    """Documents defect 2 as it behaves with reacquisition disabled."""
    recorder = _Recorder(reacquire=False)
    _storm_a_hidden(recorder)
    assert recorder.tracker.get_track(1).status == "missing"
    assert len(recorder.tracker.all_tracks) == 3


def test_without_reacquisition_missing_tracks_are_lost_after_the_time_limit():
    recorder = _Recorder(reacquire=False)
    recorder.update(_scan(0, [_storm_a(1), _storm_b(2)]))
    recorder.update(_scan(10, [_storm_b(1)]))
    recorder.update(_scan(20, [_storm_b(1)]))
    assert recorder.tracker.get_track(1).status == "missing"
    recorder.update(_scan(25, [_storm_b(1)]))
    assert recorder.tracker.get_track(1).status == "lost"


def test_reacquisition_requires_a_scan_loader():
    with pytest.raises(ValueError):
        StormTracker(reacquire=True)


def test_reacquisition_window_matches_the_tracker_continuity_limit():
    assert MAX_REACQUISITION_MINUTES == MAX_TEMPORAL_CONTINUITY_MINUTES


def test_expiring_a_scan_does_not_change_a_tracker_that_never_reacquires():
    recorder = _Recorder(reacquire=False)
    recorder.update(_scan(0, [_storm_a(1), _storm_b(2)]))
    recorder.update(_scan(5, [_storm_b(1)]))
    recorder.tracker.expire_scan("KTLX", T0)
    assert recorder.tracker.get_track(1).status == "missing"


def test_focus_flag_never_stays_on_a_track_that_is_not_active():
    recorder = _Recorder()
    recorder.update(_scan(0, [_storm_a(1), _storm_b(2)]))
    assert recorder.tracker.get_track(1).is_primary_focus, "fixture: storm A is the focus"
    recorder.update(_scan(5, [_storm_b(1)]))
    assert not recorder.tracker.get_track(1).is_primary_focus
    recorder.update(_scan(10, [_storm_a(1, row=51, lat=35.301), _storm_b(2)]))
    assert sum(track.is_primary_focus for track in recorder.tracker.all_tracks) == 1
