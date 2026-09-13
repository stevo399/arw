from datetime import datetime, timedelta

from fastapi.testclient import TestClient

import src.server as server
from src.detection import DetectedObject, IntensityLayerData
from src.history.records import dumps, loads
from src.tracking.trends import compute_trend
from src.tracking.types import MAX_TREND_SAMPLES, TrendSample, Track
from tests.unit.test_tracking_reacquisition import _Recorder, _storm_a_hidden

T0 = datetime(2026, 9, 12, 18, 0)


def _samples(areas, cores=None, reacquired_index=None):
    cores = cores or [0.0] * len(areas)
    return [
        TrendSample(T0 + timedelta(minutes=5 * i), area, core, 50.0, reacquired=(i == reacquired_index))
        for i, (area, core) in enumerate(zip(areas, cores))
    ]


def test_consistent_growth_is_medium_confidence():
    trend = compute_trend(_samples([100.0, 120.0, 150.0], cores=[0.0, 4.0, 10.0]))
    assert (trend.area, trend.core_area, trend.confidence) == ("growing", "growing", "medium")
    assert trend.sample_count == 3


def test_decay_and_steady_are_classified():
    assert compute_trend(_samples([150.0, 130.0, 100.0])).area == "decaying"
    assert compute_trend(_samples([100.0, 104.0, 98.0])).area == "steady"


def test_direction_change_within_window_lowers_confidence():
    trend = compute_trend(_samples([100.0, 80.0, 150.0]))
    assert trend.area == "growing" and trend.confidence == "low"


def test_no_core_is_reported_as_none():
    assert compute_trend(_samples([100.0, 120.0, 150.0])).core_area == "none"


def test_fewer_than_three_samples_is_insufficient():
    trend = compute_trend(_samples([100.0, 150.0]))
    assert (trend.area, trend.confidence) == ("insufficient", "none")


def test_window_with_a_reacquired_sample_is_insufficient():
    trend = compute_trend(_samples([100.0, 120.0, 150.0], reacquired_index=2))
    assert trend.area == "insufficient"
    assert "reacquired" in trend.reason


def test_add_position_records_core_area_and_caps_samples():
    track = Track(track_id=1, status="active")
    obj = DetectedObject(
        1, 35.0, -97.0, 40.0, 90.0, 62.0, "severe core", 100.0,
        layers=[
            IntensityLayerData("heavy precipitation", 40.0, 50.0, 60.0),
            IntensityLayerData("intense precipitation", 50.0, 60.0, 12.0),
            IntensityLayerData("severe core", 60.0, float("inf"), 3.0),
        ],
    )
    for i in range(MAX_TREND_SAMPLES + 2):
        track.add_position(T0 + timedelta(minutes=5 * i), obj)
    assert len(track.trend_samples) == MAX_TREND_SAMPLES
    assert track.trend_samples[-1].core_area_km2 == 15.0
    assert loads(dumps(track)) == track


def test_reacquired_match_marks_its_trend_sample():
    recorder = _Recorder()
    _storm_a_hidden(recorder)
    samples = recorder.tracker.get_track(1).trend_samples
    assert [sample.reacquired for sample in samples] == [False, True]


def test_tracks_endpoint_reports_trend(monkeypatch):
    recorder = _Recorder()
    _storm_a_hidden(recorder)
    scan = recorder.stored[max(recorder.stored)]
    scan.tracking = recorder.tracker.snapshot()
    monkeypatch.setattr(server, "_live_or_ingest", lambda site_id, dt=None: (scan, False))
    data = TestClient(server.app).get("/tracks/KTLX").json()
    trends = {track["track_id"]: track["trend"] for track in data["tracks"]}
    assert trends[1]["area"] == "insufficient"
    assert set(trends[1]) == {"area", "core_area", "confidence", "reason", "sample_count"}
