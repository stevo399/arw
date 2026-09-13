from datetime import datetime, timedelta

from fastapi.testclient import TestClient
import pytest

import src.history.store as store
import src.server as server
from src.history.store import TRACKER_STATE_FILENAME, HistoryRegistry, RadarHistory
from tests.unit.test_tracker import _make_object, _make_scan

T0 = datetime(2026, 9, 12, 18, 0)
LOCATION = {"latitude": "35.33", "longitude": "-97.28"}


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(server, "rank_sites", lambda lat, lon: [{"site_id": "KTLX"}])
    monkeypatch.setattr(server, "_schedule_live_refresh", lambda *a, **k: False)
    monkeypatch.setattr(server, "fetch_scan", lambda *a, **k: pytest.fail("must not fetch"))
    return TestClient(server.app)


def _retain(minutes):
    with server._histories.use("KTLX") as history:
        for minute in minutes:
            history.add_live_scan(_make_scan(
                "KTLX", T0 + timedelta(minutes=minute),
                [_make_object(1, 35.3, -97.3), _make_object(2, 35.0, -97.0)],
            ))


def test_history_lists_retained_scans_newest_first(client):
    _retain([0, 5, 10])
    data = client.get("/map/history", params=LOCATION).json()
    assert data["site_id"] == "KTLX"
    assert [scan["timestamp"] for scan in data["scans"]] == [
        (T0 + timedelta(minutes=m)).isoformat() for m in (10, 5, 0)
    ]
    assert all(scan["tracked"] and scan["object_count"] == 2 for scan in data["scans"])


def test_history_for_a_radar_with_nothing_retained_is_empty(client):
    assert client.get("/map/history", params=LOCATION).json()["scans"] == []


def test_listed_timestamp_round_trips_to_tracks_and_status(client):
    _retain([0, 5])
    oldest = client.get("/map/history", params=LOCATION).json()["scans"][-1]["timestamp"]

    status = client.get("/map/status", params={**LOCATION, "datetime": oldest}).json()
    tracks = client.get("/tracks/KTLX", params={"datetime": oldest}).json()

    assert status["available"] is True and status["refresh_state"] == "ready"
    assert status["scan_timestamp"] == oldest
    assert tracks["tracking_context"] == "tracked"
    assert tracks["timestamp"] == oldest
    assert [len(track["positions"]) for track in tracks["tracks"]] == [1, 1]


def test_rebuilt_continuity_is_reported_and_spoken(client, monkeypatch, tmp_path):
    directory = tmp_path / "KTLX" / "history"
    history = RadarHistory.open("KTLX", directory)
    for minute in (0, 5):
        history.add_live_scan(_make_scan("KTLX", T0 + timedelta(minutes=minute), [_make_object(1, 35.3, -97.3)]))
    (directory / TRACKER_STATE_FILENAME).unlink()
    monkeypatch.setattr(server, "_histories", HistoryRegistry(tmp_path))

    tracks = client.get("/tracks/KTLX").json()
    summary = client.get("/summary/KTLX").json()

    assert tracks["continuity_rebuilt"] is True
    assert summary["continuity_rebuilt"] is True
    assert "storm tracking was rebuilt after a restart" in summary["text"]


def test_live_status_reports_history_write_errors(client, monkeypatch, tmp_path):
    monkeypatch.setattr(server, "_histories", HistoryRegistry(tmp_path))

    def failing_save(scan, directory):
        raise OSError("disk full")

    monkeypatch.setattr(store, "save_compact_scan", failing_save)
    _retain([0])
    status = client.get("/live/KTLX/status").json()
    assert status["available"] is True
    assert status["history_write_error"] == "OSError: disk full"


class _CapturingExecutor:
    def __init__(self):
        self.jobs = []

    def submit(self, fn, *args):
        self.jobs.append((fn, args))

    def run_all(self):
        jobs, self.jobs = self.jobs, []
        for fn, args in jobs:
            fn(*args)


@pytest.fixture
def background(monkeypatch):
    executor = _CapturingExecutor()
    monkeypatch.setattr(server, "_refresh_executor", executor)
    server._historical_requests.clear()
    yield executor
    server._historical_requests.clear()


SPECIFIC_TIME = {**LOCATION, "datetime": "2026-09-12T17:00:00"}


def test_specific_time_not_retained_is_prepared_off_the_request(client, monkeypatch, background):
    monkeypatch.setattr(server, "_ingest_to_buffer", lambda *a, **k: pytest.fail("status must not ingest"))

    first = client.get("/map/status", params=SPECIFIC_TIME).json()
    second = client.get("/map/status", params=SPECIFIC_TIME).json()

    assert (first["available"], first["refresh_state"], first["queued"]) == (False, "updating", True)
    assert (second["available"], second["refresh_state"], second["queued"]) == (False, "updating", False)
    assert len(background.jobs) == 1

    prepared = _make_scan("KTLX", T0 - timedelta(hours=1), [_make_object(1, 35.3, -97.3)])
    monkeypatch.setattr(server, "_ingest_to_buffer", lambda site_id, dt=None: prepared)
    background.run_all()
    ready = client.get("/map/status", params=SPECIFIC_TIME).json()

    assert (ready["available"], ready["refresh_state"], ready["queued"]) == (True, "ready", False)
    assert ready["scan_timestamp"] == prepared.reflectivity_data.timestamp
    assert background.jobs == []


def test_failed_specific_time_preparation_is_reported_then_retried(client, monkeypatch, background):
    def fail(*_args, **_kwargs):
        raise RuntimeError("volume unavailable")

    monkeypatch.setattr(server, "_ingest_to_buffer", fail)
    client.get("/map/status", params=SPECIFIC_TIME)
    background.run_all()

    failed = client.get("/map/status", params=SPECIFIC_TIME).json()
    retry = client.get("/map/status", params=SPECIFIC_TIME).json()

    assert failed["available"] is False
    assert "volume unavailable" in failed["last_refresh_error"]
    assert retry["queued"] is True and len(background.jobs) == 1
