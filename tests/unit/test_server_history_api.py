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
