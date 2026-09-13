from datetime import datetime, timedelta
from pathlib import Path

import pytest

import src.server as server
from tests.unit.test_tracker import _make_object, _make_scan

T0 = datetime(2026, 9, 12, 18, 0)


@pytest.fixture(autouse=True)
def _clean_caches():
    server._historical_scans.clear()
    server._map_layers.clear()
    yield
    server._historical_scans.clear()
    server._map_layers.clear()


@pytest.fixture
def volumes(tmp_path, monkeypatch):
    """Real files standing in for downloaded volumes, with scripted analysis."""
    scans, processed = {}, []

    def add(name: str, minute: int) -> str:
        path = tmp_path / name
        path.write_bytes(b"volume")
        key = str(path.resolve())
        scan = _make_scan("KTLX", T0 + timedelta(minutes=minute), [_make_object(1, 35.3, -97.3)])
        scan.source_path = key
        scans[key] = scan
        return key

    def process(site_id, filepath):
        key = str(Path(filepath).resolve())
        processed.append(key)
        return scans[key]

    monkeypatch.setattr(server, "_process_scan_file", process)
    return add, processed


def _serve_live(monkeypatch, paths):
    order = iter(paths)
    monkeypatch.setattr(server, "fetch_scan", lambda site_id, dt=None: next(order))


def test_live_volumes_enter_the_ring_with_tracking(monkeypatch, volumes):
    add, _ = volumes
    _serve_live(monkeypatch, [add("v1", 0), add("v2", 5)])
    server._ingest_to_buffer("KTLX")
    latest = server._ingest_to_buffer("KTLX")
    assert [len(t.positions) for t in latest.tracking.active_tracks] == [2]
    with server._histories.use("KTLX") as history:
        assert [scan.timestamp for scan in history.scans()] == [T0, T0 + timedelta(minutes=5)]


def test_exact_retained_timestamp_is_served_without_fetching(monkeypatch, volumes):
    add, processed = volumes
    _serve_live(monkeypatch, [add("v1", 0), add("v2", 5)])
    server._ingest_to_buffer("KTLX")
    server._ingest_to_buffer("KTLX")
    monkeypatch.setattr(server, "fetch_scan", lambda *a, **k: pytest.fail("must not fetch"))

    older = server._ingest_to_buffer("KTLX", T0)

    assert older.timestamp == T0
    assert [len(t.positions) for t in older.tracking.active_tracks] == [1]
    assert len(processed) == 2


def test_nearest_volume_that_is_retained_is_served_from_the_ring(monkeypatch, volumes):
    add, processed = volumes
    first = add("v1", 0)
    _serve_live(monkeypatch, [first])
    server._ingest_to_buffer("KTLX")
    monkeypatch.setattr(server, "fetch_scan", lambda site_id, dt=None: first)

    served = server._ingest_to_buffer("KTLX", T0 + timedelta(minutes=1))

    assert served.tracking is not None
    assert len(processed) == 1


def test_historical_volume_is_processed_once_and_never_tracked(monkeypatch, volumes):
    add, processed = volumes
    old = add("old", -60)
    monkeypatch.setattr(server, "fetch_scan", lambda site_id, dt=None: old)

    first = server._ingest_to_buffer("KTLX", T0 - timedelta(hours=1))
    second = server._ingest_to_buffer("KTLX", T0 - timedelta(hours=1))

    assert first.tracking is None and second.tracking is None
    assert processed == [old]
    with server._histories.use("KTLX") as history:
        assert history.scans() == []
        assert history.tracker.last_scan_timestamp is None


def test_historical_cache_is_bounded(monkeypatch, volumes):
    add, _ = volumes
    monkeypatch.setattr(server, "_max_historical_scans", 2)
    paths = [add(f"old{i}", -60 - i) for i in range(3)]
    for path in paths:
        monkeypatch.setattr(server, "fetch_scan", lambda site_id, dt=None, path=path: path)
        server._ingest_to_buffer("KTLX", T0 - timedelta(hours=2))
    assert len(server._historical_scans) == 2
    assert ("KTLX", paths[0]) not in server._historical_scans


def test_live_request_serves_newest_retained_scan_and_queues_refresh(monkeypatch, volumes):
    add, _ = volumes
    _serve_live(monkeypatch, [add("v1", 0)])
    server._ingest_to_buffer("KTLX")
    scheduled = []
    monkeypatch.setattr(server, "_schedule_live_refresh", lambda site_id: scheduled.append(site_id) or True)

    scan, updating = server._live_or_ingest("ktlx")

    assert scan.timestamp == T0 and updating is True and scheduled == ["KTLX"]


def test_live_status_reports_newest_retained_scan(monkeypatch, volumes):
    add, _ = volumes
    _serve_live(monkeypatch, [add("v1", 0)])
    server._ingest_to_buffer("KTLX")
    status = server.get_live_scan_status("KTLX")
    assert status["available"] is True
    assert status["scan_timestamp"] == T0.isoformat()
