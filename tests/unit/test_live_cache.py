from unittest.mock import MagicMock
from types import SimpleNamespace

import src.server as server


def _clear_live_state():
    with server._state_lock:
        server._processed_scans.clear()
        server._map_layers.clear()
        server._live_scans.clear()
        server._refreshing_sites.clear()
        server._refresh_started_at.clear()
        server._refresh_errors.clear()


def test_live_request_returns_completed_scan_and_queues_refresh(monkeypatch):
    _clear_live_state()
    completed = MagicMock()
    completed.site_id = "KIWA"
    with server._state_lock:
        server._live_scans["KIWA"] = completed

    scheduled = []
    monkeypatch.setattr(
        server, "_schedule_live_refresh", lambda site_id: scheduled.append(site_id) or True
    )

    try:
        result, updating = server._live_or_ingest("kiwa")
        assert result is completed
        assert updating is True
        assert scheduled == ["KIWA"]
    finally:
        _clear_live_state()


def test_live_refresh_is_coalesced(monkeypatch):
    _clear_live_state()
    executor = MagicMock()
    monkeypatch.setattr(server, "_refresh_executor", executor)
    monkeypatch.setattr(server, "_refresh_interval_seconds", 60)

    try:
        assert server._schedule_live_refresh("KIWA") is True
        assert server._schedule_live_refresh("KIWA") is True
        assert executor.submit.call_count == 1
    finally:
        _clear_live_state()


def test_prepared_map_layers_are_built_once_per_real_volume(monkeypatch, tmp_path):
    _clear_live_state()
    source = tmp_path / "KIWA20260907_222941_V06"
    source.write_bytes(b"radar")
    scan = SimpleNamespace(
        site_id="KIWA",
        source_path=str(source),
        reflectivity_data=SimpleNamespace(timestamp="2026-09-07T22:29:41Z"),
    )
    builders = [
        "build_storm_geojson",
        "build_storm_intensity_geojson",
        "build_storm_audiom_geojson",
        "build_storm_centroid_geojson",
        "build_precipitation_field_geojson",
    ]
    mocks = []
    for name in builders:
        mock = MagicMock(return_value={"type": "FeatureCollection", "features": []})
        monkeypatch.setattr(server, name, mock)
        mocks.append(mock)

    try:
        first = server._prepared_map_layers(scan)
        second = server._prepared_map_layers(scan)
        assert first is second
        assert all(mock.call_count == 1 for mock in mocks)
    finally:
        _clear_live_state()
