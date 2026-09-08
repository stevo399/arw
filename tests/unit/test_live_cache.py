from unittest.mock import MagicMock
from types import SimpleNamespace
from threading import Event, Thread

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


def test_failed_live_refresh_can_be_retried_without_waiting_the_normal_interval(monkeypatch):
    _clear_live_state()
    monkeypatch.setattr(server, "_schedule_recurring_refresh", lambda _site_id: None)
    monkeypatch.setattr(server, "_ingest_to_buffer", MagicMock(side_effect=RuntimeError("temporary fetch failure")))
    try:
        server._refresh_live_scan("KOHX")
        with server._state_lock:
            last_started = server._refresh_started_at["KOHX"]
        elapsed = (server.datetime.now() - last_started).total_seconds()
        assert elapsed >= server._refresh_interval_seconds - server.LIVE_REFRESH_FAILURE_RETRY_SECONDS
    finally:
        _clear_live_state()


def test_location_relevance_filter_hides_remote_interpretations_but_records_them():
    near = {
        "type": "Feature",
        "properties": {
            "object_id": 1, "centroid_lat": 33.5, "centroid_lon": -112.1,
            "distance_km": 20.0, "description": "Nearby echo.", "soundPriority": 500,
        },
    }
    remote_band = {
        "type": "Feature",
        "properties": {
            "object_id": 2, "centroid_lat": 31.7415, "centroid_lon": -113.344,
            "distance_km": 233.0, "description": "Remote echo.", "soundPriority": 650,
        },
    }
    geojson = {"type": "FeatureCollection", "metadata": {}, "features": [near, remote_band]}

    filtered = server._filter_map_features_by_relevance(
        geojson, 33.4484, -112.0741, 75.0, 0.5
    )

    assert [feature["properties"]["object_id"] for feature in filtered["features"]] == [1]
    assert filtered["metadata"]["relevanceRadiusMiles"] == 75.0
    assert filtered["metadata"]["relevanceOmittedObjectCount"] == 1
    assert filtered["metadata"]["relevanceOmittedFeatureCount"] == 1
    assert [round(value, 4) for value in filtered["bbox"]] == [
        -113.3736, 32.3614, -110.7746, 34.5354
    ]


def test_relevance_filter_demotes_elevated_beam_when_a_wide_radius_is_requested():
    feature = {
        "type": "Feature",
        "properties": {
            "object_id": 2, "centroid_lat": 31.7415, "centroid_lon": -113.344,
            "distance_km": 233.0, "description": "Remote echo.", "soundPriority": 650,
        },
    }
    geojson = {"type": "FeatureCollection", "metadata": {}, "features": [feature]}

    filtered = server._filter_map_features_by_relevance(
        geojson, 33.4484, -112.0741, 200.0, 0.5
    )

    properties = filtered["features"][0]["properties"]
    assert properties["beamHeightKmAboveRadar"] == 5.2
    assert properties["surfacePrecipitationObservable"] is False
    assert properties["soundPriority"] == 250
    assert "elevated evidence" in properties["description"]


def test_empty_local_result_has_a_truthful_geographic_location_anchor():
    geojson = {"type": "FeatureCollection", "metadata": {}, "features": []}

    filtered = server._filter_map_features_by_relevance(
        geojson, 33.4484, -112.0741, 75.0, 0.5
    )

    assert filtered["metadata"]["relevanceEmpty"] is True
    assert filtered["metadata"]["relevanceLocationAnchor"] is True
    anchor = filtered["features"]
    assert len(anchor) == 1
    assert anchor[0]["geometry"] == {
        "type": "Point", "coordinates": [-112.0741, 33.4484]
    }
    assert anchor[0]["properties"]["isLocationAnchor"] is True
    assert "no nearby interpreted echoes" in anchor[0]["properties"]["name"]


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


def test_completed_map_layer_does_not_wait_for_another_site_ingest(tmp_path):
    _clear_live_state()
    source = tmp_path / "KIWA20260907_222941_V06"
    source.write_bytes(b"radar")
    scan = SimpleNamespace(
        site_id="KIWA",
        source_path=str(source),
        reflectivity_data=SimpleNamespace(timestamp="2026-09-07T22:29:41Z"),
    )
    expected = {"audiom": {"type": "FeatureCollection", "features": []}}
    with server._state_lock:
        server._map_layers[server._map_layer_cache_key(scan)] = expected

    finished = Event()
    result = []

    def read_cached_layer():
        result.append(server._prepared_map_layers(scan))
        finished.set()

    server._ingest_lock.acquire()
    worker = Thread(target=read_cached_layer)
    try:
        worker.start()
        assert finished.wait(0.5), "cached map layer waited for an unrelated ingest"
        assert result == [expected]
    finally:
        server._ingest_lock.release()
        worker.join(timeout=1)
        _clear_live_state()
