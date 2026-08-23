from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import numpy as np
from src.parser import ReflectivityData
from src.server import app

client = TestClient(app)


def _make_reflectivity_data(fill_value: float, elevation_angles: list[float] | None = None) -> ReflectivityData:
    grid = np.full((360, 500), fill_value)
    return ReflectivityData(
        reflectivity=grid,
        azimuths=np.linspace(0, 359, 360),
        ranges_m=np.linspace(2000, 250000, 500),
        radar_lat=35.3331,
        radar_lon=-97.2778,
        elevation_angle=0.5,
        elevation_angles=elevation_angles or [0.5],
        timestamp="2026-04-08T18:30:00Z",
    )


def test_root_returns_200():
    resp = client.get("/")
    assert resp.status_code == 200
    assert "ARW" in resp.json()["name"]


def test_radar_page_returns_form():
    resp = client.get("/radar")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "ARW Radar Map" in resp.text
    assert 'id="search-form"' in resp.text
    assert 'id="audiom-frame"' in resp.text


def test_radar_config_returns_non_secret_settings():
    resp = client.get("/radar/config")
    assert resp.status_code == 200
    data = resp.json()
    assert data["audiom_map_url"]
    assert data["audiom_rules_path"] == "/rules/arw-storms.json"
    assert "api_key" not in data


def test_cors_allows_audiom_origin():
    resp = client.options(
        "/map/storms.geojson",
        headers={
            "Origin": "https://www.audiom.net",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert resp.status_code == 200
    assert resp.headers["access-control-allow-origin"] == "https://www.audiom.net"


def test_sites_endpoint_returns_200():
    with patch("src.server.geocode_city_state", return_value=(35.4676, -97.5164)):
        resp = client.get("/sites?city=Oklahoma+City&state=OK")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) > 0
    assert "site_id" in data[0]


def test_sites_endpoint_missing_params_returns_422():
    resp = client.get("/sites")
    assert resp.status_code == 422


def test_scan_endpoint_returns_200():
    mock_ref = _make_reflectivity_data(0.0, [0.5, 1.5])
    with patch("src.server.fetch_scan", return_value="/fake/path"), \
         patch("src.server.parse_radar_file", return_value=MagicMock()), \
         patch("src.server.extract_reflectivity_from_radar", return_value=mock_ref), \
         patch("src.server.extract_velocity", return_value=None):
        resp = client.get("/scan/KTLX")
    assert resp.status_code == 200
    data = resp.json()
    assert data["site_id"] == "KTLX"
    assert "elevation_angles" in data


def test_objects_endpoint_returns_200():
    mock_ref = _make_reflectivity_data(np.nan)
    with patch("src.server.fetch_scan", return_value="/fake/path"), \
         patch("src.server.parse_radar_file", return_value=MagicMock()), \
         patch("src.server.extract_reflectivity_from_radar", return_value=mock_ref), \
         patch("src.server.extract_velocity", return_value=None):
        resp = client.get("/objects/KTLX")
    assert resp.status_code == 200
    data = resp.json()
    assert "objects" in data
    assert data["object_count"] == 0


def test_storm_map_layer_endpoint_returns_geojson_layer():
    mock_ref = _make_reflectivity_data(np.nan)
    mock_ref.reflectivity[85:95, 195:205] = 45.0
    with patch("src.server.geocode_city_state", return_value=(35.4676, -97.5164)), \
         patch("src.server.fetch_scan", return_value="/fake/path"), \
         patch("src.server.parse_radar_file", return_value=MagicMock()), \
         patch("src.server.extract_reflectivity_from_radar", return_value=mock_ref), \
         patch("src.server.extract_velocity", return_value=None):
        resp = client.get("/map/storms?city=Oklahoma+City&state=OK&date=2026-04-10")
    assert resp.status_code == 200
    data = resp.json()
    assert data["layer_type"] == "FeatureLayer"
    assert data["geometryType"] == "esriGeometryPolygon"
    assert data["spatialReference"]["wkid"] == 4326
    assert data["site"]["site_id"] == "KTLX"
    assert data["geojson"]["type"] == "FeatureCollection"
    assert data["intensity_geojson"]["type"] == "FeatureCollection"
    assert data["audiom_geojson"]["type"] == "FeatureCollection"
    assert data["centroid_geojson"]["type"] == "FeatureCollection"
    assert data["feature_count"] == len(data["geojson"]["features"])
    assert data["feature_count"] == len(data["centroid_geojson"]["features"])
    assert len(data["audiom_geojson"]["features"]) >= data["feature_count"]
    assert data["geojson"]["features"][0]["geometry"]["type"] == "Polygon"
    assert data["intensity_geojson"]["features"][0]["geometry"]["type"] == "Polygon"
    assert data["centroid_geojson"]["features"][0]["geometry"]["type"] == "Point"
    assert data["geojson"]["features"][0]["properties"]["peak_dbz"] == 45.0


def test_storm_map_layer_endpoint_accepts_zipcode():
    mock_ref = _make_reflectivity_data(np.nan)
    with patch("src.server.geocode_zipcode", return_value=(35.4676, -97.5164)), \
         patch("src.server.fetch_scan", return_value="/fake/path"), \
         patch("src.server.parse_radar_file", return_value=MagicMock()), \
         patch("src.server.extract_reflectivity_from_radar", return_value=mock_ref), \
         patch("src.server.extract_velocity", return_value=None):
        resp = client.get("/map/storms?zipcode=73102")
    assert resp.status_code == 200
    data = resp.json()
    assert data["location"]["label"] == "73102"
    assert data["site"]["site_id"] == "KTLX"


def test_storm_map_geojson_endpoint_returns_raw_feature_collection():
    mock_ref = _make_reflectivity_data(np.nan)
    mock_ref.reflectivity[85:95, 195:205] = 45.0
    with patch("src.server.geocode_city_state", return_value=(35.4676, -97.5164)), \
         patch("src.server.fetch_scan", return_value="/fake/path"), \
         patch("src.server.parse_radar_file", return_value=MagicMock()), \
         patch("src.server.extract_reflectivity_from_radar", return_value=mock_ref), \
         patch("src.server.extract_velocity", return_value=None):
        resp = client.get("/map/storms.geojson?city=Oklahoma+City&state=OK")
    assert resp.status_code == 200
    data = resp.json()
    assert data["type"] == "FeatureCollection"
    assert data["metadata"]["mapType"] == "heatmap"
    assert data["metadata"]["currentStat"] == "heat_value"
    assert data["features"][0]["geometry"]["type"] == "Polygon"
    rule_types = {feature["properties"]["ruleType"] for feature in data["features"]}
    assert "storm_heavy_footprint" in rule_types
    assert "radar_heavy_rain" in rule_types


def test_storm_map_layer_endpoint_requires_location():
    resp = client.get("/map/storms")
    assert resp.status_code == 422


def test_summary_endpoint_returns_200():
    mock_ref = _make_reflectivity_data(np.nan)
    with patch("src.server.fetch_scan", return_value="/fake/path"), \
         patch("src.server.parse_radar_file", return_value=MagicMock()), \
         patch("src.server.extract_reflectivity_from_radar", return_value=mock_ref), \
         patch("src.server.extract_velocity", return_value=None):
        resp = client.get("/summary/KTLX")
    assert resp.status_code == 200
    data = resp.json()
    assert "text" in data


def test_tracks_endpoint_returns_200():
    mock_ref = _make_reflectivity_data(np.nan)
    with patch("src.server.fetch_scan", return_value="/fake/path"), \
         patch("src.server.parse_radar_file", return_value=MagicMock()), \
         patch("src.server.extract_reflectivity_from_radar", return_value=mock_ref), \
         patch("src.server.extract_velocity", return_value=None):
        resp = client.get("/tracks/KTLX")
    assert resp.status_code == 200
    data = resp.json()
    assert "tracks" in data
    assert "active_count" in data
    assert "recent_events" in data


def test_tracks_endpoint_includes_motion_confidence_fields():
    mock_ref = _make_reflectivity_data(np.nan)
    with patch("src.server.fetch_scan", return_value="/fake/path"), \
         patch("src.server.parse_radar_file", return_value=MagicMock()), \
         patch("src.server.extract_reflectivity_from_radar", return_value=mock_ref), \
         patch("src.server.extract_velocity", return_value=None):
        client.get("/tracks/KTLX")
    with patch("src.server.fetch_scan", return_value="/fake/path"), \
         patch("src.server.parse_radar_file", return_value=MagicMock()), \
         patch("src.server.extract_reflectivity_from_radar", return_value=mock_ref), \
         patch("src.server.extract_velocity", return_value=None):
        resp = client.get("/tracks/KTLX")
    assert resp.status_code == 200
    data = resp.json()
    if data["tracks"]:
        motion = data["tracks"][0]["motion"]
        assert "confidence_label" in motion
        assert "confidence_score" in motion
        assert "confidence_reason" in motion
        assert "source" in motion
        identity = data["tracks"][0]["identity"]
        assert "label" in identity
        assert "score" in identity
        assert "reason" in identity
        focus = data["tracks"][0]["focus"]
        assert "label" in focus
        assert "score" in focus
        assert "reason" in focus
        assert "recent_reported_heading_sequence" in focus
        assert "reported_heading_stability_label" in focus
        assert "reported_heading_stability_score" in focus
        assert "reported_heading_stability_reason" in focus


def test_motion_endpoint_missing_track_returns_404():
    resp = client.get("/motion/KTLX/999")
    assert resp.status_code == 404


def test_velocity_endpoint_returns_200():
    mock_ref = _make_reflectivity_data(np.nan)
    with patch("src.server.fetch_scan", return_value="/fake/path"), \
         patch("src.server.parse_radar_file", return_value=MagicMock()), \
         patch("src.server.extract_reflectivity_from_radar", return_value=mock_ref), \
         patch("src.server.extract_velocity", return_value=None):
        resp = client.get("/velocity/KTLX")
    assert resp.status_code == 200
    data = resp.json()
    assert "regions" in data
    assert "rotation_signatures" in data
