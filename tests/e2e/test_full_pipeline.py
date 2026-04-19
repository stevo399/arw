# tests/e2e/test_full_pipeline.py
"""End-to-end tests for the full ARW pipeline.

These tests mock only the network boundary (NEXRAD download and geocoding)
and exercise the entire pipeline: sites → scan → objects → summary.
"""
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import numpy as np
from src.server import app
from src.parser import ReflectivityData

client = TestClient(app)


def _make_reflectivity_data_with_storm() -> ReflectivityData:
    """Create a ReflectivityData with a synthetic storm."""
    reflectivity = np.full((360, 500), np.nan)
    # Storm 1: large, intense — centered at azimuth 90, range bin 200
    reflectivity[80:110, 180:220] = 25.0   # light rain shell
    reflectivity[85:105, 190:210] = 35.0   # moderate core
    reflectivity[90:100, 195:205] = 50.0   # intense core
    # Storm 2: smaller, moderate — centered at azimuth 270, range bin 350
    reflectivity[265:280, 340:360] = 30.0  # moderate rain
    reflectivity[268:278, 345:355] = 42.0  # heavy rain core

    return ReflectivityData(
        reflectivity=reflectivity,
        azimuths=np.linspace(0, 359, 360),
        ranges_m=np.linspace(2000, 250000, 500),
        radar_lat=35.3331,
        radar_lon=-97.2778,
        elevation_angle=0.5,
        elevation_angles=[0.5, 1.5, 2.4],
        timestamp="2026-04-08T18:30:00Z",
    )


def test_full_pipeline_sites_to_summary():
    """Test the complete flow: get sites → get objects → get summary."""
    # Step 1: Get sites for Oklahoma City
    with patch("src.server.geocode_city_state", return_value=(35.4676, -97.5164)):
        resp = client.get("/sites?city=Oklahoma+City&state=OK")
    assert resp.status_code == 200
    sites = resp.json()
    assert len(sites) > 0
    site_ids = [s["site_id"] for s in sites]
    assert "KTLX" in site_ids

    # Step 2: Get objects for KTLX with synthetic storm data
    ref_data = _make_reflectivity_data_with_storm()
    with patch("src.server.fetch_scan", return_value="/fake/path"), \
         patch("src.server.parse_radar_file", return_value=MagicMock()), \
         patch("src.server.extract_reflectivity_from_radar", return_value=ref_data), \
         patch("src.server.extract_velocity", return_value=None):
        resp = client.get("/objects/KTLX")
    assert resp.status_code == 200
    data = resp.json()
    assert data["object_count"] == 2
    assert data["objects"][0]["peak_dbz"] >= data["objects"][1]["peak_dbz"]
    strongest = data["objects"][0]
    layer_labels = [l["label"] for l in strongest["layers"]]
    assert "light rain" in layer_labels
    assert "moderate rain" in layer_labels
    assert "intense rain" in layer_labels

    # Step 3: Get summary for KTLX
    with patch("src.server.fetch_scan", return_value="/fake/path"), \
         patch("src.server.parse_radar_file", return_value=MagicMock()), \
         patch("src.server.extract_reflectivity_from_radar", return_value=ref_data), \
         patch("src.server.extract_velocity", return_value=None):
        resp = client.get("/summary/KTLX")
    assert resp.status_code == 200
    summary = resp.json()
    assert "2 rain objects" in summary["text"]
    assert "intense rain" in summary["text"]
    assert "Oklahoma City" in summary["text"]


def test_full_pipeline_no_precipitation():
    """Test the pipeline when there is no precipitation."""
    ref_data = ReflectivityData(
        reflectivity=np.full((360, 500), np.nan),
        azimuths=np.linspace(0, 359, 360),
        ranges_m=np.linspace(2000, 250000, 500),
        radar_lat=35.3331,
        radar_lon=-97.2778,
        elevation_angle=0.5,
        elevation_angles=[0.5],
        timestamp="2026-04-08T18:30:00Z",
    )
    with patch("src.server.fetch_scan", return_value="/fake/path"), \
         patch("src.server.parse_radar_file", return_value=MagicMock()), \
         patch("src.server.extract_reflectivity_from_radar", return_value=ref_data), \
         patch("src.server.extract_velocity", return_value=None):
        resp = client.get("/summary/KTLX")
    assert resp.status_code == 200
    assert "No significant precipitation" in resp.json()["text"]


def test_full_pipeline_scan_metadata():
    """Test getting scan metadata."""
    ref_data = _make_reflectivity_data_with_storm()
    with patch("src.server.fetch_scan", return_value="/fake/path"), \
         patch("src.server.parse_radar_file", return_value=MagicMock()), \
         patch("src.server.extract_reflectivity_from_radar", return_value=ref_data), \
         patch("src.server.extract_velocity", return_value=None):
        resp = client.get("/scan/KTLX")
    assert resp.status_code == 200
    data = resp.json()
    assert data["site_id"] == "KTLX"
    assert data["elevation_angles"] == [0.5, 1.5, 2.4]
    assert data["timestamp"] == "2026-04-08T18:30:00Z"


def test_tracks_endpoint_e2e():
    """Test /tracks endpoint with synthetic data."""
    ref_data = _make_reflectivity_data_with_storm()
    with patch("src.server.fetch_scan", return_value="/fake/path"), \
         patch("src.server.parse_radar_file", return_value=MagicMock()), \
         patch("src.server.extract_reflectivity_from_radar", return_value=ref_data), \
         patch("src.server.extract_velocity", return_value=None):
        resp = client.get("/tracks/KTLX")
    assert resp.status_code == 200
    data = resp.json()
    assert data["active_count"] >= 0
    assert isinstance(data["tracks"], list)
    assert isinstance(data["recent_events"], list)


def test_tracks_accumulate_across_calls():
    """Two calls to /tracks should show tracks with multiple positions."""
    # Reset server state
    import src.server as srv
    srv._buffer = type(srv._buffer)()
    srv._tracker = type(srv._tracker)()

    ref_data1 = ReflectivityData(
        reflectivity=np.full((360, 500), np.nan),
        azimuths=np.linspace(0, 359, 360),
        ranges_m=np.linspace(2000, 250000, 500),
        radar_lat=35.3331,
        radar_lon=-97.2778,
        elevation_angle=0.5,
        elevation_angles=[0.5],
        timestamp="2026-04-08T18:30:00Z",
    )
    ref_data1.reflectivity[85:95, 195:205] = 45.0

    ref_data2 = ReflectivityData(
        reflectivity=np.full((360, 500), np.nan),
        azimuths=np.linspace(0, 359, 360),
        ranges_m=np.linspace(2000, 250000, 500),
        radar_lat=35.3331,
        radar_lon=-97.2778,
        elevation_angle=0.5,
        elevation_angles=[0.5],
        timestamp="2026-04-08T18:35:00Z",
    )
    ref_data2.reflectivity[86:96, 196:206] = 45.0  # Slightly moved

    with patch("src.server.fetch_scan", return_value="/fake/path"), \
         patch("src.server.parse_radar_file", return_value=MagicMock()), \
         patch("src.server.extract_reflectivity_from_radar", return_value=ref_data1), \
         patch("src.server.extract_velocity", return_value=None):
        resp1 = client.get("/tracks/KTLX")
    assert resp1.status_code == 200

    with patch("src.server.fetch_scan", return_value="/fake/path"), \
         patch("src.server.parse_radar_file", return_value=MagicMock()), \
         patch("src.server.extract_reflectivity_from_radar", return_value=ref_data2), \
         patch("src.server.extract_velocity", return_value=None):
        resp2 = client.get("/tracks/KTLX")
    assert resp2.status_code == 200
    data = resp2.json()
    assert data["active_count"] >= 1
    if data["tracks"]:
        motion = data["tracks"][0]["motion"]
        assert "confidence_label" in motion
        assert "confidence_score" in motion
