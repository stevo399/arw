from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import numpy as np
from src.server import app

client = TestClient(app)


def test_root_returns_200():
    resp = client.get("/")
    assert resp.status_code == 200
    assert "ARW" in resp.json()["name"]


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
    mock_ref = MagicMock()
    mock_ref.reflectivity = np.zeros((360, 500))
    mock_ref.azimuths = np.linspace(0, 359, 360)
    mock_ref.ranges_m = np.linspace(2000, 250000, 500)
    mock_ref.elevation_angle = 0.5
    mock_ref.elevation_angles = [0.5, 1.5]
    mock_ref.timestamp = "2026-04-08T18:30:00Z"
    mock_ref.radar_lat = 35.3331
    mock_ref.radar_lon = -97.2778
    with patch("src.server.fetch_scan", return_value="/fake/path"), \
         patch("src.server.extract_reflectivity", return_value=mock_ref):
        resp = client.get("/scan/KTLX")
    assert resp.status_code == 200
    data = resp.json()
    assert data["site_id"] == "KTLX"
    assert "elevation_angles" in data


def test_objects_endpoint_returns_200():
    mock_ref = MagicMock()
    mock_ref.reflectivity = np.full((360, 500), np.nan)
    mock_ref.azimuths = np.linspace(0, 359, 360)
    mock_ref.ranges_m = np.linspace(2000, 250000, 500)
    mock_ref.radar_lat = 35.3331
    mock_ref.radar_lon = -97.2778
    mock_ref.timestamp = "2026-04-08T18:30:00Z"
    mock_ref.elevation_angle = 0.5
    mock_ref.elevation_angles = [0.5]
    with patch("src.server.fetch_scan", return_value="/fake/path"), \
         patch("src.server.extract_reflectivity", return_value=mock_ref):
        resp = client.get("/objects/KTLX")
    assert resp.status_code == 200
    data = resp.json()
    assert "objects" in data
    assert data["object_count"] == 0


def test_summary_endpoint_returns_200():
    mock_ref = MagicMock()
    mock_ref.reflectivity = np.full((360, 500), np.nan)
    mock_ref.azimuths = np.linspace(0, 359, 360)
    mock_ref.ranges_m = np.linspace(2000, 250000, 500)
    mock_ref.radar_lat = 35.3331
    mock_ref.radar_lon = -97.2778
    mock_ref.timestamp = "2026-04-08T18:30:00Z"
    mock_ref.elevation_angle = 0.5
    mock_ref.elevation_angles = [0.5]
    with patch("src.server.fetch_scan", return_value="/fake/path"), \
         patch("src.server.extract_reflectivity", return_value=mock_ref):
        resp = client.get("/summary/KTLX")
    assert resp.status_code == 200
    data = resp.json()
    assert "text" in data


def test_tracks_endpoint_returns_200():
    mock_ref = MagicMock()
    mock_ref.reflectivity = np.full((360, 500), np.nan)
    mock_ref.azimuths = np.linspace(0, 359, 360)
    mock_ref.ranges_m = np.linspace(2000, 250000, 500)
    mock_ref.radar_lat = 35.3331
    mock_ref.radar_lon = -97.2778
    mock_ref.timestamp = "2026-04-08T18:30:00Z"
    mock_ref.elevation_angle = 0.5
    mock_ref.elevation_angles = [0.5]
    with patch("src.server.fetch_scan", return_value="/fake/path"), \
         patch("src.server.extract_reflectivity", return_value=mock_ref):
        resp = client.get("/tracks/KTLX")
    assert resp.status_code == 200
    data = resp.json()
    assert "tracks" in data
    assert "active_count" in data
    assert "recent_events" in data


def test_motion_endpoint_missing_track_returns_404():
    resp = client.get("/motion/KTLX/999")
    assert resp.status_code == 404
