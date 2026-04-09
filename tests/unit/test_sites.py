import math
from src.sites import (
    NEXRAD_SITES,
    geocode_city_state,
    haversine_distance_km,
    compute_beam_height_m,
    rank_sites,
)


def test_nexrad_sites_has_entries():
    assert len(NEXRAD_SITES) > 150
    ktlx = next(s for s in NEXRAD_SITES if s["site_id"] == "KTLX")
    assert ktlx["name"] == "Oklahoma City"
    assert abs(ktlx["latitude"] - 35.3331) < 0.01
    assert abs(ktlx["longitude"] - (-97.2778)) < 0.01
    assert ktlx["elevation_m"] > 0


def test_haversine_distance_km():
    # OKC to Dallas is roughly 300 km
    dist = haversine_distance_km(35.4676, -97.5164, 32.7767, -96.7970)
    assert 290 < dist < 320


def test_compute_beam_height_m():
    # At 100 km distance, 0.5 deg elevation, ~0m radar elevation
    height = compute_beam_height_m(distance_km=100.0, radar_elevation_m=0.0)
    # Should be roughly 1600-1700 m (geometric + earth curvature)
    assert 1400 < height < 2000


def test_compute_beam_height_m_increases_with_distance():
    h1 = compute_beam_height_m(distance_km=50.0, radar_elevation_m=0.0)
    h2 = compute_beam_height_m(distance_km=200.0, radar_elevation_m=0.0)
    assert h2 > h1


def test_compute_beam_height_m_includes_radar_elevation():
    h1 = compute_beam_height_m(distance_km=100.0, radar_elevation_m=0.0)
    h2 = compute_beam_height_m(distance_km=100.0, radar_elevation_m=500.0)
    assert h2 - h1 == 500.0


def test_rank_sites_returns_sorted_by_beam_height():
    results = rank_sites(lat=35.4676, lon=-97.5164)
    assert len(results) > 0
    # Check sorted ascending by beam_height_m
    for i in range(len(results) - 1):
        assert results[i]["beam_height_m"] <= results[i + 1]["beam_height_m"]


def test_rank_sites_filters_high_beam_height():
    results = rank_sites(lat=35.4676, lon=-97.5164)
    for r in results:
        assert r["beam_height_m"] <= 10000.0


def test_rank_sites_includes_distance():
    results = rank_sites(lat=35.4676, lon=-97.5164)
    for r in results:
        assert r["distance_km"] > 0
