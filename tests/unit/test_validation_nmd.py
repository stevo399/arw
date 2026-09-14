from datetime import datetime
from pathlib import Path

import pytest

from src.sites import haversine_distance_km
from src.validation.nmd import parse_nmd

FIXTURES = Path("tests/fixtures/level3")
KMVX = (47.5278, -97.3256)


def test_a_product_with_two_mesocyclones_is_parsed_from_its_table():
    when, detections = parse_nmd((FIXTURES / "MVX_NMD_2026_09_12_00_05_32").read_bytes(), "KMVX")
    assert when == datetime(2026, 9, 12, 0, 5, 32)
    assert [(d.circulation_id, d.azimuth_deg, d.strength_rank, d.storm_id, d.tvs) for d in detections] == [
        ("916", 147.0, 6, "T1", False),
        ("872", 123.0, 5, "B1", False),
    ]
    assert detections[0].range_km == pytest.approx(110 * 1.852)
    # 204 km at 147 degrees from KMVX lies to the SSE.
    assert detections[0].latitude < KMVX[0] and detections[0].longitude > KMVX[1]
    assert haversine_distance_km(*KMVX, detections[0].latitude, detections[0].longitude) == pytest.approx(203.7, abs=2.0)


def test_a_product_without_detections_has_none():
    when, detections = parse_nmd((FIXTURES / "TLX_NMD_2026_04_10_05_55_23").read_bytes(), "KTLX")
    assert detections == []
