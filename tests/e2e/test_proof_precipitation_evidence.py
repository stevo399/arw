"""Proof (defect 1, 2026-09-12): the live pipeline's precipitation layer keeps
its dual-polarization evidence after the pipeline releases the dual-pol grids.

Before the fix, `_process_scan_file` set rhohv/zdr/gate_classification to
None and the layer, rendered later from that scan, reported every band as
median_rhohv None / "uncertain".  This compares the layer the live pipeline
publishes with the layer built from the same sweep before release.
"""

import dataclasses

import pytest

import src.server as server
from src.buffer import BufferedScan
from src.map_layer import build_precipitation_field_geojson
from tests.e2e.geojson_difference import first_difference

VOLUMES = [
    ("KTLX", "cache/KTLX/KTLX20260410_224209_V06"),
    ("KEMX", "cache/KEMX/KEMX20260712_022646_V06"),
]


@pytest.mark.parametrize("site_id,path", VOLUMES)
def test_precipitation_evidence_survives_dual_pol_release(monkeypatch, site_id, path):
    captured = {}
    original = server.refresh_quality_advisory

    def capture(*args, **kwargs):
        sweep, quality, advisory = original(*args, **kwargs)
        # A shallow copy keeps the grids even after the server sets the
        # original object's attributes to None.
        captured["sweep"] = dataclasses.replace(sweep)
        return sweep, quality, advisory

    monkeypatch.setattr(server, "refresh_quality_advisory", capture)
    released = server._process_scan_file(site_id, path)

    assert released.reflectivity_data.rhohv is None
    assert released.reflectivity_data.gate_classification is None
    unreleased = BufferedScan(
        timestamp=released.timestamp,
        site_id=released.site_id,
        reflectivity_data=captured["sweep"],
        detected_objects=released.detected_objects,
        labeled_grid=released.labeled_grid,
        object_masks=released.object_masks,
    )
    expected = build_precipitation_field_geojson(unreleased)
    actual = build_precipitation_field_geojson(released)

    difference = first_difference(actual, expected)
    assert difference is None, difference
    assert expected["features"]
    for feature in actual["features"]:
        properties = feature["properties"]
        assert properties["median_rhohv"] is not None, properties["id"]
        assert properties["median_zdr"] is not None, properties["id"]
        assert properties["class_fractions"], properties["id"]
