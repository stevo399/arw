"""Proof (compact scan history, Task 3): a compact scan rebuilds every map
layer bit-identically on real Level II volumes, and is small.

Compares the layers ARW publishes from the processed scan with the layers
built from `CompactScan.from_buffered_scan(scan).to_buffered_scan()`.
Sizes are printed for the test report (run with -s).
"""

import numpy as np
import pytest

import src.server as server
from src.history.compact_scan import CompactScan
from src.map_layer import (
    build_precipitation_field_geojson,
    build_storm_audiom_centroid_geojson,
    build_storm_centroid_geojson,
    build_storm_geojson,
    build_storm_intensity_geojson,
)
from tests.e2e.geojson_difference import first_difference

VOLUMES = [
    ("KTLX", "cache/KTLX/KTLX20260410_224209_V06"),
    ("KEMX", "cache/KEMX/KEMX20260712_022646_V06"),
    ("KIWA", "cache/KIWA/KIWA20260712_170029_V06"),
]
BUILDERS = [
    build_storm_geojson,
    build_storm_intensity_geojson,
    build_storm_audiom_centroid_geojson,
    build_storm_centroid_geojson,
    build_precipitation_field_geojson,
]


@pytest.mark.parametrize("site_id,path", VOLUMES)
def test_compact_scan_rebuilds_all_layers_identically(site_id, path):
    original = server._process_scan_file(site_id, path)
    compact = CompactScan.from_buffered_scan(original)
    rebuilt = compact.to_buffered_scan()

    assert compact.reflectivity.kind == "stepped", "real reflectivity must be on the 0.5 dBZ step"
    assert np.array_equal(
        rebuilt.reflectivity_data.reflectivity, original.reflectivity_data.reflectivity, equal_nan=True
    )
    for obj in original.detected_objects:
        assert np.array_equal(rebuilt.object_masks[obj.object_id], original.object_masks[obj.object_id])
    for builder in BUILDERS:
        difference = first_difference(builder(rebuilt), builder(original))
        assert difference is None, f"{builder.__name__}: {difference}"

    full_scan_array_bytes = (
        original.reflectivity_data.reflectivity.nbytes
        + original.labeled_grid.nbytes
        + sum(mask.nbytes for mask in original.object_masks.values())
    )
    print(
        f"\n{site_id}: objects={compact.object_count} compact_bytes={compact.nbytes} "
        f"(reflectivity={compact.reflectivity.nbytes} labels={compact.labels.nbytes} "
        f"records={len(compact.records)}) full_scan_array_bytes={full_scan_array_bytes}"
    )
    assert compact.nbytes < 1_000_000
