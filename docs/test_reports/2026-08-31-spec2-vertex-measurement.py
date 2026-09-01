"""Vertex-count measurement script for Spec 2, Task 7, fix round 1.

Re-derives, against the current HEAD of `spec2-shapes`, the numbers used in
`docs/test_reports/2026-08-31-spec2-shape-truth.md`'s vertex-count section
(Finding 1 of `.superpowers/sdd/2026-08-29-storm-shape-contours/
task-7-fix-brief.md`) and the pre-filter figures cited by
`MIN_PRECIP_FRAGMENT_AREA_KM2`'s own comment in `src/map_layer.py`
(Finding 4 of the same brief).

THE UNIT THAT MATTERS: a map viewer loads a GeoJSON *Feature*, not a
disjoint polygon fragment inside it. `build_precipitation_field_geojson`
emits one Feature per intensity band, and that Feature's geometry can be a
MultiPolygon holding hundreds of separate patches. Counting vertices per
fragment (as the first cut of this report did) understates the per-shape
burden by roughly two orders of magnitude and, worse, inverts which layer
looks heavier. Every "per Feature" number below is: parse the Feature's
geometry with `shapely.geometry.shape`, then for every polygon in it (a
MultiPolygon has `.geoms`; a Polygon is itself) sum `len(exterior.coords)`
plus `len(ring.coords)` for every interior ring, summed across all polygons
in that one Feature. That total is one data point. One Feature = one row.
No closure-duplicate coordinate is subtracted.

Run: .venv/Scripts/python.exe docs/test_reports/2026-08-31-spec2-vertex-measurement.py
"""

import sys
from pathlib import Path

import numpy as np
import pyart
from shapely.geometry import shape as shapely_shape

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.buffer import BufferedScan
from src.detection import detect_objects_with_grid
from src.geometry import align_field_by_azimuth
import src.map_layer as map_layer
from src.map_layer import build_storm_geojson, build_precipitation_field_geojson, PRECIP_FIELD_LEVELS
from src.contours import exclusive_bands
from src.parser import extract_sweep_data, extract_velocity
from src.preprocess import preprocess_sweep
from src.velocity import detect_rotation_signatures

REFERENCE_VOLUMES = {
    "KTLX (Moore)": "cache/KTLX/KTLX20130520_195527_V06.gz",
    "KEMX": "cache/KEMX/KEMX20260712_022646_V06",
    "KIWA (clear air)": "cache/KIWA/KIWA20260712_170029_V06",
}


def _velocity_aligned(raw_sweep, vel_data):
    """Mirror of src.server._velocity_aligned_to_reflectivity (same as the
    e2e walking-truth test's own helper, kept identical for comparability)."""
    if vel_data is None or not vel_data.sweeps:
        return None
    velocity_sweep = vel_data.sweeps[0]
    if not np.array_equal(
        np.asarray(velocity_sweep.ranges_m, dtype=float),
        np.asarray(raw_sweep.ranges_m, dtype=float),
    ):
        return None
    return align_field_by_azimuth(
        velocity_sweep.velocity, velocity_sweep.azimuths, raw_sweep.azimuths
    )


def load_scan(path: str) -> BufferedScan:
    """Same pipeline as tests/e2e/test_proof_shape_truth.py::_load_scan and
    src/server.py::_ingest_to_buffer, reading a cached file directly instead
    of fetching over the network."""
    radar = pyart.io.read_nexrad_archive(str(REPO_ROOT / path))
    raw_sweep = extract_sweep_data(radar)
    vel_data = extract_velocity(radar)
    preliminary_rotations = detect_rotation_signatures(vel_data) if vel_data else []
    aligned_velocity = _velocity_aligned(raw_sweep, vel_data)
    ref_data, _quality, _advisory = preprocess_sweep(
        raw_sweep, preliminary_rotations, velocity=aligned_velocity
    )
    result = detect_objects_with_grid(
        reflectivity=ref_data.reflectivity,
        azimuths=ref_data.azimuths,
        ranges_m=ref_data.ranges_m,
        radar_lat=ref_data.radar_lat,
        radar_lon=ref_data.radar_lon,
        elevation_deg=ref_data.elevation_angle,
        elevations=ref_data.elevations,
        gate_classification=ref_data.gate_classification,
    )
    return BufferedScan(
        timestamp=ref_data.timestamp,
        site_id=path.split("/")[1],
        reflectivity_data=ref_data,
        detected_objects=result.objects,
        labeled_grid=result.labeled_grid,
        object_masks=result.object_masks,
    )


def _polygons_of(geom):
    if geom.geom_type == "Polygon":
        return [geom]
    if geom.geom_type == "MultiPolygon":
        return list(geom.geoms)
    return []


def feature_vertex_count(feature) -> int:
    """The unit that matters: total vertices across every polygon in ONE
    GeoJSON Feature's geometry."""
    geom = shapely_shape(feature["geometry"])
    total = 0
    for poly in _polygons_of(geom):
        total += len(poly.exterior.coords)
        for ring in poly.interiors:
            total += len(ring.coords)
    return total


def fragment_vertex_counts(geometry):
    """Per-fragment (disjoint polygon piece) vertex counts. NOT the unit a
    map viewer loads -- retained only to reproduce the legacy piece-count
    figure `MIN_PRECIP_FRAGMENT_AREA_KM2`'s comment cites (Finding 4) and to
    report the pre/post filter delta, which are fragment-scoped quantities
    by definition."""
    counts = []
    for poly in _polygons_of(geometry):
        n = len(poly.exterior.coords)
        for ring in poly.interiors:
            n += len(ring.coords)
        counts.append(n)
    return counts


def stats(counts):
    arr = np.array(counts)
    return dict(
        n=len(arr), min=int(arr.min()), median=float(np.median(arr)),
        p95=float(np.percentile(arr, 95)), max=int(arr.max()), total=int(arr.sum()),
    )


def main():
    print("Loading volumes...")
    scans = {label: load_scan(path) for label, path in REFERENCE_VOLUMES.items()}

    print("\n=== Finding 1: per-Feature vertex counts, current HEAD ===")
    print("\n-- storm footprint layer (build_storm_geojson), per Feature --")
    pooled_storm = []
    for label, scan in scans.items():
        storm_gj = build_storm_geojson(scan)
        counts = [feature_vertex_count(f) for f in storm_gj["features"]]
        pooled_storm.extend(counts)
        print(f"{label:<18} {stats(counts)}")
    print(f"{'POOLED':<18} {stats(pooled_storm)}")

    print("\n-- precipitation field layer (build_precipitation_field_geojson), per Feature --")
    pooled_precip = []
    for label, scan in scans.items():
        precip_gj = build_precipitation_field_geojson(scan)
        counts = [feature_vertex_count(f) for f in precip_gj["features"]]
        pooled_precip.extend(counts)
        print(f"{label:<18} {stats(counts)}")
        print(f"   omittedFragmentCount={precip_gj['metadata']['omittedFragmentCount']} "
              f"omittedFragmentAreaKm2={precip_gj['metadata']['omittedFragmentAreaKm2']}")
    print(f"{'POOLED':<18} {stats(pooled_precip)}")

    # --- pre/post filter delta, pooled across all 3 volumes (fragment-level
    # totals; the vertex TOTAL is convention-invariant -- same whether summed
    # per-Feature or per-fragment -- but fragment COUNT is fragment-scoped) ---
    original_threshold = map_layer.MIN_PRECIP_FRAGMENT_AREA_KM2
    print(f"\ncurrent MIN_PRECIP_FRAGMENT_AREA_KM2 = {original_threshold}")

    map_layer.MIN_PRECIP_FRAGMENT_AREA_KM2 = -1.0  # disable the drop entirely
    try:
        pre_total_vertices = 0
        pre_fragment_count = 0
        for label, scan in scans.items():
            gj = build_precipitation_field_geojson(scan)
            for f in gj["features"]:
                counts = fragment_vertex_counts(shapely_shape(f["geometry"]))
                pre_total_vertices += sum(counts)
                pre_fragment_count += len(counts)
    finally:
        map_layer.MIN_PRECIP_FRAGMENT_AREA_KM2 = original_threshold

    post_total_vertices = 0
    post_fragment_count = 0
    dropped_count = 0
    dropped_area_km2 = 0.0
    for label, scan in scans.items():
        gj = build_precipitation_field_geojson(scan)
        for f in gj["features"]:
            counts = fragment_vertex_counts(shapely_shape(f["geometry"]))
            post_total_vertices += sum(counts)
            post_fragment_count += len(counts)
        dropped_count += gj["metadata"]["omittedFragmentCount"]
        dropped_area_km2 += gj["metadata"]["omittedFragmentAreaKm2"]

    print("\n=== filter delta (pooled across all 3 volumes) ===")
    print(f"pre-filter:  {pre_fragment_count} fragments, {pre_total_vertices} vertices")
    print(f"post-filter: {post_fragment_count} fragments, {post_total_vertices} vertices")
    pct = 100.0 * (post_total_vertices - pre_total_vertices) / pre_total_vertices
    print(f"vertex change: {pct:.1f}%")
    print(f"dropped fragments (sum of metadata): {dropped_count}")
    print(f"dropped fragment area km2 (sum of metadata): {round(dropped_area_km2, 1)}")

    # --- Finding 4: KTLX-only, pre-filter piece count + vertex total (the
    # figure MIN_PRECIP_FRAGMENT_AREA_KM2's own comment cites) ---
    map_layer.MIN_PRECIP_FRAGMENT_AREA_KM2 = -1.0
    try:
        ktlx_scan = scans["KTLX (Moore)"]
        ktlx_gj = build_precipitation_field_geojson(ktlx_scan)
        ktlx_pieces = 0
        ktlx_vertices = 0
        for f in ktlx_gj["features"]:
            counts = fragment_vertex_counts(shapely_shape(f["geometry"]))
            ktlx_pieces += len(counts)
            ktlx_vertices += sum(counts)
    finally:
        map_layer.MIN_PRECIP_FRAGMENT_AREA_KM2 = original_threshold

    print("\n=== Finding 4: KTLX-only pre-filter piece count + vertex total ===")
    print(f"pieces={ktlx_pieces} vertices={ktlx_vertices}")

    # Empirical check the comment also makes: raising DEFAULT_SIMPLIFY_M to
    # 2000 m does not merge pieces (simplification smooths outlines, it does
    # not dissolve fragment boundaries).
    ktlx_sweep = ktlx_scan.reflectivity_data
    bands_2000m = exclusive_bands(ktlx_sweep.reflectivity, PRECIP_FIELD_LEVELS, ktlx_sweep, simplify_m=2000.0)
    pieces_2000m = sum(
        len(_polygons_of(geom)) for geom in bands_2000m.values() if not geom.is_empty
    )
    print(f"KTLX pieces at 2000 m simplify tolerance (sanity check): {pieces_2000m}")


if __name__ == "__main__":
    main()
