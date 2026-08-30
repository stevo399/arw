import numpy as np
import pyart
import pytest
from shapely.geometry import Polygon as ShapelyPolygon

from src.contours import (
    DEFAULT_SIMPLIFY_M,
    contour_mask,
    polar_vertices_to_lonlat,
    simplify_ground_metres,
)
from src.geometry import gate_coordinates
from src.parser import extract_sweep_data

REFERENCE_VOLUME = "cache/KEMX/KEMX20260712_022646_V06"


@pytest.fixture(scope="module")
def sweep():
    return extract_sweep_data(pyart.io.read_nexrad_archive(REFERENCE_VOLUME))


def test_integer_vertices_match_gate_coordinates(sweep):
    """At whole-number indices, conversion must land exactly on the gate."""
    lat, lon = gate_coordinates(
        sweep.azimuths, sweep.ranges_m, sweep.elevations,
        sweep.radar_lat, sweep.radar_lon,
    )
    rows = np.array([0.0, 10.0, 359.0, 700.0])
    cols = np.array([5.0, 100.0, 800.0, 1500.0])

    out = polar_vertices_to_lonlat(rows, cols, sweep)

    for i, (r, c) in enumerate(zip(rows.astype(int), cols.astype(int))):
        assert out[i, 0] == pytest.approx(float(lon[r, c]), abs=1e-9)
        assert out[i, 1] == pytest.approx(float(lat[r, c]), abs=1e-9)


def test_fractional_vertex_lies_between_its_neighbours(sweep):
    """A vertex at row 10.5 must sit between the row 10 and row 11 gates."""
    lat, lon = gate_coordinates(
        sweep.azimuths, sweep.ranges_m, sweep.elevations,
        sweep.radar_lat, sweep.radar_lon,
    )
    out = polar_vertices_to_lonlat(np.array([10.5]), np.array([500.0]), sweep)

    lo, hi = sorted([float(lat[10, 500]), float(lat[11, 500])])
    assert lo <= out[0, 1] <= hi


def test_azimuth_interpolation_is_seam_safe(sweep):
    """A vertex interpolated across the 0/360 wrap must not flip 180 degrees.

    The reference volume's azimuth array wraps once. Interpolating raw azimuths
    across that wrap produces a bearing 180 degrees wrong -- the same defect
    that once reported a storm due north as due south.
    """
    azimuths = np.asarray(sweep.azimuths, dtype=float)
    seam = int(np.flatnonzero(np.abs(np.diff(azimuths)) > 180.0)[0])

    just_before = polar_vertices_to_lonlat(
        np.array([float(seam)]), np.array([1000.0]), sweep
    )
    straddling = polar_vertices_to_lonlat(
        np.array([seam + 0.5]), np.array([1000.0]), sweep
    )

    # The straddling vertex must be close to its neighbour, not on the far side
    # of the radar.
    assert abs(straddling[0, 0] - just_before[0, 0]) < 0.05
    assert abs(straddling[0, 1] - just_before[0, 1]) < 0.05


def test_simplify_removes_detail_below_tolerance(sweep):
    """A wobble finer than the tolerance is flattened; the shape survives."""
    # A ~10 km square near the radar, with a 20 m zigzag along one edge.
    lat0, lon0 = sweep.radar_lat, sweep.radar_lon
    d = 0.05  # roughly 5 km
    coords = [(lon0 - d, lat0 - d), (lon0 + d, lat0 - d)]
    for i in range(20):
        coords.append((lon0 + d + (0.0002 if i % 2 else 0.0), lat0 - d + i * d / 10))
    coords += [(lon0 + d, lat0 + d), (lon0 - d, lat0 + d)]
    poly = ShapelyPolygon(coords)

    out = simplify_ground_metres(poly, lat0, lon0, 250.0)

    assert out.is_valid
    assert len(out.exterior.coords) < len(poly.exterior.coords)
    # Area must not move much: simplification removes wobble, not the shape.
    assert out.area == pytest.approx(poly.area, rel=0.05)


def test_simplify_tolerance_is_in_metres_not_degrees(sweep):
    """250 must mean 250 metres. If it were read as degrees the shape would vanish."""
    lat0, lon0 = sweep.radar_lat, sweep.radar_lon
    d = 0.05
    poly = ShapelyPolygon([
        (lon0 - d, lat0 - d), (lon0 + d, lat0 - d),
        (lon0 + d, lat0 + d), (lon0 - d, lat0 + d),
    ])

    out = simplify_ground_metres(poly, lat0, lon0, DEFAULT_SIMPLIFY_M)

    assert not out.is_empty
    assert out.area == pytest.approx(poly.area, rel=0.01)


def _mask_sweep(sweep, mask_builder):
    """Build a boolean mask shaped like the sweep, via a callback."""
    mask = np.zeros(np.asarray(sweep.reflectivity).shape, dtype=bool)
    mask_builder(mask)
    return mask


def test_contour_mask_returns_multipolygon(sweep):
    mask = _mask_sweep(sweep, lambda m: m.__setitem__((slice(100, 140), slice(300, 340)), True))
    out = contour_mask(mask, sweep)
    assert out.geom_type in ("Polygon", "MultiPolygon")
    assert out.is_valid
    assert not out.is_empty


def test_contour_mask_produces_a_hole(sweep):
    """A donut mask must yield a polygon with an interior ring.

    The convex hull fills this in completely -- which is the bug: a user
    walking into the gap would be told there is a storm there.
    """
    def build(m):
        m[100:160, 300:360] = True
        m[120:140, 320:340] = False

    out = contour_mask(_mask_sweep(sweep, build), sweep)
    polys = list(out.geoms) if out.geom_type == "MultiPolygon" else [out]
    assert sum(len(p.interiors) for p in polys) >= 1


def test_contour_mask_keeps_disjoint_blobs_separate(sweep):
    def build(m):
        m[100:120, 300:320] = True
        m[400:420, 900:920] = True

    out = contour_mask(_mask_sweep(sweep, build), sweep)
    assert out.geom_type == "MultiPolygon"
    assert len(out.geoms) == 2


def test_contour_mask_empty_mask_is_empty_not_invented(sweep):
    """No echo means no shape. The old code drew a square at the centroid."""
    out = contour_mask(np.zeros(np.asarray(sweep.reflectivity).shape, dtype=bool), sweep)
    assert out.is_empty


def test_contour_mask_joins_a_shape_across_the_seam(sweep):
    """A blob straddling due north is ONE polygon, not two.

    This defect has already appeared three times in this project: in centroid
    interpolation, in connected-component labelling, and it would appear here
    without wrap padding.
    """
    def build(m):
        m[-15:, 300:340] = True
        m[:15, 300:340] = True

    out = contour_mask(_mask_sweep(sweep, build), sweep)
    polys = list(out.geoms) if out.geom_type == "MultiPolygon" else [out]
    assert len(polys) == 1, f"seam split the shape into {len(polys)} pieces"
