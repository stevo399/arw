import numpy as np
import pyart
import pytest
from pyart.core.transforms import geographic_to_cartesian_aeqd
from shapely.geometry import Point, Polygon as ShapelyPolygon
from shapely.ops import transform as shapely_transform, unary_union

from src.contours import (
    DEFAULT_SIMPLIFY_M,
    _repair_if_invalid,
    contour_field,
    contour_mask,
    exclusive_bands,
    polar_vertices_to_lonlat,
    simplify_ground_metres,
)
from src.geometry import gate_areas_km2, gate_coordinates, gate_latlon
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


def _area_km2(geom, sweep):
    if geom is None or geom.is_empty:
        return 0.0
    projected = shapely_transform(
        lambda x, y: geographic_to_cartesian_aeqd(
            np.asarray(x), np.asarray(y), sweep.radar_lon, sweep.radar_lat
        ),
        geom,
    )
    return projected.area / 1e6


def _probe(sweep, ray: int, gate: int, range_gates: float = 0.0, beamwidths: float = 0.0):
    """A lon/lat point offset from one gate's centre in gate-widths.

    `range_gates` is measured in whole range bins along the beam and
    `beamwidths` in whole ray spacings across it, so 0.45 lands just inside
    that gate's own edge and 0.6 lands outside it, in the neighbouring cell.
    """
    ranges_m = np.asarray(sweep.ranges_m, dtype=float)
    azimuths = np.asarray(sweep.azimuths, dtype=float)
    gate_depth_m = float(np.median(np.diff(ranges_m)))
    beamwidth_deg = float(np.median(np.abs(np.diff(np.unwrap(azimuths, period=360.0)))))
    lat, lon = gate_latlon(
        azimuth_deg=float(azimuths[ray]) + beamwidths * beamwidth_deg,
        range_m=float(ranges_m[gate]) + range_gates * gate_depth_m,
        elevation_deg=float(np.asarray(sweep.elevations, dtype=float)[ray]),
        radar_lat=sweep.radar_lat,
        radar_lon=sweep.radar_lon,
    )
    return Point(lon, lat)


def test_isolated_gate_contours_to_its_gate_edge_not_its_gate_centre(sweep):
    """A single no-data-surrounded echoing gate must be drawn out to its edges.

    A gate is a patch of ground, not a point. The boundary between an
    echoing gate and a gate the radar reported nothing for belongs half a
    range gate out along the beam and half a beamwidth out across it -- at
    that gate's own outer edge.

    Before `_fill_no_data`, no-data was a flat -9999.0, which drove the
    marching-squares crossing fraction to essentially zero and collapsed the
    boundary onto the echoing gate's CENTRE: an isolated 18 dBZ gate drew
    4.7e-8 km2 against a 0.4383 km2 ground footprint, and every shape in the
    system was inset by half a gate all the way round its no-data boundary.
    That is silence about weather that is there, which for a user who reads
    this map by walking it is the worst failure direction available.

    The four inside probes below all sit within the gate's own footprint and
    all fell outside the pre-fix polygon.
    """
    ray, gate = 200, 600
    field = np.full(np.asarray(sweep.reflectivity).shape, np.nan)
    field[ray, gate] = 18.0

    drawn = contour_field(field, [15.0], sweep, simplify_m=0.0)[15.0]
    assert not drawn.is_empty

    for label, probe in (
        ("outward along the beam", _probe(sweep, ray, gate, range_gates=0.45)),
        ("inward along the beam", _probe(sweep, ray, gate, range_gates=-0.45)),
        ("clockwise across the beam", _probe(sweep, ray, gate, beamwidths=0.45)),
        ("anticlockwise across the beam", _probe(sweep, ray, gate, beamwidths=-0.45)),
    ):
        assert drawn.contains(probe), (
            f"{label}: a point still inside this gate's own footprint is outside "
            "the drawn shape -- the no-data boundary has collapsed back toward "
            "the gate centre"
        )

    for label, probe in (
        ("outward along the beam", _probe(sweep, ray, gate, range_gates=0.7)),
        ("clockwise across the beam", _probe(sweep, ray, gate, beamwidths=0.7)),
    ):
        assert not drawn.contains(probe), (
            f"{label}: the shape reaches into a neighbouring cell the radar "
            "reported nothing for"
        )


def test_isolated_gate_field_contour_matches_its_binary_mask_contour(sweep):
    """The echo/no-data boundary must behave exactly like an echo/no-echo one.

    `contour_mask` contours a binary mask, where 0 already puts the crossing
    half a gate out. That is the reference behaviour: contouring a FIELD
    whose only echo is one gate must produce the same shape as contouring the
    mask of that same gate. Pre-fix the field path produced 4.7e-8 km2 and
    the mask path 0.2310 km2 for the same gate -- a factor of 5 million
    between two functions in the same module describing the same gate.

    HONEST LIMIT, recorded here rather than in a report nobody opens: both
    numbers are close to HALF the gate's true 0.4383 km2 ground footprint,
    because marching squares cuts each convex corner of a cell diagonally.
    An isolated gate is drawn as the diamond inscribed in its footprint, so
    its area is ~0.5x that footprint however the no-data boundary is placed;
    an n x n block loses 0.5 of one gate in total, so the error vanishes with
    size (measured on this geometry: 2x2 0.86, 3x3 0.92, 5x5 0.99, 20x20
    1.00). This test therefore pins the two paths to each other and to the
    diamond, and does NOT claim full-footprint area agreement for a single
    gate, which marching squares cannot deliver at any sentinel value.
    """
    ray, gate = 200, 600
    field = np.full(np.asarray(sweep.reflectivity).shape, np.nan)
    field[ray, gate] = 18.0
    mask = np.zeros(field.shape, dtype=bool)
    mask[ray, gate] = True

    field_area = _area_km2(contour_field(field, [15.0], sweep, simplify_m=0.0)[15.0], sweep)
    mask_area = _area_km2(contour_mask(mask, sweep, simplify_m=0.0), sweep)
    footprint = float(
        gate_areas_km2(sweep.azimuths, sweep.ranges_m, sweep.elevation_angle)[gate]
    )

    assert field_area == pytest.approx(mask_area, rel=0.05)
    assert field_area == pytest.approx(0.5 * footprint, rel=0.05)


def test_echo_in_the_last_range_bin_reaches_that_bin_s_outer_edge(sweep):
    """Echo at the end of the range axis must not stop dead on the gate centre.

    contourpy cannot place a vertex outside the array it was handed, so
    without the no-data column `_pad_for_contouring` adds at each end of the
    range axis, a shape reaching the first or last range bin was cut off at
    that bin's centre. The matching coordinate extension lives in
    `_padded_ranges`; without it `np.interp` would clamp the vertex straight
    back onto the same centre.
    """
    n_gates = np.asarray(sweep.reflectivity).shape[1]
    for gate in (0, n_gates - 1):
        field = np.full(np.asarray(sweep.reflectivity).shape, np.nan)
        field[195:205, gate] = 25.0
        drawn = contour_field(field, [15.0], sweep, simplify_m=0.0)[15.0]
        outward = 0.45 if gate else -0.45
        assert drawn.contains(_probe(sweep, 200, gate, range_gates=outward)), (
            f"gate {gate}: the shape stops at the outermost bin's centre "
            "instead of its edge"
        )


def test_inner_range_padding_never_produces_a_negative_range(sweep):
    """Half a gate inside the first range bin must stay a real, positive range."""
    from src.contours import _padded_ranges

    padded, index = _padded_ranges(sweep)
    assert np.all(np.diff(padded) > 0), "extended range axis is not increasing"
    assert padded[0] >= 0.0
    assert index[0] == -1.0 and index[-1] == len(np.asarray(sweep.ranges_m))
    # The innermost vertex a contour can produce sits at index -0.5.
    assert float(np.interp(-0.5, index, padded)) >= 0.0


def test_field_contour_joins_a_blob_across_the_seam_with_the_right_area(sweep):
    """The seam, re-verified for the FIELD path after the padding change.

    `_pad_for_contouring` now pads the range axis as well as the ray axis and
    computes a neighbour maximum that must treat azimuth as periodic. This
    project has had four separate defects at the 0/360 seam, one of which
    reported a storm due north as due south, so a padding change is
    re-checked here rather than assumed: a blob straddling due north must
    come back as ONE polygon carrying the ground area its gates cover.
    """
    shape = np.asarray(sweep.reflectivity).shape
    field = np.full(shape, np.nan)
    field[-15:, 300:340] = 25.0
    field[:15, 300:340] = 25.0

    drawn = contour_field(field, [15.0], sweep, simplify_m=0.0)[15.0]
    polys = list(drawn.geoms) if drawn.geom_type == "MultiPolygon" else [drawn]
    assert len(polys) == 1, f"seam split the shape into {len(polys)} pieces"

    per_bin = gate_areas_km2(sweep.azimuths, sweep.ranges_m, sweep.elevation_angle)
    gate_summed = float(np.broadcast_to(per_bin[None, :], shape)[np.isfinite(field)].sum())
    assert _area_km2(drawn, sweep) == pytest.approx(gate_summed, rel=0.05)


def _ramp_field(sweep):
    """A field rising from 10 to 60 dBZ across a block, NaN elsewhere."""
    field = np.full(np.asarray(sweep.reflectivity).shape, np.nan)
    block = field[100:200, 300:400]
    field[100:200, 300:400] = np.linspace(10.0, 60.0, block.shape[1])[None, :]
    return field


def test_contour_field_levels_are_nested(sweep):
    """Higher levels must sit inside lower ones: 40+ is contained by 30+."""
    result = contour_field(_ramp_field(sweep), [20.0, 30.0, 40.0], sweep)
    assert result[40.0].area < result[30.0].area < result[20.0].area
    assert result[30.0].buffer(1e-9).contains(result[40.0])


def test_exclusive_bands_do_not_overlap(sweep):
    """Walking must place the user in exactly one band."""
    bands = exclusive_bands(_ramp_field(sweep), [20.0, 30.0, 40.0], sweep)
    geoms = list(bands.values())
    # A band silently going empty would trivially satisfy "no overlap" while
    # hiding weather from the user -- rule that out explicitly.
    assert all(not g.is_empty for g in geoms), "a band must not be silently empty"
    for i, a in enumerate(geoms):
        for b in geoms[i + 1:]:
            assert a.intersection(b).area < 1e-12


def test_exclusive_bands_tile_the_cumulative_region(sweep):
    """The bands together must cover exactly the 20+ region, no more, no less."""
    field = _ramp_field(sweep)
    cumulative = contour_field(field, [20.0], sweep)[20.0]
    bands = exclusive_bands(field, [20.0, 30.0, 40.0], sweep)
    union = unary_union(list(bands.values()))
    assert union.area == pytest.approx(cumulative.area, rel=0.01)


def test_exclusive_band_has_a_hole_where_the_next_band_sits(sweep):
    """A 20-30 band wrapped around a 30+ core is an annulus, not a filled blob.

    The convex hull fills this middle in -- so today, walking from the edge
    inward, the user passes through overlapping claims about the same ground.
    """
    field = np.full(np.asarray(sweep.reflectivity).shape, np.nan)
    field[100:200, 300:400] = 25.0
    field[130:170, 330:370] = 45.0

    bands = exclusive_bands(field, [20.0, 40.0], sweep)
    low = bands[(20.0, 40.0)]
    polys = list(low.geoms) if low.geom_type == "MultiPolygon" else [low]
    assert sum(len(p.interiors) for p in polys) >= 1


def test_highest_band_is_open_ended(sweep):
    bands = exclusive_bands(_ramp_field(sweep), [20.0, 30.0], sweep)
    assert (30.0, float("inf")) in bands


def test_contour_field_all_nan_returns_multipolygon(sweep):
    """No data anywhere must still come back as MultiPolygon, not GeometryCollection.

    unary_union of an all-empty piece list returns GEOMETRYCOLLECTION EMPTY,
    not MULTIPOLYGON EMPTY. Still empty (no invented geometry), but the wrong
    type would serialise to a different GeoJSON "type" for downstream
    consumers than `exclusive_bands` produces for the same input.
    """
    field = np.full(np.asarray(sweep.reflectivity).shape, np.nan)
    result = contour_field(field, [20.0, 30.0, 40.0], sweep)
    for level, region in result.items():
        assert region.geom_type == "MultiPolygon", f"level {level} was {region.geom_type}"
        assert region.is_empty


def test_simplify_falls_back_to_raw_bands_when_it_would_empty_one(sweep, monkeypatch, caplog):
    """A band that had real area must never be simplified out of existence.

    `shapely.coverage_simplify` is documented as assuming a valid polygonal
    coverage and leaving its result undefined otherwise, and
    `shapely.coverage_is_valid` reports False on real volumes here (traced in
    `_raw_bands`'s docstring to outer-boundary edges, which is benign -- but
    benign-today is not a guarantee). If the operation ever does go wrong,
    the user-visible failure is a band vanishing: weather present on the
    radar, absent from the map, with nothing said about it.

    This forces that outcome and asserts the module discards the whole
    simplification pass and returns the exact unsimplified bands instead.
    """
    import src.contours as contours

    field = np.full(np.asarray(sweep.reflectivity).shape, np.nan)
    field[100:200, 300:400] = 25.0
    field[130:170, 330:370] = 45.0
    levels = [20.0, 40.0]

    raw = contours._raw_bands(field, levels, sweep)
    assert all(g.area > 0 for g in raw.values()), "fixture must give both bands area"

    monkeypatch.setattr(
        contours, "coverage_simplify",
        lambda geoms, tolerance: [ShapelyPolygon() for _ in geoms],
    )
    with caplog.at_level("WARNING", logger="src.contours"):
        out = contours.exclusive_bands(field, levels, sweep)

    assert [r.getMessage() for r in caplog.records], "the fallback must say so"
    assert "falling back to unsimplified bands" in caplog.records[0].getMessage()
    for key, geom in out.items():
        assert not geom.is_empty, f"band {key} was lost"
        assert geom.geom_type == "MultiPolygon"
        assert geom.area == pytest.approx(raw[key].area, rel=1e-12)


class _FakeInvalidGeom:
    """A minimal stand-in for a shapely geometry that is invalid, reports a
    non-zero pre-repair area, and collapses to empty under buffer(0).

    Neither this task nor the prior review round could construct a real
    difference/simplify result that actually exercises this path on live
    radar data, so the warning path is exercised directly against a
    contrived input satisfying `_repair_if_invalid`'s three preconditions
    (invalid, non-zero area, buffer(0) empties it) rather than against real
    geometry.
    """

    is_valid = False
    area = 1.23456

    def buffer(self, distance):
        return _FakeEmptyGeom()


class _FakeEmptyGeom:
    is_empty = True


def test_repair_if_invalid_warns_when_buffer_zero_empties_real_content(caplog):
    """A band that had real area before repair must not vanish in silence."""
    with caplog.at_level("WARNING", logger="src.contours"):
        result = _repair_if_invalid(_FakeInvalidGeom(), (20.0, 30.0))

    assert result.is_empty
    assert len(caplog.records) == 1
    message = caplog.records[0].getMessage()
    assert caplog.records[0].levelname == "WARNING"
    assert "(20.0, 30.0)" in message
    assert "1.23456" in message
