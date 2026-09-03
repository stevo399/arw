"""Turn polar radar fields into geographic polygons.

This module knows nothing about storms, GeoJSON or Audiom. It takes an array
laid out as (rays x gates) and returns shapely geometry in lon/lat.

Contouring happens on the polar grid itself rather than on a resampled
Cartesian grid, so the radar's own resolution is preserved: no single grid
spacing can match a radar at both 20 km and 200 km, and inventing detail at
range is the worse failure for a user exploring a shape by touch.
"""

import logging

import numpy as np
from contourpy import contour_generator
from shapely import coverage_simplify
from pyart.core.transforms import (
    antenna_to_cartesian,
    cartesian_to_geographic_aeqd,
    geographic_to_cartesian_aeqd,
)
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import transform, unary_union

logger = logging.getLogger(__name__)

# Douglas-Peucker simplification with tolerance t may displace any vertex by
# up to t. A tolerance equal to the full 250 m gate depth can therefore
# displace a vertex enough to swallow a gate-wide notch whole -- the
# displacement bound has to stay a fraction of the gate, not equal to it.
# 100 m keeps displacement under half a gate while still discarding
# sub-gate jitter that isn't real structure.
DEFAULT_SIMPLIFY_M = 100.0

# Rows of wrap-around padding added before contouring so a shape crossing due
# north is not severed by the array edge. Marching squares needs one row of
# overlap; two is defensive.
SEAM_PAD_RAYS = 2

# Value used for a no-data gate that has no finite neighbour at all. Such a
# gate is never on a level crossing (every gate it touches is also no-data),
# so the only requirement is that it sit unambiguously below every real
# level. See `_fill_no_data` for the gates that ARE on a crossing, which do
# not use this value.
NO_DATA_FLOOR = -9999.0

# Level membership is HALF-OPEN AND CLOSED AT THE BOTTOM: a gate belongs to
# a level if its reflectivity is >= that level. That is the convention the
# rest of the system already uses -- `src.map_layer._band_gate_mask`,
# `src.detection.classify_intensity`, `src.shape_truth.measure_walk_truth`
# all bucket with `>=` -- so the geometry is what had to move.
#
# contourpy's `filled(level, inf)` is strict: a value sitting exactly ON the
# level is treated as on the boundary, not inside, and `filled(20.0, inf)`
# over a block of exact 20.0s returns zero rings. NEXRAD reflectivity is
# quantized to 0.5 dBZ, so exact hits on the integer band levels are not a
# corner case: 5,912 gates on KTLX and 8,748 on KEMX sit exactly on a band
# level, 6.5% and 7.8% of their >= 15 dBZ ground. Every one of those was in
# the evidence a band reports (`_band_evidence`) and outside the polygon
# that evidence was attached to.
#
# Contouring at `level - LEVEL_MEMBERSHIP_EPSILON` makes the geometry agree.
# The epsilon has to be small enough not to reach the next quantum down --
# any value in (0, 0.5) does that for quantized data -- and large enough to
# survive float64 arithmetic at these magnitudes, where one ulp near 60 dBZ
# is ~7e-15. 1e-6 dBZ is two millionths of the smallest real difference the
# data can express, so it shifts an interpolated boundary between two gates
# differing by even 1 dBZ by one part in a million of a gate.
LEVEL_MEMBERSHIP_EPSILON = 1e-6


def _contour_level(level: float) -> float:
    """The value to hand contourpy so that `>= level` means inside."""
    return float(level) - LEVEL_MEMBERSHIP_EPSILON


def _padded_axes(sweep):
    """Ray-axis arrays extended past the seam, kept monotonic in azimuth.

    Azimuth is unwrapped before padding so that interpolation across the
    0/360 boundary is continuous. Callers fold the result back with % 360.
    """
    azimuths = np.unwrap(np.asarray(sweep.azimuths, dtype=float), period=360.0)
    elevations = np.asarray(sweep.elevations, dtype=float)
    padded_azimuths = np.concatenate([azimuths, azimuths[:SEAM_PAD_RAYS] + 360.0])
    padded_elevations = np.concatenate([elevations, elevations[:SEAM_PAD_RAYS]])
    return padded_azimuths, padded_elevations


def _padded_ranges(sweep):
    """Range axis extended one gate beyond each end of the sweep.

    Contouring runs on a field padded with a ring of no-data one gate outside
    the first and last range bins (`_pad_for_contouring`), so a contour
    vertex can legitimately land at gate index -0.5 or n_gates - 0.5 -- half
    a gate beyond the outermost gate centre, which is that gate's own outer
    edge. `np.interp` clamps outside its `xp` range, so without extending the
    axis those vertices would collapse straight back onto the first/last gate
    centre, which is exactly the half-gate inset this padding exists to
    remove.

    The inner extension is clamped at zero: a range gate half a gate inside
    the first gate centre must never come out negative, and must never be
    allowed to wrap through the radar origin into the opposite azimuth.
    """
    ranges_m = np.asarray(sweep.ranges_m, dtype=float)
    if len(ranges_m) < 2:
        return ranges_m, np.arange(len(ranges_m), dtype=float)
    inner = max(float(ranges_m[0] - (ranges_m[1] - ranges_m[0])), 0.0)
    outer = float(ranges_m[-1] + (ranges_m[-1] - ranges_m[-2]))
    return (
        np.concatenate([[inner], ranges_m, [outer]]),
        np.arange(-1.0, len(ranges_m) + 1.0),
    )


def _neighbour_max(field: np.ndarray) -> np.ndarray:
    """Largest finite value among each gate's four neighbours, else NaN.

    The azimuth axis is periodic -- ray 0 and ray N-1 both point just either
    side of due north -- so it is rolled, not edge-padded. The range axis is
    not periodic: beyond the first and last range bin there is genuinely
    nothing, which is what `_pad_for_contouring` then represents explicitly.

    Four-neighbour, not eight: marching squares computes a level crossing on
    the axis-aligned edges of each quad, so only axis-aligned neighbours can
    ever produce one.
    """
    field = np.asarray(field, dtype=float)
    neighbours = np.full((4,) + field.shape, np.nan)
    neighbours[0] = np.roll(field, 1, axis=0)
    neighbours[1] = np.roll(field, -1, axis=0)
    neighbours[2, :, 1:] = field[:, :-1]
    neighbours[3, :, :-1] = field[:, 1:]
    finite = np.isfinite(neighbours)
    if not finite.any():
        return np.full(field.shape, np.nan)
    largest = np.max(np.where(finite, neighbours, -np.inf), axis=0)
    return np.where(np.isfinite(largest), largest, np.nan)


def _seam_padded(array: np.ndarray) -> np.ndarray:
    return np.vstack([array, array[:SEAM_PAD_RAYS]])


def _pad_for_contouring(field: np.ndarray):
    """Field and neighbour-max arrays padded for contouring.

    Two paddings are applied:

    - `SEAM_PAD_RAYS` wrap-around rows, so a shape crossing due north is not
      severed by the array edge (unchanged behaviour).
    - one column of no-data at each end of the range axis, so echo in the
      first or last range bin contours out to that bin's own edge instead of
      stopping dead on its centre. `_padded_ranges` supplies the matching
      coordinate extension.

    Returned column index c corresponds to gate index c - 1.

    The neighbour-max array is returned alongside because `_fill_no_data`
    needs it once per level and it does not depend on the level.
    """
    field = np.asarray(field, dtype=float)
    neighbour_max = _neighbour_max(field)

    padded_field = _seam_padded(field)
    padded_neighbours = _seam_padded(neighbour_max)

    edge = np.full((padded_field.shape[0], 1), np.nan)
    padded_field = np.hstack([edge, padded_field, edge])
    # The two synthetic no-data columns each have exactly one real
    # neighbour -- the range bin they sit against -- so their neighbour-max
    # is that bin's own value, which is what puts the contour half a gate
    # outside it rather than on it.
    padded_neighbours = np.hstack([
        _seam_padded(field[:, :1]),
        padded_neighbours,
        _seam_padded(field[:, -1:]),
    ])
    return padded_field, padded_neighbours


def _fill_no_data(padded_field: np.ndarray, neighbour_max: np.ndarray, level: float):
    """Replace no-data with a value that puts the level crossing on the gate edge.

    A gate is a physical patch of ground, not a point. The boundary between
    an echoing gate and a gate the radar reported nothing for is a hard mask
    edge, and it belongs half a gate out from the echoing gate's centre --
    at that gate's own outer edge -- not on the centre itself.

    Marching squares places a crossing between two nodes `v` and `s` at the
    fraction `(v - level) / (v - s)`. The previous implementation used a flat
    `-9999.0` for no-data, which drives that fraction to essentially zero:
    the boundary collapsed onto the last echoing gate's CENTRE, insetting
    every shape by half a gate all the way round its no-data boundary, and
    shrinking an isolated echoing gate to almost nothing (measured: 4.7e-8
    km2 for an isolated 18 dBZ gate whose ground footprint is 0.4383 km2).

    Reflecting the neighbour's value about the level -- `s = 2*level - v` --
    puts that fraction at exactly 0.5, which is the gate edge.

    A no-data gate can border several echoing gates with different values,
    and only one of them can be matched exactly. The LARGEST is used, which
    makes every crossing fraction <= 0.5: the region can then reach a
    neighbouring gate's edge but never cross into a cell the radar reported
    nothing for. Using the smallest (or the mean) would recover a further
    0.5-2% of area on the cached volumes but would let the boundary run up
    to half a gate into unmeasured ground, which is the fabrication this
    codebase refuses elsewhere. Measured pooled area ratios against
    gate-summed ground, KTLX/KEMX: max 0.9922/1.0084, mean 0.9968/-, min
    1.0037/-.

    Gates whose neighbour-max is not above the level cannot be on a crossing
    at all, so they take `NO_DATA_FLOOR`; this also keeps the sentinel
    strictly below the level, so no-data can never read as echo.
    """
    sentinel = np.where(
        neighbour_max > level, 2.0 * float(level) - neighbour_max, NO_DATA_FLOOR
    )
    return np.where(np.isfinite(padded_field), padded_field, sentinel)


def polar_vertices_to_lonlat(rows, cols, sweep) -> np.ndarray:
    """Convert fractional (ray, gate) index positions to lon/lat.

    Contour vertices land between gate centres, so azimuth, range and
    elevation are all interpolated. Row indices may exceed the ray count when
    seam padding is in play; the padded azimuth axis handles that
    continuously. Column indices may likewise fall just outside
    [0, n_gates - 1] when the range-edge no-data padding is in play; the
    extended range axis (`_padded_ranges`) handles that, and is the reason
    this must not fall back on `np.interp`'s clamping.

    Computes each vertex directly with `antenna_to_cartesian` /
    `cartesian_to_geographic_aeqd` on paired 1-D arrays, rather than routing
    through `src.geometry.gate_coordinates` and taking the diagonal of its
    meshed output. Measured on a real storm mask (KTLX 2013-05-20 19:55:27),
    a single contour ring of 8,921 vertices drove the meshed diagonal
    approach to ~8.9 GB peak memory; the full mask's ~24,800 vertices would
    have needed a ~9.8 GB (n x n x 2 x float64) mesh just for the output
    arrays. `antenna_to_cartesian` and `cartesian_to_geographic_aeqd` are
    both already elementwise over equal-length arrays, so calling them on
    paired (not meshed) arrays here costs O(n), not O(n^2).
    """
    padded_azimuths, padded_elevations = _padded_axes(sweep)
    padded_ranges, gate_index = _padded_ranges(sweep)

    ray_index = np.arange(len(padded_azimuths), dtype=float)

    azimuth = np.interp(rows, ray_index, padded_azimuths) % 360.0
    elevation = np.interp(rows, ray_index, padded_elevations)
    range_m = np.interp(cols, gate_index, padded_ranges)

    range_km = range_m / 1000.0
    x, y, _z = antenna_to_cartesian(range_km, azimuth, elevation)
    lon, lat = cartesian_to_geographic_aeqd(x, y, sweep.radar_lon, sweep.radar_lat)
    return np.column_stack([lon, lat])


def simplify_ground_metres(geom, radar_lat: float, radar_lon: float, tolerance_m: float):
    """Simplify a lon/lat geometry with a tolerance expressed in ground metres.

    Shapely simplifies in whatever units it is handed. Degrees are wrong here:
    a degree of longitude shrinks with latitude, so a degree tolerance is
    anisotropic and varies across the sweep. Projecting to a local
    azimuthal-equidistant frame centred on the radar makes the units metres.

    Py-ART's `geographic_to_cartesian_aeqd(lon, lat, lon_0, lat_0)` returns
    `(x, y)` and `cartesian_to_geographic_aeqd(x, y, lon_0, lat_0)` returns
    `(lon, lat)` -- confirmed against the installed Py-ART source. shapely's
    `transform` calls the callback with the geometry's own (x, y), which for
    a lon/lat geometry means (lon, lat); both callbacks below preserve that
    (x, y) calling convention while changing the units it represents.
    """
    if geom.is_empty:
        return geom

    def to_metres(x, y, z=None):
        return geographic_to_cartesian_aeqd(np.asarray(x), np.asarray(y), radar_lon, radar_lat)

    def to_degrees(x, y, z=None):
        return cartesian_to_geographic_aeqd(np.asarray(x), np.asarray(y), radar_lon, radar_lat)

    projected = transform(to_metres, geom)
    simplified = projected.simplify(tolerance_m, preserve_topology=True)
    return transform(to_degrees, simplified)


def _rings_to_polygons(points_list, codes_list):
    """Assemble contourpy filled output into shapely polygons.

    contourpy's 'OuterCode' fill type marks the start of each ring with code 1.
    Within one entry the first ring is the outer boundary and any that follow
    are holes -- so nesting is already resolved for us.
    """
    polygons = []
    for points, codes in zip(points_list, codes_list):
        points = np.asarray(points, dtype=float)
        codes = np.asarray(codes)
        starts = np.flatnonzero(codes == 1)
        bounds = list(starts) + [len(points)]
        rings = [points[a:b] for a, b in zip(starts, bounds[1:])]
        rings = [r for r in rings if len(r) >= 4]
        if not rings:
            continue
        polygons.append(Polygon(rings[0], rings[1:] or None))
    return polygons


# Column offset between the contoured array and the sweep's own gate index,
# from the single no-data column `_pad_for_contouring` (and `contour_mask`)
# prepend to the range axis.
RANGE_PAD_COLUMNS = 1


def _polygons_to_geographic(polygons, sweep):
    """Map index-space contour polygons into lon/lat, dropping empty results.

    contourpy yields vertices as (x, y) = (column, row) -- verified against a
    mask deliberately taller than it is wide: the x column of the returned
    vertices tracked the mask's column span and the y column tracked its row
    span. That is why coordinate index 1 is passed as rows and index 0 as
    columns. `RANGE_PAD_COLUMNS` is subtracted from the columns to undo the
    range-edge padding, so what reaches `polar_vertices_to_lonlat` is the
    sweep's own gate index.
    """
    geographic = []
    for polygon in polygons:
        shell_xy = np.asarray(polygon.exterior.coords)
        shell = polar_vertices_to_lonlat(
            shell_xy[:, 1], shell_xy[:, 0] - RANGE_PAD_COLUMNS, sweep
        )
        holes = []
        for ring in polygon.interiors:
            ring_xy = np.asarray(ring.coords)
            converted = polar_vertices_to_lonlat(
                ring_xy[:, 1], ring_xy[:, 0] - RANGE_PAD_COLUMNS, sweep
            )
            if len(converted) >= 4:
                holes.append(converted)
        candidate = Polygon(shell, holes or None)
        if not candidate.is_valid:
            candidate = candidate.buffer(0)
        if not candidate.is_empty:
            geographic.append(candidate)
    return geographic


def contour_mask(mask: np.ndarray, sweep, simplify_m: float = DEFAULT_SIMPLIFY_M):
    """Geographic outline of a boolean gate mask.

    The mask is binary, so the contour hugs the gate boundary -- there is no
    meaningful value to interpolate between 'echo' and 'no echo'.

    Returns an empty MultiPolygon when the mask holds nothing. Callers must not
    substitute an invented shape.

    See `_polygons_to_geographic` for the (x, y) = (column, row) convention
    contourpy returns and for the range-padding column offset.
    """
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return MultiPolygon()

    # Wrap rows for the seam, plus one empty column at each end of the range
    # axis so a mask reaching the first or last range bin contours out to
    # that bin's edge instead of stopping on its centre. A binary field needs
    # no value reflection: 0 already puts the 0.5 crossing exactly half a
    # gate out. `_padded_ranges` supplies the matching coordinate extension,
    # hence the -1 column offset below.
    padded = np.vstack([mask, mask[:SEAM_PAD_RAYS]]).astype(float)
    edge = np.zeros((padded.shape[0], 1))
    padded = np.hstack([edge, padded, edge])

    generator = contour_generator(z=padded, name="serial", fill_type="OuterCode")
    points_list, codes_list = generator.filled(0.5, np.inf)

    geographic = _polygons_to_geographic(
        _rings_to_polygons(points_list, codes_list), sweep
    )
    if not geographic:
        return MultiPolygon()

    # The wrap padding duplicates a sliver of geometry; union dissolves it, and
    # in lon/lat the two copies genuinely coincide because the seam is an
    # artefact of the array, not of the world.
    merged = unary_union(geographic)
    merged = simplify_ground_metres(merged, sweep.radar_lat, sweep.radar_lon, simplify_m)

    if merged.geom_type == "Polygon":
        return MultiPolygon([merged])
    return merged


def _contour_at_level_raw(field: np.ndarray, level: float, sweep, padded=None):
    """Cumulative super-level-set region ('at or above `level`'), unsimplified.

    `padded` is the `(padded_field, neighbour_max)` pair from
    `_pad_for_contouring`. It does not depend on the level, so `_raw_bands`
    builds it once and passes it in for every level rather than rebuilding
    two full-sweep arrays per level.

    No `simplify_ground_metres` call here on purpose. Set inclusion between
    two levels of the *same* field is exact in index space ({field >= 40} is
    always a subset of {field >= 30}), and the polar-to-geographic map is a
    smooth, invertible coordinate change, so that inclusion survives the
    conversion. Simplifying each level's polygon independently with
    Douglas-Peucker does NOT preserve that inclusion -- two nearby but
    genuinely different curves, simplified separately, can end up crossing
    each other by hundreds of metres to a few kilometres over a long,
    gently-curved azimuthal span (observed directly on this field: level 30
    and level 40 contours of a 100-ray-wide test ramp, simplified
    independently, failed containment by ~0.00026 sq deg, several sq km, in
    scattered slivers along the arc). Callers must simplify only after
    combining levels (see `_simplify_bands_as_coverage`), never before.
    """
    if padded is None:
        padded = _pad_for_contouring(field)
    padded_field, neighbour_max = padded
    # NaN means 'no data'. It must not read as 'below the level' in a way
    # that closes the contour around the last echoing gate's CENTRE -- see
    # `_fill_no_data`, which puts that boundary on the gate's edge instead.
    # See LEVEL_MEMBERSHIP_EPSILON: contourpy's `filled` is strict, and this
    # system buckets gates with `>=` everywhere else. The same effective
    # level goes into `_fill_no_data`, so the gate-edge reflection stays
    # centred on the level the contour is actually drawn at.
    effective_level = _contour_level(level)
    filled = _fill_no_data(padded_field, neighbour_max, effective_level)

    generator = contour_generator(z=filled, name="serial", fill_type="OuterCode")
    points_list, codes_list = generator.filled(effective_level, np.inf)

    geographic = _polygons_to_geographic(
        _rings_to_polygons(points_list, codes_list), sweep
    )
    if not geographic:
        return MultiPolygon()

    # The wrap padding duplicates a sliver of geometry; union dissolves it.
    merged = unary_union(geographic)
    return MultiPolygon([merged]) if merged.geom_type == "Polygon" else merged


def _repair_if_invalid(geom, label):
    """Return `geom` unchanged if valid; otherwise repair it with buffer(0).

    buffer(0) can turn a self-intersecting geometry into a clean one, but for
    a genuinely degenerate case it can also collapse the geometry to nothing.
    Silently returning that empty result would mean a band vanishes and the
    user is never told there was weather there. `is_empty` is not checked
    here because an already-empty band isn't a repair problem -- only a
    non-empty input that repair turns empty is.
    """
    if geom.is_valid:
        return geom
    try:
        pre_repair_area = geom.area
    except Exception:
        pre_repair_area = float("nan")
    repaired = geom.buffer(0)
    if pre_repair_area > 0 and repaired.is_empty:
        logger.warning(
            "buffer(0) repair emptied band %s: pre-repair area was %.6g "
            "(this band's weather will not appear in the output)",
            label, pre_repair_area,
        )
    return repaired


def _raw_bands(field: np.ndarray, ordered_levels, sweep):
    """Exact, unsimplified exclusive bands between consecutive levels.

    Each band is built from `cumulative[lower].difference(cumulative[upper])`
    where both operands are raw (unsimplified) regions of the *same* field.
    Because `cumulative[upper]` is a true geometric subset of
    `cumulative[lower]` (see `_contour_at_level_raw`), this difference is
    clean: no new intersection vertices need to be invented, so the shared
    boundary between one band and its neighbour is the same ring object in
    both differences, before any simplification touches it. That is what
    keeps `exclusive_bands` from re-opening a gap or overlap at the seam
    between two adjacent bands.

    WHAT THIS GUARANTEES, stated precisely because the previous wording
    ("guarantees ... exactly such a coverage") was checked and found to
    overclaim. Measured on the raw output of this function, real volumes:

      - interior-disjoint: worst pairwise band overlap 0.000000000 km2
        (KTLX and KEMX);
      - exact tiling, no gaps: sum of the bands' own areas equals the area
        of their union to full double precision -- 14374.946031 km2 on
        KTLX, 34411.490698 km2 on KEMX, difference 0.000000000 km2.

    What it does NOT guarantee is that `shapely.coverage_is_valid` returns
    True, and it never did. It returns False on both volumes (182 flagged
    edges on KTLX, 23 on KEMX, longest 1718.6 m). Those edges were traced:
    all 182 of 182 and all 23 of 23 lie on the coverage's OWN OUTER
    BOUNDARY, where by definition there is no neighbouring band to match
    them, and each starts or ends at a point where three bands pinch
    together. They are not overlaps, not gaps, and not mismatched interior
    edges -- the two measurements above rule all three out directly. See
    `_simplify_bands_as_coverage` for what that means for the operation
    downstream, and `tests/e2e/test_proof_shape_truth.py` for the real-data
    assertions that pin all of it.
    """
    padded = _pad_for_contouring(field)
    cumulative_raw = {
        level: _contour_at_level_raw(field, level, sweep, padded)
        for level in ordered_levels
    }

    bands = {}
    for index, lower in enumerate(ordered_levels):
        upper = ordered_levels[index + 1] if index + 1 < len(ordered_levels) else float("inf")
        region = cumulative_raw[lower]
        if upper != float("inf"):
            region = region.difference(cumulative_raw[upper])
        region = _repair_if_invalid(region, (lower, upper))
        bands[(lower, upper)] = region
    return bands


def _simplify_bands_as_coverage(bands: dict, sweep, simplify_m: float) -> dict:
    """Simplify a dict of edge-matched, non-overlapping bands as one coverage.

    `shapely.coverage_simplify` (Visvalingam-Whyatt) simplifies a set of
    polygons that share edges *together*, moving a shared vertex once for
    every polygon that references it -- unlike calling `.simplify()` on each
    band separately, which treats each ring as its own problem and lets two
    neighbours drift apart or across each other along a boundary they are
    supposed to share exactly.

    THE PRECONDITION, and how far it actually holds. Shapely documents
    `coverage_simplify` as assuming a valid polygonal coverage and leaves
    the result undefined otherwise, so this is not a detail to wave at.
    `_raw_bands` (see its docstring for the measurements) produces bands
    that are interior-disjoint and tile their union exactly, with every
    shared interior edge coming from the same ring object in both
    neighbours. `shapely.coverage_is_valid` nevertheless reports False on
    real volumes -- and on the INPUT to this function every edge it flags
    (182 on KTLX, 23 on KEMX) was traced to the coverage's own OUTER
    boundary, where there is no neighbour to match and nothing for joint
    simplification to keep in step. The operation's meaningful precondition
    therefore holds; the library's boolean does not report it.

    On the OUTPUT the picture is the same: 177 flagged edges on KTLX, of
    which 175 are outer-boundary and 2 are an interior matched PAIR -- one
    edge from the 15-20 band and one from the 20-30 band, at the same place,
    every vertex of each sitting 0.000000 m from the other's boundary. They
    describe the same curve split into different segments, which is what
    GEOS's edge matching objects to; they do not describe different ground.
    KEMX and KIWA have no interior flagged edge at all. Overlap and tiling
    gap are 0.000000000 km2 on all three.

    That is a claim about today's geometry, so it is not left resting on
    itself. Two things back it up:

      - a runtime guard here: a band that had real area before
        simplification and comes back empty afterwards means weather has
        silently vanished from the map, so the whole simplification pass is
        discarded and the exact, unsimplified bands are returned instead. An
        unsimplified band is heavier to serialise; it is never wrong.
      - real-data assertions in `tests/e2e/test_proof_shape_truth.py`
        (`test_precipitation_bands_are_exclusive_and_nested`) covering
        exclusivity, exact tiling and nesting on all three cached volumes,
        so a future volume or a GEOS upgrade that breaks this fails the
        suite instead of passing it quietly. The expensive geometric checks
        live there rather than here because `exclusive_bands` runs once per
        detected object on the storm layer, where a per-call union costs
        ~2.3 s (measured, KTLX) and would be paid dozens of times per
        request.
    """
    keys = list(bands.keys())
    geoms = [bands[key] for key in keys]

    def to_metres(x, y, z=None):
        return geographic_to_cartesian_aeqd(np.asarray(x), np.asarray(y), sweep.radar_lon, sweep.radar_lat)

    def to_degrees(x, y, z=None):
        return cartesian_to_geographic_aeqd(np.asarray(x), np.asarray(y), sweep.radar_lon, sweep.radar_lat)

    non_empty = [i for i, g in enumerate(geoms) if not g.is_empty]
    if not non_empty:
        return {key: MultiPolygon() for key in keys}

    projected = [transform(to_metres, geoms[i]) for i in non_empty]
    simplified_metres = coverage_simplify(projected, tolerance=simplify_m)

    result = {key: MultiPolygon() for key in keys}
    for i, geom in zip(non_empty, simplified_metres):
        # coverage_simplify's own output is valid in the metres frame it ran
        # in (checked directly: every band comes back geom.is_valid there).
        # The nonlinear aeqd->lon/lat projection can still fold a near-acute
        # vertex into a self-intersection on the way back out, so validity
        # has to be re-checked and repaired *after* that projection, in the
        # degree space downstream operations actually run in -- not before
        # it, and not in the metres frame. buffer(0) repairs it locally; it
        # does not undo the edge-matching coverage_simplify already did
        # between neighbouring bands, since projection and repair are both
        # applied identically, per band, to output that was already
        # edge-matched going in.
        back = transform(to_degrees, geom)
        back = _repair_if_invalid(back, keys[i])
        if not back.is_empty:
            result[keys[i]] = MultiPolygon([back]) if back.geom_type == "Polygon" else back

    emptied = [
        keys[i] for i in non_empty if geoms[i].area > 0.0 and result[keys[i]].is_empty
    ]
    if emptied:
        # Discard the whole pass rather than the affected bands only:
        # coverage_simplify moves shared vertices jointly, so if it has gone
        # wrong badly enough to delete a band, the bands either side of the
        # deleted one are the ones whose shared boundary it was moving, and
        # they cannot be trusted separately. `_raw_bands` output is exact and
        # exclusive; it is only heavier.
        logger.warning(
            "coverage_simplify emptied %d band(s) that had real area (%s); "
            "falling back to unsimplified bands so no weather is lost",
            len(emptied), emptied,
        )
        return {
            key: _as_multipolygon_or_empty(bands[key]) for key in keys
        }
    return result


def _as_multipolygon_or_empty(geom):
    if geom is None or geom.is_empty:
        return MultiPolygon()
    return MultiPolygon([geom]) if geom.geom_type == "Polygon" else geom


def exclusive_bands(field: np.ndarray, levels, sweep, simplify_m: float = DEFAULT_SIMPLIFY_M):
    """Non-overlapping bands, each covering one level up to the next.

    Built by geometric difference on raw (unsimplified) geometry, so a band
    wrapped around a stronger core is an annulus with a hole -- which is what
    that region physically is. Simplification runs once, jointly, across all
    bands (`_simplify_bands_as_coverage`), so the user walking inward is
    always in exactly one band and crosses a real boundary, not a
    simplification artefact.
    """
    field = np.asarray(field, dtype=float)
    ordered = sorted(float(level) for level in levels)
    raw = _raw_bands(field, ordered, sweep)
    return _simplify_bands_as_coverage(raw, sweep, simplify_m)


def contour_field(field: np.ndarray, levels, sweep, simplify_m: float = DEFAULT_SIMPLIFY_M):
    """Cumulative contours: each level is the region at or above that value.

    Internally reuses the same exclusive-band machinery as `exclusive_bands`
    (raw difference, then one joint coverage-simplify pass) and reconstitutes
    each cumulative level as the union of that level's band and every band
    above it. Because those bands are simplified together as a single
    coverage, the resulting cumulative levels nest exactly: `contour_field`
    and `exclusive_bands` can never disagree about where a boundary sits.
    """
    field = np.asarray(field, dtype=float)
    ordered = sorted(float(level) for level in levels)
    raw = _raw_bands(field, ordered, sweep)
    simplified_bands = _simplify_bands_as_coverage(raw, sweep, simplify_m)

    band_keys = list(simplified_bands.keys())  # same order as `ordered`
    cumulative = {}
    for index, level in enumerate(ordered):
        pieces = [simplified_bands[key] for key in band_keys[index:]]
        merged = unary_union(pieces)
        # unary_union of an all-empty input list returns GEOMETRYCOLLECTION
        # EMPTY, not MULTIPOLYGON EMPTY (e.g. an all-NaN field, or a level
        # above the field's maximum). Still empty, so no invented geometry,
        # but the wrong type would serialise to the wrong GeoJSON "type" for
        # downstream consumers -- normalise it the same way `exclusive_bands`
        # already does for its own empty case.
        if merged.is_empty:
            cumulative[level] = MultiPolygon()
        elif merged.geom_type == "Polygon":
            cumulative[level] = MultiPolygon([merged])
        else:
            cumulative[level] = merged
    return cumulative
