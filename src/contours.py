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

# The radar's finest real resolution: one range gate is 250 m deep. Simplifying
# below this would discard measurement; above it would invent detail.
DEFAULT_SIMPLIFY_M = 250.0

# Rows of wrap-around padding added before contouring so a shape crossing due
# north is not severed by the array edge. Marching squares needs one row of
# overlap; two is defensive.
SEAM_PAD_RAYS = 2


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


def polar_vertices_to_lonlat(rows, cols, sweep) -> np.ndarray:
    """Convert fractional (ray, gate) index positions to lon/lat.

    Contour vertices land between gate centres, so azimuth, range and
    elevation are all interpolated. Row indices may exceed the ray count when
    seam padding is in play; the padded azimuth axis handles that continuously.

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
    ranges_m = np.asarray(sweep.ranges_m, dtype=float)

    ray_index = np.arange(len(padded_azimuths), dtype=float)
    gate_index = np.arange(len(ranges_m), dtype=float)

    azimuth = np.interp(rows, ray_index, padded_azimuths) % 360.0
    elevation = np.interp(rows, ray_index, padded_elevations)
    range_m = np.interp(cols, gate_index, ranges_m)

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


def contour_mask(mask: np.ndarray, sweep, simplify_m: float = DEFAULT_SIMPLIFY_M):
    """Geographic outline of a boolean gate mask.

    The mask is binary, so the contour hugs the gate boundary -- there is no
    meaningful value to interpolate between 'echo' and 'no echo'.

    Returns an empty MultiPolygon when the mask holds nothing. Callers must not
    substitute an invented shape.

    contourpy yields vertices as (x, y) = (column, row) -- verified against a
    mask deliberately taller than it is wide: the x column of the returned
    vertices tracked the mask's column span and the y column tracked its row
    span. That is why the conversion calls below pass exterior/interior
    coordinate index 1 as rows and index 0 as columns.
    """
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return MultiPolygon()

    padded = np.vstack([mask, mask[:SEAM_PAD_RAYS]]).astype(float)

    generator = contour_generator(z=padded, name="serial", fill_type="OuterCode")
    points_list, codes_list = generator.filled(0.5, np.inf)

    geographic = []
    for polygon in _rings_to_polygons(points_list, codes_list):
        shell = polar_vertices_to_lonlat(
            np.asarray(polygon.exterior.coords)[:, 1],
            np.asarray(polygon.exterior.coords)[:, 0],
            sweep,
        )
        holes = [
            polar_vertices_to_lonlat(
                np.asarray(ring.coords)[:, 1], np.asarray(ring.coords)[:, 0], sweep
            )
            for ring in polygon.interiors
        ]
        candidate = Polygon(shell, [h for h in holes if len(h) >= 4])
        if not candidate.is_valid:
            candidate = candidate.buffer(0)
        if not candidate.is_empty:
            geographic.append(candidate)

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


def _contour_at_level_raw(field: np.ndarray, level: float, sweep):
    """Cumulative super-level-set region ('at or above `level`'), unsimplified.

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
    padded = np.vstack([field, field[:SEAM_PAD_RAYS]])
    # NaN means 'no data', which must not read as 'below the level' in a way
    # that closes a contour around it. Push it well below any real level.
    filled = np.where(np.isfinite(padded), padded, -9999.0)

    generator = contour_generator(z=filled, name="serial", fill_type="OuterCode")
    points_list, codes_list = generator.filled(level, np.inf)

    geographic = []
    for polygon in _rings_to_polygons(points_list, codes_list):
        shell_xy = np.asarray(polygon.exterior.coords)
        shell = polar_vertices_to_lonlat(shell_xy[:, 1], shell_xy[:, 0], sweep)
        holes = []
        for ring in polygon.interiors:
            ring_xy = np.asarray(ring.coords)
            converted = polar_vertices_to_lonlat(ring_xy[:, 1], ring_xy[:, 0], sweep)
            if len(converted) >= 4:
                holes.append(converted)
        candidate = Polygon(shell, holes or None)
        if not candidate.is_valid:
            candidate = candidate.buffer(0)
        if not candidate.is_empty:
            geographic.append(candidate)

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
    """
    cumulative_raw = {level: _contour_at_level_raw(field, level, sweep) for level in ordered_levels}

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
    supposed to share exactly. `_raw_bands` guarantees the input here is
    exactly such a coverage (edge-matched, non-overlapping) before
    simplification ever runs.
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
    return result


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
