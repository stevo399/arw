"""Turn polar radar fields into geographic polygons.

This module knows nothing about storms, GeoJSON or Audiom. It takes an array
laid out as (rays x gates) and returns shapely geometry in lon/lat.

Contouring happens on the polar grid itself rather than on a resampled
Cartesian grid, so the radar's own resolution is preserved: no single grid
spacing can match a radar at both 20 km and 200 km, and inventing detail at
range is the worse failure for a user exploring a shape by touch.
"""

import numpy as np
from contourpy import contour_generator
from pyart.core.transforms import (
    antenna_to_cartesian,
    cartesian_to_geographic_aeqd,
    geographic_to_cartesian_aeqd,
)
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import transform, unary_union

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
