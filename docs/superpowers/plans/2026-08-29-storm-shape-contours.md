# True Storm Shapes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace convex-hull storm polygons with contours of the actual echo, so that walking across a shape in Audiom means there is genuinely weather there.

**Architecture:** One new module, `src/contours.py`, turns a polar radar field into geographic shapely polygons — marching squares on the `(rays × gates)` array, vertices mapped through `geometry.gate_coordinates`, seam handled by wrapping, simplification performed in ground metres via a local equidistant projection. `src/map_layer.py` consumes it and loses its hull machinery entirely. A second GeoJSON layer contours the whole precipitation field independently of object detection.

**Tech Stack:** Python 3.11+, contourpy 1.3.3 (marching squares), shapely 2.1.2 (rings, holes, difference, simplification), NumPy, Py-ART (projection helpers), pytest.

**Spec:** `docs/superpowers/specs/2026-08-29-storm-shape-contours-design.md`

## Current status (2026-09-07)

Implementation and validation are complete in the `spec2-shapes` working tree,
pending the restoration commit. The Sep 2 contour safeguards were restored after
`6017bac` accidentally replaced them with older code: no-data boundaries use
gate edges, range-axis ends are padded, level membership matches `>=`, and a
failed coverage-simplification pass falls back to raw geometry. Cache-backed
proof: 10 passed; full suite: 390 passed, 4 strict xfailed. The proof report
contains the current measurements and its two explicit limitations (the
resolution-floor guard is one-sided, and KIWA's single-object union FN is
reported rather than asserted). The task checkboxes below are retained as the
historical execution log rather than rewritten as though every original success
criterion had been established literally.

## Global Constraints

- **Every new test must be proven to FAIL against the convex-hull implementation before acceptance.** The hull is the known-broken version and is available in git history. Spec 1 shipped nine tests that passed against the very bug they claimed to cover; this is the guard against a tenth.
- **`DEFAULT_SIMPLIFY_M = 100.0`** — deliberately below one full 250 m gate so simplification cannot swallow a gate-wide notch. Simplification tolerance is always expressed in ground metres, never in index space or degrees.
- **Never invent geometry.** If a mask yields no valid contour, omit the feature and record why. `_fallback_square` (a square drawn at a centroid when hulling failed) is deleted, not ported.
- **Never decimate vertices by index.** To reduce vertex count, raise the tolerance and re-simplify. Index decimation biases the boundary.
- **Success is measured by whether the map lies when walked on:** false-positive rate below 5%, false-negative rate below 5%, on every cached volume tested. If a target cannot be met, report the measured value — do not relax the threshold.
- **Preserve the four strict xfails from Spec 1.** They remain strict.
- **Test runner:** `.venv/Scripts/python.exe -m pytest`
- **Baseline at plan start:** 324 passed, 4 xfailed. **Current restoration validation:** 390 passed, 4 xfailed.
- **Reference volumes:** `cache/KTLX/KTLX20130520_195527_V06.gz` (Newcastle–Moore EF5, dense convection), `cache/KEMX/KEMX20260712_022646_V06` (spec 1 evidence volume), `cache/KIWA/KIWA20260712_170029_V06` (clear air).

---

### Task 1: `src/contours.py` — polar mask to geographic polygon

The core of the spec. Everything else consumes this.

**Files:**
- Create: `src/contours.py`
- Modify: `pyproject.toml`
- Test: `tests/unit/test_contours.py`

**Interfaces:**
- Consumes: `src.geometry.gate_coordinates`, `src.parser.SweepData`
- Produces:
  - `DEFAULT_SIMPLIFY_M: float` (250.0)
  - `SEAM_PAD_RAYS: int` (2)
  - `contour_mask(mask: np.ndarray, sweep, simplify_m: float = DEFAULT_SIMPLIFY_M) -> MultiPolygon`
  - `polar_vertices_to_lonlat(rows, cols, sweep) -> np.ndarray` shaped `(n, 2)`
  - `simplify_ground_metres(geom, radar_lat, radar_lon, tolerance_m) -> BaseGeometry`

- [ ] **Step 1: Declare the dependencies**

`shapely` and `contourpy` are currently present only transitively via Py-ART. Relying on a transitive dependency for core geometry is how a future Py-ART upgrade silently breaks the map.

Add to `pyproject.toml` `dependencies`:

```toml
    "contourpy>=1.3.0",
    "shapely>=2.0",
```

Run `uv sync` (or confirm the venv already satisfies them — contourpy 1.3.3 and shapely 2.1.2 are installed).

- [ ] **Step 2: Write the failing test for vertex conversion**

Contourpy returns vertices at *fractional* index positions, so conversion must interpolate azimuth, range and elevation between gate centres — and the azimuth interpolation must be seam-safe.

```python
# tests/unit/test_contours.py
import numpy as np
import pyart
import pytest

from src.contours import polar_vertices_to_lonlat
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
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_contours.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.contours'`

- [ ] **Step 4: Implement vertex conversion**

```python
# src/contours.py
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
    """
    padded_azimuths, padded_elevations = _padded_axes(sweep)
    ranges_m = np.asarray(sweep.ranges_m, dtype=float)

    ray_index = np.arange(len(padded_azimuths), dtype=float)
    gate_index = np.arange(len(ranges_m), dtype=float)

    azimuth = np.interp(rows, ray_index, padded_azimuths) % 360.0
    elevation = np.interp(rows, ray_index, padded_elevations)
    range_m = np.interp(cols, gate_index, ranges_m)

    from src.geometry import gate_coordinates

    lat, lon = gate_coordinates(
        azimuths=azimuth,
        ranges_m=range_m,
        elevation_deg=elevation,
        radar_lat=sweep.radar_lat,
        radar_lon=sweep.radar_lon,
    )
    # gate_coordinates builds a full (n_rays, n_gates) mesh; the diagonal holds
    # the paired (azimuth[i], range[i]) vertices we asked for.
    return np.column_stack([np.diagonal(lon), np.diagonal(lat)])
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_contours.py -v`
Expected: PASS (3 tests)

If the diagonal trick proves too memory-hungry for large vertex counts (it builds an n×n mesh), replace it with a direct per-vertex computation using the same Doviak & Zrnić maths, and say so in your report. Correctness first; measure before optimising.

- [ ] **Step 6: Write the failing test for ground-metre simplification**

```python
# append to tests/unit/test_contours.py
from shapely.geometry import Polygon as ShapelyPolygon

from src.contours import simplify_ground_metres


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
```

- [ ] **Step 7: Run to verify failure**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_contours.py -k simplify -v`
Expected: FAIL with `ImportError: cannot import name 'simplify_ground_metres'`

- [ ] **Step 8: Implement ground-metre simplification**

Shapely's `simplify` works in the coordinate units it is given. Degrees are the wrong unit — a degree of longitude shrinks with latitude, so a degree tolerance is anisotropic and range-dependent. Project into a local azimuthal-equidistant frame centred on the radar, where units are metres, simplify there, and project back.

```python
# append to src/contours.py

def simplify_ground_metres(geom, radar_lat: float, radar_lon: float, tolerance_m: float):
    """Simplify a lon/lat geometry with a tolerance expressed in ground metres.

    Shapely simplifies in whatever units it is handed. Degrees are wrong here:
    a degree of longitude shrinks with latitude, so a degree tolerance is
    anisotropic and varies across the sweep. Projecting to a local
    azimuthal-equidistant frame centred on the radar makes the units metres.
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
```

Note `cartesian_to_geographic_aeqd` returns `(lon, lat)` while `geographic_to_cartesian_aeqd` takes `(lon, lat)` — confirm the argument order against the installed Py-ART before relying on it, and state in your report what you found.

- [ ] **Step 9: Run to verify passing**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_contours.py -k simplify -v`
Expected: PASS (2 tests)

- [ ] **Step 10: Write the failing test for `contour_mask`**

```python
# append to tests/unit/test_contours.py
from src.contours import contour_mask


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
    n_rays = np.asarray(sweep.reflectivity).shape[0]

    def build(m):
        m[-15:, 300:340] = True
        m[:15, 300:340] = True

    out = contour_mask(_mask_sweep(sweep, build), sweep)
    polys = list(out.geoms) if out.geom_type == "MultiPolygon" else [out]
    assert len(polys) == 1, f"seam split the shape into {len(polys)} pieces"
```

- [ ] **Step 11: Run to verify failure**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_contours.py -k contour_mask -v`
Expected: FAIL with `ImportError: cannot import name 'contour_mask'`

- [ ] **Step 12: Implement `contour_mask`**

```python
# append to src/contours.py

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
```

Note contourpy yields vertices as `(x, y)` = `(column, row)`, which is why the conversion call passes coordinate index 1 as rows and index 0 as columns. Verify that ordering against a known asymmetric mask before trusting it, and say what you found.

- [ ] **Step 13: Run to verify passing**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_contours.py -v`
Expected: PASS (10 tests)

- [ ] **Step 14: Commit**

```bash
git add src/contours.py tests/unit/test_contours.py pyproject.toml
git commit -m "feat: add polar-to-geographic contouring with seam and hole support"
```

---

### Task 2: Continuous-field contours and exclusive bands

Object footprints are binary; intensity bands are sampled from a continuous field where interpolating dBZ between gate centres is physically meaningful.

**Files:**
- Modify: `src/contours.py`
- Test: `tests/unit/test_contours.py`

**Interfaces:**
- Consumes: `contour_mask` internals from Task 1 (`_rings_to_polygons`, `polar_vertices_to_lonlat`, `simplify_ground_metres`, `SEAM_PAD_RAYS`)
- Produces:
  - `contour_field(field, levels, sweep, simplify_m=DEFAULT_SIMPLIFY_M) -> dict[float, MultiPolygon]` — cumulative, each level meaning "at or above"
  - `exclusive_bands(field, levels, sweep, simplify_m=DEFAULT_SIMPLIFY_M) -> dict[tuple[float, float], MultiPolygon]` — each band is `level_n` up to `level_n+1`; the highest is open-ended with upper bound `float("inf")`

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/unit/test_contours.py
from src.contours import contour_field, exclusive_bands


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
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_contours.py -k "field or band" -v`
Expected: FAIL with `ImportError: cannot import name 'contour_field'`

- [ ] **Step 3: Implement both functions**

```python
# append to src/contours.py

def _contour_at_level(field: np.ndarray, level: float, sweep, simplify_m: float):
    """Cumulative contour of everything at or above one level."""
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

    merged = unary_union(geographic)
    merged = simplify_ground_metres(merged, sweep.radar_lat, sweep.radar_lon, simplify_m)
    return MultiPolygon([merged]) if merged.geom_type == "Polygon" else merged


def contour_field(field: np.ndarray, levels, sweep, simplify_m: float = DEFAULT_SIMPLIFY_M):
    """Cumulative contours: each level is the region at or above that value."""
    field = np.asarray(field, dtype=float)
    return {
        float(level): _contour_at_level(field, float(level), sweep, simplify_m)
        for level in sorted(levels)
    }


def exclusive_bands(field: np.ndarray, levels, sweep, simplify_m: float = DEFAULT_SIMPLIFY_M):
    """Non-overlapping bands, each covering one level up to the next.

    Built by geometric difference, so a band wrapped around a stronger core is
    an annulus with a hole -- which is what that region physically is. The user
    walking inward is always in exactly one band and crosses a real boundary.
    """
    ordered = sorted(float(level) for level in levels)
    cumulative = contour_field(field, ordered, sweep, simplify_m)

    bands = {}
    for index, lower in enumerate(ordered):
        upper = ordered[index + 1] if index + 1 < len(ordered) else float("inf")
        region = cumulative[lower]
        if upper != float("inf"):
            region = region.difference(cumulative[upper])
        if not region.is_valid:
            region = region.buffer(0)
        if region.geom_type == "Polygon":
            region = MultiPolygon([region])
        bands[(lower, upper)] = region
    return bands
```

- [ ] **Step 4: Run to verify passing**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_contours.py -v`
Expected: PASS (15 tests)

- [ ] **Step 5: Commit**

```bash
git add src/contours.py tests/unit/test_contours.py
git commit -m "feat: add continuous-field contours and exclusive intensity bands"
```

---

### Task 3: The walking-truth metric

The spec's success criterion, and the only measurement in this project that describes the user's experience rather than an internal property. Built before the shapes are wired in, so the convex-hull baseline can be captured first.

**Files:**
- Create: `src/shape_truth.py`
- Test: `tests/unit/test_shape_truth.py`

**Interfaces:**
- Consumes: `src.geometry.gate_coordinates`
- Produces:
  - `WalkTruth` dataclass with `false_positive_rate: float`, `false_negative_rate: float`, `n_inside: int`, `n_outside: int`
  - `measure_walk_truth(geom, field, sweep, level, n_samples=2000, seed=0) -> WalkTruth`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_shape_truth.py
import numpy as np
import pyart
import pytest
from shapely.geometry import MultiPolygon

from src.contours import contour_mask
from src.parser import extract_sweep_data
from src.shape_truth import measure_walk_truth

REFERENCE_VOLUME = "cache/KEMX/KEMX20260712_022646_V06"


@pytest.fixture(scope="module")
def sweep():
    return extract_sweep_data(pyart.io.read_nexrad_archive(REFERENCE_VOLUME))


def test_perfect_shape_scores_near_zero_on_both_rates(sweep):
    """A contour of a block should describe that block almost exactly."""
    field = np.full(np.asarray(sweep.reflectivity).shape, np.nan)
    field[100:200, 300:400] = 40.0
    mask = np.isfinite(field) & (field >= 20.0)

    truth = measure_walk_truth(contour_mask(mask, sweep), field, sweep, 20.0)

    assert truth.false_positive_rate < 0.05
    assert truth.false_negative_rate < 0.05


def test_oversized_shape_is_caught_as_false_positive(sweep):
    """A shape larger than its echo must score badly -- that is the hull's failure."""
    field = np.full(np.asarray(sweep.reflectivity).shape, np.nan)
    field[100:200, 300:400] = 40.0
    mask = np.isfinite(field) & (field >= 20.0)

    honest = contour_mask(mask, sweep)
    inflated = MultiPolygon([honest.buffer(0.05)]) if honest.geom_type == "Polygon" \
        else MultiPolygon([honest.buffer(0.05)])

    truth = measure_walk_truth(inflated, field, sweep, 20.0)
    assert truth.false_positive_rate > 0.2


def test_undersized_shape_is_caught_as_false_negative(sweep):
    field = np.full(np.asarray(sweep.reflectivity).shape, np.nan)
    field[100:200, 300:400] = 40.0
    mask = np.isfinite(field) & (field >= 20.0)

    honest = contour_mask(mask, sweep)
    shrunk = honest.buffer(-0.02)
    truth = measure_walk_truth(shrunk, field, sweep, 20.0)
    assert truth.false_negative_rate > 0.2


def test_measurement_is_deterministic(sweep):
    field = np.full(np.asarray(sweep.reflectivity).shape, np.nan)
    field[100:200, 300:400] = 40.0
    mask = np.isfinite(field) & (field >= 20.0)
    geom = contour_mask(mask, sweep)

    a = measure_walk_truth(geom, field, sweep, 20.0, seed=7)
    b = measure_walk_truth(geom, field, sweep, 20.0, seed=7)
    assert a.false_positive_rate == b.false_positive_rate
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_shape_truth.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.shape_truth'`

- [ ] **Step 3: Implement the metric**

```python
# src/shape_truth.py
"""Measure whether a shape lies when walked on.

The project owner explores Audiom by walking across a feature's whole surface
at a step size of his choosing. So the question that matters is not whether a
polygon looks right, but whether standing at a point inside it means there is
genuinely weather at that point.

  false positive: inside the shape, no echo present   -- a lie walked into
  false negative: echo present, outside the shape     -- weather not reported
"""

from dataclasses import dataclass

import numpy as np
from shapely.geometry import Point
from shapely.prepared import prep

from src.geometry import gate_coordinates


@dataclass
class WalkTruth:
    false_positive_rate: float
    false_negative_rate: float
    n_inside: int
    n_outside: int


def measure_walk_truth(geom, field, sweep, level: float, n_samples: int = 2000, seed: int = 0):
    """Sample gates and compare shape membership against real echo.

    Sampling gates rather than uniform points keeps the measurement in the
    radar's own frame: every sample is somewhere the radar actually looked.
    """
    field = np.asarray(field, dtype=float)
    has_echo = np.isfinite(field) & (field >= level)

    lat, lon = gate_coordinates(
        sweep.azimuths, sweep.ranges_m, sweep.elevations,
        sweep.radar_lat, sweep.radar_lon,
    )

    rng = np.random.default_rng(seed)
    minx, miny, maxx, maxy = geom.bounds if not geom.is_empty else (0, 0, 0, 0)
    within_bbox = (lon >= minx) & (lon <= maxx) & (lat >= miny) & (lat <= maxy)
    candidates = np.flatnonzero(within_bbox.ravel())
    if candidates.size == 0:
        return WalkTruth(0.0, 0.0, 0, 0)

    chosen = rng.choice(candidates, size=min(n_samples, candidates.size), replace=False)
    flat_lon = lon.ravel()[chosen]
    flat_lat = lat.ravel()[chosen]
    flat_echo = has_echo.ravel()[chosen]

    prepared = prep(geom)
    inside = np.array([prepared.contains(Point(x, y)) for x, y in zip(flat_lon, flat_lat)])

    n_inside = int(inside.sum())
    n_outside = int((~inside).sum())
    false_positive = int((inside & ~flat_echo).sum())
    false_negative = int((~inside & flat_echo).sum())

    return WalkTruth(
        false_positive_rate=false_positive / n_inside if n_inside else 0.0,
        false_negative_rate=false_negative / n_outside if n_outside else 0.0,
        n_inside=n_inside,
        n_outside=n_outside,
    )
```

- [ ] **Step 4: Run to verify passing**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_shape_truth.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Capture the convex-hull baseline**

The hull is still in `src/map_layer.py` at this point. Measure it on all three reference volumes, for the largest six objects in each, and record the numbers in your report. These are the figures every later task is judged against.

Use the existing pipeline: `extract_sweep_data`, `extract_velocity`, `align_field_by_azimuth`, `detect_rotation_signatures`, `apply_quality_control`, `detect_objects_with_grid`, then `map_layer.object_mask_to_polygon` for the hull.

Expected, from the spec's own measurement: roughly 24% false-positive on the Moore volume. If you measure something materially different, say so — the spec's figure was computed by area, and a sampled rate may differ.

- [ ] **Step 6: Commit**

```bash
git add src/shape_truth.py tests/unit/test_shape_truth.py
git commit -m "feat: add walking-truth metric for shape fidelity"
```

---

### Task 4: Storm footprints become contours

**Files:**
- Modify: `src/map_layer.py:96-158` (removing `_sample_mask_points`, `_fallback_square`, `mask_to_polygon`, `object_mask_to_polygon`), `src/map_layer.py:12` (`MAX_HULL_POINTS`), `src/map_layer.py:4` (the `ConvexHull` import)
- Test: `tests/unit/test_map_layer.py`

**Interfaces:**
- Consumes: `contour_mask` (Task 1), `measure_walk_truth` (Task 3)
- Produces: `object_geometry(scan, obj) -> dict` — a GeoJSON geometry mapping, `Polygon` or `MultiPolygon`

- [ ] **Step 1: Write the failing test**

The existing tests in this file build `BufferedScan` inline. Add a helper so the new tests stay readable, then the tests themselves.

```python
# append to tests/unit/test_map_layer.py
from src.map_layer import object_geometry


def _scan_with_mask(mask: np.ndarray, peak_dbz: float = 50.0) -> BufferedScan:
    """A BufferedScan whose single object occupies exactly `mask`."""
    n_rays, n_gates = mask.shape
    reflectivity = np.where(mask, peak_dbz, np.nan)
    return BufferedScan(
        timestamp=datetime(2026, 4, 10, 20, 0),
        site_id="KTLX",
        reflectivity_data=SweepData(
            reflectivity=reflectivity,
            azimuths=np.linspace(0.0, 359.0, n_rays),
            ranges_m=np.linspace(20000.0, 60000.0, n_gates),
            radar_lat=35.3331,
            radar_lon=-97.2778,
            elevation_angle=0.5,
            elevations=np.full(n_rays, 0.5),
            elevation_angles=[0.5],
            radar_alt_m=390.0,
            timestamp="2026-04-10T20:00:00Z",
        ),
        detected_objects=[
            DetectedObject(
                object_id=1,
                centroid_lat=35.2,
                centroid_lon=-96.9,
                distance_km=30.0,
                bearing_deg=90.0,
                peak_dbz=peak_dbz,
                peak_label="intense precipitation",
                area_km2=100.0,
            )
        ],
        labeled_grid=mask.astype(int),
        object_masks={1: mask},
    )


def test_object_geometry_emits_a_polygon_type():
    mask = np.zeros((60, 60), dtype=bool)
    mask[20:40, 20:40] = True
    geometry = object_geometry(_scan_with_mask(mask), _scan_with_mask(mask).detected_objects[0])
    assert geometry["type"] in ("Polygon", "MultiPolygon")


def test_object_geometry_carries_an_interior_ring_for_a_donut():
    """The convex hull fills this middle in. Walking into the gap would then
    report a storm that is not there."""
    mask = np.zeros((60, 60), dtype=bool)
    mask[15:45, 15:45] = True
    mask[25:35, 25:35] = False

    scan = _scan_with_mask(mask)
    geometry = object_geometry(scan, scan.detected_objects[0])

    if geometry["type"] == "Polygon":
        rings = [geometry["coordinates"]]
    else:
        rings = geometry["coordinates"]
    assert any(len(part) > 1 for part in rings), "no interior ring emitted"


def test_object_geometry_keeps_two_blobs_separate():
    mask = np.zeros((60, 60), dtype=bool)
    mask[5:15, 5:15] = True
    mask[40:50, 40:50] = True

    scan = _scan_with_mask(mask)
    geometry = object_geometry(scan, scan.detected_objects[0])

    assert geometry["type"] == "MultiPolygon"
    assert len(geometry["coordinates"]) == 2


def test_object_geometry_returns_none_rather_than_inventing_a_shape():
    """The previous implementation drew a square at the centroid when hulling
    failed, presenting a fabricated shape as measurement."""
    mask = np.zeros((60, 60), dtype=bool)
    scan = _scan_with_mask(mask)
    assert object_geometry(scan, scan.detected_objects[0]) is None
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_map_layer.py -k geometry -v`
Expected: FAIL — `object_geometry` does not exist

- [ ] **Step 3: Replace the hull machinery**

Delete from `src/map_layer.py`: the `ConvexHull, QhullError` import, `MAX_HULL_POINTS`, `_sample_mask_points`, `_fallback_square`, `mask_to_polygon`, `object_mask_to_polygon`.

Add:

```python
from shapely.geometry import mapping

from src.contours import contour_mask


def object_geometry(scan: BufferedScan, obj: DetectedObject) -> dict[str, Any] | None:
    """GeoJSON geometry for a detected object, or None if it has no valid shape.

    Returns None rather than inventing a placeholder. The previous
    implementation drew a square at the centroid when hulling failed, which
    presented a fabricated shape as measurement.
    """
    mask = scan.object_masks.get(obj.object_id)
    if mask is None:
        return None
    geom = contour_mask(mask, scan.reflectivity_data)
    if geom.is_empty:
        return None
    return mapping(geom)
```

Update `storm_object_to_feature` to use it, and to skip the object entirely when geometry is None. Update `build_storm_geojson` and `build_storm_audiom_geojson` to filter out skipped objects, and record the count of skipped objects in the layer metadata so the omission is visible rather than silent.

- [ ] **Step 4: Run to verify passing**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_map_layer.py -v`
Expected: PASS. Several existing tests will need their expectations recomputed — polygon vertex counts and coordinate values change. Recompute them; never loosen an assertion into a range to get green.

- [ ] **Step 5: Measure the improvement**

On all three reference volumes, measure the walking-truth rates for the six largest objects and compare against the Task 3 hull baseline. Record both in your report.

- [ ] **Step 6: Run the full suite**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: green, with the four Spec 1 xfails still xfailing.

- [ ] **Step 7: Commit**

```bash
git add src/map_layer.py tests/unit/test_map_layer.py
git commit -m "feat: storm footprints are contours of real echo, not convex hulls"
```

---

### Task 5: Exclusive intensity bands

**Files:**
- Modify: `src/map_layer.py:213-286` (`storm_intensity_layer_to_feature`), `src/map_layer.py:327` (`build_storm_intensity_geojson`)
- Test: `tests/unit/test_map_layer.py`

**Interfaces:**
- Consumes: `exclusive_bands` (Task 2)
- Produces: `build_storm_intensity_geojson` emitting non-overlapping band features with holes

- [ ] **Step 1: Write the failing test**

```python
# append to tests/unit/test_map_layer.py
from shapely.geometry import shape as shapely_shape


def _scan_with_core() -> BufferedScan:
    """A 25 dBZ object wrapped around a 45 dBZ core."""
    reflectivity = np.full((60, 60), np.nan)
    reflectivity[15:45, 15:45] = 25.0
    reflectivity[25:35, 25:35] = 45.0
    mask = np.isfinite(reflectivity)
    scan = _scan_with_mask(mask)
    scan.reflectivity_data.reflectivity = reflectivity
    scan.detected_objects[0].layers = [
        IntensityLayerData("light precipitation", 20.0, 30.0, 60.0),
        IntensityLayerData("heavy precipitation", 40.0, 50.0, 10.0),
    ]
    return scan


def test_intensity_bands_do_not_overlap():
    """Walking must place the user in exactly one band.

    Today each band is its own convex hull, filled to its own middle, so
    walking inward crosses several overlapping claims about the same ground.
    """
    geojson = build_storm_intensity_geojson(_scan_with_core())
    geoms = [shapely_shape(f["geometry"]) for f in geojson["features"]]

    for i, a in enumerate(geoms):
        for b in geoms[i + 1:]:
            assert a.intersection(b).area < 1e-12


def test_weaker_band_has_a_hole_where_the_core_sits():
    geojson = build_storm_intensity_geojson(_scan_with_core())
    weaker = next(
        shapely_shape(f["geometry"])
        for f in geojson["features"]
        if f["properties"]["min_dbz"] == 20.0
    )
    parts = list(weaker.geoms) if weaker.geom_type == "MultiPolygon" else [weaker]
    assert sum(len(p.interiors) for p in parts) >= 1


def test_band_rule_types_are_unchanged():
    """These identifiers are an external contract with Audiom's styling.

    Spec 1 established that renaming them breaks the user's map silently.
    """
    geojson = build_storm_intensity_geojson(_scan_with_core())
    rule_types = {f["properties"]["ruleType"] for f in geojson["features"]}
    assert rule_types <= {
        "radar_light_rain", "radar_moderate_rain", "radar_heavy_rain",
        "radar_intense_rain", "radar_severe_core",
    }
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_map_layer.py -k "band" -v`
Expected: FAIL on overlap and on the missing hole — today's bands are separate convex hulls that overlap and each fill their own middle. The `ruleType` test should already pass; if it does not, that is a separate regression to report.

- [ ] **Step 3: Rebuild band construction on `exclusive_bands`**

Replace the per-layer hull with a call to `exclusive_bands` over the object's reflectivity, clipped to the object's own footprint via shapely intersection. Keep every existing feature property — `ruleType`, `heat_value`, `min_dbz`, `max_dbz`, areas, distance and bearing.

- [ ] **Step 4: Run to verify passing, then the full suite**

- [ ] **Step 5: Commit**

```bash
git add src/map_layer.py tests/unit/test_map_layer.py
git commit -m "feat: intensity bands are exclusive annuli, not overlapping hulls"
```

---

### Task 6: The precipitation-field layer

Independent of object detection, so light rain and small showers appear at all. Today they are invisible: detection ignores everything below 20 dBZ and discards anything under 4 km².

**Files:**
- Create: nothing
- Modify: `src/map_layer.py`, `src/server.py` (new endpoint alongside `/map/storms.geojson` at line 402), `src/models.py` if a response model is needed
- Test: `tests/unit/test_map_layer.py`, `tests/smoke/test_server_smoke.py`

**Interfaces:**
- Consumes: `exclusive_bands` (Task 2)
- Produces:
  - `PRECIP_FIELD_LEVELS = (15.0, 20.0, 30.0, 40.0, 50.0, 60.0)`
  - `build_precipitation_field_geojson(scan) -> dict`
  - endpoint `GET /map/precipitation.geojson`

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/unit/test_map_layer.py
from src.map_layer import build_precipitation_field_geojson


def _scan_with_light_rain(rhohv_value: float = 0.99) -> BufferedScan:
    """Echo at 17 dBZ -- below the 20 dBZ object threshold -- plus a tiny patch."""
    reflectivity = np.full((60, 60), np.nan)
    reflectivity[10:30, 10:30] = 17.0        # light rain, forms no object
    reflectivity[50:52, 50:52] = 35.0        # patch far below 4 km2
    mask = np.isfinite(reflectivity)
    scan = _scan_with_mask(mask)
    scan.reflectivity_data.reflectivity = reflectivity
    scan.reflectivity_data.rhohv = np.where(mask, rhohv_value, np.nan)
    scan.reflectivity_data.zdr = np.where(mask, 1.4, np.nan)
    return scan


def test_precipitation_layer_shows_rain_below_the_object_threshold():
    """17 dBZ forms no detected object today, so it is invisible on the map."""
    geojson = build_precipitation_field_geojson(_scan_with_light_rain())
    bands = [f["properties"]["min_dbz"] for f in geojson["features"]]
    assert 15.0 in bands


def test_precipitation_layer_shows_patches_below_the_area_threshold():
    geojson = build_precipitation_field_geojson(_scan_with_light_rain())
    strong = [f for f in geojson["features"] if f["properties"]["min_dbz"] == 30.0]
    assert strong, "a 2x2 gate patch at 35 dBZ produced no feature"


def test_every_band_carries_its_evidence():
    geojson = build_precipitation_field_geojson(_scan_with_light_rain())
    for feature in geojson["features"]:
        properties = feature["properties"]
        assert "median_rhohv" in properties
        assert "median_zdr" in properties
        assert "dominant_class" in properties
        assert "class_fractions" in properties


def test_low_correlation_reports_uncertain_rather_than_guessing():
    """On the clear-air KIWA volume both precipitation and biological classes
    sit at correlation 0.58-0.64 -- indistinguishable, and far too low to be
    rain. No label is supportable there, so the band must say so."""
    geojson = build_precipitation_field_geojson(_scan_with_light_rain(rhohv_value=0.60))
    classes = {f["properties"]["dominant_class"] for f in geojson["features"]}
    assert classes == {"uncertain"}


def test_high_correlation_still_names_a_class():
    geojson = build_precipitation_field_geojson(_scan_with_light_rain(rhohv_value=0.99))
    classes = {f["properties"]["dominant_class"] for f in geojson["features"]}
    assert "uncertain" not in classes
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_map_layer.py -k precipitation -v`
Expected: FAIL with `ImportError: cannot import name 'build_precipitation_field_geojson'`

- [ ] **Step 3: Implement the builder and endpoint**

Contour `scan.reflectivity_data.reflectivity` with `exclusive_bands` at `PRECIP_FIELD_LEVELS`. For each band, compute the evidence attributes from the gates falling within that dBZ range:

```python
# add to src/map_layer.py
from src.contours import exclusive_bands
from src.qc.classifier import CODE_TO_CLASS

PRECIP_FIELD_LEVELS = (15.0, 20.0, 30.0, 40.0, 50.0, 60.0)

# ARW-tuned. Below this median correlation coefficient no class label is
# supportable: on clear-air volumes both precipitation and biological classes
# collapse to 0.58-0.64, indistinguishable and far too low to be rain.
UNCERTAIN_RHOHV = 0.85


def _band_evidence(sweep, lower: float, upper: float) -> dict[str, Any]:
    """Evidence attributes for one intensity band.

    Within a real storm, polarimetry separates rain from insects cleanly:
    precipitation sits near rhohv 0.99 with zdr around +1.4, biological near
    rhohv 0.89 with zdr above +4. In clear air both collapse to rhohv ~0.6,
    where no label is supportable -- so the band reports uncertainty rather
    than guessing, and always carries the raw numbers so the call can be
    checked.
    """
    field = np.asarray(sweep.reflectivity, dtype=float)
    in_band = np.isfinite(field) & (field >= lower)
    if upper != float("inf"):
        in_band &= field < upper

    evidence: dict[str, Any] = {
        "median_rhohv": None,
        "median_zdr": None,
        "dominant_class": "uncertain",
        "class_fractions": {},
    }
    if not in_band.any():
        return evidence

    if sweep.rhohv is not None:
        values = np.asarray(sweep.rhohv, dtype=float)[in_band]
        values = values[np.isfinite(values)]
        if values.size:
            evidence["median_rhohv"] = round(float(np.median(values)), 4)
    if sweep.zdr is not None:
        values = np.asarray(sweep.zdr, dtype=float)[in_band]
        values = values[np.isfinite(values)]
        if values.size:
            evidence["median_zdr"] = round(float(np.median(values)), 4)

    if sweep.gate_classification is not None:
        codes = np.asarray(sweep.gate_classification)[in_band]
        total = float(codes.size)
        fractions = {
            name: round(float(np.count_nonzero(codes == code)) / total, 4)
            for code, name in CODE_TO_CLASS.items()
        }
        evidence["class_fractions"] = {k: v for k, v in fractions.items() if v > 0.0}

    median_rhohv = evidence["median_rhohv"]
    if median_rhohv is not None and median_rhohv >= UNCERTAIN_RHOHV and evidence["class_fractions"]:
        evidence["dominant_class"] = max(
            evidence["class_fractions"], key=evidence["class_fractions"].get
        )
    return evidence


def build_precipitation_field_geojson(scan: BufferedScan) -> dict[str, Any]:
    """The whole precipitation field, independent of object detection.

    Detection ignores echo below 20 dBZ and discards anything under 4 km2, so
    light rain and small showers are invisible on the storm layer. This layer
    has neither threshold.
    """
    sweep = scan.reflectivity_data
    bands = exclusive_bands(sweep.reflectivity, PRECIP_FIELD_LEVELS, sweep)

    features = []
    for (lower, upper), geometry in bands.items():
        if geometry.is_empty:
            continue
        properties = {
            "id": f"{scan.site_id}-precip-{int(lower)}",
            "ruleName": "Precipitation field",
            "ruleType": _intensity_rule_type(classify_intensity(lower)),
            "min_dbz": lower,
            "max_dbz": None if upper == float("inf") else upper,
            "heat_value": lower,
            "site_id": scan.site_id,
            "timestamp": _timestamp_to_str(sweep.timestamp),
            "passable": True,
            "soundPriority": 700,
            "minstep": "100m",
            "maxstep": "300mi",
        }
        properties.update(_band_evidence(sweep, lower, upper))
        features.append({
            "type": "Feature",
            "id": properties["id"],
            "geometry": mapping(geometry),
            "properties": properties,
        })

    return {
        "type": "FeatureCollection",
        "metadata": _storm_metadata("ARW precipitation field"),
        "features": features,
    }
```

Then add the endpoint in `src/server.py`, mirroring `get_storm_map_geojson` at line 402:

```python
@app.get("/map/precipitation.geojson")
def get_precipitation_map_geojson(
    site_id: str = Query(..., description="Radar site identifier"),
    datetime_value: str | None = Query(None, alias="datetime"),
):
    buffered = _ingest_to_buffer(site_id, _parse_datetime(datetime_value))
    return JSONResponse(_json_safe(build_precipitation_field_geojson(buffered)))
```

Match the existing endpoint's parameter handling exactly — read `get_storm_map_geojson` first rather than assuming, since it may take location parameters rather than a site id.

- [ ] **Step 4: Run to verify passing, then the full suite**

- [ ] **Step 5: Commit**

```bash
git add src/map_layer.py src/server.py src/models.py tests/
git commit -m "feat: add precipitation-field layer contoured to 15 dBZ"
```

---

### Task 7: The validation proof

The spec's success criteria, measured on real volumes and recorded as a committed report.

**Files:**
- Create: `tests/e2e/test_proof_shape_truth.py`, `docs/test_reports/2026-08-29-spec2-shape-truth.md`
- Test: itself

**Interfaces:**
- Consumes: everything above

- [ ] **Step 1: Write the proof**

Five assertions, each on real cached volumes:

1. **Walking truth.** False-positive and false-negative rates below 5% for every object tested across all three reference volumes. Record the hull baseline alongside.
2. **Holes are empty.** Sample points inside interior rings; assert no echo at that band's level. A hole that is not genuinely empty is worse than no hole, because the user trusts it.
3. **The seam.** A storm straddling due north yields one polygon whose area matches its gate-summed area. This defect has appeared three times in this project already.
4. **Area agreement.** Contour polygon area against gate-summed area, within 5%, on every cached volume. The hull fails this by 32%.
5. **No invented detail.** No polygon edge shorter than the radar's resolution at that range — 250 m radially, and `range × 0.5°` azimuthally. An edge finer than the measurement asserts knowledge the radar does not have.

- [ ] **Step 2: Prove each fails against the hull**

For assertions 1, 4 and 5, check out the pre-Task-4 `map_layer.py` from git into a scratch path, run the same measurement against hull geometry, and confirm each assertion fails. Record the observed hull values. Do not skip this: it is the whole guard.

- [ ] **Step 3: Measure and record vertex counts**

The spec deliberately left the vertex budget unresolved rather than inventing a cap. Measure what `DEFAULT_SIMPLIFY_M = 250.0` actually produces: minimum, median, maximum and 95th percentile vertex counts per polygon across all three reference volumes, for both layers.

Today's hull output is 8–18 vertices, mean 12. Contours in the hundreds are the honest expectation. Report the real numbers — whether they are workable in Audiom is a question for the project owner, and he cannot answer it without them.

- [ ] **Step 4: Write the report**

`docs/test_reports/2026-08-29-spec2-shape-truth.md`, containing: the walking-truth table with hull baseline against contour result per volume and object; the vertex-count distribution; the area-agreement figures; and an explicit statement of anything that failed or could not be established.

- [ ] **Step 5: Run the full suite**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: green, four Spec 1 xfails preserved and still strict.

- [ ] **Step 6: Commit**

```bash
git add tests/e2e/test_proof_shape_truth.py docs/test_reports/2026-08-29-spec2-shape-truth.md
git commit -m "test: prove shapes tell the truth when walked on"
```

---

## Completion Checklist

From the spec's success criteria:

- [ ] False-positive and false-negative walking rates below 5% on all cached test volumes, hull baseline recorded alongside (Task 7)
- [ ] Contour area agrees with gate-summed area within 5% (Task 7)
- [ ] A storm straddling due north yields one polygon of correct area (Tasks 1, 7)
- [ ] Points inside interior rings contain no echo (Task 7)
- [ ] No polygon edge finer than the range-dependent gate resolution (Task 7)
- [ ] Every new test demonstrated to fail against the convex-hull implementation (all tasks)
- [ ] `ConvexHull` and `_fallback_square` removed from the codebase (Task 4)
- [ ] Vertex-count distribution measured and reported for the project owner's decision (Task 7)
- [ ] Full suite green; the four Spec 1 strict xfails preserved (Task 7)
