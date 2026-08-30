# Spec 2: True storm shapes

**Date:** 2026-08-29
**Status:** Design — awaiting review
**Builds on:** Spec 1 (`2026-08-22-radar-data-layer-correctness-design.md`), branch `spec1-data-layer`
**Blocks:** nothing. Spec 3 (SCIT) and Spec 4 (hail/mesocyclone) are independent.

## 1. Purpose

ARW emits GeoJSON so blind users can explore weather in Audiom, overlaid on a
map of their own city. Today every storm polygon is a `ConvexHull` of its gate
mask — the shape you get by stretching a rubber band around the storm.

That is not the storm's shape. A convex hull bridges every concavity, fills
every notch, and closes every gap.

### How the user actually explores a shape

Stated directly by the project owner:

> "When exploring a map in Audiom, I walk around on each of the features. I can
> get a good idea of how they are shaped because I can walk across their entire
> surfaces with whatever step size I choose, and I can trace their edges."

Both halves matter, and together they set the requirement:

- **Walking the surface** means the polygon's *interior* must be true. If part
  of the shape contains no echo, the user walks across it and is told there is a
  storm where there is none.
- **Tracing edges** means the *boundary* must be real, at a resolution fine
  enough to feel. A twelve-vertex hull traces as a crude polygon, not as a storm.

### Measured cost of the current implementation

On `cache/KTLX/KTLX20130520_195527_V06.gz` (the Newcastle–Moore EF5 volume),
comparing each object's gate-summed area against its convex hull:

| Object | Gates | True area | Hull area | Inflation |
|---|---|---|---|---|
| 8 | 14,903 | 1,290 km² | 2,235 km² | **1.73×** |
| 13 | 2,251 | 1,454 km² | 1,907 km² | 1.31× |
| 16 | 1,733 | 1,309 km² | 1,554 km² | 1.19× |
| 27 | 6,457 | 1,140 km² | 1,440 km² | 1.26× |
| 30 | 2,784 | 974 km² | 1,182 km² | 1.21× |
| 7 | 28,400 | 949 km² | 1,059 km² | 1.12× |

Overall **1.32× inflation — 24% of the hull area contains no echo**, and 42% for
the worst object. This is a supercell day, where storms are comparatively
blobby; a bowing squall line or a hook echo is worse.

Current output carries 8–18 vertices per polygon, mean 12.

## 2. Scope

### In scope

- A contouring module producing geographic polygons from polar radar fields
- Storm object footprints as true contours, with interior rings
- Intensity bands as exclusive annuli built by geometric difference
- A new precipitation-field layer, independent of object detection
- Evidence attributes carried on every shape
- Validation measuring whether the map lies when walked on

### Out of scope

| Deferred | Reason |
|---|---|
| SCIT cell identification | Spec 3 |
| Hail and mesocyclone algorithms | Spec 4 |
| Classifier calibration | Tracked by two strict xfails from Spec 1 |
| Spatial audio scene | Phase 5 |

## 3. Approach

Three were considered.

**Resampling to a Cartesian grid and contouring there** was rejected: it
interpolates twice, and no single grid spacing matches a radar's resolution at
both 20 km and 200 km. Near the radar it discards detail; far out it invents
it. For a user walking with an arbitrary step size, invented detail is the worse
failure.

**Dissolving each echo gate's true quadrilateral** was rejected on cost: exactly
truthful, but a 14,903-gate storm yields thousands of boundary vertices, and
simplifying afterwards reintroduces the error the approach avoided.

**Chosen: contour the polar grid natively, then map vertices to geography.**
Marching squares runs on the `(rays × gates)` array; each resulting vertex is
converted through `geometry.gate_coordinates`. This preserves the radar's own
resolution with no resampling.

The two layers get different treatment, because they are different kinds of
data:

- An **object footprint** is a *binary* mask. Its contour should hug the gate
  boundary — there is no meaningful value to interpolate between "echo" and
  "no echo".
- An **intensity band** is sampled from a *continuous* field. Interpolating dBZ
  between two gate centres is physically meaningful, so marching squares with
  interpolation is correct there.

## 4. Architecture

### New module: `src/contours.py`

Single responsibility: turn a polar field into geographic polygons. It knows
nothing about storms, GeoJSON or Audiom.

```python
contour_mask(mask, sweep, simplify_m=DEFAULT_SIMPLIFY_M) -> MultiPolygon

contour_field(field, levels, sweep, simplify_m=DEFAULT_SIMPLIFY_M)
    -> dict[float, MultiPolygon]      # cumulative: each level is "at or above"

exclusive_bands(field, levels, sweep, simplify_m=DEFAULT_SIMPLIFY_M)
    -> dict[tuple[float, float], MultiPolygon]   # each band is level_n .. level_n+1
```

`exclusive_bands` lives here rather than in `map_layer.py` because the
difference operation is geometry, not presentation, and both layers need it.
The highest level's band is open-ended (60 dBZ and above).

`sweep` is a `SweepData`, supplying azimuths, ranges, per-ray elevations and the
radar position. Output geometry is shapely, in lon/lat.

### Removed from `src/map_layer.py`

- `mask_to_polygon`, `_sample_mask_points`, `MAX_HULL_POINTS`
- the `ConvexHull` / `QhullError` import
- `_fallback_square`

`_fallback_square` drew a square at the centroid when hulling failed. That is a
fabricated shape presented as measurement. With contours there is no failure
mode that warrants inventing geometry: if a mask yields no valid contour, the
feature is omitted and the reason recorded in the layer's metadata.

### Dependencies

`shapely` and the contouring engine are currently present only transitively via
Py-ART. Both are declared explicitly in `pyproject.toml`. Relying on a
transitive dependency for core geometry is how a future Py-ART upgrade breaks
the map silently.

## 5. Geometry details

### The azimuth seam

The azimuth axis is periodic: ray 0 and ray N−1 both point just either side of
due north. A contour crossing due north would be severed by the array edge.

This is the same defect class that has already appeared three times in this
project — in centroid interpolation (a storm due north reported due south), in
connected-component labelling (one storm becoming two objects), and now here.

Handled by padding the array with a copy of its own leading rows before
contouring, then letting shapely's union dissolve the duplicated overlap. Once
vertices are in lon/lat the seam does not exist, so the pieces genuinely
coincide.

### Simplification in ground metres

One index step in range is 250 m everywhere. One index step in azimuth is 218 m
at 25 km, 873 m at 100 km, 1,745 m at 200 km. Tolerance must therefore be
computed on the ground, never in index space.

`DEFAULT_SIMPLIFY_M = 250.0` — the radar's finest real resolution. This
guarantees no detail the radar actually measured is simplified away, and none is
invented. At long range it retains more vertices than the data strictly
supports, which is harmless.

**Vertex budget.** No hard cap is specified, because none is known to be needed
and inventing one risks discarding real detail to satisfy a guess. Implementation
must *measure* the vertex counts the 250 m tolerance actually produces on the
cached volumes and record them in the report. If any polygon proves impractical
for Audiom, the tolerance is *raised* and simplification re-run — vertices are
never decimated by index, which would bias the boundary toward whichever end the
decimation started from. The current hull output is 8–18 vertices; the honest
expectation is contours in the hundreds, and whether that is workable is a
question for the project owner once the real numbers exist.

### Holes and multiple parts

Shapely nests interior rings within their parent automatically and emits
`MultiPolygon` for disconnected pieces. Audiom performs point-in-polygon and
will not report a feature the user is not inside, so a hole behaves correctly
with no special handling. Confirmed with the project owner.

## 6. Layer 1 — storm objects

The footprint becomes `contour_mask` of the object's gate mask: a MultiPolygon
carrying interior rings wherever the storm genuinely contains clear air.

Intensity bands become **exclusive annuli**, built by geometric difference:

```
band(30, 40) = contour_field(reflectivity, 30) - contour_field(reflectivity, 40)
```

producing a ring with a hole — which is what a 30–40 dBZ region physically is:
moderate rain wrapped around a heavier core.

Today those bands are separate convex hulls that overlap one another and each
fill their own middle, so walking inward passes through several overlapping
claims about the same ground. With difference geometry the user is always in
exactly one intensity band, and stepping inward crosses a real boundary.

Existing feature properties are unchanged: object id, peak dBZ, area, distance
and bearing, rotation, and the `class_fractions` added in Spec 1.

## 7. Layer 2 — the precipitation field

Contoured directly from the reflectivity field, independent of object detection.
No 20 dBZ object threshold and no 4 km² minimum area, so light rain and small
showers appear — both are currently invisible on the map.

**Floor: 15 dBZ.** Levels: 15, 20, 30, 40, 50, 60, built as exclusive bands via
`exclusive_bands`, exactly as Layer 1's intensity bands are. The user is always
inside exactly one band of this layer, and stepping between them crosses a real
boundary.

Rationale for the floor. On the clear-air KIWA volumes, **92% of all valid echo
sits below 10 dBZ** and is overwhelmingly receiver noise, insects and clear-air
return; contouring it would blanket the county on a cloudless day. Between 10
and 15 dBZ is very light drizzle that frequently evaporates before reaching the
ground. From 15 dBZ upward is weather a person would feel.

### Evidence carried on every band

Each band carries median correlation coefficient, median ZDR, the dominant gate
class, and per-class gate fractions.

This matters because of what was measured about the classifier. Within an actual
storm, polarimetry separates rain from insects well:

| | Correlation | ZDR |
|---|---|---|
| precipitation | 0.992 (0.955–0.998) | +1.44 dB |
| biological | 0.893 (0.608–1.002) | +4.44 dB |

That is the textbook signature: rain is uniform so correlation approaches 1;
insects are irregular and horizontally elongated, so correlation falls and ZDR
climbs.

In clear air, both classes sit at correlation 0.58–0.64 — indistinguishable, and
far too low to be rain. There is no precipitation in a clear-air scan, so any
"precipitation" label there is noise.

**Therefore:** where the evidence supports a call, the band names its class.
Where it does not — correlation low across the scan — the band reports
`uncertain` rather than guessing. The underlying numbers travel with the shape
in both cases, so nothing is asserted that the user cannot check.

The layer is served separately so it can be left off when only storms are
wanted.

## 8. Validation

Spec 1 shipped nine tests that passed against the very bug they claimed to
cover. Every test here must be **proven to fail against the convex-hull
implementation** before acceptance. The hull is the known-broken version and is
available in git history.

### The headline metric: does walking the surface tell the truth?

Sample points at random inside each polygon. For each, locate the nearest gate
and ask whether it carries echo at or above that band's level.

- **False-positive rate** — inside the shape, no echo present. This is the lie
  the user walks into. The convex hull scores ~24% on the Moore volume, 42% on
  its worst object.
- **False-negative rate** — echo present, outside the shape. Weather the map
  failed to report.

These two numbers are the spec's success criteria, measured on real cached
volumes and directly comparable to the hull baseline. They require no
interpretation: a 3% false-positive rate means three steps in a hundred are a
lie.

**Target:** false-positive rate below 5%, false-negative rate below 5%, on every
cached volume tested. If either cannot be met, the measured value is reported
rather than the threshold relaxed.

### Holes

Sample points inside interior rings; assert no echo is present. A hole that is
not genuinely empty is worse than no hole, because the user trusts it.

### The seam

A storm straddling due north must produce one polygon whose area matches its
gate-summed area — not two polygons, and not one with a wedge missing.

### Area agreement

Contour polygon area against gate-summed area, on every cached volume. The hull
fails this by 32%. Divergence indicates a broken contour, a severed seam, or a
misassigned hole.

### No invented detail

No polygon edge shorter than the radar's resolution at that range. An edge finer
than the measurement asserts knowledge the radar does not have.

## 9. Success criteria

1. False-positive and false-negative walking rates below 5% on all cached test
   volumes, with the hull baseline recorded alongside for comparison.
2. Contour area agrees with gate-summed area within 5%.
3. A storm straddling due north yields one polygon of correct area.
4. Points inside interior rings contain no echo.
5. No polygon edge finer than the range-dependent gate resolution.
6. Every new test demonstrated to fail against the convex-hull implementation.
7. `ConvexHull` and `_fallback_square` removed from the codebase.
8. Full suite green; the four strict xfails from Spec 1 preserved and still
   strict.
