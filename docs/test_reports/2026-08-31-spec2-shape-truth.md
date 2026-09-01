# Spec 2, Task 7: storm shapes tell the truth when walked on

**Fix round 1 (2026-09-01):** this report was corrected after review found the vertex-count
table measured the wrong unit (per polygon fragment instead of per GeoJSON Feature) and stated
its conclusion backwards, and after commit `1803346` landed a 0.5 km² minimum-fragment-area
filter (`MIN_PRECIP_FRAGMENT_AREA_KM2` in `src/map_layer.py`) that changed the precipitation
layer's geometry. Every number below was re-measured against current HEAD of `spec2-shapes`
(commit `1803346`) rather than carried forward. See the "Vertex-count distribution" section for
the corrected table and what changed, and `docs/test_reports/2026-08-31-spec2-vertex-measurement.py`
for the script used. Storm-layer numbers (walking truth, holes, seam, area agreement, the
resolution-floor guard) do not touch the precipitation layer and were re-run fresh to confirm
they are unaffected, rather than assumed unaffected.

Proof test: `tests/e2e/test_proof_shape_truth.py`. Run: `.venv/Scripts/python.exe -m pytest tests/e2e/test_proof_shape_truth.py -v -s`
-- **9 passed** in 198.6s. Full suite: `.venv/Scripts/python.exe -m pytest -q` -- **378 passed, 4 xfailed** in 690.3s,
0 regressions, the four Spec 1 xfails unchanged and still strict (the passed count is 4 higher than the previously
committed report's 374 because commit `1803346` itself added 4 new unit tests for the fragment filter to
`tests/unit/test_map_layer.py`, all passing -- not a change caused by this fix round).

Reference volumes: `cache/KTLX/KTLX20130520_195527_V06.gz` (Moore, OK tornado), `cache/KEMX/KEMX20260712_022646_V06`,
`cache/KIWA/KIWA20260712_170029_V06` (clear air). Hull baseline recovered from git history at
`0bdd90d^:src/map_layer.py` (one commit before Task 4 deleted `ConvexHull`/`MAX_HULL_POINTS`/`_fallback_square`/
`mask_to_polygon`/`object_mask_to_polygon`), executed from that historical source text at test time -- never
reimplemented, never restored to `src/`. `DEFAULT_SIMPLIFY_M` at measurement time: **100.0 m**
(`src/contours.py`, lowered from 250.0 in an earlier fix round -- Douglas-Peucker may displace a vertex by up to
the tolerance, and a full-gate 250 m tolerance could swallow a gate-wide notch whole).

## 1. Walking truth

`measure_walk_truth` (area-weighted; tests echo *presence*, not object membership) at the 20 dBZ detection
threshold, `n_samples=2000, seed=0`, against each object's actual display footprint. Top 6 objects by area per
volume (all objects for KIWA, which produced only one).

| Site | Object | Gates | Area (km2) | Contour FP | Hull FP | Contour FN | Hull FN |
|---|---|---|---|---|---|---|---|
| KTLX (Moore) | 13 | 2251 | 1454.32 | **0.0000** | 0.1941 | 0.1040 | 0.1616 |
| KTLX (Moore) | 16 | 1733 | 1308.69 | **0.0000** | 0.1516 | 0.0393 | 0.0932 |
| KTLX (Moore) | 8 | 14903 | 1290.36 | **0.0000** | 0.1298 | 0.3129 | 0.2299 |
| KTLX (Moore) | 27 | 6457 | 1140.39 | **0.0000** | 0.1237 | 0.0962 | 0.0929 |
| KTLX (Moore) | 30 | 2784 | 974.04 | **0.0000** | 0.1146 | 0.1035 | 0.1475 |
| KTLX (Moore) | 7 | 28400 | 948.89 | **0.0068** | 0.0538 | 0.4049 | 0.4848 |
| KEMX | 55 | 8194 | 4661.27 | **0.0000** | 0.4006 | 0.0935 | 0.1133 |
| KEMX | 3 | 27600 | 3845.73 | **0.0000** | 0.2790 | 0.2373 | 0.3635 |
| KEMX | 49 | 9829 | 2917.14 | **0.0000** | 0.0857 | 0.3199 | 0.4154 |
| KEMX | 48 | 7840 | 2828.72 | **0.0000** | 0.3394 | 0.3204 | 0.4323 |
| KEMX | 5 | 5551 | 1297.09 | **0.0000** | 0.0508 | 0.7557 | 0.8174 |
| KEMX | 44 | 4713 | 854.49 | **0.0000** | 0.3412 | 0.1921 | 0.2565 |
| KIWA (clear air) | 1 | 150 | 5.43 | **0.0336** | 0.4115 | 0.0948 | 0.1158 |

**Pooled mean false-positive rate: contour 0.31%, hull 20.58%.** Every object individually passes the contour's
5% bound (`test_contour_walking_truth_false_positive_under_five_percent`); the hull's pooled mean and its worst
single object (KIWA, 41.15%) both fail that same bound
(`test_hull_baseline_walking_truth_fails_the_five_percent_bound`) -- the guard: this is not a check the old
implementation could have passed.

### False-negative scope -- read this before citing an FN number

Per-object FN above ranges from 3.9% to 75.6% and is **not asserted against a bound and should not be read as
"this object's shape is wrong."** `measure_walk_truth` samples only within a shape's own bounding box. For false
negatives that means every gate counted "outside" this one object that carries real echo is counted as a miss --
including echo that genuinely belongs to a *different*, separately drawn storm nearby, plus ordinary speckle and
sub-threshold blobs that never became a detected object at all. None of that is this object's boundary lying; a
high per-object FN is largely a measurement-scope artifact, not a shape defect. (This is not new to Task 7 --
Task 3's report already found a KEMX object at 84% FN under the hull for exactly this reason, a detached lobe
that fell outside the hull's own bounding box.)

The meaningful, asserted claim instead measures against the **union of every object's footprint emitted for the
whole volume** -- "is there echo this map fails to represent anywhere":

| Site | Union FN | Objects unioned | n_inside | n_outside |
|---|---|---|---|---|
| KTLX (Moore) | **0.0005** | 36 | 163 | 1837 |
| KEMX | **0.0056** | 81 | 222 | 1778 |
| KIWA (clear air) | 0.0948 | 1 | 150 | 338 |

KTLX and KEMX (asserted, `< 0.05`) both pass with wide margin: 0.05% and 0.56% of nearby ground area carries
unrepresented >= 20 dBZ echo. **KIWA is reported, not asserted.** It detected exactly one object in this clear-air
scan, so its "union" is that single 150-gate object and the union scope collapses back to the per-object scope --
its 9.48% is a real number, honestly reported, but it is not evidence the sweep-wide claim fails; it reflects that
there was essentially nothing else in the sweep to union against.

## 2. Holes are empty

Interior rings of every contoured footprint were sampled (up to 200 points per hole, seeded) and checked for echo
at the 20 dBZ level. Three outcomes were distinguished for a sample that DOES carry echo, because not all of them
are a lie to the user:

| Site | Holes | Sampled inside | Echo found | Same-object boundary artifact | Claimed by another storm | **Unclaimed** |
|---|---|---|---|---|---|---|
| KTLX (Moore) | 762 | 2032 | 79 | 61 | 0 | **18 (0.89%)** |
| KEMX | 878 | 2508 | 38 | 0 | 0 | **38 (1.52%)** |
| KIWA (clear air) | 1 | 0 | 0 | 0 | 0 | 0 |

- **Same-object boundary artifact** (KTLX only, 61 of 79): the sampled gate genuinely belongs to this exact
  object's own raster mask; the hole boundary and the mask disagree by a sliver, consistent with
  `DEFAULT_SIMPLIFY_M` (100 m) displacing a vertex enough to nominally exclude a handful of the object's own edge
  gates from its own hole ring. The echo is real and already attributed to this object -- not a fabricated empty
  claim.
- **Unclaimed** (the genuine failure mode): real echo, inside a hole, belonging to no drawn shape at all. Both
  volumes pass the asserted `< 5%` bound (0.89%, 1.52%) but this is **not zero** and is reported honestly rather
  than rounded away. Plausible source (not confirmed further): fragments the hierarchy split produced or the
  4 km2 minimum-object-area floor dropped entirely, small enough to fall below detection but still >= 20 dBZ,
  sitting geometrically inside a larger neighbour's hole.

## 3. The seam

No cached reference volume is guaranteed to have a real storm sitting exactly on due north today, so this uses a
synthetic mask (matching `tests/unit/test_contours.py`'s own seam-test pattern) built on KTLX's real sweep
geometry -- only the mask is synthetic; the azimuth/range-to-lat/lon conversion is the genuine production
pipeline.

**Result: 1 polygon** (not split at the array boundary). Polygon area 214.1194 km2 vs. gate-summed area
213.4122 km2 -- **relative difference 0.33%**, well inside the 5% bound.

## 4. Area agreement

Whole-volume total: every detected object's own contour/hull area summed, against every object's own gate-summed
area summed.

| Site | Contour area (km2) | Hull area (km2) | Gate-summed area (km2) | Contour ratio | Hull ratio |
|---|---|---|---|---|---|
| KTLX (Moore) | 11531.3 | 13448.6 | 11446.2 | **1.0074** | 1.1749 |
| KEMX | 22513.7 | 31947.6 | 21802.9 | **1.0326** | 1.4653 |
| KIWA (clear air) | 5.4 | 9.0 | 5.4 | **0.9867** | 1.6593 |

Contour passes the 5% bound on all three volumes (worst deviation 3.26%, KEMX). Hull fails all three by a wide
margin (17.5%-65.9% over) -- worst single-volume deviation 65.9% (KIWA), consistent in direction and order of
magnitude with the spec's cited ~1.32x hull inflation (pooled across all three volumes here: hull totals
45,405 km2 against 33,255 km2 gate-summed, a **1.365x** ratio).

## 5. Resolution-floor guard -- deliberately one-sided, not a hull comparison

**Correction (fix round 1):** this section previously read as though the assertion below
distinguished the contour implementation from the hull it replaced. It does not, and by
construction cannot: a convex hull is coarser than the radar's resolution floor, not finer. It
omits detail; it never invents any, so there is no formulation of "no invented detail" a
12-vertex hull could violate. The assertion below is a **one-sided guard against a failure mode
only the contour implementation can have** (e.g. simplification being disabled or broken) -- it
is legitimate and worth keeping, but it is not evidence that the contour beats the hull, and this
report must not be read as claiming that.

Floor per edge: `min(250 m radial gate depth, range x 0.5 deg azimuthal beamwidth)` at that edge's own midpoint
range. Measured for every edge of every emitted polygon, both implementations, all three volumes.

| Site | Impl | Edges | Below floor | Below **half** floor |
|---|---|---|---|---|
| KTLX (Moore) | contour | 8367 | 1686 (20.2%) | **0 (0.0%)** |
| KTLX (Moore) | hull | 536 | 119 (22.2%) | **0 (0.0%)** |
| KEMX | contour | 14537 | 2430 (16.7%) | **0 (0.0%)** |
| KEMX | hull | 961 | 188 (19.6%) | **0 (0.0%)** |
| KIWA (clear air) | contour | 65 | 9 (13.8%) | **0 (0.0%)** |
| KIWA (clear air) | hull | 12 | 1 (8.3%) | **0 (0.0%)** |

Taken completely literally (`edge_length >= floor`), this assertion **does not discriminate between the contour
and the hull it replaced** -- both show 8-22% of edges landing marginally under their local floor. Investigated
further before accepting that at face value: the violating edges are not fabricated fine detail. Their
edge-length-to-floor ratio never drops below 0.5 on either implementation (0 edges out of 8367+536+14537+961+65+12
= 24478 total edges measured, across all three volumes, both implementations); the contour's worst 10th-percentile
violation still sits at 73.6% of its floor, and the hull's violations cluster at essentially exactly the floor
(ratio median 0.997) -- consistent with gate-grid quantization and aeqd floating-point round-trip precision
landing a hair under a hard cutoff, not with either implementation asserting knowledge the radar does not have.
**No polygon edge on either implementation, anywhere measured, falls below half its local resolution floor** --
that is the bound the committed test actually asserts
(`test_resolution_floor_guard_no_edge_below_half_floor`), and it is what "no invented detail" is substantively
protecting against. The literal, un-relaxed formulation could not be established as hull-discriminating -- because
no formulation of it can be, per the correction above -- and is reported as such rather than hidden or quietly
threshold-shopped until it passed only for the contour.

## Vertex-count distribution (a deliverable measurement, not a pass/fail gate)

**Correction (fix round 1):** this table was first published counting vertices **per disjoint
polygon fragment**. That is not the unit that matters. `build_storm_geojson` and
`build_precipitation_field_geojson` each emit **one GeoJSON Feature per object / per intensity
band**, and a map viewer loads a Feature, not a fragment inside it -- a Feature's geometry can be
a MultiPolygon holding hundreds of patches. Counting per fragment understated the precipitation
layer's per-shape burden by roughly two orders of magnitude and, worse, stated the comparison
between the two layers **backwards**: the first version of this table said the largest single
precipitation band reached "11,870" vertices, making the precipitation layer look lighter than
the storm layer. Measured per Feature -- the object actually loaded -- the largest precipitation
band is **53,410 vertices**, and the precipitation layer is **roughly 9x heavier** than the storm
layer by pooled total, not lighter. `DEFAULT_SIMPLIFY_M` was deliberately shipped without an
invented vertex cap; this is what it actually produces, re-measured against current HEAD
(commit `1803346`). The measurement script is checked in at
`docs/test_reports/2026-08-31-spec2-vertex-measurement.py`.

**Per-Feature vertex counts -- the unit a map viewer loads -- both layers, side by side:**

| Layer | Site | n Features | min verts/Feature | median | p95 | max | total verts |
|---|---|---|---|---|---|---|---|
| Storm footprint | KTLX (Moore) | 36 | 10 | 80 | 1,002 | 2,222 | 9,166 |
| Storm footprint | KEMX | 81 | 12 | 35 | 759 | 3,535 | 15,510 |
| Storm footprint | KIWA (clear air) | 1 | 67 | 67 | 67 | 67 | 67 |
| Storm footprint | **POOLED** | **118** | **10** | **40** | **823** | **3,535** | **24,743** |
| Precipitation band | KTLX (Moore) | 6 | 42 | 19,065 | 24,083 | 24,608 | 88,336 |
| Precipitation band | KEMX | 6 | 20 | 14,999 | 51,382 | 53,410 | 130,028 |
| Precipitation band | KIWA (clear air) | 2 | 505 | 1,141 | 1,713 | 1,777 | 2,282 |
| Precipitation band | **POOLED** | **14** | **20** | **11,844** | **48,138** | **53,410** | **220,646** |

Read per Feature, the precipitation layer is far heavier to load than the storm layer: its
largest single Feature (53,410 vertices) is roughly **15x** the storm layer's largest Feature
(3,535), and its pooled total (220,646) is roughly **9x** the storm layer's pooled total (24,743)
despite emitting only 14 Features against the storm layer's 118 -- because it has no 4 km2
object-detection area floor and each Feature is one whole intensity band's MultiPolygon rather
than one object's.

**Cross-check against the fix brief's independently-measured calibration table.** For the
storm-footprint row, every column above matches the brief's calibration exactly, including the
decimal (p95 = 823.45 -> 823). For the precipitation-band row, `n` (14), `min` (20), and `median`
(11,844) match exactly; `p95`, `max`, and `total` do not (this report: 48,138 / 53,410 / 220,646;
brief's calibration: 49,613 / 56,875 / 227,887 -- roughly 3-6% higher). Investigated rather than
silently adopting either set, per the fix brief's instruction:

- The mismatch is confined to the two largest, most fragment-dense Features (KEMX's 15-20 dBZ
  and 20-30 dBZ bands) -- the other 12 of 14 precipitation Features, and every storm-layer
  number without exception, reproduce bit-for-bit against the calibration table.
- Ruled out: non-determinism (identical result across repeated runs in this process and
  environment), a replaced or duplicated cache file (one copy of each reference volume on disk,
  hashes checked directly), dependency-version drift (`shapely` has been pinned to `2.1.2` / GEOS
  `3.13.1` in `uv.lock` across this commit's entire history, not a floating minimum), and a
  geometry-type edge case silently dropping vertices (every precipitation Feature's geometry
  parses as a valid `MultiPolygon`; none is a `GeometryCollection`).
- The **pre-filter** fragment counts (unaffected by any threshold decision) reproduce this
  report's own earlier pre-filter measurement exactly -- 16,996 total pieces pooled across all
  three volumes, 7,535 for KTLX alone -- which rules out a difference in the underlying
  contour/simplify geometry itself, the thing coverage-simplify's floating-point behaviour would
  most plausibly have disturbed.
- Swept the fragment-drop threshold against the measured fragment-area distribution: only 58 of
  16,996 fragments sit within 2% of the 0.5 km2 cutoff -- far too few to account for the
  ~1,400-fragment gap between this measurement's dropped count (14,941) and the brief's stated
  16,355 -- so a floating-point boundary effect on the drop decision does not explain it either.
- No further explanation was found from this side within the scope of this task. Since the storm
  layer and every pre-filter (topology-only) precipitation figure reproduce exactly, the
  discrepancy looks confined to how the *post-filter* precipitation aggregate was produced on the
  calibration side, not to this measurement -- but that could not be confirmed further without
  access to how the calibration table was generated. The numbers in this report are the ones a
  command in this repository, against this commit, actually produces; per the fix brief's
  instruction not to adopt either set silently, they are reported as measured rather than
  reconciled to the calibration table.

**What the fragment filter changed** (`MIN_PRECIP_FRAGMENT_AREA_KM2 = 0.5`, commit `1803346`),
pooled across all three volumes: pre-filter, the precipitation layer contoured **16,996 fragments
totalling 299,394 vertices**; post-filter (current HEAD, as shipped) it contours **2,055
fragments totalling 220,646 vertices (a 26.3% reduction)**. **14,941 fragments totalling 951.0
km2** were dropped as sub-0.5 km2 speckle -- now discoverable per band in the layer's own
metadata via `omittedFragmentCount` / `omittedFragmentAreaKm2` (per volume: KTLX 7,022 fragments
/ 340.8 km2, KEMX 6,030 / 541.6 km2, KIWA 1,889 / 68.5 km2). The `MIN_PRECIP_FRAGMENT_AREA_KM2`
comment in `src/map_layer.py` previously cited a pre-filter figure from an earlier, non-reproducing
measurement (7,876 pieces / 131,472 vertices for KTLX alone); it has been corrected to the figure
re-derived here (7,535 pieces / 127,049 vertices, KTLX only, pre-filter), which matches this
report's own pre-filter measurement and the pre-filter figure independently reported before
commit `1803346` existed.

**Fragment-level counts** (the wrong unit for judging per-Feature load -- kept here only because
it is the quantity `MIN_PRECIP_FRAGMENT_AREA_KM2` was tuned against, and it is what the layer's
own source comment now cites):

| Site | n polygons (fragments) | min | median | max | p95 |
|---|---|---|---|---|---|
| KTLX (Moore) | 513 | 3 | 16 | 10,379 | 525 |
| KEMX | 1,492 | 3 | 11 | 11,870 | 164 |
| KIWA (clear air) | 50 | 5 | 29 | 324 | 129 |
| **POOLED** | **2,055** | **3** | **12** | **11,870** | **218** |

**Hull baseline** (for scale, storm footprint layer only -- the hull has no precipitation-field equivalent):

| Site | n polygons | min | median | max | p95 |
|---|---|---|---|---|---|
| KTLX (Moore) | 36 | 6 | 14 | 40 | 28 |
| KEMX | 81 | 5 | 11 | 39 | 19 |
| KIWA (clear air) | 1 | 12 | 12 | 12 | 12 |
| **POOLED** | **118** | **5** | **12** | **40** | **24** |

The old hull's pooled median (12) matches the design-spec-cited "8-18 vertices, mean 12" baseline exactly,
corroborating both this measurement and the earlier one. The honest finding for the project owner, corrected:
**the precipitation-field layer is the heavier of the two layers to load, not the lighter one.** Per Feature, the
storm footprint layer runs from 10 to 3,535 vertices (pooled median 40, p95 823 -- roughly 3x-60x the hull's
vertex count depending on percentile); the precipitation-field layer runs from 20 to 53,410 vertices (pooled
median 11,844 -- roughly 300x the storm layer's median) despite emitting far fewer Features overall (14 vs. 118),
because it has no 4 km2 area floor and each Feature is one whole intensity band's entire MultiPolygon rather than
one object. Whether this vertex density is workable for keyboard/spatial-audio exploration in Audiom is not
something this report decides -- it is the number the project owner asked for so he can decide it himself.

## What failed or could not be established

- **Assertion 5 ("no invented detail") is a deliberately one-sided resolution-floor guard, not a
  hull-comparison, and cannot be made into one.** A convex hull is coarser than the resolution
  floor, not finer, so no formulation of "no invented detail" exists that a 12-vertex hull could
  violate -- see the corrected section above. The committed test
  (`test_resolution_floor_guard_no_edge_below_half_floor`) asserts the substantive claim that
  held up under investigation (no edge below half its local resolution floor, true for both
  implementations) rather than the literal one (no edge below the floor, true for neither). This
  is disclosed, not hidden, and is not evidence the contour beats the hull on this axis -- it
  guards only against the contour's own simplification breaking.
- **The precipitation-band vertex-count calibration cross-check did not fully reconcile.** See
  the "Vertex-count distribution" section above: `n`, `min`, and `median` match the fix brief's
  independently-measured calibration exactly, but `p95`, `max`, and `total` are 3-6% lower here,
  confined to the two largest bands. Several causes were ruled out (non-determinism, cache-file
  drift, dependency-version drift, geometry-type dropout, threshold boundary sensitivity); no
  further explanation was found within this task's scope. Reported as measured, not reconciled.
- **Holes are not perfectly empty.** 0.89% (KTLX) and 1.52% (KEMX) of hole-interior samples carry real,
  unclaimed echo -- small, within the asserted 5% bound, but non-zero and not explained further than "plausibly a
  hierarchy-split fragment or area-floor-dropped blob." Not chased further per the task's scope (measure and
  report, not re-engineer detection).
- **KIWA's false-negative number cannot be given the strong "sweep-wide" reading** the other two volumes get,
  because it detected only one object; its 9.48% union FN is reported honestly rather than asserted against the
  5% bound the other two volumes clear.
- Everything else measured (walking-truth false positives, the seam, area agreement, and the hull-fails guard for
  all three) passed cleanly with wide margins on real cached volumes, and the hull baseline was proven, on the
  same objects and the same metric, to fail every bound the contour passes.
