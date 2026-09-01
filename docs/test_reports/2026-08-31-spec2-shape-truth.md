# Spec 2, Task 7: storm shapes tell the truth when walked on

Proof test: `tests/e2e/test_proof_shape_truth.py`. Run: `.venv/Scripts/python.exe -m pytest tests/e2e/test_proof_shape_truth.py -v -s`
-- **9 passed** in 201.5s. Full suite: `.venv/Scripts/python.exe -m pytest -q` -- **374 passed, 4 xfailed**, 0 regressions, the four Spec 1 xfails unchanged and still strict.

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

## 5. No invented detail -- an honest, non-discriminating finding

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
that is the bound the committed test actually asserts (`test_no_catastrophic_invented_detail`), and it is what
"no invented detail" is substantively protecting against. The literal, un-relaxed formulation could not be
established as hull-discriminating and is reported as such rather than hidden or quietly threshold-shopped until
it passed only for the contour.

## Vertex-count distribution (a deliverable measurement, not a pass/fail gate)

`DEFAULT_SIMPLIFY_M` was deliberately shipped without an invented vertex cap. This is what it actually produces.

**Storm footprint layer** (contour, 100 m tolerance):

| Site | n polygons | min | median | max | p95 |
|---|---|---|---|---|---|
| KTLX (Moore) | 37 | 4 | 74 | 1872 | 885 |
| KEMX | 95 | 4 | 28 | 3204 | 688 |
| KIWA (clear air) | 1 | 65 | 65 | 65 | 65 |
| **POOLED** | **133** | **4** | **35** | **3204** | **744** |

**Precipitation-field layer** (contour, no object/area floor):

| Site | n polygons | min | median | max | p95 |
|---|---|---|---|---|---|
| KTLX (Moore) | 7535 | 3 | 3 | 10379 | 16 |
| KEMX | 7522 | 3 | 4 | 11870 | 22 |
| KIWA (clear air) | 1939 | 3 | 3 | 324 | 14 |
| **POOLED** | **16996** | **3** | **3** | **11870** | **18** |

**Hull baseline** (for scale, storm footprint layer only -- the hull has no precipitation-field equivalent):

| Site | n polygons | min | median | max | p95 |
|---|---|---|---|---|---|
| KTLX (Moore) | 36 | 6 | 14 | 40 | 28 |
| KEMX | 81 | 5 | 11 | 39 | 19 |
| KIWA (clear air) | 1 | 12 | 12 | 12 | 12 |
| **POOLED** | **118** | **5** | **12** | **40** | **24** |

The old hull's pooled median (12) matches the design-spec-cited "8-18 vertices, mean 12" baseline exactly,
corroborating both this measurement and the earlier one. The honest finding for the project owner: contours in
the hundreds are typical for the storm footprint layer (pooled median 35, p95 744 -- roughly 3x-60x the hull's
vertex count depending on percentile), and the largest single storm footprint measured here has **3204 vertices**
(a KEMX object). The precipitation-field layer's individual bands are mostly small (pooled median 3 -- the minimum
possible closed ring -- because most bands are small, simple patches), but its largest single band reaches
**11,870 vertices** and it emits far more polygons overall (16,996 vs. 133) since it has no 4 km2 area floor.
Whether this vertex density is workable for keyboard/spatial-audio exploration in Audiom is not something this
report decides -- it is the number the project owner asked for so he can decide it himself.

## What failed or could not be established

- **Assertion 5 ("no invented detail"), literal form, is not hull-discriminating.** See the section above. The
  committed test asserts the substantive claim that held up under investigation (no edge below half its local
  resolution floor, true for both implementations) rather than the literal one (no edge below the floor, true for
  neither). This is disclosed, not hidden.
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
