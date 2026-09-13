# Geographic placement of storms and velocity features (2026-09-13)

Owner requirement: everything on the map is placed at its true geographic location.

## 1. What was wrong

A radar sweep is stored as a grid of rays (azimuth) by gates (range). Ray 0 is wherever the antenna
pointed when the sweep began, and it is adjacent to the last ray. Several computations treated ray
and gate *numbers* as coordinates.

| Defect | Effect, measured on real volumes |
|---|---|
| Storm centre was the reflectivity-weighted mean of ray and gate indices | A storm straddling ray 0 averaged to the far side of the radar. 10 of 49 sampled volumes contained one, misplaced by 12 to 451 km (for example, a 7.8 km² KEYX storm 62 km from the radar was placed 115 km from its true position). Large storms elsewhere were 1 to 4 km off, because a mean in azimuth and range lies on an arc rather than at the shape's centre. |
| Velocity regions and shear couplets were labelled without wrap-around and centred the same way | A region straddling ray 0 became two regions. KMLB 01:18:57Z had 5 such regions. |
| Velocity regions were merged across tilts by comparing masks cell by cell | Each tilt starts at a different azimuth (KTLX 22:47:37Z: 84, 128 and 176 degrees). On three real volumes no region ever matched across tilts; every region had `sweep_count` 1. |
| Rotation candidates were merged across tilts using the index-based centre | Inherited the centre error. |
| The threshold hierarchy that finds a storm's cores labelled without wrap-around | One core straddling ray 0 counted as two, splitting the storm in two (synthetic case; not observed in the 49-volume sample). |
| A multi-core storm's remaining gates went to the nearest core in ray and gate numbers | At long range one ray spans kilometres and one gate 250 m, so gates went to a core that was farther away on the ground. |
| Distance to a storm was its slant range | Map distance is ground distance. The difference is small at low elevation. |

Storm outlines were already correct: contouring pads the sweep across ray 0. On KEYX 18:29:21Z,
KTLX 22:47:37Z and KJAX 17:59:03Z, every gate of every storm lies inside its drawn outline, including
the 5 storms straddling ray 0. Outline area is 0.92 to 1.05 of the storm's area; the difference is
simplification.

## 2. Fixes

- **Storm centre (`ef8a22b`).** Owner decision: area times reflectivity weighting.
  - Gates are projected to the radar-centred azimuthal equidistant plane.
  - The centre is weighted by reflectivity times the ground area each gate covers.
  - Distance and bearing are taken from that centre (`geometry.weighted_geographic_centroid`).
- **Velocity (`ef8a22b`).**
  - Regions and shear couplets are labelled with wrap-around and use the same centre, weighted by
    speed or shear times area.
  - Regions merge across tilts after each tilt's masks are re-ordered by azimuth.
  - Rotation candidates merge by ground distance.
- **Splits (`e0e44ea`).**
  - The hierarchy labels with wrap-around when a storm spans every ray.
  - Remaining gates go to the core nearest on the ground, with core centres weighted like storm
    centres.

## 3. Verification

**Unit tests (synthetic):**
- a storm straddling ray 0 is placed due north, not due south;
- a uniform storm from 50 to 150 km has its centre at 108.3 km (area weighting);
- a 90-degree arc at 100 km has its centre at 90.0 km;
- the same checks for a velocity region straddling ray 0 and a velocity arc;
- regions merge across tilts starting 137 rays apart;
- one core straddling ray 0 stays one storm;
- a gate 5 km from core A and 21 km from core B, but nearer B in index terms, goes to A.

All failed before the fix, except the rotation couplet straddling ray 0. That test passed by
accident, because its two halves merged afterwards, and is kept as a guard.

**Real-data proof (`tests/e2e/test_proof_geographic_placement.py`):**
- **Storm centres.** On KEYX 18:29:21Z, KTLX 22:47:37Z and KJAX 17:59:03Z, every storm's centre is
  within 0.25 km of a reference centre computed differently: gate latitudes and longitudes averaged
  as unit vectors on the sphere, weighted by reflectivity times ground area. Distance and bearing
  agree with the centre. Each volume must contain a storm straddling ray 0. Against the previous
  code the proof fails: 115.28 km on KEYX, and distance mismatches on the other two.
- **Velocity regions.** On KTLX 22:47:37Z and KMLB 01:18:57Z, every region is within 0.25 km of the
  same reference centre, and each volume contains a region straddling ray 0. On KTLX 22:47:37Z,
  regions merge across its three tilts.

**Velocity output, old against new:**

| Volume | Regions | Regions seen on 2+ tilts | Rotation candidates |
|---|---|---|---|
| KTLX 22:47:37Z | 135 -> 122 | 0 -> 12 | 116 -> 115 |
| KMLB 01:18:57Z | 146 -> 139 | 0 -> 1 | 240 -> 242 |
| KEYX 18:29:21Z | 70 -> 57 | 0 -> 8 | 115 -> 115 |

**Splits, old against new, 49 sampled volumes across 9 radars:**
- Storm counts and total covered area are identical in every volume, with no speed change.
- Up to 16 storms per busy volume received different gates between their cores (KJAX 22:13:23Z:
  14 of 142). Quiet volumes were unchanged.
- The seam-core split did not occur in the sample.

**Full suite:** 562 passed, 3 xfailed (the pre-existing expected failures).

**Tracking benchmark** (`2026-09-13-benchmark-after-placement.json` against
`2026-09-13-benchmark-after-alignment.json`):

- **`lower_complexity_extended` and `merge_split_extended`:** every metric unchanged; only
  couplet wording in summaries changed.
- **`merge_split_regression`:** one heading-reversal scan fewer; one reported speed changed from 5
  to 4 mph.
- **`dense_cached_extended`:** the focus storm changed from track 1 ("severe core, 30 miles E") to
  track 2 ("85 miles NE") for the whole window.
  - Explained by instrumenting the first scan (KTLX 23:19:24Z). The ground-distance split moved
    about 600 km² between two cores of one complex 80 to 95 miles NE: storm 21 went from 962 to
    1,559 km², storm 22 from 1,873 to 1,277 km².
  - Storm 21's focus score rose to 13.01, above the 30-miles-E storm's 11.59. The two had been
    nearly tied (11.63 and 11.55).
  - Other metrics moved slightly with that focus change: merges 40 -> 41, splits 22 -> 25, new
    tracks 91 -> 92, lost 36 -> 37, focus heading flips 2 -> 5.
- **`dense_cached_quick`:**
  - Merges 13 -> 14, splits 7 -> 8.
  - One focus switch.
  - At 23:51:18Z speech now says "moving S at 14 mph". That motion comes from the defective
    scene-wide estimate (section 4), not from the storm.

**New finding (not fixed):** speech introduces the focus storm as "Strongest", but focus is chosen
by a score mixing peak, area, persistence, identity and closeness. After the change above, speech
calls storm 21 (64.5 dBZ) the strongest while the storm 30 miles E peaks at 66.0 dBZ.

## 4. Still open

- The scene-wide motion estimate converts a polar-grid shift using a centroid of ray indices, so its
  heading depends on where the sweep began (compact-scan-history report, section 7c). To be
  replaced in the motion redesign.
- Tracking's per-storm local motion window is the bounding box of ray and gate indices, which for a
  storm straddling ray 0 spans every ray. Also part of the motion redesign.
- Speech calls the focus storm "Strongest" although focus is not chosen by strength (section 3).
- A storm with one tracked position is reported "stationary" with high confidence. Its motion is
  unknown. Owner decision: use nearby storms' measured motion, spoken as such, or say motion is not
  yet known.
