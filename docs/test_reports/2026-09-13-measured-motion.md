# Measured storm motion (2026-09-13) -- branch `measured-motion`

Design and plan: `docs/superpowers/specs/2026-09-13-measured-motion-design.md`.

**Evidence behind every rule:**
- `2026-09-13-motion-prediction-evaluation.md`: centre-based estimates.
- `2026-09-13-pattern-motion-evaluation.md`: pattern matching and its guards.
- `2026-09-13-isolated-storm-motion-evaluation.md`: isolated, stationary, fast and distant
  storms, found by the benchmark and the all-windows check below.

## What changed

**Before:**
- Reported motion came from a least-squares fit of a storm's centre track, falling back to a
  scene-wide phase correlation whose heading depended on beam order.
- A storm with one position was reported "stationary" with high confidence.
- Storm matching used the same scene estimate.

**Now:**
- **Measurement.** Each storm seen in a scan is matched against the previous scan: every echo
  within 10 km of it, on a 0.5 km ground grid, by normalized cross-correlation
  (`src/tracking/pattern_motion.py`). A match is a velocity only when all of these hold:
  - its peak is not on the search edge;
  - it is within 150 km/h in any direction;
  - the storm is within 350 km of the radar.
- **Acceptance** (`guard_matches`). A match is accepted when:
  - with 3 or more other matches within 60 km, it is within 25 km/h of their vector median;
  - with fewer, it correlates at 0.6 or better;
  - and in either case it is no faster than 80 km/h.

  A match those rules reject is still accepted when it agrees within 10 km/h with the same storm's
  previous valid match.
- **Reporting** (`report_motion`), by the first rule that applies:
  1. A storm seen in two or more scans with an accepted velocity from the last 20 minutes reports
     the median of its last three. Confidence is high with three, medium with fewer.
  2. Otherwise nearby storms' accepted velocity within 60 km, spoken as "likely moving NE at 20 mph
     with nearby storms". A storm seen once always takes this rule (owner decision).
  3. Otherwise "motion not yet known", with no speed.

  Speeds below 6 km/h, the measured noise bound, are "nearly stationary"; "stationary" is never
  said.
- **Matching** uses each track's reported velocity to predict its position and move its outline,
  and reacquisition uses the same velocity over the elapsed time. `motion_field.py` and the scene
  estimate are removed.
- **API.** `TrackMotion` speeds are null when motion is unknown; sources are `pattern_match`,
  `nearby_storms` and `not_measured`.
- **State.** Tracker state version 2, so state built with the old motion is rebuilt rather than
  mixed with it.

## Verification

**Unit tests:**
- Matcher: synthetic shifts, a storm straddling due north, an absent storm.
- Every guard and rescue rule.
- Reporting rules, the noise threshold, stale velocities, and the records round trip.
- Tracker on synthetic volumes through the real detector:
  - first scan unknown;
  - established storms measured;
  - a newcomer takes nearby storms' motion;
  - a lone storm unknown;
  - a stationary storm among moving ones reports its own motion once corroborated.
- Association moves outlines by measured motion, including across ray 0.
- Speech and API wording.

**Real-data proof** (`tests/e2e/test_proof_measured_motion.py`, KTLX 22:42-23:14Z): reported motion
predicts the next centre and outline better than no motion, and no storm is reported stationary.

**All cached windows** (`scripts/investigations/reported_motion_eval.py`; 213 scans, 9 radars;
`2026-09-13-reported-motion-evaluation.json`). Every storm step predicted from the motion reported
in the previous scan:

| Targets | n | Median error km (reported / no motion) | p90 km | Mean outline IoU |
|---|---|---|---|---|
| All | 6,137 | **1.29** / 2.23 | **7.57** / 7.73 | **0.452** / 0.359 |
| Clean (no merge, split or reacquisition) | 5,097 | **1.15** / 2.13 | **6.90** / 7.40 | **0.459** / 0.360 |
| Measured (`pattern_match`) | 4,692 | **1.21** / 2.22 | **6.98** / 7.51 | **0.492** / 0.392 |
| Nearby storms | 1,445 | **1.59** / 2.27 | 8.61 / **8.43** | **0.322** / 0.252 |

- **By radar:** reported motion's median error beat no motion at every radar except KSOX (21
  steps, 0.46 against 0.45 km).
- **Fastest report:** 79.5 km/h. The same check had earlier found 116 and 117 km/h artifacts,
  fixed by the range cutoff and the 80 km/h corroboration rule.
- **Reported sources:** 6,757 measured, 2,645 nearby storms, 1,761 unknown (first scans and storms
  with nothing measured).

**Tracking benchmark** (`2026-09-13-benchmark-after-measured-motion.json` against
`2026-09-13-benchmark-after-placement.json`):

- **Storm matching:** merges, new tracks, lost tracks and fragmentation are identical in every
  window; splits fell 8 -> 7 in `dense_cached_quick`.
- **Max speed:** 31 -> 42 mph (both dense windows), 3 -> 0 (`lower_complexity_extended`), 27 -> 48
  (`merge_split_regression`), 29 -> 36 (`merge_split_extended`). Before the guard fixes these were
  up to 118 mph; see the isolated-storm report.
- **Speech:**
  - Every window's first scan: "stationary" -> "motion not yet known".
  - Dense KTLX focus storm: "tracking uncertain" and inconsistent S or SSE headings -> "moving ESE
    at 13-18 mph" for the whole window. Focus heading flips 5 -> 0.
  - KEYX focus echo: "nearly stationary"; for one scan, before its second agreeing match, "likely
    moving NNE at 22 mph with nearby storms".
- **Mean uncertain tracks** rose (for example 6.67 -> 19.67) because the counter now includes
  storms whose motion is unknown, which are all storms on a window's first scan.

**Reacquisition survey** (same 213 scans):

| | After alignment | Measured motion |
|---|---|---|
| Storms that went missing | 1,159 | 1,150 |
| Reacquired | 40 | 48 |
| Lost | 785 | 764 |
| Implied speed of reacquired storms, median / max km/h | 11.9 / 26.1 | 18.4 / 57.1 |
| Reacquired with zero raw overlap | 6 | 20 |

Matching now follows measured motion, so a storm that moved several kilometres during a missed scan
can be found. Reacquired displacements reach 16.7 km over 18.6 minutes on the KTLX 2026-04-10
storm day, where measured speeds reach 68 km/h.

**Full suite:** 567 passed, 3 xfailed (the pre-existing expected failures).

**Live check** (current code on a separate port and history directory, 2026-09-14):
- **First scans** (KJAX 04:14:53Z, KMLB 04:14:18Z, KIWA 04:15:50Z): every storm "motion not yet
  known".
- **KJAX 04:19:55Z** (50 storms): 12 measured, 35 nearby storms, 3 unknown. Headings E to SE,
  maximum 34 km/h.
- **KIWA 04:20:56Z** (22 storms): 17 measured, 5 nearby storms. Mostly NE, maximum 24 km/h, 7 nearly
  stationary.

## Limits

- **Template context.** A stationary storm within 10 km of moving storms can be measured with their
  motion, because the template includes surrounding echo.
- **Rule delays.** A stationary storm among moving storms, and a genuinely fast storm, report their
  own motion one scan late, after a second agreeing match. Before that, the stationary storm
  reports nearby storms' motion (spoken as such) and the fast storm may report unknown.
- **Nearby-storms tail.** Motion taken from nearby storms has a slightly worse worst-tenth error
  than assuming no motion (8.61 against 8.43 km); its median and outline overlap are much better.
- **Beyond 350 km** no motion is measured.
- **Focus wording.** Speech still introduces the focus storm as "Strongest", although focus is not
  chosen by strength (placement report, section 3).
