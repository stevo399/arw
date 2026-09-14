# Accepting pattern motion for storms neighbours cannot judge (2026-09-13)

## Why

The first tracking benchmark on measured motion (commit `e07e528`) reported speeds up to 118 mph.
Instrumenting the benchmark showed three causes:

1. **Square search window.** The 150 km/h search limit applied to east and north separately, so a
   diagonal match reached 190 km/h (KEYX 23:19:28Z).
2. **Stale velocity.** After an 18.5-minute scan gap, that bad velocity was still reported
   (KEYX 23:37:55Z).
3. **Isolated storms unguarded.** The neighbour-consistency guard needs three other matches within
   60 km, so isolated storms had no guard. Two weak storms 390-440 km from KSOX matched at 119 and
   165 km/h with correlations 0.30 and 0.64, and their velocities became nearby storms' motion.

Causes 1 and 2 were fixed directly (`b12824b`): matches beyond 150 km/h in any direction are
rejected, and velocities older than 20 minutes are ignored. Cause 3 needed evidence.

## Method

**Script:** `scripts/investigations/isolated_storm_motion_eval.py` (ran on `e07e528`); summary in
`2026-09-13-isolated-storm-motion-evaluation.json`.

- Every cached back-to-back window (213 scans) runs through the production tracker.
- For every raw pattern match, the script records the match's correlation, speed, number of judging
  neighbours, and the same storm's previous raw match.
- At the next scan it scores the raw velocity and no motion by centre error and outline IoU. This
  gives 6,698 scored targets.

**Eligible targets:** the match is not on the search edge and is within 150 km/h, and the next step
was not a merge, split or reacquisition. A storm is **isolated** when it has fewer than 3 judging
neighbours (868 targets).

**Scoring rules:** each rule accepts or rejects a match. A rejected storm is scored as no motion.
All rules are scored on the same targets.

## Results: isolated storms (868)

| Rule | Accepted | Median error km | p90 error km | Mean IoU |
|---|---|---|---|---|
| Accept all | 100% | 1.23 | 7.39 | 0.452 |
| No motion | 0% | 1.85 | 5.86 | 0.396 |
| **Correlation >= 0.6** | **88%** | **1.26** | **5.84** | **0.454** |
| Correlation >= 0.7 | 77% | 1.29 | 5.57 | 0.452 |
| Correlation >= 0.8 | 49% | 1.50 | 5.70 | 0.433 |
| Speed <= 100 km/h | 97% | 1.21 | 5.93 | 0.459 |
| Correlation >= 0.6 and speed <= 100 km/h | 88% | 1.26 | 5.76 | 0.455 |
| Agrees with previous match within 25 km/h | 59% | 1.39 | 5.54 | 0.446 |
| Correlation >= 0.7, or agrees with previous | 83% | 1.26 | 5.64 | 0.457 |

**Storms neighbours can judge (4,596)**, for comparison:

| Rule | Median error km | p90 error km | Mean IoU |
|---|---|---|---|
| Accept all | 1.29 | 8.60 | 0.438 |
| Production neighbour guard | 1.23 | 7.24 | 0.449 |
| No motion | 2.20 | 7.64 | 0.351 |

Adding a correlation threshold to the neighbour guard did not help: the median rose to 1.28 km and
p90 changed only to 7.20 km.

## Decision

An isolated storm's match is accepted only when its correlation is at least 0.6.
- It keeps most of the median gain over no motion (1.26 against 1.85 km).
- It brings the worst tenth back to no motion's level (5.84 against 5.86 km).
- It is the simplest rule that does so.

Rules adding a speed cap or agreement with the previous match gave slightly smaller tails at the
cost of extra assumptions; a hard speed cap would reject real fast storms. Both KSOX false matches
above are now rejected: one by the speed limit (165 km/h), the other by correlation (0.30).

## Follow-up: a stationary echo among moving storms

The benchmark on the guards above (commit `c15b027`) still reported the KEYX focus storm
"likely moving NE at 20-27 mph with nearby storms", although its centre stayed 39-40 miles WNW for
20 minutes.

**What instrumentation showed (KEYX 23:42-23:56Z):**
- The storm's own pattern match was about 0 km/h with correlation 0.79 to 0.88, at every scan.
- Within 60 km, a few other echoes also matched at about 0 km/h, and a larger group of storms moved
  NE at about 40 km/h.
- The neighbours' vector median was the moving group's, so the storm's correct match was rejected
  and nearby storms' motion was reported in its place.

**What the evaluation data showed.** Of the 431 matches the neighbour guard rejected, most were
wrong (median error 4.88 km against 2.04 km for no motion). Where production then reported nearby
storms' motion, that helped on average (1.44 against 1.94 km).

The exception was a rejected match that agreed within 10 km/h with the same storm's previous
match. On those 44, the storm's own match predicted best: median 1.24 km, against 1.48 for what
production reported and 1.66 for no motion, with mean IoU 0.481.

**Rule, scored on all 5,464 eligible matches.** Accepting any match the guards reject when it
agrees within 10 km/h with that storm's previous valid match:
- raised acceptance from 90.3% to 91.3%;
- improved median error from 1.239 to 1.232 km and mean IoU from 0.4500 to 0.4508;
- left the 90th percentile unchanged (7.05 against 7.06 km).

Tolerances of 5 and 15 km/h gave the same picture. Adopted at 10 km/h. Edge peaks and matches
beyond the speed limit are never rescued.

A stationary storm therefore reports nearby storms' motion for one scan, until its second agreeing
match corroborates the first.
