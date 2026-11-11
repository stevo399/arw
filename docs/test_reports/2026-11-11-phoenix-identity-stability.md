# Phoenix storm identity stability (KIWA, 2026-10-30 and 2026-11-01)

## Problem

During two storms that approached Phoenix, storms changed identity between
scans. Motion could not be reported at times, and some storms disappeared for
one or two scans and then came back.

## Method

`scripts/investigations/identity_stability_eval.py` sends each cached KIWA
volume through the server's scan path (`_process_scan_file`), and then through
`StormTracker` with reacquisition on. The script logs identity events for each
object of 20 km2 or more.

- 2026-10-30, 22:01 to 23:51 UTC: 23 volumes, no gaps.
- 2026-11-01, 23:34 to 00:54 UTC: 13 volumes, with some 8 to 12 minute gaps.

Area-weighted life is the number of scans that a track keeps its identity,
weighted by object area. Large storms count the most. An inversion is a merge
where the surviving track is younger than the track that it absorbed.

## Root causes

### 1. Splits flicker (2026-10-30)

The 20 dBZ outline of the complex stayed between 6,200 and 7,700 gates in each
scan. The threshold-hierarchy splitter cut that same outline into 1, 1, 1, 4,
2, 2 and 1 pieces in consecutive scans. A 50 dBZ core of only 6 gates
(`MIN_SEED_PIXELS`) could seed its own storm, and cores that small come and go
between scans. At 22:53 the 9-scan track of the complex broke into four pieces,
and its identity moved to younger tracks. Object centroids moved faster than
100 km/h on 48 steps, while the storms moved at about 40 km/h.

### 2. The assignment forced poor pairs (2026-11-01)

The cost matrix used `UNMATCHED_COST = 10` for "no candidate". The solver had
no cheap way to leave a track unmatched, so it matched every track that it
could. At 00:54:

| Pair | Cost |
|---|---|
| Line track (age 12) to the line (3,291 km2) | 0.48 |
| Line track to a 14 km2 sliver (0.15% overlap) | 1.01 |
| 1-scan track from 99 km away to the line | 1.84 |

The solver chose sliver + far track (2.85), instead of line track + far track
unmatched (0.48 + 10). The line lost its 12-scan history.

## Results

| Variant | 10-30 life | 11-01 life | 10-30 inversions | 11-01 inversions |
|---|---:|---:|---:|---:|
| Before | 9.0 | 8.0 | 18 | 12 |
| Gated assignment, no-match cost 1.0 | 9.2 | 9.3 | 17 | 8 |
| Seed core 40 gates | 12.7 | 8.0 | 17 | 12 |
| Both (adopted) | 13.2 | 9.3 | 16 | 8 |
| Both, and the oldest track survives each merge | 13.8 | 8.0 | 0 | 0 |

The "oldest track survives" rule removes inversions by definition, but it made
2026-11-01 worse. It is not adopted. The remaining inversions need a better
survivor rule, for example the track that gives the most gates to the new
object.

## Changes

- `src/tracking/association.py`: `gated_assignment` gives each track and each
  object its own "no match" choice at `NO_MATCH_COST = 1.0`. Stage 1 and stage 2
  (reacquisition) use it.
- `src/detection.py`: `MIN_SEED_PIXELS` goes from 6 to 40.

## Not reproduced

On 2026-11-01, the user reported that direction was unknown less than an hour
before the storm arrived. In this replay, the line had measured motion (NE, about
40 mph) from 23:38 UTC. Most small cells near Phoenix lived for one scan, so
they had unknown or nearby-storm motion. A tracker restart, or volumes that
the live app skipped, could explain the difference. This is not confirmed.

## Open items

- A survivor rule for merges that does not lose long histories.
- `MIN_SEED_PIXELS` counts gates. A gate covers more ground at long range, so
  a minimum core area in km2 can be more consistent.
