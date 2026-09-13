# Compact scan history: proofs, benchmark and live checks (2026-09-12)

Branch `radar-history`. Spec: `docs/superpowers/specs/2026-09-12-compact-scan-history-design.md`
(Amendments 1 and 2). Plans: `docs/superpowers/plans/2026-09-12-radar-correctness-fixes.md`,
`docs/superpowers/plans/2026-09-12-compact-scan-history.md`.

All proofs used Level II volumes already in `cache/`; no radar data was downloaded except in
the live checks.

## 1. Compact scan size and losslessness

Proof: `tests/e2e/test_proof_compact_scan.py` (3 passed). Every one of the five layer builders
(footprints, intensity bands, Audiom centroids, centroids, precipitation field) produces
byte-identical GeoJSON from the compact scan and from the original processed scan. Reflectivity
is bit-identical, including NaN positions, and every object mask equals the original.

| Volume | Objects | Compact scan | Reflectivity | Labels | Records | Arrays retained per scan before |
|---|---|---|---|---|---|---|
| KTLX 2026-04-10 22:42:09Z | 63 | 272 KB | 250 KB | 9 KB | 6 KB | 98.9 MB |
| KEMX 2026-07-12 02:26:46Z | 81 | 262 KB | 234 KB | 15 KB | 7 KB | 122.7 MB |
| KIWA 2026-07-12 17:00:29Z | 1 | 180 KB | 171 KB | 1 KB | 1 KB | 17.1 MB |

Records here carry no tracking snapshot; in the live pipeline, tracked scans measured
277-280 KB (KTLX) and 148-152 KB (KSOX) including their snapshots (section 3).

## 2. Identical tracking

Proof: `test_history_tracking_is_identical_to_retained_scan_tracking` (passed), on the 7-scan KTLX
window 2026-04-10 22:42:09Z to 23:14:07Z.

- With reacquisition disabled, a `RadarHistory` -- whose tracker reloads its previous scan from
  compact records -- produced the same tracking snapshot, byte for byte, as a tracker retaining
  full scans, at every scan. The final exported tracker state was also identical.
- With reacquisition enabled, no storm was reacquired in this window, so its snapshots were
  identical too (`first_reacquisition_divergence: null`). **Reacquisition is therefore proven by
  the synthetic tests only** (`tests/unit/test_tracking_reacquisition.py`, 14 tests), not yet by
  an observed real-data reacquisition. (Real-data reacquisitions were surveyed on 2026-09-13; see
  section 7c.)

| Scan | Objects | Active tracks | Reacquired |
|---|---|---|---|
| 22:42:09 | 63 | 63 (was 63) | none |
| 22:47:37 | 53 | 53 (was 50) | none |
| 22:52:55 | 51 | 51 (was 49) | none |
| 22:58:05 | 49 | 49 (was 47) | none |
| 23:03:32 | 50 | 50 (was 48) | none |
| 23:08:50 | 61 | 61 (was 57) | none |
| 23:14:07 | 57 | 57 (was 54) | none |

Active-track counts are from the 2026-09-13 rerun after the storm-identity fix (section 7b); the
counts in parentheses, from 2026-09-12, were lower because continuing storms had been merged away.

## 3. Memory

Proof: `test_retained_memory_is_bounded_and_no_full_grids_remain` (passed). Live-style ingest
through the server of five scans each on KTLX (22:42-23:03Z) and KSOX (22:35-23:09Z), with all
four map layers and the precipitation layer rendered for every scan.

| Measurement | Result | Limit |
|---|---|---|
| Retained compact history, 10 scans | **2.14 MB** | 3.08 MB |
| Retained rendered map layers | **32.1 MB** | 88.3 MB |
| Full-size grids retained after ingest | **none** | none |
| Previous pipeline, arrays for 2 scans on 2 radars | 395.7 MB | -- |
| Transient traced peak during ingest and rendering | **2.65 GB** | not gated (spec A6) |

**Update 2026-09-13 -- the transient peak's root cause was found and fixed (`55fc07c`).** Measuring
each pipeline stage on real KTLX volumes showed object detection peaking at 2,243 MB and still
holding 2,226 MB when it returned (KEMX: 3,557 MB / 3,539 MB); every other stage peaked at 356 MB
or less. Detection kept a full-grid mask (1.3 MB) for every threshold-hierarchy node of every
object; one KEMX storm complex of 36,222 gates produced 1,329 nodes and a 1,765 MB build peak.
Components are now labeled once per threshold inside the blob's bounding box, with full-grid masks
built only for split seeds. Against the previous code on the same volumes, every object, mask,
label and hierarchy node (2,429 KTLX, 7,075 KEMX) is identical, and:

| Volume | Detection peak before | after | Held after detection before | after |
|---|---|---|---|---|
| KTLX 22:42:09Z (63 objects) | 2,243 MB | **97 MB** | 2,226 MB | **89 MB** |
| KEMX 02:26:46Z (81 objects) | 3,557 MB | **121 MB** | 3,539 MB | **113 MB** |

Largest remaining per-stage peaks on KTLX: Level II parse 356 MB (159 MB held through processing),
quality-control preprocessing 148 MB and its refresh 148 MB, intensity-band layer 107 MB,
precipitation layer 100 MB.

The memory proof, rerun after the fix on the same ten scans with every layer rendered, measured a
transient traced peak of **546 MB** (was 2,655 MB). Retained history 2.15 MB, retained rendered
layers 32.1 MB, and no full-size grids retained, all unchanged.

Findings to act on:

- **Rendered GeoJSON is now the largest retained memory.** A busy KTLX scan's rendered layers
  measured 5.9-6.7 MB; KSOX's 0.14-0.22 MB. With the defaults (newest scan pinned for up to 20
  radars, plus 10 others) a busy day could hold roughly 200 MB of rendered layers. Bounded, but
  worth reducing.
- **The transient peak is large.** 2.65 GB of Python-traced allocations at peak across ingest,
  tracking and rendering. Retained memory between scans is solved; brief peaks are not, and which
  step dominates was not measured. Tracing also slowed the run (about 7 minutes per busy scan).

## 4. Restart and browsing

- `test_restart_mid_window_continues_identically` (passed): reopening the KTLX history from disk
  before its fourth scan produced tracker state identical to an uninterrupted history, with
  `continuity_rebuilt` false.
- `test_historical_request_leaves_live_tracking_unchanged` (passed): after three live KTLX scans,
  a request for 21:49:29Z returned no tracking context and left the ring and tracker state
  byte-identical.

## 5. Tracking benchmark (offline manifest)

Baseline `2026-09-12-benchmark-before-history.json` was captured before any history tracker
change; `2026-09-12-benchmark-after-history.json` after Task 11. Object counts, merges, splits,
new tracks, heading flips and fragmentation are **identical** in every window, confirming stage-1
association did not change. Every difference follows from the `missing` status (spec A9):

| Window | Metric | Before | After | Explanation |
|---|---|---|---|---|
| all | mean_active_tracks | 57.67 / 62.25 / 20.4 / 17.86 | 48.67 / 51.12 / 18.6 / 15.86 | missing storms are no longer counted as active |
| all | mean_uncertain_tracks | 8.0 / 11.88 / 1.2 / 1.86 | 6.0 / 7.5 / 1.0 / 1.57 | missing tracks (identity lowered by a miss) are no longer active |
| all | lost_tracks_total | 15 / 77 / 9 / 14 | 0 / 51 / 0 / 3 | a track is lost only once unrecoverable, up to 20 minutes |
| dense_cached_quick | focus at 23:56:21Z | track 8, suppressed motion, continuity low | track 62, S at 7 mph, continuity medium | track 8 missed that scan; before it stayed active and kept focus on a storm absent from the scan |
| dense_cached_quick | focus_switches, summary_motion_published_count, focus_motion_suppressed_source_scans, focus_low_continuity_scans, focus_low_motion_scans, mean_focus_* | 1, 1, 1, 1, 1 | 2, 2, 0, 0, 0 | the same focus change |
| merge_split_* | mean_focus_identity_confidence, mean_focus_continuity, mean_focus_selection_margin | 0.74 / 0.70-0.75 / 1.27-1.7 | 0.78 / 0.73-0.77 / 1.68-2.05 | low-identity missing tracks no longer compete for focus |
| dense_cached_extended | mean_focus_selection_margin | 1.93 | 1.99 | the same |

The 23:56:21Z cause was confirmed by instrumenting the benchmark's own pipeline: at that scan
track 8 is `missing` (1 missed scan, identity 0.62) and track 62 holds focus.

That instrumentation also exposed a defect introduced by the missing status: the missing track
kept `is_primary_focus = True`, so a reacquired storm would have returned as a second focus.
Fixed (`0bea661`); a full benchmark rerun after the fix was byte-identical to the after-file,
confirming published focus did not change.

## 6. Defects found and fixed during this work

| Defect | Effect | Fix |
|---|---|---|
| 1. Precipitation evidence stripped | every band reported no RhoHV, "uncertain" | evidence precomputed during processing |
| 2. Missed storm never re-matched | a storm missing one scan got a new identity | stage-2 reacquisition with the missing status |
| 3. History requests reset live tracking | viewing an older scan wiped every live track | analysis separated from tracking; per-scan snapshots |
| 4. Summary matched a stale track | a storm could be spoken with another storm's motion | only tracks seen in the scan match |
| Specific-time status blocked | Weather Kitten reported ARW unavailable while it worked, and polled the latest scan | background preparation; requested time forwarded |
| Stale focus flag on missing tracks | a reacquired storm would be a second focus | all flags cleared before focus selection |
| In-use radar eviction (plan review) | the radar being used could be dropped from memory | eviction skips the in-use radar |

## 7. Live checks (real NEXRAD, KJAX, 2026-09-13)

ARW on branch `radar-history`, port 8765, history under `cache/KJAX/history/`.

- **Retention:** three live volumes (04:42:20Z, 04:46:54Z, 04:51:27Z) retained, all `tracked`; each
  `.arwscan` file 283-285 KB; `tracker_state.json` 199 KB after three scans; no history write error.
- **Scan-owned tracking:** `/tracks?datetime=04:42:20Z` returned 31 tracks each with 1 position
  (its own time); the newest scan returned 13 tracks with 3 positions.
- **Evidence on an older scan:** the 04:42:20Z precipitation layer carried median RhoHV on every
  band (0.805, 0.968, 0.995, 0.992, 0.992).
- **Weather Kitten against live ARW:** `format_radar_snapshot` stepped newest to oldest and back on
  all four layers (16 steps). Every map pinned `datetime` to that scan's own timestamp, position
  text was correct at each step ("Scan 1 of 3 retained scans, 04:42 UTC, 31 detected echoes,
  tracked."), and the rendered section marked exactly one scan `aria-current`.
- **Restart:** after stopping and restarting ARW, `/map/history` listed the same three scans and
  `/tracks` at 04:51:27Z the same 24 track IDs with the same position counts,
  `continuity_rebuilt: false`. The next live scan (05:00:16Z) continued them: 9 tracks reached 4
  positions, not rebuilt.
- **Not verified here:** a keyboard-and-NVDA pass through the Weather Kitten page (needs the
  owner), and an observed real reacquisition (none occurred).

## 7b. Storm identity: a continuing storm was merged away (fixed 2026-09-13)

**Found:** the newest KJAX scan had 29 objects but 24 active tracks, and the offline benchmark showed
the same pattern (49 objects / 47 active tracks; 45 / 44). Every object mapped to exactly one track,
but some tracks were matched to their own object and then marked `merged` into a neighbour in the
same update (for example object 40 -> track 38, `merged_into` 12). Such a storm was present in the
scan but absent from `/tracks`, without motion in speech, and ineligible for focus.

**Root cause:** association listed every track that scored against an object as a merge candidate
for it, including tracks the global assignment had matched to a different object. The mirror case
also existed: an object matched to one track could be listed as a split child of another.

**Owner decision:** a storm that is the same storm as a previous track must not become a different
one. **Fix (`e32a835`):** only an unmatched track can merge, and only an unmatched object can be a
split child. An April tracker test had encoded the defect (storm A, best matched to its own
continuation with 60% overlap, merged away and its continuation given a new identity); it now
asserts that A keeps its track.

**Verified:** on the benchmark pipeline active tracks now equal objects in every scan (55/55, 49/49,
45/45), with no object on a merged track.

**Benchmark effect** (`2026-09-13-benchmark-after-identity-fix.json` against
`2026-09-12-benchmark-after-history.json`), all in the direction of fewer false identity changes:

| Window | Merges | Splits | New tracks after first scan | Fragmentation proxy |
|---|---|---|---|---|
| dense_cached_quick | 15 -> 8 | 9 -> 7 | 34 -> 27 | 0.228 -> 0.181 |
| dense_cached_extended | 53 -> 29 | 33 -> 24 | 131 -> 100 | 0.313 -> 0.239 |
| lower_complexity_extended | 5 -> 2 | 1 -> 1 | 15 -> 5 | 0.625 -> 0.208 |
| merge_split_regression | 12 -> 5 | 6 -> 5 | 23 -> 16 | 0.242 -> 0.168 |
| merge_split_extended | 14 -> 6 | 11 -> 8 | 41 -> 28 | 0.353 -> 0.241 |

The clearest case, `lower_complexity_extended`: a steady scene of four storms previously created
three new tracks every scan (tracks seen 4, 7, 10, 12, 13, 19), meaning three of the four storms were
renamed each scan; after the fix it created none while the scene was stable (4 tracks seen through
23:44Z). The focus storm is track 1 throughout. Its selection margin fell (5.16 -> 2.76) because its
competitors are now persistent tracks rather than brand-new ones, and at 23:18Z it now has real
track history, so motion is published from it (ENE 3 mph). Mean uncertain tracks rose slightly in
each window (for example 7.5 -> 8.75); the likely cause is continuing storms keeping their lowered
identity scores instead of being replaced by new tracks, which was not verified case by case.

## 7c. Scans compared by beam index instead of azimuth (fixed 2026-09-13)

**Found:** a survey of every cached back-to-back window (23 windows, 213 scans, 6 radars) found
only 4 reacquisitions among 1,517 storms that went missing. An audit of the rejected candidates
found 140 near candidates; 135 of them had exactly zero raw overlap with the returning echo, even
when stationary.

**Root cause:** masks and reflectivity from two scans were compared cell by cell, but a Level II
volume starts at whatever azimuth the antenna is pointing, so row 0 differs between volumes
(measured 336, 343, 341 and 8 degrees on consecutive KEYX volumes; 177 degrees apart on a KTLX
pair). On a KTLX pair, all 38 matched storms had zero raw overlap. The same misalignment made the
per-storm local motion estimate useless: its median quality was 0.00.

**Owner decision:** align within tracking. **Fix (`091d193`):** `src/tracking/alignment.py`
re-indexes the earlier scan onto the later scan's beam order (nearest azimuth, circular) before
any comparison, in stage 1 and for each scan loaded for reacquisition. The re-indexed azimuth array
keeps each row's true observed azimuth, so geographic conversions stay exact. When beam orders
already match, alignment returns the scan unchanged. Tests use rotated synthetic volumes
(`tests/unit/test_tracking_alignment.py`).

**Verified on real pairs:** raw overlap of matched storms is 0.74, 0.34 and 0.66 (was 0 in all).
Median local motion quality rose from 0.00 to 0.89. Local shifts larger than 20 pixels fell from
10 of 48 to 4 of 48.

**Reacquisition survey rerun** (same 23 windows, 213 scans):

| | Before alignment | After |
|---|---|---|
| Storms that went missing | 1,517 | 1,159 |
| Reacquired | 4 | 40 |
| Lost | 1,041 | 785 |
| Still missing at window end | 472 | 334 |

The 40 reacquisitions (KIWA 12, KTLX 9, KJAX 7, KMLB 6, KEYX 5, KFWS 1), each checked for physical
plausibility:

- Gap: 29 after one missed scan, 7 after two, 4 after three; 7.8 to 18.6 minutes.
- Displacement: 0.08 to 6.17 km (median 1.9). Implied speed is 0.5 to 26 km/h, all slow and
  physically reasonable.
- Advected overlap: 0.16 to 0.77 (median 0.27). Raw overlap was zero in 6 cases, all 2.5 to 4.9 km
  displacements over 10 to 19 minutes.
- Size and strength: weak and small (peak 22.5 to 57.5 dBZ, median 29.5; area 4 to 39 km²),
  mostly at long range (median 236 km), where echoes near the detection threshold drop out for a
  scan. Area after/before ratio is 0.30 to 2.74; peak change is -5.0 to +7.5 dBZ.
- Competition: one case had a competing track. At KMLB 03:10Z, track 80 (cost 0.332) won over
  track 87 (0.357); the lower cost won.
- Identity after reacquisition is 0.08 to 0.23, below the 0.4 cap, so these storms are reported
  with low identity confidence.

**Rejection audit rerun:** 41 candidates were eligible and 40 were reacquired. The one eligible
candidate not reacquired was track 87 above, which lost to a lower-cost track.

Rejections within 5 km for low advected overlap fell from 140 to 58:

- Before alignment, 139 of the 140 had zero raw overlap, including stationary echoes.
- Now 28 of the 58 have zero raw overlap, and every one of those is displaced 2.4 to 5.0 km, which
  is plausible for small echoes.
- The highest advected overlap among the 58 is 0.13, against the 0.15 threshold; only 4 exceed
  0.10.

No remaining rejection pattern points to a comparison artifact.

**Benchmark effect** (`2026-09-13-benchmark-after-alignment.json` against
`2026-09-13-benchmark-after-identity-fix.json`):

| Window | Merges | Splits | New tracks | Lost | Max speed mph | Mean uncertain | Fragmentation |
|---|---|---|---|---|---|---|---|
| dense_cached_quick | 8 -> 13 | 7 -> 7 | 27 -> 25 | - | - | 7.67 -> 6.0 | 0.181 -> 0.168 |
| dense_cached_extended | 29 -> 40 | 24 -> 22 | 100 -> 91 | 47 -> 36 | 87 -> 32 | 8.75 -> 6.62 | 0.239 -> 0.217 |
| lower_complexity_extended | unchanged | unchanged | unchanged | - | - | 0.17 -> 0.0 | unchanged |
| merge_split_regression | unchanged | unchanged | unchanged | - | 35 -> 27 | 1.2 -> 1.0 | unchanged |
| merge_split_extended | unchanged | unchanged | unchanged | - | 35 -> 29 | 1.71 -> 1.29 | unchanged |

- Fewer new, lost and uncertain tracks and lower maximum speeds follow from working local motion
  guidance and real overlap.
- The 87 mph outlier is gone.
- Merges rose in both dense windows. The likely cause is that overlaps between merging storms are
  now measured instead of reading zero. This was not verified case by case.
- `dense_cached_extended` mean focus selection margin fell from 2.02 to 0.79. This was not
  investigated.
- `merge_split_regression` gained one heading-reversal scan: a 82 to 90 degree change crossing the
  metric's threshold, with unchanged speech.

`dense_cached_quick` speech changed:

- **Before:** the strongest storm was a different track each scan (9, 8, 60), with "moving SSE at
  19 mph", then "moving S at 7 mph".
- **After:** the strongest storm stays track 9 at 96, 97 and 94 miles NNE, and its motion is
  reported as "tracking uncertain" twice.

Instrumenting that window showed that neither reported motion came from the storm. Both came from
the scene-wide motion estimate, identical for every track: 183 degrees at 2.01 km, then 325
degrees at 1.01 km. Its raw phase-correlation shifts were exactly 0 rays and -8 gates, then 0 rays
and -4 gates. The motion guard withheld the resulting S-to-NW flip. That estimate is defective, as
described next.

**New defect found (not fixed, predates this branch):** `estimate_scan_geographic_motion_field`
has two problems:

- It correlates the polar grid downsampled by 4, so shifts are quantized to 1 km in range and
  2 degrees in azimuth (about 5 km tangentially at 150 km).
- It turns the shift into a direction at a weighted centroid of *row indices*. That linear average
  of a circular quantity depends on where the volume's beam 0 lies, not on where the echoes are.

A purely radial 1 to 2 km shift was therefore reported toward whatever bearing the index average
fell on. Reported motion falls back to this field whenever a track's own history motion is not
publishable, or disagrees with the field. The blend with the storm's local estimate discards the
local estimate whenever it differs from the scene estimate by more than 0.03 degrees, as both local
estimates did here. Spoken motion can therefore state a wrong direction.

## 8. Known limits

- Scene-wide motion estimate is quantized and its heading depends on beam order (section 7c). Not
  yet fixed.
- Peak memory while processing and rendering is 546 MB (measured; section 3), led by the 356 MB
  Level II parse. Detection's multi-gigabyte peak is fixed.
- The intensity-band layer took 123.5 s (KTLX, 53 storms) and 327.1 s (KEMX, 81 storms) because it
  contoured the whole field once per storm. Fixed 2026-09-13 with byte-identical output: 8.0 s and
  14.0 s, now dominated by joint coverage simplification per level set.
- Rendered layers dominate retained memory on busy days (section 3).
- `Track.positions` is uncapped, so motion fits span a track's lifetime (issue #1).
- Lost and merged tracks accumulate in tracker state for the life of a site's tracker, so
  `tracker_state.json` grows during a long session.
