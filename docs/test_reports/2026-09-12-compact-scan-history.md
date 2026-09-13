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
  an observed real-data reacquisition.

| Scan | Objects | Active tracks | Reacquired |
|---|---|---|---|
| 22:42:09 | 63 | 63 | none |
| 22:47:37 | 53 | 50 | none |
| 22:52:55 | 51 | 49 | none |
| 22:58:05 | 49 | 47 | none |
| 23:03:32 | 50 | 48 | none |
| 23:08:50 | 61 | 57 | none |
| 23:14:07 | 57 | 54 | none |

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

## 7b. Defect found during the live check, not fixed (owner decision)

The newest KJAX scan had 29 objects but 24 active tracks; the offline benchmark showed the same
pattern (49 objects, 47 active; 45 objects, 44 active). Instrumenting the benchmark pipeline: every
object maps to exactly one track, but in the same update some tracks are first assigned an object as
a split parent and then marked `merged` into another track (for example object 40 -> track 38,
`merged_into` 12, with object 40 as its current object). The merge step checks only that a track is
`active`, not whether it already holds an object in this scan. Such a storm is present in the scan
but represented only by a merged track: absent from `/tracks`, without motion in speech, and
ineligible for focus. This predates the history work (the old tracker's inflated active count hid
it); fixing it changes merge and split lineage, so it is left for an owner decision.

## 8. Known limits

- Reacquisition has not yet been observed on real data (section 2).
- Transient peak memory during ingest and rendering is large and not bounded (section 3).
- Rendered layers dominate retained memory on busy days (section 3).
- `Track.positions` is uncapped, so motion fits span a track's lifetime (issue #1).
- Lost and merged tracks accumulate in tracker state for the life of a site's tracker, so
  `tracker_state.json` grows during a long session.
