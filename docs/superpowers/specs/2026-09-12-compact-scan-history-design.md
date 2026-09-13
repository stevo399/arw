# Compact Scan History Design

## Problem

ARW keeps storm continuity by holding whole processed scans in memory. On a
convective day two held scans approach a gigabyte, which is why the replay
buffer, processed-scan cache and live-scan cache were all cut to two scans
(uncommitted `src/buffer.py` / `src/server.py` work as of 2026-09-12). That
bound makes the memory tolerable but leaves ARW unable to:

- reacquire a storm that drops out for a scan or two,
- let a user step back through recent scans on any of the four map layers,
- compute multi-scan growth and decay trends, or
- keep a radar's continuity when the user switches to another radar and back,
  or when the server restarts.

The goal is a per-radar history of the five most recent scans that preserves
everything the four map layers and the tracker read, at a small fraction of
the current size.

## Measurements and findings

All measured on real cached Level II volumes on 2026-09-12.

### Where the memory goes

KTLX 2026-04-10 22:42:09Z, 63 storm objects, reflectivity grid 720 x 1832:

| Held per scan today | Size |
|---|---|
| `object_masks`: one full-grid boolean array per object | 83.1 MB |
| Reflectivity field, float64 | 10.6 MB |
| `labeled_grid`, int32 | 5.3 MB |

The 83 MB of masks mark 48,016 gates in total. Mask memory scales with object
count, so a 300-object scene is roughly 400 MB per scan.

### The masks are redundant

On KTLX 2026-04-10 22:42:09Z (63 objects) and KEMX 2026-07-12 02:26:46Z (81
objects), no gate belongs to two objects, and `labeled_grid == object_id`
reproduces every mask exactly. A single label grid carries all mask
information.

### The fields store losslessly in their Level II encoding

| Field | Observed quantization | Storage |
|---|---|---|
| Reflectivity | every gate on the 0.5 dBZ step | 1 byte |
| RhoHV | 254 distinct values, 1/300 step | 1 byte |
| ZDR | 1,048 distinct values, 1/32 dB step | 2 bytes |
| QC gate class | int8 codes | 1 byte |

Compressed (zlib level 6) sizes for the KTLX volume:

| Stored field | Every gate | Gates >= 15 dBZ only |
|---|---|---|
| Reflectivity | 250 KB | 74 KB |
| RhoHV | 350 KB | 87 KB |
| ZDR | 442 KB | 120 KB |
| QC gate class | 86 KB | 86 KB |
| Object label grid (uint16) | 11 KB | 11 KB |
| **Total** | **~1.1 MB** | **~0.4 MB** |

**Reflectivity below 15 dBZ must be kept.** Detection and motion estimation
threshold at 20 dBZ, and the lowest precipitation level is 15 dBZ, but contour
geometry does not depend only on gates above a level. `src/contours.py` uses
contourpy, which places each boundary by interpolating between neighbouring
gate values, and fills no-data gates from their neighbours. A 12 dBZ gate beside
an 18 dBZ gate puts the 15 dBZ edge between them. Replacing the 12 with no-data
moves that edge. The same holds at the storm layer's 20 dBZ edge. Reflectivity is
therefore stored at every gate.

RhoHV, ZDR and gate class are read only for gates inside a band
(`_band_gate_mask`, `lower >= 15`), so RhoHV and ZDR are stored only where
reflectivity is at least 15 dBZ. Gate class is stored at every gate, since it
measured the same size either way.

Chosen layout: reflectivity 250 KB + RhoHV 87 KB + ZDR 120 KB + gate class
86 KB + label grid 11 KB, about **0.55 MB per scan**, or about 2.8 MB for a
five-scan radar history.

### Defect: precipitation-layer evidence is stripped

The uncommitted `_process_scan_file` change sets `rhohv`, `zdr` and
`gate_classification` to `None` before returning. The precipitation-field
layer is built later, from that stripped scan, and `_band_evidence` reads those
fields. On the KTLX volume, the first three bands built with the fields report
median RhoHV 0.9883-0.9917 and class `precipitation`. Built from the stripped
scan, the same bands report RhoHV `None` and class `uncertain`, with no
indication that evidence is missing. This design restores the fields. The
defect must not be committed on its own in the meantime.

### Defect: a storm that misses one scan can never be re-matched

`associate_tracks` finds a track's previous mask only through `obj_to_track`,
which holds the tracks matched in the latest scan. A track that misses a scan is
absent from it, is skipped at `src/tracking/association.py:187`, and becomes
`lost` after `MAX_MISSED_SCANS = 2` (`src/tracker.py:663`). The missed-scan
allowance can never produce a match.

### Defect: browsing history resets live tracking

Every request path runs `_process_scan_file`, which calls `tracker.update` on
the site's single live tracker (`src/server.py:331-333`). A request for an
older scan, such as Weather Kitten's specific-time mode or ARW's own "Previous
hour" button, produces a negative time gap and resets the tracker
(`src/tracker.py:516`), discarding every live track for that radar.

## Design

### 1. Data model and storage

**`CompactScan`** (`src/history/compact_scan.py`) is immutable, one per
processed volume:

- Identity: site ID, scan timestamp, source Level II filename, schema version.
- Geometry needed to rebuild gate coordinates exactly: azimuths, ranges,
  per-ray elevations, radar latitude/longitude/altitude, nominal elevation
  angle and elevation-angle list.
- Coded fields, each zlib-compressed: reflectivity at every gate (uint8,
  0.5 dBZ, with a distinct no-data code), QC gate class at every gate (uint8),
  and RhoHV (uint8) and ZDR (uint16) at gates with reflectivity >= 15 dBZ, with
  the no-data code elsewhere. Rebuilt RhoHV/ZDR below 15 dBZ are NaN. Nothing
  reads them there today, and the round-trip test (section 4) pins that.
- One object label grid (uint16, zlib-compressed) replacing `object_masks`.
- Detected objects, scan quality, and rotation signatures as plain records.
- `to_sweep()` rebuilds a `SweepData`, and `to_buffered_scan()` rebuilds a
  `BufferedScan` with masks derived from the label grid. Map-layer builders,
  detection consumers and the tracker keep their current input types.

**Encoding is verified, never assumed.** When packing, each field is checked
against its expected quantization step. If any finite value is off-step, that
field is stored as compressed float32 (or float64 if float32 does not
round-trip), and the fallback is logged with site, timestamp and field. A
change in a future Level II build can cost size, never accuracy. The same
applies if the object count exceeds the uint16 label range.

**`RadarHistory`** (`src/history/store.py`), one per radar:

- A ring of the five newest `CompactScan`s on the live path (section 3).
- The radar's `StormTracker`.
- It replaces `ReplayBuffer`, `_processed_scans`, `_live_scans` and the
  tracker's retained `_prev_scan` BufferedScan. Full-size arrays exist only
  transiently, for the scan pair being associated or the layer being rendered.
- A global in-memory cap on radars (least recently used, default 20, set by
  environment variable). An evicted radar's ring and tracker state reload from
  disk on next use.

**Rendered map layers** stay cached by scan identity, but separately bounded:
the newest scan per in-memory radar plus a global LRU of other scans (default
10). An evicted rendering is rebuilt from its snapshot on demand, and the
result is identical.

**On disk:**

- `cache/<site>/history/<YYYYMMDD_HHMMSS>.arwscan` (one `.npz` container per
  scan), pruned to the ring.
- `cache/<site>/history/tracker_state.json`, versioned.
- Every write goes to a temporary file in the same directory, then an atomic
  rename.

### 2. Tracker changes

**Reacquisition.**

- Each track records `last_seen_ref = (scan timestamp, object id)`, updated
  whenever it is matched. Association uses it instead of `obj_to_track` to find
  a track's last mask.
- Matching runs in two stages:
  1. Tracks seen in the previous scan match exactly as today: same costs, same
     global assignment, same thresholds.
  2. Active tracks with `_missed_scans` of 1 or more then compete only for
     objects left unclaimed. Each one's mask is rebuilt from the snapshot named
     by `last_seen_ref` and shifted by the current scene pixel motion scaled
     from the adjacent-scan interval to the full elapsed time since last seen.
- A stage-2 match must pass `MIN_ADVECTED_OVERLAP_PCT` on the shifted mask and
  `MAX_STORM_SPEED_KMH` over the full elapsed time.
- The elapsed time since last seen must be at most
  `MAX_TEMPORAL_CONTINUITY_MINUTES` (20). This preserves the tracker's rule that
  a long outage cannot make an unrelated echo appear persistent.
- Tracks with status `lost` are never reacquired. `MAX_MISSED_SCANS` is
  unchanged.
- A reacquisition emits a `reacquired` event. Identity confidence is lowered
  in proportion to the scans missed, and identity diagnostics carry
  `event_context="reacquired"` with a reason naming the missed-scan count, so
  speech can express uncertainty instead of stating identity as fact.

**Trends.**

- Each track keeps a per-scan sample of total area, area per intensity band,
  peak dBZ and scan timestamp, capped at the last 6 entries like
  `rotation_history` and `motion_history`.
- A pure function reports total-area and >= 50 dBZ core-area trend (growing,
  steady or decaying) with a confidence label. It returns `insufficient` when
  there are fewer than 3 samples, or when any sample in the window came from a
  reacquired match.

**Deliberately unchanged:** association costs for stage 1, the motion fit,
motion publishability guards, and focus selection. The existing replay
benchmarks must therefore hold unchanged for continuous tracks.

**Out of scope, tracked separately:** `Track.positions` is never capped, so the
motion fit spans a track's whole lifetime. Changing that alters published
motion and needs its own validation. See
https://github.com/stevo399/arw/issues/1.

### 3. Live path, historical path, restart, errors

**Two separate paths.**

- **Live path:** the volume returned for a latest-scan request or by the live
  refresh worker. If its timestamp is newer than the ring's newest, it is
  processed, fed to the radar's tracker, packed and added to the ring. Volumes
  the refresh skipped are not backfilled, which is today's behavior.
- **Historical path:** a `datetime=` request. If the timestamp exactly matches
  a scan in the ring, it is served from the ring with tracking context.
  Otherwise the volume is processed and packed but **never touches the live
  tracker**. It is held in a separate LRU (default 12 scans), and responses
  state that tracking context is unavailable.

**ARW API.**

- `GET /map/history` accepts the same location parameters as
  `/map/storms.geojson` (city/state, zipcode, or latitude/longitude). It
  returns the selected site and its retained scans, newest first. Each entry
  has exact timestamp, object count, and `tracked`.
- `/map/status` reports a ring scan requested by exact timestamp as ready
  immediately.
- `/map/storms.geojson`, `/map/precipitation.geojson`, `/objects`, `/tracks`
  and `/summary` serve ring scans through their existing `datetime=` parameter.
- `/tracks` and `/summary` add `continuity_rebuilt` (section 3, restart) and
  `tracking_context` (`tracked` or `unavailable`).
- ARW's own `/radar` page is not changed.

**Weather Kitten** (`Documents/weatherKitten`, separate repository):

- While a map is shown, the radar panel adds a "Recent scans" heading, followed
  by:
  - **Previous scan** and **Next scan** links. Each keeps the selected layer,
    sets `radar_time_mode=timestamp` and pins the exact scan timestamp. Where a
    link does not apply, plain text replaces it ("This is the oldest retained
    scan" / "This is the newest retained scan").
  - A list of retained scans (up to 5), each a link such as "22:37 UTC, 14
    storms, tracked". The shown scan carries `aria-current="true"`.
- Links reload the page to `#radar-map`, following the panel's existing form
  pattern, with no new JavaScript and no custom keyboard shortcuts (Alt+Left is
  browser Back).
- The existing polite status region announces the position, for example "Scan
  3 of 5 retained scans, 22:37 UTC, tracked."
- Automatic refresh continues to apply only to latest mode, so a newer scan
  never moves a user who is reviewing an older one.
- If ARW returns no retained scans, the section says so and shows no links.
- Weather Kitten has uncommitted changes (`app.py`, `templates/index.html`,
  `tests/test_perf.py`, `.env.example`) that must be committed or set aside
  before this work starts.

**Restart recovery.**

- On first use of a radar after startup, its ring loads from disk, then
  `tracker_state.json` is loaded.
- Saved tracker state is used only if all hold: the schema version matches, its
  last processed scan timestamp equals the newest snapshot's, and (when the next
  live volume arrives) the gap is within `MAX_TEMPORAL_CONTINUITY_MINUTES`.
- If version or timestamp checks fail, the ring's snapshots are replayed
  through a fresh tracker, and `continuity_rebuilt` is `true` until the next
  live scan. Speech mentions it once.
- If the gap to the next live volume is too long, the tracker resets as it does
  today.

**Errors.**

- An unreadable or corrupt snapshot is logged, deleted and skipped. The rest of
  the ring stays usable.
- An unreadable `tracker_state.json` is treated as a failed check (replay).
- A failed disk write keeps the in-memory ring and tracker, and the error is
  reported through `/live/{site}/status`. Live tracking continues.
- Encoding fallbacks are covered in section 1.

### 4. Testing

Tests use real cached Level II volumes unless marked synthetic.

**Unit (ARW)**

- Lossless round trip on KTLX, KEMX and KIWA: reflectivity (including NaN
  positions) and gate class are identical at every gate; RhoHV and ZDR are
  identical at every gate >= 15 dBZ. Every derived mask equals the original, and
  geometry is identical.
- Consumer guard: a test asserts no map-layer builder reads RhoHV or ZDR
  outside `_band_gate_mask` gates, by building all four layers from a sweep
  whose sub-15 dBZ RhoHV/ZDR are replaced with extreme values and checking the
  output is unchanged.
- Encoding guard (synthetic): one off-step value triggers the full-precision
  fallback and a log entry, and never rounds.
- Store: a sixth scan evicts the oldest. Radars beyond the in-memory cap are
  evicted LRU and reload from disk. An interrupted write leaves no partial
  file. A corrupt snapshot is skipped and deleted.
- Tracker state: save and load round trip. Each rejection path falls back to
  replay and sets `continuity_rebuilt`.
- Reacquisition (synthetic):
  - a storm hidden for one scan returns with its ID and lowered confidence;
  - a different storm appearing nearby is not matched;
  - nothing is matched after 20 minutes or once `lost`;
  - a continuous track always keeps its match over a returning one.
- Trends: growing, steady, decaying, and `insufficient` for fewer than 3
  samples or a reacquired sample.

**Proof on real data**

1. **Identical output.** Replay the existing benchmark manifests through the
   old and new pipelines. All map layers' GeoJSON must be identical, including
   precipitation evidence. For continuous tracks, association results, motion
   and focus must be identical. Only reacquisitions may differ, and each is
   listed in the report for review.
2. **Precipitation evidence present.** Every precipitation band carries
   RhoHV, ZDR and class fractions when the source volume has them.
3. **Memory bound.** Process the dense KTLX window live-style, five scans each
   for two radars, with all four layers requested for every scan. Measure and
   record separately:
   - compact history: must be at most 10 scans x the largest single measured
     compact scan in the window, plus 10%;
   - rendered-layer cache: must be at most (newest scan per radar + the LRU
     limit) x the largest measured rendered scan, plus 10%;
   - no full-size `SweepData`, label grid or mask array retained once requests
     finish (checked by reference inspection, not only by process memory).
   Also record the old pipeline's retained memory on the same window for
   comparison. Limits come from the measurement, not a guess. If measured
   rendered layers are too large for the default LRU, lower the default and
   record why.
4. **Restart.** Process 3 scans, restart, process 2 more. Track IDs and ages
   continue.
5. **Browsing does not break tracking.** A request for a scan an hour old,
   made mid-window, leaves live track IDs and histories unchanged.

**Smoke and end to end**

- Server smoke test for `/map/history`.
- Live end-to-end run against a running ARW that hits `/summary` and
  `/map/history` with real NEXRAD data.
- Weather Kitten Flask tests with ARW's `/map/history` stubbed: link
  parameters keep the layer and pin the exact timestamp; edge text replaces a
  missing link; `aria-current` marks the shown scan; empty history shows the
  explanation.
- Weather Kitten live run against a running ARW stepping through every retained
  scan on each of the four layers.

Results go in `docs/test_reports/`, including the reacquisition list and
memory measurements.

## Out of scope

- Windowing or capping the motion fit (issue #1).
- Backfilling volumes the live refresh skipped.
- Tracking context for historical scans outside the ring.
- Changes to ARW's own `/radar` page.

## Amendment 1 (2026-09-12, during implementation planning)

Read this section before the rest of the spec. Where it conflicts with an
earlier section, it overrides it.

### A1. Dual-pol evidence is precomputed, not stored (owner decision)

`_band_evidence(sweep, lower, upper)` summarizes every gate in one band across
the whole sweep. The bands are fixed (`precipitation_band_keys()`: 15-20,
20-30, 30-40, 40-50, 50-60, 60+), and after processing nothing else reads
RhoHV, ZDR or per-gate class. The evidence for all six bands is therefore
computed once, during processing, while the fields still exist, and stored as
`precipitation_band_evidence`. The precipitation layer uses it when present.

Verified on KTLX 2026-04-10 22:42:09Z, KEMX 2026-07-12 02:26:46Z and KIWA
2026-07-12 17:00:29Z: every emitted band has median RhoHV, median ZDR and class
fractions (KIWA's clear-air bands correctly remain `uncertain`).

Consequences:

- **Defect 1** is fixed without retaining any grid (Plan 1).
- `CompactScan` no longer stores RhoHV, ZDR or gate class. It stores
  reflectivity at every gate, the label grid, records, and band evidence.
  Reflectivity plus labels measured 0.26 MB on KTLX. Records add the objects
  and that scan's tracking snapshot, which are not yet measured on a busy
  scene; the Plan 2 proofs measure the full size and set the memory limits
  from it.
- The consumer-guard test in section 4 is replaced by a test that the layer
  built from precomputed evidence equals the layer built from the full fields.
- Raw dual-pol remains available by re-parsing the Level II file in `cache/`.

### A2. A scan carries the tracking state from its own time

Serving an older scan with the live tracker's current tracks would describe
storms at the wrong time. So each tracked scan stores a `TrackingSnapshot`
(copies of active tracks plus that scan's events) taken immediately after it
was tracked. `/tracks`, `/summary` and `/objects` serve the requested scan's own
snapshot and never read the live tracker. A scan with no snapshot (the
historical path) reports `tracking_context: "unavailable"` with no tracks.

### A3. Defect 4: speech attached a storm to another storm's track

`src/summary.py:25` and `:37` find a storm's track by matching
`track.current_object.object_id`. A track that missed the latest scan stays
`active` with its previous scan's object, and object numbers are reassigned
every scan. Verified with a synthetic two-scan case: storm B (lon -96.0) was
attached to storm A's track (lon -98.5) because A was not detected and B took
A's old object number. The summary would speak B with A's motion and history.
Fix: only tracks seen in the scan being summarized (`_missed_scans == 0`) can
be matched (Plan 1).

### A4. The tracker loads its previous scan instead of retaining it

`StormTracker(scan_loader=None, reacquire=False)`. With no loader, behavior is
exactly as today. With a loader, the tracker keeps only
`(site_id, timestamp)` of its previous scan after `release_previous_scan()`,
and loads the full scan through the loader at the next update. Reacquisition
masks come from the same loader.

### A5. Reacquisition covers a storm absent for exactly one scan

With `MAX_MISSED_SCANS = 2` unchanged, a track becomes `lost` in the scan of
its second miss, before it could be reacquired. Reacquisition therefore
applies to a storm missing from exactly one scan and back in the next.
Allowing two missed scans means changing `MAX_MISSED_SCANS`, which changes
which tracks are active for focus and speech, and needs its own benchmark
validation. It is not part of this work.

### A6. Transient memory during a tracker update is measured, not bounded

Association still builds full-grid masks for both scans of the pair while an
update runs. This design bounds **retained** memory. The memory proof records
the transient peak during an update separately, and does not gate on it.

### A7. Encoding fallback stores the original dtype

If a field fails the exact-step check, it is stored in its original dtype
(zlib-compressed), not float32, so the fallback is lossless by construction.

### A8. Delivery is three plans

1. `docs/superpowers/plans/2026-09-12-radar-correctness-fixes.md`: defects 1,
   3 and 4, independent of the new history.
2. `docs/superpowers/plans/2026-09-12-compact-scan-history.md`: compact
   scans, per-radar history, tracker persistence, reacquisition (defect 2),
   trends, API, proofs.
3. `docs/superpowers/plans/2026-09-12-weather-kitten-scan-stepping.md`:
   Weather Kitten recent-scan navigation.

## Amendment 2 (2026-09-12, owner decision before implementation)

### A9. Track loss follows reacquirability ("missing" status); replaces A5

Product integrity outranks keeping spoken output unchanged. A retained
history of five scans that can only bridge one missed scan is inconsistent,
so track loss now follows whether a storm can still be reacquired:

- A track that is not matched in a scan changes from `active` to **`missing`**.
  A missing track is not active: it is excluded from focus selection,
  summaries, `/tracks` and merge/split handling, so ARW never describes a
  storm that is not in the scan being described.
- A missing track remains a stage-2 reacquisition candidate. Reacquired, it
  returns to `active` with low identity confidence (unchanged from section 2).
- A missing track becomes **`lost`** when it can no longer be reacquired:
  - it has been unseen for more than `MAX_REACQUISITION_MINUTES` (20), or
  - reacquisition is enabled and the scan it was last seen in can no longer
    be loaded (it has left the radar's retained history).
- `MAX_MISSED_SCANS` is removed. Without reacquisition (for example the
  offline benchmark script), only the 20-minute rule applies.

Consequences:

- Focus and speech change wherever the old tracker kept a missed storm
  active. The Task 11 benchmark comparison therefore reviews and explains
  every difference instead of expecting identical output.
- The "identical tracking" proof still compares two trackers running the
  same code (retained scans against loaded scans), so it still expects
  byte-identical snapshots.
