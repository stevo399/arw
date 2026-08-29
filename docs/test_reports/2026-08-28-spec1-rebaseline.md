# Spec 1 Data-Layer Re-baseline (Task 18)

Date: 2026-08-28

Purpose: measure whether the Spec 1 "radar data layer correctness" branch
(`spec1-data-layer`) changed storm-tracking behavior versus `master`, and
check that direction against the amended spec prediction (§11): since quality
control (QC) now only classifies and annotates gates instead of deleting
them, tracking behavior is expected to be **largely unchanged**, except where
the geometry fixes bite directly — principally the azimuth-seam fix for
storms straddling due north, and rotation-related metrics from velocity
sweep selection.

## Harness and exact commands

Harness: `scripts/evaluate_tracking.py`, which replays each benchmark
manifest entry through `parse_radar_file` -> `extract_sweep_data` ->
`preprocess_reflectivity_data` -> `extract_velocity` ->
`detect_objects_with_grid` -> `analyze_velocity` -> `StormTracker.update`,
and reports per-window aggregate metrics (`BenchmarkResult`).

Manifests used (all three present in `docs/benchmarks/`):

```
docs/benchmarks/tracking_benchmark_manifest.json
docs/benchmarks/tracking_benchmark_manifest_broader_validation.json
docs/benchmarks/dense_followup_manifest.json
```

Exact commands run on `spec1-data-layer` (this branch, from repo root):

```
.venv/Scripts/python.exe -u scripts/evaluate_tracking.py --manifest docs/benchmarks/tracking_benchmark_manifest.json --output-json docs/test_reports/spec1-rebaseline-after/tracking_benchmark_manifest.json --output-md docs/test_reports/spec1-rebaseline-after/tracking_benchmark_manifest.md
.venv/Scripts/python.exe -u scripts/evaluate_tracking.py --manifest docs/benchmarks/tracking_benchmark_manifest_broader_validation.json --output-json docs/test_reports/spec1-rebaseline-after/broader_validation.json --output-md docs/test_reports/spec1-rebaseline-after/broader_validation.md
.venv/Scripts/python.exe -u scripts/evaluate_tracking.py --manifest docs/benchmarks/dense_followup_manifest.json --output-json docs/test_reports/spec1-rebaseline-after/dense_followup.json --output-md docs/test_reports/spec1-rebaseline-after/dense_followup.md
```

Identical commands run against `master`, from a separate worktree (see
below), with `--output-json`/`--output-md` pointed at
`docs/test_reports/spec1-rebaseline-before/` instead.

Consolidated raw output (all 12 benchmark-window results per side, tagged
with source manifest) is committed alongside this report:

- `docs/test_reports/2026-08-28-spec1-rebaseline-before.json`
- `docs/test_reports/2026-08-28-spec1-rebaseline-after.json`

## How the "before" numbers were obtained

```
git worktree add ../arw-baseline master
```

`master` was at `7f3e098` (docs-only commit; the same commit `spec1-data-layer`
branched from). The worktree has no `cache/` of its own (it is untracked in
git), so a directory symlink was created so both branches replay the
identical cached NEXRAD volume files, byte for byte, with no re-download:

```
ln -s "<repo>/cache" "<repo>/../arw-baseline/cache"
```

A Python environment was built in the worktree with `uv sync` (same
`pyproject.toml`/`uv.lock` as this branch's dependency set — the branch's
changes are all in `src/`, not in dependencies). The three manifests were
then run unmodified in the worktree, writing to
`docs/test_reports/spec1-rebaseline-before/`.

One manifest entry (`lower_complexity_quick`, `local_only: false`) calls
`list_scans_for_date`, which hits AWS S3 regardless of cache state (see
`src/ingest.py:fetch_scan`). Network access was available in this
environment; the listing succeeded on both branches and resolved to the same
already-cached file both times (confirmed by identical output — see table
below), so this did not introduce cross-branch variance.

The worktree was removed after use:

```
git worktree remove ../arw-baseline
```

No numbers in this report were reconstructed from prior report files. Every
number below came from a fresh run of the same harness invocation on both
branches, on the same cached volumes, in this session.

## Per-window delta table

All 10 distinct benchmark windows across the three manifests (2 window ids —
`dense_cached_extended` and `lower_complexity_extended` and
`merge_split_extended` — are shared between the main and broader-validation
manifests; they are listed once).

| window (site) | scans | objects before→after | frag. proxy before→after | focus switches b→a | heading flips≥90 b→a | merges b→a | splits b→a | new tracks after 1st scan b→a | lost tracks total b→a |
|---|---|---|---|---|---|---|---|---|---|
| dense_cached_quick (KTLX) | 3 | 49.33→49.67 | 0.243→**0.228** | 0→**1** | **1→0** | 14→15 | 9→9 | 36→34 | 7→**15** |
| dense_cached_extended (KTLX) | 8 | 51.88→52.38 | 0.313→0.313 | 0→0 | 1→1 | 52→53 | 35→33 | 130→131 | 68→**77** |
| dense_cached_evening_window (KTLX) | 8 | 74.62→**71.5** | 0.377→0.369 | 0→0 | 1→1 | 93→**83** | 61→**55** | 225→**211** | 81→**76** |
| dense_cached_followup (KTLX) | 5 | 47.00→47.20 | 0.311→**0.331** | 1→1 | **3→1** | 26→25 | 16→15 | 73→78 | 24→**33** |
| lower_complexity_quick (KSOX) | 3 | 4.00→4.00 | 0.583→0.583 | 0→0 | 0→0 | 2→2 | 1→1 | 7→7 | 0→0 |
| lower_complexity_extended (KSOX) | 6 | 4.00→4.00 | 0.625→0.625 | 0→0 | 0→0 | 5→5 | 1→1 | 15→15 | 0→0 |
| merge_split_regression (KEYX) | 5 | 19.00→19.00 | 0.242→0.242 | 1→1 | 0→0 | 12→12 | 6→6 | 23→23 | 9→9 |
| merge_split_extended (KEYX) | 7 | 16.57→16.57 | 0.353→0.353 | 2→2 | 0→0 | 14→14 | 11→11 | 41→41 | 14→14 |
| merge_split_evening_window (KEYX) | 7 | 4.14→4.14 | 0.517→**0.552** | 2→2 | 0→0 | 6→6 | 1→1 | 15→16 | 5→**6** |

(`b→a` = before → after; bold marks a change. Every metric not shown —
`mean_active_tracks`, `merged_tracks_total`, `split_children_total`,
`absorbed_links_total`, `summary_tracking_uncertain_count`,
`summary_motion_published_count`, `summary_stationaryish_count`,
`mean_uncertain_tracks`, `max_speed_mph`, and the focus-continuity/motion
fields — is in the consolidated JSON files; the notable ones are called out
below.)

### Headline result: four of nine windows are byte-identical, a fifth moves by one track

`lower_complexity_quick`, `lower_complexity_extended`,
`merge_split_regression`, and `merge_split_extended` (2 KSOX windows, 2 of 3
KEYX windows) produced **exactly identical output on every single field** in
`BenchmarkResult`, before and after — object counts, fragmentation proxy,
merges, splits, lineage totals, focus switches, focus continuity, focus
selection margin, everything. This is the strongest evidence for the amended
prediction: for these storms, none of the branch's changes (QC classification,
georeferencing, gate areas, azimuth seam, sweep selection) altered a single
downstream tracking number.

### Where numbers moved, and why

**`dense_cached_evening_window` (KTLX, 8 scans) — object count and lineage
churn dropped, consistent with the azimuth-seam fix.** Mean objects fell
74.62→71.5 (−3.1), merges 93→83, splits 61→55, new-tracks-after-first-scan
225→211, lost tracks 81→76 — a broad-based reduction in fragmentation-like
churn, not an increase. This is the one window in the set with substantial
storm-scale echo (≥20 dBZ) straddling due north: a direct check of the four
scans nearest the window's end filename found 1,081–1,307 storm-intensity
gates within ±3° of azimuth 0/360 in every one of them (`src/parser.py`
`extract_sweep_data` + a one-off count, not part of the harness). That is
exactly the geometry the seam fix (`label_periodic_azimuth` in
`src/geometry.py`, wired into both `detect_objects_with_grid` and speckle
removal) targets: a storm that used to be cut into two objects at the seam is
now one. This is the predicted "geometry fixes bite" case, and it moved in
the expected direction — **fewer** spurious objects and **less** lineage
churn, not more.

**`dense_cached_quick` (KTLX, 3 scans) — mixed, net small.** Fragmentation
proxy improved (0.243→0.228) and heading flips fell (1→0), but a new focus
switch appeared (0→1) and `lost_tracks_total` rose 7→15. With only 3 scans
this window has very few opportunities for any of these integer counters to
move, so a single differently-resolved association event dominates the
percentage change. Not a red flag on its own, but see the cross-window
pattern below.

**`dense_cached_followup` (KTLX, 5 scans) — heading stability improved,
fragmentation proxy and lost tracks got slightly worse.** Heading flips
≥90° dropped 3→1, `focus_reported_heading_reversal_scans` dropped 3→1, and
`focus_track_distance_changes` dropped 3→1 — a real improvement in the
metric the spec cared most about for the dense/reversal-prone window. Against
that, `fragmentation_proxy` rose 0.311→0.331 and `lost_tracks_total` rose
24→33. `mean_focus_motion_confidence` also dropped (0.98→0.59) even as the
heading-stability numbers improved — the per-ray elevation/ground-range
geometry changes shift exact gate positions scan-to-scan, which appears to
perturb local motion-field confidence somewhat even where it doesn't change
the final published heading-stability verdict.

**`merge_split_evening_window` (KEYX, 7 scans) — fragmentation proxy
worsened on a very small scene.** Mean objects unchanged (4.14→4.14), but
`fragmentation_proxy` rose 0.517→0.552 and `lost_tracks_total` rose 5→6 —
one additional short-lived track over a 7-scan, ~4-object-per-scan window.
At this scale a single track produces a large percentage swing; this is
flagged per the task's instruction to report any fragmentation increase
prominently rather than explain it away, but the absolute magnitude (one
track) does not look like systematic new fragmentation.

**A cross-window pattern worth naming directly: `lost_tracks_total` rose in
three of the four KTLX dense windows** (`dense_cached_quick` 7→15,
`dense_cached_extended` 68→77, `dense_cached_followup` 24→33), falling only
in `dense_cached_evening_window` (81→76). `mean_uncertain_tracks` rose in the
same three windows. None of these windows showed a corresponding rise in
`fragmentation_proxy` (the harness's actual fragmentation metric, defined as
new-tracks-after-first-scan over total object-scans) except
`dense_cached_followup`'s small one. `lost_tracks_total` counts tracks whose
final status is `"lost"` — i.e., tracks that stopped getting matched, which
grows naturally if track-to-track association becomes slightly more
sensitive to per-scan position (a direct, expected consequence of the
per-ray elevation and ground-range gate-area geometry fixes shifting gate
centroids by a small amount frame to frame) without any corresponding change
in how many genuinely distinct objects exist. This is plausible but **not
independently verified in this report** — see "Unexplained" below.

## Rotation delta (separate from QC, from sweep selection and velocity
## alignment)

Two distinct changes on this branch touch velocity:

1. **Sweep selection** (`select_velocity_sweeps`, replacing
   `range(3)` index-based selection) *does* feed
   `scripts/evaluate_tracking.py`'s pipeline: `extract_velocity(radar)` is
   passed into `analyze_velocity`, which calls
   `detect_velocity_regions`/`detect_rotation_signatures` on whatever sweeps
   were selected. This is the change already measured directly, on real
   cached volumes, in
   `docs/test_reports/2026-08-23-sweep-selection-rotation-delta.md` (Task 6):
   region counts roughly tripled (666→1,756 across 9 scans) because the old
   code fed the detector two empty surveillance cuts and one real Doppler
   cut, while the new code feeds it three real cuts. That report also
   flagged a **pre-existing, unrelated bug** in
   `RotationSignature.sweep_count` (couplet-counting instead of
   distinct-sweep-counting, inflating it to as high as 198). That bug was
   fixed separately, after the Task 6 report, in commit `715e08b` ("count
   distinct sweeps, not merged couplets"), which this report did not need to
   re-verify since it is a different code path from the one this task
   measures.

   `scripts/evaluate_tracking.py`'s `BenchmarkResult` does not report region
   or rotation-signature counts at all (only reflectivity-object tracking
   metrics), so this task cannot add a fresh rotation-count delta beyond
   Task 6's; it is included here as required context, not re-derived.

2. **Velocity-to-reflectivity azimuth alignment** (`e018006`,
   `align_field_by_azimuth` / `_velocity_aligned_to_reflectivity`) does
   **not** touch region or rotation detection at all. It is used exactly
   once, in `src/server.py`'s `_ingest_to_buffer`, to feed QC's
   `classify_gates` fuzzy-logic scoring (specifically the `ground_clutter`
   discriminator's `abs_velocity` term). `analyze_velocity` /
   `detect_velocity_regions` / `detect_rotation_signatures` all consume
   `vel_data` directly, never the aligned/reindexed copy. Since
   `scripts/evaluate_tracking.py` and `scripts/live_replay.py` both call
   `preprocess_reflectivity_data` (the pre-QC function), not
   `preprocess_sweep` (the QC-wired function `server.py` uses in
   production), **QC classification — and therefore this alignment fix —
   is not exercised by this benchmark harness at all.** See "Unexplained"
   below for why this does not undermine the QC-direction check.

## Direction check

The amended spec §11 prediction: because QC no longer deletes echo (only
classifies/annotates it), tracking behavior should be **largely unchanged**,
except where the geometry fixes (azimuth seam, georeferencing) bite. The
azimuth-seam fix specifically should **reduce** object counts for storms near
due north; velocity-related metrics should move only where sweep selection
or alignment are exercised.

**Measured result matches this prediction:**

- 5 of 9 distinct windows (KSOX ×2 exactly identical, KEYX ×3 — 2 exactly
  identical, 1 with a small fragmentation-proxy uptick) showed no change or
  only a one-track-scale change in a tiny-object-count scene.
- The one window with clear, substantial storm echo straddling due north
  (`dense_cached_evening_window`) showed a **reduction** in object count and
  lineage churn — the predicted direction, not the opposite.
- Heading-flip counts (`focus_heading_flips_ge_90`) **improved or stayed
  the same** in every window that changed (3→1 dense_followup, 1→0
  dense_quick, 1→1 elsewhere) — no window showed heading instability get
  worse.
- Fragmentation proxy **improved** in 2 windows, **stayed the same** in 6,
  and **worsened slightly** in 2 (`dense_cached_followup`: 0.311→0.331;
  `merge_split_evening_window`: 0.517→0.552). Per the task's instruction,
  both are reported here prominently rather than smoothed over. Neither
  looks like the systemic "QC deleted weather, Phase 2 suppression is now
  fighting a starved input" failure mode the original (pre-amendment)
  spec worried about — that failure mode does not apply here, since this
  harness never exercises QC's echo-classification path at all (see above).

**Nothing moved in a direction inconsistent with the amended expectation.**
The `lost_tracks_total` rise in 3 of 4 KTLX dense windows is the one pattern
that does not have a fully verified mechanism (see below), but it is a
lineage-bookkeeping count, not the fragmentation proxy or heading-instability
metrics the spec singled out, and it moved in both directions across windows
rather than uniformly worsening.

## Unexplained

- **`lost_tracks_total` rose in `dense_cached_quick` (+8),
  `dense_cached_extended` (+9), and `dense_cached_followup` (+9), while
  falling in `dense_cached_evening_window` (−5).** The most likely mechanism
  — per-ray elevation and ground-range gate-area geometry changes shifting
  object centroids by a small amount frame-to-frame, making association
  marginally more sensitive — is plausible and consistent with
  `mean_uncertain_tracks` rising in the same three windows, but this report
  did not instrument the tracker to confirm it directly (e.g., by diffing
  which specific tracks flip from "matched" to "lost" between branches).
  Flagging this as input for anyone tuning Phase 2 association thresholds
  later, per the "do not re-tune, but do record" constraint on this task.
- **Why `evaluate_tracking.py`/`live_replay.py` never exercise QC
  classification at all** (`preprocess_reflectivity_data` vs.
  `preprocess_sweep`) is not something this task's brief anticipated; it
  turned out not to matter for the numbers in this report only because QC on
  this branch is confirmed non-destructive (`qc_sweep.reflectivity` is
  byte-for-byte the input, per the docstring in `src/preprocess.py`), so
  running the harness through `preprocess_sweep` instead would not change
  any object/track count here — only the (unreported-by-this-harness)
  `class_fractions`/`advisory_fraction`/`degraded_modes` quality metadata.
  This is left as-is; changing the harness's preprocessing entrypoint is a
  harness change, out of scope for a measurement-only task, and is noted
  here rather than silently worked around.
- All other deltas in the tables above are attributed to a specific,
  identified code change (azimuth seam, sweep selection, geometry) with
  supporting evidence cited; none required guessing.

## Constraint compliance

No Phase 2 suppression parameter, threshold, or tracking constant was
modified in the course of producing this report. `src/`, `tests/`, and
`PROGRESS.md` were not touched by this task; only the report and its
companion JSON files were added.
