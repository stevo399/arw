# Tracking Best-Practice Roadmap

Date: 2026-04-11

## Purpose

This roadmap turns the source-backed tracking recommendations into explicit engineering tasks. It is written to improve the generic tracker architecture, not to create radar-specific exceptions or site-specific tuning rules.

The target direction remains:

- keep the staged pipeline
- upgrade weak stages instead of collapsing the design
- validate against live radar data, not only synthetic tests

## Non-Negotiable Task Close-Out Rule

When any task below is completed, do all of the following before marking it done:

1. run relevant smoke tests
2. run live data fetches, API rendering checks, and output analysis when appropriate to the task
3. commit if green
4. mark the task as done in this document

If a task changes no runtime behavior, note that explicitly and still run the minimum relevant verification.

## Guiding Constraints

- no hardcoded logic for individual radar sites
- no per-site exception tables as a substitute for generic tracking behavior
- thresholds and confidence policies must remain tunable and documented
- product-layer focus logic must remain separate from scientific identity continuity
- each task should leave the architecture more replaceable, not less

## Task 1: Add explicit preprocessing and scan-quality metadata

Status: completed

Goal:

- create a real stage before detection that handles basic reflectivity QC and records scan-quality metadata for downstream confidence penalties

Why:

- source-backed best practice favors input conditioning before identification and tracking
- this reduces pressure to patch clutter-like behavior inside later tracking stages

Scope:

- introduce a preprocessing module
- add scan-quality flags or a scan-quality score to the buffered scan or tracking inputs
- wire quality metadata into downstream diagnostics without breaking existing public endpoints

Deliverables:

- preprocessing stage in code
- documented quality fields and intended downstream use
- unit tests for QC behavior and metadata propagation

Validation:

- smoke tests for server startup and summary/tracks/motion endpoints
- live fetch or replay validation showing no regression in simple and dense scenes
- output analysis noting whether low-quality scans now degrade confidence instead of producing brittle object structure

Completion notes:

- added `src/preprocess.py` with a conservative preprocessing stage that removes tiny weak speckle and emits scan-quality metadata
- threaded `scan_quality` through `BufferedScan`, ingest, replay tooling, and tracker confidence handling
- kept the public API contract unchanged for this task
- targeted verification:
  - `uv run pytest tests/unit/test_preprocess.py tests/unit/test_buffer.py tests/unit/test_tracker.py tests/unit/test_live_replay_contracts.py tests/smoke/test_server_smoke.py -q`
  - `35 passed in 2.98s`
- live replay validation:
  - `uv run python scripts/live_replay.py KSOX --date 2026-04-10 --quick`
  - `uv run python scripts/live_replay.py KTLX --date 2026-04-10 --quick --local-only`
- observed behavior:
  - no summary or merge/split regressions in the checked windows
  - replay output now includes scan-quality score and flags for debugging
  - current quality scoring is conservative because masked sweep area is still counted heavily; this should be refined later without removing the new preprocessing stage

## Task 2: Replace heuristic core splitting with multilevel segmentation hierarchy

Status: completed

Goal:

- evolve from the current parent-blob-plus-core-split heuristic toward a fuller multilevel segmentation model

Why:

- the current segmentation is directionally right but still simpler than stronger multithreshold practice
- better hierarchy should reduce both false merges and false fragment births

Scope:

- preserve current object masks and endpoint contracts where practical
- support a threshold ladder and parent-child object structure internally
- keep the segmentation conservative enough that simpler scenes do not explode into fragments

Deliverables:

- internal multilevel segmentation data model
- revised segmentation logic
- tests covering crowded scenes, simple scenes, and continuity across scans

Validation:

- smoke tests
- live replay runs over a dense scene, a lower-complexity scene, and a known split/merge-sensitive window
- output analysis focused on object counts, merge/split plausibility, and continuity stability

Completion notes:

- replaced the old two-threshold seed scan with a multilevel threshold hierarchy inside detection
- hierarchy nodes are now built across a threshold ladder and attached to detected objects for downstream segmentation metadata
- split selection now comes from persistent high-threshold branches rather than a one-off seed heuristic
- kept endpoint-visible object contracts stable while enriching internal segmentation metadata
- targeted verification:
  - `uv run pytest tests/unit/test_detection.py tests/unit/test_tracking_segmentation.py tests/unit/test_tracking_association.py tests/unit/test_tracker.py tests/unit/test_summary.py tests/e2e/test_full_pipeline.py tests/smoke/test_server_smoke.py -q`
  - `62 passed in 3.47s`
- live replay validation:
  - `uv run python scripts/live_replay.py KSOX --date 2026-04-10 --quick`
  - `uv run python scripts/live_replay.py KTLX --date 2026-04-10 --quick --local-only`
- observed behavior:
  - lower-complexity replay remained stable at `3 / 2 / 7` objects with plausible motion and event counts
  - dense replay counts dropped from the prior `53 / 64 / 65` window to `48 / 50 / 50`, indicating less fragmentation under the hierarchy-based split logic
  - dense-scene focus and spoken motion remained stable while merge/split counts became less noisy

## Task 3: Strengthen association with advected geometry features

Status: completed

Goal:

- make geometry continuity a stronger part of association than it is today

Why:

- the tracker already preserves masks, so it should use that geometry more fully
- this should improve continuity decisions in crowded or irregular scenes

Scope:

- add advected overlap or IoU-like scoring after applying the motion prior
- keep the global assignment interface stable
- maintain interpretability of candidate scores and diagnostics

Deliverables:

- expanded cost model with geometry-aware features
- diagnostics explaining why a match won
- unit tests for ambiguous geometry cases

Validation:

- smoke tests
- live replay checks in crowded scenes where centroid-only continuity is weak
- API rendering and output analysis verifying fewer false continuity jumps and fewer misleading structural events

Completion notes:

- added motion-advected mask IoU support to the association layer so continuity scoring can use predicted geometry, not just raw overlap and centroid distance
- extended `AssociationScore` to carry `advected_overlap_score` for downstream diagnostics
- kept the global assignment interface and public API contracts stable
- targeted verification:
  - `uv run pytest tests/unit/test_tracking_association.py tests/unit/test_tracker.py tests/unit/test_tracking_types.py tests/unit/test_motion.py tests/unit/test_summary.py tests/e2e/test_full_pipeline.py tests/smoke/test_server_smoke.py -q`
  - `62 passed in 3.37s`
- live replay validation:
  - `uv run python scripts/live_replay.py KTLX --date 2026-04-10 --quick --local-only`
  - `uv run python scripts/live_replay.py KSOX --date 2026-04-10 --quick`
- observed behavior:
  - no summary, merge/split, or motion regressions in the checked windows
  - lower-complexity replay remained stable at `3 / 2 / 7` objects
  - dense replay remained stable at `48 / 50 / 50` objects with the same focal storm and plausible spoken motion
  - the main gain from this task is stronger geometry-aware scoring in ambiguous cases, reflected in new unit coverage and diagnostics rather than a dramatic replay-output shift in these short windows

## Task 4: Move from event bookkeeping toward lineage state

Status: completed

Goal:

- evolve merge/split handling from latest-scan event normalization into richer lineage state over time

Why:

- one-scan event cleanup solved the immediate corruption bugs, but long-term tracking quality needs a clearer parent/child/absorbed model

Scope:

- maintain parent, child, and absorbed relationships in track state
- derive current event payloads from lineage state instead of treating them as the primary truth
- keep public API behavior stable unless correctness requires a change

Deliverables:

- lineage-oriented internal model
- compatibility layer for current event output
- tests for repeated segmentation oscillation, sustained lineage chains, and merge-followed-by-split scenarios

Validation:

- smoke tests
- live replay of known merge/split-sensitive windows
- output analysis verifying no duplicate/self-merge regressions and better lineage coherence across multiple scans

Completion notes:

- added persistent lineage fields to track state: `parent_track_ids`, `child_track_ids`, and `absorbed_track_ids`
- merge and split handling now records lineage relationships in track state while keeping the latest-scan event list as a derived compatibility view
- kept public API behavior stable for this task; lineage is now an internal source of truth rather than a required public contract
- targeted verification:
  - `uv run pytest tests/unit/test_tracker.py tests/unit/test_tracking_types.py tests/unit/test_tracking_association.py tests/unit/test_summary.py tests/smoke/test_server_smoke.py -q`
  - `45 passed in 3.24s`
- live replay validation:
  - `uv run python scripts/live_replay.py KTLX --date 2026-04-10 --quick --local-only`
  - `uv run python scripts/live_replay.py KSOX --date 2026-04-10 --quick`
- observed behavior:
  - no summary, motion, or merge/split regressions in the checked windows
  - dense replay remained stable at `48 / 50 / 50` objects with unchanged focal summaries
  - lower-complexity replay remained stable at `3 / 2 / 7` objects
  - the primary gain is persistent lineage state for debugging and future API evolution, not an intentional public-output change in this task

## Task 5: Upgrade motion guidance beyond one global prior

Status: completed

Goal:

- move the motion stage from one bulk scene estimate toward a tiered model that can support local variation

Why:

- one scene-level phase-correlation estimate is a reasonable baseline, but it can be wrong when motion differs across the domain

Scope:

- keep the current global phase-correlation estimate as a baseline
- add regional or local motion guidance in a way that can be inspected and turned off for comparison
- keep public motion conservative and confidence-gated

Deliverables:

- motion module supporting more than one spatial prior
- diagnostics showing which motion source guided each track
- unit and regression tests for motion disagreement scenarios

Validation:

- smoke tests
- live replay over broader windows with spatially heterogeneous storm motion
- API rendering and output analysis focused on reduced false-motion fires and stable spoken motion behavior

Completion notes:

- kept the global phase-correlation motion prior and added a per-track local ROI phase-correlation estimate
- blended local and global guidance conservatively by quality and consistency
- used per-track blended motion guidance for association prediction and track-level fallback motion
- corrected an initial simple-scene regression by requiring local guidance to be both strong and consistent before it influences reported motion
- targeted verification:
  - `uv run pytest tests/unit/test_tracking_motion_field.py tests/unit/test_motion.py tests/unit/test_tracking_association.py tests/unit/test_tracker.py tests/unit/test_summary.py tests/smoke/test_server_smoke.py -q`
  - `64 passed in 3.14s`
- live replay validation:
  - `uv run python scripts/live_replay.py KTLX --date 2026-04-10 --quick --local-only`
  - `uv run python scripts/live_replay.py KSOX --date 2026-04-10 --quick`
- clean-data report:
  - `docs/test_reports/2026-04-11-task5-motion-guidance-report.md`
- observed behavior:
  - dense replay remained stable with plausible spoken motion and no new uncertainty noise
  - lower-complexity replay returned to stationary/nearly-stationary output after the conservative blend gate was added
  - this task now has a dedicated reviewable report with raw replay lines and clean summary text for later comparison

## Task 6: Add quantitative tracking evaluation

Status: pending

Goal:

- complement replay sanity checks with explicit tracking metrics

Why:

- live replay is necessary but not sufficient for best-practice claims
- objective metrics reduce anecdotal tuning drift

Scope:

- define a compact benchmark set of replay windows covering simple, dense, split/merge, growth/decay, and degraded-quality cases
- compute metrics such as identity switches, false merges, false splits, fragmentation, and motion error where adjudication exists
- include summary-focus stability metrics for the product layer

Deliverables:

- benchmark dataset manifest
- metric computation scripts
- evaluation report template

Validation:

- smoke tests only if runtime/API paths changed
- live replay and evaluation runs on the benchmark set
- output analysis that compares before/after metrics, not only anecdotal examples

## Task 7: Calibrate confidence scores and diagnostics

Status: pending

Goal:

- make identity and motion confidence more interpretable and more testable

Why:

- confidence is already central to public-safe motion, but it should become easier to evaluate and trust

Scope:

- define what identity confidence and motion confidence are intended to mean
- verify score monotonicity and calibration behavior on the benchmark set
- expose better machine-readable diagnostics without cluttering user-facing output

Deliverables:

- confidence score definitions
- calibration tests or reports
- improved diagnostics in replay and API debug paths

Validation:

- smoke tests
- live replay plus benchmark evaluation
- output analysis verifying that confidence downgrades align with actual ambiguous tracking situations

## Task Order

Recommended execution order:

1. Task 1: preprocessing and scan-quality metadata
2. Task 2: multilevel segmentation hierarchy
3. Task 3: advected geometry-aware association
4. Task 4: lineage state
5. Task 5: upgraded motion guidance
6. Task 6: quantitative evaluation
7. Task 7: confidence calibration and diagnostics

Rationale:

- preprocessing and segmentation shape the rest of the tracker
- association and lineage should sit on top of improved object structure
- richer motion guidance should be evaluated after the geometry pipeline is stronger
- evaluation and calibration should become first-class as the architecture stabilizes

## Source Basis

- staged generic tracker requirement: `docs/superpowers/specs/2026-04-10-phase2-tracker-refactor-design.md`
- current tracker capabilities and open quality work: `PROGRESS.md`
- live replay methodology and regression evidence: `docs/test_reports/2026-04-10-live-replay-harness.md`
- best-practice alignment discussed in: `docs/test_reports/2026-04-11-tracking-system-report.md`

External methodology sources:

1. Raut, B. A., et al. (2021). "An Adaptive Tracking Algorithm for Convection in Simulated and Remote Sensing Data." DOI: https://doi.org/10.1175/JAMC-D-20-0119.1
2. Han, L., et al. (2009). "3D Convective Storm Identification, Tracking, and Forecasting: An Enhanced TITAN Algorithm." DOI: https://doi.org/10.1175/2008JTECHA1084.1
3. Sokolowsky, G. A., et al. (2024). "`tobac` v1.5: introducing fast 3D tracking, splits and mergers, and other enhancements for identifying and analysing meteorological phenomena." https://gmd.copernicus.org/articles/17/5309/2024/gmd-17-5309-2024.html
4. Lakshmanan, V., and Stumpf, G. J. (2007). WDSS-II background/process reference. https://www.weather.gov/media/mdl/LakshmananStumpf2007_WDSS.pdf
