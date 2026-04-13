# ARW Progress

## Completed
- Phase 1: site database, reflectivity ingest, rain object detection, speech summaries
- Phase 2: tracker refactor with segmentation, motion guidance, global association, confidence-aware motion, and live replay validation
  - ReplayBuffer: 2-hour in-memory scan storage with auto-eviction and site-switch reset
  - Tracking package split into segmentation, association, motion-field, motion, events, and tracker modules
  - Global association: single-use per-scan track assignment, deduplicated merge events, no self-merges
  - Geometry-aware association: primary matching now also considers motion-advected mask overlap instead of relying only on raw overlap and distance
  - Lineage state: tracks now retain persistent parent/child/absorbed relationships while latest-scan events remain a derived view
  - Motion guidance: per-track local ROI motion now blends with the global phase-correlation prior under conservative consistency rules
  - Quantitative evaluation: benchmark-manifest replay metrics now track focus switches, heading flips, fragmentation proxy, and lineage/event totals across reviewable live windows
  - Confidence calibration: tracks now expose machine-readable identity diagnostics alongside motion confidence, with ambiguity-aware identity scoring and benchmarked focus-confidence metrics
  - Motion: confidence-aware output with uncertain/stationary/nearly-stationary states
  - Motion provenance: reported motion now distinguishes track-history, motion-field, and suppressed outputs
  - Motion guidance: scan-based phase correlation is now preferred over weighted centroid drift for scene motion
  - Segmentation: low-threshold blobs can now be conservatively partitioned around multiple intense internal cores
  - Segmentation hierarchy: internal segmentation now uses a multilevel threshold hierarchy instead of a one-off high-threshold seed scan
  - Echo significance: very small weak echoes are filtered so they do not inflate object counts in simpler scenes
  - Summary focus: strongest-object selection now follows a relevance-aware primary focus track across scans
  - Focus hysteresis: focus handoff now requires a meaningful challenger win instead of switching on minor score changes
  - Replay harness: local-only mode now falls back to the most recent cached scans for a requested date
  - Live replay harness: multi-scan replay script for summary, merge/split, and motion sanity diagnostics
  - Updated speech summaries with confidence-aware motion and total scene coverage
  - New endpoints: /tracks/{site_id}, /motion/{site_id}/{track_id}
- Live validation:
  - merge/split regression replay remains free of duplicate or self-merge events
  - dense-scene replay suppresses absurd spoken motion, uses field-guided fallback motion, and reports scene-level coverage
  - lower-complexity replay count noise improved further after small-weak-object filtering
  - hierarchy-based segmentation further reduced dense-scene fragmentation without destabilizing summary focus
  - geometry-aware association passed regression replay without reintroducing continuity or event noise
  - lineage-state refactor preserved replay behavior while giving merges and splits persistent internal relationships
  - local-plus-global motion guidance passed replay validation after rejecting inconsistent local motion in simpler scenes
  - benchmark evaluation now produces reviewable metric reports with clean replay snapshots for dense, simpler, and merge/split-sensitive live windows
  - confidence-calibration replay now keeps simpler and merge/split-sensitive focus tracks at medium/high identity confidence while still flagging the dense heading-reversal scan as a low-identity case
  - dense 5-scan follow-up now downgrades unstable spoken focus motion to `tracking uncertain` across the reversal-heavy middle of the window instead of speaking each turn as fact
  - motion publication now carries a deeper continuity-ambiguity suppression path, while focus-summary publishability remains a separate product-layer guard
  - focus continuity is now a first-class diagnostic with reviewable evaluation metrics instead of being inferred indirectly from motion confidence
  - focus continuity is now the primary publishability metric for unstable focus-motion summaries in dense scenes, and the dense 5-scan follow-up aligns `3` degraded-continuity scans with `tracking uncertain` output
  - live replay diagnostics now print focus track identity and focus continuity directly so dense-scene summary suppression can be reviewed against the same operational output used for fetch/render checks
  - dense-scene strongest-object summaries are materially more stable across short replay windows
  - longer dense-scene replay windows now keep the late-window focus anchored on the nearer active storm field
  - local-only regression replay now works reliably even when only part of the requested day is cached
- Full test suite: 112 tests all passing

## In Progress
- Residual dense-scene validation for broader replay periods and focus-lineage edge cases, including heading-flip behavior in evolving rain objects

## Next
- Validate focus stability over more replay periods and more storm morphologies
- Stress-test the new confidence diagnostics across broader replay windows and future benchmark additions
- Continue refining conservative multithreshold segmentation so simpler scenes do not fragment unnecessarily
- Phase 3: Velocity ingestion, velocity region detection
- NVGT frontend integration with the REST API

## Blockers / Decisions
- No blocker for current API behavior. Remaining work is quality-related: focus stability and dense-scene continuity are improved in the tested windows, but broader replay coverage is still needed for evolving scenes with abrupt heading changes.
