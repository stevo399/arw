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
  - benchmark evaluation now includes broader cached replay windows plus lineage-pressure and summary-publishability counts, and the local-only replay harness backfills partially cached windows to keep extended validation honest
  - publishable motion is now also gated by agreement with recent short-horizon track trajectory, which suppressed the late dense-scene reversal pair in live replay without regressing the simpler low-motion window
  - focus continuity now penalizes low-confidence motion under high structural pressure while ignoring heading-instability penalties for stationaryish motion, which aligned dense late-window continuity with the uncertain summaries and removed the false low-continuity hits from the simpler window
  - live replay and API focus diagnostics now expose focus selection margin and runner-up identity, making it clear when dense-scene ambiguity is genuine challenger pressure versus a bad focus handoff
  - focus continuity now also penalizes abrupt reported-motion reversals under structural pressure, which removed the earlier dense-window `WNW` spoken outlier without adding uncertainty to the simpler validation window
  - benchmark evaluation now surfaces focus selection margin, runner-up identity, reported-motion reversal counts, and focus motion source directly in reviewable snapshots
  - dense-scene strongest-object summaries are materially more stable across short replay windows
  - longer dense-scene replay windows now keep the late-window focus anchored on the nearer active storm field
  - local-only regression replay now works reliably even when only part of the requested day is cached
  - replay and benchmark manifests can now target earlier same-day cached windows via an explicit end-of-window scan selector, and short live-style replays validated that behavior across distinct storm morphologies without site-specific logic
  - dense-scene detection runtime is now substantially lower after vectorizing hierarchy-parent assignment, split-pixel allocation, and per-object area accumulation, which restored full broader-manifest evaluation as a practical validation step
  - focus continuity now distinguishes strong focus dominance from true focus ambiguity in dense scenes, allowing stable mid-window motion to publish when challenger pressure is low while still suppressing reversal-heavy scans
  - replay, benchmark, and API diagnostics now expose the recent reported heading sequence directly, making dense-scene suppression decisions reviewable from the actual motion-history pattern instead of only aggregate flip counts
  - focus continuity now classifies recent reported-heading history as insufficient, stable, coherent-turn, mixed, or unstable, which lets dense live replays preserve one-direction turning motion while suppressing reversal-prone heading sequences under structural pressure
  - all `mixed` heading stability cases are now suppressed under dense structural pressure (>= 4 merge/split events), closing the last remaining heading-stability calibration gap
  - broader replay validation across 4 additional windows (KTLX afternoon, KTLX earlier evening, KEYX later evening, KTLX late window) confirmed no regressions from the mixed-suppression change
- Full test suite: 164 tests all passing
- Phase 3: velocity ingestion, velocity region detection, rotation signature detection
  - Parser refactored: `parse_radar_file()` returns pyart Radar object; `extract_reflectivity_from_radar()` and `extract_velocity()` operate on the same radar object (no re-reading)
  - Multi-sweep velocity extraction: up to 3 lowest sweeps with Py-ART region-based dealiasing
  - Velocity region detection: connected-component labeling on thresholded inbound/outbound velocity fields with cross-sweep merging
  - Rotation signature detection: gate-to-gate shear detection (NWS criteria: >= 15 m/s across < 5 km), strength classification (weak/moderate/strong), multi-sweep confirmation
  - Rain object association: velocity regions and rotation signatures matched to nearest DetectedObject by haversine distance
  - Buffer and pipeline integration: BufferedScan carries velocity data, server and replay scripts use the new parse-once pipeline
  - Track rotation history: per-scan rotation entries on tracks with 6-scan cap, enabling persistence detection
  - API: new `/velocity/{site_id}` endpoint, `/objects` gains `max_inbound_ms`/`max_outbound_ms`/`rotation_strength`, `/tracks` gains `rotation_history`
  - Summary integration: rotation language in spoken summaries with persistence patterns ("new rotation", "persistent rotation", "rotation weakening"), standalone rotation reports for secondary objects
  - Live validation: 2 KTLX replay windows confirmed working rotation detection, persistence language, and no tracking regressions
- Full test suite: 195 tests all passing

## In Progress
- None — Phase 3 velocity work is complete

## Next
- Phase 4: Hail detection, debris scoring
- Native web app frontend (replaces earlier NVGT plan)
- Continue refining conservative multithreshold segmentation so simpler scenes do not fragment unnecessarily

## Blockers / Decisions
- No blockers. Phase 3 velocity pipeline is complete and validated against live NEXRAD data.
