# ARW Progress

## Completed
- Phase 1: site database, reflectivity ingest, rain object detection, speech summaries
- Phase 2: tracker refactor with segmentation, motion guidance, global association, confidence-aware motion, and live replay validation
  - ReplayBuffer: 2-hour in-memory scan storage with auto-eviction and site-switch reset
  - Tracking package split into segmentation, association, motion-field, motion, events, and tracker modules
  - Global association: single-use per-scan track assignment, deduplicated merge events, no self-merges
  - Motion: confidence-aware output with uncertain/stationary/nearly-stationary states
  - Motion provenance: reported motion now distinguishes track-history, motion-field, and suppressed outputs
  - Motion guidance: scan-based phase correlation is now preferred over weighted centroid drift for scene motion
  - Segmentation: low-threshold blobs can now be conservatively partitioned around multiple intense internal cores
  - Echo significance: very small weak echoes are filtered so they do not inflate object counts in simpler scenes
  - Live replay harness: multi-scan replay script for summary, merge/split, and motion sanity diagnostics
  - Updated speech summaries with confidence-aware motion and total scene coverage
  - New endpoints: /tracks/{site_id}, /motion/{site_id}/{track_id}
- Live validation:
  - merge/split regression replay remains free of duplicate or self-merge events
  - dense-scene replay suppresses absurd spoken motion, uses field-guided fallback motion, and reports scene-level coverage
  - lower-complexity replay count noise improved further after small-weak-object filtering
- Full test suite: 112 tests all passing

## In Progress
- Residual strongest-object stability tuning and conservative segmentation refinement

## Next
- Continue improving strongest-object stability in crowded scenes without introducing site-specific logic
- Continue refining conservative multithreshold segmentation so simpler scenes do not fragment unnecessarily
- Phase 3: Velocity ingestion, velocity region detection
- NVGT frontend integration with the REST API

## Blockers / Decisions
- No blocker for current API behavior. Remaining work is quality-related: dense scenes still need stronger focal-object stability, but simple-scene fragmentation is materially improved after the latest filtering pass.
