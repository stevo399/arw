# ARW Progress

## Completed
- Phase 1: site database, reflectivity ingest, rain object detection, speech summaries
- Phase 2: tracker refactor with segmentation, motion guidance, global association, confidence-aware motion, and live replay validation
  - ReplayBuffer: 2-hour in-memory scan storage with auto-eviction and site-switch reset
  - Tracking package split into segmentation, association, motion-field, motion, events, and tracker modules
  - Global association: single-use per-scan track assignment, deduplicated merge events, no self-merges
  - Motion: confidence-aware output with uncertain/stationary/nearly-stationary states
  - Live replay harness: multi-scan replay script for summary, merge/split, and motion sanity diagnostics
  - Updated speech summaries with confidence-aware motion and total scene coverage
  - New endpoints: /tracks/{site_id}, /motion/{site_id}/{track_id}
- Live validation:
  - merge/split regression replay remains free of duplicate or self-merge events
  - dense-scene replay suppresses absurd spoken motion and now reports scene-level coverage
  - lower-complexity replay remains stable with plausible summaries
- Full test suite: 112 tests all passing

## In Progress
- Final tracker validation closeout and residual dense-scene tuning

## Next
- Reduce dense-scene raw speed spikes that still appear in diagnostics even when spoken summaries correctly downgrade them
- Continue improving strongest-object stability in crowded scenes without introducing site-specific logic
- Phase 3: Velocity ingestion, velocity region detection
- NVGT frontend integration with the REST API

## Blockers / Decisions
- No blocker for current API behavior. Remaining issue is quality-related: dense scenes can still produce raw high-speed track diagnostics even though summary output suppresses them as uncertain.
