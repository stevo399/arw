# ARW Progress

## Completed
- Phase 1: site database, reflectivity ingest, rain object detection, speech summaries
- Phase 2: motion tracking with multi-scan trajectories, replay buffer, merge/split detection
  - ReplayBuffer: 2-hour in-memory scan storage with auto-eviction and site-switch reset
  - StormTracker: persistent Track objects with hybrid overlap+centroid matching
  - Motion: linear regression velocity computation over position history
  - Merge/split detection with event reporting
  - Updated speech summaries with motion info
  - New endpoints: /tracks/{site_id}, /motion/{site_id}/{track_id}
- Full test suite: 81 tests (68 unit, 8 smoke, 5 e2e) all passing

## In Progress
- None

## Next
- Phase 3: Velocity ingestion, velocity region detection
- NVGT frontend integration with the REST API

## Blockers / Decisions
- None
