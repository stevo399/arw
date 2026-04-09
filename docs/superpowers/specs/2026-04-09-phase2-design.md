# ARW Phase 2 Design Spec

## Overview

Phase 2 adds storm motion tracking and a replay buffer to the ARW backend. Users can see where storms are heading, how fast they're moving, and review merge/split events. Motion info is integrated into the existing speech summary and exposed through new dedicated endpoints.

**Architecture:** Builds on the Phase 1 FastAPI backend. New modules for buffering, tracking, and motion computation. No new external dependencies.

## Replay Buffer

Stores up to 2 hours of parsed scan data in memory, ordered by timestamp.

**Data structure:** An ordered collection of `BufferedScan` objects, each containing:
- `timestamp` (datetime)
- `site_id` (str)
- `reflectivity_data` (ReflectivityData from parser.py)
- `detected_objects` (list of DetectedObject from detection.py)
- `labeled_grid` (numpy ndarray from scipy.ndimage.label — needed for pixel overlap matching between scans)
- `object_masks` (dict mapping object_id to boolean mask — extracted from labeled_grid for efficient overlap computation)

**Behavior:**
- When a new scan is ingested, it's parsed, objects are detected, and the result is appended to the buffer
- Scans older than 2 hours are evicted automatically
- Buffer is per-site — if the user switches radars, the buffer resets
- The buffer exposes: current scan, previous scan, all scans, scan count, and time range

**Storage:** In-memory only. At ~360x1832 float32 per scan, each scan is ~2.5 MB. 30 scans = ~75 MB. Acceptable.

## Track Objects & Lifecycle

A Track is a persistent object that follows one storm across multiple scans. Each track has:
- `track_id` — unique integer, monotonically increasing
- `positions` — list of (timestamp, centroid_lat, centroid_lon, distance_km, bearing_deg) entries
- `peak_history` — list of (timestamp, peak_dbz, peak_label) entries
- `current_object` — reference to the most recent DetectedObject
- `status` — `active`, `merged`, `split`, `lost`
- `merged_into` — track_id this track merged into (if status is `merged`)
- `split_from` — track_id this track split from (if status is `split`)
- `first_seen` / `last_seen` — timestamps

**Lifecycle:**
1. **New track:** When a detected object can't be matched to any existing track, a new track is created
2. **Updated:** When a detected object matches an existing track, the track's positions and peak history are appended
3. **Merged:** When two or more existing tracks match to a single new object, all but one track are marked `merged` and the surviving track continues
4. **Split:** When one existing track matches to two or more new objects, new child tracks are created (status `split_from` = parent), parent continues with the larger piece
5. **Lost:** When an active track has no match for 2 consecutive scans, it's marked `lost`

## Object Matching Algorithm

**Input:** Objects from the previous scan (with their tracks) and objects from the new scan.

**Matching strategy — hybrid overlap + centroid:**

1. For each new object, compute pixel overlap percentage with each previous-scan object
2. If overlap >= 30%, consider it a candidate match
3. If no overlap candidates, fall back to centroid proximity — match if centroid moved <= the maximum plausible distance (based on max storm speed of 120 km/h x scan interval)
4. Build a cost matrix (overlap percentage as primary score, centroid distance as tiebreaker)
5. Use greedy assignment: best matches first, resolving conflicts by highest overlap

**Detecting merges and splits:**
- **Merge:** Multiple previous objects match to one new object — surviving track absorbs, others marked `merged`
- **Split:** One previous object matches to multiple new objects — parent continues with largest piece, new child tracks created for the rest

**Constants:**
- `MIN_OVERLAP_PCT = 0.30` — minimum pixel overlap to consider a match
- `MAX_STORM_SPEED_KMH = 120` — ceiling for centroid fallback matching
- `MAX_MISSED_SCANS = 2` — scans without a match before marking `lost`

## Motion Computation

**Input:** A track's position history (list of timestamped lat/lon points).

**Method:** Linear regression over the position history to compute velocity vector.
- Fit lat vs time and lon vs time separately using numpy least-squares
- Slopes give degrees/second in each axis, convert to km/h and compass bearing
- With 2+ points, we get a velocity. With 3+ points, the regression smooths out noise.
- Single-point tracks (just appeared): motion reported as "stationary — tracking"

**Output per track:**
- `speed_kmh` — magnitude of velocity vector
- `speed_mph` — converted for speech
- `heading_deg` — compass direction of motion (0=N, 90=E)
- `heading_label` — 16-point cardinal (reuse `degrees_to_bearing` from detection.py)

**Edge cases:**
- Track with only 1 position: speed = 0, heading = None, label = "stationary"
- Track with positions but negligible movement (< 2 km/h): label = "nearly stationary"

## API Endpoints

Two new endpoints, plus modification to existing summary:

| Method | Path | Input | Output |
|--------|------|-------|--------|
| `GET` | `/tracks/{site_id}` | `?datetime=...` (optional) | All active tracks with motion vectors, merge/split events |
| `GET` | `/motion/{site_id}/{track_id}` | | Detailed motion report for single track: full position history, speed, heading, status |

**Modified endpoint:**
- `GET /summary/{site_id}` — now includes motion info appended to each object's description

**New response models:**
- `TrackPosition` — timestamp, lat, lon, distance_km, bearing_deg
- `TrackMotion` — speed_kmh, speed_mph, heading_deg, heading_label
- `StormTrack` — track_id, status, positions, motion, current peak, merged_into, split_from, first_seen, last_seen
- `TracksResponse` — site_id, timestamp, active_count, tracks list, recent_events (merges/splits)
- `TrackDetailResponse` — full single-track detail with complete history

Replay buffer populates automatically as scans are requested over time.

## Updated Summary Format

**Phase 1 format:**
> "Oklahoma City: 2 rain objects detected. Strongest: intense rain, 25 miles W of the radar. Covering approximately 47 square miles."

**Phase 2 format (with motion):**
> "Oklahoma City: 2 rain objects detected. Strongest: intense rain, 25 miles W of the radar, moving NE at 35 mph. Covering approximately 47 square miles."

**With merge/split events:**
> "Oklahoma City: 2 rain objects detected. Strongest: intense rain, 25 miles W of the radar, moving NE at 35 mph. Note: 2 storms merged in the last scan. Covering approximately 47 square miles."

**Stationary/new:**
> "...25 miles W of the radar, stationary."

The summary generator receives the tracks list alongside objects so it can look up motion for each detected object by matching track to current_object.

## Source Layout

**New files:**
```
src/
├── buffer.py          # ReplayBuffer class, BufferedScan dataclass
├── tracker.py         # Track objects, matching algorithm, merge/split detection
├── motion.py          # Linear regression velocity computation
tests/
├── unit/
│   ├── test_buffer.py
│   ├── test_tracker.py
│   └── test_motion.py
```

**Modified files:**
```
src/
├── server.py          # New /tracks and /motion endpoints, buffer integration
├── summary.py         # Add motion info to generate_summary
├── models.py          # New Pydantic models for tracks/motion
tests/
├── unit/
│   └── test_summary.py  # Updated tests for motion in summary
├── smoke/
│   └── test_server_smoke.py  # Smoke tests for new endpoints
└── e2e/
    └── test_full_pipeline.py  # E2E tests with motion tracking
```

**No new dependencies.** Linear regression uses `numpy.polyfit` (already installed).
