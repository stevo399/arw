# ARW Phase 2: Motion Tracking & Replay Buffer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add storm motion tracking with persistent multi-scan trajectories, merge/split detection, and a 2-hour replay buffer to the ARW backend.

**Architecture:** Three new modules (buffer, tracker, motion) build on Phase 1's detection pipeline. The replay buffer stores parsed scans with labeled grids. The tracker maintains persistent Track objects across scans using hybrid overlap+centroid matching. Motion is computed via linear regression over track position history. Two new API endpoints expose tracks/motion, and the existing summary endpoint gains motion info.

**Tech Stack:** Python 3.11+, FastAPI, numpy (linear regression), scipy (labeled grids from Phase 1), existing Phase 1 modules

---

## File Structure

```
src/
├── buffer.py          # NEW: ReplayBuffer class, BufferedScan dataclass
├── tracker.py         # NEW: Track dataclass, StormTracker class (matching, merge/split)
├── motion.py          # NEW: compute_motion() linear regression velocity
├── detection.py       # MODIFY: expose labeled_grid and object_masks from detect_objects
├── models.py          # MODIFY: add TrackPosition, TrackMotion, StormTrack, TracksResponse, TrackDetailResponse, TrackEvent
├── summary.py         # MODIFY: accept optional tracks, add motion to summary text
├── server.py          # MODIFY: integrate buffer+tracker, add /tracks and /motion endpoints
tests/
├── unit/
│   ├── test_buffer.py     # NEW
│   ├── test_tracker.py    # NEW
│   ├── test_motion.py     # NEW
│   ├── test_detection.py  # MODIFY: test new return values
│   ├── test_summary.py    # MODIFY: test motion in summaries
│   └── test_models.py     # MODIFY: test new models
├── smoke/
│   └── test_server_smoke.py  # MODIFY: smoke tests for new endpoints
└── e2e/
    └── test_full_pipeline.py  # MODIFY: e2e with tracking
```

---

### Task 1: Extend detect_objects to Return Labeled Grid and Masks

**Files:**
- Modify: `src/detection.py`
- Modify: `tests/unit/test_detection.py`

- [ ] **Step 1: Write the failing tests**

Add these tests to the bottom of `tests/unit/test_detection.py`:

```python
from src.detection import detect_objects_with_grid, DetectionResult


def test_detect_objects_with_grid_returns_result():
    reflectivity = np.full((360, 500), np.nan)
    reflectivity[85:95, 195:205] = 45.0
    azimuths = np.linspace(0, 359, 360)
    ranges_m = np.linspace(2000, 250000, 500)
    result = detect_objects_with_grid(
        reflectivity=reflectivity,
        azimuths=azimuths,
        ranges_m=ranges_m,
        radar_lat=35.0,
        radar_lon=-97.0,
    )
    assert isinstance(result, DetectionResult)
    assert len(result.objects) == 1
    assert result.labeled_grid.shape == reflectivity.shape
    assert len(result.object_masks) == 1
    assert result.object_masks[1].shape == reflectivity.shape
    assert result.object_masks[1].dtype == bool


def test_detect_objects_with_grid_masks_match_objects():
    reflectivity = np.full((360, 500), np.nan)
    reflectivity[10:20, 50:60] = 35.0
    reflectivity[200:210, 300:310] = 55.0
    azimuths = np.linspace(0, 359, 360)
    ranges_m = np.linspace(2000, 250000, 500)
    result = detect_objects_with_grid(
        reflectivity=reflectivity,
        azimuths=azimuths,
        ranges_m=ranges_m,
        radar_lat=35.0,
        radar_lon=-97.0,
    )
    assert len(result.objects) == 2
    assert len(result.object_masks) == 2
    for obj in result.objects:
        assert obj.object_id in result.object_masks
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_detection.py::test_detect_objects_with_grid_returns_result -v`
Expected: FAIL — `ImportError: cannot import name 'detect_objects_with_grid'`

- [ ] **Step 3: Write the implementation**

Add to the bottom of `src/detection.py`:

```python
@dataclass
class DetectionResult:
    """Result of object detection including labeled grid for tracking."""
    objects: list[DetectedObject]
    labeled_grid: np.ndarray
    object_masks: dict[int, np.ndarray]


def detect_objects_with_grid(
    reflectivity: np.ndarray,
    azimuths: np.ndarray,
    ranges_m: np.ndarray,
    radar_lat: float,
    radar_lon: float,
) -> DetectionResult:
    """Detect rain objects and return labeled grid + masks for tracking.

    Same as detect_objects but also returns the scipy labeled grid and
    per-object boolean masks needed for overlap-based tracking.
    """
    valid = ~np.isnan(reflectivity) & (reflectivity >= MIN_DBZ_THRESHOLD)
    labeled, num_features = label(valid)

    objects = []
    object_masks = {}
    for i in range(1, num_features + 1):
        obj_mask = labeled == i
        obj = compute_object_properties(
            obj_mask=obj_mask,
            reflectivity=reflectivity,
            azimuths=azimuths,
            ranges_m=ranges_m,
            radar_lat=radar_lat,
            radar_lon=radar_lon,
            object_id=i,
        )
        if obj is not None:
            objects.append(obj)
            object_masks[obj.object_id] = obj_mask

    objects.sort(key=lambda o: o.peak_dbz, reverse=True)
    return DetectionResult(
        objects=objects,
        labeled_grid=labeled,
        object_masks=object_masks,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_detection.py -v`
Expected: All 11 tests PASS (9 existing + 2 new).

- [ ] **Step 5: Commit**

```bash
git add src/detection.py tests/unit/test_detection.py
git commit -m "feat: add detect_objects_with_grid returning labeled grid and masks"
```

---

### Task 2: New Pydantic Models for Tracks & Motion

**Files:**
- Modify: `src/models.py`
- Modify: `tests/unit/test_models.py`

- [ ] **Step 1: Write the failing tests**

Add these tests to the bottom of `tests/unit/test_models.py`:

```python
from src.models import TrackPosition, TrackMotion, StormTrack, TracksResponse, TrackDetailResponse, TrackEvent


def test_track_position_model():
    pos = TrackPosition(
        timestamp="2026-04-08T18:30:00Z",
        latitude=35.5,
        longitude=-97.3,
        distance_km=40.2,
        bearing_deg=270.0,
    )
    assert pos.latitude == 35.5


def test_track_motion_model():
    motion = TrackMotion(
        speed_kmh=56.3,
        speed_mph=35,
        heading_deg=45.0,
        heading_label="NE",
    )
    assert motion.speed_mph == 35
    assert motion.heading_label == "NE"


def test_track_motion_model_stationary():
    motion = TrackMotion(
        speed_kmh=0.0,
        speed_mph=0,
        heading_deg=None,
        heading_label="stationary",
    )
    assert motion.heading_deg is None


def test_storm_track_model():
    track = StormTrack(
        track_id=1,
        status="active",
        positions=[
            TrackPosition(timestamp="2026-04-08T18:30:00Z", latitude=35.5, longitude=-97.3, distance_km=40.2, bearing_deg=270.0),
        ],
        motion=TrackMotion(speed_kmh=56.3, speed_mph=35, heading_deg=45.0, heading_label="NE"),
        peak_dbz=55.0,
        peak_label="intense rain",
        merged_into=None,
        split_from=None,
        first_seen="2026-04-08T18:20:00Z",
        last_seen="2026-04-08T18:30:00Z",
    )
    assert track.track_id == 1
    assert track.status == "active"


def test_track_event_model():
    event = TrackEvent(
        event_type="merge",
        timestamp="2026-04-08T18:30:00Z",
        description="Tracks 2, 3 merged into track 1",
        involved_track_ids=[1, 2, 3],
    )
    assert event.event_type == "merge"


def test_tracks_response_model():
    resp = TracksResponse(
        site_id="KTLX",
        timestamp="2026-04-08T18:30:00Z",
        active_count=0,
        tracks=[],
        recent_events=[],
    )
    assert resp.active_count == 0


def test_track_detail_response_model():
    resp = TrackDetailResponse(
        track_id=1,
        status="active",
        positions=[],
        motion=TrackMotion(speed_kmh=0.0, speed_mph=0, heading_deg=None, heading_label="stationary"),
        peak_history=[],
        merged_into=None,
        split_from=None,
        first_seen="2026-04-08T18:20:00Z",
        last_seen="2026-04-08T18:30:00Z",
    )
    assert resp.track_id == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_models.py::test_track_position_model -v`
Expected: FAIL — `ImportError: cannot import name 'TrackPosition'`

- [ ] **Step 3: Write the implementation**

Add to the bottom of `src/models.py`:

```python
class TrackPosition(BaseModel):
    timestamp: str
    latitude: float
    longitude: float
    distance_km: float
    bearing_deg: float


class TrackMotion(BaseModel):
    speed_kmh: float
    speed_mph: int
    heading_deg: float | None
    heading_label: str


class PeakHistoryEntry(BaseModel):
    timestamp: str
    peak_dbz: float
    peak_label: str


class StormTrack(BaseModel):
    track_id: int
    status: str
    positions: list[TrackPosition]
    motion: TrackMotion
    peak_dbz: float
    peak_label: str
    merged_into: int | None
    split_from: int | None
    first_seen: str
    last_seen: str


class TrackEvent(BaseModel):
    event_type: str
    timestamp: str
    description: str
    involved_track_ids: list[int]


class TracksResponse(BaseModel):
    site_id: str
    timestamp: str
    active_count: int
    tracks: list[StormTrack]
    recent_events: list[TrackEvent]


class TrackDetailResponse(BaseModel):
    track_id: int
    status: str
    positions: list[TrackPosition]
    motion: TrackMotion
    peak_history: list[PeakHistoryEntry]
    merged_into: int | None
    split_from: int | None
    first_seen: str
    last_seen: str
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_models.py -v`
Expected: All 14 tests PASS (6 existing + 8 new).

- [ ] **Step 5: Commit**

```bash
git add src/models.py tests/unit/test_models.py
git commit -m "feat: add Pydantic models for tracks, motion, and events"
```

---

### Task 3: Replay Buffer

**Files:**
- Create: `src/buffer.py`
- Create: `tests/unit/test_buffer.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_buffer.py
from datetime import datetime, timedelta
import numpy as np
from src.buffer import ReplayBuffer, BufferedScan
from src.parser import ReflectivityData
from src.detection import DetectedObject


def _make_buffered_scan(site_id: str, timestamp: datetime, num_objects: int = 1) -> BufferedScan:
    """Create a minimal BufferedScan for testing."""
    ref_data = ReflectivityData(
        reflectivity=np.full((360, 500), np.nan),
        azimuths=np.linspace(0, 359, 360),
        ranges_m=np.linspace(2000, 250000, 500),
        radar_lat=35.3331,
        radar_lon=-97.2778,
        elevation_angle=0.5,
        elevation_angles=[0.5],
        timestamp=timestamp.isoformat(),
    )
    objects = [
        DetectedObject(
            object_id=i + 1,
            centroid_lat=35.5 + i * 0.1,
            centroid_lon=-97.3,
            distance_km=40.0 + i * 10,
            bearing_deg=270.0,
            peak_dbz=45.0,
            peak_label="heavy rain",
            area_km2=100.0,
            layers=[],
        )
        for i in range(num_objects)
    ]
    labeled_grid = np.zeros((360, 500), dtype=int)
    object_masks = {}
    for i in range(num_objects):
        mask = np.zeros((360, 500), dtype=bool)
        mask[85 + i * 20:95 + i * 20, 195:205] = True
        labeled_grid[mask] = i + 1
        object_masks[i + 1] = mask
    return BufferedScan(
        timestamp=timestamp,
        site_id=site_id,
        reflectivity_data=ref_data,
        detected_objects=objects,
        labeled_grid=labeled_grid,
        object_masks=object_masks,
    )


def test_buffer_initially_empty():
    buf = ReplayBuffer()
    assert buf.scan_count == 0
    assert buf.current_scan is None
    assert buf.previous_scan is None


def test_buffer_add_scan():
    buf = ReplayBuffer()
    scan = _make_buffered_scan("KTLX", datetime(2026, 4, 8, 18, 30))
    buf.add_scan(scan)
    assert buf.scan_count == 1
    assert buf.current_scan is scan
    assert buf.previous_scan is None


def test_buffer_two_scans():
    buf = ReplayBuffer()
    scan1 = _make_buffered_scan("KTLX", datetime(2026, 4, 8, 18, 30))
    scan2 = _make_buffered_scan("KTLX", datetime(2026, 4, 8, 18, 35))
    buf.add_scan(scan1)
    buf.add_scan(scan2)
    assert buf.scan_count == 2
    assert buf.current_scan is scan2
    assert buf.previous_scan is scan1


def test_buffer_evicts_old_scans():
    buf = ReplayBuffer(max_age_minutes=120)
    base_time = datetime(2026, 4, 8, 16, 0)
    # Add scans spanning 3 hours
    for i in range(36):  # every 5 min for 3 hours
        scan = _make_buffered_scan("KTLX", base_time + timedelta(minutes=i * 5))
        buf.add_scan(scan)
    # Should have evicted scans older than 2 hours from the latest
    latest = buf.current_scan.timestamp
    for scan in buf.all_scans:
        age = latest - scan.timestamp
        assert age <= timedelta(minutes=120)


def test_buffer_resets_on_site_change():
    buf = ReplayBuffer()
    scan1 = _make_buffered_scan("KTLX", datetime(2026, 4, 8, 18, 30))
    buf.add_scan(scan1)
    assert buf.scan_count == 1
    scan2 = _make_buffered_scan("KFWS", datetime(2026, 4, 8, 18, 35))
    buf.add_scan(scan2)
    assert buf.scan_count == 1  # Reset, only new scan
    assert buf.current_scan.site_id == "KFWS"


def test_buffer_all_scans_ordered():
    buf = ReplayBuffer()
    times = [datetime(2026, 4, 8, 18, i * 5) for i in range(5)]
    for t in times:
        buf.add_scan(_make_buffered_scan("KTLX", t))
    scans = buf.all_scans
    for i in range(len(scans) - 1):
        assert scans[i].timestamp <= scans[i + 1].timestamp


def test_buffer_time_range():
    buf = ReplayBuffer()
    scan1 = _make_buffered_scan("KTLX", datetime(2026, 4, 8, 18, 0))
    scan2 = _make_buffered_scan("KTLX", datetime(2026, 4, 8, 18, 30))
    buf.add_scan(scan1)
    buf.add_scan(scan2)
    start, end = buf.time_range
    assert start == datetime(2026, 4, 8, 18, 0)
    assert end == datetime(2026, 4, 8, 18, 30)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_buffer.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.buffer'`

- [ ] **Step 3: Write the implementation**

```python
# src/buffer.py
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import deque
import numpy as np
from src.parser import ReflectivityData
from src.detection import DetectedObject


@dataclass
class BufferedScan:
    """A single scan stored in the replay buffer."""
    timestamp: datetime
    site_id: str
    reflectivity_data: ReflectivityData
    detected_objects: list[DetectedObject]
    labeled_grid: np.ndarray
    object_masks: dict[int, np.ndarray]


class ReplayBuffer:
    """Stores up to max_age_minutes of parsed scan data in memory."""

    def __init__(self, max_age_minutes: int = 120):
        self._scans: deque[BufferedScan] = deque()
        self._max_age = timedelta(minutes=max_age_minutes)
        self._current_site: str | None = None

    def add_scan(self, scan: BufferedScan) -> None:
        """Add a scan to the buffer. Resets if site changes. Evicts old scans."""
        if self._current_site is not None and scan.site_id != self._current_site:
            self._scans.clear()
        self._current_site = scan.site_id
        self._scans.append(scan)
        self._evict_old()

    def _evict_old(self) -> None:
        """Remove scans older than max_age from the latest scan."""
        if not self._scans:
            return
        cutoff = self._scans[-1].timestamp - self._max_age
        while self._scans and self._scans[0].timestamp < cutoff:
            self._scans.popleft()

    @property
    def scan_count(self) -> int:
        return len(self._scans)

    @property
    def current_scan(self) -> BufferedScan | None:
        return self._scans[-1] if self._scans else None

    @property
    def previous_scan(self) -> BufferedScan | None:
        return self._scans[-2] if len(self._scans) >= 2 else None

    @property
    def all_scans(self) -> list[BufferedScan]:
        return list(self._scans)

    @property
    def time_range(self) -> tuple[datetime, datetime] | None:
        if not self._scans:
            return None
        return (self._scans[0].timestamp, self._scans[-1].timestamp)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_buffer.py -v`
Expected: All 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/buffer.py tests/unit/test_buffer.py
git commit -m "feat: add replay buffer with 2-hour eviction and site reset"
```

---

### Task 4: Motion Computation

**Files:**
- Create: `src/motion.py`
- Create: `tests/unit/test_motion.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_motion.py
import math
from datetime import datetime, timedelta
from src.motion import compute_motion, MotionVector

NEARLY_STATIONARY_KMH = 2.0


def test_motion_single_point():
    """Single position = stationary."""
    positions = [
        (datetime(2026, 4, 8, 18, 30), 35.0, -97.0),
    ]
    motion = compute_motion(positions)
    assert isinstance(motion, MotionVector)
    assert motion.speed_kmh == 0.0
    assert motion.speed_mph == 0
    assert motion.heading_deg is None
    assert motion.heading_label == "stationary"


def test_motion_two_points_moving_north():
    """Two points moving north should give ~0 deg heading."""
    positions = [
        (datetime(2026, 4, 8, 18, 0), 35.0, -97.0),
        (datetime(2026, 4, 8, 18, 30), 35.5, -97.0),  # ~55 km north in 30 min
    ]
    motion = compute_motion(positions)
    assert motion.speed_kmh > 50
    assert motion.speed_kmh < 150
    assert motion.heading_deg is not None
    assert abs(motion.heading_deg - 0) < 20 or abs(motion.heading_deg - 360) < 20
    assert motion.heading_label in ("N", "NNE", "NNW")


def test_motion_two_points_moving_east():
    """Two points moving east should give ~90 deg heading."""
    positions = [
        (datetime(2026, 4, 8, 18, 0), 35.0, -97.0),
        (datetime(2026, 4, 8, 18, 30), 35.0, -96.5),  # ~45 km east in 30 min
    ]
    motion = compute_motion(positions)
    assert motion.speed_kmh > 40
    assert motion.heading_deg is not None
    assert 60 < motion.heading_deg < 120
    assert motion.heading_label in ("E", "ENE", "ESE")


def test_motion_nearly_stationary():
    """Barely moving should be labeled nearly stationary."""
    positions = [
        (datetime(2026, 4, 8, 18, 0), 35.0, -97.0),
        (datetime(2026, 4, 8, 18, 30), 35.0001, -97.0),  # ~11 meters in 30 min
    ]
    motion = compute_motion(positions)
    assert motion.speed_kmh < NEARLY_STATIONARY_KMH
    assert motion.heading_label == "nearly stationary"


def test_motion_three_points_smooths():
    """Three points should produce a smoother estimate than two."""
    positions = [
        (datetime(2026, 4, 8, 18, 0), 35.0, -97.0),
        (datetime(2026, 4, 8, 18, 5), 35.05, -97.0),
        (datetime(2026, 4, 8, 18, 10), 35.1, -97.0),
    ]
    motion = compute_motion(positions)
    assert motion.speed_kmh > 0
    assert motion.heading_deg is not None
    assert abs(motion.heading_deg - 0) < 20 or abs(motion.heading_deg - 360) < 20


def test_motion_mph_conversion():
    """Speed in mph should be km/h divided by 1.60934, rounded."""
    positions = [
        (datetime(2026, 4, 8, 18, 0), 35.0, -97.0),
        (datetime(2026, 4, 8, 19, 0), 35.0, -96.0),  # ~90 km east in 1 hour
    ]
    motion = compute_motion(positions)
    expected_mph = round(motion.speed_kmh / 1.60934)
    assert motion.speed_mph == expected_mph
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_motion.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.motion'`

- [ ] **Step 3: Write the implementation**

```python
# src/motion.py
import math
from dataclasses import dataclass
from datetime import datetime
import numpy as np
from src.detection import degrees_to_bearing

KM_PER_DEGREE_LAT = 111.32
KM_PER_MILE = 1.60934
NEARLY_STATIONARY_KMH = 2.0


@dataclass
class MotionVector:
    """Computed motion for a storm track."""
    speed_kmh: float
    speed_mph: int
    heading_deg: float | None
    heading_label: str


def compute_motion(positions: list[tuple[datetime, float, float]]) -> MotionVector:
    """Compute motion vector from a list of (timestamp, lat, lon) positions.

    Uses linear regression over position history.
    Single-point: stationary. < 2 km/h: nearly stationary.
    """
    if len(positions) < 2:
        return MotionVector(speed_kmh=0.0, speed_mph=0, heading_deg=None, heading_label="stationary")

    # Convert timestamps to seconds from first position
    t0 = positions[0][0]
    times_s = np.array([(p[0] - t0).total_seconds() for p in positions])
    lats = np.array([p[1] for p in positions])
    lons = np.array([p[2] for p in positions])

    # Linear regression: fit lat and lon vs time
    lat_slope = np.polyfit(times_s, lats, 1)[0]  # degrees per second
    lon_slope = np.polyfit(times_s, lons, 1)[0]  # degrees per second

    # Convert to km/h
    mean_lat = np.mean(lats)
    km_per_degree_lon = KM_PER_DEGREE_LAT * math.cos(math.radians(mean_lat))

    lat_kmh = lat_slope * 3600 * KM_PER_DEGREE_LAT
    lon_kmh = lon_slope * 3600 * km_per_degree_lon

    speed_kmh = math.sqrt(lat_kmh ** 2 + lon_kmh ** 2)
    speed_kmh = round(speed_kmh, 1)

    if speed_kmh < NEARLY_STATIONARY_KMH:
        return MotionVector(
            speed_kmh=speed_kmh,
            speed_mph=round(speed_kmh / KM_PER_MILE),
            heading_deg=None,
            heading_label="nearly stationary",
        )

    # Heading: atan2(east, north) to get compass bearing
    heading_rad = math.atan2(lon_kmh, lat_kmh)
    heading_deg = math.degrees(heading_rad) % 360
    heading_deg = round(heading_deg, 1)
    heading_label = degrees_to_bearing(heading_deg)

    speed_mph = round(speed_kmh / KM_PER_MILE)

    return MotionVector(
        speed_kmh=speed_kmh,
        speed_mph=speed_mph,
        heading_deg=heading_deg,
        heading_label=heading_label,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_motion.py -v`
Expected: All 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/motion.py tests/unit/test_motion.py
git commit -m "feat: add linear regression motion computation"
```

---

### Task 5: Storm Tracker — Track Dataclass and Core Matching

**Files:**
- Create: `src/tracker.py`
- Create: `tests/unit/test_tracker.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_tracker.py
from datetime import datetime, timedelta
import numpy as np
from src.tracker import StormTracker, Track
from src.detection import DetectedObject
from src.buffer import BufferedScan
from src.parser import ReflectivityData


def _make_object(obj_id: int, lat: float, lon: float, peak_dbz: float = 45.0) -> DetectedObject:
    return DetectedObject(
        object_id=obj_id,
        centroid_lat=lat,
        centroid_lon=lon,
        distance_km=40.0,
        bearing_deg=270.0,
        peak_dbz=peak_dbz,
        peak_label="heavy rain",
        area_km2=100.0,
        layers=[],
    )


def _make_scan(
    site_id: str,
    timestamp: datetime,
    objects: list[DetectedObject],
    grid_shape: tuple[int, int] = (360, 500),
    masks: dict[int, np.ndarray] | None = None,
) -> BufferedScan:
    ref_data = ReflectivityData(
        reflectivity=np.full(grid_shape, np.nan),
        azimuths=np.linspace(0, 359, grid_shape[0]),
        ranges_m=np.linspace(2000, 250000, grid_shape[1]),
        radar_lat=35.3331,
        radar_lon=-97.2778,
        elevation_angle=0.5,
        elevation_angles=[0.5],
        timestamp=timestamp.isoformat(),
    )
    labeled_grid = np.zeros(grid_shape, dtype=int)
    if masks is None:
        masks = {}
        for obj in objects:
            mask = np.zeros(grid_shape, dtype=bool)
            row = min(85 + (obj.object_id - 1) * 40, grid_shape[0] - 15)
            mask[row:row + 10, 195:205] = True
            labeled_grid[mask] = obj.object_id
            masks[obj.object_id] = mask
    else:
        for obj_id, mask in masks.items():
            labeled_grid[mask] = obj_id
    return BufferedScan(
        timestamp=timestamp,
        site_id=site_id,
        reflectivity_data=ref_data,
        detected_objects=objects,
        labeled_grid=labeled_grid,
        object_masks=masks,
    )


def test_tracker_first_scan_creates_tracks():
    tracker = StormTracker()
    t = datetime(2026, 4, 8, 18, 30)
    objects = [_make_object(1, 35.5, -97.3), _make_object(2, 35.8, -97.1)]
    scan = _make_scan("KTLX", t, objects)
    tracker.update(scan)
    assert len(tracker.active_tracks) == 2
    for track in tracker.active_tracks:
        assert track.status == "active"
        assert len(track.positions) == 1


def test_tracker_second_scan_updates_tracks():
    tracker = StormTracker()
    t1 = datetime(2026, 4, 8, 18, 30)
    t2 = datetime(2026, 4, 8, 18, 35)
    obj1 = _make_object(1, 35.5, -97.3)
    obj2 = _make_object(1, 35.51, -97.29)  # Slightly moved
    mask1 = np.zeros((360, 500), dtype=bool)
    mask1[85:95, 195:205] = True
    mask2 = np.zeros((360, 500), dtype=bool)
    mask2[86:96, 196:206] = True  # Overlaps with mask1
    scan1 = _make_scan("KTLX", t1, [obj1], masks={1: mask1})
    scan2 = _make_scan("KTLX", t2, [obj2], masks={1: mask2})
    tracker.update(scan1)
    tracker.update(scan2)
    assert len(tracker.active_tracks) == 1
    track = tracker.active_tracks[0]
    assert len(track.positions) == 2


def test_tracker_unmatched_object_creates_new_track():
    tracker = StormTracker()
    t1 = datetime(2026, 4, 8, 18, 30)
    t2 = datetime(2026, 4, 8, 18, 35)
    obj1 = _make_object(1, 35.5, -97.3)
    obj2 = _make_object(1, 36.5, -96.0)  # Far away, no overlap
    mask1 = np.zeros((360, 500), dtype=bool)
    mask1[85:95, 195:205] = True
    mask2 = np.zeros((360, 500), dtype=bool)
    mask2[300:310, 400:410] = True  # No overlap with mask1
    scan1 = _make_scan("KTLX", t1, [obj1], masks={1: mask1})
    scan2 = _make_scan("KTLX", t2, [obj2], masks={1: mask2})
    tracker.update(scan1)
    tracker.update(scan2)
    # Original track should be active (1 missed scan), new track created
    all_tracks = tracker.all_tracks
    assert len(all_tracks) >= 2


def test_tracker_lost_after_missed_scans():
    tracker = StormTracker()
    t1 = datetime(2026, 4, 8, 18, 30)
    obj1 = _make_object(1, 35.5, -97.3)
    mask1 = np.zeros((360, 500), dtype=bool)
    mask1[85:95, 195:205] = True
    scan1 = _make_scan("KTLX", t1, [obj1], masks={1: mask1})
    tracker.update(scan1)
    # Two empty scans = lost
    for i in range(2):
        t = t1 + timedelta(minutes=(i + 1) * 5)
        empty_scan = _make_scan("KTLX", t, [])
        tracker.update(empty_scan)
    lost = [t for t in tracker.all_tracks if t.status == "lost"]
    assert len(lost) == 1


def test_tracker_merge_detection():
    tracker = StormTracker()
    t1 = datetime(2026, 4, 8, 18, 30)
    t2 = datetime(2026, 4, 8, 18, 35)
    # Two separate objects in scan 1
    obj_a = _make_object(1, 35.5, -97.3, peak_dbz=50.0)
    obj_b = _make_object(2, 35.5, -97.28, peak_dbz=40.0)
    mask_a = np.zeros((360, 500), dtype=bool)
    mask_a[85:95, 195:205] = True
    mask_b = np.zeros((360, 500), dtype=bool)
    mask_b[85:95, 205:215] = True
    scan1 = _make_scan("KTLX", t1, [obj_a, obj_b], masks={1: mask_a, 2: mask_b})
    # One merged object in scan 2, overlapping both mask_a and mask_b
    obj_merged = _make_object(1, 35.5, -97.29, peak_dbz=52.0)
    mask_merged = np.zeros((360, 500), dtype=bool)
    mask_merged[85:95, 195:215] = True  # Covers both previous masks
    scan2 = _make_scan("KTLX", t2, [obj_merged], masks={1: mask_merged})
    tracker.update(scan1)
    tracker.update(scan2)
    merged = [t for t in tracker.all_tracks if t.status == "merged"]
    assert len(merged) >= 1
    events = tracker.recent_events
    merge_events = [e for e in events if e["event_type"] == "merge"]
    assert len(merge_events) >= 1


def test_tracker_split_detection():
    tracker = StormTracker()
    t1 = datetime(2026, 4, 8, 18, 30)
    t2 = datetime(2026, 4, 8, 18, 35)
    # One object in scan 1
    obj1 = _make_object(1, 35.5, -97.3, peak_dbz=50.0)
    mask1 = np.zeros((360, 500), dtype=bool)
    mask1[85:95, 195:215] = True
    scan1 = _make_scan("KTLX", t1, [obj1], masks={1: mask1})
    # Two objects in scan 2, each overlapping the original
    obj_a = _make_object(1, 35.5, -97.31, peak_dbz=48.0)
    obj_b = _make_object(2, 35.5, -97.28, peak_dbz=35.0)
    mask_a = np.zeros((360, 500), dtype=bool)
    mask_a[85:95, 195:205] = True  # Left half
    mask_b = np.zeros((360, 500), dtype=bool)
    mask_b[85:95, 205:215] = True  # Right half
    scan2 = _make_scan("KTLX", t2, [obj_a, obj_b], masks={1: mask_a, 2: mask_b})
    tracker.update(scan1)
    tracker.update(scan2)
    active = tracker.active_tracks
    assert len(active) >= 2
    events = tracker.recent_events
    split_events = [e for e in events if e["event_type"] == "split"]
    assert len(split_events) >= 1


def test_tracker_get_track_by_id():
    tracker = StormTracker()
    t = datetime(2026, 4, 8, 18, 30)
    scan = _make_scan("KTLX", t, [_make_object(1, 35.5, -97.3)])
    tracker.update(scan)
    track = tracker.get_track(1)
    assert track is not None
    assert track.track_id == 1
    assert tracker.get_track(999) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_tracker.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.tracker'`

- [ ] **Step 3: Write the implementation**

```python
# src/tracker.py
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
from src.buffer import BufferedScan
from src.detection import DetectedObject
from src.motion import compute_motion, MotionVector
from src.sites import haversine_distance_km

MIN_OVERLAP_PCT = 0.30
MAX_STORM_SPEED_KMH = 120.0
MAX_MISSED_SCANS = 2


@dataclass
class TrackPosition:
    timestamp: datetime
    latitude: float
    longitude: float
    distance_km: float
    bearing_deg: float


@dataclass
class PeakEntry:
    timestamp: datetime
    peak_dbz: float
    peak_label: str


@dataclass
class Track:
    track_id: int
    status: str  # "active", "merged", "split", "lost"
    positions: list[TrackPosition] = field(default_factory=list)
    peak_history: list[PeakEntry] = field(default_factory=list)
    current_object: DetectedObject | None = None
    merged_into: int | None = None
    split_from: int | None = None
    first_seen: datetime | None = None
    last_seen: datetime | None = None
    _missed_scans: int = 0

    def add_position(self, timestamp: datetime, obj: DetectedObject) -> None:
        self.positions.append(TrackPosition(
            timestamp=timestamp,
            latitude=obj.centroid_lat,
            longitude=obj.centroid_lon,
            distance_km=obj.distance_km,
            bearing_deg=obj.bearing_deg,
        ))
        self.peak_history.append(PeakEntry(
            timestamp=timestamp,
            peak_dbz=obj.peak_dbz,
            peak_label=obj.peak_label,
        ))
        self.current_object = obj
        self.last_seen = timestamp
        self._missed_scans = 0
        if self.first_seen is None:
            self.first_seen = timestamp

    def get_motion(self) -> MotionVector:
        pos_tuples = [
            (p.timestamp, p.latitude, p.longitude)
            for p in self.positions
        ]
        return compute_motion(pos_tuples)


def _compute_overlap(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Compute overlap percentage between two boolean masks.

    Returns the fraction of mask_a's pixels that overlap with mask_b.
    """
    if not np.any(mask_a):
        return 0.0
    intersection = np.sum(mask_a & mask_b)
    return float(intersection) / float(np.sum(mask_a))


class StormTracker:
    """Tracks storms across multiple radar scans."""

    def __init__(self):
        self._tracks: list[Track] = []
        self._next_id: int = 1
        self._recent_events: list[dict] = []
        self._prev_scan: BufferedScan | None = None
        self._obj_to_track: dict[int, int] = {}  # object_id -> track_id for current scan

    def _create_track(self, timestamp: datetime, obj: DetectedObject) -> Track:
        track = Track(track_id=self._next_id, status="active")
        self._next_id += 1
        track.add_position(timestamp, obj)
        self._tracks.append(track)
        return track

    def update(self, scan: BufferedScan) -> None:
        """Process a new scan: match objects to tracks, detect merges/splits."""
        self._recent_events.clear()
        timestamp = scan.timestamp

        if self._prev_scan is None:
            # First scan: create a track for each object
            self._obj_to_track.clear()
            for obj in scan.detected_objects:
                track = self._create_track(timestamp, obj)
                self._obj_to_track[obj.object_id] = track.track_id
            self._prev_scan = scan
            return

        prev_masks = self._prev_scan.object_masks
        new_masks = scan.object_masks
        prev_objects = {obj.object_id: obj for obj in self._prev_scan.detected_objects}
        new_objects = {obj.object_id: obj for obj in scan.detected_objects}

        # Compute scan interval for centroid fallback
        dt_hours = (timestamp - self._prev_scan.timestamp).total_seconds() / 3600.0
        max_distance_km = MAX_STORM_SPEED_KMH * dt_hours if dt_hours > 0 else 50.0

        # Build match candidates: new_obj_id -> list of (prev_obj_id, overlap_pct)
        match_candidates: dict[int, list[tuple[int, float]]] = {
            nid: [] for nid in new_objects
        }
        # Also track reverse: prev_obj_id -> list of (new_obj_id, overlap_pct)
        reverse_candidates: dict[int, list[tuple[int, float]]] = {
            pid: [] for pid in prev_objects
        }

        for new_id, new_mask in new_masks.items():
            for prev_id, prev_mask in prev_masks.items():
                overlap = _compute_overlap(prev_mask, new_mask)
                if overlap >= MIN_OVERLAP_PCT:
                    match_candidates[new_id].append((prev_id, overlap))
                    reverse_candidates.setdefault(prev_id, []).append((new_id, overlap))

        # Centroid fallback for unmatched new objects
        for new_id in list(match_candidates.keys()):
            if not match_candidates[new_id] and new_id in new_objects:
                new_obj = new_objects[new_id]
                for prev_id, prev_obj in prev_objects.items():
                    dist = haversine_distance_km(
                        prev_obj.centroid_lat, prev_obj.centroid_lon,
                        new_obj.centroid_lat, new_obj.centroid_lon,
                    )
                    if dist <= max_distance_km:
                        match_candidates[new_id].append((prev_id, 0.0))
                        reverse_candidates.setdefault(prev_id, []).append((new_id, 0.0))

        # Greedy assignment: sort all candidate pairs by overlap desc
        all_pairs = []
        for new_id, candidates in match_candidates.items():
            for prev_id, overlap in candidates:
                all_pairs.append((overlap, prev_id, new_id))
        all_pairs.sort(reverse=True)

        assigned_new: set[int] = set()
        assigned_prev: set[int] = set()
        assignments: dict[int, list[int]] = {}  # new_obj_id -> [prev_obj_ids]
        reverse_assignments: dict[int, list[int]] = {}  # prev_obj_id -> [new_obj_ids]

        for overlap, prev_id, new_id in all_pairs:
            # Allow many-to-one for merges and one-to-many for splits
            if new_id not in assignments:
                assignments[new_id] = []
            if prev_id not in reverse_assignments:
                reverse_assignments[prev_id] = []
            assignments[new_id].append(prev_id)
            reverse_assignments[prev_id].append(new_id)
            assigned_new.add(new_id)
            assigned_prev.add(prev_id)

        # Deduplicate assignments
        for key in assignments:
            assignments[key] = list(dict.fromkeys(assignments[key]))
        for key in reverse_assignments:
            reverse_assignments[key] = list(dict.fromkeys(reverse_assignments[key]))

        # Process assignments: detect merges, splits, and simple updates
        new_obj_to_track: dict[int, int] = {}
        processed_new: set[int] = set()

        # Handle merges: new object matched to multiple previous objects
        for new_id, prev_ids in assignments.items():
            if len(prev_ids) <= 1:
                continue
            # Merge: multiple previous -> one new
            surviving_track_id = None
            merged_track_ids = []
            for pid in prev_ids:
                tid = self._obj_to_track.get(pid)
                if tid is not None:
                    if surviving_track_id is None:
                        surviving_track_id = tid
                    else:
                        merged_track_ids.append(tid)

            if surviving_track_id is not None:
                surviving = self.get_track(surviving_track_id)
                if surviving is not None:
                    surviving.add_position(timestamp, new_objects[new_id])
                    new_obj_to_track[new_id] = surviving_track_id
                    for mtid in merged_track_ids:
                        mt = self.get_track(mtid)
                        if mt is not None and mt.status == "active":
                            mt.status = "merged"
                            mt.merged_into = surviving_track_id
                    self._recent_events.append({
                        "event_type": "merge",
                        "timestamp": timestamp.isoformat(),
                        "description": f"Tracks {', '.join(str(t) for t in merged_track_ids)} merged into track {surviving_track_id}",
                        "involved_track_ids": [surviving_track_id] + merged_track_ids,
                    })
            processed_new.add(new_id)

        # Handle splits: one previous object matched to multiple new objects
        for prev_id, new_ids in reverse_assignments.items():
            if len(new_ids) <= 1:
                continue
            unprocessed = [nid for nid in new_ids if nid not in processed_new]
            if len(unprocessed) <= 1:
                continue
            parent_tid = self._obj_to_track.get(prev_id)
            if parent_tid is None:
                continue
            parent_track = self.get_track(parent_tid)
            if parent_track is None:
                continue
            # Parent continues with largest piece
            largest = max(unprocessed, key=lambda nid: new_objects[nid].area_km2)
            parent_track.add_position(timestamp, new_objects[largest])
            new_obj_to_track[largest] = parent_tid
            processed_new.add(largest)
            child_ids = []
            for nid in unprocessed:
                if nid == largest:
                    continue
                child = self._create_track(timestamp, new_objects[nid])
                child.split_from = parent_tid
                new_obj_to_track[nid] = child.track_id
                processed_new.add(nid)
                child_ids.append(child.track_id)
            self._recent_events.append({
                "event_type": "split",
                "timestamp": timestamp.isoformat(),
                "description": f"Track {parent_tid} split into tracks {', '.join(str(c) for c in child_ids)}",
                "involved_track_ids": [parent_tid] + child_ids,
            })

        # Handle simple 1:1 matches
        for new_id, prev_ids in assignments.items():
            if new_id in processed_new:
                continue
            if len(prev_ids) == 1:
                prev_id = prev_ids[0]
                tid = self._obj_to_track.get(prev_id)
                if tid is not None:
                    track = self.get_track(tid)
                    if track is not None and track.status == "active":
                        track.add_position(timestamp, new_objects[new_id])
                        new_obj_to_track[new_id] = tid
                        processed_new.add(new_id)

        # Create new tracks for unmatched new objects
        for new_id, obj in new_objects.items():
            if new_id not in processed_new:
                track = self._create_track(timestamp, obj)
                new_obj_to_track[new_id] = track.track_id

        # Increment missed scans for unmatched active tracks
        matched_track_ids = set(new_obj_to_track.values())
        for track in self._tracks:
            if track.status == "active" and track.track_id not in matched_track_ids:
                track._missed_scans += 1
                if track._missed_scans >= MAX_MISSED_SCANS:
                    track.status = "lost"

        self._obj_to_track = new_obj_to_track
        self._prev_scan = scan

    @property
    def active_tracks(self) -> list[Track]:
        return [t for t in self._tracks if t.status == "active"]

    @property
    def all_tracks(self) -> list[Track]:
        return list(self._tracks)

    @property
    def recent_events(self) -> list[dict]:
        return list(self._recent_events)

    def get_track(self, track_id: int) -> Track | None:
        for t in self._tracks:
            if t.track_id == track_id:
                return t
        return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_tracker.py -v`
Expected: All 8 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/tracker.py tests/unit/test_tracker.py
git commit -m "feat: add storm tracker with merge/split detection and multi-scan trajectories"
```

---

### Task 6: Update Summary with Motion Info

**Files:**
- Modify: `src/summary.py`
- Modify: `tests/unit/test_summary.py`

- [ ] **Step 1: Write the failing tests**

Replace the contents of `tests/unit/test_summary.py` with:

```python
# tests/unit/test_summary.py
from src.summary import generate_summary, km_to_miles
from src.detection import DetectedObject, IntensityLayerData, degrees_to_bearing
from src.motion import MotionVector
from src.tracker import Track


def _make_object(obj_id=1, distance_km=40.2, bearing_deg=270.0, peak_dbz=45.0,
                 peak_label="heavy rain", area_km2=120.5) -> DetectedObject:
    return DetectedObject(
        object_id=obj_id,
        centroid_lat=35.5,
        centroid_lon=-97.3,
        distance_km=distance_km,
        bearing_deg=bearing_deg,
        peak_dbz=peak_dbz,
        peak_label=peak_label,
        area_km2=area_km2,
        layers=[],
    )


def _make_track(track_id=1, obj: DetectedObject | None = None,
                speed_kmh=56.3, heading_label="NE") -> Track:
    track = Track(track_id=track_id, status="active")
    if obj is not None:
        track.current_object = obj
    return track


def test_km_to_miles():
    assert km_to_miles(1.60934) == 1
    assert km_to_miles(0.0) == 0
    assert km_to_miles(100.0) == 62


def test_generate_summary_no_objects():
    text = generate_summary(
        site_id="KTLX",
        site_name="Oklahoma City",
        timestamp="2026-04-08T18:30:00Z",
        objects=[],
    )
    assert text == "Oklahoma City: No significant precipitation detected."


def test_generate_summary_single_object_no_tracks():
    """Phase 1 behavior: no tracks passed, no motion info."""
    obj = _make_object()
    text = generate_summary(
        site_id="KTLX",
        site_name="Oklahoma City",
        timestamp="2026-04-08T18:30:00Z",
        objects=[obj],
    )
    assert "Oklahoma City" in text
    assert "1 rain object" in text
    assert "heavy rain" in text
    assert "25 miles" in text
    assert "W" in text
    assert "47 square miles" in text
    assert "moving" not in text


def test_generate_summary_with_motion():
    """Phase 2: tracks passed, motion info included."""
    obj = _make_object()
    track = _make_track(obj=obj)
    track._motion_override = MotionVector(speed_kmh=56.3, speed_mph=35, heading_deg=45.0, heading_label="NE")
    text = generate_summary(
        site_id="KTLX",
        site_name="Oklahoma City",
        timestamp="2026-04-08T18:30:00Z",
        objects=[obj],
        tracks=[track],
    )
    assert "moving NE at 35 mph" in text


def test_generate_summary_stationary():
    """Stationary track shows 'stationary'."""
    obj = _make_object()
    track = _make_track(obj=obj)
    track._motion_override = MotionVector(speed_kmh=0.0, speed_mph=0, heading_deg=None, heading_label="stationary")
    text = generate_summary(
        site_id="KTLX",
        site_name="Oklahoma City",
        timestamp="2026-04-08T18:30:00Z",
        objects=[obj],
        tracks=[track],
    )
    assert "stationary" in text


def test_generate_summary_with_merge_event():
    obj = _make_object()
    track = _make_track(obj=obj)
    track._motion_override = MotionVector(speed_kmh=56.3, speed_mph=35, heading_deg=45.0, heading_label="NE")
    events = [{"event_type": "merge", "description": "Tracks 2, 3 merged into track 1"}]
    text = generate_summary(
        site_id="KTLX",
        site_name="Oklahoma City",
        timestamp="2026-04-08T18:30:00Z",
        objects=[obj],
        tracks=[track],
        events=events,
    )
    assert "merged" in text.lower()


def test_generate_summary_multiple_objects():
    obj1 = _make_object(obj_id=1, peak_dbz=55.0, peak_label="intense rain", area_km2=200.0)
    obj2 = _make_object(obj_id=2, peak_dbz=30.0, peak_label="moderate rain", area_km2=50.0)
    text = generate_summary(
        site_id="KTLX",
        site_name="Oklahoma City",
        timestamp="2026-04-08T18:30:00Z",
        objects=[obj1, obj2],
    )
    assert "2 rain objects" in text
    assert "intense rain" in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_summary.py -v`
Expected: FAIL — `generate_summary() got an unexpected keyword argument 'tracks'`

- [ ] **Step 3: Write the implementation**

Replace `src/summary.py` with:

```python
# src/summary.py
from src.detection import DetectedObject, degrees_to_bearing
from src.motion import MotionVector

KM_PER_MILE = 1.60934


def km_to_miles(km: float) -> int:
    """Convert kilometers to miles, rounded to nearest whole number."""
    return round(km / KM_PER_MILE)


def km2_to_mi2(km2: float) -> int:
    """Convert square kilometers to square miles, rounded to nearest whole number."""
    return round(km2 / (KM_PER_MILE ** 2))


def _get_motion_for_object(obj: DetectedObject, tracks) -> MotionVector | None:
    """Find the motion vector for a detected object by matching to its track."""
    if tracks is None:
        return None
    for track in tracks:
        if track.current_object is not None and track.current_object.object_id == obj.object_id:
            if hasattr(track, '_motion_override'):
                return track._motion_override
            return track.get_motion()
    return None


def _format_motion(motion: MotionVector | None) -> str:
    """Format motion info for speech."""
    if motion is None:
        return ""
    if motion.heading_label == "stationary":
        return ", stationary"
    if motion.heading_label == "nearly stationary":
        return ", nearly stationary"
    return f", moving {motion.heading_label} at {motion.speed_mph} mph"


def generate_summary(
    site_id: str,
    site_name: str,
    timestamp: str,
    objects: list[DetectedObject],
    tracks=None,
    events: list[dict] | None = None,
) -> str:
    """Generate a speech-ready text summary of detected rain objects.

    Args:
        site_id: Radar site ID.
        site_name: Radar site display name.
        timestamp: Scan timestamp.
        objects: Detected objects sorted by peak_dbz descending.
        tracks: Optional list of Track objects for motion info.
        events: Optional list of recent merge/split event dicts.
    """
    if not objects:
        return f"{site_name}: No significant precipitation detected."

    count = len(objects)
    obj_word = "rain object" if count == 1 else "rain objects"
    strongest = objects[0]
    distance_mi = km_to_miles(strongest.distance_km)
    bearing = degrees_to_bearing(strongest.bearing_deg)
    area_mi2 = km2_to_mi2(strongest.area_km2)

    motion = _get_motion_for_object(strongest, tracks)
    motion_str = _format_motion(motion)

    parts = [
        f"{site_name}: {count} {obj_word} detected. "
        f"Strongest: {strongest.peak_label}, "
        f"{distance_mi} miles {bearing} of the radar{motion_str}."
    ]

    # Add merge/split events
    if events:
        merge_count = sum(1 for e in events if e["event_type"] == "merge")
        split_count = sum(1 for e in events if e["event_type"] == "split")
        if merge_count > 0:
            storms_word = "storm" if merge_count == 1 else "storms"
            parts.append(f" Note: {merge_count} {storms_word} merged in the last scan.")
        if split_count > 0:
            storms_word = "storm" if split_count == 1 else "storms"
            parts.append(f" Note: {split_count} {storms_word} split in the last scan.")

    parts.append(f" Covering approximately {area_mi2} square miles.")

    return "".join(parts)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_summary.py -v`
Expected: All 8 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/summary.py tests/unit/test_summary.py
git commit -m "feat: add motion and merge/split events to speech summary"
```

---

### Task 7: Integrate Buffer & Tracker into Server

**Files:**
- Modify: `src/server.py`
- Modify: `tests/smoke/test_server_smoke.py`

- [ ] **Step 1: Write the failing smoke tests**

Add these tests to the bottom of `tests/smoke/test_server_smoke.py`:

```python
def test_tracks_endpoint_returns_200():
    mock_ref = MagicMock()
    mock_ref.reflectivity = np.full((360, 500), np.nan)
    mock_ref.azimuths = np.linspace(0, 359, 360)
    mock_ref.ranges_m = np.linspace(2000, 250000, 500)
    mock_ref.radar_lat = 35.3331
    mock_ref.radar_lon = -97.2778
    mock_ref.timestamp = "2026-04-08T18:30:00Z"
    mock_ref.elevation_angle = 0.5
    mock_ref.elevation_angles = [0.5]
    with patch("src.server.fetch_scan", return_value="/fake/path"), \
         patch("src.server.extract_reflectivity", return_value=mock_ref):
        resp = client.get("/tracks/KTLX")
    assert resp.status_code == 200
    data = resp.json()
    assert "tracks" in data
    assert "active_count" in data
    assert "recent_events" in data


def test_motion_endpoint_missing_track_returns_404():
    resp = client.get("/motion/KTLX/999")
    assert resp.status_code == 404
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/smoke/test_server_smoke.py::test_tracks_endpoint_returns_200 -v`
Expected: FAIL — no `/tracks/` route

- [ ] **Step 3: Write the implementation**

Replace `src/server.py` with:

```python
# src/server.py
from datetime import datetime
from fastapi import FastAPI, Query, HTTPException
from src.models import (
    RadarSite, ScanMeta, ObjectsResponse, SummaryResponse, RainObject, IntensityLayer,
    TracksResponse, StormTrack, TrackPosition, TrackMotion, TrackEvent,
    TrackDetailResponse, PeakHistoryEntry,
)
from src.sites import geocode_city_state, rank_sites, NEXRAD_SITES
from src.ingest import fetch_scan
from src.parser import extract_reflectivity
from src.detection import detect_objects, detect_objects_with_grid
from src.summary import generate_summary
from src.buffer import ReplayBuffer, BufferedScan
from src.tracker import StormTracker

app = FastAPI(title="ARW - Accessible Radar Workstation", version="0.2.0")

# Module-level state for buffer and tracker
_buffer = ReplayBuffer()
_tracker = StormTracker()


def _find_site_name(site_id: str) -> str:
    """Look up the display name for a NEXRAD site."""
    for site in NEXRAD_SITES:
        if site["site_id"] == site_id.upper():
            return site["name"]
    return site_id


def _parse_datetime(dt_str: str | None) -> datetime | None:
    """Parse an optional datetime query parameter."""
    if dt_str is None:
        return None
    return datetime.fromisoformat(dt_str)


def _ingest_to_buffer(site_id: str, dt: datetime | None = None) -> BufferedScan:
    """Fetch a scan, detect objects, and add to buffer + tracker."""
    filepath = fetch_scan(site_id.upper(), dt)
    ref_data = extract_reflectivity(filepath)
    result = detect_objects_with_grid(
        reflectivity=ref_data.reflectivity,
        azimuths=ref_data.azimuths,
        ranges_m=ref_data.ranges_m,
        radar_lat=ref_data.radar_lat,
        radar_lon=ref_data.radar_lon,
    )
    scan_timestamp = datetime.fromisoformat(ref_data.timestamp) if isinstance(ref_data.timestamp, str) else ref_data.timestamp
    buffered = BufferedScan(
        timestamp=scan_timestamp,
        site_id=site_id.upper(),
        reflectivity_data=ref_data,
        detected_objects=result.objects,
        labeled_grid=result.labeled_grid,
        object_masks=result.object_masks,
    )
    _buffer.add_scan(buffered)
    _tracker.update(buffered)
    return buffered


def _track_to_model(track) -> StormTrack:
    """Convert internal Track to Pydantic StormTrack model."""
    motion = track.get_motion()
    return StormTrack(
        track_id=track.track_id,
        status=track.status,
        positions=[
            TrackPosition(
                timestamp=p.timestamp.isoformat() if isinstance(p.timestamp, datetime) else p.timestamp,
                latitude=p.latitude,
                longitude=p.longitude,
                distance_km=p.distance_km,
                bearing_deg=p.bearing_deg,
            )
            for p in track.positions
        ],
        motion=TrackMotion(
            speed_kmh=motion.speed_kmh,
            speed_mph=motion.speed_mph,
            heading_deg=motion.heading_deg,
            heading_label=motion.heading_label,
        ),
        peak_dbz=track.peak_history[-1].peak_dbz if track.peak_history else 0.0,
        peak_label=track.peak_history[-1].peak_label if track.peak_history else "unknown",
        merged_into=track.merged_into,
        split_from=track.split_from,
        first_seen=track.first_seen.isoformat() if track.first_seen else "",
        last_seen=track.last_seen.isoformat() if track.last_seen else "",
    )


@app.get("/")
def root():
    return {"name": "ARW - Accessible Radar Workstation", "version": "0.2.0"}


@app.get("/sites", response_model=list[RadarSite])
def get_sites(city: str = Query(...), state: str = Query(...)):
    lat, lon = geocode_city_state(city, state)
    ranked = rank_sites(lat, lon)
    return [RadarSite(**site) for site in ranked]


@app.get("/scan/{site_id}", response_model=ScanMeta)
def get_scan(site_id: str, datetime: str | None = Query(None)):
    dt = _parse_datetime(datetime)
    buffered = _ingest_to_buffer(site_id, dt)
    return ScanMeta(
        site_id=site_id.upper(),
        timestamp=buffered.reflectivity_data.timestamp,
        elevation_angles=buffered.reflectivity_data.elevation_angles,
    )


@app.get("/objects/{site_id}", response_model=ObjectsResponse)
def get_objects(site_id: str, datetime: str | None = Query(None)):
    dt = _parse_datetime(datetime)
    buffered = _ingest_to_buffer(site_id, dt)
    rain_objects = [
        RainObject(
            object_id=obj.object_id,
            centroid_lat=obj.centroid_lat,
            centroid_lon=obj.centroid_lon,
            distance_km=obj.distance_km,
            bearing_deg=obj.bearing_deg,
            peak_dbz=obj.peak_dbz,
            peak_label=obj.peak_label,
            area_km2=obj.area_km2,
            layers=[
                IntensityLayer(
                    label=layer.label,
                    min_dbz=layer.min_dbz,
                    max_dbz=layer.max_dbz,
                    area_km2=layer.area_km2,
                )
                for layer in obj.layers
            ],
        )
        for obj in buffered.detected_objects
    ]
    return ObjectsResponse(
        site_id=site_id.upper(),
        timestamp=buffered.reflectivity_data.timestamp,
        object_count=len(rain_objects),
        objects=rain_objects,
    )


@app.get("/summary/{site_id}", response_model=SummaryResponse)
def get_summary(site_id: str, datetime: str | None = Query(None)):
    dt = _parse_datetime(datetime)
    buffered = _ingest_to_buffer(site_id, dt)
    site_name = _find_site_name(site_id)
    text = generate_summary(
        site_id=site_id.upper(),
        site_name=site_name,
        timestamp=buffered.reflectivity_data.timestamp,
        objects=buffered.detected_objects,
        tracks=_tracker.active_tracks,
        events=_tracker.recent_events,
    )
    return SummaryResponse(
        site_id=site_id.upper(),
        timestamp=buffered.reflectivity_data.timestamp,
        text=text,
    )


@app.get("/tracks/{site_id}", response_model=TracksResponse)
def get_tracks(site_id: str, datetime: str | None = Query(None)):
    dt = _parse_datetime(datetime)
    buffered = _ingest_to_buffer(site_id, dt)
    active = _tracker.active_tracks
    events = _tracker.recent_events
    return TracksResponse(
        site_id=site_id.upper(),
        timestamp=buffered.reflectivity_data.timestamp,
        active_count=len(active),
        tracks=[_track_to_model(t) for t in active],
        recent_events=[
            TrackEvent(
                event_type=e["event_type"],
                timestamp=e["timestamp"],
                description=e["description"],
                involved_track_ids=e["involved_track_ids"],
            )
            for e in events
        ],
    )


@app.get("/motion/{site_id}/{track_id}", response_model=TrackDetailResponse)
def get_motion(site_id: str, track_id: int):
    track = _tracker.get_track(track_id)
    if track is None:
        raise HTTPException(status_code=404, detail=f"Track {track_id} not found")
    motion = track.get_motion()
    return TrackDetailResponse(
        track_id=track.track_id,
        status=track.status,
        positions=[
            TrackPosition(
                timestamp=p.timestamp.isoformat() if isinstance(p.timestamp, datetime) else p.timestamp,
                latitude=p.latitude,
                longitude=p.longitude,
                distance_km=p.distance_km,
                bearing_deg=p.bearing_deg,
            )
            for p in track.positions
        ],
        motion=TrackMotion(
            speed_kmh=motion.speed_kmh,
            speed_mph=motion.speed_mph,
            heading_deg=motion.heading_deg,
            heading_label=motion.heading_label,
        ),
        peak_history=[
            PeakHistoryEntry(
                timestamp=p.timestamp.isoformat() if isinstance(p.timestamp, datetime) else p.timestamp,
                peak_dbz=p.peak_dbz,
                peak_label=p.peak_label,
            )
            for p in track.peak_history
        ],
        merged_into=track.merged_into,
        split_from=track.split_from,
        first_seen=track.first_seen.isoformat() if track.first_seen else "",
        last_seen=track.last_seen.isoformat() if track.last_seen else "",
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/smoke/test_server_smoke.py -v`
Expected: All 8 tests PASS (6 existing + 2 new). Note: existing smoke tests need their mocks updated since server.py now uses `detect_objects_with_grid`. If the existing tests break, update them to mock `detect_objects_with_grid` instead of `detect_objects`.

- [ ] **Step 5: Commit**

```bash
git add src/server.py tests/smoke/test_server_smoke.py
git commit -m "feat: integrate buffer and tracker into server with /tracks and /motion endpoints"
```

---

### Task 8: Update E2E Tests for Tracking

**Files:**
- Modify: `tests/e2e/test_full_pipeline.py`

- [ ] **Step 1: Write the new e2e tests**

Add these tests to the bottom of `tests/e2e/test_full_pipeline.py`:

```python
from src.detection import DetectionResult


def _make_detection_result_with_storm() -> tuple:
    """Create a ReflectivityData and matching DetectionResult with synthetic storms."""
    reflectivity = np.full((360, 500), np.nan)
    reflectivity[80:110, 180:220] = 25.0
    reflectivity[85:105, 190:210] = 35.0
    reflectivity[90:100, 195:205] = 50.0
    reflectivity[265:280, 340:360] = 30.0
    reflectivity[268:278, 345:355] = 42.0

    ref_data = ReflectivityData(
        reflectivity=reflectivity,
        azimuths=np.linspace(0, 359, 360),
        ranges_m=np.linspace(2000, 250000, 500),
        radar_lat=35.3331,
        radar_lon=-97.2778,
        elevation_angle=0.5,
        elevation_angles=[0.5, 1.5, 2.4],
        timestamp="2026-04-08T18:30:00Z",
    )
    return ref_data


def test_tracks_endpoint_e2e():
    """Test /tracks endpoint with synthetic data."""
    ref_data = _make_detection_result_with_storm()
    with patch("src.server.fetch_scan", return_value="/fake/path"), \
         patch("src.server.extract_reflectivity", return_value=ref_data):
        resp = client.get("/tracks/KTLX")
    assert resp.status_code == 200
    data = resp.json()
    assert data["active_count"] >= 0
    assert isinstance(data["tracks"], list)
    assert isinstance(data["recent_events"], list)


def test_tracks_accumulate_across_calls():
    """Two calls to /tracks should show tracks with multiple positions."""
    # Reset server state
    import src.server as srv
    srv._buffer = type(srv._buffer)()
    srv._tracker = type(srv._tracker)()

    ref_data1 = ReflectivityData(
        reflectivity=np.full((360, 500), np.nan),
        azimuths=np.linspace(0, 359, 360),
        ranges_m=np.linspace(2000, 250000, 500),
        radar_lat=35.3331,
        radar_lon=-97.2778,
        elevation_angle=0.5,
        elevation_angles=[0.5],
        timestamp="2026-04-08T18:30:00Z",
    )
    ref_data1.reflectivity[85:95, 195:205] = 45.0

    ref_data2 = ReflectivityData(
        reflectivity=np.full((360, 500), np.nan),
        azimuths=np.linspace(0, 359, 360),
        ranges_m=np.linspace(2000, 250000, 500),
        radar_lat=35.3331,
        radar_lon=-97.2778,
        elevation_angle=0.5,
        elevation_angles=[0.5],
        timestamp="2026-04-08T18:35:00Z",
    )
    ref_data2.reflectivity[86:96, 196:206] = 45.0  # Slightly moved

    with patch("src.server.fetch_scan", return_value="/fake/path"), \
         patch("src.server.extract_reflectivity", return_value=ref_data1):
        resp1 = client.get("/tracks/KTLX")
    assert resp1.status_code == 200

    with patch("src.server.fetch_scan", return_value="/fake/path"), \
         patch("src.server.extract_reflectivity", return_value=ref_data2):
        resp2 = client.get("/tracks/KTLX")
    assert resp2.status_code == 200
    data = resp2.json()
    assert data["active_count"] >= 1
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `uv run pytest tests/e2e/ -v`
Expected: All e2e tests PASS.

- [ ] **Step 3: Run the full test suite**

Run: `uv run pytest tests/ -v`
Expected: All tests PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/e2e/test_full_pipeline.py
git commit -m "feat: add e2e tests for tracking across scans"
```

---

### Task 9: Final Verification & Push

- [ ] **Step 1: Run the complete test suite**

Run: `uv run pytest tests/ -v --tb=short`
Expected: All tests PASS.

- [ ] **Step 2: Verify the server starts**

Run: `uv run python -c "from src.server import app; print('Server module loads OK')"`
Expected: `Server module loads OK`

- [ ] **Step 3: Push to remote**

```bash
git push
```

- [ ] **Step 4: Update PROGRESS.md**

```markdown
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
- Full test suite: unit, smoke, and e2e tests all passing

## In Progress
- None

## Next
- Phase 3: Velocity ingestion, velocity region detection
- NVGT frontend integration with the REST API

## Blockers / Decisions
- None
```
