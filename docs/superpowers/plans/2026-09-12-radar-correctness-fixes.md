# Radar Correctness Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix three verified defects that make ARW speak or map wrong information: precipitation evidence silently stripped (defect 1), browsing history resetting live tracking (defect 3), and speech attaching a storm to another storm's track (defect 4).

**Architecture:** Precipitation-band evidence is computed during processing while dual-pol grids still exist and stored on the scan. Scan analysis is separated from tracking so only the newest live volume advances a site's tracker, and each tracked scan carries a snapshot of the tracker state from its own time, which `/tracks` and `/summary` serve. Summary track lookup ignores tracks that missed the scan being described.

**Tech Stack:** Python 3.11 (`.venv`), FastAPI, NumPy, pytest.

**Spec:** `docs/superpowers/specs/2026-09-12-compact-scan-history-design.md` (read **Amendment 1** first; this plan implements A1's defect-1 fix, A2 and A3, and the "browsing history resets live tracking" defect).

## Global Constraints

- Run Python through the project venv: `.venv/Scripts/python.exe`.
- Accuracy outranks delivery speed. Never weaken an assertion to make a test pass; investigate instead.
- Every commit needs green tests for the affected code: at minimum `.venv/Scripts/python.exe -m pytest tests/unit tests/smoke -q` (baseline on 2026-09-12: 400 passed).
- Real-data proofs read Level II volumes already in `cache/` and make no network calls.
- Commit messages end with:
  ```
  Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_011VMHfi1nbuux7187WtgQTt
  ```

## Prerequisite (owner decision, before Task 1)

The working tree has uncommitted work (`README.md`, `src/buffer.py`, `src/detection.py`, `src/map_layer.py`, `src/server.py` and five test files). This plan builds on it. It contains defect 1, which Task 1 fixes. The owner decides how that work is committed before Task 1 starts. Do not commit it on the owner's behalf without being told to.

## File Structure

- `src/map_layer.py`: adds `precipitation_band_keys()` and `compute_precipitation_band_evidence()`; the precipitation builder uses precomputed evidence when present.
- `src/buffer.py`: `BufferedScan` gains `precipitation_band_evidence` and `tracking`; new `TrackingSnapshot`.
- `src/server.py`: `_process_scan_file` becomes analysis-only; new `_track_live_scan`; `/tracks` and `/summary` serve the scan's own snapshot.
- `src/tracker.py`: `last_scan_timestamp` property and `snapshot()`.
- `src/summary.py`: track lookup ignores tracks that missed this scan.
- `src/models.py`: `tracking_context` on `TracksResponse` and `SummaryResponse`.
- Tests: `tests/unit/test_map_layer.py`, `tests/unit/test_summary.py`, `tests/unit/test_live_cache.py`, new `tests/e2e/test_proof_precipitation_evidence.py`.

---

### Task 1: Keep precipitation dual-pol evidence after the grids are released (defect 1)

**Files:**
- Modify: `src/map_layer.py` (after `_band_evidence`, and inside `build_precipitation_field_geojson`)
- Modify: `src/buffer.py` (`BufferedScan`)
- Modify: `src/server.py` (`_process_scan_file`, imports from `src.map_layer`)
- Test: `tests/unit/test_map_layer.py`
- Test: `tests/e2e/test_proof_precipitation_evidence.py` (create)

**Interfaces:**
- Produces: `precipitation_band_keys() -> list[tuple[float, float]]`, `compute_precipitation_band_evidence(sweep) -> dict[tuple[float, float], dict[str, Any]]` in `src.map_layer`; `BufferedScan.precipitation_band_evidence: dict[tuple[float, float], dict[str, Any]] | None = None`.

- [ ] **Step 1: Write the failing unit tests**

Append to `tests/unit/test_map_layer.py` (add the imports at the top of the file if absent):

```python
import dataclasses
import json

from src.contours import exclusive_bands
from src.map_layer import (
    PRECIP_FIELD_LEVELS,
    compute_precipitation_band_evidence,
    precipitation_band_keys,
)


def _dual_pol_precipitation_sweep() -> SweepData:
    reflectivity = np.full((360, 500), np.nan)
    reflectivity[100:160, 200:300] = 25.0
    reflectivity[115:145, 230:270] = 45.0
    rhohv = np.full(reflectivity.shape, 0.99)
    rhohv[115:145, 230:270] = 0.95
    zdr = np.full(reflectivity.shape, 1.25)
    classes = np.ones(reflectivity.shape, dtype=np.int8)  # 1 = precipitation
    return SweepData(
        reflectivity=reflectivity,
        azimuths=np.linspace(0, 359, 360),
        ranges_m=np.linspace(2000, 250000, 500),
        elevation_angle=0.5,
        elevations=np.full(360, 0.5),
        elevation_angles=[0.5],
        radar_lat=35.3331,
        radar_lon=-97.2778,
        radar_alt_m=390.0,
        timestamp="2026-09-12T18:00:00Z",
        rhohv=rhohv,
        zdr=zdr,
        gate_classification=classes,
    )


def _precipitation_scan(sweep: SweepData, evidence=None) -> BufferedScan:
    return BufferedScan(
        timestamp=datetime(2026, 9, 12, 18, 0),
        site_id="KTLX",
        reflectivity_data=sweep,
        detected_objects=[],
        labeled_grid=np.zeros(sweep.reflectivity.shape, dtype=np.int32),
        object_masks={},
        precipitation_band_evidence=evidence,
    )


def test_precipitation_band_keys_match_the_contoured_bands():
    sweep = _dual_pol_precipitation_sweep()
    bands = exclusive_bands(sweep.reflectivity, PRECIP_FIELD_LEVELS, sweep)
    assert precipitation_band_keys() == list(bands.keys())


def test_precipitation_layer_is_identical_from_precomputed_evidence():
    sweep = _dual_pol_precipitation_sweep()
    expected = build_precipitation_field_geojson(_precipitation_scan(sweep))
    evidence = compute_precipitation_band_evidence(sweep)
    released = dataclasses.replace(sweep, rhohv=None, zdr=None, gate_classification=None)

    actual = build_precipitation_field_geojson(_precipitation_scan(released, evidence))

    assert json.dumps(actual, sort_keys=True) == json.dumps(expected, sort_keys=True)
    assert expected["features"], "fixture must produce precipitation features"
    assert all(f["properties"]["median_rhohv"] is not None for f in expected["features"])


def test_released_dual_pol_without_precomputed_evidence_is_the_defect():
    """Documents defect 1: this is what the live pipeline published."""
    sweep = _dual_pol_precipitation_sweep()
    released = dataclasses.replace(sweep, rhohv=None, zdr=None, gate_classification=None)
    layer = build_precipitation_field_geojson(_precipitation_scan(released))
    assert all(f["properties"]["median_rhohv"] is None for f in layer["features"])


def test_precomputed_evidence_is_not_shared_with_published_features():
    sweep = _dual_pol_precipitation_sweep()
    evidence = compute_precipitation_band_evidence(sweep)
    layer = build_precipitation_field_geojson(_precipitation_scan(sweep, evidence))
    layer["features"][0]["properties"]["class_fractions"]["precipitation"] = -1.0
    assert all(
        band["class_fractions"].get("precipitation") != -1.0 for band in evidence.values()
    )
```

`datetime` must be imported in that test file (`from datetime import datetime`). Check the existing imports, and add only what's missing.

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_map_layer.py -k "precipitation_band_keys or precomputed_evidence or released_dual_pol" -v`
Expected: FAIL with `ImportError: cannot import name 'compute_precipitation_band_evidence'` (the whole module fails to import).

- [ ] **Step 3: Add the field to `BufferedScan`**

In `src/buffer.py`, change the typing import and add the field at the end of `BufferedScan`:

```python
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
from typing import Any
```

```python
    echo_advisory: EchoAdvisory | None = None
    # Evidence for each precipitation band (see
    # src.map_layer.compute_precipitation_band_evidence), computed while the
    # dual-pol grids still exist.  The live pipeline releases those grids
    # before the precipitation layer is rendered.
    precipitation_band_evidence: dict[tuple[float, float], dict[str, Any]] | None = None
```

- [ ] **Step 4: Add the evidence functions and use them in the builder**

In `src/map_layer.py`, directly after `_band_evidence`:

```python
def precipitation_band_keys() -> list[tuple[float, float]]:
    """The (lower, upper) bands the precipitation layer contours.

    Mirrors `src.contours._raw_bands`: consecutive sorted levels, the last
    band open-ended.
    """
    ordered = sorted(float(level) for level in PRECIP_FIELD_LEVELS)
    return [
        (lower, ordered[index + 1] if index + 1 < len(ordered) else float("inf"))
        for index, lower in enumerate(ordered)
    ]


def compute_precipitation_band_evidence(sweep) -> dict[tuple[float, float], dict[str, Any]]:
    """Evidence for every precipitation band, computed while dual-pol exists.

    `_band_evidence` depends only on the sweep and the band's own gates, so
    this is exactly what the layer would compute from the full fields.  It
    lets the live pipeline release the RhoHV/ZDR/class grids before the
    layer is rendered without the layer losing its evidence.
    """
    return {key: _band_evidence(sweep, *key) for key in precipitation_band_keys()}
```

In `build_precipitation_field_geojson`, after `gate_area = _gate_area_grid(sweep)` add:

```python
    precomputed_evidence = getattr(scan, "precipitation_band_evidence", None)
```

and replace `properties.update(_band_evidence(sweep, lower, upper))` with:

```python
        evidence = (
            precomputed_evidence[(lower, upper)]
            if precomputed_evidence is not None
            else _band_evidence(sweep, lower, upper)
        )
        # Copy the nested dict: published features must never alias the
        # scan's stored evidence.
        properties.update({**evidence, "class_fractions": dict(evidence["class_fractions"])})
```

- [ ] **Step 5: Run the unit tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_map_layer.py -v`
Expected: all PASS.

- [ ] **Step 6: Compute evidence in the live pipeline before release**

In `src/server.py`, add `compute_precipitation_band_evidence` to the existing `from src.map_layer import (...)` list. In `_process_scan_file`, directly after the `refresh_quality_advisory(...)` call and **before** the block that sets `ref_data.velocity = None` and the other fields to None, add:

```python
    # The precipitation layer is rendered later, after the dual-pol grids
    # below are released.  Its per-band evidence depends only on this sweep
    # and each band's own gates, so compute it now, exactly.
    precipitation_band_evidence = compute_precipitation_band_evidence(ref_data)
```

and with the other `buffered.` assignments at the end of the function:

```python
    buffered.precipitation_band_evidence = precipitation_band_evidence
```

- [ ] **Step 7: Write the real-data proof**

Create `tests/e2e/test_proof_precipitation_evidence.py`:

```python
"""Proof (defect 1, 2026-09-12): the live pipeline's precipitation layer keeps
its dual-polarization evidence after the pipeline releases the dual-pol grids.

Before the fix, `_process_scan_file` set rhohv/zdr/gate_classification to
None and the layer, rendered later from that scan, reported every band as
median_rhohv None / "uncertain".  This compares the layer the live pipeline
publishes with the layer built from the same sweep before release.
"""

import dataclasses
import json

import pytest

import src.server as server
from src.buffer import BufferedScan
from src.map_layer import build_precipitation_field_geojson

VOLUMES = [
    ("KTLX", "cache/KTLX/KTLX20260410_224209_V06"),
    ("KEMX", "cache/KEMX/KEMX20260712_022646_V06"),
]


@pytest.fixture(autouse=True)
def _clean_server_state():
    yield
    with server._state_lock:
        server._buffers.clear()
        server._trackers.clear()


@pytest.mark.parametrize("site_id,path", VOLUMES)
def test_precipitation_evidence_survives_dual_pol_release(monkeypatch, site_id, path):
    captured = {}
    original = server.refresh_quality_advisory

    def capture(*args, **kwargs):
        sweep, quality, advisory = original(*args, **kwargs)
        # A shallow copy keeps the grids even after the server sets the
        # original object's attributes to None.
        captured["sweep"] = dataclasses.replace(sweep)
        return sweep, quality, advisory

    monkeypatch.setattr(server, "refresh_quality_advisory", capture)
    released = server._process_scan_file(site_id, path)

    assert released.reflectivity_data.rhohv is None
    assert released.reflectivity_data.gate_classification is None
    unreleased = BufferedScan(
        timestamp=released.timestamp,
        site_id=released.site_id,
        reflectivity_data=captured["sweep"],
        detected_objects=released.detected_objects,
        labeled_grid=released.labeled_grid,
        object_masks=released.object_masks,
    )
    expected = build_precipitation_field_geojson(unreleased)
    actual = build_precipitation_field_geojson(released)

    assert json.dumps(actual, sort_keys=True) == json.dumps(expected, sort_keys=True)
    assert expected["features"]
    for feature in actual["features"]:
        properties = feature["properties"]
        assert properties["median_rhohv"] is not None, properties["id"]
        assert properties["median_zdr"] is not None, properties["id"]
        assert properties["class_fractions"], properties["id"]
```

- [ ] **Step 8: Run the proof**

Run: `.venv/Scripts/python.exe -m pytest tests/e2e/test_proof_precipitation_evidence.py -v`
Expected: 2 PASS (several minutes: contouring runs twice per volume). To confirm the proof detects the defect, temporarily comment out the `buffered.precipitation_band_evidence = ...` line, rerun (expected FAIL on `median_rhohv is not None`), then restore it.

- [ ] **Step 9: Run the suite and commit**

Run: `.venv/Scripts/python.exe -m pytest tests/unit tests/smoke -q`
Expected: all pass.

```bash
git add src/map_layer.py src/buffer.py src/server.py tests/unit/test_map_layer.py tests/e2e/test_proof_precipitation_evidence.py
git commit -m "$(cat <<'EOF'
Keep precipitation dual-pol evidence after releasing grids

The live pipeline released RhoHV, ZDR and gate class before the
precipitation layer was rendered, so every band reported no RhoHV and
"uncertain". Band evidence is now computed during processing and used
by the layer; output is identical to building from the full fields.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_011VMHfi1nbuux7187WtgQTt
EOF
)"
```

---

### Task 2: Never attach a storm to a track that missed this scan (defect 4)

**Files:**
- Modify: `src/summary.py:20-39` (`_get_motion_for_object`, `_get_track_for_object`)
- Test: `tests/unit/test_summary.py`

**Interfaces:**
- Produces: `src.summary._seen_this_scan(track) -> bool`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_summary.py`:

```python
from datetime import datetime, timedelta

import numpy as np

from src.detection import DetectedObject
from src.summary import _get_motion_for_object, _get_track_for_object
from src.tracker import StormTracker
from src.tracking.types import Track
from tests.unit.test_tracking_association import _make_object, _make_scan


def _storm(object_id: int, lon: float, peak: float) -> DetectedObject:
    return DetectedObject(object_id, 35.0, lon, 40.0, 90.0, peak, "heavy precipitation", 100.0)


def test_track_lookup_ignores_a_track_that_missed_this_scan():
    t1 = datetime(2026, 9, 12, 18, 0)
    stale = Track(track_id=1, status="active")
    stale.add_position(t1, _storm(1, -98.5, 55.0))
    stale._missed_scans = 1
    stale._motion_override = "storm A motion"
    current = Track(track_id=2, status="active")
    current.add_position(t1 + timedelta(minutes=5), _storm(1, -96.0, 45.0))
    current._motion_override = "storm B motion"
    storm_b = _storm(1, -96.0, 45.0)

    assert _get_track_for_object(storm_b, [stale, current]) is current
    assert _get_motion_for_object(storm_b, [stale, current]) == "storm B motion"


def test_tracker_scenario_where_a_missed_storm_shares_the_object_number():
    """Reproduces defect 4 through the real tracker.

    Storm A is not detected in the second scan, so storm B -- the only
    object -- is numbered 1, which was A's number in the first scan.
    """
    shape = (360, 500)

    def mask(row: int, col: int) -> np.ndarray:
        grid = np.zeros(shape, dtype=bool)
        grid[row:row + 10, col:col + 10] = True
        return grid

    t1 = datetime(2026, 4, 8, 18, 30)
    tracker = StormTracker()
    tracker.update(_make_scan(
        "KTLX", t1,
        [_make_object(1, 35.0, -98.5, 55.0), _make_object(2, 35.0, -96.0, 45.0)],
        {1: mask(50, 50), 2: mask(200, 400)},
    ))
    tracker.update(_make_scan(
        "KTLX", t1 + timedelta(minutes=5),
        [_make_object(1, 35.0, -96.0, 45.0)],
        {1: mask(200, 400)},
    ))

    storm_b = _make_object(1, 35.0, -96.0, 45.0)
    track = _get_track_for_object(storm_b, tracker.active_tracks)
    assert track is not None
    assert track.current_object.centroid_lon == -96.0
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_summary.py -k "missed" -v`
Expected: both FAIL. The first returns `stale`. The second finds the track at lon -98.5.

- [ ] **Step 3: Fix the lookups**

In `src/summary.py`, add above `_get_motion_for_object`:

```python
def _seen_this_scan(track) -> bool:
    """Whether this track was matched in the scan being summarized.

    A track that missed the scan stays active while holding its previous
    scan's object.  Object numbers are reassigned every scan, so that stale
    object's number can now belong to a different storm.
    """
    return getattr(track, "_missed_scans", 0) == 0
```

and in both `_get_motion_for_object` and `_get_track_for_object` change the match condition to:

```python
        if (
            track.current_object is not None
            and _seen_this_scan(track)
            and track.current_object.object_id == obj.object_id
        ):
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_summary.py -v`
Expected: all PASS.

- [ ] **Step 5: Run the suite and commit**

Run: `.venv/Scripts/python.exe -m pytest tests/unit tests/smoke -q`
Expected: all pass.

```bash
git add src/summary.py tests/unit/test_summary.py
git commit -m "$(cat <<'EOF'
Stop speaking a storm with another storm's track

A track that missed a scan kept its old object number, which detection
had reassigned to a different storm, and the summary matched it first.
Only tracks seen in the summarized scan are now matched.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_011VMHfi1nbuux7187WtgQTt
EOF
)"
```

---

### Task 3: Only the newest live volume advances tracking; scans serve their own tracking state (defect 3)

**Files:**
- Modify: `src/buffer.py` (new `TrackingSnapshot`; `BufferedScan.tracking`)
- Modify: `src/tracker.py` (`last_scan_timestamp`, `snapshot()`)
- Modify: `src/server.py` (`_process_scan_file`, new `_track_live_scan`, `_ingest_to_buffer`, `get_tracks`, `get_summary`)
- Modify: `src/models.py` (`TracksResponse`, `SummaryResponse`)
- Test: `tests/unit/test_live_cache.py`, `tests/unit/test_tracker.py`

**Interfaces:**
- Consumes: `BufferedScan` from Task 1.
- Produces: `src.buffer.TrackingSnapshot(active_tracks: list, recent_events: list[dict], continuity_rebuilt: bool = False)`; `BufferedScan.tracking: TrackingSnapshot | None = None`; `StormTracker.last_scan_timestamp -> datetime | None`; `StormTracker.snapshot() -> TrackingSnapshot`; `src.server._track_live_scan(site_id: str, buffered: BufferedScan) -> None`; `TracksResponse.tracking_context` and `SummaryResponse.tracking_context` (`"tracked"` or `"unavailable"`).

- [ ] **Step 1: Write the failing tracker tests**

Append to `tests/unit/test_tracker.py`:

```python
def test_snapshot_is_independent_of_later_tracker_updates():
    tracker = StormTracker()
    t1 = datetime(2026, 9, 12, 18, 0)
    tracker.update(_make_scan("KTLX", t1, [_make_object(1, 35.3, -97.3)]))
    snapshot = tracker.snapshot()
    tracker.update(_make_scan("KTLX", t1 + timedelta(minutes=5), [_make_object(1, 35.3, -97.3)]))

    assert [len(track.positions) for track in snapshot.active_tracks] == [1]
    assert [len(track.positions) for track in tracker.active_tracks] == [2]
    assert snapshot.continuity_rebuilt is False


def test_last_scan_timestamp_follows_updates():
    tracker = StormTracker()
    assert tracker.last_scan_timestamp is None
    t1 = datetime(2026, 9, 12, 18, 0)
    tracker.update(_make_scan("KTLX", t1, [_make_object(1, 35.3, -97.3)]))
    assert tracker.last_scan_timestamp == t1
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_tracker.py -k "snapshot_is_independent or last_scan_timestamp" -v`
Expected: FAIL with `AttributeError: 'StormTracker' object has no attribute 'snapshot'`.

- [ ] **Step 3: Add `TrackingSnapshot` and the tracker methods**

In `src/buffer.py`, above `BufferedScan`:

```python
@dataclass
class TrackingSnapshot:
    """Tracker state as it stood immediately after one scan was tracked.

    Served with that scan, so a scan is never described with tracks from a
    later scan.  `active_tracks` holds deep copies of `Track` objects (typed
    loosely to avoid importing the tracker here).
    """
    active_tracks: list[Any]
    recent_events: list[dict]
    continuity_rebuilt: bool = False
```

and at the end of `BufferedScan`:

```python
    # Set only when the live tracker tracked this scan.  None means the scan
    # came from the historical path and has no tracking context.
    tracking: TrackingSnapshot | None = None
```

In `src/tracker.py`, change the import to `from src.buffer import BufferedScan, TrackingSnapshot`, add `from copy import deepcopy`, and add to `StormTracker` (after `recent_events`):

```python
    @property
    def last_scan_timestamp(self) -> datetime | None:
        """Timestamp of the most recent scan this tracker processed."""
        return self._prev_scan.timestamp if self._prev_scan is not None else None

    def snapshot(self) -> TrackingSnapshot:
        """Deep copy of the state a consumer needs to describe the latest scan."""
        return TrackingSnapshot(
            active_tracks=deepcopy(self.active_tracks),
            recent_events=deepcopy(self._recent_events),
        )
```

- [ ] **Step 4: Run the tracker tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_tracker.py -v`
Expected: all PASS.

- [ ] **Step 5: Write the failing server tests**

Append to `tests/unit/test_live_cache.py`:

```python
from datetime import datetime, timedelta

from fastapi.testclient import TestClient

from tests.unit.test_tracker import _make_object, _make_scan

T0 = datetime(2026, 9, 12, 18, 0)


def _scripted_site(monkeypatch, scans_by_path, live_paths):
    live_order = iter(live_paths)
    monkeypatch.setattr(
        server,
        "fetch_scan",
        lambda site_id, dt=None: "/old" if dt is not None else next(live_order),
    )
    monkeypatch.setattr(server, "_process_scan_file", lambda site_id, path: scans_by_path[path])


def _two_live_one_old():
    return {
        "/live-1": _make_scan("KTLX", T0, [_make_object(1, 35.3, -97.3)]),
        "/live-2": _make_scan("KTLX", T0 + timedelta(minutes=5), [_make_object(1, 35.3, -97.3)]),
        "/old": _make_scan("KTLX", T0 - timedelta(hours=1), [_make_object(1, 35.0, -98.0)]),
    }


def test_historical_request_never_advances_or_resets_live_tracking(monkeypatch):
    _clear_live_state()
    _scripted_site(monkeypatch, _two_live_one_old(), ["/live-1", "/live-2"])
    try:
        server._ingest_to_buffer("KTLX")
        server._ingest_to_buffer("KTLX")
        tracker = server._site_tracker("KTLX")
        live_state = [(track.track_id, len(track.positions)) for track in tracker.active_tracks]
        assert live_state == [(1, 2)]

        old = server._ingest_to_buffer("KTLX", T0 - timedelta(hours=1))

        assert old.tracking is None
        assert [(t.track_id, len(t.positions)) for t in tracker.active_tracks] == live_state
        assert tracker.last_scan_timestamp == T0 + timedelta(minutes=5)
    finally:
        _clear_live_state()


def test_live_volume_older_than_the_tracked_one_is_not_tracked(monkeypatch):
    _clear_live_state()
    scans = _two_live_one_old()
    _scripted_site(monkeypatch, scans, ["/live-2", "/live-1"])
    try:
        server._ingest_to_buffer("KTLX")
        stale_live = server._ingest_to_buffer("KTLX")
        assert stale_live.tracking is None
        assert server._site_tracker("KTLX").last_scan_timestamp == T0 + timedelta(minutes=5)
    finally:
        _clear_live_state()


def test_tracks_for_an_older_scan_come_from_that_scan(monkeypatch):
    _clear_live_state()
    _scripted_site(monkeypatch, _two_live_one_old(), ["/live-1", "/live-2"])
    try:
        first = server._ingest_to_buffer("KTLX")
        server._ingest_to_buffer("KTLX")
        monkeypatch.setattr(server, "_live_or_ingest", lambda site_id, dt=None: (first, False))

        data = TestClient(server.app).get(
            "/tracks/KTLX", params={"datetime": "2026-09-12T18:00:00"}
        ).json()

        assert data["tracking_context"] == "tracked"
        assert [len(track["positions"]) for track in data["tracks"]] == [1]
    finally:
        _clear_live_state()


def test_untracked_scan_reports_no_tracks_and_unavailable_context(monkeypatch):
    _clear_live_state()
    untracked = _make_scan("KTLX", T0 - timedelta(hours=1), [_make_object(1, 35.0, -98.0)])
    monkeypatch.setattr(server, "_live_or_ingest", lambda site_id, dt=None: (untracked, False))
    try:
        client = TestClient(server.app)
        tracks = client.get("/tracks/KTLX", params={"datetime": "2026-09-12T17:00:00"}).json()
        summary = client.get("/summary/KTLX", params={"datetime": "2026-09-12T17:00:00"}).json()

        assert tracks["tracking_context"] == "unavailable"
        assert tracks["tracks"] == []
        assert tracks["recent_events"] == []
        assert summary["tracking_context"] == "unavailable"
    finally:
        _clear_live_state()
```

In the existing `test_different_sites_can_ingest_concurrently`, the fake `slow_process` returns a `SimpleNamespace` that cannot be tracked. Add this line next to its other `monkeypatch.setattr` calls:

```python
    monkeypatch.setattr(server, "_track_live_scan", lambda *_args: None)
```

- [ ] **Step 6: Run them to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_live_cache.py -v`
Expected: the four new tests FAIL. For example, the historical request resets the tracker, or `KeyError: 'tracking_context'`. `test_different_sites_can_ingest_concurrently` fails with `AttributeError` because `_track_live_scan` doesn't exist yet.

- [ ] **Step 7: Separate analysis from tracking in the server**

In `src/server.py`:

1. In `_process_scan_file`, delete these lines:

```python
    tracker = _site_tracker(site_id)
    _site_buffer(site_id).add_scan(buffered)
    tracker.update(buffered)
    for obj in annotated_objects:
        obj.temporal_status = tracker.temporal_status_for_current_object(obj.object_id)
```

and change its docstring to:

```python
    """Run the analysis pipeline for one already-selected volume.

    Analysis only: this never touches tracking state.  The live path decides
    separately whether the result advances the site's tracker.
    """
```

2. Add after `_process_scan_file`:

```python
def _track_live_scan(site_id: str, buffered: BufferedScan) -> None:
    """Advance a site's live tracker with its newest volume.

    Only a volume newer than the last one tracked is accepted.  Tracking an
    older volume would make the tracker see a negative time gap and discard
    every live track for the site.
    """
    tracker = _site_tracker(site_id)
    last_tracked = tracker.last_scan_timestamp
    if last_tracked is not None and buffered.timestamp <= last_tracked:
        return
    _site_buffer(site_id).add_scan(buffered)
    tracker.update(buffered)
    for obj in buffered.detected_objects:
        obj.temporal_status = tracker.temporal_status_for_current_object(obj.object_id)
    buffered.tracking = tracker.snapshot()
```

3. In `_ingest_to_buffer`, in the cached branch replace

```python
                if dt is None:
                    with _state_lock:
```

with

```python
                if dt is None:
                    if cached.tracking is None:
                        _track_live_scan(normalized_site, cached)
                    with _state_lock:
```

and directly after `buffered = _process_scan_file(normalized_site, filepath)` add:

```python
        if dt is None:
            _track_live_scan(normalized_site, buffered)
```

4. Add below `_track_live_scan`:

```python
def _scan_tracking(buffered) -> TrackingSnapshot | None:
    tracking = getattr(buffered, "tracking", None)
    return tracking if isinstance(tracking, TrackingSnapshot) else None
```

and change the import to `from src.buffer import ReplayBuffer, BufferedScan, TrackingSnapshot`.

5. In `get_summary`, replace the `tracks=`/`events=` arguments and the response:

```python
    tracking = _scan_tracking(buffered)
    text = generate_summary(
        site_id=site_id.upper(),
        site_name=site_name,
        timestamp=buffered.reflectivity_data.timestamp,
        objects=buffered.detected_objects,
        tracks=tracking.active_tracks if tracking is not None else [],
        events=tracking.recent_events if tracking is not None else [],
    )
    return SummaryResponse(
        site_id=site_id.upper(),
        timestamp=buffered.reflectivity_data.timestamp,
        text=text,
        tracking_context="tracked" if tracking is not None else "unavailable",
    )
```

6. In `get_tracks`, replace the `tracker = ...`, `active = ...` and `events = ...` lines with:

```python
    tracking = _scan_tracking(buffered)
    active = tracking.active_tracks if tracking is not None else []
    events = tracking.recent_events if tracking is not None else []
```

and add `tracking_context="tracked" if tracking is not None else "unavailable",` to the `TracksResponse(...)` arguments.

In `src/models.py`, add to both `SummaryResponse` and `TracksResponse`:

```python
    # "tracked": served with the tracker state from this scan's own time.
    # "unavailable": a historical scan the live tracker never processed.
    tracking_context: str = "unavailable"
```

- [ ] **Step 8: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_live_cache.py tests/unit/test_tracker.py -v`
Expected: all PASS.

- [ ] **Step 9: Remove the now-unneeded proof cleanup, then run the full suite**

`_process_scan_file` no longer touches tracking, so delete the `_clean_server_state` fixture (and nothing else) from `tests/e2e/test_proof_precipitation_evidence.py`.

Run: `.venv/Scripts/python.exe -m pytest tests/unit tests/smoke tests/e2e/test_full_pipeline.py tests/e2e/test_proof_precipitation_evidence.py -q`
Expected: all pass. If a smoke or e2e test now sees no tracks because repeated calls reuse one timestamp, that is correct behavior. Change its fixture to give the second scan a later timestamp. Never remove the assertion.

- [ ] **Step 10: Commit**

```bash
git add src/buffer.py src/tracker.py src/server.py src/models.py tests/unit/test_live_cache.py tests/unit/test_tracker.py tests/e2e/test_proof_precipitation_evidence.py
git commit -m "$(cat <<'EOF'
Keep browsing history from resetting live tracking

Every request, including historical ones, advanced the site's single
tracker, so viewing an older scan reset every live track. Analysis is
now separate from tracking: only a newer live volume is tracked. Each
tracked scan carries its own tracker snapshot, and /tracks and /summary
report tracking_context "unavailable" for untracked scans instead of
showing the live tracker's later state.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_011VMHfi1nbuux7187WtgQTt
EOF
)"
```

---

### Task 4: Live check and progress record

**Files:**
- Modify: `PROGRESS.md`

- [ ] **Step 1: Live data check against real NEXRAD**

Start the server in the background: `.venv/Scripts/python.exe -m uvicorn src.server:app --port 8000`. Then, for one active site (KTLX, or a site with current weather):

```bash
curl -s "http://localhost:8000/summary/KTLX"
curl -s "http://localhost:8000/tracks/KTLX" | .venv/Scripts/python.exe -c "import json,sys; d=json.load(sys.stdin); print(d['tracking_context'], d['active_count'])"
curl -s "http://localhost:8000/map/precipitation.geojson?latitude=35.33&longitude=-97.28" | .venv/Scripts/python.exe -c "import json,sys; d=json.load(sys.stdin); print([(f['properties']['min_dbz'], f['properties']['median_rhohv']) for f in d['features']])"
```

Expected: the summary text is returned; `/tracks` reports `tracked`; every precipitation feature has a non-null `median_rhohv` (if the radar has precipitation). Then request `/tracks/KTLX?datetime=<a time about one hour earlier>` and re-request `/tracks/KTLX`. Expected: the historical response says `unavailable`, and the live response afterwards still lists the same track IDs. Stop the server.

- [ ] **Step 2: Update PROGRESS.md**

Add a dated section recording defects 1, 3 and 4 as fixed with their commits and proofs, the live check results, and that the compact scan history plan (`docs/superpowers/plans/2026-09-12-compact-scan-history.md`) is next.

- [ ] **Step 3: Commit**

```bash
git add PROGRESS.md
git commit -m "$(cat <<'EOF'
Record radar correctness fixes in progress notes

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_011VMHfi1nbuux7187WtgQTt
EOF
)"
```
