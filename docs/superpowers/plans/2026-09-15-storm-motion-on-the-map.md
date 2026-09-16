# Storm Motion on the Map Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Audiom radar map convey where a storm is going, by carrying the motion ARW already computes through to the map layer's properties and spoken description.

**Architecture:** ARW already computes motion in the tracking engine (`MotionVector`, reached via `Track.get_motion()`), but `src/map_layer.py` only ever receives `(BufferedScan, DetectedObject)` and has no access to tracks. Rather than threading tracks into the map layer, follow the established pattern used by `rotation`, `temporal_status` and `class_fractions`: the history store attaches the analysis result to the `DetectedObject`, and `map_layer` reads it via the object. This adds one dataclass field, one tracker lookup mirroring `temporal_status_for_current_object`, one assignment in the store, and read-only changes in `map_layer`.

**Tech Stack:** Python 3.11+, pytest, uv. No new dependencies. No network calls.

**Spec:** `docs/superpowers/specs/2026-09-15-level3-companion-ingest-design.md` — section "What this changes about the map", gap **A**, and Decision 5, which records that this work is independent of Level III and should proceed separately.

## Global Constraints

- **Only the Ingest Manager makes network calls** (CLAUDE.md). This plan makes none; no task may add one.
- **Run tests before committing.** No commit without green tests for the affected code (CLAUDE.md).
- **Commit frequently and sensibly** — one commit per task (CLAUDE.md).
- **Speak nothing you cannot back.** House precedent is commit `83aa64f` "Speak no rotation until a level holds up live". Motion is stated in the spoken description only when it is known and not downgraded; otherwise the description says nothing about motion. Properties are emitted as `null`, exactly as the `rotation_*` properties already are.
- **Do not change footprint geometry.** This plan touches properties and description text only. `shape_truth` rates must be untouched because no shape is touched.
- **Test commands:** `uv run pytest <path> -v` from the repo root.

---

## File Structure

| File | Change | Responsibility |
|---|---|---|
| `src/detection.py` | Modify (~line 61) | Add the `motion` field to the `DetectedObject` dataclass, alongside `rotation` and `temporal_status`. |
| `src/tracker.py` | Modify (after line 905) | Add `motion_for_current_object`, mirroring `temporal_status_for_current_object`: the canonical object-id → motion lookup. |
| `src/history/store.py` | Modify (lines 110, 203) | Attach motion to each object during live association, alongside the existing `temporal_status` assignment. |
| `src/map_layer.py` | Modify (lines 118, 348, 499) | Emit motion properties, speak motion in the description, advertise speed as a sonification dimension. |
| `tests/unit/test_tracker.py` | Modify | Cover the new tracker lookup. |
| `tests/unit/test_history_store.py` | Modify | Cover attachment to the object. |
| `tests/unit/test_map_layer.py` | Modify | Cover emitted properties, description text, and metadata. |

Tasks 1–5 are sequential: each depends on the field or method added by the one before.

---

### Task 1: Motion lookup on the tracker and the object field

**Files:**
- Modify: `src/detection.py:61` (after the `temporal_status` field)
- Modify: `src/tracker.py:905` (after `temporal_status_for_current_object`)
- Test: `tests/unit/test_tracker.py`

**Interfaces:**
- Consumes: `StormTracker._obj_to_track: dict[int, int]`, `StormTracker.get_track(track_id)`, `Track.get_motion()` (assigned at `src/tracker.py:908`), `MotionVector` from `src.tracking.motion`.
- Produces: `DetectedObject.motion: MotionVector | None = None`, and `StormTracker.motion_for_current_object(object_id: int) -> MotionVector | None`. Tasks 2–5 rely on both names exactly as written.

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_tracker.py`. This reuses the existing `_make_object` and `_make_scan` helpers already at the top of that file, and mirrors the setup of `test_tracker_matches_overlapping_objects` (around line 85).

```python
def test_motion_for_current_object_returns_track_motion():
    tracker = StormTracker()
    t1 = datetime(2026, 4, 8, 18, 30)
    t2 = t1 + timedelta(minutes=5)
    obj1 = _make_object(1, 35.0, -97.5)
    obj2 = _make_object(1, 35.05, -97.5)
    mask1 = np.zeros((360, 500), dtype=bool)
    mask1[85:95, 195:205] = True
    mask2 = np.zeros((360, 500), dtype=bool)
    mask2[86:96, 196:206] = True
    tracker.update(_make_scan("KTLX", t1, [obj1], masks={1: mask1}))
    tracker.update(_make_scan("KTLX", t2, [obj2], masks={1: mask2}))

    motion = tracker.motion_for_current_object(obj2.object_id)

    assert motion is not None
    assert motion.heading_label


def test_motion_for_current_object_returns_none_for_untracked_object():
    tracker = StormTracker()
    assert tracker.motion_for_current_object(999) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_tracker.py::test_motion_for_current_object_returns_track_motion tests/unit/test_tracker.py::test_motion_for_current_object_returns_none_for_untracked_object -v`

Expected: FAIL — `AttributeError: 'StormTracker' object has no attribute 'motion_for_current_object'`

- [ ] **Step 3: Add the `motion` field to `DetectedObject`**

In `src/detection.py`, directly after the `temporal_status: str = "not_checked"` field (line 61), add:

```python
    # Set by live cross-scan association, from this object's own track. None
    # when the object has no track yet, or when the track cannot state a
    # motion. Never inferred from a neighbouring storm.
    motion: "MotionVector | None" = None
```

The annotation is a string, so no import is needed at runtime — matching how `rotation: "RotationSignature | None"` is already declared on line 50.

- [ ] **Step 4: No tracker change — the lookup already exists**

**Corrected during execution (2026-09-15).** `StormTracker.motion_for_current_object` already exists at `src/tracker.py:888`, directly above `temporal_status_for_current_object`. It had **zero callers in `src/` and zero tests** — a bridge that was built and never wired up. Do not add a second one.

```python
    def motion_for_current_object(self, object_id: int):
        """Return the matched track's current motion, if one exists."""
        track_id = self._obj_to_track.get(object_id)
        track = self.get_track(track_id) if track_id is not None else None
        return track.last_motion if track is not None else None
```

It differs from `src/summary.py:_get_motion_for_object` in two ways, both verified harmless:

- It reads `track.last_motion` rather than `track.get_motion()`. But `Track.get_motion` *is* `_get_track_motion` (`src/tracker.py:147`), which returns `last_motion if last_motion is not None else unknown_motion()`. The only difference is `None` versus an `unknown_motion()` vector — and `None` is preferable here, because Task 3 maps it to clean null properties and Task 4 already declines to speak an `"unknown"` heading.
- It does not honour `_motion_override`. That attribute is set **only in tests** (`tests/unit/test_summary.py`, `tests/unit/test_live_replay_contracts.py`); nothing in `src/` sets it. In production the two functions therefore agree, so the map cannot contradict the spoken summary.

The two tests added in Step 1 stay: they are the first coverage this method has ever had, and Task 2 is about to make it load-bearing.

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_tracker.py -v`

Expected: PASS, including the pre-existing tracker tests.

- [ ] **Step 6: Commit**

```bash
git add src/detection.py src/tracker.py tests/unit/test_tracker.py
git commit -m "Give the tracker a motion lookup for the current scan's objects"
```

---

### Task 2: Attach motion to objects during live association

**Files:**
- Modify: `src/history/store.py:110`, `src/history/store.py:203`
- Test: `tests/unit/test_history_store.py`

**Interfaces:**
- Consumes: `StormTracker.motion_for_current_object(object_id)` and `DetectedObject.motion` from Task 1.
- Produces: every object passing through the history store carries `obj.motion`. Tasks 3–5 read it as `getattr(obj, "motion", None)`.

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_history_store.py`. This uses that file's existing `_feed` helper (line 23), which feeds scans built by `_scan` (line 15) — each carrying two objects that drift by `minute * 0.001` degrees of longitude, so a track accumulates real motion across scans.

```python
def test_live_scan_objects_carry_their_track_motion(tmp_path):
    history = RadarHistory.open("KTLX", tmp_path, ring_size=5)
    _feed(history, [0, 5, 10])

    newest = list(history.scans())[-1]

    assert newest.detected_objects
    for obj in newest.detected_objects:
        assert hasattr(obj, "motion")
    assert any(obj.motion is not None for obj in newest.detected_objects)
```

Three scans, not two: the first establishes tracks and the later ones give the tracker the position history it needs before it will state a motion.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_history_store.py::test_store_attaches_motion_to_current_objects -v`

Expected: FAIL — every `obj.motion` is `None` because nothing assigns it.

- [ ] **Step 3: Assign motion alongside temporal_status**

`src/history/store.py` line 110 currently reads:

```python
                obj.temporal_status = self.tracker.temporal_status_for_current_object(obj.object_id)
```

Add immediately after it, at the same indentation:

```python
                obj.motion = self.tracker.motion_for_current_object(obj.object_id)
```

Apply the identical change after line 203, which has the same `temporal_status` assignment in the second code path. Both sites must change — missing one leaves motion absent on whichever path that is.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_history_store.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/history/store.py tests/unit/test_history_store.py
git commit -m "Carry each object's own track motion through the history store"
```

---

### Task 3: Emit motion properties on storm features

**Files:**
- Modify: `src/map_layer.py:348` (`_storm_properties`)
- Test: `tests/unit/test_map_layer.py`

**Interfaces:**
- Consumes: `DetectedObject.motion` from Task 2; `MotionVector` fields `speed_mph: int | None`, `heading_deg: float | None`, `heading_label: str`, `source: str`, `confidence: MotionConfidence | None` where `MotionConfidence` has `label: str` and `score: float` (`src/tracking/types.py:79`).
- Produces: five new feature properties — `motion_speed_mph`, `motion_heading_deg`, `motion_heading_label`, `motion_confidence`, `motion_source`. Tasks 4 and 5 use these names exactly.

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_map_layer.py`. First add this helper, which mirrors the existing `_storm_feature_with_rotation` (line 105) exactly — same fixture geometry, with a motion attached instead of a rotation. Tasks 3 and 4 both use it.

```python
def _storm_feature_with_motion(motion):
    reflectivity = np.full((12, 12), np.nan)
    reflectivity[4:8, 5:9] = 50.0
    mask = ~np.isnan(reflectivity)
    obj = DetectedObject(
        object_id=1, centroid_lat=35.2, centroid_lon=-96.9, distance_km=30.0,
        bearing_deg=90.0, peak_dbz=50.0, peak_label="intense precipitation",
        area_km2=24.0,
    )
    obj.motion = motion
    scan = BufferedScan(
        timestamp=datetime(2026, 4, 10, 20, 0), site_id="KTLX",
        reflectivity_data=SweepData(
            reflectivity=reflectivity, azimuths=np.linspace(80, 100, 12),
            ranges_m=np.linspace(20000, 40000, 12), radar_lat=35.3331,
            radar_lon=-97.2778, elevation_angle=0.5, elevations=np.full(12, 0.5),
            elevation_angles=[0.5], radar_alt_m=390.0, timestamp="2026-04-10T20:00:00Z",
        ),
        detected_objects=[obj],
        labeled_grid=mask.astype(int), object_masks={1: mask},
    )
    return build_storm_geojson(scan)["features"][0]
```

Then the tests:

```python
def test_storm_features_carry_motion_properties():
    from src.tracking.motion import MotionVector
    from src.tracking.types import MotionConfidence

    props = _storm_feature_with_motion(MotionVector(
        speed_kmh=32.0,
        speed_mph=20,
        heading_deg=95.0,
        heading_label="east",
        confidence=MotionConfidence(label="measured", score=0.9),
        source="track_history",
    ))["properties"]

    assert props["motion_speed_mph"] == 20
    assert props["motion_heading_deg"] == 95.0
    assert props["motion_heading_label"] == "east"
    assert props["motion_confidence"] == "measured"
    assert props["motion_source"] == "track_history"


def test_storm_features_emit_null_motion_when_unknown():
    props = _storm_feature_with_motion(None)["properties"]

    assert props["motion_speed_mph"] is None
    assert props["motion_heading_deg"] is None
    assert props["motion_heading_label"] is None
    assert props["motion_confidence"] is None
    assert props["motion_source"] is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_map_layer.py::test_storm_features_carry_motion_properties tests/unit/test_map_layer.py::test_storm_features_emit_null_motion_when_unknown -v`

Expected: FAIL — `KeyError: 'motion_speed_mph'`.

- [ ] **Step 3: Emit the properties**

In `src/map_layer.py`, inside `_storm_properties` (starting line 348), add near the top of the function body, next to the existing `rotation = getattr(obj, "rotation", None)` on line 349:

```python
    motion = getattr(obj, "motion", None)
    motion_confidence = motion.confidence.label if motion is not None and motion.confidence is not None else None
```

Then add these five entries to the returned dictionary, directly after the `rotation_dual_pol_available` entry (line 384):

```python
        "motion_speed_mph": motion.speed_mph if motion is not None else None,
        "motion_heading_deg": motion.heading_deg if motion is not None else None,
        "motion_heading_label": motion.heading_label if motion is not None else None,
        "motion_confidence": motion_confidence,
        "motion_source": motion.source if motion is not None else None,
```

Emitting explicit `null` rather than omitting the keys matches how every `rotation_*` property already behaves when no rotation is present.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_map_layer.py -v`

Expected: PASS, including pre-existing map layer tests.

- [ ] **Step 5: Commit**

```bash
git add src/map_layer.py tests/unit/test_map_layer.py
git commit -m "Put each storm's motion on its map feature"
```

---

### Task 4: Speak motion in the storm description

**Files:**
- Modify: `src/map_layer.py:118` (`_storm_description`), and its call site at `src/map_layer.py:357`
- Test: `tests/unit/test_map_layer.py`

**Interfaces:**
- Consumes: `DetectedObject.motion` from Task 2.
- Produces: `_storm_description(obj: DetectedObject, rotation) -> str` gains a motion sentence. Signature is unchanged — motion is read from `obj`, which `_storm_description` already receives.

This is the task that makes the work reach a user. Properties alone are not spoken; Audiom reads `description`.

- [ ] **Step 1: Write the failing test**

These reuse `_storm_feature_with_motion` from Task 3.

```python
def test_description_states_known_motion():
    from src.tracking.motion import MotionVector

    description = _storm_feature_with_motion(MotionVector(
        speed_kmh=32.0, speed_mph=20, heading_deg=95.0,
        heading_label="east", source="track_history",
    ))["properties"]["description"]

    assert "Moving east at 20 mph." in description


def test_description_is_silent_on_uncertain_motion():
    from src.tracking.motion import MotionVector

    description = _storm_feature_with_motion(MotionVector(
        speed_kmh=None, speed_mph=None, heading_deg=None,
        heading_label="uncertain", source="track_history",
    ))["properties"]["description"]

    assert "uncertain" not in description
    assert "Moving" not in description


def test_description_is_silent_on_nearby_sourced_motion():
    from src.tracking.motion import MotionVector, NEARBY_SOURCE

    description = _storm_feature_with_motion(MotionVector(
        speed_kmh=32.0, speed_mph=20, heading_deg=95.0,
        heading_label="east", source=NEARBY_SOURCE,
    ))["properties"]["description"]

    assert "Moving" not in description
```

The third test enforces the Global Constraint: motion borrowed from a neighbouring storm is not this storm's own measurement and must not be spoken as if it were. `NEARBY_SOURCE` is already imported in `src/summary.py:6` from `src.tracking.motion`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_map_layer.py -k description -v`

Expected: the first test FAILS (no motion text); the two silence tests may pass vacuously — that is expected and they become meaningful guards after Step 3.

- [ ] **Step 3: Add the motion sentence**

In `src/map_layer.py`, add this helper directly above `_storm_description` (line 118):

```python
_UNSPOKEN_HEADINGS = {"uncertain", "stationary", "nearly stationary", "unknown"}


def _motion_sentence(obj: DetectedObject) -> str | None:
    """State motion only when this storm measured it itself."""
    motion = getattr(obj, "motion", None)
    if motion is None:
        return None
    if motion.heading_label in _UNSPOKEN_HEADINGS:
        return None
    if motion.source == NEARBY_SOURCE:
        return None
    if motion.speed_mph is None:
        return None
    return f"Moving {motion.heading_label} at {motion.speed_mph} mph."
```

Add the import at the top of `src/map_layer.py`:

```python
from src.tracking.motion import NEARBY_SOURCE
```

Then inside `_storm_description`, the sentence list is the local variable `parts`. Insert the motion sentence **after** the `temporal_status` if/elif/else block ends and **before** the `if rotation is not None ...` block begins:

```python
    motion_sentence = _motion_sentence(obj)
    if motion_sentence is not None:
        parts.append(motion_sentence)
```

Rotation stays last, matching the existing ordering where the most safety-critical statement is spoken at the end.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_map_layer.py -v`

Expected: PASS. Pre-existing description tests may assert exact strings; if any break, update them to include the new sentence only where a motion was set — never by loosening an assertion to `in` when it was an equality check.

- [ ] **Step 5: Commit**

```bash
git add src/map_layer.py tests/unit/test_map_layer.py
git commit -m "Say where a storm is heading when it measured that itself"
```

---

### Task 5: Advertise motion speed as a sonification dimension

**Files:**
- Modify: `src/map_layer.py:499` (`_storm_metadata`)
- Test: `tests/unit/test_map_layer.py`

**Interfaces:**
- Consumes: the `motion_speed_mph` property from Task 3.
- Produces: one additional `dataProperties` descriptor. No new function names.

- [ ] **Step 1: Write the failing test**

These use the existing `_scan_with_two_adjacent_objects()` helper (line 882), which returns a `BufferedScan` carrying two objects — two distinct speeds are required for the field to survive `_safe_audiom_data_properties`.

```python
def test_metadata_advertises_motion_speed_when_it_varies():
    from src.tracking.motion import MotionVector

    scan = _scan_with_two_adjacent_objects()
    speeds = [10, 30]
    for obj, mph in zip(scan.detected_objects, speeds):
        obj.motion = MotionVector(
            speed_kmh=mph * 1.60934, speed_mph=mph, heading_deg=90.0,
            heading_label="east", source="track_history",
        )

    metadata = build_storm_audiom_geojson(scan)["metadata"]
    fields = [descriptor["field"] for descriptor in metadata["dataProperties"]]

    assert "motion_speed_mph" in fields


def test_metadata_drops_motion_speed_when_all_objects_lack_motion():
    scan = _scan_with_two_adjacent_objects()
    for obj in scan.detected_objects:
        obj.motion = None

    metadata = build_storm_audiom_geojson(scan)["metadata"]
    fields = [descriptor["field"] for descriptor in metadata["dataProperties"]]

    assert "motion_speed_mph" not in fields
```

The second test needs no new code — `_safe_audiom_data_properties` (line 518) already strips any advertised field whose values are not at least two distinct finite numbers. It is here to prove that guard covers the new field, because a constant-valued dimension makes Audiom's pitch non-finite.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_map_layer.py -k metadata_advertises -v`

Expected: the first test FAILS — `motion_speed_mph` is not advertised.

- [ ] **Step 3: Advertise the field**

In `src/map_layer.py`, add to the `dataProperties` list inside `_storm_metadata` (line 507), after the `distance_mi` descriptor:

```python
            {"field": "motion_speed_mph", "displayName": "Motion speed mph", "type": "number"},
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_map_layer.py -v`

Expected: PASS.

- [ ] **Step 5: Run the full unit suite**

Run: `uv run pytest tests/unit -v`

Expected: PASS. If `tests/unit/test_summary.py` fails, motion has changed shape somewhere it should not have — investigate rather than adjusting the summary tests.

- [ ] **Step 6: Commit**

```bash
git add src/map_layer.py tests/unit/test_map_layer.py
git commit -m "Offer motion speed as a sonification dimension"
```

---

## Verification

After Task 5, confirm the change reaches a real map rather than only the tests:

- [ ] Start ARW: `uv run uvicorn src.server:app --host 127.0.0.1 --port 8000`
- [ ] Fetch a live layer for a site with active storms and at least two retained scans (motion needs history — a cold start has none):
      `curl -s "http://127.0.0.1:8000/map/storms.geojson?latitude=33.4484&longitude=-112.0741&mode=audiom"`
- [ ] Confirm at least one feature has a non-null `motion_speed_mph` and a `description` containing "Moving".
- [ ] Confirm every feature still carries its pre-existing 40 properties — this plan adds five, removes none.

## Out of scope

- **Arrival time / ETA.** "Reaching you in ~25 minutes" needs the selected location and a forecast position; it is a different calculation and a separate plan.
- **The `name` string.** Today it ends "58 miles WNW", a bearing from the radar. Whether motion belongs in the name Audiom announces on entry is an open presentation question, deliberately untouched here.
- **Level III STI corroboration.** Covered by the spec, not this plan. This plan depends on nothing from Level III.
