# Compact Scan History Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep the five most recent scans per radar as lossless compact records (reflectivity plus labels measured at 0.26 MB against ~100 MB for a full scan; total size including records is measured by the proofs), persisted to disk with tracker state, so ARW can reacquire a storm missing for one scan (defect 2), serve any retained scan with its own tracking context, report storm growth trends, and survive radar switches and restarts.

**Architecture:** A `CompactScan` stores reflectivity as Level II 0.5 dBZ codes, one label grid instead of per-object masks, and zlib-compressed JSON records for objects, scan quality, precipitation evidence and the tracking snapshot. A `RadarHistory` per site owns a ring of compact scans and a `StormTracker` that loads its previous scan from the ring instead of retaining it. A `HistoryRegistry` bounds radars in memory and reloads evicted ones from `cache/<site>/history/`. The server's four full-scan caches are replaced by the registry, a small LRU of historical compact scans, and a bounded rendered-layer cache.

**Tech Stack:** Python 3.11 (`.venv`), NumPy, zlib, FastAPI, pytest.

**Spec:** `docs/superpowers/specs/2026-09-12-compact-scan-history-design.md`. Read **Amendment 1** first: it overrides the original data model (no stored dual-pol), adds per-scan tracking snapshots, the tracker scan loader, and the one-missed-scan reacquisition limit.

## Global Constraints

- **Prerequisite:** `docs/superpowers/plans/2026-09-12-radar-correctness-fixes.md` is fully implemented. This plan uses its `BufferedScan.precipitation_band_evidence`, `TrackingSnapshot`, `BufferedScan.tracking`, `StormTracker.snapshot()`, `StormTracker.last_scan_timestamp`, `_track_live_scan` and `tracking_context`.
- Run Python through `.venv/Scripts/python.exe`.
- Accuracy outranks delivery speed. Lossless means bit-identical: compare arrays with `np.array_equal(..., equal_nan=True)` and GeoJSON with `json.dumps(..., sort_keys=True)`. Never loosen a comparison to make a test pass.
- **Continuous tracking must not change.** With reacquisition disabled, the new pipeline must produce identical tracker state to the old one on real data (Task 11).
- Only the Ingest Manager (`src/ingest.py`) makes network calls. Nothing in `src/history/` does.
- Defaults: 5 scans per radar (`ARW_SCANS_PER_RADAR`), 20 radars in memory (`ARW_MAX_RADARS_IN_MEMORY`), 12 historical scans (`ARW_MAX_HISTORICAL_SCANS`), 10 unpinned rendered scans (`ARW_MAX_RENDERED_SCANS`), history root `cache/` (`ARW_HISTORY_ROOT`).
- Every commit needs green `.venv/Scripts/python.exe -m pytest tests/unit tests/smoke -q`, plus any e2e file the task names.
- Commit messages end with:
  ```
  Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_011VMHfi1nbuux7187WtgQTt
  ```

## File Structure

New package `src/history/`, one responsibility per file:

- `src/history/__init__.py`: package docstring only.
- `src/history/codec.py`: lossless grid encoding (`EncodedGrid`, `encode_stepped`, `encode_integer`, `encode_raw`, `decode`).
- `src/history/records.py`: JSON encoding of registered analysis dataclasses (`dumps`, `loads`, `to_jsonable`, `from_jsonable`).
- `src/history/compact_scan.py`: `CompactScan`, `LabelMaskView`, `timestamp_key`, `verify_label_masks`.
- `src/history/disk.py`: `.arwscan` file format, atomic writes, `CompactScanLoadError`.
- `src/history/store.py`: `RadarHistory` (ring, tracker, persistence, rebuild) and `HistoryRegistry` (LRU of radars).
- `src/history/layer_cache.py`: `RenderedLayerCache`.
- `src/tracking/trends.py`: `StormTrend`, `compute_trend`.

Modified:

- `src/tracker.py`: scan loader, `release_previous_scan`, `export_state`/`from_state`, reacquisition handling, trend samples.
- `src/tracking/association.py`: stage-2 reacquisition.
- `src/tracking/types.py`: `Track.last_seen_ref`, `TrendSample`, `Track.trend_samples`.
- `src/tracking/events.py`: `normalize_reacquired_event`.
- `src/summary.py`: reacquired and rebuilt wording.
- `src/server.py`: registry integration, `/map/history`, status changes.
- `src/models.py`: history and trend models.
- `src/buffer.py`: `ReplayBuffer` removed.
- `tests/conftest.py` (new): in-memory history registry for every test.

---

### Task 1: Lossless grid codec

**Files:**
- Create: `src/history/__init__.py`
- Create: `src/history/codec.py`
- Test: `tests/unit/test_history_codec.py`

**Interfaces:**
- Produces:
  - `EncodedGrid(kind: str, stored_dtype: str, original_dtype: str, shape: tuple[int, ...], payload: bytes, step: float | None = None, offset: float | None = None)`, frozen, with `.nbytes -> int`
  - `encode_stepped(values: np.ndarray, *, step: float, offset: float, label: str) -> EncodedGrid`
  - `encode_integer(values: np.ndarray, *, label: str) -> EncodedGrid`
  - `encode_raw(values: np.ndarray) -> EncodedGrid`
  - `decode(grid: EncodedGrid) -> np.ndarray`
  - Constants `REFLECTIVITY_STEP_DBZ = 0.5`, `REFLECTIVITY_OFFSET_DBZ = -32.5`, `ZLIB_LEVEL = 6`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_history_codec.py`:

```python
import logging

import numpy as np
import pytest

from src.history.codec import (
    REFLECTIVITY_OFFSET_DBZ,
    REFLECTIVITY_STEP_DBZ,
    decode,
    encode_integer,
    encode_raw,
    encode_stepped,
)


def _reflectivity_like(shape=(360, 500), seed=7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    codes = rng.integers(1, 256, size=shape)
    values = codes * REFLECTIVITY_STEP_DBZ + REFLECTIVITY_OFFSET_DBZ
    values = values.astype(np.float64)
    values[rng.random(shape) < 0.7] = np.nan
    return values


def _encode_reflectivity(values):
    return encode_stepped(
        values, step=REFLECTIVITY_STEP_DBZ, offset=REFLECTIVITY_OFFSET_DBZ, label="test reflectivity"
    )


def test_stepped_field_round_trips_bit_identically():
    values = _reflectivity_like()
    grid = _encode_reflectivity(values)
    decoded = decode(grid)
    assert grid.kind == "stepped"
    assert decoded.dtype == values.dtype
    assert np.array_equal(decoded, values, equal_nan=True)


def test_stepped_field_preserves_float32_dtype():
    values = _reflectivity_like().astype(np.float32)
    decoded = decode(_encode_reflectivity(values))
    assert decoded.dtype == np.float32
    assert np.array_equal(decoded, values, equal_nan=True)


def test_stepped_extremes_are_exact():
    values = np.array([[-32.0, 94.5, np.nan, 0.0]])
    assert np.array_equal(decode(_encode_reflectivity(values)), values, equal_nan=True)


@pytest.mark.parametrize(
    "bad_value",
    [20.25, -32.5, 95.5, np.inf, -np.inf],
    ids=["off-step", "below-range", "above-range", "inf", "-inf"],
)
def test_unrepresentable_values_fall_back_to_original_dtype(caplog, bad_value):
    values = _reflectivity_like()
    values[0, 0] = bad_value
    with caplog.at_level(logging.WARNING, logger="src.history.codec"):
        grid = _encode_reflectivity(values)
    assert grid.kind == "raw"
    assert "test reflectivity" in caplog.text
    assert np.array_equal(decode(grid), values, equal_nan=True)


def test_all_nan_field_is_stepped_and_exact():
    values = np.full((10, 10), np.nan)
    grid = _encode_reflectivity(values)
    assert grid.kind == "stepped"
    assert np.array_equal(decode(grid), values, equal_nan=True)


def test_stepped_rejects_integer_grids():
    with pytest.raises(TypeError):
        encode_stepped(np.zeros((2, 2), dtype=np.int32), step=0.5, offset=0.0, label="x")


def test_integer_grid_uses_smallest_unsigned_storage_and_restores_dtype():
    labels = np.zeros((720, 1832), dtype=np.int32)
    labels[100:110, 200:210] = 63
    grid = encode_integer(labels, label="labels")
    assert np.dtype(grid.stored_dtype) == np.uint8
    decoded = decode(grid)
    assert decoded.dtype == np.int32
    assert np.array_equal(decoded, labels)


def test_integer_grid_widens_for_large_labels():
    labels = np.zeros((4, 4), dtype=np.int64)
    labels[0, 0] = 70_000
    grid = encode_integer(labels, label="labels")
    assert np.dtype(grid.stored_dtype) == np.uint32
    assert np.array_equal(decode(grid), labels)


def test_integer_grid_rejects_negative_labels():
    with pytest.raises(ValueError):
        encode_integer(np.array([[-1, 0]], dtype=np.int32), label="labels")


def test_raw_grid_round_trips():
    azimuths = np.linspace(0.0, 359.5, 720)
    grid = encode_raw(azimuths)
    assert np.array_equal(decode(grid), azimuths)


def test_sparse_reflectivity_compresses_far_below_float64():
    values = np.full((720, 1832), np.nan)
    values[100:140, 300:400] = 45.0
    grid = _encode_reflectivity(values)
    assert grid.nbytes < values.nbytes / 100
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_history_codec.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.history'`.

- [ ] **Step 3: Implement the codec**

Create `src/history/__init__.py`:

```python
"""Compact, lossless per-radar scan history (see the 2026-09-12 design spec)."""
```

Create `src/history/codec.py`:

```python
"""Lossless, compact storage for radar grids.

Level II moments are transmitted as integer codes with a fixed scale and
offset, so a decoded field sits exactly on a quantization step.  Storing the
code instead of the float is lossless -- but only if every value really is on
the step.  That is checked on every encode, never assumed: a grid that fails
the check is stored in its original dtype instead, and the fallback is
logged.  A surprise in a future Level II build can cost size, never accuracy.
"""

from dataclasses import dataclass
import logging
import zlib

import numpy as np

_logger = logging.getLogger(__name__)

ZLIB_LEVEL = 6
# NEXRAD Level II reflectivity: 0.5 dBZ steps from -32.0 dBZ.  Code 0 is
# reserved for no data (NaN), so code 1 is -32.0 dBZ and code 255 is 95.0 dBZ,
# which covers Level II's own range of -32.0 to 94.5 dBZ.
REFLECTIVITY_STEP_DBZ = 0.5
REFLECTIVITY_OFFSET_DBZ = -32.5


@dataclass(frozen=True)
class EncodedGrid:
    kind: str  # "stepped", "integer" or "raw"
    stored_dtype: str
    original_dtype: str
    shape: tuple[int, ...]
    payload: bytes
    step: float | None = None
    offset: float | None = None

    @property
    def nbytes(self) -> int:
        return len(self.payload)


def _compress(array: np.ndarray) -> bytes:
    return zlib.compress(np.ascontiguousarray(array).tobytes(), ZLIB_LEVEL)


def encode_raw(values: np.ndarray) -> EncodedGrid:
    """Store a grid exactly as it is, compressed."""
    array = np.asarray(values)
    return EncodedGrid(
        kind="raw",
        stored_dtype=array.dtype.str,
        original_dtype=array.dtype.str,
        shape=tuple(array.shape),
        payload=_compress(array),
    )


def encode_stepped(
    values: np.ndarray, *, step: float, offset: float, label: str
) -> EncodedGrid:
    """Store a float grid as uint8 codes when that is provably lossless."""
    array = np.asarray(values)
    if array.dtype.kind != "f":
        raise TypeError(f"{label}: stepped encoding needs a float grid, got {array.dtype}")
    finite = np.isfinite(array)
    finite_values = array[finite]
    codes = np.rint((finite_values - offset) / step)
    exact = (
        not np.any(np.isinf(array))
        and bool(np.all((codes >= 1) & (codes <= 255)))
        and np.array_equal((codes * step + offset).astype(array.dtype), finite_values)
    )
    if not exact:
        _logger.warning(
            "%s is not exactly representable on its %s step from %s; "
            "storing it in its original dtype %s",
            label, step, offset, array.dtype,
        )
        return encode_raw(array)
    stored = np.zeros(array.shape, dtype=np.uint8)
    stored[finite] = codes.astype(np.uint8)
    return EncodedGrid(
        kind="stepped",
        stored_dtype=stored.dtype.str,
        original_dtype=array.dtype.str,
        shape=tuple(array.shape),
        payload=_compress(stored),
        step=float(step),
        offset=float(offset),
    )


def encode_integer(values: np.ndarray, *, label: str) -> EncodedGrid:
    """Store a non-negative integer grid in the smallest unsigned dtype."""
    array = np.asarray(values)
    if array.dtype.kind not in "iu":
        raise TypeError(f"{label}: integer encoding needs an integer grid, got {array.dtype}")
    high = int(array.max()) if array.size else 0
    if array.size and int(array.min()) < 0:
        raise ValueError(f"{label}: negative values cannot be stored")
    stored_type = next(
        candidate
        for candidate in (np.uint8, np.uint16, np.uint32, np.uint64)
        if high <= np.iinfo(candidate).max
    )
    stored = array.astype(stored_type)
    return EncodedGrid(
        kind="integer",
        stored_dtype=stored.dtype.str,
        original_dtype=array.dtype.str,
        shape=tuple(array.shape),
        payload=_compress(stored),
    )


def decode(grid: EncodedGrid) -> np.ndarray:
    """Rebuild the exact original grid."""
    stored = np.frombuffer(
        zlib.decompress(grid.payload), dtype=np.dtype(grid.stored_dtype)
    ).reshape(grid.shape)
    original = np.dtype(grid.original_dtype)
    if grid.kind in ("raw", "integer"):
        return stored.astype(original, copy=True)
    if grid.kind == "stepped":
        decoded = np.full(grid.shape, np.nan, dtype=original)
        present = stored != 0
        decoded[present] = stored[present].astype(np.float64) * grid.step + grid.offset
        return decoded
    raise ValueError(f"unknown encoded grid kind: {grid.kind!r}")
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_history_codec.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/history/__init__.py src/history/codec.py tests/unit/test_history_codec.py
git commit -m "$(cat <<'EOF'
Add lossless compact grid codec for scan history

Reflectivity is stored as Level II 0.5 dBZ codes only when every value
round-trips exactly; anything else falls back to its original dtype
with a logged warning. Label grids use the smallest unsigned dtype.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_011VMHfi1nbuux7187WtgQTt
EOF
)"
```

---

### Task 2: JSON records for analysis objects

**Files:**
- Create: `src/history/records.py`
- Test: `tests/unit/test_history_records.py`

**Interfaces:**
- Consumes: `TrackingSnapshot` (correctness-fixes plan).
- Produces: `to_jsonable(value) -> Any`, `from_jsonable(data) -> Any`, `dumps(value) -> str`, `loads(text: str) -> Any`, `register_record_type(cls) -> cls`. Registered types: `DetectedObject`, `IntensityLayerData`, `RotationSignature`, `ScanQuality`, `TrackingSnapshot`, `Track`, `TrackPosition`, `PeakEntry`, `IdentityConfidence`, `FocusContinuity`, `MotionConfidence`, `MotionSample`, `RotationHistoryEntry`, `MotionVector`.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_history_records.py`:

```python
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from src.buffer import TrackingSnapshot
from src.detection import DetectedObject, IntensityLayerData
from src.history.records import dumps, from_jsonable, loads, to_jsonable
from src.preprocess import ScanQuality
from src.tracker import StormTracker
from src.velocity import RotationSignature
from tests.unit.test_tracker import _make_object, _make_scan


def _rotation() -> RotationSignature:
    return RotationSignature(
        centroid_lat=35.1, centroid_lon=-97.4, distance_km=30.0, bearing_deg=200.0,
        max_shear_ms=22.0, max_inbound_ms=-11.0, max_outbound_ms=11.0, diameter_km=2.5,
        sweep_count=2, elevation_angles=[0.5, 0.9], strength="moderate",
        associated_object_id=3,
    )


def test_detected_object_with_open_ended_band_and_rotation_round_trips():
    obj = DetectedObject(
        object_id=3, centroid_lat=35.1, centroid_lon=-97.4, distance_km=30.0,
        bearing_deg=200.0, peak_dbz=62.5, peak_label="severe core", area_km2=120.0,
        layers=[
            IntensityLayerData("heavy precipitation", 40.0, 50.0, 80.0),
            IntensityLayerData("severe core", 60.0, float("inf"), 4.0),
        ],
        rotation=_rotation(),
        class_fractions={"precipitation": 0.9, "hail": 0.1},
        temporal_status="persistent",
    )
    assert loads(dumps(obj)) == obj


def test_scan_quality_and_band_evidence_keys_round_trip():
    quality = ScanQuality(
        score=0.8, finite_fraction=0.3, removed_speckle_pixels=np.int64(12),
        removed_speckle_fraction=0.001, flags=["ok"], class_fractions={"precipitation": 1.0},
    )
    evidence = {(60.0, float("inf")): {"median_rhohv": 0.98, "class_fractions": {}}}
    restored_quality, restored_evidence = loads(dumps([quality, evidence]))
    assert restored_quality == quality
    assert isinstance(restored_quality.removed_speckle_pixels, int)
    assert restored_evidence == evidence


def test_aware_datetimes_and_tuples_round_trip():
    value = {"ref": (datetime(2026, 9, 12, 18, 0, tzinfo=timezone.utc), 4)}
    restored = loads(dumps(value))
    assert restored == value
    assert isinstance(restored["ref"], tuple)


def test_dicts_that_look_like_markers_are_not_misread():
    value = {"__datetime__": "not a date", "__type__": "Track"}
    assert loads(dumps(value)) == value


def test_tracker_tracks_and_snapshot_round_trip_exactly():
    tracker = StormTracker()
    t1 = datetime(2026, 9, 12, 18, 0)
    for minutes in (0, 5, 10):
        tracker.update(_make_scan(
            "KTLX", t1 + timedelta(minutes=minutes),
            [_make_object(1, 35.3, -97.3 + minutes * 0.01), _make_object(2, 35.0, -97.0)],
        ))
    snapshot = tracker.snapshot()
    restored = loads(dumps(snapshot))
    assert isinstance(restored, TrackingSnapshot)
    assert restored == snapshot
    assert dumps(restored) == dumps(snapshot)


def test_unregistered_types_are_rejected_both_ways():
    class NotRegistered:
        pass

    with pytest.raises(TypeError):
        to_jsonable(NotRegistered())
    with pytest.raises(ValueError):
        from_jsonable({"__type__": "os.system", "fields": {}})
    with pytest.raises(ValueError):
        from_jsonable({"unexpected": 1})
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_history_records.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.history.records'`.

- [ ] **Step 3: Implement the records codec**

Create `src/history/records.py`:

```python
"""JSON records for the analysis objects a compact scan keeps.

Every value is tagged, so decoding is unambiguous: dataclasses, datetimes,
tuples and dicts each carry a marker, and plain JSON objects are never
produced for data.  Only registered dataclasses can be decoded, so a record
file can never construct an arbitrary type.  NaN and infinity use Python's
JSON extensions; these files are internal and never served to clients.
"""

from dataclasses import fields, is_dataclass
from datetime import datetime
import json
from typing import Any

import numpy as np

from src.buffer import TrackingSnapshot
from src.detection import DetectedObject, IntensityLayerData
from src.preprocess import ScanQuality
from src.tracking.motion import MotionVector
from src.tracking.types import (
    FocusContinuity,
    IdentityConfidence,
    MotionConfidence,
    MotionSample,
    PeakEntry,
    RotationHistoryEntry,
    Track,
    TrackPosition,
)
from src.velocity import RotationSignature

_RECORD_TYPES: dict[str, type] = {}


def register_record_type(cls: type) -> type:
    if not is_dataclass(cls):
        raise TypeError(f"{cls.__name__} is not a dataclass")
    _RECORD_TYPES[cls.__name__] = cls
    return cls


for _cls in (
    DetectedObject, IntensityLayerData, RotationSignature, ScanQuality,
    TrackingSnapshot, Track, TrackPosition, PeakEntry, IdentityConfidence,
    FocusContinuity, MotionConfidence, MotionSample, RotationHistoryEntry,
    MotionVector,
):
    register_record_type(_cls)


def to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value)
    if isinstance(value, datetime):
        return {"__datetime__": value.isoformat()}
    if isinstance(value, tuple):
        return {"__tuple__": [to_jsonable(item) for item in value]}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {"__dict__": [[to_jsonable(k), to_jsonable(v)] for k, v in value.items()]}
    if is_dataclass(value) and not isinstance(value, type):
        name = type(value).__name__
        if _RECORD_TYPES.get(name) is not type(value):
            raise TypeError(f"{name} is not a registered record type")
        return {
            "__type__": name,
            "fields": {f.name: to_jsonable(getattr(value, f.name)) for f in fields(value)},
        }
    raise TypeError(f"cannot encode {type(value).__name__} as a record")


def from_jsonable(data: Any) -> Any:
    if isinstance(data, list):
        return [from_jsonable(item) for item in data]
    if not isinstance(data, dict):
        return data
    if set(data) == {"__datetime__"}:
        return datetime.fromisoformat(data["__datetime__"])
    if set(data) == {"__tuple__"}:
        return tuple(from_jsonable(item) for item in data["__tuple__"])
    if set(data) == {"__dict__"}:
        return {from_jsonable(k): from_jsonable(v) for k, v in data["__dict__"]}
    if set(data) == {"__type__", "fields"}:
        cls = _RECORD_TYPES.get(data["__type__"])
        if cls is None:
            raise ValueError(f"unregistered record type {data['__type__']!r}")
        return cls(**{name: from_jsonable(item) for name, item in data["fields"].items()})
    raise ValueError(f"unrecognized record structure with keys {sorted(data)}")


def dumps(value: Any) -> str:
    return json.dumps(to_jsonable(value), allow_nan=True, separators=(",", ":"))


def loads(text: str) -> Any:
    return from_jsonable(json.loads(text))
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_history_records.py -v`
Expected: all PASS. If `test_tracker_tracks_and_snapshot_round_trip_exactly` fails on an unregistered type, a `Track` field holds a dataclass missing from the registration tuple. Register it (after checking it is a plain analysis dataclass), and don't change the test.

- [ ] **Step 5: Commit**

```bash
git add src/history/records.py tests/unit/test_history_records.py
git commit -m "$(cat <<'EOF'
Add tagged JSON records for scan history objects

Registered analysis dataclasses, datetimes, tuples and non-string dict
keys round-trip exactly, including full tracker snapshots. Unregistered
types are rejected in both directions.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_011VMHfi1nbuux7187WtgQTt
EOF
)"
```

---

### Task 3: CompactScan and label-derived masks

**Files:**
- Create: `src/history/compact_scan.py`
- Test: `tests/unit/test_history_compact_scan.py`
- Test: `tests/e2e/test_proof_compact_scan.py` (create)

**Interfaces:**
- Consumes: Task 1 codec, Task 2 records, `BufferedScan` fields `precipitation_band_evidence` and `tracking`.
- Produces:
  - `SCHEMA_VERSION = 1`
  - `timestamp_key(value: datetime) -> datetime` (naive UTC)
  - `class LabelMaskView(Mapping[int, np.ndarray])` with `__init__(labels, object_ids)`
  - `verify_label_masks(scan: BufferedScan) -> None` (raises `ValueError`)
  - `CompactScan`, frozen, with fields `site_id, timestamp, scan_timestamp_text, source_path, object_count, tracked, reflectivity, labels, azimuths, ranges_m, elevations, records, schema_version`
  - `CompactScan.from_buffered_scan(scan: BufferedScan) -> CompactScan`
  - `.to_buffered_scan() -> BufferedScan`
  - `.with_tracking(tracking: TrackingSnapshot | None) -> CompactScan`
  - `.read_records() -> dict`
  - `.validate() -> None`
  - `.nbytes -> int`

- [ ] **Step 1: Write the failing unit tests**

Create `tests/unit/test_history_compact_scan.py`:

```python
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from src.detection import IntensityLayerData
from src.history.compact_scan import CompactScan, LabelMaskView, timestamp_key, verify_label_masks
from src.preprocess import ScanQuality
from src.tracker import StormTracker
from tests.unit.test_tracker import _make_object, _make_scan

T0 = datetime(2026, 9, 12, 18, 0)


def _scan_with_content():
    objects = [_make_object(1, 35.3, -97.3, 55.0), _make_object(2, 35.0, -97.0, 40.0)]
    objects[0].layers = [IntensityLayerData("severe core", 60.0, float("inf"), 4.0)]
    scan = _make_scan("KTLX", T0, objects)
    scan.reflectivity_data.reflectivity[85:95, 195:205] = 55.5
    scan.reflectivity_data.reflectivity[125:135, 195:205] = 12.0
    scan.scan_quality = ScanQuality(
        score=0.9, finite_fraction=0.01, removed_speckle_pixels=0,
        removed_speckle_fraction=0.0, flags=[],
    )
    scan.precipitation_band_evidence = {(15.0, 20.0): {"median_rhohv": None, "class_fractions": {}}}
    scan.source_path = "C:/cache/KTLX/KTLX20260912_180000_V06"
    return scan


def test_round_trip_restores_every_field_the_pipeline_reads():
    scan = _scan_with_content()
    rebuilt = CompactScan.from_buffered_scan(scan).to_buffered_scan()

    assert rebuilt.timestamp == scan.timestamp
    assert rebuilt.site_id == scan.site_id
    assert rebuilt.source_path == scan.source_path
    sweep, original = rebuilt.reflectivity_data, scan.reflectivity_data
    for name in ("reflectivity", "azimuths", "ranges_m", "elevations"):
        assert np.array_equal(getattr(sweep, name), getattr(original, name), equal_nan=True), name
        assert getattr(sweep, name).dtype == getattr(original, name).dtype, name
    for name in ("elevation_angle", "elevation_angles", "radar_lat", "radar_lon", "radar_alt_m", "timestamp"):
        assert getattr(sweep, name) == getattr(original, name), name
    assert sweep.rhohv is None and sweep.zdr is None and sweep.gate_classification is None
    assert np.array_equal(rebuilt.labeled_grid, scan.labeled_grid)
    assert rebuilt.labeled_grid.dtype == scan.labeled_grid.dtype
    assert set(rebuilt.object_masks) == set(scan.object_masks)
    for object_id, mask in scan.object_masks.items():
        assert np.array_equal(rebuilt.object_masks[object_id], mask)
    assert rebuilt.detected_objects == scan.detected_objects
    assert rebuilt.scan_quality == scan.scan_quality
    assert rebuilt.precipitation_band_evidence == scan.precipitation_band_evidence
    assert rebuilt.tracking is None


def test_rebuilt_objects_are_fresh_copies():
    compact = CompactScan.from_buffered_scan(_scan_with_content())
    first = compact.to_buffered_scan()
    first.detected_objects[0].temporal_status = "mutated"
    assert compact.to_buffered_scan().detected_objects[0].temporal_status != "mutated"


def test_tracking_snapshot_round_trips_and_sets_tracked_flag():
    tracker = StormTracker()
    scan = _scan_with_content()
    tracker.update(scan)
    scan.tracking = tracker.snapshot()
    compact = CompactScan.from_buffered_scan(scan)
    assert compact.tracked is True
    assert compact.to_buffered_scan().tracking == scan.tracking
    untracked = compact.with_tracking(None)
    assert untracked.tracked is False and untracked.to_buffered_scan().tracking is None


def test_packing_refuses_masks_the_label_grid_cannot_represent():
    scan = _scan_with_content()
    overlap = scan.object_masks[1].copy()
    overlap[125:135, 195:205] = True  # also claims object 2's gates
    scan.object_masks[1] = overlap
    with pytest.raises(ValueError, match="object 1"):
        verify_label_masks(scan)
    with pytest.raises(ValueError, match="object 1"):
        CompactScan.from_buffered_scan(scan)


def test_label_mask_view_behaves_like_the_mask_dict():
    labels = np.array([[0, 1], [2, 2]], dtype=np.int32)
    view = LabelMaskView(labels, [1, 2])
    assert list(view) == [1, 2] and len(view) == 2
    assert 2 in view and np.int64(2) in view and 3 not in view
    assert np.array_equal(view[2], np.array([[False, False], [True, True]]))
    assert view.get(3) is None
    assert dict(view.items()).keys() == {1, 2}
    with pytest.raises(KeyError):
        view[3]


def test_compact_scan_is_small_and_metadata_is_exposed():
    compact = CompactScan.from_buffered_scan(_scan_with_content())
    assert compact.object_count == 2
    assert compact.scan_timestamp_text == T0.isoformat()
    assert compact.nbytes < 50_000
    compact.validate()


def test_timestamp_key_normalizes_to_naive_utc():
    aware = datetime(2026, 9, 12, 13, 0, tzinfo=timezone(timedelta(hours=-5)))
    assert timestamp_key(aware) == T0
    assert timestamp_key(T0) == T0
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_history_compact_scan.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.history.compact_scan'`.

- [ ] **Step 3: Implement CompactScan**

Create `src/history/compact_scan.py`:

```python
"""One processed radar volume, stored compactly and losslessly.

Holds exactly what map layers, detection consumers, speech and the tracker
read after processing: reflectivity at every gate (contour interpolation
needs sub-threshold gates too), one label grid in place of per-object masks,
sweep geometry, and records (objects, scan quality, precipitation-band
evidence, tracking snapshot).  Dual-pol grids are never kept; precipitation
evidence is precomputed during processing (spec Amendment 1, A1).
"""

from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import zlib

import numpy as np

from src.buffer import BufferedScan, TrackingSnapshot
from src.history import records as record_codec
from src.history.codec import (
    REFLECTIVITY_OFFSET_DBZ,
    REFLECTIVITY_STEP_DBZ,
    ZLIB_LEVEL,
    EncodedGrid,
    decode,
    encode_integer,
    encode_raw,
    encode_stepped,
)
from src.parser import SweepData

SCHEMA_VERSION = 1


def timestamp_key(value: datetime) -> datetime:
    """Naive-UTC form used to compare scan times from any source."""
    if value.tzinfo is not None:
        return value.astimezone(timezone.utc).replace(tzinfo=None)
    return value


class LabelMaskView(Mapping):
    """Object masks derived on demand from one label grid.

    Replaces a dict of full-grid boolean masks (one per object) without
    holding any of them: each lookup computes `labels == object_id`.
    """

    def __init__(self, labels: np.ndarray, object_ids):
        self._labels = labels
        self._object_ids = tuple(int(object_id) for object_id in object_ids)
        self._id_set = frozenset(self._object_ids)

    def __getitem__(self, object_id) -> np.ndarray:
        if object_id not in self._id_set:
            raise KeyError(object_id)
        return self._labels == object_id

    def __iter__(self):
        return iter(self._object_ids)

    def __len__(self) -> int:
        return len(self._object_ids)


def verify_label_masks(scan: BufferedScan) -> None:
    """Refuse a scan whose masks are not exactly their label-grid regions.

    Detection builds the label grid by writing each object's mask in turn,
    so overlapping masks would silently lose gates.  Checked on real data
    (no overlaps on KTLX, KEMX, KIWA), but verified on every scan anyway.
    """
    labels = np.asarray(scan.labeled_grid)
    for obj in scan.detected_objects:
        mask = scan.object_masks.get(obj.object_id)
        if mask is None or not np.array_equal(labels == obj.object_id, mask):
            raise ValueError(
                f"{scan.site_id} {scan.timestamp.isoformat()}: object {obj.object_id} "
                "mask is not exactly its label-grid region"
            )


@dataclass(frozen=True)
class CompactScan:
    site_id: str
    timestamp: datetime
    scan_timestamp_text: str
    source_path: str | None
    object_count: int
    tracked: bool
    reflectivity: EncodedGrid
    labels: EncodedGrid
    azimuths: EncodedGrid
    ranges_m: EncodedGrid
    elevations: EncodedGrid
    records: bytes
    schema_version: int = SCHEMA_VERSION

    @classmethod
    def from_buffered_scan(cls, scan: BufferedScan) -> "CompactScan":
        verify_label_masks(scan)
        sweep = scan.reflectivity_data
        label = f"{scan.site_id} {scan.timestamp.isoformat()}"
        records = {
            "elevation_angle": sweep.elevation_angle,
            "elevation_angles": list(sweep.elevation_angles),
            "radar_lat": sweep.radar_lat,
            "radar_lon": sweep.radar_lon,
            "radar_alt_m": sweep.radar_alt_m,
            "sweep_timestamp": sweep.timestamp,
            "detected_objects": list(scan.detected_objects),
            "scan_quality": scan.scan_quality,
            "rotation_signatures": list(scan.rotation_signatures),
            "precipitation_band_evidence": scan.precipitation_band_evidence,
            "tracking": scan.tracking,
        }
        timestamp_text = (
            sweep.timestamp if isinstance(sweep.timestamp, str) else sweep.timestamp.isoformat()
        )
        return cls(
            site_id=scan.site_id.upper(),
            timestamp=scan.timestamp,
            scan_timestamp_text=timestamp_text,
            source_path=scan.source_path,
            object_count=len(scan.detected_objects),
            tracked=scan.tracking is not None,
            reflectivity=encode_stepped(
                sweep.reflectivity,
                step=REFLECTIVITY_STEP_DBZ,
                offset=REFLECTIVITY_OFFSET_DBZ,
                label=f"{label} reflectivity",
            ),
            labels=encode_integer(np.asarray(scan.labeled_grid), label=f"{label} labels"),
            azimuths=encode_raw(sweep.azimuths),
            ranges_m=encode_raw(sweep.ranges_m),
            elevations=encode_raw(sweep.elevations),
            records=_encode_records(records),
        )

    def read_records(self) -> dict:
        """Fresh copies of every record; safe for callers to mutate."""
        return record_codec.loads(zlib.decompress(self.records).decode("utf-8"))

    def to_buffered_scan(self) -> BufferedScan:
        records = self.read_records()
        labels = decode(self.labels)
        objects = records["detected_objects"]
        sweep = SweepData(
            reflectivity=decode(self.reflectivity),
            azimuths=decode(self.azimuths),
            ranges_m=decode(self.ranges_m),
            elevation_angle=records["elevation_angle"],
            elevations=decode(self.elevations),
            elevation_angles=records["elevation_angles"],
            radar_lat=records["radar_lat"],
            radar_lon=records["radar_lon"],
            radar_alt_m=records["radar_alt_m"],
            timestamp=records["sweep_timestamp"],
        )
        return BufferedScan(
            timestamp=self.timestamp,
            site_id=self.site_id,
            reflectivity_data=sweep,
            detected_objects=objects,
            labeled_grid=labels,
            object_masks=LabelMaskView(labels, [obj.object_id for obj in objects]),
            scan_quality=records["scan_quality"],
            source_path=self.source_path,
            rotation_signatures=records["rotation_signatures"],
            precipitation_band_evidence=records["precipitation_band_evidence"],
            tracking=records["tracking"],
        )

    def with_tracking(self, tracking: TrackingSnapshot | None) -> "CompactScan":
        records = self.read_records()
        records["tracking"] = tracking
        return replace(self, records=_encode_records(records), tracked=tracking is not None)

    def validate(self) -> None:
        """Decode everything once; raises if any part is corrupt or inconsistent."""
        reflectivity = decode(self.reflectivity)
        labels = decode(self.labels)
        records = self.read_records()
        if reflectivity.shape != labels.shape:
            raise ValueError("reflectivity and label grids differ in shape")
        if decode(self.azimuths).shape[0] != reflectivity.shape[0]:
            raise ValueError("azimuth count does not match the grid")
        if decode(self.ranges_m).shape[0] != reflectivity.shape[1]:
            raise ValueError("range count does not match the grid")
        if len(records["detected_objects"]) != self.object_count:
            raise ValueError("object count does not match the records")

    @property
    def nbytes(self) -> int:
        grids = (self.reflectivity, self.labels, self.azimuths, self.ranges_m, self.elevations)
        return sum(grid.nbytes for grid in grids) + len(self.records)


def _encode_records(records: dict) -> bytes:
    return zlib.compress(record_codec.dumps(records).encode("utf-8"), ZLIB_LEVEL)
```

- [ ] **Step 4: Run the unit tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_history_compact_scan.py -v`
Expected: all PASS.

- [ ] **Step 5: Write the real-data proof that every map layer is identical**

Create `tests/e2e/test_proof_compact_scan.py`:

```python
"""Proof (compact scan history, Task 3): a compact scan rebuilds every map
layer bit-identically on real Level II volumes, and is small.

Compares the layers ARW publishes from the processed scan with the layers
built from `CompactScan.from_buffered_scan(scan).to_buffered_scan()`.
Sizes are printed for the test report (run with -s).
"""

import json

import numpy as np
import pytest

import src.server as server
from src.history.compact_scan import CompactScan
from src.map_layer import (
    build_precipitation_field_geojson,
    build_storm_audiom_centroid_geojson,
    build_storm_centroid_geojson,
    build_storm_geojson,
    build_storm_intensity_geojson,
)

VOLUMES = [
    ("KTLX", "cache/KTLX/KTLX20260410_224209_V06"),
    ("KEMX", "cache/KEMX/KEMX20260712_022646_V06"),
    ("KIWA", "cache/KIWA/KIWA20260712_170029_V06"),
]
BUILDERS = [
    build_storm_geojson,
    build_storm_intensity_geojson,
    build_storm_audiom_centroid_geojson,
    build_storm_centroid_geojson,
    build_precipitation_field_geojson,
]


def _canonical(geojson) -> str:
    return json.dumps(geojson, sort_keys=True)


@pytest.mark.parametrize("site_id,path", VOLUMES)
def test_compact_scan_rebuilds_all_layers_identically(site_id, path):
    original = server._process_scan_file(site_id, path)
    compact = CompactScan.from_buffered_scan(original)
    rebuilt = compact.to_buffered_scan()

    assert compact.reflectivity.kind == "stepped", "real reflectivity must be on the 0.5 dBZ step"
    assert np.array_equal(
        rebuilt.reflectivity_data.reflectivity, original.reflectivity_data.reflectivity, equal_nan=True
    )
    for obj in original.detected_objects:
        assert np.array_equal(rebuilt.object_masks[obj.object_id], original.object_masks[obj.object_id])
    for builder in BUILDERS:
        assert _canonical(builder(rebuilt)) == _canonical(builder(original)), builder.__name__

    retained_before = original.reflectivity_data.reflectivity.nbytes + original.labeled_grid.nbytes + sum(
        mask.nbytes for mask in original.object_masks.values()
    )
    print(
        f"\n{site_id}: objects={compact.object_count} compact_bytes={compact.nbytes} "
        f"(reflectivity={compact.reflectivity.nbytes} labels={compact.labels.nbytes} "
        f"records={len(compact.records)}) full_scan_array_bytes={retained_before}"
    )
    assert compact.nbytes < 1_000_000
```

- [ ] **Step 6: Run the proof**

Run: `.venv/Scripts/python.exe -m pytest tests/e2e/test_proof_compact_scan.py -v -s`
Expected: 3 PASS, with sizes printed. Save the printed lines for the Task 11 report. If a layer differs, stop and investigate. Don't loosen the comparison.

- [ ] **Step 7: Commit**

```bash
git add src/history/compact_scan.py tests/unit/test_history_compact_scan.py tests/e2e/test_proof_compact_scan.py
git commit -m "$(cat <<'EOF'
Add CompactScan with label-derived masks

A processed scan packs into coded reflectivity, one label grid and
compressed records, and rebuilds a BufferedScan whose masks are derived
on demand. Packing refuses masks the label grid cannot represent. Real
KTLX, KEMX and KIWA volumes rebuild all five layer builders identically.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_011VMHfi1nbuux7187WtgQTt
EOF
)"
```

---

### Task 4: On-disk `.arwscan` format with atomic writes

**Files:**
- Create: `src/history/disk.py`
- Test: `tests/unit/test_history_disk.py`

**Interfaces:**
- Consumes: Task 3 `CompactScan`, `SCHEMA_VERSION`.
- Produces:
  - `HISTORY_SUFFIX = ".arwscan"`
  - `class CompactScanLoadError(Exception)`
  - `history_filename(timestamp: datetime) -> str`
  - `atomic_write_bytes(path: Path, data: bytes) -> None`
  - `serialize_compact_scan(scan) -> bytes`
  - `deserialize_compact_scan(data: bytes) -> CompactScan`
  - `save_compact_scan(scan, directory: Path) -> Path`
  - `load_compact_scan(path: Path) -> CompactScan`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_history_disk.py`:

```python
from datetime import datetime, timezone

import pytest

import src.history.disk as disk
from src.history.compact_scan import CompactScan
from src.history.disk import (
    CompactScanLoadError,
    history_filename,
    load_compact_scan,
    save_compact_scan,
)
from tests.unit.test_history_compact_scan import _scan_with_content


def _compact() -> CompactScan:
    return CompactScan.from_buffered_scan(_scan_with_content())


def test_save_and_load_round_trip_exactly(tmp_path):
    compact = _compact()
    path = save_compact_scan(compact, tmp_path)
    assert path.name == "20260912_180000.arwscan"
    assert load_compact_scan(path) == compact


def test_filename_uses_utc():
    aware = datetime(2026, 9, 12, 18, 0, 5, tzinfo=timezone.utc)
    assert history_filename(aware) == "20260912_180005.arwscan"


@pytest.mark.parametrize("damage", ["truncate", "flip"])
def test_damaged_files_raise_load_error(tmp_path, damage):
    path = save_compact_scan(_compact(), tmp_path)
    data = bytearray(path.read_bytes())
    if damage == "truncate":
        data = data[: len(data) // 2]
    else:
        data[len(data) // 2] ^= 0xFF
    path.write_bytes(bytes(data))
    with pytest.raises(CompactScanLoadError):
        load_compact_scan(path)


def test_schema_mismatch_is_a_load_error(tmp_path, monkeypatch):
    path = save_compact_scan(_compact(), tmp_path)
    monkeypatch.setattr(disk, "SCHEMA_VERSION", 999)
    with pytest.raises(CompactScanLoadError, match="schema"):
        load_compact_scan(path)


def test_failed_atomic_write_leaves_previous_file_intact(tmp_path, monkeypatch):
    compact = _compact()
    path = save_compact_scan(compact, tmp_path)
    original_bytes = path.read_bytes()

    def failing_replace(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(disk.os, "replace", failing_replace)
    with pytest.raises(OSError):
        save_compact_scan(compact.with_tracking(None), tmp_path)
    assert path.read_bytes() == original_bytes
    assert not list(tmp_path.glob("*.tmp"))
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_history_disk.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.history.disk'`.

- [ ] **Step 3: Implement the disk format**

Create `src/history/disk.py`:

```python
"""The `.arwscan` file: one CompactScan as an uncompressed NumPy .npz archive.

Grid payloads and records are already zlib-compressed; the archive only
frames them next to a JSON header.  Loading never unpickles.  Every write
goes to a temporary file in the same directory and is then atomically
renamed, so a crash can never leave a half-written scan in place.
"""

from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path

import numpy as np

from src.history.codec import EncodedGrid
from src.history.compact_scan import SCHEMA_VERSION, CompactScan

HISTORY_SUFFIX = ".arwscan"
_GRID_NAMES = ("reflectivity", "labels", "azimuths", "ranges_m", "elevations")


class CompactScanLoadError(Exception):
    """A history file is unreadable, corrupt, or from another schema."""


def history_filename(timestamp: datetime) -> str:
    if timestamp.tzinfo is not None:
        timestamp = timestamp.astimezone(timezone.utc)
    return timestamp.strftime("%Y%m%d_%H%M%S") + HISTORY_SUFFIX


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with open(temporary, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def serialize_compact_scan(scan: CompactScan) -> bytes:
    header = {
        "schema_version": scan.schema_version,
        "site_id": scan.site_id,
        "timestamp": scan.timestamp.isoformat(),
        "scan_timestamp_text": scan.scan_timestamp_text,
        "source_path": scan.source_path,
        "object_count": scan.object_count,
        "tracked": scan.tracked,
        "grids": {
            name: {
                "kind": grid.kind,
                "stored_dtype": grid.stored_dtype,
                "original_dtype": grid.original_dtype,
                "shape": list(grid.shape),
                "step": grid.step,
                "offset": grid.offset,
            }
            for name, grid in ((name, getattr(scan, name)) for name in _GRID_NAMES)
        },
    }
    arrays = {
        f"grid_{name}": np.frombuffer(getattr(scan, name).payload, dtype=np.uint8)
        for name in _GRID_NAMES
    }
    arrays["records"] = np.frombuffer(scan.records, dtype=np.uint8)
    arrays["header"] = np.frombuffer(json.dumps(header).encode("utf-8"), dtype=np.uint8)
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    return buffer.getvalue()


def deserialize_compact_scan(data: bytes) -> CompactScan:
    try:
        with np.load(io.BytesIO(data), allow_pickle=False) as archive:
            header = json.loads(archive["header"].tobytes().decode("utf-8"))
            if header.get("schema_version") != SCHEMA_VERSION:
                raise CompactScanLoadError(
                    f"schema version {header.get('schema_version')!r} is not {SCHEMA_VERSION}"
                )
            grids = {
                name: EncodedGrid(
                    kind=spec["kind"],
                    stored_dtype=spec["stored_dtype"],
                    original_dtype=spec["original_dtype"],
                    shape=tuple(spec["shape"]),
                    payload=archive[f"grid_{name}"].tobytes(),
                    step=spec["step"],
                    offset=spec["offset"],
                )
                for name, spec in header["grids"].items()
            }
            scan = CompactScan(
                site_id=header["site_id"],
                timestamp=datetime.fromisoformat(header["timestamp"]),
                scan_timestamp_text=header["scan_timestamp_text"],
                source_path=header["source_path"],
                object_count=header["object_count"],
                tracked=header["tracked"],
                records=archive["records"].tobytes(),
                **grids,
            )
        scan.validate()
        return scan
    except CompactScanLoadError:
        raise
    except Exception as exc:  # zip, zlib, JSON and shape errors all mean corrupt
        raise CompactScanLoadError(f"{type(exc).__name__}: {exc}") from exc


def save_compact_scan(scan: CompactScan, directory: Path) -> Path:
    path = Path(directory) / history_filename(scan.timestamp)
    atomic_write_bytes(path, serialize_compact_scan(scan))
    return path


def load_compact_scan(path: Path) -> CompactScan:
    try:
        data = Path(path).read_bytes()
    except OSError as exc:
        raise CompactScanLoadError(f"{type(exc).__name__}: {exc}") from exc
    return deserialize_compact_scan(data)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_history_disk.py -v`
Expected: all PASS. A flipped byte must be caught by the zip CRC or zlib's checksum. If `flip` loads successfully, the flipped byte landed in header padding: pick a byte position inside a payload instead, and never catch less.

- [ ] **Step 5: Commit**

```bash
git add src/history/disk.py tests/unit/test_history_disk.py
git commit -m "$(cat <<'EOF'
Add atomic .arwscan files for compact scan history

Compact scans are framed in a pickle-free npz with a JSON header,
validated on load, and written through a temp file plus atomic rename.
Truncated, corrupted or wrong-schema files raise CompactScanLoadError.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_011VMHfi1nbuux7187WtgQTt
EOF
)"
```

---

### Task 5: Tracker loads its previous scan and exports its state

**Files:**
- Modify: `src/tracker.py` (imports, `__init__`, `_reset_for_site`, `last_scan_timestamp`, `_refresh_track_motions` line ~318, `update`)
- Test: `tests/unit/test_tracker_persistence.py` (create)

**Interfaces:**
- Consumes: `records.dumps`/`loads` (Task 2, tests only).
- Produces: `ScanLoader = Callable[[str, datetime], BufferedScan | None]`; `TRACKER_STATE_VERSION = 1`; `StormTracker(scan_loader: ScanLoader | None = None)`; `StormTracker.release_previous_scan() -> None`; `StormTracker.export_state() -> dict`; `StormTracker.from_state(state: dict, scan_loader: ScanLoader) -> StormTracker`.

- [ ] **Step 0: Capture the tracking benchmark baseline before any tracker change**

Tasks 5, 8 and 9 modify the tracker. Their effect on the existing benchmark windows is checked in Task 11 against this baseline, so it must come from the code as it is now. Create `docs/benchmarks/tracking_benchmark_manifest_local.json` holding only the `local_only: true` entries of `docs/benchmarks/tracking_benchmark_manifest.json` (all entries except `lower_complexity_quick`, which downloads), then run:

```bash
.venv/Scripts/python.exe scripts/evaluate_tracking.py --manifest docs/benchmarks/tracking_benchmark_manifest_local.json --output-json docs/test_reports/2026-09-12-benchmark-before-history.json
```

Expected: the JSON file is written with one result per manifest entry. Commit both files:

```bash
git add docs/benchmarks/tracking_benchmark_manifest_local.json docs/test_reports/2026-09-12-benchmark-before-history.json
git commit -m "$(cat <<'EOF'
Capture offline tracking benchmark baseline before history work

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_011VMHfi1nbuux7187WtgQTt
EOF
)"
```

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_tracker_persistence.py`:

```python
from datetime import datetime, timedelta
import logging

import pytest

from src.history.records import dumps, loads
from src.tracker import StormTracker
from tests.unit.test_tracker import _make_object, _make_scan

T0 = datetime(2026, 9, 12, 18, 0)


def _scans():
    return [
        _make_scan("KTLX", T0, [_make_object(1, 35.3, -97.30), _make_object(2, 35.0, -97.0)]),
        _make_scan("KTLX", T0 + timedelta(minutes=5), [_make_object(1, 35.3, -97.29), _make_object(2, 35.0, -97.0)]),
        _make_scan("KTLX", T0 + timedelta(minutes=10), [_make_object(1, 35.3, -97.28)]),
        _make_scan("KTLX", T0 + timedelta(minutes=15), [_make_object(1, 35.3, -97.27), _make_object(2, 35.0, -97.0)]),
    ]


def _store(scans):
    stored = {(scan.site_id, scan.timestamp): scan for scan in scans}
    return lambda site_id, timestamp: stored.get((site_id, timestamp))


def test_tracker_with_loader_matches_tracker_that_retains_scans():
    retained = StormTracker()
    for scan in _scans():
        retained.update(scan)

    scans = _scans()
    loaded = StormTracker(scan_loader=_store(scans))
    for scan in scans:
        loaded.update(scan)
        loaded.release_previous_scan()

    assert loaded._prev_scan is None
    assert dumps(loaded.export_state()) == dumps(retained.export_state())


def test_restored_tracker_continues_exactly_like_an_uninterrupted_one():
    uninterrupted = StormTracker()
    for scan in _scans():
        uninterrupted.update(scan)

    scans = _scans()
    first = StormTracker(scan_loader=_store(scans))
    for scan in scans[:2]:
        first.update(scan)
        first.release_previous_scan()
    resumed = StormTracker.from_state(loads(dumps(first.export_state())), scan_loader=_store(scans))
    for scan in scans[2:]:
        resumed.update(scan)
        resumed.release_previous_scan()

    assert dumps(resumed.export_state()) == dumps(uninterrupted.export_state())


def test_unavailable_previous_scan_restarts_tracking_with_a_warning(caplog):
    scans = _scans()
    tracker = StormTracker(scan_loader=lambda site_id, timestamp: None)
    tracker.update(scans[0])
    tracker.release_previous_scan()
    with caplog.at_level(logging.WARNING, logger="src.tracker"):
        tracker.update(scans[1])
    assert "no longer available" in caplog.text
    assert [track.track_id for track in tracker.active_tracks] == [1, 2]
    assert all(len(track.positions) == 1 for track in tracker.active_tracks)


def test_release_requires_a_loader():
    tracker = StormTracker()
    tracker.update(_scans()[0])
    with pytest.raises(RuntimeError):
        tracker.release_previous_scan()


def test_state_version_mismatch_is_rejected():
    state = StormTracker().export_state()
    state["version"] = 999
    with pytest.raises(ValueError, match="version"):
        StormTracker.from_state(state, scan_loader=lambda site_id, timestamp: None)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_tracker_persistence.py -v`
Expected: FAIL with `TypeError: StormTracker.__init__() got an unexpected keyword argument 'scan_loader'`.

- [ ] **Step 3: Implement loader, release, export and restore**

In `src/tracker.py`:

1. Add imports and module constants near the top:

```python
import logging
from typing import Callable
```

```python
_logger = logging.getLogger(__name__)
TRACKER_STATE_VERSION = 1
ScanLoader = Callable[[str, datetime], BufferedScan | None]
```

2. Replace `StormTracker.__init__` and `_reset_for_site` with:

```python
    def __init__(self, scan_loader: ScanLoader | None = None):
        self._tracks: list[Track] = []
        self._next_id: int = 1
        self._recent_events: list[dict] = []
        # The full previous scan is retained only until release_previous_scan();
        # after that it is reloaded through scan_loader by (site_id, timestamp).
        self._prev_scan: BufferedScan | None = None
        self._prev_scan_ref: tuple[str, datetime] | None = None
        self._scan_loader = scan_loader
        self._obj_to_track: dict[int, int] = {}  # object_id -> track_id for current scan
        self._focus_history: list[int | None] = []

    def _reset_for_site(self) -> None:
        """Clear tracker state when switching radar sites."""
        self._tracks.clear()
        self._next_id = 1
        self._recent_events.clear()
        self._prev_scan = None
        self._prev_scan_ref = None
        self._obj_to_track.clear()
        self._focus_history.clear()

    def _previous_scan(self) -> BufferedScan | None:
        if self._prev_scan_ref is None:
            return None
        if self._prev_scan is not None:
            return self._prev_scan
        if self._scan_loader is None:
            return None
        return self._scan_loader(*self._prev_scan_ref)

    def _remember_scan(self, scan: BufferedScan) -> None:
        self._prev_scan = scan
        self._prev_scan_ref = (scan.site_id, scan.timestamp)

    def release_previous_scan(self) -> None:
        """Drop the retained full previous scan; it is reloaded when needed."""
        if self._scan_loader is None:
            raise RuntimeError("release_previous_scan requires a scan_loader to reload the scan")
        self._prev_scan = None

    def export_state(self) -> dict:
        """Everything needed to resume tracking.

        Values are live references: serialize them before the next update.
        """
        return {
            "version": TRACKER_STATE_VERSION,
            "tracks": self._tracks,
            "next_id": self._next_id,
            "recent_events": self._recent_events,
            "obj_to_track": self._obj_to_track,
            "focus_history": self._focus_history,
            "prev_scan_ref": self._prev_scan_ref,
        }

    @classmethod
    def from_state(cls, state: dict, scan_loader: ScanLoader) -> "StormTracker":
        if state.get("version") != TRACKER_STATE_VERSION:
            raise ValueError(
                f"tracker state version {state.get('version')!r} is not {TRACKER_STATE_VERSION}"
            )
        tracker = cls(scan_loader)
        tracker._tracks = list(state["tracks"])
        tracker._next_id = int(state["next_id"])
        tracker._recent_events = list(state["recent_events"])
        tracker._obj_to_track = dict(state["obj_to_track"])
        tracker._focus_history = list(state["focus_history"])
        reference = state["prev_scan_ref"]
        tracker._prev_scan_ref = tuple(reference) if reference is not None else None
        return tracker
```

3. Replace the body of the `last_scan_timestamp` property with:

```python
        return self._prev_scan_ref[1] if self._prev_scan_ref is not None else None
```

4. In `_refresh_track_motions`, replace

```python
                    timestamp=track.last_seen or (self._prev_scan.timestamp if self._prev_scan is not None else datetime.min),
```

with

```python
                    timestamp=track.last_seen or (self._prev_scan_ref[1] if self._prev_scan_ref is not None else datetime.min),
```

5. In `update`, replace everything from `if self._prev_scan is not None and self._prev_scan.site_id != scan.site_id:` down to and including the `association = associate_tracks(` line and its `previous_scan=self._prev_scan,` argument with:

```python
        if self._prev_scan_ref is not None and self._prev_scan_ref[0] != scan.site_id:
            self._reset_for_site()

        if self._prev_scan_ref is not None:
            gap_minutes = (timestamp - self._prev_scan_ref[1]).total_seconds() / 60.0
            # Cross-scan association is evidence only over adjacent operational
            # radar volumes.  Retaining a track across a long outage or a
            # historical jump could falsely make an unrelated echo appear
            # persistent, particularly because its motion search radius grows
            # with elapsed time.
            if gap_minutes <= 0 or gap_minutes > MAX_TEMPORAL_CONTINUITY_MINUTES:
                self._reset_for_site()

        previous_scan = self._previous_scan()
        if self._prev_scan_ref is not None and previous_scan is None:
            _logger.warning(
                "Previous scan %s %s is no longer available; storm tracking restarts for this site",
                *self._prev_scan_ref,
            )
            self._reset_for_site()

        if self._prev_scan_ref is None:
            # First scan: create a track for each object
            self._obj_to_track.clear()
            for obj in scan.detected_objects:
                track = self._create_track(timestamp, obj)
                track.identity_confidence = round(0.45 + (_scan_quality_factor(scan) * 0.35), 2)
                track.identity_diagnostics = self._build_identity_diagnostics(
                    score_value=track.identity_confidence,
                    scan=scan,
                    track=track,
                    reason="first-scan track initialization",
                    event_context="initial",
                )
                self._obj_to_track[obj.object_id] = track.track_id
            self._refresh_track_motions(field_estimates=None, field_dt_hours=0.0)
            self._update_primary_focus()
            self._remember_scan(scan)
            return

        new_objects = {obj.object_id: obj for obj in scan.detected_objects}
        association = associate_tracks(
            previous_scan=previous_scan,
```

Then replace the final `self._prev_scan = scan` at the end of `update` with `self._remember_scan(scan)`. Confirm with `grep -n "_prev_scan\b" src/tracker.py` that `_prev_scan` is now used only in `__init__`, `_reset_for_site`, `_previous_scan`, `_remember_scan` and `release_previous_scan`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_tracker_persistence.py tests/unit/test_tracker.py tests/unit/test_tracking_association.py tests/unit/test_tracking_focus.py -v`
Expected: all PASS.

- [ ] **Step 5: Run the suite and commit**

Run: `.venv/Scripts/python.exe -m pytest tests/unit tests/smoke -q`
Expected: all pass.

```bash
git add src/tracker.py tests/unit/test_tracker_persistence.py
git commit -m "$(cat <<'EOF'
Let the storm tracker reload its previous scan and resume from state

With a scan loader the tracker keeps only the previous scan's site and
timestamp, reloading the scan at the next update. Exported state
restores a tracker that continues exactly like an uninterrupted one.
Without a loader behavior is unchanged.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_011VMHfi1nbuux7187WtgQTt
EOF
)"
```

---

### Task 6: Per-radar history ring, persistence, rebuild, and radar LRU

**Files:**
- Create: `src/history/store.py`
- Test: `tests/unit/test_history_store.py`

**Interfaces:**
- Consumes: Tasks 3–5.
- Produces:
  - `RING_SIZE = 5`, `TRACKER_STATE_FILENAME = "tracker_state.json"`
  - `RadarHistory(site_id: str, directory: Path | None, *, ring_size: int = RING_SIZE)` with:
    - attributes `site_id`, `directory`, `ring_size`, `lock: RLock`, `tracker: StormTracker`, `last_write_error: str | None`
    - `RadarHistory.open(site_id, directory, *, ring_size=RING_SIZE) -> RadarHistory`
    - `.scans() -> list[CompactScan]` (oldest first), `.newest() -> CompactScan | None`
    - `.find_by_timestamp(when: datetime) -> CompactScan | None`, `.find_by_source(source_path: str) -> CompactScan | None`
    - `.add_live_scan(buffered: BufferedScan) -> CompactScan | None`
    - overridable `._new_tracker() -> StormTracker` and `._restore_tracker(state: dict) -> StormTracker`
  - `HistoryRegistry(root: Path | None, *, max_radars: int = 20, ring_size: int = RING_SIZE)` with `.use(site_id)` (context manager yielding a locked `RadarHistory`), `.loaded_sites() -> list[str]`, `.clear() -> None`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_history_store.py`:

```python
from datetime import datetime, timedelta
from threading import Event, Thread

import pytest

import src.history.store as store
from src.history.disk import HISTORY_SUFFIX
from src.history.records import dumps
from src.history.store import TRACKER_STATE_FILENAME, HistoryRegistry, RadarHistory
from tests.unit.test_tracker import _make_object, _make_scan

T0 = datetime(2026, 9, 12, 18, 0)


def _scan(minute: int, lon_shift: float = 0.0, site_id: str = "KTLX"):
    return _make_scan(
        site_id,
        T0 + timedelta(minutes=minute),
        [_make_object(1, 35.3, -97.30 + lon_shift), _make_object(2, 35.0, -97.0)],
    )


def _feed(history: RadarHistory, minutes):
    return [history.add_live_scan(_scan(minute, minute * 0.001)) for minute in minutes]


def test_ring_keeps_only_the_newest_scans_in_memory_and_on_disk(tmp_path):
    history = RadarHistory.open("KTLX", tmp_path, ring_size=5)
    _feed(history, [0, 5, 10, 15, 20, 25])
    assert [scan.timestamp for scan in history.scans()] == [
        T0 + timedelta(minutes=m) for m in (5, 10, 15, 20, 25)
    ]
    assert len(list(tmp_path.glob(f"*{HISTORY_SUFFIX}"))) == 5
    assert (tmp_path / TRACKER_STATE_FILENAME).is_file()


def test_every_retained_scan_is_tracked_and_newest_is_live(tmp_path):
    history = RadarHistory.open("KTLX", tmp_path)
    _feed(history, [0, 5, 10])
    assert all(scan.tracked for scan in history.scans())
    newest = history.newest().to_buffered_scan()
    assert [len(track.positions) for track in newest.tracking.active_tracks] == [3, 3]
    assert history.tracker._prev_scan is None, "full previous scan must not be retained"


def test_scan_not_newer_than_the_newest_is_refused_and_not_tracked(tmp_path):
    history = RadarHistory.open("KTLX", tmp_path)
    _feed(history, [0, 5])
    before = dumps(history.tracker.export_state())
    assert history.add_live_scan(_scan(5)) is None
    assert history.add_live_scan(_scan(3)) is None
    assert dumps(history.tracker.export_state()) == before


def test_find_by_timestamp_and_source(tmp_path):
    history = RadarHistory.open("KTLX", tmp_path)
    scan = _scan(0)
    scan.source_path = "C:/cache/KTLX/KTLX20260912_180000_V06"
    history.add_live_scan(scan)
    assert history.find_by_timestamp(T0).timestamp == T0
    assert history.find_by_timestamp(T0 + timedelta(seconds=1)) is None
    assert history.find_by_source(scan.source_path).timestamp == T0


def test_restart_with_saved_state_continues_uninterrupted(tmp_path):
    uninterrupted = RadarHistory.open("KTLX", tmp_path / "a")
    _feed(uninterrupted, [0, 5, 10, 15])

    first = RadarHistory.open("KTLX", tmp_path / "b")
    _feed(first, [0, 5])
    restarted = RadarHistory.open("KTLX", tmp_path / "b")
    _feed(restarted, [10, 15])

    assert dumps(restarted.tracker.export_state()) == dumps(uninterrupted.tracker.export_state())
    assert not restarted.newest().to_buffered_scan().tracking.continuity_rebuilt


def test_stale_tracker_state_triggers_a_flagged_rebuild(tmp_path):
    history = RadarHistory.open("KTLX", tmp_path / "current")
    _feed(history, [0, 5, 10])
    older = RadarHistory.open("KTLX", tmp_path / "older")
    _feed(older, [0, 5])
    # State saved before the newest scan was written.
    (tmp_path / "current" / TRACKER_STATE_FILENAME).write_bytes(
        (tmp_path / "older" / TRACKER_STATE_FILENAME).read_bytes()
    )

    rebuilt = RadarHistory.open("KTLX", tmp_path / "current")

    snapshots = [scan.to_buffered_scan().tracking for scan in rebuilt.scans()]
    assert all(snapshot.continuity_rebuilt for snapshot in snapshots)
    assert [len(t.positions) for t in snapshots[-1].active_tracks] == [3, 3]
    assert rebuilt.tracker.last_scan_timestamp == T0 + timedelta(minutes=10)


@pytest.mark.parametrize("damage", ["missing", "garbage"])
def test_missing_or_unreadable_tracker_state_rebuilds(tmp_path, damage):
    history = RadarHistory.open("KTLX", tmp_path)
    _feed(history, [0, 5])
    state_path = tmp_path / TRACKER_STATE_FILENAME
    if damage == "missing":
        state_path.unlink()
    else:
        state_path.write_text("{not json", encoding="utf-8")
    rebuilt = RadarHistory.open("KTLX", tmp_path)
    assert rebuilt.newest().to_buffered_scan().tracking.continuity_rebuilt


def test_corrupt_snapshot_is_skipped_and_deleted(tmp_path):
    history = RadarHistory.open("KTLX", tmp_path)
    _feed(history, [0, 5, 10])
    damaged = sorted(tmp_path.glob(f"*{HISTORY_SUFFIX}"))[1]
    damaged.write_bytes(b"not an archive")

    reopened = RadarHistory.open("KTLX", tmp_path)

    assert [scan.timestamp for scan in reopened.scans()] == [T0, T0 + timedelta(minutes=10)]
    assert not damaged.exists()


def test_write_failure_is_reported_and_live_tracking_continues(tmp_path, monkeypatch):
    history = RadarHistory.open("KTLX", tmp_path)

    def failing_save(scan, directory):
        raise OSError("disk full")

    monkeypatch.setattr(store, "save_compact_scan", failing_save)
    _feed(history, [0, 5])
    assert history.last_write_error == "OSError: disk full"
    assert len(history.scans()) == 2
    assert [len(t.positions) for t in history.tracker.active_tracks] == [2, 2]


def test_in_memory_history_never_touches_disk(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    history = RadarHistory.open("KTLX", None)
    _feed(history, [0, 5])
    assert len(history.scans()) == 2
    assert list(tmp_path.iterdir()) == []


def test_registry_evicts_least_recently_used_idle_radar_and_reloads_it(tmp_path):
    registry = HistoryRegistry(tmp_path, max_radars=2)
    with registry.use("KTLX") as history:
        history.add_live_scan(_scan(0))
    with registry.use("KEMX"):
        pass
    with registry.use("KIWA"):
        pass
    assert registry.loaded_sites() == ["KEMX", "KIWA"]
    with registry.use("ktlx") as reloaded:
        assert [scan.timestamp for scan in reloaded.scans()] == [T0]


def test_registry_never_evicts_a_radar_in_use(tmp_path):
    registry = HistoryRegistry(tmp_path, max_radars=1)
    holding, release = Event(), Event()

    def hold_ktlx():
        with registry.use("KTLX"):
            holding.set()
            release.wait(2)

    worker = Thread(target=hold_ktlx)
    worker.start()
    try:
        assert holding.wait(2)
        with registry.use("KEMX"):
            # Both radars are in use: neither may be evicted, even though
            # the registry is over its limit until one is released.
            assert registry.loaded_sites() == ["KTLX", "KEMX"]
        assert "KTLX" in registry.loaded_sites()
    finally:
        release.set()
        worker.join(2)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_history_store.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.history.store'`.

- [ ] **Step 3: Implement the store**

Create `src/history/store.py`:

```python
"""Per-radar compact scan history: ring, tracker, persistence and rebuild.

A RadarHistory owns one radar's newest tracked scans and the tracker that
produced them.  The tracker never retains a full previous scan; it reloads
it from the ring.  Every retained scan carries the tracking snapshot from its
own time.  On disk each scan is one `.arwscan` file next to the tracker
state; after a restart the state is used only if it matches the newest
retained scan, otherwise continuity is rebuilt from the ring and flagged.
"""

from collections import OrderedDict
from contextlib import contextmanager
from datetime import datetime
import logging
from pathlib import Path
from threading import RLock

from src.buffer import BufferedScan
from src.history import records as record_codec
from src.history.compact_scan import CompactScan, timestamp_key, verify_label_masks
from src.history.disk import (
    HISTORY_SUFFIX,
    CompactScanLoadError,
    atomic_write_bytes,
    history_filename,
    load_compact_scan,
    save_compact_scan,
)
from src.tracker import StormTracker

_logger = logging.getLogger(__name__)

RING_SIZE = 5
TRACKER_STATE_FILENAME = "tracker_state.json"


def _unlink_quietly(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        _logger.warning("Unable to delete radar history file %s", path, exc_info=True)


class RadarHistory:
    def __init__(self, site_id: str, directory: Path | None, *, ring_size: int = RING_SIZE):
        self.site_id = site_id.upper()
        self.directory = Path(directory) if directory is not None else None
        self.ring_size = max(1, int(ring_size))
        self.lock = RLock()
        self.last_write_error: str | None = None
        self._ring: list[CompactScan] = []
        self.tracker = self._new_tracker()

    @classmethod
    def open(cls, site_id: str, directory: Path | None, *, ring_size: int = RING_SIZE) -> "RadarHistory":
        history = cls(site_id, directory, ring_size=ring_size)
        if history.directory is not None and history.directory.is_dir():
            with history.lock:
                history._restore_from_disk()
        return history

    def _new_tracker(self) -> StormTracker:
        return StormTracker(scan_loader=self._load_for_tracker)

    def _restore_tracker(self, state: dict) -> StormTracker:
        return StormTracker.from_state(state, self._load_for_tracker)

    # -- reads -----------------------------------------------------------

    def scans(self) -> list[CompactScan]:
        with self.lock:
            return list(self._ring)

    def newest(self) -> CompactScan | None:
        with self.lock:
            return self._ring[-1] if self._ring else None

    def find_by_timestamp(self, when: datetime) -> CompactScan | None:
        key = timestamp_key(when)
        with self.lock:
            return next((scan for scan in self._ring if timestamp_key(scan.timestamp) == key), None)

    def find_by_source(self, source_path: str) -> CompactScan | None:
        with self.lock:
            return next((scan for scan in self._ring if scan.source_path == source_path), None)

    def _load_for_tracker(self, site_id: str, timestamp: datetime) -> BufferedScan | None:
        if site_id.upper() != self.site_id:
            return None
        compact = self.find_by_timestamp(timestamp)
        return compact.to_buffered_scan() if compact is not None else None

    # -- live path -------------------------------------------------------

    def add_live_scan(self, buffered: BufferedScan) -> CompactScan | None:
        """Track and retain this radar's newest volume.

        Returns None, without tracking, for a volume that is not newer than
        the newest retained scan.
        """
        with self.lock:
            newest = self.newest()
            if newest is not None and timestamp_key(buffered.timestamp) <= timestamp_key(newest.timestamp):
                return None
            verify_label_masks(buffered)  # refuse before the tracker advances
            self.tracker.update(buffered)
            for obj in buffered.detected_objects:
                obj.temporal_status = self.tracker.temporal_status_for_current_object(obj.object_id)
            buffered.tracking = self.tracker.snapshot()
            compact = CompactScan.from_buffered_scan(buffered)
            self._ring.append(compact)
            evicted = self._ring[: -self.ring_size]
            self._ring = self._ring[-self.ring_size:]
            self.tracker.release_previous_scan()
            self._persist([compact], evicted)
            return compact

    # -- persistence -----------------------------------------------------

    def _persist(self, written: list[CompactScan], evicted: list[CompactScan]) -> None:
        if self.directory is None:
            return
        try:
            for scan in written:
                save_compact_scan(scan, self.directory)
            self._write_tracker_state()
        except OSError as exc:
            _logger.exception("Unable to write radar history for %s", self.site_id)
            self.last_write_error = f"{type(exc).__name__}: {exc}"
            return
        self.last_write_error = None
        for scan in evicted:
            _unlink_quietly(self.directory / history_filename(scan.timestamp))

    def _write_tracker_state(self) -> None:
        document = {
            "site_id": self.site_id,
            "last_scan_timestamp": self.tracker.last_scan_timestamp,
            "state": self.tracker.export_state(),
        }
        atomic_write_bytes(
            self.directory / TRACKER_STATE_FILENAME,
            record_codec.dumps(document).encode("utf-8"),
        )

    def _restore_from_disk(self) -> None:
        for leftover in self.directory.glob("*.tmp"):
            _unlink_quietly(leftover)
        loaded: list[CompactScan] = []
        for path in sorted(self.directory.glob(f"*{HISTORY_SUFFIX}")):
            try:
                scan = load_compact_scan(path)
            except CompactScanLoadError as exc:
                _logger.warning("Discarding unreadable radar history file %s: %s", path, exc)
                _unlink_quietly(path)
                continue
            if scan.site_id != self.site_id:
                _logger.warning("Ignoring %s history file %s in %s's directory", scan.site_id, path, self.site_id)
                continue
            loaded.append(scan)
        loaded.sort(key=lambda scan: timestamp_key(scan.timestamp))
        for stale in loaded[: -self.ring_size]:
            _unlink_quietly(self.directory / history_filename(stale.timestamp))
        self._ring = loaded[-self.ring_size:]
        if not self._ring:
            return
        restored = self._load_saved_tracker()
        if restored is not None:
            self.tracker = restored
        else:
            self._rebuild_tracking()

    def _load_saved_tracker(self) -> StormTracker | None:
        path = self.directory / TRACKER_STATE_FILENAME
        try:
            document = record_codec.loads(path.read_text(encoding="utf-8"))
            if document["site_id"] != self.site_id:
                raise ValueError(f"state belongs to {document['site_id']}")
            saved_at = document["last_scan_timestamp"]
            newest_at = self._ring[-1].timestamp
            if saved_at is None or timestamp_key(saved_at) != timestamp_key(newest_at):
                raise ValueError(f"state was saved at {saved_at}, newest retained scan is {newest_at}")
            return self._restore_tracker(document["state"])
        except Exception as exc:  # missing, unreadable, stale or incompatible
            _logger.warning(
                "Saved tracker state for %s is unusable (%s); rebuilding continuity from retained scans",
                self.site_id, exc,
            )
            return None

    def _rebuild_tracking(self) -> None:
        self.tracker = self._new_tracker()
        for index, compact in enumerate(list(self._ring)):
            buffered = compact.to_buffered_scan()
            self.tracker.update(buffered)
            for obj in buffered.detected_objects:
                obj.temporal_status = self.tracker.temporal_status_for_current_object(obj.object_id)
            snapshot = self.tracker.snapshot()
            snapshot.continuity_rebuilt = True
            buffered.tracking = snapshot
            self._ring[index] = CompactScan.from_buffered_scan(buffered)
            self.tracker.release_previous_scan()
        self._persist(list(self._ring), [])


class HistoryRegistry:
    """Radar histories in memory, least recently used evicted first.

    An evicted radar's ring and tracker state reload from disk on next use.
    A radar whose lock is held (in use) is never evicted.
    """

    def __init__(self, root: Path | None, *, max_radars: int = 20, ring_size: int = RING_SIZE):
        self._root = Path(root) if root is not None else None
        self._max_radars = max(1, int(max_radars))
        self._ring_size = ring_size
        self._histories: OrderedDict[str, RadarHistory] = OrderedDict()
        self._lock = RLock()

    @contextmanager
    def use(self, site_id: str):
        key = site_id.upper()
        with self._lock:
            history = self._histories.get(key)
            if history is None:
                directory = None if self._root is None else self._root / key / "history"
                history = RadarHistory.open(key, directory, ring_size=self._ring_size)
                self._histories[key] = history
            self._histories.move_to_end(key)
            history.lock.acquire()
            self._evict_idle(in_use=key)
        try:
            yield history
        finally:
            history.lock.release()

    def _evict_idle(self, in_use: str) -> None:
        for key in list(self._histories):
            if len(self._histories) <= self._max_radars:
                return
            if key == in_use:
                # This thread holds its lock, and RLock is reentrant, so a
                # non-blocking acquire would wrongly succeed.
                continue
            candidate = self._histories[key]
            if candidate.lock.acquire(blocking=False):
                try:
                    del self._histories[key]
                finally:
                    candidate.lock.release()

    def loaded_sites(self) -> list[str]:
        with self._lock:
            return list(self._histories)

    def clear(self) -> None:
        with self._lock:
            self._histories.clear()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_history_store.py -v`
Expected: all PASS.

`_evict_idle` runs while the calling thread holds the just-used radar's lock. Because `RLock` is reentrant, a non-blocking acquire on that radar would succeed, so it's skipped explicitly by key. `test_registry_never_evicts_a_radar_in_use` fails without that skip: the older radar is busy in another thread, and the loop reaches the in-use one.

- [ ] **Step 5: Commit**

```bash
git add src/history/store.py tests/unit/test_history_store.py
git commit -m "$(cat <<'EOF'
Add per-radar scan history with persistence and radar LRU

RadarHistory keeps the five newest tracked scans and their tracker,
writes each scan and the tracker state atomically, and on reopen uses
saved state only when it matches the newest scan, otherwise rebuilding
continuity and flagging it. Corrupt files are skipped and deleted, write
failures are reported without stopping tracking, and HistoryRegistry
evicts idle radars least recently used.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_011VMHfi1nbuux7187WtgQTt
EOF
)"
```

---

### Task 7: Server uses the history registry; full-scan caches removed

**Files:**
- Create: `src/history/layer_cache.py`
- Create: `tests/conftest.py`
- Modify: `src/server.py` (globals, `_site_tracker`, `_track_live_scan`, `_ingest_to_buffer`, `_live_or_ingest`, `_refresh_live_scan`, rendered-layer helpers, `get_live_scan_status`, `get_map_status`, `get_motion`)
- Modify: `src/buffer.py` (remove `ReplayBuffer`)
- Modify: `tests/unit/test_buffer.py` (remove `ReplayBuffer` tests), `tests/unit/test_live_cache.py`, `tests/e2e/test_full_pipeline.py`
- Test: `tests/unit/test_history_layer_cache.py` (create), `tests/unit/test_server_history.py` (create)

**Interfaces:**
- Consumes: Task 6 `HistoryRegistry`, `RadarHistory`; Task 3 `CompactScan`.
- Produces:
  - `RenderedLayerCache(max_unpinned_scans: int = 10)` with `.get(key) -> dict | None`, `.put(key, layers, *, pinned: bool) -> None`, `__contains__`, `__len__`, `.clear()`
  - Server:
    - `_histories: HistoryRegistry`, `_historical_scans: OrderedDict[tuple[str, str], CompactScan]`, `_map_layers: RenderedLayerCache`
    - `_newest_live_scan(site_id) -> CompactScan | None`
    - `_track_live_scan(site_id, buffered) -> CompactScan | None`
    - `_ingest_to_buffer(site_id, dt=None) -> BufferedScan` (the `publish_live` parameter is removed)

- [ ] **Step 1: Write the failing layer-cache tests**

Create `tests/unit/test_history_layer_cache.py`:

```python
from src.history.layer_cache import RenderedLayerCache


def test_unpinned_scans_are_evicted_least_recently_used():
    cache = RenderedLayerCache(max_unpinned_scans=2)
    cache.put(("KTLX", "a"), {"audiom": {"a": 1}}, pinned=False)
    cache.put(("KTLX", "b"), {"audiom": {}}, pinned=False)
    assert cache.get(("KTLX", "a")) == {"audiom": {"a": 1}}  # refreshes "a"
    cache.put(("KTLX", "c"), {"audiom": {}}, pinned=False)
    assert ("KTLX", "b") not in cache
    assert ("KTLX", "a") in cache and ("KTLX", "c") in cache


def test_newest_scan_per_site_is_never_evicted():
    cache = RenderedLayerCache(max_unpinned_scans=1)
    cache.put(("KTLX", "newest"), {"audiom": {}}, pinned=True)
    cache.put(("KEMX", "newest"), {"audiom": {}}, pinned=True)
    for name in ("x", "y", "z"):
        cache.put(("KIWA", name), {"audiom": {}}, pinned=False)
    assert ("KTLX", "newest") in cache and ("KEMX", "newest") in cache
    assert len(cache) == 3


def test_pinning_a_newer_scan_unpins_the_previous_one():
    cache = RenderedLayerCache(max_unpinned_scans=0)
    cache.put(("KTLX", "first"), {"audiom": {}}, pinned=True)
    cache.put(("KTLX", "second"), {"audiom": {}}, pinned=True)
    assert ("KTLX", "first") not in cache
    assert ("KTLX", "second") in cache
```

- [ ] **Step 2: Implement the layer cache and run its tests**

Create `src/history/layer_cache.py`:

```python
"""Bounded cache of rendered GeoJSON map layers.

Each radar's newest retained scan is pinned and never evicted, since that is
what live maps request.  Other rendered scans (stepped-back or historical)
are kept least recently used up to a limit; an evicted rendering is rebuilt
identically from its compact scan on demand.
"""

from collections import OrderedDict
from threading import RLock

LayerKey = tuple[str, str]  # (site_id, source volume path)


class RenderedLayerCache:
    def __init__(self, max_unpinned_scans: int = 10):
        self._max_unpinned = max(0, int(max_unpinned_scans))
        self._entries: OrderedDict[LayerKey, dict[str, dict]] = OrderedDict()
        self._pinned: dict[str, LayerKey] = {}
        self._lock = RLock()

    def get(self, key: LayerKey) -> dict[str, dict] | None:
        with self._lock:
            layers = self._entries.get(key)
            if layers is not None:
                self._entries.move_to_end(key)
            return layers

    def put(self, key: LayerKey, layers: dict[str, dict], *, pinned: bool) -> None:
        with self._lock:
            self._entries[key] = layers
            self._entries.move_to_end(key)
            if pinned:
                self._pinned[key[0]] = key
            pinned_keys = set(self._pinned.values())
            unpinned = [entry for entry in self._entries if entry not in pinned_keys]
            for entry in unpinned[: max(0, len(unpinned) - self._max_unpinned)]:
                del self._entries[entry]

    def __contains__(self, key: LayerKey) -> bool:
        with self._lock:
            return key in self._entries

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._pinned.clear()
```

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_history_layer_cache.py -v`
Expected: all PASS.

- [ ] **Step 3: Keep tests off the real cache directory**

Create `tests/conftest.py`:

```python
import pytest

import src.server as server
from src.history.store import HistoryRegistry


@pytest.fixture(autouse=True)
def _in_memory_radar_history(monkeypatch):
    """No test may write radar history into the repository's cache/ directory.

    Tests that need persistence construct their own registry or RadarHistory
    on tmp_path.
    """
    monkeypatch.setattr(server, "_histories", HistoryRegistry(None))
```

- [ ] **Step 4: Write the failing server tests**

Create `tests/unit/test_server_history.py`:

```python
from datetime import datetime, timedelta
from pathlib import Path

import pytest

import src.server as server
from tests.unit.test_tracker import _make_object, _make_scan

T0 = datetime(2026, 9, 12, 18, 0)


@pytest.fixture(autouse=True)
def _clean_caches():
    server._historical_scans.clear()
    server._map_layers.clear()
    yield
    server._historical_scans.clear()
    server._map_layers.clear()


@pytest.fixture
def volumes(tmp_path, monkeypatch):
    """Real files standing in for downloaded volumes, with scripted analysis."""
    scans, processed = {}, []

    def add(name: str, minute: int) -> str:
        path = tmp_path / name
        path.write_bytes(b"volume")
        key = str(path.resolve())
        scan = _make_scan("KTLX", T0 + timedelta(minutes=minute), [_make_object(1, 35.3, -97.3)])
        scan.source_path = key
        scans[key] = scan
        return key

    def process(site_id, filepath):
        key = str(Path(filepath).resolve())
        processed.append(key)
        return scans[key]

    monkeypatch.setattr(server, "_process_scan_file", process)
    return add, processed


def _serve_live(monkeypatch, paths):
    order = iter(paths)
    monkeypatch.setattr(server, "fetch_scan", lambda site_id, dt=None: next(order))


def test_live_volumes_enter_the_ring_with_tracking(monkeypatch, volumes):
    add, _ = volumes
    _serve_live(monkeypatch, [add("v1", 0), add("v2", 5)])
    server._ingest_to_buffer("KTLX")
    latest = server._ingest_to_buffer("KTLX")
    assert [len(t.positions) for t in latest.tracking.active_tracks] == [2]
    with server._histories.use("KTLX") as history:
        assert [scan.timestamp for scan in history.scans()] == [T0, T0 + timedelta(minutes=5)]


def test_exact_retained_timestamp_is_served_without_fetching(monkeypatch, volumes):
    add, processed = volumes
    _serve_live(monkeypatch, [add("v1", 0), add("v2", 5)])
    server._ingest_to_buffer("KTLX")
    server._ingest_to_buffer("KTLX")
    monkeypatch.setattr(server, "fetch_scan", lambda *a, **k: pytest.fail("must not fetch"))

    older = server._ingest_to_buffer("KTLX", T0)

    assert older.timestamp == T0
    assert [len(t.positions) for t in older.tracking.active_tracks] == [1]
    assert len(processed) == 2


def test_nearest_volume_that_is_retained_is_served_from_the_ring(monkeypatch, volumes):
    add, processed = volumes
    first = add("v1", 0)
    _serve_live(monkeypatch, [first])
    server._ingest_to_buffer("KTLX")
    monkeypatch.setattr(server, "fetch_scan", lambda site_id, dt=None: first)

    served = server._ingest_to_buffer("KTLX", T0 + timedelta(minutes=1))

    assert served.tracking is not None
    assert len(processed) == 1


def test_historical_volume_is_processed_once_and_never_tracked(monkeypatch, volumes):
    add, processed = volumes
    old = add("old", -60)
    monkeypatch.setattr(server, "fetch_scan", lambda site_id, dt=None: old)

    first = server._ingest_to_buffer("KTLX", T0 - timedelta(hours=1))
    second = server._ingest_to_buffer("KTLX", T0 - timedelta(hours=1))

    assert first.tracking is None and second.tracking is None
    assert processed == [old]
    with server._histories.use("KTLX") as history:
        assert history.scans() == []
        assert history.tracker.last_scan_timestamp is None


def test_historical_cache_is_bounded(monkeypatch, volumes):
    add, _ = volumes
    monkeypatch.setattr(server, "_max_historical_scans", 2)
    paths = [add(f"old{i}", -60 - i) for i in range(3)]
    for path in paths:
        monkeypatch.setattr(server, "fetch_scan", lambda site_id, dt=None, path=path: path)
        server._ingest_to_buffer("KTLX", T0 - timedelta(hours=2))
    assert len(server._historical_scans) == 2
    assert ("KTLX", paths[0]) not in server._historical_scans


def test_live_request_serves_newest_retained_scan_and_queues_refresh(monkeypatch, volumes):
    add, _ = volumes
    _serve_live(monkeypatch, [add("v1", 0)])
    server._ingest_to_buffer("KTLX")
    scheduled = []
    monkeypatch.setattr(server, "_schedule_live_refresh", lambda site_id: scheduled.append(site_id) or True)

    scan, updating = server._live_or_ingest("ktlx")

    assert scan.timestamp == T0 and updating is True and scheduled == ["KTLX"]


def test_live_status_reports_newest_retained_scan(monkeypatch, volumes):
    add, _ = volumes
    _serve_live(monkeypatch, [add("v1", 0)])
    server._ingest_to_buffer("KTLX")
    status = server.get_live_scan_status("KTLX")
    assert status["available"] is True
    assert status["scan_timestamp"] == T0.isoformat()
```

- [ ] **Step 5: Run them to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_server_history.py -v`
Expected: FAIL with `AttributeError: module 'src.server' has no attribute '_historical_scans'`.

- [ ] **Step 6: Integrate the registry into the server**

In `src/server.py`:

1. Imports. Replace `from src.buffer import ReplayBuffer, BufferedScan, TrackingSnapshot` with `from src.buffer import BufferedScan, TrackingSnapshot`, change `from src.ingest import fetch_scan` to `from src.ingest import CACHE_DIR, fetch_scan`, and add:

```python
from src.history.compact_scan import CompactScan
from src.history.layer_cache import RenderedLayerCache
from src.history.store import RING_SIZE, HistoryRegistry
```

2. Globals. Delete `_buffers`, `_trackers`, `_processed_scans`, the `_map_layers` dict, `_live_scans`, `_processed_scans_per_site`, `_max_processed_scans`, `_replay_scans_per_site` and their comments. Add in their place:

```python
# Each radar keeps its newest tracked scans as compact, lossless records
# (sizes measured in docs/test_reports) plus its tracker, persisted under
# cache/<site>/history/.
# Full-size grids exist only while a request or tracker update needs them.
_history_root = Path(os.getenv("ARW_HISTORY_ROOT", CACHE_DIR))
_histories = HistoryRegistry(
    _history_root,
    max_radars=int(os.getenv("ARW_MAX_RADARS_IN_MEMORY", "20")),
    ring_size=int(os.getenv("ARW_SCANS_PER_RADAR", str(RING_SIZE))),
)
# Volumes requested by time that are not in a radar's ring.  Never tracked.
_historical_scans: OrderedDict[tuple[str, str], CompactScan] = OrderedDict()
_max_historical_scans = int(os.getenv("ARW_MAX_HISTORICAL_SCANS", "12"))
_map_layers = RenderedLayerCache(
    max_unpinned_scans=int(os.getenv("ARW_MAX_RENDERED_SCANS", "10"))
)
```

3. Replace `_site_buffer` and `_site_tracker` with:

```python
def _site_tracker(site_id: str) -> StormTracker:
    with _histories.use(site_id) as history:
        return history.tracker


def _newest_live_scan(site_id: str) -> CompactScan | None:
    with _histories.use(site_id) as history:
        return history.newest()
```

4. Replace `_track_live_scan` with:

```python
def _track_live_scan(site_id: str, buffered: BufferedScan) -> CompactScan | None:
    """Track and retain a site's newest live volume.

    Returns None for a volume that is not newer than the newest retained
    scan; such a volume is never tracked (tracking an older volume would
    reset every live track for the site).
    """
    with _histories.use(site_id) as history:
        return history.add_live_scan(buffered)
```

5. Delete `_remember_processed_scan`. Replace `_ingest_to_buffer` with:

```python
def _source_key(filepath: str) -> str | None:
    """Identity of a real downloaded volume.  Test doubles are never shared."""
    path = Path(filepath)
    return str(path.resolve()) if path.is_file() else None


def _remember_historical_scan(key: tuple[str, str], compact: CompactScan) -> None:
    with _state_lock:
        _historical_scans[key] = compact
        _historical_scans.move_to_end(key)
        while len(_historical_scans) > max(1, _max_historical_scans):
            _historical_scans.popitem(last=False)


def _ingest_to_buffer(site_id: str, dt: datetime | None = None) -> BufferedScan:
    """Return a completed interpretation, analyzing each local volume once.

    A time that exactly matches a retained scan is served from the ring with
    no network call.  Only a live request (dt is None) for a volume newer
    than the newest retained one advances tracking.  Everything else is the
    historical path: analyzed once, cached compactly, never tracked.
    """
    normalized_site = site_id.upper()
    if dt is not None:
        with _histories.use(normalized_site) as history:
            retained = history.find_by_timestamp(dt)
        if retained is not None:
            return retained.to_buffered_scan()

    with _site_ingest_lock(normalized_site):
        filepath = fetch_scan(normalized_site, dt)
        source_key = _source_key(filepath)
        historical: CompactScan | None = None
        if source_key is not None:
            with _histories.use(normalized_site) as history:
                retained = history.find_by_source(source_key)
            if retained is not None:
                return retained.to_buffered_scan()
            with _state_lock:
                historical = _historical_scans.get((normalized_site, source_key))
                if historical is not None:
                    _historical_scans.move_to_end((normalized_site, source_key))

        buffered = (
            historical.to_buffered_scan()
            if historical is not None
            else _process_scan_file(normalized_site, filepath)
        )
        if dt is None and _track_live_scan(normalized_site, buffered) is not None:
            if source_key is not None:
                with _state_lock:
                    _historical_scans.pop((normalized_site, source_key), None)
            return buffered
        if source_key is not None and historical is None:
            _remember_historical_scan(
                (normalized_site, source_key), CompactScan.from_buffered_scan(buffered)
            )
        return buffered
```

6. Replace the body of `_live_or_ingest` after its docstring with:

```python
    normalized_site = site_id.upper()
    if dt is not None:
        return _ingest_to_buffer(normalized_site, dt), False
    newest = _newest_live_scan(normalized_site)
    if newest is None:
        return _ingest_to_buffer(normalized_site), False
    return newest.to_buffered_scan(), _schedule_live_refresh(normalized_site)
```

7. In `_refresh_live_scan`, replace

```python
        buffered = _ingest_to_buffer(normalized_site, publish_live=False)
        with _state_lock:
            _live_scans[normalized_site] = buffered
            _refresh_errors.pop(normalized_site, None)
```

with

```python
        _ingest_to_buffer(normalized_site)
        with _state_lock:
            _refresh_errors.pop(normalized_site, None)
```

8. Rendered layers. Add:

```python
def _is_newest_retained(buffered) -> bool:
    source_path = getattr(buffered, "source_path", None)
    newest = _newest_live_scan(buffered.site_id)
    return newest is not None and source_path is not None and newest.source_path == source_path
```

In `_prepared_map_layers`, replace each

```python
        with _state_lock:
            cached = _map_layers.get(cache_key)
```

(both the first one and the one inside `with _map_build_lock:`) with `cached = _map_layers.get(cache_key)` at the same indentation. Then replace

```python
        if cache_key is not None:
            with _state_lock:
                _map_layers[cache_key] = layers
```

with

```python
        if cache_key is not None:
            _map_layers.put(cache_key, layers, pinned=_is_newest_retained(buffered))
```

Replace `_prepared_precipitation_layer` with:

```python
def _prepared_precipitation_layer(buffered: BufferedScan) -> dict:
    """Build the much larger whole-field layer only when it is requested."""
    cache_key = _map_layer_cache_key(buffered)
    if cache_key is not None:
        cached = (_map_layers.get(cache_key) or {}).get("precipitation")
        if cached is not None:
            return cached

    with _map_build_lock:
        if cache_key is not None:
            cached = (_map_layers.get(cache_key) or {}).get("precipitation")
            if cached is not None:
                return cached
        precipitation = build_precipitation_field_geojson(buffered)
        if cache_key is not None:
            layers = dict(_map_layers.get(cache_key) or {})
            layers["precipitation"] = precipitation
            _map_layers.put(cache_key, layers, pinned=_is_newest_retained(buffered))
        return precipitation
```

9. In `get_live_scan_status`, replace

```python
    with _state_lock:
        completed = _live_scans.get(normalized_site)
```

with

```python
    completed = _newest_live_scan(normalized_site)
    with _state_lock:
```

and replace `timestamp = None if completed is None else completed.reflectivity_data.timestamp` with `timestamp = None if completed is None else completed.scan_timestamp_text`.

10. In `get_map_status`'s latest path, replace

```python
    with _state_lock:
        completed = _live_scans.get(site_id)
```

with

```python
    completed = _newest_live_scan(site_id)
    with _state_lock:
```

and replace `timestamp = None if completed is None else completed.reflectivity_data.timestamp` with `timestamp = None if completed is None else completed.scan_timestamp_text`.

11. In `get_motion`, replace `track = _site_tracker(site_id).get_track(track_id)` with:

```python
    with _histories.use(site_id) as history:
        track = deepcopy(history.tracker.get_track(track_id))
```

12. In `src/buffer.py`, delete the `ReplayBuffer` class and the now-unused `deque` and `timedelta` imports. In `tests/unit/test_buffer.py`, delete every `ReplayBuffer` test and its import. If nothing is left, delete the file.

13. In `tests/unit/test_live_cache.py`, replace `_clear_live_state` with:

```python
def _clear_live_state():
    with server._state_lock:
        server._historical_scans.clear()
        server._refreshing_sites.clear()
        server._refresh_started_at.clear()
        server._refresh_errors.clear()
        server._site_ingest_locks.clear()
    server._map_layers.clear()
    server._histories.clear()
```

Then update the existing tests:

- Delete `test_live_request_returns_completed_scan_and_queues_refresh` (replaced in `test_server_history.py`).
- Delete `test_processed_scan_cache_is_bounded_per_site_and_evicts_map_layers` (replaced by `test_historical_cache_is_bounded` and the layer-cache tests).
- In `test_live_refresh_publishes_completed_scan_without_waiting_for_map_build`, set `ingested = []`, patch `_ingest_to_buffer` with `lambda site_id, *args, **kwargs: ingested.append(site_id)`, and replace the `_live_scans` assertion with `assert ingested == ["KJAX"]`.
- In `test_completed_map_layer_does_not_wait_for_another_site_ingest`, replace the `server._map_layers[...] = expected` assignment (and its `with server._state_lock:`) with `server._map_layers.put(server._map_layer_cache_key(scan), expected, pinned=False)`.

In `tests/e2e/test_full_pipeline.py`, replace `srv._buffers.clear()` and `srv._trackers.clear()` with `srv._histories.clear()`.

- [ ] **Step 7: Run the server tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_server_history.py tests/unit/test_live_cache.py -v`
Expected: all PASS.

- [ ] **Step 8: Verify nothing still references the removed caches**

Run: `grep -rn "_live_scans\|_processed_scans\|_buffers\b\|_trackers\b\|ReplayBuffer\|publish_live\|_remember_processed_scan\|_site_buffer" src scripts tests --include=*.py`
Expected: no output.

- [ ] **Step 9: Run the full suite and commit**

Run: `.venv/Scripts/python.exe -m pytest tests/unit tests/smoke tests/e2e/test_full_pipeline.py tests/e2e/test_proof_precipitation_evidence.py -q`
Expected: all pass.

```bash
git add src/history/layer_cache.py src/server.py src/buffer.py tests/conftest.py tests/unit/test_history_layer_cache.py tests/unit/test_server_history.py tests/unit/test_live_cache.py tests/unit/test_buffer.py tests/e2e/test_full_pipeline.py
git commit -m "$(cat <<'EOF'
Serve radar scans from compact per-radar history

The server's replay buffers, trackers, processed-scan and live-scan
caches are replaced by the history registry, a bounded LRU of compact
historical scans, and a rendered-layer cache that always keeps each
radar's newest scan. A retained scan requested by time is served with
its own tracking snapshot and no network call.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_011VMHfi1nbuux7187WtgQTt
EOF
)"
```

---

### Task 8: Reacquire missing storms and lose tracks only when unrecoverable (defect 2)

> **Revision (owner decision, spec Amendment 2, A9). This overrides the task text below where they differ.**
>
> 1. **Status rule.** In `StormTracker.update`, an active track not matched this scan becomes `status = "missing"` (its `_missed_scans` still increments, and its identity diagnostics are still "track missed a scan"). Delete `MAX_MISSED_SCANS`.
> 2. **Candidates.** `associate_tracks` must run stage 2 even when no track is `active`. Its early return applies only when there are no active *and* no missing tracks. Stage-2 candidates are tracks with `status == "missing"` and a `last_seen_ref`, taken from the full `tracks` argument (not the active list). Stage 1 is unchanged.
> 3. **Unrecoverable tracks.** Add `AssociationResult.unreacquirable_track_ids: set[int]`. Stage 2 adds each candidate whose elapsed time exceeds `MAX_REACQUISITION_MINUTES`, or whose last-seen scan the loader cannot return.
> 4. **Lost.** After applying reacquired matches, the tracker marks a `missing` track `lost` if it is in `unreacquirable_track_ids`, or if more than `MAX_REACQUISITION_MINUTES` have passed since `track.last_seen`. The time rule is applied by the tracker itself, so it also works without reacquisition. A reacquired track's status is set back to `"active"`.
> 4b. **Expiry without a candidate (implementation refinement).** Stage 2 only loads a last-seen scan when some object is unclaimed. A storm whose scan leaves the history while nothing reappears would otherwise stay `missing` until the time limit. `StormTracker.expire_scan(site_id, timestamp)` marks missing tracks last seen in that scan `lost`, and `RadarHistory.add_live_scan` calls it for every scan it evicts. The association-time check remains a backstop, covered by `test_unloadable_last_seen_scan_makes_the_track_lost` and `test_missing_track_is_lost_when_its_scan_leaves_the_ring`.
> 5. **Tests.** Replace `test_lost_track_is_never_reacquired` with two tests:
>    - `test_missing_track_is_not_active_and_not_described`: after one miss, the track is `missing`, is absent from `active_tracks`, and is not picked by the summary.
>    - `test_track_is_lost_when_its_last_seen_scan_leaves_history`: the loader drops the last-seen scan, so the track becomes `lost` and is not reacquired.
>
>    Change `test_storm_unseen_longer_than_the_continuity_limit_is_not_reacquired` to assert `lost` at the scan where 20 minutes is exceeded, and `missing` before it. Add `test_storm_missing_for_two_scans_is_reacquired` (5-minute scans, absent at 5 and 10, back at 15). Update `tests/unit/test_tracker.py::test_tracker_lost_after_missed_scans` so that two empty scans 5 minutes apart leave the track `missing`, and a scan more than 20 minutes after it was last seen makes it `lost`. In `test_a_different_nearby_storm_is_not_taken_for_the_missing_one` and `test_without_reacquisition_the_returning_storm_gets_a_new_identity`, the old track is `missing` (not `lost`) after the scan shown.
> 6. **Task 11 Step 3** no longer expects `IDENTICAL` benchmark output. Record every changed metric, explain each one by the missing-status or reacquisition rule, and put that table in the report. An unexplained difference is a defect.


**Files:**
- Modify: `src/tracking/types.py` (`Track.last_seen_ref`, `Track.add_position`)
- Modify: `src/tracking/events.py` (`normalize_reacquired_event`)
- Modify: `src/tracking/association.py` (`AssociationResult`, `associate_tracks`, new `_reacquire_missed_tracks`)
- Modify: `src/tracker.py` (`reacquire` option, reacquired-match handling)
- Modify: `src/history/store.py` (`_new_tracker`, `_restore_tracker` enable reacquisition)
- Modify: `src/summary.py` (reacquired wording)
- Test: `tests/unit/test_tracking_reacquisition.py` (create), `tests/unit/test_summary.py`, `tests/unit/test_history_store.py`

**Interfaces:**
- Consumes: Task 5 `scan_loader`; Task 6 `RadarHistory`.
- Produces:
  - `Track.last_seen_ref: tuple[datetime, int] | None = None`
  - `normalize_reacquired_event(timestamp: datetime, track_id: int, missed_scans: int) -> dict` (`event_type == "reacquired"`)
  - `MAX_REACQUISITION_MINUTES = 20.0`
  - `AssociationResult.reacquired_matches: dict[int, int]`, `AssociationResult.reacquisition_scores: list[AssociationScore]`
  - `associate_tracks(..., reacquisition_loader: Callable[[datetime], BufferedScan | None] | None = None)`
  - `StormTracker(scan_loader=None, *, reacquire: bool = False)`, `StormTracker.from_state(state, scan_loader, *, reacquire: bool = False)`
  - `REACQUIRED_IDENTITY_CAP = 0.4`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_tracking_reacquisition.py`:

```python
from datetime import datetime, timedelta

import numpy as np
import pytest

from src.tracker import MAX_MISSED_SCANS, MAX_TEMPORAL_CONTINUITY_MINUTES, MEDIUM_CONFIDENCE, StormTracker
from src.tracking.association import MAX_REACQUISITION_MINUTES
from tests.unit.test_tracking_association import _make_object, _make_scan

SHAPE = (360, 500)
T0 = datetime(2026, 9, 12, 18, 0)


def _mask(row: int, col: int, rows: int = 10, cols: int = 10) -> np.ndarray:
    grid = np.zeros(SHAPE, dtype=bool)
    grid[row:row + rows, col:col + cols] = True
    return grid


def _storm_a(object_id, row=50, lat=35.30):
    return _make_object(object_id, lat, -97.50, 55.0), _mask(row, 100)


def _storm_b(object_id):
    return _make_object(object_id, 35.0, -97.0, 45.0), _mask(200, 300)


def _scan(minute, storms):
    objects = [obj for obj, _ in storms]
    masks = {obj.object_id: mask for obj, mask in storms}
    return _make_scan("KTLX", T0 + timedelta(minutes=minute), objects, masks)


class _Recorder:
    """A tracker whose loader serves every scan it has processed."""

    def __init__(self, reacquire=True):
        self.stored = {}
        self.tracker = StormTracker(
            scan_loader=lambda site_id, timestamp: self.stored.get(timestamp),
            reacquire=reacquire,
        )

    def update(self, scan):
        self.tracker.update(scan)
        self.stored[scan.timestamp] = scan


def _storm_a_hidden_for_one_scan(recorder, return_minute=10, gap=5):
    recorder.update(_scan(0, [_storm_a(1), _storm_b(2)]))
    recorder.update(_scan(gap, [_storm_b(1)]))
    recorder.update(_scan(return_minute, [_storm_a(1, row=51, lat=35.301), _storm_b(2)]))


def test_storm_missing_for_one_scan_keeps_its_identity_with_low_confidence():
    recorder = _Recorder()
    _storm_a_hidden_for_one_scan(recorder)
    track = recorder.tracker.get_track(1)

    assert track.status == "active"
    assert [position.latitude for position in track.positions] == [35.30, 35.301]
    assert track.identity_diagnostics.event_context == "reacquired"
    assert track.identity_diagnostics.missed_scans == 1
    assert "1 missed scan" in track.identity_diagnostics.reason
    assert track.identity_confidence < MEDIUM_CONFIDENCE
    assert track.identity_diagnostics.label == "low"
    assert [event["event_type"] for event in recorder.tracker.recent_events] == ["reacquired"]
    assert recorder.tracker.temporal_status_for_current_object(1) == "persistent"
    assert len(recorder.tracker.all_tracks) == 2


def test_a_different_nearby_storm_is_not_taken_for_the_missing_one():
    recorder = _Recorder()
    recorder.update(_scan(0, [_storm_a(1), _storm_b(2)]))
    recorder.update(_scan(5, [_storm_b(1)]))
    nearby = (_make_object(1, 35.28, -97.50, 55.0), _mask(80, 100))  # no overlap with A
    recorder.update(_scan(10, [nearby, _storm_b(2)]))

    assert recorder.tracker.get_track(1).status == "lost"
    assert len(recorder.tracker.all_tracks) == 3
    assert not any(e["event_type"] == "reacquired" for e in recorder.tracker.recent_events)


def test_storm_unseen_longer_than_the_continuity_limit_is_not_reacquired():
    recorder = _Recorder()
    _storm_a_hidden_for_one_scan(recorder, return_minute=30, gap=15)
    assert recorder.tracker.get_track(1).status == "lost"
    assert len(recorder.tracker.all_tracks) == 3


def test_lost_track_is_never_reacquired():
    recorder = _Recorder()
    recorder.update(_scan(0, [_storm_a(1), _storm_b(2)]))
    for minute in range(5, 5 * (MAX_MISSED_SCANS + 1), 5):
        recorder.update(_scan(minute, [_storm_b(1)]))
    assert recorder.tracker.get_track(1).status == "lost"
    recorder.update(_scan(5 * (MAX_MISSED_SCANS + 1), [_storm_a(1, row=51), _storm_b(2)]))
    assert recorder.tracker.get_track(1).status == "lost"
    assert len(recorder.tracker.all_tracks) == 3


def test_continuously_tracked_storm_always_keeps_a_contested_object():
    recorder = _Recorder()
    # A and D have adjacent masks but centroids 16.7 km apart -- beyond the
    # 10 km plausible distance for a 5-minute scan -- so the second scan does
    # not merge A into D.
    storm_a = (_make_object(1, 35.45, -97.50, 55.0), _mask(50, 100))
    storm_d = (_make_object(2, 35.30, -97.48, 50.0), _mask(50, 112))
    recorder.update(_scan(0, [storm_a, storm_d]))
    recorder.update(_scan(5, [(_make_object(1, 35.30, -97.48, 50.0), _mask(50, 112))]))
    # Covers A's last mask and D's mask, centred where A was last seen, so A
    # alone would pass every reacquisition check.  D was tracked continuously.
    contested = (_make_object(1, 35.45, -97.50, 55.0), _mask(50, 100, cols=22))
    recorder.update(_scan(10, [contested]))

    assert recorder.tracker.get_track(2).current_object.centroid_lat == 35.45
    assert recorder.tracker.get_track(1).status == "lost"
    assert not any(e["event_type"] == "reacquired" for e in recorder.tracker.recent_events)


def test_without_reacquisition_the_returning_storm_gets_a_new_identity():
    """Documents defect 2 as it behaves with reacquisition disabled."""
    recorder = _Recorder(reacquire=False)
    _storm_a_hidden_for_one_scan(recorder)
    assert recorder.tracker.get_track(1).status == "lost"
    assert len(recorder.tracker.all_tracks) == 3


def test_reacquisition_requires_a_scan_loader():
    with pytest.raises(ValueError):
        StormTracker(reacquire=True)


def test_reacquisition_window_matches_the_tracker_continuity_limit():
    assert MAX_REACQUISITION_MINUTES == MAX_TEMPORAL_CONTINUITY_MINUTES
```

Append to `tests/unit/test_history_store.py`:

```python
from tests.unit.test_tracking_reacquisition import _scan as _reacquisition_scan
from tests.unit.test_tracking_reacquisition import _storm_a, _storm_b


def test_radar_history_reacquires_through_its_ring(tmp_path):
    history = RadarHistory.open("KTLX", tmp_path)
    history.add_live_scan(_reacquisition_scan(0, [_storm_a(1), _storm_b(2)]))
    history.add_live_scan(_reacquisition_scan(5, [_storm_b(1)]))
    history.add_live_scan(_reacquisition_scan(10, [_storm_a(1, row=51, lat=35.301), _storm_b(2)]))

    tracking = history.newest().to_buffered_scan().tracking
    assert [event["event_type"] for event in tracking.recent_events] == ["reacquired"]
    assert history.tracker.get_track(1).status == "active"
```

Append to `tests/unit/test_summary.py`:

```python
from src.summary import generate_summary
from src.tracking.types import IdentityConfidence


def test_reacquired_focus_storm_is_described_as_possibly_the_same_storm():
    storm = _storm(1, -97.5, 55.0)
    track = Track(track_id=4, status="active")
    track.add_position(datetime(2026, 9, 12, 18, 0), storm)
    track.identity_diagnostics = IdentityConfidence(
        label="low", score=0.4, reason="reacquired after 1 missed scan", event_context="reacquired"
    )
    text = generate_summary("KTLX", "Oklahoma City", "2026-09-12T18:00:00Z", [storm], tracks=[track], events=[])
    assert "possibly the same storm seen before a missed scan" in text
    assert "tracking uncertain" in text
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_tracking_reacquisition.py tests/unit/test_history_store.py tests/unit/test_summary.py -v`
Expected: FAIL. The reacquisition module fails to import `MAX_REACQUISITION_MINUTES`, the store test sees no `reacquired` event, and the summary test lacks the wording.

- [ ] **Step 3: Record where each track was last seen**

In `src/tracking/types.py`, add to `Track` (before `_missed_scans`):

```python
    # (scan timestamp, object id) of the last scan this track was matched in.
    # Reacquisition rebuilds the track's mask from that scan.
    last_seen_ref: tuple[datetime, int] | None = None
```

and at the end of `Track.add_position`:

```python
        self.last_seen_ref = (timestamp, obj.object_id)
```

In `src/tracking/events.py`, add:

```python
def normalize_reacquired_event(timestamp: datetime, track_id: int, missed_scans: int) -> dict:
    scans_word = "scan" if missed_scans == 1 else "scans"
    return {
        "event_type": "reacquired",
        "timestamp": timestamp.isoformat(),
        "description": f"Track {track_id} reacquired after {missed_scans} missed {scans_word}",
        "involved_track_ids": [track_id],
    }
```

- [ ] **Step 4: Add stage-2 association**

In `src/tracking/association.py`:

1. Add imports `from collections.abc import Callable` and `GeographicMotionFieldEstimate` to the `src.tracking.motion_field` import list.

2. Add below `UNMATCHED_COST`:

```python
# Must equal src.tracker.MAX_TEMPORAL_CONTINUITY_MINUTES (asserted in tests):
# a storm unseen for longer than that is not evidence of the same storm.
MAX_REACQUISITION_MINUTES = 20.0
```

3. Add to `AssociationResult`:

```python
    # Stage 2: tracks that missed the previous scan, matched to objects stage 1
    # left unclaimed.  Kept apart from candidate_scores so stage-1 ambiguity
    # margins and identity confidence are unaffected.
    reacquired_matches: dict[int, int] = field(default_factory=dict)  # new_obj_id -> track_id
    reacquisition_scores: list[AssociationScore] = field(default_factory=list)
```

4. Add the keyword parameter to `associate_tracks`:

```python
    obj_to_track: dict[int, int],
    reacquisition_loader: Callable[[datetime], BufferedScan | None] | None = None,
) -> AssociationResult:
```

and replace its final `return result` with:

```python
    if reacquisition_loader is not None:
        _reacquire_missed_tracks(
            result=result,
            active_tracks=active_tracks,
            current_scan=current_scan,
            new_objects=new_objects,
            new_masks=new_masks,
            pixel_motion=pixel_motion,
            geo_motion=geo_motion,
            dt_hours=dt_hours,
            reacquisition_loader=reacquisition_loader,
        )
    return result
```

5. Add the function after `associate_tracks`:

```python
def _reacquire_missed_tracks(
    *,
    result: AssociationResult,
    active_tracks: list[Track],
    current_scan: BufferedScan,
    new_objects: dict,
    new_masks,
    pixel_motion,
    geo_motion: GeographicMotionFieldEstimate,
    dt_hours: float,
    reacquisition_loader: Callable[[datetime], BufferedScan | None],
) -> None:
    """Stage 2: tracks that missed the previous scan compete for unclaimed objects.

    Runs only after stage 1 is final, so a continuously tracked storm always
    keeps its match.  A missed track's mask comes from the scan where it was
    last seen, shifted by the scene motion scaled to the full elapsed time,
    and a match must pass the advected-overlap threshold and the maximum
    storm speed over that elapsed time.
    """
    claimed = set(result.primary_matches)
    for split_ids in result.split_candidates.values():
        claimed.update(split_ids)
    unclaimed = [object_id for object_id in new_objects if object_id not in claimed]
    candidates = [
        track for track in active_tracks
        if track._missed_scans >= 1 and track.last_seen_ref is not None
    ]
    if not unclaimed or not candidates or dt_hours <= 0:
        return

    seen_scans: dict[datetime, BufferedScan | None] = {}
    new_mask_cache: dict[int, np.ndarray] = {}
    cost_matrix = np.full((len(candidates), len(unclaimed)), UNMATCHED_COST, dtype=float)
    for row, track in enumerate(candidates):
        seen_at, seen_object_id = track.last_seen_ref
        elapsed_hours = (current_scan.timestamp - seen_at).total_seconds() / 3600.0
        if elapsed_hours <= 0 or elapsed_hours * 60.0 > MAX_REACQUISITION_MINUTES:
            continue
        if seen_at not in seen_scans:
            seen_scans[seen_at] = reacquisition_loader(seen_at)
        seen_scan = seen_scans[seen_at]
        prev_mask = seen_scan.object_masks.get(seen_object_id) if seen_scan is not None else None
        if prev_mask is None:
            continue
        scale = elapsed_hours / dt_hours
        scaled_motion = GeographicMotionFieldEstimate(
            delta_lat=geo_motion.delta_lat * scale,
            delta_lon=geo_motion.delta_lon * scale,
            quality=geo_motion.quality,
            source=f"reacquisition:{geo_motion.source}",
        )
        last_object = track.current_object
        predicted_lat, predicted_lon = predict_latlon_position(
            last_object.centroid_lat, last_object.centroid_lon, scaled_motion
        )
        max_distance_km = MAX_STORM_SPEED_KMH * elapsed_hours
        for col, object_id in enumerate(unclaimed):
            if object_id not in new_mask_cache:
                new_mask_cache[object_id] = new_masks[object_id]
            new_mask = new_mask_cache[object_id]
            if new_mask.shape != prev_mask.shape:
                continue
            new_object = new_objects[object_id]
            score = _candidate_score(
                track=track,
                new_object=new_object,
                prev_mask=prev_mask,
                new_mask=new_mask,
                max_distance_km=max_distance_km,
                predicted_lat=predicted_lat,
                predicted_lon=predicted_lon,
                motion_shift_rows=-pixel_motion.shift_rows * scale,
                motion_shift_cols=-pixel_motion.shift_cols * scale,
            )
            if score is None or score.advected_overlap_score < MIN_ADVECTED_OVERLAP_PCT:
                continue
            centroid_km = haversine_distance_km(
                last_object.centroid_lat, last_object.centroid_lon,
                new_object.centroid_lat, new_object.centroid_lon,
            )
            if centroid_km > max(max_distance_km, 5.0):
                continue
            result.reacquisition_scores.append(score)
            cost_matrix[row, col] = score.total_cost

    for row, col in zip(*linear_sum_assignment(cost_matrix)):
        if cost_matrix[row, col] >= UNMATCHED_COST:
            continue
        object_id, track_id = unclaimed[col], candidates[row].track_id
        result.reacquired_matches[object_id] = track_id
        result.unmatched_new_ids.discard(object_id)
        result.unmatched_track_ids.discard(track_id)
```

- [ ] **Step 5: Apply reacquired matches in the tracker**

In `src/tracker.py`:

1. Import `normalize_reacquired_event` alongside the other event helpers, and add below `MEDIUM_CONFIDENCE`:

```python
# A reacquired identity is never better than low confidence: the storm was
# not observed for at least one scan, so speech must not state it as fact.
REACQUIRED_IDENTITY_CAP = 0.4
```

2. Change `__init__`'s signature and add the flag:

```python
    def __init__(self, scan_loader: ScanLoader | None = None, *, reacquire: bool = False):
        if reacquire and scan_loader is None:
            raise ValueError("reacquisition needs a scan_loader for the scans where tracks were last seen")
```

with `self._reacquire = reacquire` stored next to `self._scan_loader`. In `from_state`, add the keyword `*, reacquire: bool = False` and construct with `cls(scan_loader, reacquire=reacquire)`.

3. In `update`, pass the loader into `associate_tracks(...)` by adding the argument:

```python
            reacquisition_loader=(
                (lambda seen_at: self._scan_loader(scan.site_id, seen_at))
                if self._reacquire
                else None
            ),
```

4. In `update`, directly after the `for new_id, track_id in association.primary_matches.items():` loop and before `# Create new tracks for unmatched new objects`, insert:

```python
        for new_id, track_id in association.reacquired_matches.items():
            if new_id in new_obj_to_track:
                continue
            track = self.get_track(track_id)
            if track is None or track.status != "active":
                continue
            missed_scans = track._missed_scans
            track.add_position(timestamp, new_objects[new_id])
            track.rotation_history.append(RotationHistoryEntry(
                timestamp=timestamp,
                rotation=getattr(new_objects[new_id], "rotation", None),
            ))
            if len(track.rotation_history) > 6:
                track.rotation_history = track.rotation_history[-6:]
            score = next(
                candidate for candidate in association.reacquisition_scores
                if candidate.track_id == track_id and candidate.object_id == new_id
            )
            match_quality = self._match_quality(score)
            track.identity_confidence = round(
                min(REACQUIRED_IDENTITY_CAP, match_quality * _scan_quality_factor(scan) / (1 + missed_scans)),
                2,
            )
            scans_word = "scan" if missed_scans == 1 else "scans"
            track.identity_diagnostics = self._build_identity_diagnostics(
                score_value=track.identity_confidence,
                scan=scan,
                track=track,
                reason=f"reacquired after {missed_scans} missed {scans_word}",
                match_quality=match_quality,
                event_context="reacquired",
            )
            track.identity_diagnostics.missed_scans = missed_scans
            self._recent_events.append(normalize_reacquired_event(timestamp, track_id, missed_scans))
            new_obj_to_track[new_id] = track_id
```

In `src/history/store.py`, enable reacquisition in the live history:

```python
    def _new_tracker(self) -> StormTracker:
        return StormTracker(scan_loader=self._load_for_tracker, reacquire=True)

    def _restore_tracker(self, state: dict) -> StormTracker:
        return StormTracker.from_state(state, self._load_for_tracker, reacquire=True)
```

- [ ] **Step 6: Say it in speech**

In `src/summary.py`, add:

```python
def _was_reacquired(track) -> bool:
    diagnostics = getattr(track, "identity_diagnostics", None)
    return diagnostics is not None and diagnostics.event_context == "reacquired"
```

and in `generate_summary`, directly after `motion_str = _format_motion(motion, track=focus_track, events=events)`:

```python
    if focus_track is not None and _was_reacquired(focus_track):
        # Identity rests on shape overlap across a scan in which the storm was
        # not seen; say so rather than state continuity or motion as fact.
        motion_str = ", possibly the same storm seen before a missed scan, tracking uncertain"
```

- [ ] **Step 7: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_tracking_reacquisition.py tests/unit/test_history_store.py tests/unit/test_summary.py tests/unit/test_tracking_association.py tests/unit/test_tracker.py tests/unit/test_tracker_persistence.py -v`
Expected: all PASS.

- [ ] **Step 8: Run the suite and commit**

Run: `.venv/Scripts/python.exe -m pytest tests/unit tests/smoke -q`
Expected: all pass.

```bash
git add src/tracking/types.py src/tracking/events.py src/tracking/association.py src/tracker.py src/history/store.py src/summary.py tests/unit/test_tracking_reacquisition.py tests/unit/test_history_store.py tests/unit/test_summary.py
git commit -m "$(cat <<'EOF'
Reacquire a storm missing from one scan instead of renaming it

A track that missed the previous scan could never be matched again.
After continuous tracks are matched, missed tracks now compete for the
remaining objects using their mask from the scan where they were last
seen, shifted by scene motion over the elapsed time. Matches need the
advected-overlap threshold, the speed limit and the 20-minute
continuity limit, carry low identity confidence and a reacquired event,
and speech calls the storm possibly the same one.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_011VMHfi1nbuux7187WtgQTt
EOF
)"
```

---

### Task 9: Storm growth and decay trends

**Files:**
- Modify: `src/tracking/types.py` (`CORE_MIN_DBZ`, `TrendSample`, `Track.trend_samples`, `Track.add_position`)
- Create: `src/tracking/trends.py`
- Modify: `src/tracker.py` (mark reacquired samples)
- Modify: `src/history/records.py` (register `TrendSample`)
- Modify: `src/models.py` (`TrackTrend`, `StormTrack.trend`)
- Modify: `src/server.py` (`_track_to_model`)
- Test: `tests/unit/test_tracking_trends.py` (create)

**Interfaces:**
- Consumes: Task 8 reacquired handling.
- Produces:
  - `CORE_MIN_DBZ = 50.0`, `MAX_TREND_SAMPLES = 6`
  - `TrendSample(timestamp, area_km2, core_area_km2, peak_dbz, reacquired=False)`, `Track.trend_samples: list[TrendSample]`
  - `StormTrend(area: str, core_area: str, confidence: str, reason: str, sample_count: int)`
  - `compute_trend(samples: list[TrendSample]) -> StormTrend`
  - `StormTrack.trend: TrackTrend | None`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_tracking_trends.py`:

```python
from datetime import datetime, timedelta

from fastapi.testclient import TestClient

import src.server as server
from src.detection import DetectedObject, IntensityLayerData
from src.history.records import dumps, loads
from src.tracking.trends import compute_trend
from src.tracking.types import MAX_TREND_SAMPLES, TrendSample, Track
from tests.unit.test_tracking_reacquisition import _Recorder, _storm_a_hidden_for_one_scan

T0 = datetime(2026, 9, 12, 18, 0)


def _samples(areas, cores=None, reacquired_index=None):
    cores = cores or [0.0] * len(areas)
    return [
        TrendSample(T0 + timedelta(minutes=5 * i), area, core, 50.0, reacquired=(i == reacquired_index))
        for i, (area, core) in enumerate(zip(areas, cores))
    ]


def test_consistent_growth_is_medium_confidence():
    trend = compute_trend(_samples([100.0, 120.0, 150.0], cores=[0.0, 4.0, 10.0]))
    assert (trend.area, trend.core_area, trend.confidence) == ("growing", "growing", "medium")
    assert trend.sample_count == 3


def test_decay_and_steady_are_classified():
    assert compute_trend(_samples([150.0, 130.0, 100.0])).area == "decaying"
    assert compute_trend(_samples([100.0, 104.0, 98.0])).area == "steady"


def test_direction_change_within_window_lowers_confidence():
    trend = compute_trend(_samples([100.0, 80.0, 150.0]))
    assert trend.area == "growing" and trend.confidence == "low"


def test_no_core_is_reported_as_none():
    assert compute_trend(_samples([100.0, 120.0, 150.0])).core_area == "none"


def test_fewer_than_three_samples_is_insufficient():
    trend = compute_trend(_samples([100.0, 150.0]))
    assert (trend.area, trend.confidence) == ("insufficient", "none")


def test_window_with_a_reacquired_sample_is_insufficient():
    trend = compute_trend(_samples([100.0, 120.0, 150.0], reacquired_index=2))
    assert trend.area == "insufficient"
    assert "reacquired" in trend.reason


def test_add_position_records_core_area_and_caps_samples():
    track = Track(track_id=1, status="active")
    obj = DetectedObject(
        1, 35.0, -97.0, 40.0, 90.0, 62.0, "severe core", 100.0,
        layers=[
            IntensityLayerData("heavy precipitation", 40.0, 50.0, 60.0),
            IntensityLayerData("intense precipitation", 50.0, 60.0, 12.0),
            IntensityLayerData("severe core", 60.0, float("inf"), 3.0),
        ],
    )
    for i in range(MAX_TREND_SAMPLES + 2):
        track.add_position(T0 + timedelta(minutes=5 * i), obj)
    assert len(track.trend_samples) == MAX_TREND_SAMPLES
    assert track.trend_samples[-1].core_area_km2 == 15.0
    assert loads(dumps(track)) == track


def test_reacquired_match_marks_its_trend_sample():
    recorder = _Recorder()
    _storm_a_hidden_for_one_scan(recorder)
    samples = recorder.tracker.get_track(1).trend_samples
    assert [sample.reacquired for sample in samples] == [False, True]


def test_tracks_endpoint_reports_trend(monkeypatch):
    recorder = _Recorder()
    _storm_a_hidden_for_one_scan(recorder)
    scan = recorder.stored[max(recorder.stored)]
    scan.tracking = recorder.tracker.snapshot()
    monkeypatch.setattr(server, "_live_or_ingest", lambda site_id, dt=None: (scan, False))
    data = TestClient(server.app).get("/tracks/KTLX").json()
    trends = {track["track_id"]: track["trend"] for track in data["tracks"]}
    assert trends[1]["area"] == "insufficient"
    assert set(trends[1]) == {"area", "core_area", "confidence", "reason", "sample_count"}
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_tracking_trends.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.tracking.trends'`.

- [ ] **Step 3: Implement samples and trends**

In `src/tracking/types.py`, add below the imports:

```python
# Reflectivity at or above this marks a storm's intense core (the
# "intense precipitation" and "severe core" intensity bands).
CORE_MIN_DBZ = 50.0
MAX_TREND_SAMPLES = 6


@dataclass
class TrendSample:
    timestamp: datetime
    area_km2: float
    core_area_km2: float
    peak_dbz: float
    # True when this sample came from a reacquired match; trends spanning it
    # are not reported, since the storm was unobserved for a scan.
    reacquired: bool = False
```

Add to `Track` (next to `rotation_history`):

```python
    trend_samples: list[TrendSample] = field(default_factory=list)
```

and at the end of `Track.add_position`:

```python
        self.trend_samples.append(TrendSample(
            timestamp=timestamp,
            area_km2=obj.area_km2,
            core_area_km2=sum(layer.area_km2 for layer in obj.layers if layer.min_dbz >= CORE_MIN_DBZ),
            peak_dbz=obj.peak_dbz,
        ))
        if len(self.trend_samples) > MAX_TREND_SAMPLES:
            self.trend_samples = self.trend_samples[-MAX_TREND_SAMPLES:]
```

Create `src/tracking/trends.py`:

```python
"""Storm growth and decay from a track's recent per-scan samples.

Area comes from one low-elevation reflectivity sweep, so a trend is at best
medium confidence.  A window that includes a reacquired match is not
reported: the storm was unobserved for a scan inside it.
"""

from dataclasses import dataclass

from src.tracking.types import TrendSample

MIN_TREND_SAMPLES = 3
# Relative change from the first to the last sample that counts as growth or decay.
TREND_CHANGE_FRACTION = 0.15


@dataclass
class StormTrend:
    area: str  # "growing", "steady", "decaying" or "insufficient"
    core_area: str  # the same labels, or "none" when no >= 50 dBZ core was present
    confidence: str  # "medium", "low" or "none"
    reason: str
    sample_count: int


def _classify(first: float, last: float) -> str:
    change = (last - first) / max(first, 1.0)
    if change > TREND_CHANGE_FRACTION:
        return "growing"
    if change < -TREND_CHANGE_FRACTION:
        return "decaying"
    return "steady"


def _consistent(values: list[float], label: str) -> bool:
    steps = [later - earlier for earlier, later in zip(values, values[1:])]
    if label == "growing":
        return all(step >= 0 for step in steps)
    if label == "decaying":
        return all(step <= 0 for step in steps)
    return True


def compute_trend(samples: list[TrendSample]) -> StormTrend:
    window = list(samples)
    count = len(window)
    if count < MIN_TREND_SAMPLES:
        return StormTrend("insufficient", "insufficient", "none", f"needs {MIN_TREND_SAMPLES} scans, has {count}", count)
    if any(sample.reacquired for sample in window):
        return StormTrend("insufficient", "insufficient", "none", "window includes a reacquired match", count)
    areas = [sample.area_km2 for sample in window]
    cores = [sample.core_area_km2 for sample in window]
    area = _classify(areas[0], areas[-1])
    core = "none" if max(cores) == 0.0 else _classify(cores[0], cores[-1])
    consistent = _consistent(areas, area) and (core == "none" or _consistent(cores, core))
    return StormTrend(
        area=area,
        core_area=core,
        confidence="medium" if consistent else "low",
        reason="consistent across the window" if consistent else "direction changes within the window",
        sample_count=count,
    )
```

In `src/tracker.py`'s reacquired-match block (Task 8), directly after `track.add_position(timestamp, new_objects[new_id])`, add:

```python
            track.trend_samples[-1].reacquired = True
```

In `src/history/records.py`, import `TrendSample` from `src.tracking.types` and add it to the registration tuple.

In `src/models.py`, add:

```python
class TrackTrend(BaseModel):
    area: str
    core_area: str
    confidence: str
    reason: str
    sample_count: int
```

and to `StormTrack`: `trend: TrackTrend | None = None`.

In `src/server.py`, import `compute_trend` from `src.tracking.trends` and `TrackTrend` from `src.models`. In `_track_to_model`, add the argument:

```python
        trend=TrackTrend(**asdict(compute_trend(getattr(track, "trend_samples", [])))),
```

(with `from dataclasses import asdict` added to the imports).

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_tracking_trends.py tests/unit/test_history_records.py tests/unit/test_tracker_persistence.py -v`
Expected: all PASS.

- [ ] **Step 5: Run the suite and commit**

Run: `.venv/Scripts/python.exe -m pytest tests/unit tests/smoke -q`
Expected: all pass.

```bash
git add src/tracking/types.py src/tracking/trends.py src/tracker.py src/history/records.py src/models.py src/server.py tests/unit/test_tracking_trends.py
git commit -m "$(cat <<'EOF'
Report storm area and core growth trends per track

Each track keeps its last six per-scan samples of total area, >= 50 dBZ
core area and peak. /tracks reports growing, steady or decaying with at
most medium confidence, and insufficient with fewer than three samples
or when a reacquired match falls inside the window.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_011VMHfi1nbuux7187WtgQTt
EOF
)"
```

---

### Task 10: History API, ready status for retained scans, rebuilt-continuity reporting

**Files:**
- Modify: `src/models.py` (`HistoryScan`, `MapHistoryResponse`, `continuity_rebuilt` on `TracksResponse` and `SummaryResponse`)
- Modify: `src/server.py` (new `get_map_history`, `get_map_status` datetime path, `get_live_scan_status`, `get_tracks`, `get_summary`)
- Modify: `src/summary.py` (`generate_summary(..., continuity_rebuilt=False)`)
- Test: `tests/unit/test_server_history_api.py` (create), `tests/smoke/test_server_smoke.py`

**Interfaces:**
- Consumes: Tasks 6–9.
- Produces:
  - `GET /map/history`: city/state, zipcode, or latitude/longitude; returns `{site_id, location, scans: [{timestamp, object_count, tracked}]}`, newest first
  - `/map/status?datetime=<retained timestamp>` is ready without ingest
  - `/live/{site}/status` adds `history_write_error`
  - `/tracks` and `/summary` add `continuity_rebuilt: bool`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_server_history_api.py`:

```python
from datetime import datetime, timedelta

from fastapi.testclient import TestClient
import pytest

import src.history.store as store
import src.server as server
from src.history.store import TRACKER_STATE_FILENAME, HistoryRegistry, RadarHistory
from tests.unit.test_tracker import _make_object, _make_scan

T0 = datetime(2026, 9, 12, 18, 0)
LOCATION = {"latitude": "35.33", "longitude": "-97.28"}


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(server, "rank_sites", lambda lat, lon: [{"site_id": "KTLX"}])
    monkeypatch.setattr(server, "_schedule_live_refresh", lambda *a, **k: False)
    monkeypatch.setattr(server, "fetch_scan", lambda *a, **k: pytest.fail("must not fetch"))
    return TestClient(server.app)


def _retain(minutes):
    with server._histories.use("KTLX") as history:
        for minute in minutes:
            history.add_live_scan(_make_scan(
                "KTLX", T0 + timedelta(minutes=minute),
                [_make_object(1, 35.3, -97.3), _make_object(2, 35.0, -97.0)],
            ))


def test_history_lists_retained_scans_newest_first(client):
    _retain([0, 5, 10])
    data = client.get("/map/history", params=LOCATION).json()
    assert data["site_id"] == "KTLX"
    assert [scan["timestamp"] for scan in data["scans"]] == [
        (T0 + timedelta(minutes=m)).isoformat() for m in (10, 5, 0)
    ]
    assert all(scan["tracked"] and scan["object_count"] == 2 for scan in data["scans"])


def test_history_for_a_radar_with_nothing_retained_is_empty(client):
    assert client.get("/map/history", params=LOCATION).json()["scans"] == []


def test_listed_timestamp_round_trips_to_tracks_and_status(client):
    _retain([0, 5])
    oldest = client.get("/map/history", params=LOCATION).json()["scans"][-1]["timestamp"]

    status = client.get("/map/status", params={**LOCATION, "datetime": oldest}).json()
    tracks = client.get("/tracks/KTLX", params={"datetime": oldest}).json()

    assert status["available"] is True and status["refresh_state"] == "ready"
    assert status["scan_timestamp"] == oldest
    assert tracks["tracking_context"] == "tracked"
    assert tracks["timestamp"] == oldest
    assert [len(track["positions"]) for track in tracks["tracks"]] == [1, 1]


def test_rebuilt_continuity_is_reported_and_spoken(client, monkeypatch, tmp_path):
    directory = tmp_path / "KTLX" / "history"
    history = RadarHistory.open("KTLX", directory)
    for minute in (0, 5):
        history.add_live_scan(_make_scan("KTLX", T0 + timedelta(minutes=minute), [_make_object(1, 35.3, -97.3)]))
    (directory / TRACKER_STATE_FILENAME).unlink()
    monkeypatch.setattr(server, "_histories", HistoryRegistry(tmp_path))

    tracks = client.get("/tracks/KTLX").json()
    summary = client.get("/summary/KTLX").json()

    assert tracks["continuity_rebuilt"] is True
    assert summary["continuity_rebuilt"] is True
    assert "storm tracking was rebuilt after a restart" in summary["text"]


def test_live_status_reports_history_write_errors(client, monkeypatch, tmp_path):
    monkeypatch.setattr(server, "_histories", HistoryRegistry(tmp_path))

    def failing_save(scan, directory):
        raise OSError("disk full")

    monkeypatch.setattr(store, "save_compact_scan", failing_save)
    _retain([0])
    status = client.get("/live/KTLX/status").json()
    assert status["available"] is True
    assert status["history_write_error"] == "OSError: disk full"
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_server_history_api.py -v`
Expected: FAIL with 404 for `/map/history` and `KeyError` for the new fields.

- [ ] **Step 3: Implement the API changes**

In `src/models.py`:

```python
class HistoryScan(BaseModel):
    # Pass back unchanged as `datetime=` to select exactly this scan.
    timestamp: str
    object_count: int
    tracked: bool


class MapHistoryResponse(BaseModel):
    site_id: str
    location: dict
    scans: list[HistoryScan]  # newest first
```

and add `continuity_rebuilt: bool = False` to both `TracksResponse` and `SummaryResponse`.

In `src/summary.py`, add the keyword parameter `continuity_rebuilt: bool = False` to `generate_summary` (documented in its docstring), and directly before `parts.append(f" Covering approximately {area_mi2} square miles.")` add:

```python
    if continuity_rebuilt:
        parts.append(" Note: storm tracking was rebuilt after a restart, so storm identities may have changed.")
```

In `src/server.py`:

1. Import `HistoryScan` and `MapHistoryResponse` from `src.models`.

2. Add the endpoint after `get_map_status`:

```python
@app.get("/map/history", response_model=MapHistoryResponse)
def get_map_history(
    city: str | None = Query(None),
    state: str | None = Query(None),
    zipcode: str | None = Query(None),
    latitude: float | None = Query(None),
    longitude: float | None = Query(None),
):
    """Retained scans for the radar ARW selects for a location, newest first.

    Never ingests: it lists only what that radar's history already holds.
    Each timestamp can be passed back unchanged as `datetime=` to the map,
    status, objects, tracks and summary endpoints to select exactly that scan.
    """
    lat, lon, label = _resolve_map_location(city, state, zipcode, latitude, longitude)
    ranked_sites = rank_sites(lat, lon)
    if not ranked_sites:
        raise HTTPException(status_code=404, detail="No radar site found for that location")
    site_id = ranked_sites[0]["site_id"].upper()
    with _histories.use(site_id) as history:
        retained = list(reversed(history.scans()))
    return MapHistoryResponse(
        site_id=site_id,
        location={"latitude": lat, "longitude": lon, "label": label},
        scans=[
            HistoryScan(timestamp=scan.scan_timestamp_text, object_count=scan.object_count, tracked=scan.tracked)
            for scan in retained
        ],
    )
```

3. In `get_map_status`, replace the historical branch's `completed = _ingest_to_buffer(site_id, requested_datetime)` and its `"scan_timestamp": completed.reflectivity_data.timestamp,` with:

```python
        with _histories.use(site_id) as history:
            retained = history.find_by_timestamp(requested_datetime)
        scan_timestamp = (
            retained.scan_timestamp_text
            if retained is not None
            else _ingest_to_buffer(site_id, requested_datetime).reflectivity_data.timestamp
        )
```

and `"scan_timestamp": scan_timestamp,`.

4. In `get_live_scan_status`, replace `completed = _newest_live_scan(normalized_site)` with:

```python
    with _histories.use(normalized_site) as history:
        completed = history.newest()
        history_write_error = history.last_write_error
```

and add `"history_write_error": history_write_error,` to the returned dict. Its comment: a failed disk write leaves live tracking running, but the history will not survive a restart until writes succeed.

5. In `get_tracks`, add `continuity_rebuilt=bool(tracking is not None and tracking.continuity_rebuilt),` to `TracksResponse(...)`. In `get_summary`, pass `continuity_rebuilt=bool(tracking is not None and tracking.continuity_rebuilt)` to `generate_summary` and add the same value to `SummaryResponse(...)`.

Add to `tests/smoke/test_server_smoke.py`:

```python
def test_map_history_requires_location():
    assert client.get("/map/history").status_code == 422


def test_map_history_returns_200_for_a_location(monkeypatch):
    monkeypatch.setattr("src.server.rank_sites", lambda _lat, _lon: [{"site_id": "KTLX"}])
    resp = client.get("/map/history", params={"latitude": 35.33, "longitude": -97.28})
    assert resp.status_code == 200
    assert resp.json()["site_id"] == "KTLX"
    assert isinstance(resp.json()["scans"], list)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_server_history_api.py tests/smoke/test_server_smoke.py -v`
Expected: all PASS.

- [ ] **Step 5: Run the suite and commit**

Run: `.venv/Scripts/python.exe -m pytest tests/unit tests/smoke -q`
Expected: all pass.

```bash
git add src/models.py src/server.py src/summary.py tests/unit/test_server_history_api.py tests/smoke/test_server_smoke.py
git commit -m "$(cat <<'EOF'
Expose retained radar scans and rebuilt continuity through the API

GET /map/history lists a location's radar scans newest first with
timestamps that select exactly that scan elsewhere. /map/status is
ready immediately for a retained scan. /tracks and /summary report
continuity_rebuilt and the summary says so; /live status reports
history write errors.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_011VMHfi1nbuux7187WtgQTt
EOF
)"
```

---

### Task 11: Real-data proofs, benchmark comparison, live check, report and docs

**Files:**
- Create: `tests/e2e/test_proof_compact_history.py`
- Create: `docs/test_reports/2026-09-12-compact-scan-history.md`
- Create: `docs/test_reports/2026-09-12-compact-history-proof.json` (written by the proof)
- Create: `docs/test_reports/2026-09-12-benchmark-after-history.json`
- Modify: `README.md` (history environment variables), `PROGRESS.md`, `.gitignore`

**Interfaces:**
- Consumes: everything above.

- [ ] **Step 1: Write the proofs**

Create `tests/e2e/test_proof_compact_history.py`:

```python
"""Proofs (compact scan history, Task 11) on real cached Level II volumes.

1. Identical tracking: with reacquisition disabled, a RadarHistory -- whose
   tracker reloads previous scans from compact records -- produces the same
   per-scan tracking snapshot, byte for byte, as a tracker that retains full
   scans (the previous pipeline).  With reacquisition enabled, snapshots are
   identical until the first reacquisition; every reacquisition is recorded
   for review.
2. Memory: after live-style ingest of five scans on two radars with all map
   layers rendered, no full-size grid is retained, and retained history and
   rendered-layer bytes are within limits derived from measured sizes.
3. Restart: reopening a radar's history mid-window continues identically.
4. Browsing: a historical request leaves live tracking unchanged.

Results are written to docs/test_reports/2026-09-12-compact-history-proof.json.
"""

from copy import deepcopy
from dataclasses import replace
from datetime import datetime
import gc
import json
from pathlib import Path
import tracemalloc

import numpy as np
import pytest

import src.server as server
from src.history.layer_cache import RenderedLayerCache
from src.history.records import dumps
from src.history.store import HistoryRegistry, RadarHistory
from src.tracker import StormTracker

KTLX_WINDOW = [
    f"cache/KTLX/KTLX20260410_{stamp}_V06"
    for stamp in ("224209", "224737", "225255", "225805", "230332", "230850", "231407")
]
KSOX_WINDOW = [
    f"cache/KSOX/KSOX20260410_{stamp}_V06"
    for stamp in ("223503", "224345", "225226", "230106", "230940")
]
KTLX_HOUR_EARLIER = "cache/KTLX/KTLX20260410_214929_V06"
REPORT = Path("docs/test_reports/2026-09-12-compact-history-proof.json")


def _analyze(path: str):
    return server._process_scan_file(Path(path).parent.name, str(Path(path).resolve()))


def _twin(buffered):
    """Same arrays (read-only), independent object records."""
    return replace(buffered, detected_objects=deepcopy(buffered.detected_objects), tracking=None)


def _record(section: str, payload) -> None:
    report = json.loads(REPORT.read_text()) if REPORT.exists() else {}
    report[section] = payload
    REPORT.write_text(json.dumps(report, indent=2, default=str))


class _NoReacquireHistory(RadarHistory):
    def _new_tracker(self):
        return StormTracker(scan_loader=self._load_for_tracker)

    def _restore_tracker(self, state):
        return StormTracker.from_state(state, self._load_for_tracker)


def test_history_tracking_is_identical_to_retained_scan_tracking(tmp_path):
    previous = StormTracker()
    plain = _NoReacquireHistory.open("KTLX", tmp_path / "plain")
    reacquiring = RadarHistory.open("KTLX", tmp_path / "reacquiring")
    rows, first_divergence = [], None

    for index, path in enumerate(KTLX_WINDOW):
        original = _analyze(path)
        old_input, reacquire_input = _twin(original), _twin(original)
        previous.update(old_input)
        for obj in old_input.detected_objects:
            obj.temporal_status = previous.temporal_status_for_current_object(obj.object_id)
        old_snapshot = dumps(previous.snapshot())

        plain.add_live_scan(original)
        plain_snapshot = dumps(plain.newest().to_buffered_scan().tracking)
        reacquiring.add_live_scan(reacquire_input)
        reacquired_tracking = reacquiring.newest().to_buffered_scan().tracking
        reacquired_events = [e for e in reacquired_tracking.recent_events if e["event_type"] == "reacquired"]

        assert plain_snapshot == old_snapshot, f"scan {index} ({path}) tracking differs"
        if first_divergence is None and dumps(reacquired_tracking) != old_snapshot:
            first_divergence = index
        rows.append({
            "scan": Path(path).name,
            "objects": len(original.detected_objects),
            "active_tracks": len(reacquired_tracking.active_tracks),
            "reacquired": reacquired_events,
        })
        del original, old_input, reacquire_input

    assert dumps(plain.tracker.export_state()) == dumps(previous.export_state())
    if first_divergence is not None:
        assert rows[first_divergence]["reacquired"], "reacquisition run diverged without a reacquisition"
    _record("identical_tracking", {"scans": rows, "first_reacquisition_divergence": first_divergence})


def test_retained_memory_is_bounded_and_no_full_grids_remain(tmp_path, monkeypatch):
    registry = HistoryRegistry(tmp_path / "history")
    monkeypatch.setattr(server, "_histories", registry)
    monkeypatch.setattr(server, "_map_layers", RenderedLayerCache(max_unpinned_scans=10))
    monkeypatch.setattr(server, "_schedule_live_refresh", lambda *a, **k: False)
    grid_gates = None
    old_pipeline_bytes = 0
    compact_sizes, rendered_sizes = [], []
    tracemalloc.start()

    for site_id, window in (("KTLX", KTLX_WINDOW[:5]), ("KSOX", KSOX_WINDOW)):
        paths = iter(str(Path(p).resolve()) for p in window)
        monkeypatch.setattr(server, "fetch_scan", lambda site, dt=None, paths=paths: next(paths))
        for _ in window:
            buffered = server._ingest_to_buffer(site_id)
            gates = buffered.reflectivity_data.reflectivity.size
            grid_gates = gates if grid_gates is None else min(grid_gates, gates)
            # What the previous pipeline retained for this one scan.
            old_pipeline_bytes = max(
                old_pipeline_bytes,
                buffered.reflectivity_data.reflectivity.nbytes
                + np.asarray(buffered.labeled_grid).nbytes
                + sum(mask.nbytes for mask in buffered.object_masks.values()),
            )
            layers = server._prepared_map_layers(buffered)
            precipitation = server._prepared_precipitation_layer(buffered)
            rendered_sizes.append(len(json.dumps(layers)) + len(json.dumps(precipitation)))
            del buffered, layers, precipitation
        with registry.use(site_id) as history:
            compact_sizes.extend(scan.nbytes for scan in history.scans())

    transient_peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    gc.collect()
    retained_grids = [
        (obj.shape, obj.dtype.str)
        for obj in gc.get_objects()
        if isinstance(obj, np.ndarray) and obj.size >= grid_gates
    ]
    retained_compact = sum(compact_sizes)
    retained_rendered = sum(
        len(json.dumps(server._map_layers.get(key)))
        for key in list(server._map_layers._entries)
    )
    compact_limit = 10 * max(compact_sizes) * 1.1
    rendered_limit = (2 + 10) * max(rendered_sizes) * 1.1

    _record("memory", {
        "compact_scan_bytes": compact_sizes,
        "retained_compact_bytes": retained_compact,
        "compact_limit_bytes": compact_limit,
        "rendered_scan_bytes": rendered_sizes,
        "retained_rendered_bytes": retained_rendered,
        "rendered_limit_bytes": rendered_limit,
        "old_pipeline_largest_scan_array_bytes": old_pipeline_bytes,
        "old_pipeline_two_scans_two_radars_bytes": 4 * old_pipeline_bytes,
        "transient_peak_traced_bytes": transient_peak,
        "retained_full_grids": retained_grids,
    })
    assert len(compact_sizes) == 10
    assert retained_grids == [], f"full-size grids still retained: {retained_grids}"
    assert retained_compact <= compact_limit
    assert retained_rendered <= rendered_limit


def test_restart_mid_window_continues_identically(tmp_path):
    uninterrupted = RadarHistory.open("KTLX", tmp_path / "uninterrupted")
    restarted = RadarHistory.open("KTLX", tmp_path / "restarted")
    for index, path in enumerate(KTLX_WINDOW[:5]):
        original = _analyze(path)
        twin = _twin(original)
        uninterrupted.add_live_scan(original)
        if index == 3:
            restarted = RadarHistory.open("KTLX", tmp_path / "restarted")
        restarted.add_live_scan(twin)
        del original, twin

    assert dumps(restarted.tracker.export_state()) == dumps(uninterrupted.tracker.export_state())
    assert not restarted.newest().to_buffered_scan().tracking.continuity_rebuilt
    _record("restart", {"restart_before_scan": 4, "identical": True})


def test_historical_request_leaves_live_tracking_unchanged(monkeypatch):
    monkeypatch.setattr(server, "_schedule_live_refresh", lambda *a, **k: False)
    live = iter(str(Path(p).resolve()) for p in KTLX_WINDOW[:3])
    monkeypatch.setattr(server, "fetch_scan", lambda site, dt=None: next(live))
    for _ in range(3):
        server._ingest_to_buffer("KTLX")
    with server._histories.use("KTLX") as history:
        before_state = dumps(history.tracker.export_state())
        before_ring = [scan.timestamp for scan in history.scans()]

    monkeypatch.setattr(server, "fetch_scan", lambda site, dt=None: str(Path(KTLX_HOUR_EARLIER).resolve()))
    earlier = server._ingest_to_buffer("KTLX", datetime(2026, 4, 10, 21, 49, 29))

    assert earlier.tracking is None
    with server._histories.use("KTLX") as history:
        assert dumps(history.tracker.export_state()) == before_state
        assert [scan.timestamp for scan in history.scans()] == before_ring
    _record("browsing", {"historical_volume": Path(KTLX_HOUR_EARLIER).name, "live_tracking_unchanged": True})
```

- [ ] **Step 2: Run the proofs**

Run: `.venv/Scripts/python.exe -m pytest tests/e2e/test_proof_compact_history.py -v -s`
Expected: 4 PASS (several minutes). If `retained_grids` is not empty, find what holds each array with `gc.get_referrers`. Fix the retention rather than relaxing the assertion. If the plain and old snapshots differ, stop: that is a lossless-storage or loader defect, and it must be found before anything else.

- [ ] **Step 3: Compare the tracking benchmark with the Task 5 baseline**

Run:

```bash
.venv/Scripts/python.exe scripts/evaluate_tracking.py --manifest docs/benchmarks/tracking_benchmark_manifest_local.json --output-json docs/test_reports/2026-09-12-benchmark-after-history.json
.venv/Scripts/python.exe -c "import json; a=json.load(open('docs/test_reports/2026-09-12-benchmark-before-history.json')); b=json.load(open('docs/test_reports/2026-09-12-benchmark-after-history.json')); print('IDENTICAL' if a == b else 'DIFFERENT'); [print(x['benchmark_id'], k, x[k], y[k]) for x, y in zip(a, b) for k in x if x[k] != y[k]]"
```

Expected: `IDENTICAL`. `scripts/evaluate_tracking.py` uses `StormTracker()` without a loader or reacquisition, so Tasks 5, 8 and 9 must not change its results. Any difference is a regression to investigate. Don't commit it as expected output.

- [ ] **Step 4: Live check against real NEXRAD**

Start the server: `.venv/Scripts/python.exe -m uvicorn src.server:app --port 8000`. For a radar with current precipitation, poll `/live/<SITE>/status` until available. Then, over at least three live scans (about 15 minutes, or trigger refreshes with `/map/status?refresh=true`):

```bash
curl -s "http://localhost:8000/map/history?latitude=<lat>&longitude=<lon>"
curl -s "http://localhost:8000/summary/<SITE>"
curl -s "http://localhost:8000/tracks/<SITE>?datetime=<oldest timestamp from /map/history>"
curl -s "http://localhost:8000/map/precipitation.geojson?latitude=<lat>&longitude=<lon>&datetime=<oldest timestamp>"
```

Expected:
- `/map/history` lists up to 5 scans, newest first, all `tracked: true`.
- The older scan's tracks show fewer positions than the newest.
- The precipitation layer for the older scan has non-null `median_rhohv`.
- `cache/<SITE>/history/` holds one `.arwscan` per listed scan plus `tracker_state.json`.

Then stop and restart the server and repeat `/map/history` and `/tracks/<SITE>`. Expected: the same scans and track IDs, `continuity_rebuilt: false`. Record the site, times and observations for the report.

- [ ] **Step 5: Write the report and docs**

Create `docs/test_reports/2026-09-12-compact-scan-history.md` with these sections, filled from the proof JSON, the Task 3 proof output, the benchmark comparison and the live check:

1. Compact scan sizes per volume against the full scan's array bytes.
2. Identical tracking result and the reacquisition list.
3. Memory: retained compact and rendered bytes against their limits, old-pipeline comparison, transient peak, and retained full grids (must be none).
4. Restart and browsing results.
5. Benchmark comparison.
6. Live check.
7. Known limits:
   - reacquisition covers exactly one missed scan (spec A5)
   - transient memory during an update is not bounded (A6)
   - `Track.positions` is uncapped (issue #1)
   - lost and merged tracks accumulate in tracker state for the life of a site's tracker, so `tracker_state.json` grows during a long session

In `README.md`, document `ARW_HISTORY_ROOT`, `ARW_SCANS_PER_RADAR`, `ARW_MAX_RADARS_IN_MEMORY`, `ARW_MAX_HISTORICAL_SCANS`, `ARW_MAX_RENDERED_SCANS` and `GET /map/history`. Remove any documentation of the deleted `ARW_PROCESSED_SCANS_PER_SITE`, `ARW_MAX_PROCESSED_SCANS` and `ARW_REPLAY_SCANS_PER_SITE`.

Add `cache/*/history/` to `.gitignore` under a `# Radar history (regenerable runtime state)` comment.

Update `PROGRESS.md`: completed work with commits, proof and benchmark results, known limits, and that Weather Kitten scan stepping (`docs/superpowers/plans/2026-09-12-weather-kitten-scan-stepping.md`) is next.

- [ ] **Step 6: Run everything and commit**

Run: `.venv/Scripts/python.exe -m pytest tests -q`
Expected: all pass except the 4 pre-existing strict xfails.

```bash
git add tests/e2e/test_proof_compact_history.py docs/test_reports/2026-09-12-compact-scan-history.md docs/test_reports/2026-09-12-compact-history-proof.json docs/test_reports/2026-09-12-benchmark-after-history.json README.md PROGRESS.md .gitignore
git commit -m "$(cat <<'EOF'
Prove compact scan history on real radar data

Real KTLX and KSOX windows show tracking identical to the retained-scan
pipeline with reacquisition disabled, no full-size grids retained after
ingest, restart continuing identically, and historical requests leaving
live tracking untouched. The offline tracking benchmark is unchanged.
Documents the new history settings and endpoint.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_011VMHfi1nbuux7187WtgQTt
EOF
)"
```
