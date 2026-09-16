# Level III STI as an Ingest Precondition Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A live volume is not tracked or served until its NEXRAD Level III storm-tracking companion has been fetched, or a 120-second deadline has passed — and once published, that decision never changes.

**Architecture:** ARW's live path is `_refresh_live_scan` → `_ingest_to_buffer(site, dt=None)` → `fetch_scan` → `_process_scan_file` → `_track_live_scan`. The gate goes between analysis and tracking: if the volume's NST companion is not yet in S3 and the volume completed less than 120 seconds ago, the volume is not tracked, so `_newest_live_scan` keeps returning the previous fully-analyzed volume. Past the deadline it is tracked without the companion and that outcome is recorded on the scan, so it is never revisited. The companion parse mirrors `src/validation/nmd.py`: raw-bytes regex over the Level III tabular block, no new dependency.

**Tech Stack:** Python 3.11+, boto3 (already a dependency, used by `src/ingest.py` for the `unidata-nexrad-level3` bucket), Py-ART's `cartesian_to_geographic_aeqd` for az/range → lat/lon, pytest, uv. **No new dependencies.**

**Spec:** `docs/superpowers/specs/2026-09-15-level3-companion-ingest-design.md`

## Global Constraints

- **Only the Ingest Manager makes network calls** (CLAUDE.md). All S3 access in this plan goes through `src/ingest.py`. No other module may call the network.
- **`scanTimestamp` → GeoJSON content must be a function.** A volume is published exactly once, fully analyzed. Never re-publish a volume with different content under the same timestamp.
- **Deadline, then freeze.** 120 seconds past volume completion. When it expires, publish without the companion and never revisit that volume.
- **Companion products are NST only** in this plan. NMD, EET, DVL, HHC are out of scope.
- **Level III publish timing, measured 2026-09-15 over 29 KIWA volumes:** volume cadence mean 272s; Level III publishes mean 277s after volume *start*, i.e. within ~5s of volume completion, worst observed 45s after. The 120s deadline is ~2.7x that worst case.
- **STI reports motion direction FROM; ARW reports direction TOWARD.** Do not compare them without adding 180. This plan stores STI movement verbatim and makes no comparison; any later corroboration must apply the offset.
- **Run tests before committing.** No commit without green tests for the affected code (CLAUDE.md).
- **Test commands:** `uv run pytest <path> -v` from the repo root. The full suite takes ~21 minutes; run targeted files during tasks and the full suite once at the end.

---

## File Structure

| File | Change | Responsibility |
|---|---|---|
| `src/level3/__init__.py` | Create | Package marker for Level III product parsers. |
| `src/level3/sti.py` | Create | Parse the STI (product 58) tabular block into typed storm cells. Pure function over bytes — no network, no I/O. |
| `src/ingest.py` | Modify (after `download_level3`, line ~137) | `latest_sti_key_for_volume` and `fetch_sti_for_volume`: the only network access in this plan. |
| `src/server.py` | Modify (`_ingest_to_buffer`, lines 472-516) | The precondition gate and the deadline. |
| `src/buffer.py` | Modify | Carry the companion outcome on the scan so the freeze is auditable. |
| `tests/unit/test_level3_sti.py` | Create | Parser tests against the two real fixtures. |
| `tests/unit/test_ingest_sti_companion.py` | Create | Key selection and fetch, with S3 mocked. |
| `tests/unit/test_live_companion_gate.py` | Create | The gate, the deadline, and the freeze. |
| `tests/fixtures/level3/IWA_NST_2026_09_16_01_34_20` | **Already present** | Real KIWA product, 4 storm cells, all with movement. |
| `tests/fixtures/level3/IWA_NST_2026_09_16_02_45_06` | **Already present** | Real KIWA product, 2 storm cells, one of them `NEW` (no movement yet). |

Tasks are sequential: 1 (parser) → 2 (fetch) → 3 (gate) → 4 (freeze record).

---

### Task 1: Parse the STI tabular block

**Files:**
- Create: `src/level3/__init__.py` (empty file)
- Create: `src/level3/sti.py`
- Test: `tests/unit/test_level3_sti.py`

**Interfaces:**
- Consumes: `NEXRAD_SITES` from `src.sites` (a list of dicts with `site_id`, `latitude`, `longitude`); `cartesian_to_geographic_aeqd` from `pyart.core.transforms`.
- Produces: `StormCell` (frozen dataclass: `cell_id: str`, `azimuth_deg: float`, `range_km: float`, `latitude: float`, `longitude: float`, `movement_deg: float | None`, `movement_kt: float | None`) and `parse_sti(data: bytes, site_id: str) -> tuple[datetime | None, list[StormCell]]`. Tasks 2-4 use these names exactly.

The format, confirmed by reading the real fixtures: the tabular block is **plain ASCII in the raw bytes**, each 80-character line preceded by a 2-byte big-endian length (`\x00\x50`). No decompression. A header line reads `RADAR ID 524  DATE/TIME 09:16:26/02:45:06` — that is `MM:DD:YY/HH:MM:SS`, a **two-digit year**. Cell rows look like:

```
  E3     273/ 63   209/  8     274/ 62   276/ 61   278/ 61   279/ 60    0.8/ 0.9
  M3     315/ 58     NEW       NO DATA   NO DATA   NO DATA   NO DATA    0.0/ 0.0
```

Columns: cell id, current AZRAN (deg/NM), movement (deg/kt) **or the literal `NEW`** for a cell with no track yet, four forecast positions, forecast/mean error.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_level3_sti.py`:

```python
from datetime import datetime
from pathlib import Path

from src.level3.sti import StormCell, parse_sti

FIXTURES = Path("tests/fixtures/level3")


def test_four_cell_product_is_parsed_from_its_table():
    raw = (FIXTURES / "IWA_NST_2026_09_16_01_34_20").read_bytes()

    when, cells = parse_sti(raw, "KIWA")

    assert when == datetime(2026, 9, 16, 1, 34, 20)
    assert [c.cell_id for c in cells] == ["E3", "G3", "W2", "H3"]
    e3 = cells[0]
    assert e3.azimuth_deg == 273.0
    assert round(e3.range_km, 1) == 116.7
    assert e3.movement_deg == 209.0
    assert e3.movement_kt == 8.0


def test_a_new_cell_has_no_movement_yet():
    raw = (FIXTURES / "IWA_NST_2026_09_16_02_45_06").read_bytes()

    when, cells = parse_sti(raw, "KIWA")

    assert when == datetime(2026, 9, 16, 2, 45, 6)
    assert [c.cell_id for c in cells] == ["J3", "M3"]
    assert cells[0].movement_deg == 172.0
    assert cells[1].movement_deg is None
    assert cells[1].movement_kt is None


def test_cells_are_georeferenced_from_the_radar():
    raw = (FIXTURES / "IWA_NST_2026_09_16_02_45_06").read_bytes()

    _, cells = parse_sti(raw, "KIWA")

    # J3: azimuth 295, 73 NM from KIWA (33.2892, -111.670) -- west-northwest
    # and well outside the metro, verified against the product's own forecast
    # positions and against RainViewer's mosaic for the same volume.
    assert round(cells[0].latitude, 2) == 33.80
    assert round(cells[0].longitude, 2) == -113.00


def test_a_product_with_no_table_yields_no_cells():
    when, cells = parse_sti(b"not a level three product", "KIWA")

    assert when is None
    assert cells == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_level3_sti.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.level3'`

- [ ] **Step 3: Create the package marker**

Create `src/level3/__init__.py` as an empty file.

- [ ] **Step 4: Write the parser**

Create `src/level3/sti.py`. This mirrors `src/validation/nmd.py` in structure — raw-bytes regex, Py-ART aeqd georeferencing, no new dependency:

```python
"""NWS Level III storm tracking information (STI, product 58) from its tabular text.

The product's alphanumeric table lists each tracked cell as

  STORM    CURRENT POSITION              FORECAST POSITIONS               ERROR
   ID     AZRAN     MOVEMENT    15 MIN    30 MIN    45 MIN    60 MIN    FCST/MEAN
    E3     273/ 63   209/  8     274/ 62   276/ 61   278/ 61   279/ 60    0.8/ 0.9
    M3     315/ 58     NEW       NO DATA   NO DATA   NO DATA   NO DATA    0.0/ 0.0

with azimuth in degrees and range in nautical miles from the radar. A cell the
algorithm has only just identified prints NEW in place of a movement.

MOVEMENT is the direction the cell is coming FROM, the meteorological
convention. ARW reports storm motion as the direction of travel, so the two
differ by 180 degrees and must never be compared without that offset. This
parser stores the product's value verbatim and makes no comparison.
"""

from dataclasses import dataclass
from datetime import datetime
import math
import re

import numpy as np
from pyart.core.transforms import cartesian_to_geographic_aeqd

from src.sites import NEXRAD_SITES

KM_PER_NM = 1.852
_TIME = re.compile(rb"DATE/TIME\s+(\d\d):(\d\d):(\d\d)/(\d\d):(\d\d):(\d\d)")
_ROW = re.compile(rb"^([A-Z]\d+)\s+(\d{1,3})/\s*(\d{1,3})\s+(?:(\d{1,3})/\s*(\d{1,3})|NEW)\b")


@dataclass(frozen=True)
class StormCell:
    cell_id: str
    azimuth_deg: float
    range_km: float
    latitude: float
    longitude: float
    movement_deg: float | None
    movement_kt: float | None


def parse_sti(data: bytes, site_id: str) -> tuple[datetime | None, list[StormCell]]:
    site = next(s for s in NEXRAD_SITES if s["site_id"] == site_id)
    stamp = _TIME.search(data)
    when = None
    if stamp is not None:
        month, day, year, hour, minute, second = (int(g) for g in stamp.groups())
        when = datetime(2000 + year, month, day, hour, minute, second)
    cells: list[StormCell] = []
    for line in re.split(rb"[\r\n]+|\x00[\x00-\xff]", data):
        line = line.strip(b"\x00 ")
        match = _ROW.match(line)
        if match is None:
            continue
        cell_id, azimuth, range_nm, move_deg, move_kt = match.groups()
        range_km = int(range_nm) * KM_PER_NM
        bearing = math.radians(int(azimuth))
        lon, lat = cartesian_to_geographic_aeqd(
            np.array([range_km * 1000.0 * math.sin(bearing)]),
            np.array([range_km * 1000.0 * math.cos(bearing)]),
            site["longitude"], site["latitude"],
        )
        cells.append(StormCell(
            cell_id=cell_id.decode(),
            azimuth_deg=float(int(azimuth)),
            range_km=range_km,
            latitude=float(np.ravel(lat)[0]),
            longitude=float(np.ravel(lon)[0]),
            movement_deg=None if move_deg is None else float(int(move_deg)),
            movement_kt=None if move_kt is None else float(int(move_kt)),
        ))
    return when, cells
```

The split pattern `\x00[\x00-\xff]` consumes the 2-byte line-length prefix; `[\r\n]+` is kept so a text-form product still parses.

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_level3_sti.py -v`
Expected: PASS, 4 tests.

- [ ] **Step 6: Commit**

```bash
git add src/level3/ tests/unit/test_level3_sti.py tests/fixtures/level3/IWA_NST_2026_09_16_01_34_20 tests/fixtures/level3/IWA_NST_2026_09_16_02_45_06
git commit -m "Read the NWS storm tracking table for a volume"
```

---

### Task 2: Fetch a volume's STI companion

**Files:**
- Modify: `src/ingest.py`, immediately after `download_level3` (ends line ~137)
- Test: `tests/unit/test_ingest_sti_companion.py`

**Interfaces:**
- Consumes: `list_level3_keys(site_id, product, start, end) -> list[str]` and `download_level3(key) -> str` (both already in `src/ingest.py`); `level3_key_time(key) -> datetime`.
- Produces: `STI_PRODUCT = "NST"`, `sti_key_for_volume(site_id: str, volume_time: datetime) -> str | None`, and `fetch_sti_for_volume(site_id: str, volume_time: datetime) -> bytes | None`. Tasks 3-4 use these names exactly.

A Level III product's key time equals the volume time exactly — verified against KIWA `2026-09-16T01:34:20Z`, where ARW's `scanTimestamp` and the product's key both read `01_34_20`. So the lookup is an exact match, not a nearest-neighbour search. Search a one-second window and require equality.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_ingest_sti_companion.py`:

```python
from datetime import datetime

import src.ingest as ingest


def test_key_for_a_volume_matches_the_volume_time_exactly(monkeypatch):
    monkeypatch.setattr(ingest, "list_level3_keys", lambda site, product, start, end: [
        "IWA_NST_2026_09_16_01_34_20",
    ])

    key = ingest.sti_key_for_volume("KIWA", datetime(2026, 9, 16, 1, 34, 20))

    assert key == "IWA_NST_2026_09_16_01_34_20"


def test_no_key_when_the_companion_has_not_published_yet(monkeypatch):
    monkeypatch.setattr(ingest, "list_level3_keys", lambda site, product, start, end: [])

    assert ingest.sti_key_for_volume("KIWA", datetime(2026, 9, 16, 1, 34, 20)) is None


def test_a_key_for_a_different_volume_is_not_accepted(monkeypatch):
    monkeypatch.setattr(ingest, "list_level3_keys", lambda site, product, start, end: [
        "IWA_NST_2026_09_16_01_29_41",
    ])

    assert ingest.sti_key_for_volume("KIWA", datetime(2026, 9, 16, 1, 34, 20)) is None


def test_fetch_returns_the_product_bytes(monkeypatch, tmp_path):
    product = tmp_path / "IWA_NST_2026_09_16_01_34_20"
    product.write_bytes(b"storm table bytes")
    monkeypatch.setattr(ingest, "sti_key_for_volume", lambda site, when: "IWA_NST_2026_09_16_01_34_20")
    monkeypatch.setattr(ingest, "download_level3", lambda key: str(product))

    assert ingest.fetch_sti_for_volume("KIWA", datetime(2026, 9, 16, 1, 34, 20)) == b"storm table bytes"


def test_fetch_returns_none_when_no_companion_exists(monkeypatch):
    monkeypatch.setattr(ingest, "sti_key_for_volume", lambda site, when: None)

    assert ingest.fetch_sti_for_volume("KIWA", datetime(2026, 9, 16, 1, 34, 20)) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_ingest_sti_companion.py -v`
Expected: FAIL — `AttributeError: module 'src.ingest' has no attribute 'sti_key_for_volume'`

- [ ] **Step 3: Write the fetch helpers**

In `src/ingest.py`, immediately after `download_level3`:

```python
STI_PRODUCT = "NST"


def sti_key_for_volume(site_id: str, volume_time: datetime) -> str | None:
    """The storm-tracking product for exactly this volume, or None.

    A Level III product's key time equals its volume time, so this is an
    exact match. Accepting a near miss would silently attach a neighbouring
    volume's storm cells to this one.
    """
    keys = list_level3_keys(site_id, STI_PRODUCT, volume_time, volume_time)
    for key in keys:
        if level3_key_time(key) == volume_time:
            return key
    return None


def fetch_sti_for_volume(site_id: str, volume_time: datetime) -> bytes | None:
    """Download this volume's storm-tracking companion, or None if unpublished."""
    key = sti_key_for_volume(site_id, volume_time)
    if key is None:
        return None
    with open(download_level3(key), "rb") as handle:
        return handle.read()
```

`list_level3_keys` already filters on `start <= level3_key_time(key) <= end`, so passing the volume time as both bounds narrows the listing; the equality check then guards against any boundary looseness.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_ingest_sti_companion.py -v`
Expected: PASS, 5 tests.

- [ ] **Step 5: Commit**

```bash
git add src/ingest.py tests/unit/test_ingest_sti_companion.py
git commit -m "Fetch the storm tracking companion for one volume"
```

---

### Task 3: Hold a volume back until its companion exists

**Files:**
- Modify: `src/buffer.py` (the `BufferedScan` dataclass)
- Modify: `src/server.py:472-516` (`_ingest_to_buffer`)
- Test: `tests/unit/test_live_companion_gate.py`

**Interfaces:**
- Consumes: `fetch_sti_for_volume` from Task 2; `parse_sti` and `StormCell` from Task 1; `scan_time(filename) -> datetime` (`src/ingest.py:138`); `_track_live_scan(site_id, buffered) -> CompactScan | None` (`src/server.py:368`).
- Produces: `BufferedScan.sti_cells: list | None` and `BufferedScan.companion_state: str` (one of `"present"`, `"deadline_expired"`, `"not_applicable"`); `STI_DEADLINE_SECONDS = 120.0` and `_companion_ready(site_id, volume_time) -> tuple[list | None, str] | None` in `src/server.py`. Task 4 uses these names exactly.

`_companion_ready` returns `None` to mean **hold this volume back**; otherwise a `(cells, state)` pair to publish with.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_live_companion_gate.py`:

```python
from datetime import datetime, timedelta, timezone

import pytest

import src.server as server


def _volume_time(age_seconds: float) -> datetime:
    return datetime.now(timezone.utc) - timedelta(seconds=age_seconds)


def test_a_volume_with_its_companion_publishes_immediately(monkeypatch):
    raw = open("tests/fixtures/level3/IWA_NST_2026_09_16_02_45_06", "rb").read()
    monkeypatch.setattr(server, "fetch_sti_for_volume", lambda site, when: raw)

    result = server._companion_ready("KIWA", datetime(2026, 9, 16, 2, 45, 6))

    assert result is not None
    cells, state = result
    assert state == "present"
    assert [c.cell_id for c in cells] == ["J3", "M3"]


def test_a_fresh_volume_without_its_companion_is_held_back(monkeypatch):
    monkeypatch.setattr(server, "fetch_sti_for_volume", lambda site, when: None)

    assert server._companion_ready("KIWA", _volume_time(10)) is None


def test_past_the_deadline_the_volume_publishes_without_the_companion(monkeypatch):
    monkeypatch.setattr(server, "fetch_sti_for_volume", lambda site, when: None)

    result = server._companion_ready("KIWA", _volume_time(server.STI_DEADLINE_SECONDS + 1))

    assert result is not None
    cells, state = result
    assert cells is None
    assert state == "deadline_expired"


def test_a_companion_fetch_failure_does_not_stall_past_the_deadline(monkeypatch):
    def boom(site, when):
        raise RuntimeError("s3 unavailable")
    monkeypatch.setattr(server, "fetch_sti_for_volume", boom)

    assert server._companion_ready("KIWA", _volume_time(10)) is None
    result = server._companion_ready("KIWA", _volume_time(server.STI_DEADLINE_SECONDS + 1))
    assert result == (None, "deadline_expired")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_live_companion_gate.py -v`
Expected: FAIL — `AttributeError: module 'src.server' has no attribute '_companion_ready'`

- [ ] **Step 3: Add the companion fields to BufferedScan**

In `src/buffer.py`, add to the `BufferedScan` dataclass, after the existing optional fields so no positional call site breaks:

```python
    # The Level III storm-tracking cells for this exact volume, or None when
    # the companion had not published before the deadline. Set once, when the
    # volume is first analyzed, and never revisited: a published volume's
    # content must stay a function of its scanTimestamp.
    sti_cells: list | None = None
    companion_state: str = "not_applicable"
```

- [ ] **Step 4: Write the gate**

In `src/server.py`, add near the other module constants (beside `ELEVATED_BEAM_HEIGHT_KM`, line 126):

```python
STI_DEADLINE_SECONDS = 120.0
```

Add the import alongside the existing `src.ingest` imports:

```python
from src.ingest import fetch_sti_for_volume
from src.level3.sti import parse_sti
```

Then add the gate function above `_ingest_to_buffer`:

```python
def _companion_ready(site_id: str, volume_time: datetime) -> tuple[list | None, str] | None:
    """Decide whether this volume may be published yet.

    Returns None to hold the volume back — its companion has not published and
    the deadline has not passed, so the previous fully-analyzed volume stays
    newest. Otherwise returns the cells (or None past the deadline) and the
    state to record. Measured 2026-09-15: Level III lands within ~5s of volume
    completion, worst observed 45s, so the deadline is reached only when the
    product is genuinely missing rather than merely late.
    """
    try:
        raw = fetch_sti_for_volume(site_id, volume_time)
    except Exception:
        _logger.exception("Unable to fetch the storm tracking companion for %s", site_id)
        raw = None
    if raw is not None:
        _, cells = parse_sti(raw, site_id)
        return cells, "present"
    reference = volume_time if volume_time.tzinfo else volume_time.replace(tzinfo=timezone.utc)
    age = (datetime.now(timezone.utc) - reference).total_seconds()
    if age >= STI_DEADLINE_SECONDS:
        return None, "deadline_expired"
    return None
```

Ensure `timezone` is imported from `datetime` at the top of `src/server.py`; add it to the existing import if absent.

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_live_companion_gate.py -v`
Expected: PASS, 4 tests.

- [ ] **Step 6: Commit**

```bash
git add src/buffer.py src/server.py tests/unit/test_live_companion_gate.py
git commit -m "Decide whether a volume may publish yet"
```

---

### Task 4: Wire the gate into the live ingest path and freeze the outcome

**Files:**
- Modify: `src/server.py:472-516` (`_ingest_to_buffer`)
- Test: `tests/unit/test_live_companion_gate.py` (extend)

**Interfaces:**
- Consumes: `_companion_ready(site_id, volume_time)` from Task 3; `BufferedScan.sti_cells` / `.companion_state` from Task 3; `scan_time(filename)` from `src/ingest.py:138`.
- Produces: no new names. `_ingest_to_buffer` gains the gate.

The live branch is the only one gated. A historical request (`dt is not None`) is exact and synchronous and must stay so — `companion_state` remains `"not_applicable"` there.

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_live_companion_gate.py`:

```python
def test_a_held_back_volume_is_not_tracked(monkeypatch, tmp_path):
    """Holding back must leave the previous volume newest, not publish a
    partially analyzed one under a timestamp a client will treat as final."""
    tracked = []
    monkeypatch.setattr(server, "_companion_ready", lambda site, when: None)
    monkeypatch.setattr(server, "_track_live_scan", lambda site, buffered: tracked.append(buffered))
    monkeypatch.setattr(server, "fetch_scan", lambda site, dt=None: "cache/KIWA/KIWA20260916_024506_V06")
    monkeypatch.setattr(server, "_source_key", lambda path: None)
    monkeypatch.setattr(server, "_process_scan_file", lambda site, path: server.BufferedScan(
        timestamp=datetime(2026, 9, 16, 2, 45, 6), site_id="KIWA",
        reflectivity_data=None, detected_objects=[], labeled_grid=None, object_masks={},
    ))

    with pytest.raises(server.CompanionPending):
        server._ingest_to_buffer("KIWA")

    assert tracked == []


def test_a_published_volume_records_the_companion_outcome(monkeypatch):
    cells = ["sentinel cell"]
    tracked = []
    monkeypatch.setattr(server, "_companion_ready", lambda site, when: (cells, "present"))
    monkeypatch.setattr(server, "_track_live_scan", lambda site, buffered: tracked.append(buffered) or object())
    monkeypatch.setattr(server, "fetch_scan", lambda site, dt=None: "cache/KIWA/KIWA20260916_024506_V06")
    monkeypatch.setattr(server, "_source_key", lambda path: None)
    monkeypatch.setattr(server, "_process_scan_file", lambda site, path: server.BufferedScan(
        timestamp=datetime(2026, 9, 16, 2, 45, 6), site_id="KIWA",
        reflectivity_data=None, detected_objects=[], labeled_grid=None, object_masks={},
    ))

    buffered = server._ingest_to_buffer("KIWA")

    assert buffered.companion_state == "present"
    assert buffered.sti_cells == cells
    assert tracked and tracked[0].companion_state == "present"
```

`BufferedScan` is a plain dataclass with no runtime validation (`src/buffer.py:27`), so `reflectivity_data=None` constructs fine. Both tests monkeypatch `_track_live_scan`, so nothing downstream inspects those fields.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_live_companion_gate.py -v`
Expected: FAIL — `AttributeError: module 'src.server' has no attribute 'CompanionPending'`

- [ ] **Step 3: Wire the gate in**

In `src/server.py`, define the signal beside the other module-level classes:

```python
class CompanionPending(Exception):
    """This volume's Level III companion has not published and the deadline
    has not passed. The caller keeps serving the previous volume."""
```

In `_ingest_to_buffer`, replace the live-tracking branch. It currently reads:

```python
        if dt is None and _track_live_scan(normalized_site, buffered) is not None:
```

Insert immediately before that line:

```python
        if dt is None:
            decision = _companion_ready(normalized_site, buffered.timestamp)
            if decision is None:
                raise CompanionPending(normalized_site)
            buffered.sti_cells, buffered.companion_state = decision
```

Leave the `if dt is None and _track_live_scan(...)` line unchanged beneath it.

- [ ] **Step 4: Let the refresher treat a pending companion as a normal wait**

In `_refresh_live_scan` (line 561), a held-back volume is not an error — it is the gate working. Change the body so `CompanionPending` neither logs an exception nor records a refresh error:

```python
    try:
        _ingest_to_buffer(normalized_site)
        with _state_lock:
            _refresh_errors.pop(normalized_site, None)
    except CompanionPending:
        # Not a failure: the volume is analyzed but its companion has not
        # published yet. Retry soon rather than waiting a full refresh
        # interval, and leave the previous volume live and unchanged.
        with _state_lock:
            _refresh_errors.pop(normalized_site, None)
            _refresh_started_at[normalized_site] = datetime.now() - timedelta(
                seconds=_refresh_interval_seconds - LIVE_REFRESH_FAILURE_RETRY_SECONDS
            )
    except Exception as exc:  # pragma: no cover - exercised by deployment failures
```

Leave the existing `except Exception` body and the `finally` block unchanged.

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_live_companion_gate.py tests/unit/test_ingest_sti_companion.py tests/unit/test_level3_sti.py -v`
Expected: PASS.

- [ ] **Step 6: Run the full suite**

Run: `uv run pytest -q`
Expected: PASS. Takes ~21 minutes. If `tests/e2e/test_proof_compact_history.py` fails, check whether its hand-rolled reference tracker needs to mirror a new attachment, as it did for motion — that file's reference must match what the live store does.

- [ ] **Step 7: Commit**

```bash
git add src/server.py tests/unit/test_live_companion_gate.py
git commit -m "Hold a live volume back until its companion is in hand"
```

---

### Task 5: Answer a cold-start request honestly instead of erroring

**Files:**
- Modify: `src/server.py:623-635` (`_live_or_ingest`)
- Test: `tests/unit/test_live_companion_gate.py` (extend)

**Interfaces:**
- Consumes: `CompanionPending` from Task 4; `LIVE_REFRESH_FAILURE_RETRY_SECONDS` (already in `src/server.py`).
- Produces: no new names. `_live_or_ingest` translates `CompanionPending` into an HTTP 503.

**Found during self-review of this plan.** `_live_or_ingest` has a cold-start branch that runs ingest *on the request path*:

```python
newest = _newest_live_scan(normalized_site)
if newest is None:
    return _ingest_to_buffer(normalized_site), False
```

For a site with retained scans this is unreachable — the gate holds the new volume back and the previous one is served, exactly as the spec wants. But on a **cold** site there is no previous volume, so `CompanionPending` would escape to all five GeoJSON endpoints (`src/server.py` lines 1093, 1104, 1164, 1213, 1258) as an unhandled 500.

`/map/status` already handles this correctly without any change — a held-back volume never becomes `completed`, so `available` stays false and a polling client waits. Only the GeoJSON endpoints need the translation. 503 with `Retry-After` is the honest answer: the request is valid, the data genuinely is not ready yet.

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_live_companion_gate.py`:

```python
from fastapi import HTTPException


def test_a_cold_site_waiting_on_its_companion_answers_503_not_500(monkeypatch):
    monkeypatch.setattr(server, "_newest_live_scan", lambda site: None)

    def pending(site, dt=None):
        raise server.CompanionPending(site)
    monkeypatch.setattr(server, "_ingest_to_buffer", pending)

    with pytest.raises(HTTPException) as caught:
        server._live_or_ingest("KIWA")

    assert caught.value.status_code == 503
    assert caught.value.headers["Retry-After"] == str(int(server.LIVE_REFRESH_FAILURE_RETRY_SECONDS))


def test_a_warm_site_is_unaffected_by_the_gate(monkeypatch):
    """A site with a retained scan serves it; the gate only ever withholds the
    newer volume, so the previous fully-analyzed one stays available."""
    class _Retained:
        def to_buffered_scan(self):
            return "previous fully-analyzed volume"

    monkeypatch.setattr(server, "_newest_live_scan", lambda site: _Retained())
    monkeypatch.setattr(server, "_schedule_live_refresh", lambda site, force=False: True)

    buffered, refreshing = server._live_or_ingest("KIWA")

    assert buffered == "previous fully-analyzed volume"
    assert refreshing is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_live_companion_gate.py -k "cold_site or warm_site" -v`
Expected: the cold-site test FAILS with `CompanionPending` escaping instead of `HTTPException`.

- [ ] **Step 3: Translate the signal**

In `src/server.py`, change the cold-start branch of `_live_or_ingest`:

```python
    newest = _newest_live_scan(normalized_site)
    if newest is None:
        try:
            return _ingest_to_buffer(normalized_site), False
        except CompanionPending as exc:
            # A cold site has no previous volume to fall back to, and this one
            # is not publishable yet. The request is valid; the data is not
            # ready. Never serve a volume whose analysis is still incomplete.
            raise HTTPException(
                status_code=503,
                detail="Preparing the requested radar scan.",
                headers={"Retry-After": str(int(LIVE_REFRESH_FAILURE_RETRY_SECONDS))},
            ) from exc
    return newest.to_buffered_scan(), _schedule_live_refresh(normalized_site)
```

`HTTPException` is already imported at `src/server.py:14`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_live_companion_gate.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/server.py tests/unit/test_live_companion_gate.py
git commit -m "Tell a cold-start caller to wait rather than failing"
```

---

## Verification

- [ ] Start ARW: `uv run uvicorn src.server:app --host 127.0.0.1 --port 8000`
- [ ] Request a live layer for a site with active storms:
      `curl -s "http://127.0.0.1:8000/map/storms.geojson?latitude=33.4484&longitude=-112.0741&mode=audiom" | head -c 400`
- [ ] Confirm the served `metadata.scanTimestamp` corresponds to a volume whose companion exists in the bucket — check with
      `uv run python -c "from datetime import datetime; import src.ingest as i; print(i.sti_key_for_volume('KIWA', datetime(YYYY,M,D,H,M,S)))"`
      using the served timestamp. A non-None key means the gate published a volume with its companion in hand.
- [ ] Request the same layer twice and confirm the two responses are byte-identical for the same `scanTimestamp` — the invariant this plan exists to hold.

## Out of scope

- **Annotating features with `nws_cell_ids`.** That is spec Decision 4 (annotate vs. split), still open. This plan stores the cells on the scan and puts nothing new in the GeoJSON.
- **Segmentation seeds.** Depends on Decision 4.
- **Other companions.** NMD, EET, DVL, HHC. The mechanism here is what makes them additive.
- **Corroborating ARW motion against STI movement.** Requires the 180-degree offset described in Global Constraints; a separate plan.
- **`_rebuild_tracking`.** It replays retained scans whose companion decision was already frozen; it must not re-fetch or re-decide.
