# Truthful Rotation Detection Implementation Plan (Part A of #8/#9)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Validate rotation detection against confirmed events, fix it by measurement, and run it in the live server, so rotation evidence reaches speech and the map only at levels that beat null cases.

**Architecture:**
- The ingest manager gains fetchers for SPC storm reports, NWS Level III mesocyclone detections (NMD), and Level II volumes in a time window.
- Pure validation modules parse reports and NMD products and compute pre-registered metrics.
- A corpus builder assembles tornado, severe and clear-air cases.
- An evaluator runs production analysis once per volume and scores every detector configuration from a sweep.
- The detector takes a `RotationDetectorConfig`: defaults stay byte-identical until the sweep selects new values by a rule written here before any measurement.

**Tech Stack:** Python 3.11, numpy, Py-ART, boto3 (unsigned S3), stdlib `urllib`/`csv`, pytest.

**Spec:** `docs/superpowers/specs/2026-09-14-detection-truth-design.md` (Part A). Part B (classifier) gets its own plan after this one.

## Outcome (2026-09-14): executed through Task 10 and merged; no rotation is spoken

- **Tasks 1-10 are done.** The pre-registered selection chose configuration 1: shear 15 m/s, fold
  rejection, couplets across the beam only, ground-overlap merging. Only rank 3 (persistent) passed
  the speak rule.
- **Task 10's live check failed the speak rule's intent.** Persistent evidence was spoken for
  one-gate noise couplets in ordinary rain at KJAX and KAMA, with no severe reports.
- **Owner decision:** `SPOKEN_ROTATION_EVIDENCE` is empty. Analysis still runs live, and assessments
  stay in the data, labelled.
- **Extra work beyond the plan:**
  - speedups with byte-identical output;
  - compact scans keep velocity regions;
  - map descriptions follow the spoken set.
- **Not done:** Part B (classifier, #8) has no plan yet.

Full results and the live-check evidence: `docs/test_reports/2026-09-14-rotation-corpus.md`. Follow-up: issue #9.

## Global Constraints

- Only `src/ingest.py` makes network calls (CLAUDE.md). Validation modules and scripts read the cache.
- Labels never come from the thing being scored; results are reported per case, never only pooled (spec, Principles).
- Every threshold change records its population, source and false-alarm / missed-signal trade-off before changing production (`docs/validation/rotation-signal-corpus.json` `calibration_policy`).
- Level III bucket: `unidata-nexrad-level3`, unsigned. Key format `SSS_PRD_YYYY_MM_DD_HH_MM_SS`, where `SSS` is the site id without the leading K.
- SPC report CSVs: `https://www.spc.noaa.gov/climo/reports/{yymmdd}_rpts_{torn|hail|wind}.csv`. The SPC day runs 12Z to 12Z, so an `HHMM` before 1200 belongs to the next UTC date.
- **Pre-registered metrics (fixed now, before any measurement):**
  - Evidence rank: `unconfirmed`=1, `vertically_confirmed`=2, `corroborated`=2, `persistent`=3. "At level L" means rank >= L.
  - **Tornado event hit at L:** any assessment at L within 10 km of the report location, in any case volume whose start time is 20 minutes before to 10 minutes after the report.
  - **NMD agreement:** per volume, using the NMD product closest in time within 3 minutes; an assessment and a detection match within 5 km. Report matched, ARW-only and NMD-only.
  - **Null-storm probe:** every detected storm object centroid in tornado, hail and wind case volumes that is more than 50 km from every SPC report (any kind, within 60 minutes) and from every NMD detection in that volume. A probe is a false alarm at L when an assessment at L lies within 10 km of it.
  - **Clear-air false alarms at L:** assessments at L in clear-air volumes.
  - **Speak rule:** L is spoken only if all of these hold:
    1. clear-air false alarms at L == 0;
    2. tornado event hits at L >= 2;
    3. tornado hit fraction at L >= 3 × null-storm false-alarm fraction at L.
  - **Configuration selection rule:** among swept configurations, choose the one with the most tornado event hits at rank 1 whose null-storm false-alarm fraction at rank 1 does not exceed the baseline detector's. Break ties by the lower null-storm false-alarm fraction, then the higher NMD matched count, then the earlier grid order.

## Corrections made during execution (2026-09-14)

- **Azimuthal pairs.** The rule is adjacent rays at the **same gate** only. The plan's
  "`delta_az == 0` skipped" still admitted diagonal pairs, which mix radial convergence in. The
  test `test_azimuthal_pairs_only_rejects_convergence_and_keeps_rotation` caught this.
- **Diameter test.** A candidate's diameter comes from its shear boundary, not the whole velocity
  block. The large-candidate test uses a long range extent (40 gates, about 18 km).
- **Clear air.** A report-free noon volume is `clear_air` only when under 1% of valid gates reach
  35 dBZ (spec B1). Otherwise it is `null_storm`, scored only by null-storm probes: three such
  volumes held 3-18% strong echo.
- **Baseline and sweep in one run.** The sweep output's first entry is the baseline configuration.
  `docs/test_reports/2026-09-14-rotation-baseline.json` is that entry, extracted, not a separate
  run.

---

### File Structure

**Create:**
- `src/validation/__init__.py` — package marker.
- `src/validation/spc.py` — `StormReport`, `parse_spc_csv`.
- `src/validation/nmd.py` — `MesocycloneDetection`, `parse_nmd`.
- `src/validation/rotation_metrics.py` — metric functions and the speak rule, pure.
- `scripts/build_rotation_corpus.py` — fetches and writes `docs/validation/rotation-signal-corpus-v2.json`.
- `scripts/evaluate_rotation_corpus_v2.py` — runs the analysis once per volume and scores configurations.
- Tests: `tests/unit/test_validation_spc.py`, `tests/unit/test_validation_nmd.py`, `tests/unit/test_rotation_metrics.py`, `tests/unit/test_ingest_validation.py`, `tests/unit/test_rotation_detector_config.py`.
- Fixtures: `tests/fixtures/level3/MVX_NMD_2026_09_12_00_05_32` (two detections) and `tests/fixtures/level3/TLX_NMD_2026_04_10_05_55_23` (none).

**Modify:**
- `src/ingest.py` — SPC, Level III and time-window Level II fetchers.
- `src/velocity.py` — `RotationDetectorConfig`, threaded through detection and `analyze_velocity`.
- `src/summary.py` — spoken evidence levels.
- `src/server.py` — live velocity and rotation analysis.
- `scripts/evaluate_tracking.py`, `scripts/live_replay.py` — the assessed production path.

---

### Task 1: SPC storm reports

**Files:**
- Create: `src/validation/__init__.py`, `src/validation/spc.py`, `tests/unit/test_validation_spc.py`
- Modify: `src/ingest.py` (append)
- Test: `tests/unit/test_ingest_validation.py`

**Interfaces:**
- Produces: `StormReport(kind: str, time_utc: datetime, latitude: float, longitude: float, magnitude: float | None, location: str, state: str)`; `parse_spc_csv(text: str, spc_day: date, kind: str) -> list[StormReport]`; `src.ingest.fetch_spc_reports(spc_day: date, kind: str) -> str` (cached CSV path).

- [ ] **Step 1: Write the failing parser tests**

```python
# tests/unit/test_validation_spc.py
from datetime import date, datetime

from src.validation.spc import parse_spc_csv

TORNADO_CSV = (
    "Time,F_Scale,Location,County,State,Lat,Lon,Comments\n"
    "1945,UNK,3 NNE Lake Tanglewood,Randall,TX,35.10,-101.76,Brief touchdown. (AMA)\n"
    "0050,EF1,8 S Alberta,Stevens,MN,45.46,-96.06,Survey. (MPX)\n"
)
HAIL_CSV = "Time,Size,Location,County,State,Lat,Lon,Comments\n2010,275,2 N Moore,Cleveland,OK,35.38,-97.50,x\n"
WIND_CSV = "Time,Speed,Location,County,State,Lat,Lon,Comments\n2100,UNK,Town,County,OK,35.0,-97.0,x\n2105,70,Town,County,OK,35.1,-97.1,x\n"


def test_reports_before_12z_belong_to_the_next_utc_date():
    reports = parse_spc_csv(TORNADO_CSV, date(2026, 7, 12), "tornado")
    assert reports[0].time_utc == datetime(2026, 7, 12, 19, 45)
    assert reports[1].time_utc == datetime(2026, 7, 13, 0, 50)
    assert (reports[0].latitude, reports[0].longitude) == (35.10, -101.76)
    assert reports[0].magnitude is None and reports[1].magnitude == 1.0
    assert reports[1].state == "MN"


def test_hail_size_is_in_inches_and_wind_speed_in_knots():
    (hail,) = parse_spc_csv(HAIL_CSV, date(2026, 9, 12), "hail")
    assert hail.magnitude == 2.75
    unknown, known = parse_spc_csv(WIND_CSV, date(2026, 9, 12), "wind")
    assert unknown.magnitude is None and known.magnitude == 70.0


def test_header_only_file_has_no_reports():
    assert parse_spc_csv("Time,F_Scale,Location,County,State,Lat,Lon,Comments\n", date(2026, 4, 10), "tornado") == []
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest -q -p no:cacheprovider tests/unit/test_validation_spc.py`
Expected: FAIL during collection with `ModuleNotFoundError: No module named 'src.validation'`.

- [ ] **Step 3: Implement the parser**

```python
# src/validation/__init__.py
"""Validation data: storm reports and operational detections, parsed from cached files."""
```

```python
# src/validation/spc.py
"""SPC storm reports (tornado, hail, wind) parsed from the daily CSV files.

The SPC report day runs 12Z to 12Z: a report time before 1200 belongs to the
next UTC calendar date.
"""

from dataclasses import dataclass
from datetime import date, datetime, timedelta
import csv
import io
import re

MAGNITUDE_COLUMN = {"tornado": "F_Scale", "hail": "Size", "wind": "Speed"}


@dataclass(frozen=True)
class StormReport:
    kind: str
    time_utc: datetime
    latitude: float
    longitude: float
    magnitude: float | None  # tornado EF rating, hail inches, wind knots
    location: str
    state: str


def _magnitude(kind: str, raw: str) -> float | None:
    digits = re.search(r"\d+", raw or "")
    if digits is None:
        return None
    value = float(digits.group())
    return value / 100.0 if kind == "hail" else value


def parse_spc_csv(text: str, spc_day: date, kind: str) -> list[StormReport]:
    reports: list[StormReport] = []
    for row in csv.DictReader(io.StringIO(text)):
        hhmm = (row.get("Time") or "").strip()
        if not hhmm.isdigit() or len(hhmm) > 4:
            continue
        hhmm = hhmm.zfill(4)
        when = datetime(spc_day.year, spc_day.month, spc_day.day, int(hhmm[:2]), int(hhmm[2:]))
        if when.hour < 12:
            when += timedelta(days=1)
        reports.append(StormReport(
            kind=kind,
            time_utc=when,
            latitude=float(row["Lat"]),
            longitude=float(row["Lon"]),
            magnitude=_magnitude(kind, row.get(MAGNITUDE_COLUMN[kind], "")),
            location=row.get("Location", ""),
            state=row.get("State", ""),
        ))
    return reports
```

- [ ] **Step 4: Run the parser tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest -q -p no:cacheprovider tests/unit/test_validation_spc.py`
Expected: 3 passed.

- [ ] **Step 5: Write the failing fetch test**

```python
# tests/unit/test_ingest_validation.py
from datetime import date
from unittest.mock import patch

from src import ingest


def test_fetch_spc_reports_downloads_once_into_the_cache(tmp_path):
    body = b"Time,F_Scale,Location,County,State,Lat,Lon,Comments\n"
    with patch.object(ingest, "CACHE_DIR", str(tmp_path)), patch("src.ingest.urllib.request.urlopen") as urlopen:
        urlopen.return_value.__enter__.return_value.read.return_value = body
        path = ingest.fetch_spc_reports(date(2026, 9, 12), "tornado")
        again = ingest.fetch_spc_reports(date(2026, 9, 12), "tornado")
    assert path == again
    assert path.endswith("260912_rpts_torn.csv")
    assert open(path, "rb").read() == body
    assert urlopen.call_count == 1
    assert urlopen.call_args[0][0] == "https://www.spc.noaa.gov/climo/reports/260912_rpts_torn.csv"
```

- [ ] **Step 6: Run it to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest -q -p no:cacheprovider tests/unit/test_ingest_validation.py`
Expected: FAIL with `AttributeError: module 'src.ingest' has no attribute 'fetch_spc_reports'` (or no attribute `urllib`).

- [ ] **Step 7: Implement the fetcher (append to `src/ingest.py`; add `import urllib.request` and `from datetime import date` at the top)**

```python
SPC_REPORT_URL = "https://www.spc.noaa.gov/climo/reports/{yymmdd}_rpts_{name}.csv"
SPC_FILE_NAMES = {"tornado": "torn", "hail": "hail", "wind": "wind"}


def fetch_spc_reports(spc_day: date, kind: str) -> str:
    """Download one SPC daily report CSV into the cache (once). Returns the local path."""
    name = f"{spc_day:%y%m%d}_rpts_{SPC_FILE_NAMES[kind]}.csv"
    path = os.path.join(os.path.abspath(CACHE_DIR), "spc", name)
    if os.path.isfile(path):
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    url = SPC_REPORT_URL.format(yymmdd=f"{spc_day:%y%m%d}", name=SPC_FILE_NAMES[kind])
    with urllib.request.urlopen(url, timeout=60) as response:
        body = response.read()
    with open(path, "wb") as handle:
        handle.write(body)
    return path
```

- [ ] **Step 8: Run both test files**

Run: `.venv/Scripts/python.exe -m pytest -q -p no:cacheprovider tests/unit/test_validation_spc.py tests/unit/test_ingest_validation.py tests/unit/test_ingest.py`
Expected: all pass.

- [ ] **Step 9: Commit**

```bash
git add src/validation/__init__.py src/validation/spc.py src/ingest.py tests/unit/test_validation_spc.py tests/unit/test_ingest_validation.py
git commit -m "Parse and fetch SPC storm reports for validation"
```

---

### Task 2: NWS Level III mesocyclone detections

**Files:**
- Create: `src/validation/nmd.py`, `tests/unit/test_validation_nmd.py`
- Create fixtures: copy `MVX_NMD_2026_09_12_00_05_32` and `TLX_NMD_2026_04_10_05_55_23` from the scratchpad `level3` folder into `tests/fixtures/level3/` (re-download with the Step 7 fetcher if the scratchpad is gone)
- Modify: `src/ingest.py` (append), `tests/unit/test_ingest_validation.py` (append)

**Interfaces:**
- Consumes: `NEXRAD_SITES` from `src.sites`.
- Produces: `MesocycloneDetection(circulation_id: str, azimuth_deg: float, range_km: float, strength_rank: int, storm_id: str, tvs: bool, latitude: float, longitude: float)`; `parse_nmd(data: bytes, site_id: str) -> tuple[datetime | None, list[MesocycloneDetection]]`; `src.ingest.list_level3_keys(site_id: str, product: str, start: datetime, end: datetime) -> list[str]`; `src.ingest.download_level3(key: str) -> str`; `src.ingest.level3_key_time(key: str) -> datetime`.

- [ ] **Step 1: Write the failing parser tests**

```python
# tests/unit/test_validation_nmd.py
from datetime import datetime
from pathlib import Path

import pytest

from src.sites import haversine_distance_km
from src.validation.nmd import parse_nmd

FIXTURES = Path("tests/fixtures/level3")


def test_a_product_with_two_mesocyclones_is_parsed_from_its_table():
    when, detections = parse_nmd((FIXTURES / "MVX_NMD_2026_09_12_00_05_32").read_bytes(), "KMVX")
    assert when == datetime(2026, 9, 12, 0, 5, 32)
    assert [(d.circulation_id, d.azimuth_deg, d.strength_rank, d.storm_id, d.tvs) for d in detections] == [
        ("916", 147.0, 6, "T1", False),
        ("872", 123.0, 5, "B1", False),
    ]
    assert detections[0].range_km == pytest.approx(110 * 1.852)
    # KMVX is at 47.528 N, 97.325 W; 204 km at 147 degrees lies to the SSE.
    assert detections[0].latitude < 47.528 and detections[0].longitude > -97.325
    assert haversine_distance_km(47.5280, -97.3250, detections[0].latitude, detections[0].longitude) == pytest.approx(203.7, abs=2.0)


def test_a_product_without_detections_has_none():
    when, detections = parse_nmd((FIXTURES / "TLX_NMD_2026_04_10_05_55_23").read_bytes(), "KTLX")
    assert detections == []
```

- [ ] **Step 2: Copy fixtures and run the tests to verify they fail**

Run: `mkdir -p tests/fixtures/level3 && cp "<scratchpad>/level3/MVX_NMD_2026_09_12_00_05_32" "<scratchpad>/level3/TLX_NMD_2026_04_10_05_55_23" tests/fixtures/level3/ && .venv/Scripts/python.exe -m pytest -q -p no:cacheprovider tests/unit/test_validation_nmd.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.validation.nmd'`. First check the KMVX coordinates in `src/sites.py` and use those values in the test.

- [ ] **Step 3: Implement the parser**

```python
# src/validation/nmd.py
"""NWS Level III mesocyclone detection (NMD, product 141) from its tabular text.

The product's alphanumeric table lists each circulation as
  P CIRC  AZRAN   SR STM |-LOW LEVEL-| ... TVS  MOTION   MSI
  P 916  147/110  6  T1  27   34  <15  >21  89  15  27  N   ...   2676
with azimuth in degrees and range in nautical miles from the radar.
"""

from dataclasses import dataclass
from datetime import datetime
import math
import re

import numpy as np
from pyart.core.transforms import cartesian_to_geographic_aeqd

from src.sites import NEXRAD_SITES

KM_PER_NM = 1.852
_TIME = re.compile(rb"DATE:\s*(\d\d)/(\d\d)/(\d{4})\s+TIME:\s*(\d\d):(\d\d):(\d\d)")
_ROW = re.compile(rb"^P\s+(\w+)\s+(\d{1,3})/\s*(\d{1,3})\s+(\d+)[A-Z]?\s+(\w+)\s+(.*)$")


@dataclass(frozen=True)
class MesocycloneDetection:
    circulation_id: str
    azimuth_deg: float
    range_km: float
    strength_rank: int
    storm_id: str
    tvs: bool
    latitude: float
    longitude: float


def parse_nmd(data: bytes, site_id: str) -> tuple[datetime | None, list[MesocycloneDetection]]:
    site = next(s for s in NEXRAD_SITES if s["site_id"] == site_id)
    stamp = _TIME.search(data)
    when = None
    if stamp is not None:
        month, day, year, hour, minute, second = (int(g) for g in stamp.groups())
        when = datetime(year, month, day, hour, minute, second)
    detections: list[MesocycloneDetection] = []
    for line in re.split(rb"[\r\n]+|(?=P {1,2}\d)", data):
        line = line.strip(b"\x00 ")
        match = _ROW.match(line)
        if match is None:
            continue
        circulation, azimuth, range_nm, rank, storm, rest = match.groups()
        tokens = rest.split()
        # rest: RV DV BASE DEPTH STMREL% MAXRV-kft MAXRV-kts TVS ...
        if len(tokens) < 8 or tokens[7] not in (b"Y", b"N"):
            continue
        range_km = int(range_nm) * KM_PER_NM
        bearing = math.radians(int(azimuth))
        lon, lat = cartesian_to_geographic_aeqd(
            np.array([range_km * 1000.0 * math.sin(bearing)]),
            np.array([range_km * 1000.0 * math.cos(bearing)]),
            site["longitude"], site["latitude"],
        )
        detections.append(MesocycloneDetection(
            circulation_id=circulation.decode(),
            azimuth_deg=float(int(azimuth)),
            range_km=range_km,
            strength_rank=int(rank),
            storm_id=storm.decode(),
            tvs=tokens[7] == b"Y",
            latitude=float(np.ravel(lat)[0]),
            longitude=float(np.ravel(lon)[0]),
        ))
    return when, detections
```

- [ ] **Step 4: Run the parser tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest -q -p no:cacheprovider tests/unit/test_validation_nmd.py`
Expected: 2 passed. If a table row fails to match, print the raw lines, correct `_ROW` or the line split against the real bytes, and rerun. Do not relax the assertions.

- [ ] **Step 5: Write the failing Level III fetch tests (append)**

```python
# tests/unit/test_ingest_validation.py (append)
from datetime import datetime


def test_level3_keys_are_listed_per_date_and_filtered_to_the_window():
    pages = [{"Contents": [
        {"Key": "MVX_NMD_2026_09_11_23_58_00"},
        {"Key": "MVX_NMD_2026_09_12_00_05_32"},
        {"Key": "MVX_NMD_2026_09_12_00_40_00"},
    ]}]
    with patch("src.ingest._level3_client") as client:
        client.return_value.get_paginator.return_value.paginate.return_value = pages
        keys = ingest.list_level3_keys("KMVX", "NMD", datetime(2026, 9, 12, 0, 0), datetime(2026, 9, 12, 0, 30))
    assert keys == ["MVX_NMD_2026_09_12_00_05_32"]
    assert ingest.level3_key_time("MVX_NMD_2026_09_12_00_05_32") == datetime(2026, 9, 12, 0, 5, 32)


def test_download_level3_caches_by_site(tmp_path):
    with patch.object(ingest, "CACHE_DIR", str(tmp_path)), patch("src.ingest._level3_client") as client:
        path = ingest.download_level3("MVX_NMD_2026_09_12_00_05_32")
    assert path.endswith("level3\\MVX\\MVX_NMD_2026_09_12_00_05_32") or path.endswith("level3/MVX/MVX_NMD_2026_09_12_00_05_32")
    client.return_value.download_file.assert_called_once()
```

- [ ] **Step 6: Run them to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest -q -p no:cacheprovider tests/unit/test_ingest_validation.py`
Expected: FAIL with `AttributeError: ... has no attribute '_level3_client'`.

- [ ] **Step 7: Implement (append to `src/ingest.py`; add `from datetime import timedelta` at the top)**

```python
LEVEL3_BUCKET = "unidata-nexrad-level3"


def _level3_client():
    """Unsigned S3 client for NWS Level III products. Isolated for mocking."""
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config

    return boto3.client("s3", config=Config(signature_version=UNSIGNED), region_name="us-east-1")


def level3_key_time(key: str) -> datetime:
    return datetime.strptime("_".join(key.split("_")[2:8]), "%Y_%m_%d_%H_%M_%S")


def list_level3_keys(site_id: str, product: str, start: datetime, end: datetime) -> list[str]:
    """Level III product keys for a site whose product time lies within [start, end]."""
    client = _level3_client()
    site = site_id[1:] if len(site_id) == 4 else site_id
    keys: list[str] = []
    day = start.date()
    while day <= end.date():
        prefix = f"{site}_{product}_{day:%Y_%m_%d}"
        for page in client.get_paginator("list_objects_v2").paginate(Bucket=LEVEL3_BUCKET, Prefix=prefix):
            for item in page.get("Contents", []):
                if start <= level3_key_time(item["Key"]) <= end:
                    keys.append(item["Key"])
        day += timedelta(days=1)
    return sorted(keys)


def download_level3(key: str) -> str:
    site = key.split("_")[0]
    path = os.path.join(os.path.abspath(CACHE_DIR), "level3", site, key)
    if os.path.isfile(path):
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    _level3_client().download_file(LEVEL3_BUCKET, key, path)
    return path
```

- [ ] **Step 8: Run the tests**

Run: `.venv/Scripts/python.exe -m pytest -q -p no:cacheprovider tests/unit/test_ingest_validation.py tests/unit/test_validation_nmd.py`
Expected: all pass.

- [ ] **Step 9: Commit**

```bash
git add src/validation/nmd.py src/ingest.py tests/unit/test_validation_nmd.py tests/unit/test_ingest_validation.py tests/fixtures/level3
git commit -m "Parse and fetch NWS Level III mesocyclone detections"
```

---

### Task 3: Level II volumes in a time window

**Files:**
- Modify: `src/ingest.py` (append), `tests/unit/test_ingest_validation.py` (append)

**Interfaces:**
- Consumes: `list_scans_for_date`, `download_scan`.
- Produces: `src.ingest.scan_time(filename: str) -> datetime`; `src.ingest.fetch_scans_between(site_id: str, start: datetime, end: datetime) -> list[str]` (local paths, time-ordered).

- [ ] **Step 1: Write the failing test (append)**

```python
from unittest.mock import MagicMock


def test_fetch_scans_between_spans_midnight_and_filters_to_the_window():
    scans = {
        "2026-09-11": [MagicMock(filename="KMVX20260911_235500_V06")],
        "2026-09-12": [MagicMock(filename="KMVX20260912_000532_V06"), MagicMock(filename="KMVX20260912_004000_V06")],
    }
    with patch("src.ingest.list_scans_for_date", side_effect=lambda site, day: scans[day]), \
         patch("src.ingest.download_scan", side_effect=lambda site, scan: f"/cache/{scan.filename}"):
        paths = ingest.fetch_scans_between("KMVX", datetime(2026, 9, 11, 23, 50), datetime(2026, 9, 12, 0, 30))
    assert paths == ["/cache/KMVX20260911_235500_V06", "/cache/KMVX20260912_000532_V06"]
    assert ingest.scan_time("KTLX20130520_195527_V06.gz") == datetime(2013, 5, 20, 19, 55, 27)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest -q -p no:cacheprovider tests/unit/test_ingest_validation.py -k scans_between`
Expected: FAIL with `AttributeError: ... no attribute 'fetch_scans_between'`.

- [ ] **Step 3: Implement (append)**

```python
def scan_time(filename: str) -> datetime:
    """Start time of a Level II volume from its file name (KTLX20130520_195527_V06[.gz])."""
    return datetime.strptime(os.path.basename(filename)[4:19], "%Y%m%d_%H%M%S")


def fetch_scans_between(site_id: str, start: datetime, end: datetime) -> list[str]:
    """Download every Level II volume for a site starting within [start, end]."""
    paths: list[str] = []
    day = start.date()
    while day <= end.date():
        for scan in list_scans_for_date(site_id, day.strftime("%Y-%m-%d")):
            if start <= scan_time(scan.filename) <= end:
                paths.append(download_scan(site_id, scan))
        day += timedelta(days=1)
    return sorted(paths, key=scan_time)
```

- [ ] **Step 4: Run the tests**

Run: `.venv/Scripts/python.exe -m pytest -q -p no:cacheprovider tests/unit/test_ingest_validation.py tests/unit/test_ingest.py`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/ingest.py tests/unit/test_ingest_validation.py
git commit -m "Fetch Level II volumes within a time window"
```

---

### Task 4: Validation corpus v2

**Files:**
- Create: `scripts/build_rotation_corpus.py`, `docs/validation/rotation-signal-corpus-v2.json` (generated)

**Interfaces:**
- Consumes: `fetch_spc_reports`, `parse_spc_csv`, `list_level3_keys`, `download_level3`, `fetch_scans_between`, `NEXRAD_SITES`, `haversine_distance_km`.
- Produces: a manifest with schema `{"schema_version": 2, "cases": [{"id": str, "label": "tornado"|"hail"|"wind"|"clear_air", "site_id": str, "report": {"kind","time_utc","latitude","longitude","magnitude","location","state"} | null, "volumes": [path], "nmd": [path]}], "reports": [report dicts for every SPC report on the corpus days]}`.

- [ ] **Step 1: Write the builder**

```python
# scripts/build_rotation_corpus.py
"""Build the rotation validation corpus (spec 2026-09-14, A1).

Cases:
- every SPC tornado report on SPC_DAYS, nearest WSR-88D within 150 km,
  volumes from 20 min before to 10 min after the report;
- the 6 largest hail (>= 2.00 in) and 6 strongest wind (>= 65 kt) reports on
  the same days, volumes from 10 min before to 5 min after;
- clear air: the existing six KIWA 2026-07-12 volumes, plus for each tornado
  radar the volume nearest 12:00Z on its SPC day when no report of any kind
  lies within 200 km within 3 h.
Each volume gets the NWS NMD products within 5 minutes of the window.
"""
import json
import sys
from dataclasses import asdict
from datetime import date, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ingest import download_level3, fetch_scans_between, fetch_spc_reports, list_level3_keys
from src.sites import NEXRAD_SITES, haversine_distance_km
from src.validation.spc import parse_spc_csv

SPC_DAYS = [date(2026, 7, 12), date(2026, 7, 14), date(2026, 9, 7), date(2026, 9, 11), date(2026, 9, 12)]
MAX_RADAR_RANGE_KM = 150.0
SEVERE_PER_KIND = 6
KIWA_CLEAR_AIR = [f"cache/KIWA/KIWA20260712_{t}_V06" for t in ("163410", "164255", "165142", "170029", "170914", "171800")]
OUT = ROOT / "docs/validation/rotation-signal-corpus-v2.json"


def nearest_site(lat, lon):
    distance, site = min((haversine_distance_km(s["latitude"], s["longitude"], lat, lon), s["site_id"]) for s in NEXRAD_SITES)
    return (site, distance) if distance <= MAX_RADAR_RANGE_KM else (None, distance)


def relative(path):
    return str(Path(path).resolve().relative_to(ROOT)).replace("\\", "/")


def report_dict(report):
    row = asdict(report)
    row["time_utc"] = report.time_utc.isoformat()
    return row


def case_for(report, before, after, label, index):
    site, distance = nearest_site(report.latitude, report.longitude)
    if site is None:
        print(f"skip {label} {report.location} {report.state}: nearest radar {distance:.0f} km", flush=True)
        return None
    start, end = report.time_utc - before, report.time_utc + after
    volumes = [relative(p) for p in fetch_scans_between(site, start, end)]
    nmd = [relative(download_level3(k)) for k in list_level3_keys(site, "NMD", start - timedelta(minutes=5), end + timedelta(minutes=5))]
    print(f"{label} {report.time_utc} {report.location} {report.state}: {site} {distance:.0f} km, {len(volumes)} volumes, {len(nmd)} NMD", flush=True)
    return {"id": f"{label}-{report.time_utc:%Y%m%d-%H%M}-{index}", "label": label, "site_id": site,
            "report": report_dict(report), "volumes": volumes, "nmd": nmd}


def main():
    reports = []
    for day in SPC_DAYS:
        for kind in ("tornado", "hail", "wind"):
            text = Path(fetch_spc_reports(day, kind)).read_text(encoding="utf-8", errors="replace")
            reports.extend(parse_spc_csv(text, day, kind))
    cases = []
    tornadoes = [r for r in reports if r.kind == "tornado"]
    for index, report in enumerate(tornadoes):
        case = case_for(report, timedelta(minutes=20), timedelta(minutes=10), "tornado", index)
        if case:
            cases.append(case)
    for kind, floor in (("hail", 2.0), ("wind", 65.0)):
        strongest = sorted((r for r in reports if r.kind == kind and r.magnitude is not None and r.magnitude >= floor),
                           key=lambda r: (-r.magnitude, r.time_utc))[:SEVERE_PER_KIND]
        for index, report in enumerate(strongest):
            case = case_for(report, timedelta(minutes=10), timedelta(minutes=5), kind, index)
            if case:
                cases.append(case)
    cases.append({"id": "clear-air-kiwa-20260712", "label": "clear_air", "site_id": "KIWA", "report": None,
                  "volumes": KIWA_CLEAR_AIR, "nmd": []})
    for case in [c for c in cases if c["label"] == "tornado"]:
        spc_day = datetime.fromisoformat(case["report"]["time_utc"])
        noon = datetime(spc_day.year, spc_day.month, spc_day.day, 12, 0) if spc_day.hour >= 12 else datetime(spc_day.year, spc_day.month, spc_day.day, 12, 0) - timedelta(days=1)
        site = next(s for s in NEXRAD_SITES if s["site_id"] == case["site_id"])
        busy = any(abs((r.time_utc - noon).total_seconds()) <= 3 * 3600
                   and haversine_distance_km(site["latitude"], site["longitude"], r.latitude, r.longitude) <= 200.0
                   for r in reports)
        if busy:
            continue
        paths = fetch_scans_between(case["site_id"], noon - timedelta(minutes=5), noon + timedelta(minutes=5))
        if paths and not any(c["id"] == f"clear-air-{case['site_id']}-{noon:%Y%m%d}" for c in cases):
            cases.append({"id": f"clear-air-{case['site_id']}-{noon:%Y%m%d}", "label": "clear_air", "site_id": case["site_id"],
                          "report": None, "volumes": [relative(paths[0])], "nmd": []})
    OUT.write_text(json.dumps({"schema_version": 2, "built_utc": datetime.utcnow().isoformat(timespec="seconds"),
                               "spc_days": [d.isoformat() for d in SPC_DAYS], "cases": cases,
                               "reports": [report_dict(r) for r in reports]}, indent=2), encoding="utf-8")
    print("wrote", OUT, "cases", len(cases), flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the builder**

Run: `.venv/Scripts/python.exe scripts/build_rotation_corpus.py`
Expected: one line per case with site, distance, volume and NMD counts, then `wrote ... cases N`. Check the printed lines:
- every tornado case has at least 3 volumes;
- clear-air cases exist;
- no case has 0 volumes. If one does, investigate (for example, a radar outage) and record it in the corpus report. Do not silently drop it.

- [ ] **Step 3: Record the corpus in a report**

Create `docs/test_reports/2026-09-14-rotation-corpus.md` with one row per case: id, site, distance, volumes, NMD products, and NMD detections in the window (count them with `parse_nmd`). Note where no volume or no NMD exists.

- [ ] **Step 4: Commit**

```bash
git add scripts/build_rotation_corpus.py docs/validation/rotation-signal-corpus-v2.json docs/test_reports/2026-09-14-rotation-corpus.md
git commit -m "Build the rotation validation corpus from SPC reports and NWS detections"
```

---

### Task 5: Metrics, evaluator and baseline

**Files:**
- Create: `src/validation/rotation_metrics.py`, `tests/unit/test_rotation_metrics.py`, `scripts/evaluate_rotation_corpus_v2.py`
- Generated: `docs/test_reports/2026-09-14-rotation-baseline.json`, and a section in `docs/test_reports/2026-09-14-rotation-corpus.md`

**Interfaces:**
- Consumes: `StormReport`, `MesocycloneDetection`, `parse_nmd`, `analyze_velocity(vel_data, objects, object_masks, sweep, config=...)` (the `config` keyword arrives in Task 6; until then call without it), `promote_persistent_rotation_assessments`.
- Produces:
  - `EVIDENCE_RANK: dict[str, int]`
  - `Point(latitude: float, longitude: float)`
  - `VolumeResult(case_id: str, label: str, volume_time: datetime, assessments: list[tuple[Point, str]], storm_centroids: list[Point], nmd: list[MesocycloneDetection])`
  - `event_hit(results: list[VolumeResult], report: StormReport, min_rank: int) -> bool`
  - `nmd_agreement(result: VolumeResult, min_rank: int) -> tuple[int, int, int]`
  - `null_storm_probes(result: VolumeResult, reports: list[StormReport], min_rank: int) -> tuple[int, int]`
  - `clear_air_false_alarms(result: VolumeResult, min_rank: int) -> int`
  - `summarize(results: list[VolumeResult], cases: list[dict], reports: list[StormReport]) -> dict[int, dict]`
  - `speakable(rank_summary: dict) -> bool`

- [ ] **Step 1: Write the failing metric tests**

```python
# tests/unit/test_rotation_metrics.py
from datetime import datetime, timedelta

from src.validation.nmd import MesocycloneDetection
from src.validation.rotation_metrics import (
    Point, VolumeResult, clear_air_false_alarms, event_hit, nmd_agreement, null_storm_probes, speakable,
)
from src.validation.spc import StormReport

T = datetime(2026, 9, 12, 19, 40)
REPORT = StormReport("tornado", T, 29.60, -81.28, 1.0, "Palm Coast", "FL")


def volume(minutes, assessments=(), centroids=(), nmd=(), label="tornado"):
    return VolumeResult("case", label, T + timedelta(minutes=minutes), list(assessments), list(centroids), list(nmd))


def test_a_hit_needs_rank_distance_and_time():
    near = (Point(29.65, -81.28), "unconfirmed")          # 5.6 km away
    assert event_hit([volume(-5, [near])], REPORT, 1)
    assert not event_hit([volume(-5, [near])], REPORT, 2)
    assert not event_hit([volume(-25, [near])], REPORT, 1)  # before the window
    assert not event_hit([volume(-5, [(Point(29.75, -81.28), "persistent")])], REPORT, 1)  # 16.7 km


def test_nmd_agreement_counts_matches_within_5_km():
    detection = MesocycloneDetection("1", 0.0, 0.0, 5, "A1", False, 29.60, -81.28)
    result = volume(0, [(Point(29.62, -81.28), "unconfirmed"), (Point(30.5, -81.0), "unconfirmed")], nmd=[detection])
    assert nmd_agreement(result, 1) == (1, 1, 0)


def test_null_storm_probes_exclude_storms_near_reports_or_detections():
    far_storm, near_storm = Point(31.0, -83.0), Point(29.7, -81.3)
    result = volume(0, [(Point(31.02, -83.0), "unconfirmed")], centroids=[far_storm, near_storm])
    assert null_storm_probes(result, [REPORT], 1) == (1, 1)
    assert null_storm_probes(result, [REPORT], 2) == (1, 0)


def test_clear_air_false_alarms_count_only_clear_air_volumes():
    assessments = [(Point(33.0, -112.0), "unconfirmed")]
    assert clear_air_false_alarms(volume(0, assessments, label="clear_air"), 1) == 1
    assert clear_air_false_alarms(volume(0, assessments, label="tornado"), 1) == 0


def test_speak_rule():
    assert speakable({"clear_air_false_alarms": 0, "tornado_hits": 3, "tornado_events": 9, "null_probes": 100, "null_false_alarms": 5})
    assert not speakable({"clear_air_false_alarms": 1, "tornado_hits": 9, "tornado_events": 9, "null_probes": 100, "null_false_alarms": 0})
    assert not speakable({"clear_air_false_alarms": 0, "tornado_hits": 1, "tornado_events": 9, "null_probes": 100, "null_false_alarms": 0})
    assert not speakable({"clear_air_false_alarms": 0, "tornado_hits": 3, "tornado_events": 9, "null_probes": 100, "null_false_alarms": 20})
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest -q -p no:cacheprovider tests/unit/test_rotation_metrics.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.validation.rotation_metrics'`.

- [ ] **Step 3: Implement the metrics**

```python
# src/validation/rotation_metrics.py
"""Pre-registered rotation validation metrics (plan 2026-09-14, Global Constraints)."""

from dataclasses import dataclass
from datetime import datetime, timedelta

from src.sites import haversine_distance_km
from src.validation.nmd import MesocycloneDetection
from src.validation.spc import StormReport

EVIDENCE_RANK = {"unconfirmed": 1, "vertically_confirmed": 2, "corroborated": 2, "persistent": 3}
HIT_RADIUS_KM = 10.0
HIT_BEFORE = timedelta(minutes=20)
HIT_AFTER = timedelta(minutes=10)
NMD_MATCH_KM = 5.0
NULL_EXCLUSION_KM = 50.0
NULL_REPORT_WINDOW = timedelta(minutes=60)
SPEAK_MIN_HITS = 2
SPEAK_MIN_LIFT = 3.0


@dataclass(frozen=True)
class Point:
    latitude: float
    longitude: float


@dataclass
class VolumeResult:
    case_id: str
    label: str
    volume_time: datetime
    assessments: list[tuple[Point, str]]
    storm_centroids: list[Point]
    nmd: list[MesocycloneDetection]


def _distance(a: Point, lat: float, lon: float) -> float:
    return haversine_distance_km(a.latitude, a.longitude, lat, lon)


def _at_rank(result: VolumeResult, min_rank: int) -> list[Point]:
    return [point for point, level in result.assessments if EVIDENCE_RANK.get(level, 0) >= min_rank]


def event_hit(results: list[VolumeResult], report: StormReport, min_rank: int) -> bool:
    for result in results:
        if not (report.time_utc - HIT_BEFORE <= result.volume_time <= report.time_utc + HIT_AFTER):
            continue
        if any(_distance(p, report.latitude, report.longitude) <= HIT_RADIUS_KM for p in _at_rank(result, min_rank)):
            return True
    return False


def nmd_agreement(result: VolumeResult, min_rank: int) -> tuple[int, int, int]:
    points = _at_rank(result, min_rank)
    matched_points = {i for i, p in enumerate(points) if any(_distance(p, d.latitude, d.longitude) <= NMD_MATCH_KM for d in result.nmd)}
    matched_detections = sum(1 for d in result.nmd if any(_distance(p, d.latitude, d.longitude) <= NMD_MATCH_KM for p in points))
    return len(matched_points), len(points) - len(matched_points), len(result.nmd) - matched_detections


def null_storm_probes(result: VolumeResult, reports: list[StormReport], min_rank: int) -> tuple[int, int]:
    if result.label == "clear_air":
        return 0, 0
    nearby_reports = [r for r in reports if abs(r.time_utc - result.volume_time) <= NULL_REPORT_WINDOW]
    points = _at_rank(result, min_rank)
    probes = hits = 0
    for centroid in result.storm_centroids:
        if any(_distance(centroid, r.latitude, r.longitude) <= NULL_EXCLUSION_KM for r in nearby_reports):
            continue
        if any(_distance(centroid, d.latitude, d.longitude) <= NULL_EXCLUSION_KM for d in result.nmd):
            continue
        probes += 1
        hits += any(_distance(centroid, p.latitude, p.longitude) <= HIT_RADIUS_KM for p in points)
    return probes, hits


def clear_air_false_alarms(result: VolumeResult, min_rank: int) -> int:
    return len(_at_rank(result, min_rank)) if result.label == "clear_air" else 0


def summarize(results: list[VolumeResult], cases: list[dict], reports: list[StormReport]) -> dict[int, dict]:
    by_case: dict[str, list[VolumeResult]] = {}
    for result in results:
        by_case.setdefault(result.case_id, []).append(result)
    summary: dict[int, dict] = {}
    for rank in (1, 2, 3):
        tornado_cases = [c for c in cases if c["label"] == "tornado" and c["id"] in by_case]
        hits = sum(event_hit(by_case[c["id"]], next(r for r in reports if r.kind == "tornado" and r.time_utc.isoformat() == c["report"]["time_utc"] and r.latitude == c["report"]["latitude"]), rank) for c in tornado_cases)
        agreement = [nmd_agreement(r, rank) for r in results]
        probes = [null_storm_probes(r, reports, rank) for r in results]
        summary[rank] = {
            "tornado_events": len(tornado_cases),
            "tornado_hits": hits,
            "nmd_matched": sum(a[0] for a in agreement),
            "arw_only": sum(a[1] for a in agreement),
            "nmd_only": sum(a[2] for a in agreement),
            "null_probes": sum(p[0] for p in probes),
            "null_false_alarms": sum(p[1] for p in probes),
            "clear_air_false_alarms": sum(clear_air_false_alarms(r, rank) for r in results),
            "clear_air_volumes": sum(1 for r in results if r.label == "clear_air"),
        }
        summary[rank]["speakable"] = speakable(summary[rank])
    return summary


def speakable(rank_summary: dict) -> bool:
    if rank_summary["clear_air_false_alarms"] != 0 or rank_summary["tornado_hits"] < SPEAK_MIN_HITS:
        return False
    hit_fraction = rank_summary["tornado_hits"] / max(rank_summary["tornado_events"], 1)
    null_fraction = rank_summary["null_false_alarms"] / max(rank_summary["null_probes"], 1)
    return hit_fraction >= SPEAK_MIN_LIFT * null_fraction
```

- [ ] **Step 4: Run the metric tests**

Run: `.venv/Scripts/python.exe -m pytest -q -p no:cacheprovider tests/unit/test_rotation_metrics.py`
Expected: 5 passed.

- [ ] **Step 5: Write the evaluator.** It analyses each volume once, scores every configuration, and tracks per case for persistence.

```python
# scripts/evaluate_rotation_corpus_v2.py
"""Evaluate rotation detector configurations on corpus v2 (plan 2026-09-14, Task 5 and 7).

For each case, volumes are processed in time order through the production
analysis path once; each detector configuration then runs velocity analysis
on the same inputs, with persistence promoted from that configuration's own
per-track rotation history, keyed by a single identity tracker.
Usage: evaluate_rotation_corpus_v2.py OUT.json [--sweep]
"""
import itertools
import json
import sys
import time
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import src.server as server
from src.buffer import BufferedScan
from src.detection import detect_objects_with_grid
from src.parser import extract_sweep_data, extract_velocity, parse_radar_file
from src.preprocess import preprocess_sweep
from src.tracker import StormTracker
from src.velocity import RotationDetectorConfig, analyze_velocity, promote_persistent_rotation_assessments
from src.validation.nmd import parse_nmd
from src.validation.rotation_metrics import Point, VolumeResult, summarize
from src.validation.spc import StormReport

CORPUS = ROOT / "docs/validation/rotation-signal-corpus-v2.json"
BASELINE = RotationDetectorConfig.baseline()


def sweep_configs():
    grid = itertools.product(
        (15.0, 20.0, 25.0),      # min_shear_ms
        (0.0, 10.0),             # min_side_ms
        (None, 10.0),            # max_diameter_km
        (False, True),           # fold_rejection
    )
    for min_shear, min_side, max_diameter, fold in grid:
        yield RotationDetectorConfig(min_shear_ms=min_shear, min_side_ms=min_side, max_diameter_km=max_diameter,
                                     fold_rejection=fold, azimuthal_pairs_only=True, ground_overlap_merge=True)


def load_volume(path):
    radar = parse_radar_file(str(ROOT / path), scans=server.LIVE_MAP_SCAN_INDICES, include_fields=server.LIVE_MAP_FIELDS)
    raw = extract_sweep_data(radar)
    vel = extract_velocity(radar, dealias=False)
    velocity = server._velocity_aligned_to_reflectivity(raw, vel)
    ref, quality, _ = preprocess_sweep(raw, [], velocity=velocity)
    detection = detect_objects_with_grid(
        reflectivity=ref.reflectivity, azimuths=ref.azimuths, ranges_m=ref.ranges_m, radar_lat=ref.radar_lat,
        radar_lon=ref.radar_lon, elevation_deg=ref.elevation_angle, elevations=ref.elevations,
        gate_classification=ref.gate_classification,
    )
    return ref, quality, vel, detection


def main(out_path, sweep):
    corpus = json.loads(CORPUS.read_text(encoding="utf-8"))
    reports = [StormReport(**{**r, "time_utc": datetime.fromisoformat(r["time_utc"])}) for r in corpus["reports"]]
    configs = [BASELINE] + (list(sweep_configs()) if sweep else [])
    results = {i: [] for i in range(len(configs))}
    timing = {i: 0.0 for i in range(len(configs))}
    for case in corpus["cases"]:
        tracker = StormTracker()
        histories = {i: {} for i in range(len(configs))}  # track id -> (timestamp, rotation)
        nmd_products = []
        for path in case["nmd"]:
            when, detections = parse_nmd((ROOT / path).read_bytes(), case["site_id"])
            if when is not None:
                nmd_products.append((when, detections))
        for path in case["volumes"]:
            ref, quality, vel, detection = load_volume(path)
            timestamp = datetime.fromisoformat(ref.timestamp.replace("Z", "+00:00")).replace(tzinfo=None)
            tracker.update(BufferedScan(timestamp=timestamp, site_id=case["site_id"], reflectivity_data=ref,
                                        detected_objects=detection.objects, labeled_grid=detection.labeled_grid,
                                        object_masks=detection.object_masks, scan_quality=quality))
            nearest = min(nmd_products, key=lambda item: abs((item[0] - timestamp).total_seconds()), default=None)
            nmd = nearest[1] if nearest is not None and abs((nearest[0] - timestamp).total_seconds()) <= 180 else []
            centroids = [Point(o.centroid_lat, o.centroid_lon) for o in detection.objects]
            for index, config in enumerate(configs):
                started = time.perf_counter()
                _regions, assessments, _objects = analyze_velocity(vel, detection.objects, detection.object_masks, ref, config=config)
                timing[index] += time.perf_counter() - started
                track_history = {
                    obj_id: [type("Entry", (), {"timestamp": stamp, "rotation": rotation})()]
                    for obj_id, track_id in tracker._obj_to_track.items()
                    if (entry := histories[index].get(track_id)) is not None
                    for stamp, rotation in [entry]
                }
                assessments = promote_persistent_rotation_assessments(assessments, track_history, timestamp)
                for assessment in assessments:
                    if assessment.associated_object_id is not None:
                        track_id = tracker._obj_to_track.get(assessment.associated_object_id)
                        if track_id is not None:
                            histories[index][track_id] = (timestamp, assessment)
                results[index].append(VolumeResult(
                    case_id=case["id"], label=case["label"], volume_time=timestamp,
                    assessments=[(Point(a.centroid_lat, a.centroid_lon), a.evidence_level) for a in assessments if a.associated_object_id is not None],
                    storm_centroids=centroids, nmd=nmd,
                ))
            print("done", case["id"], Path(path).name, flush=True)
    output = []
    for index, config in enumerate(configs):
        output.append({"config": asdict(config), "analysis_seconds": round(timing[index], 1),
                       "summary": summarize(results[index], corpus["cases"], reports),
                       "per_case": {c["id"]: summarize([r for r in results[index] if r.case_id == c["id"]], [c], reports)[1]
                                    for c in corpus["cases"]}})
    Path(out_path).write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1], "--sweep" in sys.argv)
```

Only cell-associated assessments count, because unassociated candidates never reach speech or the map.

- [ ] **Step 6: Commit the metrics and evaluator** (the evaluator runs after Task 6 provides `RotationDetectorConfig`)

```bash
git add src/validation/rotation_metrics.py tests/unit/test_rotation_metrics.py scripts/evaluate_rotation_corpus_v2.py
git commit -m "Add pre-registered rotation validation metrics and corpus evaluator"
```

---

### Task 6: `RotationDetectorConfig`, byte-identical by default

**Files:**
- Modify: `src/velocity.py` (`_detect_shear_single_sweep`, `detect_rotation_signatures`, `_merge_cross_sweep_rotations`, `analyze_velocity`)
- Test: `tests/unit/test_rotation_detector_config.py`

**Interfaces:**
- Produces:
  - `RotationDetectorConfig(min_shear_ms: float = 15.0, min_side_ms: float = 0.0, max_diameter_km: float | None = None, fold_rejection: bool = False, azimuthal_pairs_only: bool = False, ground_overlap_merge: bool = False)`, with `RotationDetectorConfig.baseline()` equal to the defaults;
  - `FOLD_DIFFERENCE_FRACTION = 0.8`, `FOLD_SIDE_FRACTION = 0.5`, `GROUND_OVERLAP_MARGIN_KM = 1.0`;
  - `detect_rotation_signatures(vel_data, config: RotationDetectorConfig = RotationDetectorConfig())`;
  - `analyze_velocity(..., config: RotationDetectorConfig = RotationDetectorConfig())`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_rotation_detector_config.py
import numpy as np
import pytest

from src.velocity import RotationDetectorConfig, detect_rotation_signatures
from tests.unit.test_velocity import _make_sweep, _make_velocity_data


def _radial_convergence():
    """Opposite signs along the beam (same ray): convergence, not rotation."""
    grid = np.full((360, 500), np.nan)
    grid[50:60, 150:155] = 20.0
    grid[50:60, 155:160] = -20.0
    return grid


def _azimuthal_couplet():
    grid = np.full((360, 500), np.nan)
    grid[50:55, 150:160] = -20.0
    grid[55:60, 150:160] = 20.0
    return grid


def test_default_config_still_counts_radial_pairs_for_baseline_measurement():
    assert detect_rotation_signatures(_make_velocity_data([_make_sweep(_radial_convergence())]))


def test_azimuthal_pairs_only_rejects_convergence_and_keeps_rotation():
    config = RotationDetectorConfig(azimuthal_pairs_only=True)
    assert detect_rotation_signatures(_make_velocity_data([_make_sweep(_radial_convergence())]), config) == []
    assert detect_rotation_signatures(_make_velocity_data([_make_sweep(_azimuthal_couplet())]), config)


def test_min_side_requires_both_inbound_and_outbound_strength():
    grid = np.full((360, 500), np.nan)
    grid[50:55, 150:160] = -4.0
    grid[55:60, 150:160] = 30.0
    velocity = _make_velocity_data([_make_sweep(grid)])
    assert detect_rotation_signatures(velocity, RotationDetectorConfig(azimuthal_pairs_only=True))
    assert detect_rotation_signatures(velocity, RotationDetectorConfig(azimuthal_pairs_only=True, min_side_ms=10.0)) == []


def test_fold_rejection_drops_a_jump_near_twice_nyquist():
    grid = np.full((360, 500), np.nan)
    grid[50:55, 150:160] = 24.0     # Nyquist 26.2 in _make_sweep: +24 next to -24 is a fold
    grid[55:60, 150:160] = -24.0
    velocity = _make_velocity_data([_make_sweep(grid)])
    assert detect_rotation_signatures(velocity, RotationDetectorConfig(azimuthal_pairs_only=True))
    assert detect_rotation_signatures(velocity, RotationDetectorConfig(azimuthal_pairs_only=True, fold_rejection=True)) == []


def test_ground_overlap_merge_needs_candidates_to_overlap_not_just_lie_within_10_km():
    low = _azimuthal_couplet()
    high = np.full((360, 500), np.nan)
    high[60:65, 150:160] = -20.0     # about 8 km away at ~75 km range, 5 degrees of azimuth
    high[65:70, 150:160] = 20.0
    velocity = _make_velocity_data([_make_sweep(low, elevation=0.48), _make_sweep(high, elevation=0.88)])
    merged_by_distance = detect_rotation_signatures(velocity, RotationDetectorConfig(azimuthal_pairs_only=True))
    merged_by_overlap = detect_rotation_signatures(velocity, RotationDetectorConfig(azimuthal_pairs_only=True, ground_overlap_merge=True))
    assert max(s.sweep_count for s in merged_by_distance) == 2
    assert max(s.sweep_count for s in merged_by_overlap) == 1


def test_max_diameter_drops_candidates_larger_than_a_mesocyclone():
    grid = np.full((360, 500), np.nan)
    grid[40:55, 150:160] = -20.0
    grid[55:70, 150:160] = 20.0      # 30 degrees tall at ~75 km: tens of km across
    velocity = _make_velocity_data([_make_sweep(grid)])
    assert detect_rotation_signatures(velocity, RotationDetectorConfig(azimuthal_pairs_only=True))
    assert detect_rotation_signatures(velocity, RotationDetectorConfig(azimuthal_pairs_only=True, max_diameter_km=10.0)) == []
```

Before running, check the distance of the `high` couplet from the `low` one against `_make_sweep`'s ranges (`np.linspace(2000, 230000, 500)`, gate 155 is about 72 km). The rows 5 degrees apart must be at least 5 km and at most 10 km apart, with both diameters under 2 km. If not, shift `high`'s rows until they are, and write the measured distance in a comment.

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest -q -p no:cacheprovider tests/unit/test_rotation_detector_config.py`
Expected: FAIL with `ImportError: cannot import name 'RotationDetectorConfig'`.

- [ ] **Step 3: Implement.** In `src/velocity.py`, add after the constants:

```python
FOLD_DIFFERENCE_FRACTION = 0.8   # of twice the Nyquist velocity
FOLD_SIDE_FRACTION = 0.5         # of the Nyquist velocity, on both sides
GROUND_OVERLAP_MARGIN_KM = 1.0


@dataclass(frozen=True)
class RotationDetectorConfig:
    """Rotation candidate rules.  Defaults reproduce the pre-2026-09-14 detector
    exactly; values are chosen by the corpus sweep (plan 2026-09-14, Task 7)."""

    min_shear_ms: float = MIN_SHEAR_MS
    min_side_ms: float = 0.0
    max_diameter_km: float | None = None
    fold_rejection: bool = False
    azimuthal_pairs_only: bool = False
    ground_overlap_merge: bool = False

    @classmethod
    def baseline(cls) -> "RotationDetectorConfig":
        return cls()
```

Give `_detect_shear_single_sweep` the parameters `nyquist_velocity: float | None` and `config: RotationDetectorConfig`.
- In the neighbour loop, change `for delta_az in [-1, 0, 1]` so that, when `config.azimuthal_pairs_only`, `delta_az == 0` pairs are skipped (`if config.azimuthal_pairs_only and delta_az == 0: continue` after the existing `(0, 0)` skip).
- Replace `strong_shear = ~np.isnan(shear) & (shear >= MIN_SHEAR_MS)` with:

```python
            strong_shear = ~np.isnan(shear) & (shear >= config.min_shear_ms)
            if config.min_side_ms > 0.0:
                strong_shear &= (np.minimum(v1, v2) <= -config.min_side_ms) & (np.maximum(v1, v2) >= config.min_side_ms)
            if config.fold_rejection and nyquist_velocity:
                fold = (np.abs(v1 - v2) >= FOLD_DIFFERENCE_FRACTION * 2.0 * nyquist_velocity) & (
                    np.minimum(np.abs(v1), np.abs(v2)) >= FOLD_SIDE_FRACTION * nyquist_velocity
                )
                strong_shear &= ~fold
```

- After `diameter_km` is computed, add `if config.max_diameter_km is not None and diameter_km > config.max_diameter_km: continue`.
- Give `_merge_cross_sweep_rotations` a `config` parameter. Replace `if distance_km <= ROTATION_MERGE_DISTANCE_KM:` with:

```python
                limit_km = (
                    (existing_sig.diameter_km + sig.diameter_km) / 2.0 + GROUND_OVERLAP_MARGIN_KM
                    if config.ground_overlap_merge
                    else ROTATION_MERGE_DISTANCE_KM
                )
                if distance_km <= limit_km:
```

- `detect_rotation_signatures(vel_data, config=RotationDetectorConfig())` passes `nyquist_velocity=sweep.nyquist_velocity, config=config` to each sweep call, and `config` to the merge.
- `analyze_velocity(..., config=RotationDetectorConfig())` passes it to `detect_rotation_signatures`.

- [ ] **Step 4: Run the new tests and the existing velocity tests**

Run: `.venv/Scripts/python.exe -m pytest -q -p no:cacheprovider tests/unit/test_rotation_detector_config.py tests/unit/test_velocity.py`
Expected: all pass.

- [ ] **Step 5: Prove the defaults are byte-identical on real volumes**

Run a script that loads the pre-change `src/velocity.py` from `git show HEAD:src/velocity.py` as a separate module (register it in `sys.modules`). Run `detect_rotation_signatures` from both modules on `cache/KTLX/KTLX20130520_195527_V06.gz`, `cache/KIWA/KIWA20260712_163410_V06` and `cache/KTLX/KTLX20260410_224737_V06` (velocity from `extract_velocity(parse_radar_file(path))`). Assert the `dataclasses.astuple` lists are equal.
Expected: equal for all three; print the candidate counts.

- [ ] **Step 6: Run the evaluator baseline** (sweep off)

Run: `.venv/Scripts/python.exe scripts/evaluate_rotation_corpus_v2.py docs/test_reports/2026-09-14-rotation-baseline.json`
Expected: finishes, with `done` lines for every volume. Add a "Baseline" section to `docs/test_reports/2026-09-14-rotation-corpus.md`:
- for ranks 1-3: tornado hits out of events, NMD matched / ARW-only / NMD-only, null probes and false alarms, clear-air false alarms, speakable;
- the per-case table;
- analysis seconds.

- [ ] **Step 7: Commit**

```bash
git add src/velocity.py tests/unit/test_rotation_detector_config.py docs/test_reports/2026-09-14-rotation-baseline.json docs/test_reports/2026-09-14-rotation-corpus.md
git commit -m "Make rotation detector rules configurable; measure the baseline on corpus v2"
```

---

### Task 7: Sweep, select, adopt

**Files:**
- Modify: `src/velocity.py` (defaults), `tests/unit/test_rotation_detector_config.py`, `tests/unit/test_velocity.py` (only where a test encoded radial pairs or 10 km merging as correct; each change is explained in a comment)
- Generated: `docs/test_reports/2026-09-14-rotation-sweep.json`; a "Sweep and selection" section in the corpus report

**Interfaces:**
- Consumes: the evaluator with `--sweep`.
- Produces: `RotationDetectorConfig` defaults equal to the selected configuration; `RotationDetectorConfig.baseline()` still returns the pre-change rules (explicit values, no longer the defaults).

- [ ] **Step 1: Run the sweep**

Run: `.venv/Scripts/python.exe scripts/evaluate_rotation_corpus_v2.py docs/test_reports/2026-09-14-rotation-sweep.json --sweep`
Expected: `summary` for the baseline plus 24 configurations.

- [ ] **Step 2: Select by the pre-registered rule.** Write a short script that reads the JSON and applies the selection rule from Global Constraints exactly. It prints the chosen configuration and a table of every configuration: tornado hits (rank 1), null false-alarm fraction, NMD matched, clear-air false alarms, analysis seconds. Record the table and the choice in the report. If no configuration meets the baseline's null false-alarm fraction, the selection is the baseline with `azimuthal_pairs_only=True` and `ground_overlap_merge=True` (the two physics corrections), and the report says so.

- [ ] **Step 3: Change the defaults**

Edit `RotationDetectorConfig` so its field defaults are the selected values. Make `baseline()` return `cls(min_shear_ms=15.0, min_side_ms=0.0, max_diameter_km=None, fold_rejection=False, azimuthal_pairs_only=False, ground_overlap_merge=False)`. Update the class docstring with the report path and the selected values.

- [ ] **Step 4: Update tests that encoded the old rules.** Change `test_default_config_still_counts_radial_pairs_for_baseline_measurement` to use `RotationDetectorConfig.baseline()`. Run `tests/unit/test_velocity.py`. Any test failing because it relied on radial pairs or 10 km merging is corrected to the physical rule, with a comment naming this plan. A failure for any other reason is investigated, not edited.

Run: `.venv/Scripts/python.exe -m pytest -q -p no:cacheprovider tests/unit/test_rotation_detector_config.py tests/unit/test_velocity.py`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/velocity.py tests/unit/test_rotation_detector_config.py tests/unit/test_velocity.py docs/test_reports/2026-09-14-rotation-sweep.json docs/test_reports/2026-09-14-rotation-corpus.md
git commit -m "Adopt the rotation detector configuration selected on corpus v2"
```

---

### Task 8: Speak only validated evidence levels

**Files:**
- Modify: `src/summary.py`
- Test: `tests/unit/test_summary.py` (append)

**Interfaces:**
- Consumes: the selected configuration's `summary[rank]["speakable"]` from Task 7.
- Produces: `SPOKEN_ROTATION_EVIDENCE: frozenset[str]` in `src/summary.py`, holding the evidence levels whose rank was speakable. `_format_rotation` returns `""` for any other level. The standalone-rotation sentences obey the same set.

- [ ] **Step 1: Write the failing test.** Use the speakable ranks from Task 7; the example assumes rank 1 was not speakable and rank 2 was.

```python
def test_rotation_evidence_below_the_validated_level_is_not_spoken():
    from src.summary import SPOKEN_ROTATION_EVIDENCE
    obj = _make_object()
    for level, spoken in (("unconfirmed", "unconfirmed" in SPOKEN_ROTATION_EVIDENCE), ("vertically_confirmed", "vertically_confirmed" in SPOKEN_ROTATION_EVIDENCE)):
        rotation = RotationSignature(35.5, -97.5, 40.0, 270.0, 30.0, -15.0, 15.0, 2.0, 2, [0.5, 0.9], "moderate", associated_object_id=obj.object_id, evidence_level=level)
        text = generate_summary("KTLX", "Oklahoma City", "2026-04-08T18:30:00Z", [replace(obj, rotation=rotation)], None)
        assert (("rotation" in text) or ("couplet" in text)) == spoken, (level, text)
```

(Add `from dataclasses import replace` and `from src.velocity import RotationSignature` to the imports if absent.)

- [ ] **Step 2: Run it to verify it fails** — `ImportError: cannot import name 'SPOKEN_ROTATION_EVIDENCE'`.

- [ ] **Step 3: Implement.** Add near the top of `src/summary.py`:

```python
# Evidence levels that beat null cases on the rotation corpus
# (docs/test_reports/2026-09-14-rotation-corpus.md, speak rule).
SPOKEN_ROTATION_EVIDENCE: frozenset[str] = frozenset({...})  # fill with the levels whose rank was speakable in Task 7
```

Then in `_format_rotation`, after the `rotation is None` branch:

```python
    if rotation.evidence_level not in SPOKEN_ROTATION_EVIDENCE:
        return ""
```

and filter `standalone_rotations` with `obj.rotation.evidence_level in SPOKEN_ROTATION_EVIDENCE`.

If no rank was speakable, the set is empty. The report states that no rotation evidence is spoken until the corpus supports it.

- [ ] **Step 4: Run summary tests**

Run: `.venv/Scripts/python.exe -m pytest -q -p no:cacheprovider tests/unit/test_summary.py tests/unit/test_live_replay_contracts.py`
Expected: all pass. Tests that asserted speech of an unvalidated level are updated, with the reason written in the test.

- [ ] **Step 5: Commit**

```bash
git add src/summary.py tests/unit/test_summary.py tests/unit/test_live_replay_contracts.py
git commit -m "Speak rotation evidence only at levels validated on the corpus"
```

---

### Task 9: Benchmark and replay scripts use the assessed path

**Files:**
- Modify: `scripts/evaluate_tracking.py:250-275`, `scripts/live_replay.py:280-300`

**Interfaces:**
- Consumes: `analyze_velocity(vel_data, objects, object_masks, sweep)`.

- [ ] **Step 1:** In both scripts, call `analyze_velocity(vel_data, detection.objects, detection.object_masks, reflectivity)` (the preprocessed sweep variable in each script) instead of `analyze_velocity(vel_data, detection.objects)`.

- [ ] **Step 2:** Run `.venv/Scripts/python.exe -m pytest -q -p no:cacheprovider tests/unit/test_tracking_evaluation.py tests/unit/test_live_replay_contracts.py`. Expected: pass.

- [ ] **Step 3:** Commit.

```bash
git add scripts/evaluate_tracking.py scripts/live_replay.py
git commit -m "Benchmark and replay rotation through the production assessment path"
```

---

### Task 10: Live pipeline, proofs, verification

**Files:**
- Modify: `src/server.py:310-340` (`_process_scan_file`)
- Create: `tests/e2e/test_proof_rotation_corpus.py`
- Modify: `tests/e2e/test_proof_clutter_persistence.py` and `tests/e2e/test_proof_known_severe_cases.py`, only where their rotation-count expectations change, with the new measured numbers
- Docs: corpus report, `PROGRESS.md`

**Interfaces:**
- Consumes: `analyze_velocity`, `promote_persistent_rotation_assessments`, the tracker's `rotation_history_for_current_object`.

- [ ] **Step 1: Measure cost first.** Time `analyze_velocity(vel_data, objects, masks, ref_data)` with the selected configuration on the densest cached volumes (`cache/KJAX/KJAX20260912_172959_V06`, `cache/KEMX/KEMX20260712_022646_V06`, `cache/KTLX/KTLX20260410_224737_V06`). Record the seconds in the report. If any volume exceeds 5 s, profile it (cProfile) and reduce the cost before wiring. Candidate extraction limited to the union of object footprints (dilated by the maximum couplet distance) is the planned reduction. It needs a byte-identical assessment proof against the unrestricted run on the three volumes.

- [ ] **Step 2: Wire it.** In `_process_scan_file`, replace `regions: list = []`, `rotations: list = []` and `annotated_objects = result.objects` with:

```python
    regions, rotations, annotated_objects = analyze_velocity(
        vel_data, result.objects, result.object_masks, ref_data,
    )
```

Update the comment above to state the measured cost and that only validated evidence is spoken (`SPOKEN_ROTATION_EVIDENCE`). Persistence promotion needs the tracker. In `_track_live_scan` (the live tracking path), after the tracker update, apply `promote_persistent_rotation_assessments` using `tracker.rotation_history_for_current_object`, as `scripts/evaluate_rotation_signal_corpus.py` does, and store the result on the buffered scan and its objects.

- [ ] **Step 3: Real-data proof.** Create `tests/e2e/test_proof_rotation_corpus.py`. It loads `docs/test_reports/2026-09-14-rotation-sweep.json`, finds the selected configuration, and asserts its speakable ranks equal `SPOKEN_ROTATION_EVIDENCE`. It then runs the production analysis on the six KIWA clear-air volumes and asserts no spoken-level assessment. A test that recomputes the full corpus is too slow for the suite, so the suite checks the recorded result against the code's constant and the clear-air invariant directly.

- [ ] **Step 4: Full suite.** Run `.venv/Scripts/python.exe -m pytest -q -p no:cacheprovider`. Expected: all pass. Any strict xfail that now passes is removed together with its documented defect, and the report says which.

- [ ] **Step 5: Tracking benchmark.** Run `scripts/evaluate_tracking.py --manifest docs/benchmarks/tracking_benchmark_manifest_local.json --output-json docs/test_reports/2026-09-14-benchmark-after-rotation.json` and explain every difference against `2026-09-13-benchmark-after-measured-motion.json` in the report.

- [ ] **Step 6: Live check.** Start the server on port 8011 with a scratch `ARW_HISTORY_ROOT`. Request `/summary` and `/velocity` for two radars, one with active storms and one quiet, across two scans. Record rotation assessments, spoken text, and ingest time against the 2026-09-13 timings. Stop the server.

- [ ] **Step 7: Docs and commit.** Add results to the corpus report and a "Rotation detection (2026-09-14)" section to `PROGRESS.md`, then comment on issue #9 with the outcome (close it only if the speak rule and clear-air invariant both hold).

```bash
git add src/server.py tests/e2e docs PROGRESS.md
git commit -m "Run validated rotation analysis in the live pipeline"
```

---

## Self-review notes

- **Spec coverage:**
  - A1 -> Tasks 1-4; A2 -> Task 5 and Task 6 Step 6; A3 -> Tasks 6-7; A4 -> Tasks 8-10.
  - Script alignment (A3) -> Task 9.
  - Part B -> separate plan after Task 10, as the spec orders.
- **Measured values:** thresholds are selected in Task 7 by the rule in Global Constraints. `SPOKEN_ROTATION_EVIDENCE` in Task 8 is filled from Task 7's recorded `speakable` flags.
- **Names:** `RotationDetectorConfig` fields are used identically in Tasks 5, 6 and 7. `analyze_velocity(..., config=...)` is defined in Task 6 and consumed in Tasks 5 and 10.
