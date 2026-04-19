# Phase 4: Hail Detection and Debris Scoring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add hail detection (dual-pol + MESH diameter estimation) and tornado debris signature scoring to the ARW pipeline.

**Architecture:** A single `src/hazards.py` module handles all hail/debris analysis. Dual-pol extraction is added to `src/parser.py`. Freezing level fetch is added to `src/ingest.py`. The pipeline flows: `analyze_velocity()` → `extract_dual_pol()` → `fetch_freezing_level()` → `analyze_hazards()` → `BufferedScan`.

**Tech Stack:** Python, NumPy, SciPy (CCL), Py-ART, xarray/cfgrib (GRIB2 parsing for RAP data), requests (NOMADS HTTP fetch)

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `src/parser.py` | Modify | Add `DualPolData` dataclass and `extract_dual_pol(radar)` function |
| `src/ingest.py` | Modify | Add `FreezingLevel` dataclass and `fetch_freezing_level(lat, lon)` function |
| `src/hazards.py` | Create | `HailAssessment`, `DebrisSignature`, `HazardAnalysis` dataclasses; `analyze_hazards()` entry point; hail discrimination, MESH calculation, TDS detection |
| `src/buffer.py` | Modify | Add `dual_pol_data`, `hail_assessments`, `debris_signatures` fields to `BufferedScan` |
| `src/tracking/types.py` | Modify | Add `HailHistoryEntry` dataclass and `hail_history` field to `Track` |
| `src/tracking/__init__.py` | Modify | Export `HailHistoryEntry` |
| `src/tracker.py` | Modify | Record hail history at all `add_position` call sites (same pattern as rotation_history) |
| `src/models.py` | Modify | Add `HailAssessmentModel`, `DebrisSignatureModel`, `HazardResponse`, `HailHistoryEntryModel`; extend `RainObject` and `StormTrack` |
| `src/server.py` | Modify | Wire `extract_dual_pol`, `fetch_freezing_level`, `analyze_hazards` into pipeline; add `/hazards/{site_id}` endpoint; extend `/objects` with hail fields; extend `_track_to_model` with hail_history |
| `src/summary.py` | Modify | Add hail per-object annotation, debris standalone alerts, priority reordering |
| `scripts/live_replay.py` | Modify | Add hazard counts to diagnostic output |
| `tests/unit/test_dual_pol_parser.py` | Create | Tests for dual-pol extraction |
| `tests/unit/test_hazards.py` | Create | Tests for hail discrimination, MESH, TDS detection |
| `tests/unit/test_summary.py` | Modify | Tests for hail and debris speech language |
| `tests/smoke/test_server_smoke.py` | Modify | Update patches, add `/hazards` smoke test |
| `tests/e2e/test_full_pipeline.py` | Modify | Update patches for dual-pol + hazards pipeline |

---

### Task 1: Dual-Pol Extraction — Tests

**Files:**
- Create: `tests/unit/test_dual_pol_parser.py`

- [ ] **Step 1: Write the dual-pol extraction test file**

```python
# tests/unit/test_dual_pol_parser.py
import numpy as np
from unittest.mock import MagicMock
from src.parser import extract_dual_pol, DualPolData


def _make_mock_radar_with_dual_pol():
    """Create a mock Py-ART radar with dual-pol fields on sweep 0."""
    radar = MagicMock()
    radar.nsweeps = 2
    radar.fixed_angle = {"data": np.array([0.5, 1.5])}
    radar.latitude = {"data": np.array([35.3331])}
    radar.longitude = {"data": np.array([-97.2778])}

    def get_start_end(sweep_index):
        start = sweep_index * 360
        end = start + 359
        return (start, end)

    radar.get_start_end.side_effect = get_start_end

    zdr_data = np.random.uniform(-2, 6, (720, 1832)).astype(np.float32)
    cc_data = np.random.uniform(0.8, 1.0, (720, 1832)).astype(np.float32)
    ref_data = np.random.uniform(0, 60, (720, 1832)).astype(np.float32)

    radar.fields = {
        "reflectivity": {"data": np.ma.array(ref_data, mask=False)},
        "differential_reflectivity": {"data": np.ma.array(zdr_data, mask=False)},
        "cross_correlation_ratio": {"data": np.ma.array(cc_data, mask=False)},
    }
    radar.azimuth = {"data": np.tile(np.linspace(0, 359, 360), 2)}
    radar.range = {"data": np.linspace(0, 459750, 1832)}
    radar.time = {"units": "seconds since 2026-04-10T17:21:34Z"}
    return radar


def test_extract_dual_pol_returns_data():
    radar = _make_mock_radar_with_dual_pol()
    result = extract_dual_pol(radar)
    assert isinstance(result, DualPolData)
    assert result.zdr.shape == (360, 1832)
    assert result.cc.shape == (360, 1832)
    assert result.radar_lat == 35.3331
    assert result.radar_lon == -97.2778
    assert result.elevation_angle == 0.5


def test_extract_dual_pol_returns_none_missing_zdr():
    radar = _make_mock_radar_with_dual_pol()
    del radar.fields["differential_reflectivity"]
    result = extract_dual_pol(radar)
    assert result is None


def test_extract_dual_pol_returns_none_missing_cc():
    radar = _make_mock_radar_with_dual_pol()
    del radar.fields["cross_correlation_ratio"]
    result = extract_dual_pol(radar)
    assert result is None


def test_extract_dual_pol_fills_masked_with_nan():
    radar = _make_mock_radar_with_dual_pol()
    zdr = np.ma.array(np.ones((720, 1832)), mask=np.zeros((720, 1832), dtype=bool))
    zdr.mask[0, 0] = True
    radar.fields["differential_reflectivity"] = {"data": zdr}
    cc = np.ma.array(np.ones((720, 1832)) * 0.95, mask=np.zeros((720, 1832), dtype=bool))
    cc.mask[0, 0] = True
    radar.fields["cross_correlation_ratio"] = {"data": cc}
    result = extract_dual_pol(radar)
    assert np.isnan(result.zdr[0, 0])
    assert np.isnan(result.cc[0, 0])
    assert not np.isnan(result.zdr[0, 1])
    assert not np.isnan(result.cc[0, 1])


def test_extract_dual_pol_uses_sweep_zero():
    radar = _make_mock_radar_with_dual_pol()
    result = extract_dual_pol(radar)
    assert result.elevation_angle == 0.5
    assert len(result.azimuths) == 360
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_dual_pol_parser.py -v`
Expected: FAIL — `ImportError: cannot import name 'extract_dual_pol' from 'src.parser'`

- [ ] **Step 3: Commit test file**

```bash
git add tests/unit/test_dual_pol_parser.py
git commit -m "test: add dual-pol extraction tests"
```

---

### Task 2: Dual-Pol Extraction — Implementation

**Files:**
- Modify: `src/parser.py`

- [ ] **Step 1: Add DualPolData dataclass and extract_dual_pol function to src/parser.py**

Add after the existing `VelocityData` dataclass (after line 63):

```python
@dataclass
class DualPolData:
    """Dual-polarization data from the lowest sweep."""
    zdr: np.ndarray
    cc: np.ndarray
    azimuths: np.ndarray
    ranges_m: np.ndarray
    radar_lat: float
    radar_lon: float
    elevation_angle: float
    timestamp: str


def extract_dual_pol(radar) -> DualPolData | None:
    """Extract ZDR and CC from sweep 0 of a pyart Radar object.

    Returns None if either dual-pol field is missing.
    """
    if "differential_reflectivity" not in radar.fields:
        return None
    if "cross_correlation_ratio" not in radar.fields:
        return None

    sweep_start, sweep_end = radar.get_start_end(0)
    zdr = radar.fields["differential_reflectivity"]["data"][sweep_start:sweep_end + 1]
    cc = radar.fields["cross_correlation_ratio"]["data"][sweep_start:sweep_end + 1]
    azimuths = radar.azimuth["data"][sweep_start:sweep_end + 1]
    ranges_m = radar.range["data"]

    if hasattr(zdr, "filled"):
        zdr = zdr.filled(np.nan)
    if hasattr(cc, "filled"):
        cc = cc.filled(np.nan)

    return DualPolData(
        zdr=zdr,
        cc=cc,
        azimuths=azimuths,
        ranges_m=ranges_m,
        radar_lat=float(radar.latitude["data"][0]),
        radar_lon=float(radar.longitude["data"][0]),
        elevation_angle=float(radar.fixed_angle["data"][0]),
        timestamp=str(radar.time["units"]).replace("seconds since ", ""),
    )
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_dual_pol_parser.py -v`
Expected: 5 PASS

- [ ] **Step 3: Run full test suite to check for regressions**

Run: `uv run pytest tests/ -q`
Expected: All 195+ tests pass

- [ ] **Step 4: Commit**

```bash
git add src/parser.py
git commit -m "feat: add dual-pol extraction from sweep 0"
```

---

### Task 3: Freezing Level Ingest — Tests and Implementation

**Files:**
- Modify: `src/ingest.py`
- Create: `tests/unit/test_freezing_level.py`

- [ ] **Step 1: Write freezing level test file**

```python
# tests/unit/test_freezing_level.py
from unittest.mock import patch, MagicMock
from src.ingest import FreezingLevel, fetch_freezing_level


def test_fetch_freezing_level_returns_freezing_level():
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.content = b"fake grib data"
    mock_response.raise_for_status = MagicMock()

    mock_ds = MagicMock()
    mock_ds["gh"].sel.return_value = MagicMock(values=MagicMock(item=MagicMock(return_value=3500.0)))
    mock_ds.coords = {"time": MagicMock(values=MagicMock(__str__=lambda _: "2026-04-10T18:00:00"))}
    mock_ds["time"].values.item.return_value.isoformat.return_value = "2026-04-10T18:00:00"
    mock_ds.close = MagicMock()

    with patch("src.ingest.requests.get", return_value=mock_response), \
         patch("src.ingest.xr.open_dataset", return_value=mock_ds):
        result = fetch_freezing_level(35.3331, -97.2778)
    assert isinstance(result, FreezingLevel)
    assert result.height_m_asl == 3500.0
    assert result.source == "RAP"


def test_fetch_freezing_level_returns_none_on_network_error():
    with patch("src.ingest.requests.get", side_effect=Exception("network error")):
        result = fetch_freezing_level(35.3331, -97.2778)
    assert result is None


def test_fetch_freezing_level_returns_none_on_parse_error():
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.content = b"bad data"
    mock_response.raise_for_status = MagicMock()

    with patch("src.ingest.requests.get", return_value=mock_response), \
         patch("src.ingest.xr.open_dataset", side_effect=Exception("parse error")):
        result = fetch_freezing_level(35.3331, -97.2778)
    assert result is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_freezing_level.py -v`
Expected: FAIL — `ImportError: cannot import name 'FreezingLevel' from 'src.ingest'`

- [ ] **Step 3: Add FreezingLevel and fetch_freezing_level to src/ingest.py**

Add imports at the top of `src/ingest.py`:

```python
import requests
import xarray as xr
import tempfile
```

Add after the existing `fetch_scan` function at the end of the file:

```python
@dataclass
class FreezingLevel:
    """0°C isotherm height from RAP model."""
    height_m_asl: float
    timestamp: str
    source: str


def fetch_freezing_level(lat: float, lon: float) -> FreezingLevel | None:
    """Fetch the 0°C isotherm height from the latest RAP analysis via NOMADS.

    Returns None if the data is unavailable.
    """
    try:
        rap_url = (
            "https://nomads.ncep.noaa.gov/cgi-bin/filter_rap.pl"
            "?dir=%2Frap.{date}%2F{hour}"
            "&file=rap.t{hour}z.awip32f00.grib2"
            "&lev_0C_isotherm=on"
            "&var_HGT=on"
            "&subregion="
            "&toplat={top}&leftlon={left}&rightlon={right}&bottomlat={bottom}"
        )
        now = datetime.utcnow()
        analysis_hour = (now.hour // 1) * 1
        for hour_offset in range(0, 4):
            try_hour = (analysis_hour - hour_offset) % 24
            date_str = now.strftime("%Y%m%d")
            url = rap_url.format(
                date=date_str,
                hour=f"{try_hour:02d}",
                top=lat + 0.5,
                left=lon - 0.5,
                right=lon + 0.5,
                bottom=lat - 0.5,
            )
            response = requests.get(url, timeout=15)
            if response.status_code == 200 and len(response.content) > 100:
                break
        else:
            return None

        response.raise_for_status()

        with tempfile.NamedTemporaryFile(suffix=".grib2", delete=False) as tmp:
            tmp.write(response.content)
            tmp_path = tmp.name

        ds = xr.open_dataset(tmp_path, engine="cfgrib")
        height = float(ds["gh"].sel(
            latitude=lat, longitude=(lon % 360), method="nearest"
        ).values.item())
        try:
            ts = ds["time"].values.item().isoformat()
        except Exception:
            ts = date_str
        ds.close()
        os.unlink(tmp_path)

        return FreezingLevel(
            height_m_asl=height,
            timestamp=ts,
            source="RAP",
        )
    except Exception:
        return None
```

Also add the `dataclass` import at the top — change:

```python
import os
from datetime import datetime
from pathlib import Path
import nexradaws
```

to:

```python
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import nexradaws
import requests
import xarray as xr
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_freezing_level.py -v`
Expected: 3 PASS

- [ ] **Step 5: Run full test suite**

Run: `uv run pytest tests/ -q`
Expected: All tests pass

- [ ] **Step 6: Commit**

```bash
git add src/ingest.py tests/unit/test_freezing_level.py
git commit -m "feat: add freezing level fetch from NOAA RAP"
```

---

### Task 4: Hail Detection — Tests

**Files:**
- Create: `tests/unit/test_hazards.py`

- [ ] **Step 1: Write hail detection tests**

```python
# tests/unit/test_hazards.py
import numpy as np
from src.detection import DetectedObject
from src.parser import DualPolData, ReflectivityData
from src.hazards import (
    HailAssessment, DebrisSignature, HazardAnalysis,
    analyze_hazards, _assess_hail_dual_pol, _compute_mesh, _mesh_label,
)
from src.ingest import FreezingLevel

RADAR_LAT = 35.3331
RADAR_LON = -97.2778


def _make_dual_pol(zdr_grid: np.ndarray, cc_grid: np.ndarray) -> DualPolData:
    n_az, n_rng = zdr_grid.shape
    return DualPolData(
        zdr=zdr_grid,
        cc=cc_grid,
        azimuths=np.linspace(0, 359, n_az),
        ranges_m=np.linspace(2000, 230000, n_rng),
        radar_lat=RADAR_LAT,
        radar_lon=RADAR_LON,
        elevation_angle=0.5,
        timestamp="2026-04-10T18:00:00Z",
    )


def _make_ref_data(grid: np.ndarray) -> ReflectivityData:
    n_az, n_rng = grid.shape
    return ReflectivityData(
        reflectivity=grid,
        azimuths=np.linspace(0, 359, n_az),
        ranges_m=np.linspace(2000, 230000, n_rng),
        radar_lat=RADAR_LAT,
        radar_lon=RADAR_LON,
        elevation_angle=0.5,
        elevation_angles=[0.5],
        timestamp="2026-04-10T18:00:00Z",
    )


def _make_object(object_id: int, peak_dbz: float, az_center: int = 50, rng_center: int = 100) -> DetectedObject:
    from src.detection import polar_to_latlon
    azimuths = np.linspace(0, 359, 360)
    ranges_m = np.linspace(2000, 230000, 500)
    centroid_az = float(azimuths[az_center])
    centroid_range = float(ranges_m[rng_center])
    lat, lon = polar_to_latlon(RADAR_LAT, RADAR_LON, centroid_az, centroid_range)
    return DetectedObject(
        object_id=object_id,
        centroid_lat=round(lat, 4),
        centroid_lon=round(lon, 4),
        distance_km=round(centroid_range / 1000, 1),
        bearing_deg=round(centroid_az, 1),
        peak_dbz=peak_dbz,
        peak_label="intense rain" if peak_dbz >= 50 else "heavy rain",
        area_km2=50.0,
    )


# --- Hail dual-pol discrimination ---

def test_hail_likely_all_three_criteria():
    obj = _make_object(1, peak_dbz=55.0)
    assessment = _assess_hail_dual_pol(obj, zdr_min=0.5, cc_min=0.88)
    assert assessment.dual_pol_flag == "hail_likely"


def test_hail_possible_high_dbz_low_zdr_only():
    obj = _make_object(1, peak_dbz=55.0)
    assessment = _assess_hail_dual_pol(obj, zdr_min=0.5, cc_min=0.97)
    assert assessment.dual_pol_flag == "hail_possible"


def test_hail_possible_high_dbz_reduced_cc_only():
    obj = _make_object(1, peak_dbz=55.0)
    assessment = _assess_hail_dual_pol(obj, zdr_min=2.5, cc_min=0.88)
    assert assessment.dual_pol_flag == "hail_possible"


def test_no_hail_below_reflectivity_threshold():
    obj = _make_object(1, peak_dbz=40.0)
    assessment = _assess_hail_dual_pol(obj, zdr_min=0.5, cc_min=0.88)
    assert assessment.dual_pol_flag is None


def test_no_hail_normal_pol_values():
    obj = _make_object(1, peak_dbz=55.0)
    assessment = _assess_hail_dual_pol(obj, zdr_min=3.0, cc_min=0.99)
    assert assessment.dual_pol_flag is None


# --- MESH ---

def test_mesh_returns_diameter_when_freezing_level_available():
    fzl = FreezingLevel(height_m_asl=3000.0, timestamp="2026-04-10T18:00:00Z", source="RAP")
    ref_grid = np.full((360, 500), 30.0)
    ref_grid[45:55, 95:105] = 60.0
    ref_data = _make_ref_data(ref_grid)
    obj = _make_object(1, peak_dbz=60.0, az_center=50, rng_center=100)
    diameter = _compute_mesh(obj, ref_data, fzl)
    assert diameter is not None
    assert diameter > 0.0


def test_mesh_returns_none_without_freezing_level():
    ref_grid = np.full((360, 500), 60.0)
    ref_data = _make_ref_data(ref_grid)
    obj = _make_object(1, peak_dbz=60.0)
    diameter = _compute_mesh(obj, ref_data, None)
    assert diameter is None


def test_mesh_small_label():
    assert _mesh_label(0.5) == "small"


def test_mesh_moderate_label():
    assert _mesh_label(1.2) == "moderate"


def test_mesh_severe_label():
    assert _mesh_label(2.0) == "severe"


# --- Full analyze_hazards ---

def test_analyze_hazards_no_dual_pol():
    obj = _make_object(1, peak_dbz=55.0)
    ref_grid = np.full((360, 500), 55.0)
    ref_data = _make_ref_data(ref_grid)
    result = analyze_hazards(
        dual_pol=None,
        objects=[obj],
        rotation_signatures=[],
        freezing_level=None,
        reflectivity_data=ref_data,
    )
    assert isinstance(result, HazardAnalysis)
    assert len(result.hail_assessments) == 0
    assert len(result.debris_signatures) == 0


def test_analyze_hazards_returns_hail_assessment():
    zdr_grid = np.full((360, 500), 0.5)
    cc_grid = np.full((360, 500), 0.90)
    dual_pol = _make_dual_pol(zdr_grid, cc_grid)
    obj = _make_object(1, peak_dbz=55.0)
    ref_grid = np.full((360, 500), 55.0)
    ref_data = _make_ref_data(ref_grid)
    result = analyze_hazards(
        dual_pol=dual_pol,
        objects=[obj],
        rotation_signatures=[],
        freezing_level=None,
        reflectivity_data=ref_data,
    )
    assert len(result.hail_assessments) == 1
    assert result.hail_assessments[0].dual_pol_flag == "hail_likely"
```

Note: the `_mesh_label` import is already included in the main import block at the top of the file:

```python
from src.hazards import (
    HailAssessment, DebrisSignature, HazardAnalysis,
    analyze_hazards, _assess_hail_dual_pol, _compute_mesh, _mesh_label,
)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_hazards.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.hazards'`

- [ ] **Step 3: Commit test file**

```bash
git add tests/unit/test_hazards.py
git commit -m "test: add hail detection unit tests"
```

---

### Task 5: Hail Detection — Implementation

**Files:**
- Create: `src/hazards.py`

- [ ] **Step 1: Create src/hazards.py with hail detection logic**

```python
# src/hazards.py
import math
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import label

from src.detection import DetectedObject, polar_to_latlon, _range_bin_areas_km2
from src.ingest import FreezingLevel
from src.parser import DualPolData, ReflectivityData
from src.velocity import RotationSignature, _haversine_km

MIN_HAIL_DBZ = 45.0
MAX_HAIL_ZDR = 1.0
HAIL_CC_MIN = 0.85
HAIL_CC_MAX = 0.95
TDS_CC_THRESHOLD = 0.80
TDS_MIN_DBZ = 20.0
TDS_MAX_ELEVATION = 1.0
TDS_MAX_ROTATION_DISTANCE_KM = 5.0
TDS_MIN_AREA_KM2 = 1.0
METERS_PER_INCH = 0.0254


@dataclass
class HailAssessment:
    object_id: int
    dual_pol_flag: str | None
    mesh_diameter_in: float | None
    mesh_label: str | None
    max_zdr: float | None
    min_cc: float | None
    peak_dbz_above_fzl: float | None


@dataclass
class DebrisSignature:
    centroid_lat: float
    centroid_lon: float
    distance_km: float
    bearing_deg: float
    min_cc: float
    area_km2: float
    co_located_rotation_strength: str


@dataclass
class HazardAnalysis:
    hail_assessments: list[HailAssessment]
    debris_signatures: list[DebrisSignature]


def _mesh_label(diameter_in: float) -> str:
    if diameter_in >= 1.75:
        return "severe"
    if diameter_in >= 1.0:
        return "moderate"
    return "small"


def _assess_hail_dual_pol(
    obj: DetectedObject,
    zdr_min: float | None,
    cc_min: float | None,
) -> HailAssessment:
    """Classify hail likelihood using dual-pol indicators for a single object."""
    if obj.peak_dbz < MIN_HAIL_DBZ:
        return HailAssessment(
            object_id=obj.object_id,
            dual_pol_flag=None,
            mesh_diameter_in=None,
            mesh_label=None,
            max_zdr=zdr_min,
            min_cc=cc_min,
            peak_dbz_above_fzl=None,
        )

    low_zdr = zdr_min is not None and zdr_min < MAX_HAIL_ZDR
    reduced_cc = cc_min is not None and HAIL_CC_MIN <= cc_min <= HAIL_CC_MAX

    if low_zdr and reduced_cc:
        flag = "hail_likely"
    elif low_zdr or reduced_cc:
        flag = "hail_possible"
    else:
        flag = None

    return HailAssessment(
        object_id=obj.object_id,
        dual_pol_flag=flag,
        mesh_diameter_in=None,
        mesh_label=None,
        max_zdr=zdr_min,
        min_cc=cc_min,
        peak_dbz_above_fzl=None,
    )


def _object_pol_stats(
    obj: DetectedObject,
    dual_pol: DualPolData,
) -> tuple[float | None, float | None]:
    """Compute min ZDR and min CC within a small window around an object's centroid."""
    az_idx = np.argmin(np.abs(dual_pol.azimuths - (obj.bearing_deg % 360)))
    rng_idx = np.argmin(np.abs(dual_pol.ranges_m - obj.distance_km * 1000))
    half_window = 5
    az_lo = max(0, az_idx - half_window)
    az_hi = min(dual_pol.zdr.shape[0], az_idx + half_window + 1)
    rng_lo = max(0, rng_idx - half_window)
    rng_hi = min(dual_pol.zdr.shape[1], rng_idx + half_window + 1)
    zdr_patch = dual_pol.zdr[az_lo:az_hi, rng_lo:rng_hi]
    cc_patch = dual_pol.cc[az_lo:az_hi, rng_lo:rng_hi]
    valid_zdr = zdr_patch[~np.isnan(zdr_patch)]
    valid_cc = cc_patch[~np.isnan(cc_patch)]
    zdr_min = float(np.min(valid_zdr)) if len(valid_zdr) > 0 else None
    cc_min = float(np.min(valid_cc)) if len(valid_cc) > 0 else None
    return zdr_min, cc_min


def _compute_mesh(
    obj: DetectedObject,
    reflectivity_data: ReflectivityData,
    freezing_level: FreezingLevel | None,
) -> float | None:
    """Compute MESH diameter in inches for a single object.

    Uses a simplified SHI integration: for each range bin in the object's vicinity
    above the freezing level, weight reflectivity contribution using the NWS weight
    function. MESH = 2.54 * SHI^0.5 (empirical NWS formula).
    """
    if freezing_level is None:
        return None

    fzl_m = freezing_level.height_m_asl
    az_idx = np.argmin(np.abs(reflectivity_data.azimuths - (obj.bearing_deg % 360)))
    half_window = 5
    az_lo = max(0, az_idx - half_window)
    az_hi = min(reflectivity_data.reflectivity.shape[0], az_idx + half_window + 1)

    elevation_rad = math.radians(reflectivity_data.elevation_angle)
    earth_radius_m = 6371000.0

    shi = 0.0
    count = 0
    for rng_idx in range(reflectivity_data.reflectivity.shape[1]):
        range_m = float(reflectivity_data.ranges_m[rng_idx])
        beam_height_m = range_m * math.sin(elevation_rad) + (range_m ** 2) / (2 * earth_radius_m)
        if beam_height_m <= fzl_m:
            continue
        col_dbz = reflectivity_data.reflectivity[az_lo:az_hi, rng_idx]
        valid = col_dbz[~np.isnan(col_dbz)]
        if len(valid) == 0:
            continue
        max_dbz = float(np.max(valid))
        if max_dbz < 40.0:
            continue
        z_linear = 10.0 ** (max_dbz / 10.0)
        height_above_fzl_km = (beam_height_m - fzl_m) / 1000.0
        wt = _shi_weight(max_dbz)
        shi += wt * z_linear * height_above_fzl_km * 1e-6
        count += 1

    if shi <= 0.0:
        return None

    mesh_mm = 2.54 * (shi ** 0.5)
    mesh_in = round(mesh_mm / 25.4, 1)
    return max(0.1, mesh_in)


def _shi_weight(dbz: float) -> float:
    """NWS Severe Hail Index weight function."""
    if dbz < 40.0:
        return 0.0
    if dbz < 50.0:
        return (dbz - 40.0) / 10.0
    return 1.0


def _detect_debris(
    dual_pol: DualPolData,
    reflectivity_data: ReflectivityData,
    rotation_signatures: list[RotationSignature],
) -> list[DebrisSignature]:
    """Detect tornado debris signatures using strict NWS TDS criteria."""
    if dual_pol.elevation_angle > TDS_MAX_ELEVATION:
        return []

    ref_sweep0 = reflectivity_data.reflectivity
    cc = dual_pol.cc
    n_az = min(cc.shape[0], ref_sweep0.shape[0])
    n_rng = min(cc.shape[1], ref_sweep0.shape[1])
    cc_trimmed = cc[:n_az, :n_rng]
    ref_trimmed = ref_sweep0[:n_az, :n_rng]

    tds_mask = (
        ~np.isnan(cc_trimmed) &
        ~np.isnan(ref_trimmed) &
        (cc_trimmed < TDS_CC_THRESHOLD) &
        (ref_trimmed >= TDS_MIN_DBZ)
    )

    labeled_grid, count = label(tds_mask, structure=np.ones((3, 3), dtype=int))
    if count == 0:
        return []

    range_bin_areas = _range_bin_areas_km2(dual_pol.azimuths[:n_az], dual_pol.ranges_m[:n_rng])
    signatures: list[DebrisSignature] = []

    for component_id in range(1, count + 1):
        component_mask = labeled_grid == component_id
        az_indices, rng_indices = np.where(component_mask)
        area_km2 = float(np.sum(range_bin_areas[rng_indices]))
        if area_km2 < TDS_MIN_AREA_KM2:
            continue

        cc_values = cc_trimmed[component_mask]
        min_cc_val = float(np.nanmin(cc_values))

        centroid_az_idx = float(np.mean(az_indices))
        centroid_rng_idx = float(np.mean(rng_indices))
        centroid_az = float(np.interp(centroid_az_idx, range(len(dual_pol.azimuths[:n_az])), dual_pol.azimuths[:n_az]))
        centroid_range_m = float(np.interp(centroid_rng_idx, range(len(dual_pol.ranges_m[:n_rng])), dual_pol.ranges_m[:n_rng]))

        centroid_lat, centroid_lon = polar_to_latlon(
            dual_pol.radar_lat, dual_pol.radar_lon, centroid_az, centroid_range_m,
        )

        best_rotation: RotationSignature | None = None
        best_dist = float("inf")
        for rot in rotation_signatures:
            dist = _haversine_km(centroid_lat, centroid_lon, rot.centroid_lat, rot.centroid_lon)
            if dist <= TDS_MAX_ROTATION_DISTANCE_KM and dist < best_dist:
                best_dist = dist
                best_rotation = rot

        if best_rotation is None:
            continue

        signatures.append(DebrisSignature(
            centroid_lat=round(centroid_lat, 4),
            centroid_lon=round(centroid_lon, 4),
            distance_km=round(centroid_range_m / 1000.0, 1),
            bearing_deg=round(centroid_az % 360, 1),
            min_cc=round(min_cc_val, 3),
            area_km2=round(area_km2, 2),
            co_located_rotation_strength=best_rotation.strength,
        ))

    signatures.sort(key=lambda s: s.distance_km)
    return signatures


def analyze_hazards(
    dual_pol: DualPolData | None,
    objects: list[DetectedObject],
    rotation_signatures: list[RotationSignature],
    freezing_level: FreezingLevel | None,
    reflectivity_data: ReflectivityData,
) -> HazardAnalysis:
    """Run hail and debris analysis.

    Returns HazardAnalysis with per-object hail assessments and standalone debris signatures.
    """
    if dual_pol is None:
        return HazardAnalysis(hail_assessments=[], debris_signatures=[])

    hail_assessments: list[HailAssessment] = []
    for obj in objects:
        zdr_min, cc_min = _object_pol_stats(obj, dual_pol)
        assessment = _assess_hail_dual_pol(obj, zdr_min, cc_min)

        mesh_diameter = _compute_mesh(obj, reflectivity_data, freezing_level)
        if mesh_diameter is not None:
            peak_dbz_above_fzl = obj.peak_dbz
            assessment = HailAssessment(
                object_id=assessment.object_id,
                dual_pol_flag=assessment.dual_pol_flag,
                mesh_diameter_in=mesh_diameter,
                mesh_label=_mesh_label(mesh_diameter),
                max_zdr=assessment.max_zdr,
                min_cc=assessment.min_cc,
                peak_dbz_above_fzl=peak_dbz_above_fzl,
            )

        if assessment.dual_pol_flag is not None or assessment.mesh_diameter_in is not None:
            hail_assessments.append(assessment)

    debris_signatures = _detect_debris(dual_pol, reflectivity_data, rotation_signatures)

    return HazardAnalysis(
        hail_assessments=hail_assessments,
        debris_signatures=debris_signatures,
    )
```

- [ ] **Step 2: Run hail tests to verify they pass**

Run: `uv run pytest tests/unit/test_hazards.py -v`
Expected: All tests PASS

- [ ] **Step 3: Run full test suite**

Run: `uv run pytest tests/ -q`
Expected: All tests pass

- [ ] **Step 4: Commit**

```bash
git add src/hazards.py
git commit -m "feat: add hail detection and debris scoring module"
```

---

### Task 6: TDS Detection — Tests

**Files:**
- Modify: `tests/unit/test_hazards.py`

- [ ] **Step 1: Add TDS detection tests to test_hazards.py**

Append to `tests/unit/test_hazards.py`:

```python
from src.velocity import RotationSignature
from src.hazards import _detect_debris


def _make_rotation(lat: float, lon: float, strength: str = "strong") -> RotationSignature:
    return RotationSignature(
        centroid_lat=lat,
        centroid_lon=lon,
        distance_km=50.0,
        bearing_deg=90.0,
        max_shear_ms=40.0,
        max_inbound_ms=-25.0,
        max_outbound_ms=25.0,
        diameter_km=2.0,
        sweep_count=1,
        elevation_angles=[0.5],
        strength=strength,
    )


def test_tds_requires_low_cc_and_high_ref():
    cc_grid = np.full((360, 500), 0.95)
    cc_grid[50:60, 100:110] = 0.70
    zdr_grid = np.full((360, 500), 2.0)
    dual_pol = _make_dual_pol(zdr_grid, cc_grid)
    ref_grid = np.full((360, 500), np.nan)
    ref_grid[50:60, 100:110] = 30.0
    ref_data = _make_ref_data(ref_grid)
    from src.detection import polar_to_latlon
    azimuths = np.linspace(0, 359, 360)
    ranges_m = np.linspace(2000, 230000, 500)
    lat, lon = polar_to_latlon(RADAR_LAT, RADAR_LON, float(azimuths[55]), float(ranges_m[105]))
    rot = _make_rotation(lat, lon)
    result = _detect_debris(dual_pol, ref_data, [rot])
    assert len(result) >= 1
    assert result[0].co_located_rotation_strength == "strong"


def test_tds_requires_co_located_rotation():
    cc_grid = np.full((360, 500), 0.95)
    cc_grid[50:60, 100:110] = 0.70
    zdr_grid = np.full((360, 500), 2.0)
    dual_pol = _make_dual_pol(zdr_grid, cc_grid)
    ref_grid = np.full((360, 500), np.nan)
    ref_grid[50:60, 100:110] = 30.0
    ref_data = _make_ref_data(ref_grid)
    far_rot = _make_rotation(40.0, -90.0)
    result = _detect_debris(dual_pol, ref_data, [far_rot])
    assert len(result) == 0


def test_tds_requires_low_elevation():
    cc_grid = np.full((360, 500), 0.95)
    cc_grid[50:60, 100:110] = 0.70
    zdr_grid = np.full((360, 500), 2.0)
    dual_pol = _make_dual_pol(zdr_grid, cc_grid)
    dual_pol_high = DualPolData(
        zdr=dual_pol.zdr, cc=dual_pol.cc,
        azimuths=dual_pol.azimuths, ranges_m=dual_pol.ranges_m,
        radar_lat=RADAR_LAT, radar_lon=RADAR_LON,
        elevation_angle=2.0,
        timestamp=dual_pol.timestamp,
    )
    ref_grid = np.full((360, 500), np.nan)
    ref_grid[50:60, 100:110] = 30.0
    ref_data = _make_ref_data(ref_grid)
    from src.detection import polar_to_latlon
    azimuths = np.linspace(0, 359, 360)
    ranges_m = np.linspace(2000, 230000, 500)
    lat, lon = polar_to_latlon(RADAR_LAT, RADAR_LON, float(azimuths[55]), float(ranges_m[105]))
    rot = _make_rotation(lat, lon)
    result = _detect_debris(dual_pol_high, ref_data, [rot])
    assert len(result) == 0


def test_tds_no_rotation_no_debris():
    cc_grid = np.full((360, 500), 0.70)
    zdr_grid = np.full((360, 500), 2.0)
    dual_pol = _make_dual_pol(zdr_grid, cc_grid)
    ref_grid = np.full((360, 500), 30.0)
    ref_data = _make_ref_data(ref_grid)
    result = _detect_debris(dual_pol, ref_data, [])
    assert len(result) == 0


def test_tds_cc_above_threshold_no_debris():
    cc_grid = np.full((360, 500), 0.85)
    zdr_grid = np.full((360, 500), 2.0)
    dual_pol = _make_dual_pol(zdr_grid, cc_grid)
    ref_grid = np.full((360, 500), 30.0)
    ref_data = _make_ref_data(ref_grid)
    from src.detection import polar_to_latlon
    azimuths = np.linspace(0, 359, 360)
    ranges_m = np.linspace(2000, 230000, 500)
    lat, lon = polar_to_latlon(RADAR_LAT, RADAR_LON, float(azimuths[180]), float(ranges_m[250]))
    rot = _make_rotation(lat, lon)
    result = _detect_debris(dual_pol, ref_data, [rot])
    assert len(result) == 0
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_hazards.py -v`
Expected: All tests PASS (hail + TDS tests)

- [ ] **Step 3: Commit**

```bash
git add tests/unit/test_hazards.py
git commit -m "test: add TDS debris detection tests"
```

---

### Task 7: Buffer and Tracker Integration

**Files:**
- Modify: `src/buffer.py`
- Modify: `src/tracking/types.py`
- Modify: `src/tracking/__init__.py`
- Modify: `src/tracker.py`

- [ ] **Step 1: Add hazard fields to BufferedScan in src/buffer.py**

Add import at top of `src/buffer.py`:

```python
from src.hazards import HailAssessment, DebrisSignature
from src.parser import ReflectivityData, VelocityData, DualPolData
```

Replace the existing `from src.parser import ...` line.

Add three new fields to the `BufferedScan` dataclass after `rotation_signatures`:

```python
    dual_pol_data: DualPolData | None = None
    hail_assessments: list[HailAssessment] = field(default_factory=list)
    debris_signatures: list[DebrisSignature] = field(default_factory=list)
```

- [ ] **Step 2: Add HailHistoryEntry to tracking types**

Add to `src/tracking/types.py` after the `RotationHistoryEntry` dataclass:

```python
@dataclass
class HailHistoryEntry:
    timestamp: datetime
    hail_assessment: "HailAssessment | None"
```

Add field to `Track` dataclass after `rotation_history`:

```python
    hail_history: list[HailHistoryEntry] = field(default_factory=list)
```

Add import at top of `src/tracking/types.py`:

```python
from src.hazards import HailAssessment
```

- [ ] **Step 3: Export HailHistoryEntry from tracking/__init__.py**

Add `HailHistoryEntry` to the import from `src.tracking.types` and to `__all__`:

In the import line, add `HailHistoryEntry`:
```python
from src.tracking.types import (
    AssociationScore,
    HailHistoryEntry,
    MotionConfidence,
    PeakEntry,
    RotationHistoryEntry,
    SegmentedStormObject,
    Track,
    TrackPosition,
)
```

In `__all__`, add `"HailHistoryEntry"`.

- [ ] **Step 4: Record hail history in tracker.py**

Add import at top of `src/tracker.py`:

```python
from src.tracking.types import FocusContinuity, HailHistoryEntry, IdentityConfidence, MotionSample, RotationHistoryEntry, Track
```

In `_create_track` method, after the rotation_history append block, add:

```python
        track.hail_history.append(HailHistoryEntry(
            timestamp=timestamp,
            hail_assessment=self._find_hail_for_object(obj),
        ))
        if len(track.hail_history) > 6:
            track.hail_history = track.hail_history[-6:]
```

Add a helper method to `StormTracker`:

```python
    def _find_hail_for_object(self, obj: DetectedObject) -> "HailAssessment | None":
        """Look up the hail assessment for an object from the current scan's buffer."""
        if self._prev_scan is None:
            return None
        for assessment in getattr(self._prev_scan, "hail_assessments", []):
            if assessment.object_id == obj.object_id:
                return assessment
        return None
```

At each of the 4 `add_position` call sites in the `update` method (split parent, merge survivor, primary match, and unmatched new), add after the rotation_history block:

```python
            track.hail_history.append(HailHistoryEntry(
                timestamp=timestamp,
                hail_assessment=self._find_hail_for_object(new_objects[new_id]),
            ))
            if len(track.hail_history) > 6:
                track.hail_history = track.hail_history[-6:]
```

Note: For the `_find_hail_for_object` to work on the *current* scan, the `_prev_scan` won't have the current scan's hail data yet. Instead, store a reference to the current scan's hail assessments. Add a `_current_hail_assessments` field:

```python
    def __init__(self):
        ...
        self._current_hail_assessments: list = []
```

In the `update` method, at the top after clearing events:

```python
        self._current_hail_assessments = getattr(scan, "hail_assessments", [])
```

And change `_find_hail_for_object` to use it:

```python
    def _find_hail_for_object(self, obj: DetectedObject) -> "HailAssessment | None":
        for assessment in self._current_hail_assessments:
            if assessment.object_id == obj.object_id:
                return assessment
        return None
```

- [ ] **Step 5: Run full test suite**

Run: `uv run pytest tests/ -q`
Expected: All tests pass

- [ ] **Step 6: Commit**

```bash
git add src/buffer.py src/tracking/types.py src/tracking/__init__.py src/tracker.py
git commit -m "feat: integrate hazards into buffer and tracker"
```

---

### Task 8: API Models and Endpoints

**Files:**
- Modify: `src/models.py`
- Modify: `src/server.py`

- [ ] **Step 1: Add hazard models to src/models.py**

Add after `RotationHistoryEntryModel`:

```python
class HailAssessmentModel(BaseModel):
    object_id: int
    dual_pol_flag: str | None = None
    mesh_diameter_in: float | None = None
    mesh_label: str | None = None
    max_zdr: float | None = None
    min_cc: float | None = None
    peak_dbz_above_fzl: float | None = None


class DebrisSignatureModel(BaseModel):
    centroid_lat: float
    centroid_lon: float
    distance_km: float
    bearing_deg: float
    min_cc: float
    area_km2: float
    co_located_rotation_strength: str


class HazardResponse(BaseModel):
    site_id: str
    timestamp: str
    hail_assessments: list[HailAssessmentModel]
    debris_signatures: list[DebrisSignatureModel]


class HailHistoryEntryModel(BaseModel):
    timestamp: str
    dual_pol_flag: str | None = None
    mesh_diameter_in: float | None = None
    mesh_label: str | None = None
```

Add to `RainObject`, after `rotation_strength`:

```python
    hail_flag: str | None = None
    mesh_diameter_in: float | None = None
    mesh_label: str | None = None
```

Add to `StormTrack`, after `rotation_history`:

```python
    hail_history: list[HailHistoryEntryModel] = []
```

- [ ] **Step 2: Wire hazards into server pipeline**

In `src/server.py`, add imports:

```python
from src.parser import parse_radar_file, extract_reflectivity_from_radar, extract_velocity, extract_dual_pol
from src.hazards import analyze_hazards
from src.ingest import fetch_scan, fetch_freezing_level
from src.models import (
    ...,
    HailAssessmentModel, DebrisSignatureModel, HazardResponse, HailHistoryEntryModel,
)
```

Add module-level freezing level cache after `_tracker`:

```python
_freezing_level_cache: dict[str, "FreezingLevel | None"] = {}
```

In `_ingest_to_buffer`, after `analyze_velocity` and before creating `BufferedScan`:

```python
    dual_pol = extract_dual_pol(radar)

    site_upper = site_id.upper()
    if site_upper not in _freezing_level_cache:
        _freezing_level_cache.clear()
        _freezing_level_cache[site_upper] = fetch_freezing_level(
            ref_data.radar_lat, ref_data.radar_lon
        )
    freezing_level = _freezing_level_cache.get(site_upper)

    hazard_result = analyze_hazards(
        dual_pol=dual_pol,
        objects=annotated_objects,
        rotation_signatures=rotations,
        freezing_level=freezing_level,
        reflectivity_data=ref_data,
    )
```

Update the `BufferedScan` constructor to include:

```python
        dual_pol_data=dual_pol,
        hail_assessments=hazard_result.hail_assessments,
        debris_signatures=hazard_result.debris_signatures,
```

In `get_objects`, add hail fields to `RainObject`:

```python
    hail_map = {h.object_id: h for h in buffered.hail_assessments}
```

Then inside the RainObject constructor, add:

```python
            hail_flag=hail_map[obj.object_id].dual_pol_flag if obj.object_id in hail_map else None,
            mesh_diameter_in=hail_map[obj.object_id].mesh_diameter_in if obj.object_id in hail_map else None,
            mesh_label=hail_map[obj.object_id].mesh_label if obj.object_id in hail_map else None,
```

In `_track_to_model`, add hail_history after rotation_history:

```python
        hail_history=[
            HailHistoryEntryModel(
                timestamp=entry.timestamp.isoformat() if isinstance(entry.timestamp, datetime) else entry.timestamp,
                dual_pol_flag=entry.hail_assessment.dual_pol_flag if entry.hail_assessment else None,
                mesh_diameter_in=entry.hail_assessment.mesh_diameter_in if entry.hail_assessment else None,
                mesh_label=entry.hail_assessment.mesh_label if entry.hail_assessment else None,
            )
            for entry in getattr(track, 'hail_history', [])
        ],
```

- [ ] **Step 3: Add /hazards endpoint**

```python
@app.get("/hazards/{site_id}", response_model=HazardResponse)
def get_hazards(site_id: str, datetime: str | None = Query(None)):
    dt = _parse_datetime(datetime)
    buffered = _ingest_to_buffer(site_id, dt)
    hail_models = [
        HailAssessmentModel(
            object_id=h.object_id,
            dual_pol_flag=h.dual_pol_flag,
            mesh_diameter_in=h.mesh_diameter_in,
            mesh_label=h.mesh_label,
            max_zdr=h.max_zdr,
            min_cc=h.min_cc,
            peak_dbz_above_fzl=h.peak_dbz_above_fzl,
        )
        for h in buffered.hail_assessments
    ]
    debris_models = [
        DebrisSignatureModel(
            centroid_lat=d.centroid_lat,
            centroid_lon=d.centroid_lon,
            distance_km=d.distance_km,
            bearing_deg=d.bearing_deg,
            min_cc=d.min_cc,
            area_km2=d.area_km2,
            co_located_rotation_strength=d.co_located_rotation_strength,
        )
        for d in buffered.debris_signatures
    ]
    return HazardResponse(
        site_id=site_id.upper(),
        timestamp=buffered.reflectivity_data.timestamp,
        hail_assessments=hail_models,
        debris_signatures=debris_models,
    )
```

- [ ] **Step 4: Run full test suite**

Run: `uv run pytest tests/ -q`
Expected: Some smoke/e2e tests may fail due to missing patches — that's expected (fixed in Task 10)

- [ ] **Step 5: Commit**

```bash
git add src/models.py src/server.py
git commit -m "feat: add hazard API models and /hazards endpoint"
```

---

### Task 9: Summary Integration

**Files:**
- Modify: `src/summary.py`
- Modify: `tests/unit/test_summary.py`

- [ ] **Step 1: Write summary hail/debris tests**

Add to `tests/unit/test_summary.py`:

```python
from src.hazards import HailAssessment, DebrisSignature


def test_summary_includes_hail_with_mesh():
    obj = DetectedObject(
        object_id=1, centroid_lat=35.5, centroid_lon=-97.0,
        distance_km=50.0, bearing_deg=90.0, peak_dbz=60.0,
        peak_label="severe core", area_km2=100.0,
    )
    hail = HailAssessment(
        object_id=1, dual_pol_flag="hail_likely",
        mesh_diameter_in=2.0, mesh_label="severe",
        max_zdr=0.3, min_cc=0.88, peak_dbz_above_fzl=58.0,
    )
    text = generate_summary(
        site_id="KTLX", site_name="Oklahoma City",
        timestamp="2026-04-10T18:00:00Z",
        objects=[obj], tracks=None, events=None,
        hail_assessments=[hail], debris_signatures=[],
    )
    assert "Hail up to 2.0 inches" in text


def test_summary_includes_hail_without_mesh():
    obj = DetectedObject(
        object_id=1, centroid_lat=35.5, centroid_lon=-97.0,
        distance_km=50.0, bearing_deg=90.0, peak_dbz=55.0,
        peak_label="intense rain", area_km2=100.0,
    )
    hail = HailAssessment(
        object_id=1, dual_pol_flag="hail_likely",
        mesh_diameter_in=None, mesh_label=None,
        max_zdr=0.5, min_cc=0.90, peak_dbz_above_fzl=None,
    )
    text = generate_summary(
        site_id="KTLX", site_name="Oklahoma City",
        timestamp="2026-04-10T18:00:00Z",
        objects=[obj], tracks=None, events=None,
        hail_assessments=[hail], debris_signatures=[],
    )
    assert "Hail signatures detected" in text


def test_summary_includes_possible_hail():
    obj = DetectedObject(
        object_id=1, centroid_lat=35.5, centroid_lon=-97.0,
        distance_km=50.0, bearing_deg=90.0, peak_dbz=55.0,
        peak_label="intense rain", area_km2=100.0,
    )
    hail = HailAssessment(
        object_id=1, dual_pol_flag="hail_possible",
        mesh_diameter_in=None, mesh_label=None,
        max_zdr=0.5, min_cc=0.97, peak_dbz_above_fzl=None,
    )
    text = generate_summary(
        site_id="KTLX", site_name="Oklahoma City",
        timestamp="2026-04-10T18:00:00Z",
        objects=[obj], tracks=None, events=None,
        hail_assessments=[hail], debris_signatures=[],
    )
    assert "Possible hail signatures" in text


def test_summary_debris_alert_before_strongest():
    obj = DetectedObject(
        object_id=1, centroid_lat=35.5, centroid_lon=-97.0,
        distance_km=50.0, bearing_deg=90.0, peak_dbz=55.0,
        peak_label="intense rain", area_km2=100.0,
    )
    debris = DebrisSignature(
        centroid_lat=35.6, centroid_lon=-97.1,
        distance_km=25.0, bearing_deg=340.0,
        min_cc=0.65, area_km2=3.0,
        co_located_rotation_strength="strong",
    )
    text = generate_summary(
        site_id="KTLX", site_name="Oklahoma City",
        timestamp="2026-04-10T18:00:00Z",
        objects=[obj], tracks=None, events=None,
        hail_assessments=[], debris_signatures=[debris],
    )
    debris_idx = text.index("Tornado debris signature")
    strongest_idx = text.index("Strongest")
    assert debris_idx < strongest_idx
    assert "strong rotation" in text
```

- [ ] **Step 2: Update generate_summary signature and implementation in src/summary.py**

Update the `generate_summary` function signature to accept hail and debris:

```python
def generate_summary(
    site_id: str,
    site_name: str,
    timestamp: str,
    objects: list[DetectedObject],
    tracks=None,
    events: list[dict] | None = None,
    hail_assessments: list | None = None,
    debris_signatures: list | None = None,
) -> str:
```

Add hail formatting helper after `_format_rotation`:

```python
def _format_hail(obj: DetectedObject, hail_assessments: list | None) -> str:
    if not hail_assessments:
        return ""
    assessment = next((h for h in hail_assessments if h.object_id == obj.object_id), None)
    if assessment is None:
        return ""
    if assessment.mesh_diameter_in is not None:
        return f" Hail up to {assessment.mesh_diameter_in} inches."
    if assessment.dual_pol_flag == "hail_likely":
        return " Hail signatures detected."
    if assessment.dual_pol_flag == "hail_possible":
        return " Possible hail signatures."
    return ""
```

Add debris formatting helper:

```python
def _format_debris_alerts(debris_signatures: list | None) -> str:
    if not debris_signatures:
        return ""
    alerts: list[str] = []
    for debris in debris_signatures[:2]:
        bearing = degrees_to_bearing(debris.bearing_deg)
        distance_mi = km_to_miles(debris.distance_km)
        alerts.append(
            f"Tornado debris signature detected {distance_mi} miles {bearing} of the radar,"
            f" {debris.co_located_rotation_strength} rotation."
        )
    return " ".join(alerts)
```

In `generate_summary`, build the parts list with debris first, then strongest with hail:

```python
    hail_str = _format_hail(strongest, hail_assessments)
    debris_str = _format_debris_alerts(debris_signatures)

    parts = []
    if debris_str:
        parts.append(debris_str + " ")
    parts.append(
        f"{site_name}: {count} {obj_word} detected. "
        f"Strongest: {strongest.peak_label}, "
        f"{distance_mi} miles {bearing} of the radar{motion_str}."
        f"{rotation_str}{hail_str}"
    )
```

The merge/split notes and standalone rotation reports follow as before. Coverage last.

- [ ] **Step 3: Run summary tests**

Run: `uv run pytest tests/unit/test_summary.py -v`
Expected: All tests PASS (old + new)

- [ ] **Step 4: Update server.py generate_summary call to pass hail/debris**

In `get_summary` in `src/server.py`:

```python
    text = generate_summary(
        site_id=site_id.upper(),
        site_name=site_name,
        timestamp=buffered.reflectivity_data.timestamp,
        objects=buffered.detected_objects,
        tracks=_tracker.active_tracks,
        events=_tracker.recent_events,
        hail_assessments=buffered.hail_assessments,
        debris_signatures=buffered.debris_signatures,
    )
```

Also update the `summarize_scan` call in `scripts/live_replay.py`:

```python
    summary = generate_summary(
        site_id=buffered.site_id,
        site_name=site_name,
        timestamp=buffered.reflectivity_data.timestamp,
        objects=buffered.detected_objects,
        tracks=tracker.active_tracks,
        events=tracker.recent_events,
        hail_assessments=getattr(buffered, "hail_assessments", []),
        debris_signatures=getattr(buffered, "debris_signatures", []),
    )
```

- [ ] **Step 5: Commit**

```bash
git add src/summary.py tests/unit/test_summary.py src/server.py scripts/live_replay.py
git commit -m "feat: add hail and debris language to spoken summaries"
```

---

### Task 10: Update Smoke and E2E Tests

**Files:**
- Modify: `tests/smoke/test_server_smoke.py`
- Modify: `tests/e2e/test_full_pipeline.py`

- [ ] **Step 1: Update smoke test patches**

In `tests/smoke/test_server_smoke.py`, add to every `with patch(...)` block:

```python
         patch("src.server.extract_dual_pol", return_value=None), \
         patch("src.server.fetch_freezing_level", return_value=None), \
```

These go after the `extract_velocity` patch in every test function that uses it.

Add a new smoke test for `/hazards`:

```python
def test_hazards_endpoint_returns_200():
    mock_ref = _make_reflectivity_data(np.nan)
    with patch("src.server.fetch_scan", return_value="/fake/path"), \
         patch("src.server.parse_radar_file", return_value=MagicMock()), \
         patch("src.server.extract_reflectivity_from_radar", return_value=mock_ref), \
         patch("src.server.extract_velocity", return_value=None), \
         patch("src.server.extract_dual_pol", return_value=None), \
         patch("src.server.fetch_freezing_level", return_value=None):
        resp = client.get("/hazards/KTLX")
    assert resp.status_code == 200
    data = resp.json()
    assert "hail_assessments" in data
    assert "debris_signatures" in data
```

- [ ] **Step 2: Update e2e test patches**

In `tests/e2e/test_full_pipeline.py`, add to every `with patch(...)` block:

```python
         patch("src.server.extract_dual_pol", return_value=None), \
         patch("src.server.fetch_freezing_level", return_value=None), \
```

These go after the `extract_velocity` patch in every test function.

- [ ] **Step 3: Run full test suite**

Run: `uv run pytest tests/ -q`
Expected: All tests pass

- [ ] **Step 4: Commit**

```bash
git add tests/smoke/test_server_smoke.py tests/e2e/test_full_pipeline.py
git commit -m "test: update smoke and e2e patches for hazard pipeline"
```

---

### Task 11: Replay Script Integration

**Files:**
- Modify: `scripts/live_replay.py`

- [ ] **Step 1: Add hazard imports and processing to live_replay.py**

Add imports at top:

```python
from src.parser import parse_radar_file, extract_reflectivity_from_radar, extract_velocity, extract_dual_pol
from src.hazards import analyze_hazards
from src.ingest import get_cache_path, list_latest_scans, list_scans_for_date, fetch_scan, scan_is_cached, fetch_freezing_level
```

Add `hail_count` and `debris_count` fields to `ReplayDiagnostics`:

```python
    hail_count: int
    debris_count: int
```

In the `main` function, before the scan loop, add freezing level fetch:

```python
    from src.sites import NEXRAD_SITES
    site_info = next((s for s in NEXRAD_SITES if s["site_id"] == site_id), None)
    freezing_level = None
    if site_info:
        freezing_level = fetch_freezing_level(site_info["latitude"], site_info["longitude"])
```

In the scan loop, after `analyze_velocity`, add:

```python
        dual_pol = extract_dual_pol(radar)
        hazard_result = analyze_hazards(
            dual_pol=dual_pol,
            objects=annotated_objects,
            rotation_signatures=rotations,
            freezing_level=freezing_level,
            reflectivity_data=reflectivity,
        )
```

Update the `BufferedScan` constructor to include:

```python
            dual_pol_data=dual_pol,
            hail_assessments=hazard_result.hail_assessments,
            debris_signatures=hazard_result.debris_signatures,
```

In `summarize_scan`, compute hail/debris counts:

```python
    hail_count = len(getattr(buffered, "hail_assessments", []))
    debris_count = len(getattr(buffered, "debris_signatures", []))
```

Add to the return statement:

```python
        hail_count=hail_count,
        debris_count=debris_count,
```

Add to the print line:

```python
            f"HAIL={diagnostics.hail_count} "
            f"DEBRIS={diagnostics.debris_count}"
```

- [ ] **Step 2: Run full test suite**

Run: `uv run pytest tests/ -q`
Expected: All tests pass

- [ ] **Step 3: Commit**

```bash
git add scripts/live_replay.py
git commit -m "feat: add hazard counts to replay diagnostics"
```

---

### Task 12: Live Validation

**Files:**
- No new files — this is a validation task

- [ ] **Step 1: Run replay against cached KTLX afternoon window**

Run: `uv run python scripts/live_replay.py KTLX --date 2026-04-10 --scans 5 --local-only`

Expected:
- HAIL count > 0 for at least some scans (active severe weather)
- DEBRIS count may be 0 (depends on whether real TDS signatures exist in this data)
- Rotation counts should match pre-change values (~129-142)
- Focus track should remain stable
- No "tracking uncertain" regressions vs pre-change behavior
- Summary text should include hail language for high-reflectivity objects

- [ ] **Step 2: Verify no tracking regressions**

Compare output against the Phase 3 validation report (`docs/test_reports/2026-04-18-velocity-validation.md`):
- Same focus track IDs
- Same merge/split counts
- Same rotation counts
- Motion suppression behavior unchanged

- [ ] **Step 3: Run full test suite one final time**

Run: `uv run pytest tests/ -q`
Expected: All tests pass (should be ~220+ with new hazard tests)

- [ ] **Step 4: Write validation report**

Create `docs/test_reports/2026-04-19-hail-debris-validation.md` documenting:
- Replay window details
- Hail detection counts and examples
- Debris detection status
- Regression check results
- Final test count

- [ ] **Step 5: Commit validation report**

```bash
git add docs/test_reports/2026-04-19-hail-debris-validation.md
git commit -m "docs: add Phase 4 hail/debris validation report"
```

---

### Task 13: Update PROGRESS.md

**Files:**
- Modify: `PROGRESS.md`

- [ ] **Step 1: Update PROGRESS.md with Phase 4 completion**

Update the Completed section to add Phase 3 velocity details and Phase 4:

```markdown
- Phase 4: hail detection and debris scoring
  - Dual-pol extraction: ZDR and CC from sweep 0
  - Freezing level ingest: RAP model via NOMADS, cached per site switch
  - Hail detection: dual-pol discrimination (likely/possible) + MESH diameter estimation
  - TDS detection: strict NWS criteria (CC < 0.80, reflectivity >= 20 dBZ, co-located rotation, low elevation)
  - Pipeline integration: buffer, tracker hail_history, /hazards endpoint, /objects hail fields
  - Summary integration: hail per-object annotation, debris standalone alerts (highest priority)
  - Live validation: replay confirmed hail/debris detection with no tracking regressions
- Full test suite: ~220+ tests all passing
```

Update "In Progress" to "None — Phase 4 hail/debris work is complete"

Update "Next" to Phase 5.

- [ ] **Step 2: Commit**

```bash
git add PROGRESS.md
git commit -m "docs: update PROGRESS.md for Phase 4 completion"
```
