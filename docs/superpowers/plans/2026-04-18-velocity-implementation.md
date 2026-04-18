# Velocity Ingestion & Region Detection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add velocity data extraction from multiple NEXRAD sweeps, inbound/outbound region detection, rotation signature detection, and integrate into the buffer, tracker, API, and spoken summaries.

**Architecture:** A single new `src/velocity.py` module handles all velocity analysis. The parser is refactored to expose the pyart radar object so velocity can read from it without re-reading the file. Velocity data flows through the existing buffer/tracker pipeline via new fields on `BufferedScan`, `DetectedObject`, and `Track`.

**Tech Stack:** Python, Py-ART (pyart), NumPy, SciPy (connected-component labeling), FastAPI

**Spec:** `docs/superpowers/specs/2026-04-18-velocity-design.md`

---

### Task 1: Refactor parser to expose radar object

The parser currently reads the NEXRAD file and discards the pyart radar object. Split into `parse_radar_file()` that returns the radar, and refactor `extract_reflectivity()` to accept a radar object.

**Files:**
- Modify: `src/parser.py`
- Modify: `tests/unit/test_parser.py`

- [ ] **Step 1: Write failing tests for the new parser interface**

Add to `tests/unit/test_parser.py`:

```python
from src.parser import parse_radar_file, extract_reflectivity_from_radar

def test_parse_radar_file_returns_radar_object():
    mock_radar = _make_mock_radar()
    with patch("src.parser.pyart.io.read_nexrad_archive", return_value=mock_radar):
        radar = parse_radar_file("/fake/path.V06")
    assert radar is mock_radar


def test_extract_reflectivity_from_radar():
    mock_radar = _make_mock_radar()
    result = extract_reflectivity_from_radar(mock_radar)
    assert isinstance(result, ReflectivityData)
    assert result.reflectivity.shape[0] == 360
    assert result.radar_lat == 35.3331
    assert result.elevation_angle == 0.5
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_parser.py -v`
Expected: FAIL — `parse_radar_file` and `extract_reflectivity_from_radar` not defined

- [ ] **Step 3: Implement the refactored parser**

In `src/parser.py`, add `parse_radar_file` and `extract_reflectivity_from_radar`, and refactor the existing `extract_reflectivity` to call them:

```python
def parse_radar_file(filepath: str):
    """Read a NEXRAD Level II file and return the pyart Radar object."""
    return pyart.io.read_nexrad_archive(filepath)


def extract_reflectivity_from_radar(radar) -> ReflectivityData:
    """Extract reflectivity from the lowest sweep of a pyart Radar object."""
    elevation_angles = sorted(set(np.round(radar.fixed_angle["data"], 1)))
    sweep_start, sweep_end = radar.get_start_end(0)
    reflectivity = radar.fields["reflectivity"]["data"][sweep_start:sweep_end + 1]
    azimuths = radar.azimuth["data"][sweep_start:sweep_end + 1]
    ranges_m = radar.range["data"]
    if hasattr(reflectivity, "filled"):
        reflectivity = reflectivity.filled(np.nan)
    return ReflectivityData(
        reflectivity=reflectivity,
        azimuths=azimuths,
        ranges_m=ranges_m,
        radar_lat=float(radar.latitude["data"][0]),
        radar_lon=float(radar.longitude["data"][0]),
        elevation_angle=float(radar.fixed_angle["data"][0]),
        elevation_angles=[float(a) for a in elevation_angles],
        timestamp=str(radar.time["units"]).replace("seconds since ", ""),
    )


def extract_reflectivity(filepath: str) -> ReflectivityData:
    """Read a NEXRAD Level II file and extract reflectivity from the lowest sweep."""
    radar = parse_radar_file(filepath)
    return extract_reflectivity_from_radar(radar)
```

- [ ] **Step 4: Run all parser tests**

Run: `uv run pytest tests/unit/test_parser.py -v`
Expected: all pass (old tests still work via unchanged `extract_reflectivity`, new tests use new functions)

- [ ] **Step 5: Run full test suite to check for regressions**

Run: `uv run pytest tests/ -q`
Expected: all 164 tests pass

- [ ] **Step 6: Commit**

```bash
git add src/parser.py tests/unit/test_parser.py
git commit -m "refactor: expose radar object from parser for velocity access"
```

---

### Task 2: Add velocity data structures and extraction

Add `VelocitySweep`, `VelocityData` dataclasses to the parser, and `extract_velocity()` function that reads velocity from the lowest N sweeps.

**Files:**
- Modify: `src/parser.py`
- Create: `tests/unit/test_velocity_parser.py`

- [ ] **Step 1: Write failing tests for velocity extraction**

Create `tests/unit/test_velocity_parser.py`:

```python
import numpy as np
from unittest.mock import MagicMock
from src.parser import extract_velocity, VelocityData, VelocitySweep


def _make_mock_radar_with_velocity():
    """Create a mock Py-ART radar with both reflectivity and velocity fields."""
    radar = MagicMock()
    radar.nsweeps = 3
    radar.fixed_angle = {"data": np.array([0.5, 1.5, 2.4])}
    radar.latitude = {"data": np.array([35.3331])}
    radar.longitude = {"data": np.array([-97.2778])}
    radar.instrument_parameters = {
        "nyquist_velocity": {"data": np.full(1080, 26.2)}
    }

    def get_start_end(sweep_index):
        start = sweep_index * 360
        end = start + 359
        return (start, end)

    radar.get_start_end.side_effect = get_start_end

    velocity_data = np.random.uniform(-30, 30, (1080, 1832)).astype(np.float32)
    radar.fields = {
        "reflectivity": {"data": np.ma.array(np.zeros((1080, 1832)), mask=False)},
        "velocity": {"data": np.ma.array(velocity_data, mask=False)},
    }
    radar.azimuth = {"data": np.tile(np.linspace(0, 359, 360), 3)}
    radar.range = {"data": np.linspace(0, 459750, 1832)}
    return radar


def test_extract_velocity_returns_velocity_data():
    radar = _make_mock_radar_with_velocity()
    result = extract_velocity(radar)
    assert isinstance(result, VelocityData)
    assert len(result.sweeps) > 0
    assert result.radar_lat == 35.3331
    assert result.radar_lon == -97.2778


def test_extract_velocity_respects_max_sweeps():
    radar = _make_mock_radar_with_velocity()
    result = extract_velocity(radar, max_sweeps=2)
    assert len(result.sweeps) == 2


def test_extract_velocity_extracts_all_available_when_fewer_than_max():
    radar = _make_mock_radar_with_velocity()
    result = extract_velocity(radar, max_sweeps=5)
    assert len(result.sweeps) == 3


def test_extract_velocity_sweep_has_correct_fields():
    radar = _make_mock_radar_with_velocity()
    result = extract_velocity(radar, max_sweeps=1)
    sweep = result.sweeps[0]
    assert isinstance(sweep, VelocitySweep)
    assert sweep.velocity.shape == (360, 1832)
    assert sweep.elevation_angle == 0.5
    assert sweep.nyquist_velocity > 0
    assert len(sweep.azimuths) == 360
    assert len(sweep.ranges_m) == 1832


def test_extract_velocity_returns_none_when_no_velocity_field():
    radar = _make_mock_radar_with_velocity()
    del radar.fields["velocity"]
    result = extract_velocity(radar)
    assert result is None


def test_extract_velocity_fills_masked_values_with_nan():
    radar = _make_mock_radar_with_velocity()
    velocity_array = np.ma.array(
        np.ones((1080, 1832)) * 10.0,
        mask=np.zeros((1080, 1832), dtype=bool),
    )
    velocity_array.mask[0, 0] = True
    radar.fields["velocity"] = {"data": velocity_array}
    result = extract_velocity(radar, max_sweeps=1)
    assert np.isnan(result.sweeps[0].velocity[0, 0])
    assert not np.isnan(result.sweeps[0].velocity[0, 1])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_velocity_parser.py -v`
Expected: FAIL — `extract_velocity`, `VelocityData`, `VelocitySweep` not defined

- [ ] **Step 3: Implement velocity data structures and extraction**

Add to `src/parser.py`:

```python
@dataclass
class VelocitySweep:
    """Velocity data from a single radar sweep."""
    velocity: np.ndarray
    azimuths: np.ndarray
    ranges_m: np.ndarray
    elevation_angle: float
    nyquist_velocity: float


@dataclass
class VelocityData:
    """Multi-sweep velocity data from a radar volume."""
    sweeps: list[VelocitySweep]
    radar_lat: float
    radar_lon: float


def extract_velocity(radar, max_sweeps: int = 3) -> VelocityData | None:
    """Extract velocity from the lowest N sweeps of a pyart Radar object.

    Returns None if the radar has no velocity field.
    """
    if "velocity" not in radar.fields:
        return None

    # Apply Py-ART region-based dealiasing to unwrap aliased velocities
    try:
        pyart.correct.dealias_region_based(radar, field="velocity")
    except Exception:
        pass  # proceed with raw velocity if dealiasing fails

    sweeps_to_read = min(max_sweeps, radar.nsweeps)
    sweeps: list[VelocitySweep] = []

    for sweep_index in range(sweeps_to_read):
        sweep_start, sweep_end = radar.get_start_end(sweep_index)
        velocity = radar.fields["velocity"]["data"][sweep_start:sweep_end + 1]
        azimuths = radar.azimuth["data"][sweep_start:sweep_end + 1]
        ranges_m = radar.range["data"]

        if hasattr(velocity, "filled"):
            velocity = velocity.filled(np.nan)

        nyquist = float(radar.instrument_parameters["nyquist_velocity"]["data"][sweep_start])

        sweeps.append(VelocitySweep(
            velocity=velocity,
            azimuths=azimuths,
            ranges_m=ranges_m,
            elevation_angle=float(radar.fixed_angle["data"][sweep_index]),
            nyquist_velocity=nyquist,
        ))

    return VelocityData(
        sweeps=sweeps,
        radar_lat=float(radar.latitude["data"][0]),
        radar_lon=float(radar.longitude["data"][0]),
    )
```

- [ ] **Step 4: Run velocity parser tests**

Run: `uv run pytest tests/unit/test_velocity_parser.py -v`
Expected: all pass

- [ ] **Step 5: Run full test suite**

Run: `uv run pytest tests/ -q`
Expected: all pass

- [ ] **Step 6: Commit**

```bash
git add src/parser.py tests/unit/test_velocity_parser.py
git commit -m "feat: add multi-sweep velocity extraction from NEXRAD data"
```

---

### Task 3: Velocity region detection

Create `src/velocity.py` with inbound/outbound region detection using connected-component labeling on thresholded velocity fields.

**Files:**
- Create: `src/velocity.py`
- Create: `tests/unit/test_velocity.py`

- [ ] **Step 1: Write failing tests for velocity region detection**

Create `tests/unit/test_velocity.py`:

```python
import numpy as np
from src.parser import VelocitySweep, VelocityData
from src.velocity import VelocityRegion, detect_velocity_regions

RADAR_LAT = 35.3331
RADAR_LON = -97.2778


def _make_sweep(velocity_grid: np.ndarray, elevation: float = 0.5) -> VelocitySweep:
    n_az, n_rng = velocity_grid.shape
    return VelocitySweep(
        velocity=velocity_grid,
        azimuths=np.linspace(0, 359, n_az),
        ranges_m=np.linspace(2000, 230000, n_rng),
        elevation_angle=elevation,
        nyquist_velocity=26.2,
    )


def _make_velocity_data(sweeps: list[VelocitySweep]) -> VelocityData:
    return VelocityData(sweeps=sweeps, radar_lat=RADAR_LAT, radar_lon=RADAR_LON)


def test_detect_velocity_regions_finds_inbound():
    grid = np.full((360, 500), np.nan)
    grid[50:70, 100:130] = -20.0  # inbound block
    vel_data = _make_velocity_data([_make_sweep(grid)])
    regions = detect_velocity_regions(vel_data)
    inbound = [r for r in regions if r.region_type == "inbound"]
    assert len(inbound) == 1
    assert inbound[0].peak_velocity_ms <= -20.0


def test_detect_velocity_regions_finds_outbound():
    grid = np.full((360, 500), np.nan)
    grid[100:120, 200:230] = 25.0  # outbound block
    vel_data = _make_velocity_data([_make_sweep(grid)])
    regions = detect_velocity_regions(vel_data)
    outbound = [r for r in regions if r.region_type == "outbound"]
    assert len(outbound) == 1
    assert outbound[0].peak_velocity_ms >= 25.0


def test_detect_velocity_regions_ignores_weak_velocity():
    grid = np.full((360, 500), np.nan)
    grid[50:70, 100:130] = -5.0  # below 10 m/s threshold
    vel_data = _make_velocity_data([_make_sweep(grid)])
    regions = detect_velocity_regions(vel_data)
    assert len(regions) == 0


def test_detect_velocity_regions_filters_small_regions():
    grid = np.full((360, 500), np.nan)
    grid[50, 100] = -20.0  # single pixel — too small
    vel_data = _make_velocity_data([_make_sweep(grid)])
    regions = detect_velocity_regions(vel_data)
    assert len(regions) == 0


def test_detect_velocity_regions_multi_sweep_increases_sweep_count():
    grid1 = np.full((360, 500), np.nan)
    grid1[50:70, 100:130] = -20.0
    grid2 = np.full((360, 500), np.nan)
    grid2[50:70, 100:130] = -22.0  # same location, second sweep
    vel_data = _make_velocity_data([
        _make_sweep(grid1, elevation=0.5),
        _make_sweep(grid2, elevation=1.5),
    ])
    regions = detect_velocity_regions(vel_data)
    inbound = [r for r in regions if r.region_type == "inbound"]
    assert len(inbound) == 1
    assert inbound[0].sweep_count == 2


def test_detect_velocity_regions_returns_empty_for_all_nan():
    grid = np.full((360, 500), np.nan)
    vel_data = _make_velocity_data([_make_sweep(grid)])
    regions = detect_velocity_regions(vel_data)
    assert regions == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_velocity.py -v`
Expected: FAIL — `src.velocity` module not found

- [ ] **Step 3: Implement velocity region detection**

Create `src/velocity.py`:

```python
import math
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import label

from src.detection import polar_to_latlon, degrees_to_bearing, _range_bin_areas_km2
from src.parser import VelocityData

MIN_VELOCITY_MS = 10.0
MIN_REGION_AREA_KM2 = 4.0
CROSS_SWEEP_OVERLAP_THRESHOLD = 0.3


@dataclass
class VelocityRegion:
    region_type: str
    peak_velocity_ms: float
    mean_velocity_ms: float
    area_km2: float
    centroid_lat: float
    centroid_lon: float
    distance_km: float
    bearing_deg: float
    sweep_count: int
    elevation_angles: list[float]


def _detect_regions_single_sweep(
    velocity: np.ndarray,
    azimuths: np.ndarray,
    ranges_m: np.ndarray,
    radar_lat: float,
    radar_lon: float,
    elevation_angle: float,
) -> list[tuple[VelocityRegion, np.ndarray]]:
    """Detect inbound/outbound regions on a single sweep. Returns (region, mask) pairs."""
    range_bin_areas = _range_bin_areas_km2(azimuths, ranges_m)
    results: list[tuple[VelocityRegion, np.ndarray]] = []

    for region_type, condition in [
        ("inbound", velocity <= -MIN_VELOCITY_MS),
        ("outbound", velocity >= MIN_VELOCITY_MS),
    ]:
        valid = ~np.isnan(velocity) & condition
        labeled_grid, count = label(valid, structure=np.ones((3, 3), dtype=int))

        for component_id in range(1, count + 1):
            mask = labeled_grid == component_id
            az_indices, rng_indices = np.where(mask)
            area_km2 = float(np.sum(range_bin_areas[rng_indices]))

            if area_km2 < MIN_REGION_AREA_KM2:
                continue

            region_velocities = velocity[mask]
            if region_type == "inbound":
                peak = float(np.nanmin(region_velocities))
            else:
                peak = float(np.nanmax(region_velocities))
            mean = float(np.nanmean(region_velocities))

            weights = np.abs(region_velocities)
            weight_sum = float(np.sum(weights))
            if weight_sum == 0:
                continue
            centroid_az_idx = float(np.average(az_indices, weights=weights))
            centroid_rng_idx = float(np.average(rng_indices, weights=weights))
            centroid_az = float(np.interp(centroid_az_idx, range(len(azimuths)), azimuths))
            centroid_range = float(np.interp(centroid_rng_idx, range(len(ranges_m)), ranges_m))

            centroid_lat, centroid_lon = polar_to_latlon(
                radar_lat, radar_lon, centroid_az, centroid_range,
            )

            results.append((VelocityRegion(
                region_type=region_type,
                peak_velocity_ms=round(peak, 1),
                mean_velocity_ms=round(mean, 1),
                area_km2=round(area_km2, 2),
                centroid_lat=round(centroid_lat, 4),
                centroid_lon=round(centroid_lon, 4),
                distance_km=round(centroid_range / 1000.0, 1),
                bearing_deg=round(centroid_az % 360, 1),
                sweep_count=1,
                elevation_angles=[elevation_angle],
            ), mask))

    return results


def _merge_cross_sweep_regions(
    all_sweep_results: list[list[tuple[VelocityRegion, np.ndarray]]],
) -> list[VelocityRegion]:
    """Merge regions from multiple sweeps by spatial overlap."""
    if not all_sweep_results:
        return []

    merged: list[tuple[VelocityRegion, np.ndarray]] = []

    for sweep_results in all_sweep_results:
        for region, mask in sweep_results:
            matched = False
            for i, (existing_region, existing_mask) in enumerate(merged):
                if existing_region.region_type != region.region_type:
                    continue
                overlap = np.count_nonzero(mask & existing_mask)
                union = np.count_nonzero(mask | existing_mask)
                if union > 0 and overlap / union >= CROSS_SWEEP_OVERLAP_THRESHOLD:
                    if region.region_type == "inbound":
                        new_peak = min(existing_region.peak_velocity_ms, region.peak_velocity_ms)
                    else:
                        new_peak = max(existing_region.peak_velocity_ms, region.peak_velocity_ms)
                    merged[i] = (VelocityRegion(
                        region_type=existing_region.region_type,
                        peak_velocity_ms=new_peak,
                        mean_velocity_ms=round(
                            (existing_region.mean_velocity_ms + region.mean_velocity_ms) / 2, 1
                        ),
                        area_km2=max(existing_region.area_km2, region.area_km2),
                        centroid_lat=existing_region.centroid_lat,
                        centroid_lon=existing_region.centroid_lon,
                        distance_km=existing_region.distance_km,
                        bearing_deg=existing_region.bearing_deg,
                        sweep_count=existing_region.sweep_count + 1,
                        elevation_angles=existing_region.elevation_angles + region.elevation_angles,
                    ), existing_mask | mask)
                    matched = True
                    break
            if not matched:
                merged.append((region, mask))

    return [region for region, _ in merged]


def detect_velocity_regions(vel_data: VelocityData) -> list[VelocityRegion]:
    """Detect inbound/outbound velocity regions across all sweeps."""
    all_sweep_results: list[list[tuple[VelocityRegion, np.ndarray]]] = []

    for sweep in vel_data.sweeps:
        sweep_results = _detect_regions_single_sweep(
            velocity=sweep.velocity,
            azimuths=sweep.azimuths,
            ranges_m=sweep.ranges_m,
            radar_lat=vel_data.radar_lat,
            radar_lon=vel_data.radar_lon,
            elevation_angle=sweep.elevation_angle,
        )
        all_sweep_results.append(sweep_results)

    return _merge_cross_sweep_regions(all_sweep_results)
```

- [ ] **Step 4: Run velocity tests**

Run: `uv run pytest tests/unit/test_velocity.py -v`
Expected: all pass

- [ ] **Step 5: Run full test suite**

Run: `uv run pytest tests/ -q`
Expected: all pass

- [ ] **Step 6: Commit**

```bash
git add src/velocity.py tests/unit/test_velocity.py
git commit -m "feat: add velocity region detection with cross-sweep merging"
```

---

### Task 4: Rotation signature detection

Add gate-to-gate shear detection and rotation signature clustering to `src/velocity.py`.

**Files:**
- Modify: `src/velocity.py`
- Modify: `tests/unit/test_velocity.py`

- [ ] **Step 1: Write failing tests for rotation detection**

Add to `tests/unit/test_velocity.py`:

```python
from src.velocity import RotationSignature, detect_rotation_signatures


def test_detect_rotation_finds_shear_couplet():
    grid = np.full((360, 500), np.nan)
    # inbound block adjacent to outbound block — classic couplet
    grid[50:60, 150:160] = -20.0  # inbound
    grid[60:70, 150:160] = 20.0   # outbound
    vel_data = _make_velocity_data([_make_sweep(grid)])
    signatures = detect_rotation_signatures(vel_data)
    assert len(signatures) >= 1
    assert signatures[0].max_shear_ms >= 15.0


def test_detect_rotation_classifies_strength_weak():
    grid = np.full((360, 500), np.nan)
    grid[50:60, 150:160] = -10.0
    grid[60:70, 150:160] = 10.0   # 20 m/s shear — weak
    vel_data = _make_velocity_data([_make_sweep(grid)])
    signatures = detect_rotation_signatures(vel_data)
    assert len(signatures) >= 1
    assert signatures[0].strength == "weak"


def test_detect_rotation_classifies_strength_moderate():
    grid = np.full((360, 500), np.nan)
    grid[50:60, 150:160] = -15.0
    grid[60:70, 150:160] = 15.0   # 30 m/s shear — moderate
    vel_data = _make_velocity_data([_make_sweep(grid)])
    signatures = detect_rotation_signatures(vel_data)
    assert len(signatures) >= 1
    assert signatures[0].strength == "moderate"


def test_detect_rotation_classifies_strength_strong():
    grid = np.full((360, 500), np.nan)
    grid[50:60, 150:160] = -20.0
    grid[60:70, 150:160] = 20.0   # 40 m/s shear — strong
    vel_data = _make_velocity_data([_make_sweep(grid)])
    signatures = detect_rotation_signatures(vel_data)
    assert len(signatures) >= 1
    assert signatures[0].strength == "strong"


def test_detect_rotation_ignores_weak_shear():
    grid = np.full((360, 500), np.nan)
    grid[50:60, 150:160] = -5.0
    grid[60:70, 150:160] = 5.0    # 10 m/s — below threshold
    vel_data = _make_velocity_data([_make_sweep(grid)])
    signatures = detect_rotation_signatures(vel_data)
    assert len(signatures) == 0


def test_detect_rotation_returns_empty_for_no_velocity():
    grid = np.full((360, 500), np.nan)
    vel_data = _make_velocity_data([_make_sweep(grid)])
    signatures = detect_rotation_signatures(vel_data)
    assert signatures == []


def test_detect_rotation_multi_sweep_increases_sweep_count():
    grid1 = np.full((360, 500), np.nan)
    grid1[50:60, 150:160] = -20.0
    grid1[60:70, 150:160] = 20.0
    grid2 = np.full((360, 500), np.nan)
    grid2[50:60, 150:160] = -22.0
    grid2[60:70, 150:160] = 22.0
    vel_data = _make_velocity_data([
        _make_sweep(grid1, elevation=0.5),
        _make_sweep(grid2, elevation=1.5),
    ])
    signatures = detect_rotation_signatures(vel_data)
    assert len(signatures) >= 1
    assert signatures[0].sweep_count == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_velocity.py -v -k rotation`
Expected: FAIL — `detect_rotation_signatures` and `RotationSignature` not defined

- [ ] **Step 3: Implement rotation detection**

Add to `src/velocity.py`:

```python
MIN_SHEAR_MS = 15.0
MAX_COUPLET_DISTANCE_KM = 5.0
ROTATION_MERGE_DISTANCE_KM = 10.0


@dataclass
class RotationSignature:
    centroid_lat: float
    centroid_lon: float
    distance_km: float
    bearing_deg: float
    max_shear_ms: float
    max_inbound_ms: float
    max_outbound_ms: float
    diameter_km: float
    sweep_count: int
    elevation_angles: list[float]
    strength: str


def _classify_rotation_strength(shear_ms: float) -> str:
    if shear_ms >= 35.0:
        return "strong"
    if shear_ms >= 25.0:
        return "moderate"
    return "weak"


def _detect_shear_single_sweep(
    velocity: np.ndarray,
    azimuths: np.ndarray,
    ranges_m: np.ndarray,
    radar_lat: float,
    radar_lon: float,
    elevation_angle: float,
) -> list[tuple[RotationSignature, float, float]]:
    """Find gate-to-gate shear couplets on one sweep.

    Returns (signature, centroid_az, centroid_range_m) tuples for cross-sweep merging.
    """
    n_az, n_rng = velocity.shape
    range_spacing_m = float(ranges_m[1] - ranges_m[0]) if len(ranges_m) > 1 else 250.0
    az_spacing_deg = float(azimuths[1] - azimuths[0]) if len(azimuths) > 1 else 1.0

    shear_mask = np.zeros_like(velocity, dtype=bool)
    shear_values = np.full_like(velocity, np.nan)
    inbound_values = np.full_like(velocity, np.nan)
    outbound_values = np.full_like(velocity, np.nan)

    # Check azimuthal neighbors for sign changes
    for delta_az in [-1, 0, 1]:
        for delta_rng in [-1, 0, 1]:
            if delta_az == 0 and delta_rng == 0:
                continue
            shifted_az = np.roll(velocity, delta_az, axis=0)
            shifted_rng = np.roll(shifted_az, delta_rng, axis=1)

            gate_distance_km = math.sqrt(
                (delta_az * az_spacing_deg * math.pi / 180 * ranges_m.mean()) ** 2
                + (delta_rng * range_spacing_m) ** 2
            ) / 1000.0

            if gate_distance_km > MAX_COUPLET_DISTANCE_KM:
                continue

            v1 = velocity
            v2 = shifted_rng
            both_valid = ~np.isnan(v1) & ~np.isnan(v2)
            sign_change = both_valid & ((v1 < 0) != (v2 < 0))
            shear = np.where(sign_change, np.abs(v1 - v2), np.nan)
            strong_shear = ~np.isnan(shear) & (shear >= MIN_SHEAR_MS)

            new_shear = strong_shear & (~shear_mask | (shear > shear_values))
            shear_mask |= strong_shear
            shear_values = np.where(new_shear, shear, shear_values)
            inbound_values = np.where(
                new_shear,
                np.minimum(v1, v2),
                inbound_values,
            )
            outbound_values = np.where(
                new_shear,
                np.maximum(v1, v2),
                outbound_values,
            )

    labeled_shear, count = label(shear_mask, structure=np.ones((3, 3), dtype=int))
    results: list[tuple[RotationSignature, float, float]] = []

    for component_id in range(1, count + 1):
        component_mask = labeled_shear == component_id
        if np.count_nonzero(component_mask) < 2:
            continue

        az_indices, rng_indices = np.where(component_mask)
        component_shear = shear_values[component_mask]
        max_shear = float(np.nanmax(component_shear))
        max_inbound = float(np.nanmin(inbound_values[component_mask]))
        max_outbound = float(np.nanmax(outbound_values[component_mask]))

        weights = np.nan_to_num(component_shear, nan=0.0)
        weight_sum = float(np.sum(weights))
        if weight_sum == 0:
            continue
        centroid_az_idx = float(np.average(az_indices, weights=weights))
        centroid_rng_idx = float(np.average(rng_indices, weights=weights))
        centroid_az = float(np.interp(centroid_az_idx, range(len(azimuths)), azimuths))
        centroid_range = float(np.interp(centroid_rng_idx, range(len(ranges_m)), ranges_m))

        az_extent = (float(np.max(az_indices)) - float(np.min(az_indices))) * az_spacing_deg
        rng_extent = (float(np.max(rng_indices)) - float(np.min(rng_indices))) * range_spacing_m
        diameter_km = math.sqrt(
            (az_extent * math.pi / 180 * centroid_range) ** 2
            + rng_extent ** 2
        ) / 1000.0

        centroid_lat, centroid_lon = polar_to_latlon(
            radar_lat, radar_lon, centroid_az, centroid_range,
        )

        results.append((RotationSignature(
            centroid_lat=round(centroid_lat, 4),
            centroid_lon=round(centroid_lon, 4),
            distance_km=round(centroid_range / 1000.0, 1),
            bearing_deg=round(centroid_az % 360, 1),
            max_shear_ms=round(max_shear, 1),
            max_inbound_ms=round(max_inbound, 1),
            max_outbound_ms=round(max_outbound, 1),
            diameter_km=round(diameter_km, 1),
            sweep_count=1,
            elevation_angles=[elevation_angle],
            strength=_classify_rotation_strength(max_shear),
        ), centroid_az, centroid_range))

    return results


def _merge_cross_sweep_rotations(
    all_sweep_results: list[list[tuple[RotationSignature, float, float]]],
    ranges_m: np.ndarray,
) -> list[RotationSignature]:
    """Merge rotation signatures from multiple sweeps by proximity."""
    if not all_sweep_results:
        return []

    merged: list[tuple[RotationSignature, float, float]] = []

    for sweep_results in all_sweep_results:
        for sig, az, rng in sweep_results:
            matched = False
            for i, (existing_sig, existing_az, existing_rng) in enumerate(merged):
                az_dist_deg = abs(az - existing_az)
                if az_dist_deg > 180:
                    az_dist_deg = 360 - az_dist_deg
                rng_dist_m = abs(rng - existing_rng)
                approx_dist_km = math.sqrt(
                    (az_dist_deg * math.pi / 180 * rng) ** 2
                    + rng_dist_m ** 2
                ) / 1000.0

                if approx_dist_km <= ROTATION_MERGE_DISTANCE_KM:
                    merged[i] = (RotationSignature(
                        centroid_lat=existing_sig.centroid_lat,
                        centroid_lon=existing_sig.centroid_lon,
                        distance_km=existing_sig.distance_km,
                        bearing_deg=existing_sig.bearing_deg,
                        max_shear_ms=max(existing_sig.max_shear_ms, sig.max_shear_ms),
                        max_inbound_ms=min(existing_sig.max_inbound_ms, sig.max_inbound_ms),
                        max_outbound_ms=max(existing_sig.max_outbound_ms, sig.max_outbound_ms),
                        diameter_km=max(existing_sig.diameter_km, sig.diameter_km),
                        sweep_count=existing_sig.sweep_count + 1,
                        elevation_angles=existing_sig.elevation_angles + sig.elevation_angles,
                        strength=_classify_rotation_strength(
                            max(existing_sig.max_shear_ms, sig.max_shear_ms)
                        ),
                    ), existing_az, existing_rng)
                    matched = True
                    break
            if not matched:
                merged.append((sig, az, rng))

    return [sig for sig, _, _ in merged]


def detect_rotation_signatures(vel_data: VelocityData) -> list[RotationSignature]:
    """Detect rotation signatures across all sweeps."""
    all_sweep_results: list[list[tuple[RotationSignature, float, float]]] = []

    for sweep in vel_data.sweeps:
        sweep_results = _detect_shear_single_sweep(
            velocity=sweep.velocity,
            azimuths=sweep.azimuths,
            ranges_m=sweep.ranges_m,
            radar_lat=vel_data.radar_lat,
            radar_lon=vel_data.radar_lon,
            elevation_angle=sweep.elevation_angle,
        )
        all_sweep_results.append(sweep_results)

    ranges_m = vel_data.sweeps[0].ranges_m if vel_data.sweeps else np.array([])
    return _merge_cross_sweep_rotations(all_sweep_results, ranges_m)
```

- [ ] **Step 4: Run rotation tests**

Run: `uv run pytest tests/unit/test_velocity.py -v -k rotation`
Expected: all pass

- [ ] **Step 5: Run full test suite**

Run: `uv run pytest tests/ -q`
Expected: all pass

- [ ] **Step 6: Commit**

```bash
git add src/velocity.py tests/unit/test_velocity.py
git commit -m "feat: add rotation signature detection with multi-sweep merging"
```

---

### Task 5: Rain object velocity association

Add `analyze_velocity()` to `src/velocity.py` that associates velocity regions and rotation signatures with `DetectedObject`s, and add velocity fields to `DetectedObject`.

**Files:**
- Modify: `src/velocity.py`
- Modify: `src/detection.py`
- Modify: `tests/unit/test_velocity.py`

- [ ] **Step 1: Add velocity fields to DetectedObject**

In `src/detection.py`, add optional velocity fields to `DetectedObject`:

```python
@dataclass
class DetectedObject:
    object_id: int
    centroid_lat: float
    centroid_lon: float
    distance_km: float
    bearing_deg: float
    peak_dbz: float
    peak_label: str
    area_km2: float
    layers: list[IntensityLayerData] = field(default_factory=list)
    max_inbound_ms: float | None = None
    max_outbound_ms: float | None = None
    rotation: "RotationSignature | None" = None
```

Use a string annotation for `RotationSignature` to avoid circular imports (it's defined in `src/velocity.py`).

- [ ] **Step 2: Write failing tests for analyze_velocity**

Add to `tests/unit/test_velocity.py`:

```python
from src.velocity import analyze_velocity
from src.detection import DetectedObject


def _make_object(object_id, lat, lon, distance_km, bearing_deg) -> DetectedObject:
    return DetectedObject(
        object_id=object_id,
        centroid_lat=lat,
        centroid_lon=lon,
        distance_km=distance_km,
        bearing_deg=bearing_deg,
        peak_dbz=55.0,
        peak_label="intense rain",
        area_km2=50.0,
    )


def test_analyze_velocity_associates_rotation_with_object():
    grid = np.full((360, 500), np.nan)
    grid[50:60, 150:160] = -20.0
    grid[60:70, 150:160] = 20.0
    vel_data = _make_velocity_data([_make_sweep(grid)])
    # Place an object at roughly the same location as the shear couplet
    azimuths = np.linspace(0, 359, 360)
    ranges_m = np.linspace(2000, 230000, 500)
    centroid_az = float(azimuths[60])
    centroid_range = float(ranges_m[155])
    from src.detection import polar_to_latlon
    lat, lon = polar_to_latlon(RADAR_LAT, RADAR_LON, centroid_az, centroid_range)
    obj = _make_object(1, round(lat, 4), round(lon, 4), round(centroid_range / 1000, 1), round(centroid_az, 1))
    objects = [obj]
    regions, rotations, annotated = analyze_velocity(vel_data, objects)
    assert annotated[0].rotation is not None
    assert annotated[0].rotation.strength in {"weak", "moderate", "strong"}


def test_analyze_velocity_associates_inbound_with_object():
    grid = np.full((360, 500), np.nan)
    grid[50:70, 100:130] = -20.0
    vel_data = _make_velocity_data([_make_sweep(grid)])
    azimuths = np.linspace(0, 359, 360)
    ranges_m = np.linspace(2000, 230000, 500)
    centroid_az = float(azimuths[60])
    centroid_range = float(ranges_m[115])
    from src.detection import polar_to_latlon
    lat, lon = polar_to_latlon(RADAR_LAT, RADAR_LON, centroid_az, centroid_range)
    obj = _make_object(1, round(lat, 4), round(lon, 4), round(centroid_range / 1000, 1), round(centroid_az, 1))
    objects = [obj]
    regions, rotations, annotated = analyze_velocity(vel_data, objects)
    assert annotated[0].max_inbound_ms is not None
    assert annotated[0].max_inbound_ms <= -10.0


def test_analyze_velocity_leaves_distant_objects_unannotated():
    grid = np.full((360, 500), np.nan)
    grid[50:60, 150:160] = -20.0
    grid[60:70, 150:160] = 20.0
    vel_data = _make_velocity_data([_make_sweep(grid)])
    # Place object far from the couplet
    obj = _make_object(1, 40.0, -90.0, 200.0, 180.0)
    objects = [obj]
    regions, rotations, annotated = analyze_velocity(vel_data, objects)
    assert annotated[0].rotation is None
    assert annotated[0].max_inbound_ms is None
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_velocity.py -v -k analyze`
Expected: FAIL — `analyze_velocity` not defined

- [ ] **Step 4: Implement analyze_velocity**

Add to `src/velocity.py`:

```python
from src.detection import DetectedObject
from dataclasses import replace

MAX_ASSOCIATION_DISTANCE_KM = 30.0


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def analyze_velocity(
    vel_data: VelocityData | None,
    objects: list[DetectedObject],
) -> tuple[list[VelocityRegion], list[RotationSignature], list[DetectedObject]]:
    """Run full velocity analysis and associate results with detected objects.

    Returns (regions, rotation_signatures, annotated_objects).
    """
    if vel_data is None:
        return [], [], objects

    regions = detect_velocity_regions(vel_data)
    rotations = detect_rotation_signatures(vel_data)

    annotated = list(objects)
    for i, obj in enumerate(annotated):
        best_inbound: float | None = None
        best_outbound: float | None = None
        best_rotation: RotationSignature | None = None
        best_rotation_dist = float("inf")

        for region in regions:
            dist = _haversine_km(
                obj.centroid_lat, obj.centroid_lon,
                region.centroid_lat, region.centroid_lon,
            )
            if dist > MAX_ASSOCIATION_DISTANCE_KM:
                continue
            if region.region_type == "inbound":
                if best_inbound is None or region.peak_velocity_ms < best_inbound:
                    best_inbound = region.peak_velocity_ms
            else:
                if best_outbound is None or region.peak_velocity_ms > best_outbound:
                    best_outbound = region.peak_velocity_ms

        for rotation in rotations:
            dist = _haversine_km(
                obj.centroid_lat, obj.centroid_lon,
                rotation.centroid_lat, rotation.centroid_lon,
            )
            if dist < best_rotation_dist and dist <= MAX_ASSOCIATION_DISTANCE_KM:
                best_rotation_dist = dist
                best_rotation = rotation

        annotated[i] = replace(
            obj,
            max_inbound_ms=best_inbound,
            max_outbound_ms=best_outbound,
            rotation=best_rotation,
        )

    return regions, rotations, annotated
```

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/unit/test_velocity.py -v`
Expected: all pass

- [ ] **Step 6: Run full test suite**

Run: `uv run pytest tests/ -q`
Expected: all pass (check that adding optional fields to DetectedObject doesn't break existing tests)

- [ ] **Step 7: Commit**

```bash
git add src/velocity.py src/detection.py tests/unit/test_velocity.py
git commit -m "feat: associate velocity regions and rotation with rain objects"
```

---

### Task 6: Buffer and pipeline integration

Add velocity fields to `BufferedScan`, update `_ingest_to_buffer` in the server, and update the live replay script.

**Files:**
- Modify: `src/buffer.py`
- Modify: `src/server.py`
- Modify: `scripts/live_replay.py`
- Modify: `scripts/evaluate_tracking.py`

- [ ] **Step 1: Add velocity fields to BufferedScan**

In `src/buffer.py`, add imports and fields:

```python
from src.parser import ReflectivityData, VelocityData
from src.velocity import VelocityRegion, RotationSignature
```

Add to `BufferedScan`:

```python
@dataclass
class BufferedScan:
    timestamp: datetime
    site_id: str
    reflectivity_data: ReflectivityData
    detected_objects: list[DetectedObject]
    labeled_grid: np.ndarray
    object_masks: dict[int, np.ndarray]
    scan_quality: ScanQuality | None = None
    velocity_data: VelocityData | None = None
    velocity_regions: list[VelocityRegion] = field(default_factory=list)
    rotation_signatures: list[RotationSignature] = field(default_factory=list)
```

- [ ] **Step 2: Update server pipeline**

In `src/server.py`, update imports and `_ingest_to_buffer`:

```python
from src.parser import extract_reflectivity, parse_radar_file, extract_reflectivity_from_radar, extract_velocity
from src.velocity import analyze_velocity
```

Update `_ingest_to_buffer`:

```python
def _ingest_to_buffer(site_id: str, dt: datetime | None = None) -> BufferedScan:
    """Fetch a scan, detect objects, and add to buffer + tracker."""
    filepath = fetch_scan(site_id.upper(), dt)
    radar = parse_radar_file(filepath)
    raw_ref_data = extract_reflectivity_from_radar(radar)
    ref_data, scan_quality = preprocess_reflectivity_data(raw_ref_data)
    vel_data = extract_velocity(radar)
    result = detect_objects_with_grid(
        reflectivity=ref_data.reflectivity,
        azimuths=ref_data.azimuths,
        ranges_m=ref_data.ranges_m,
        radar_lat=ref_data.radar_lat,
        radar_lon=ref_data.radar_lon,
    )
    regions, rotations, annotated_objects = analyze_velocity(vel_data, result.objects)
    scan_timestamp = datetime.fromisoformat(ref_data.timestamp) if isinstance(ref_data.timestamp, str) else ref_data.timestamp
    buffered = BufferedScan(
        timestamp=scan_timestamp,
        site_id=site_id.upper(),
        reflectivity_data=ref_data,
        detected_objects=annotated_objects,
        labeled_grid=result.labeled_grid,
        object_masks=result.object_masks,
        scan_quality=scan_quality,
        velocity_data=vel_data,
        velocity_regions=regions,
        rotation_signatures=rotations,
    )
    _buffer.add_scan(buffered)
    _tracker.update(buffered)
    return buffered
```

- [ ] **Step 3: Update live_replay.py pipeline**

In `scripts/live_replay.py`, apply the same pattern: use `parse_radar_file` + `extract_reflectivity_from_radar` + `extract_velocity` + `analyze_velocity`. Update the diagnostic print line to include rotation count. Find the line that calls `extract_reflectivity(filepath)` and replace with the new pipeline, then add rotation count to the diagnostic output.

- [ ] **Step 4: Update evaluate_tracking.py pipeline**

Same pattern as live_replay.py — use the new parser functions and velocity analysis. Find the `extract_reflectivity(filepath)` call and update.

- [ ] **Step 5: Run full test suite**

Run: `uv run pytest tests/ -q`
Expected: all pass — the e2e and smoke tests mock `extract_reflectivity` so they still work; the new `parse_radar_file` path only runs in the server's `_ingest_to_buffer`

- [ ] **Step 6: Commit**

```bash
git add src/buffer.py src/server.py scripts/live_replay.py scripts/evaluate_tracking.py
git commit -m "feat: integrate velocity analysis into ingest pipeline"
```

---

### Task 7: Track rotation history

Add `rotation_history` to `Track` so rotation persistence can be reported across scans.

**Files:**
- Modify: `src/tracking/types.py`
- Modify: `src/tracker.py`
- Create: `tests/unit/test_rotation_tracking.py`

- [ ] **Step 1: Write failing tests for rotation history on tracks**

Create `tests/unit/test_rotation_tracking.py`:

```python
from datetime import datetime
from src.tracking.types import Track, RotationHistoryEntry
from src.velocity import RotationSignature


def test_track_rotation_history_starts_empty():
    track = Track(track_id=1, status="active")
    assert track.rotation_history == []


def test_rotation_history_entry_stores_signature():
    entry = RotationHistoryEntry(
        timestamp=datetime(2026, 4, 10, 21, 0),
        rotation=RotationSignature(
            centroid_lat=35.5, centroid_lon=-97.0,
            distance_km=50.0, bearing_deg=90.0,
            max_shear_ms=30.0, max_inbound_ms=-20.0, max_outbound_ms=15.0,
            diameter_km=3.0, sweep_count=2, elevation_angles=[0.5, 1.5],
            strength="moderate",
        ),
    )
    assert entry.rotation.strength == "moderate"


def test_rotation_history_entry_stores_none_for_no_rotation():
    entry = RotationHistoryEntry(
        timestamp=datetime(2026, 4, 10, 21, 0),
        rotation=None,
    )
    assert entry.rotation is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_rotation_tracking.py -v`
Expected: FAIL — `RotationHistoryEntry` not defined

- [ ] **Step 3: Add RotationHistoryEntry and rotation_history to Track**

In `src/tracking/types.py`, add:

```python
from src.velocity import RotationSignature


@dataclass
class RotationHistoryEntry:
    timestamp: datetime
    rotation: RotationSignature | None
```

Add to `Track` class:

```python
    rotation_history: list[RotationHistoryEntry] = field(default_factory=list)
```

- [ ] **Step 4: Update tracker to record rotation history**

In `src/tracker.py`, in the `update()` method where tracks call `add_position()`, add after position update:

```python
from src.tracking.types import RotationHistoryEntry

# After track.add_position(timestamp, obj):
track.rotation_history.append(RotationHistoryEntry(
    timestamp=timestamp,
    rotation=obj.rotation,
))
if len(track.rotation_history) > 6:
    track.rotation_history = track.rotation_history[-6:]
```

- [ ] **Step 5: Update tracking __init__.py exports**

Add `RotationHistoryEntry` to `src/tracking/__init__.py` imports and `__all__`.

- [ ] **Step 6: Run tests**

Run: `uv run pytest tests/unit/test_rotation_tracking.py -v`
Expected: all pass

- [ ] **Step 7: Run full test suite**

Run: `uv run pytest tests/ -q`
Expected: all pass

- [ ] **Step 8: Commit**

```bash
git add src/tracking/types.py src/tracking/__init__.py src/tracker.py tests/unit/test_rotation_tracking.py
git commit -m "feat: track rotation history across scans"
```

---

### Task 8: API models and velocity endpoint

Add Pydantic models for velocity data and a new `/velocity/{site_id}` endpoint. Enhance existing endpoints with velocity fields.

**Files:**
- Modify: `src/models.py`
- Modify: `src/server.py`
- Modify: `tests/smoke/test_server_smoke.py`

- [ ] **Step 1: Add velocity models to src/models.py**

Add to `src/models.py`:

```python
class VelocityRegionModel(BaseModel):
    region_type: str
    peak_velocity_ms: float
    mean_velocity_ms: float
    area_km2: float
    centroid_lat: float
    centroid_lon: float
    distance_km: float
    bearing_deg: float
    sweep_count: int
    elevation_angles: list[float]


class RotationSignatureModel(BaseModel):
    centroid_lat: float
    centroid_lon: float
    distance_km: float
    bearing_deg: float
    max_shear_ms: float
    max_inbound_ms: float
    max_outbound_ms: float
    diameter_km: float
    sweep_count: int
    elevation_angles: list[float]
    strength: str
    associated_object_id: int | None = None


class VelocityResponse(BaseModel):
    site_id: str
    timestamp: str
    regions: list[VelocityRegionModel]
    rotation_signatures: list[RotationSignatureModel]
```

Add optional velocity fields to `RainObject`:

```python
class RainObject(BaseModel):
    # ... existing fields ...
    max_inbound_ms: float | None = None
    max_outbound_ms: float | None = None
    rotation_strength: str | None = None
```

- [ ] **Step 2: Add velocity endpoint to server**

Add to `src/server.py`:

```python
from src.models import VelocityResponse, VelocityRegionModel, RotationSignatureModel

@app.get("/velocity/{site_id}", response_model=VelocityResponse)
def get_velocity(site_id: str, datetime: str | None = Query(None)):
    dt = _parse_datetime(datetime)
    buffered = _ingest_to_buffer(site_id, dt)
    regions = [
        VelocityRegionModel(
            region_type=r.region_type,
            peak_velocity_ms=r.peak_velocity_ms,
            mean_velocity_ms=r.mean_velocity_ms,
            area_km2=r.area_km2,
            centroid_lat=r.centroid_lat,
            centroid_lon=r.centroid_lon,
            distance_km=r.distance_km,
            bearing_deg=r.bearing_deg,
            sweep_count=r.sweep_count,
            elevation_angles=r.elevation_angles,
        )
        for r in buffered.velocity_regions
    ]
    rotation_sigs = [
        RotationSignatureModel(
            centroid_lat=s.centroid_lat,
            centroid_lon=s.centroid_lon,
            distance_km=s.distance_km,
            bearing_deg=s.bearing_deg,
            max_shear_ms=s.max_shear_ms,
            max_inbound_ms=s.max_inbound_ms,
            max_outbound_ms=s.max_outbound_ms,
            diameter_km=s.diameter_km,
            sweep_count=s.sweep_count,
            elevation_angles=s.elevation_angles,
            strength=s.strength,
            associated_object_id=None,
        )
        for s in buffered.rotation_signatures
    ]
    return VelocityResponse(
        site_id=site_id.upper(),
        timestamp=buffered.reflectivity_data.timestamp,
        regions=regions,
        rotation_signatures=rotation_sigs,
    )
```

- [ ] **Step 3: Update existing object endpoint to include velocity fields**

In the `/objects/{site_id}` endpoint, update `RainObject` construction to include:

```python
max_inbound_ms=obj.max_inbound_ms,
max_outbound_ms=obj.max_outbound_ms,
rotation_strength=obj.rotation.strength if obj.rotation is not None else None,
```

- [ ] **Step 3b: Add rotation_history to StormTrack model and tracks endpoint**

In `src/models.py`, add:

```python
class RotationHistoryEntryModel(BaseModel):
    timestamp: str
    strength: str | None = None
    max_shear_ms: float | None = None
```

Add to `StormTrack`:

```python
    rotation_history: list[RotationHistoryEntryModel] = []
```

In the `/tracks/{site_id}` endpoint in `src/server.py`, populate `rotation_history` from the track:

```python
rotation_history=[
    RotationHistoryEntryModel(
        timestamp=entry.timestamp.isoformat(),
        strength=entry.rotation.strength if entry.rotation else None,
        max_shear_ms=entry.rotation.max_shear_ms if entry.rotation else None,
    )
    for entry in track.rotation_history
],
```

- [ ] **Step 3c: Populate associated_object_id on RotationSignatureModel**

In the velocity endpoint, after building `annotated_objects` from `analyze_velocity`, build a mapping from rotation signature to object:

```python
rotation_to_object: dict[int, int] = {}
for obj in buffered.detected_objects:
    if obj.rotation is not None:
        rotation_to_object[id(obj.rotation)] = obj.object_id

rotation_sigs = [
    RotationSignatureModel(
        # ... existing fields ...
        associated_object_id=rotation_to_object.get(id(s)),
    )
    for s in buffered.rotation_signatures
]
```

Note: Since rotation signatures on objects are references to the same objects in `buffered.rotation_signatures`, `id()` matching works. If this turns out to be unreliable, fall back to matching by centroid proximity.

- [ ] **Step 4: Write smoke tests for velocity endpoint**

Add to `tests/smoke/test_server_smoke.py`:

```python
def test_velocity_endpoint_returns_200():
    mock_ref = _make_mock_reflectivity()
    with patch("src.server.fetch_scan", return_value="/fake/path.V06"), \
         patch("src.server.parse_radar_file", return_value=MagicMock()), \
         patch("src.server.extract_reflectivity_from_radar", return_value=mock_ref), \
         patch("src.server.extract_velocity", return_value=None):
        response = client.get("/velocity/KTLX")
    assert response.status_code == 200
    data = response.json()
    assert "regions" in data
    assert "rotation_signatures" in data
```

- [ ] **Step 5: Update existing smoke tests to use new parser interface**

Update all existing smoke test patches from `patch("src.server.extract_reflectivity", ...)` to use `parse_radar_file` + `extract_reflectivity_from_radar` + `extract_velocity` patches.

- [ ] **Step 6: Run smoke tests**

Run: `uv run pytest tests/smoke/ -v`
Expected: all pass

- [ ] **Step 7: Run full test suite**

Run: `uv run pytest tests/ -q`
Expected: all pass

- [ ] **Step 8: Commit**

```bash
git add src/models.py src/server.py tests/smoke/test_server_smoke.py
git commit -m "feat: add velocity API endpoint and enhance objects with velocity data"
```

---

### Task 9: Summary integration

Add rotation language to spoken summaries.

**Files:**
- Modify: `src/summary.py`
- Modify: `tests/unit/test_summary.py` (or create if not exists)

- [ ] **Step 1: Check if summary tests exist**

Run: `uv run pytest tests/ -q --collect-only 2>&1 | grep summary`

- [ ] **Step 2: Write failing tests for rotation in summaries**

Create or add to summary tests:

```python
from src.summary import generate_summary
from src.detection import DetectedObject
from src.velocity import RotationSignature


def _make_object_with_rotation(strength="moderate"):
    return DetectedObject(
        object_id=1, centroid_lat=35.5, centroid_lon=-97.0,
        distance_km=50.0, bearing_deg=90.0,
        peak_dbz=55.0, peak_label="intense rain", area_km2=100.0,
        rotation=RotationSignature(
            centroid_lat=35.5, centroid_lon=-97.0,
            distance_km=50.0, bearing_deg=90.0,
            max_shear_ms=30.0, max_inbound_ms=-20.0, max_outbound_ms=15.0,
            diameter_km=3.0, sweep_count=2, elevation_angles=[0.5, 1.5],
            strength=strength,
        ),
    )


def test_summary_includes_rotation_for_strongest_object():
    obj = _make_object_with_rotation("moderate")
    text = generate_summary(
        site_id="KTLX", site_name="Oklahoma City",
        timestamp="2026-04-10T21:00:00Z", objects=[obj],
    )
    assert "rotation" in text.lower()


def test_summary_includes_rotation_strength():
    obj = _make_object_with_rotation("strong")
    text = generate_summary(
        site_id="KTLX", site_name="Oklahoma City",
        timestamp="2026-04-10T21:00:00Z", objects=[obj],
    )
    assert "strong rotation" in text.lower()


def test_summary_no_rotation_when_none():
    obj = DetectedObject(
        object_id=1, centroid_lat=35.5, centroid_lon=-97.0,
        distance_km=50.0, bearing_deg=90.0,
        peak_dbz=55.0, peak_label="intense rain", area_km2=100.0,
    )
    text = generate_summary(
        site_id="KTLX", site_name="Oklahoma City",
        timestamp="2026-04-10T21:00:00Z", objects=[obj],
    )
    assert "rotation" not in text.lower()
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_summary.py -v -k rotation` (or wherever the tests live)
Expected: FAIL — rotation not in summary output

- [ ] **Step 4: Add rotation language to summaries**

In `src/summary.py`, update `generate_summary` to add rotation info after motion for the strongest object:

```python
def _format_rotation(obj: DetectedObject, track=None) -> str:
    """Format rotation info for speech, with persistence language if track available."""
    rotation = getattr(obj, "rotation", None)
    if rotation is None:
        if track is not None and hasattr(track, "rotation_history"):
            recent = [e for e in track.rotation_history[-3:] if e.rotation is not None]
            if len(recent) > 0 and track.rotation_history[-1].rotation is None:
                return " Rotation weakening."
        return ""
    if track is not None and hasattr(track, "rotation_history"):
        consecutive_with = 0
        for entry in reversed(track.rotation_history):
            if entry.rotation is not None:
                consecutive_with += 1
            else:
                break
        if consecutive_with >= 3:
            return f" Persistent {rotation.strength} rotation."
        elif consecutive_with <= 1:
            return f" New {rotation.strength} rotation detected."
    return f" {rotation.strength.capitalize()} rotation detected."
```

In `generate_summary`, after the motion string in the strongest-object sentence, append rotation info. The function now optionally accepts a `track` argument (the focus track from the tracker, if available) to enable persistence language:

```python
    rotation_str = _format_rotation(strongest, track=focus_track)

    parts = [
        f"{site_name}: {count} {obj_word} detected. "
        f"Strongest: {strongest.peak_label}, "
        f"{distance_mi} miles {bearing} of the radar{motion_str}."
        f"{rotation_str}"
    ]
```

Also add standalone rotation reporting for any rotation signatures not associated with the strongest object. After the merge/split notes, add:

```python
    # Report additional rotation signatures not on the strongest object
    standalone_rotations = [
        obj for obj in objects
        if obj.object_id != strongest.object_id
        and getattr(obj, "rotation", None) is not None
    ]
    for obj in standalone_rotations[:2]:  # cap at 2 extra rotation mentions
        rot = obj.rotation
        rot_bearing = degrees_to_bearing(rot.bearing_deg)
        rot_distance_mi = km_to_miles(rot.distance_km)
        parts.append(
            f" Rotation signature {rot_distance_mi} miles {rot_bearing} of the radar,"
            f" {rot.strength} shear."
        )
```

- [ ] **Step 5: Run summary tests**

Run: `uv run pytest tests/unit/test_summary.py -v`
Expected: all pass

- [ ] **Step 6: Run full test suite**

Run: `uv run pytest tests/ -q`
Expected: all pass

- [ ] **Step 7: Commit**

```bash
git add src/summary.py tests/unit/test_summary.py
git commit -m "feat: add rotation language to spoken summaries"
```

---

### Task 10: Update e2e tests and smoke tests for new pipeline

Update existing e2e and smoke tests to work with the refactored parser (since the server now calls `parse_radar_file` + `extract_reflectivity_from_radar` instead of `extract_reflectivity`).

**Files:**
- Modify: `tests/e2e/test_full_pipeline.py`
- Modify: `tests/smoke/test_server_smoke.py`

- [ ] **Step 1: Update e2e test patches**

In `tests/e2e/test_full_pipeline.py`, replace all occurrences of:
```python
patch("src.server.extract_reflectivity", return_value=ref_data)
```
with:
```python
patch("src.server.parse_radar_file", return_value=MagicMock()), \
patch("src.server.extract_reflectivity_from_radar", return_value=ref_data), \
patch("src.server.extract_velocity", return_value=None)
```

- [ ] **Step 2: Update smoke test patches**

Same pattern in `tests/smoke/test_server_smoke.py`.

- [ ] **Step 3: Run all tests**

Run: `uv run pytest tests/ -q`
Expected: all pass

- [ ] **Step 4: Commit**

```bash
git add tests/e2e/test_full_pipeline.py tests/smoke/test_server_smoke.py
git commit -m "test: update e2e and smoke tests for refactored parser pipeline"
```

---

### Task 11: Live validation with cached data

Run the velocity pipeline against real cached NEXRAD data to verify it produces reasonable results.

**Files:**
- No code changes — validation only

- [ ] **Step 1: Run live replay with velocity**

Run: `uv run python scripts/live_replay.py KTLX --date 2026-04-10 --scans 8 --local-only --end-filename KTLX20260410_214445_V06`

Check output for rotation counts and velocity region counts in the diagnostic lines.

- [ ] **Step 2: Run a second window**

Run: `uv run python scripts/live_replay.py KTLX --date 2026-04-10 --scans 8 --local-only --end-filename KTLX20260410_204923_V06`

- [ ] **Step 3: Run benchmark evaluation**

Run: `uv run python scripts/evaluate_tracking.py --manifest docs/benchmarks/tracking_benchmark_manifest_broader_validation.json --output-json docs/test_reports/2026-04-18-velocity-validation.json --output-md docs/test_reports/2026-04-18-velocity-validation.md`

Verify no regressions in tracking metrics. Check that rotation counts appear in the output.

- [ ] **Step 4: Review results and write test report**

Write a brief test report to `docs/test_reports/` documenting the velocity validation results.

- [ ] **Step 5: Commit test report**

```bash
git add docs/test_reports/
git commit -m "chore: document velocity pipeline validation results"
```

- [ ] **Step 6: Update PROGRESS.md**

Update `PROGRESS.md` with Phase 3 velocity work completed.

```bash
git add PROGRESS.md
git commit -m "chore: update progress with Phase 3 velocity work"
```
