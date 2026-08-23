# Radar Data Layer Correctness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make ARW's NEXRAD data layer conform to established radar meteorology practice — correct sweep selection, polarimetric quality control, and georeferencing — before any further interpretation work is built on top of it.

**Architecture:** Three new modules with single responsibilities. `src/geometry.py` owns all radar-to-geographic mathematics (currently scattered across three files that disagree with each other). `src/sweeps.py` owns VCP-aware sweep selection by inspecting file contents rather than indexing. `src/qc/` owns a fuzzy-logic gate classifier structured after the operational Hydrometeor Classification Algorithm, with hard protection rules that prevent quality control from discarding tornado debris. `ReflectivityData` becomes `SweepData` carrying co-registered polarimetric fields.

**Tech Stack:** Python 3.11+, Py-ART (`arm-pyart>=2.2.0`), NumPy, SciPy, FastAPI, pytest.

**Spec:** `docs/superpowers/specs/2026-08-22-radar-data-layer-correctness-design.md`

## Global Constraints

- **Only the Ingest Manager makes network calls.** Cross-check data retrieval (Level III, MRMS) must route through the ingest layer, never through `qc/` or `geometry.py`.
- **Every membership function parameter carries a provenance marker.** Either a published value with citation, or `arw-tuned` with reasoning and the case it was tuned against. No parameter may lack one.
- **Never invent a citation.** If a published value cannot be obtained, mark the parameter `arw-tuned` — do not attribute an invented number to a paper.
- **Do not re-tune Phase 2 suppression parameters in this plan.** Record benchmark deltas, leave the tuning alone. Re-tuning belongs to Spec 3. Changing the input field and the tuning that responds to it together destroys attribution.
- **Reflectivity sweep selection must not change behaviour.** Sweep 0 is already the correct surveillance cut. Preserving this isolation is what lets QC's effect be measured on its own.
- **Missing polarimetric arrays are absent, never low.** Treating a missing RhoHV array as zeros would classify an entire pre-2013 scan as clutter.
- **Test runner:** `.venv/Scripts/python.exe -m pytest`
- **Reference volume for all measurement tests:** `cache/KEMX/KEMX20260712_022646_V06`

---

### Task 1: Gate coordinates in `src/geometry.py`

Replaces `detection.polar_to_latlon`, which treats slant range as ground range and ignores elevation angle. Implements Proof 1 from spec §11.

**Files:**
- Create: `src/geometry.py`
- Test: `tests/unit/test_geometry.py`

**Interfaces:**
- Consumes: nothing (first task)
- Produces:
  - `gate_coordinates(azimuths: np.ndarray, ranges_m: np.ndarray, elevation_deg: float, radar_lat: float, radar_lon: float) -> tuple[np.ndarray, np.ndarray]` returning `(latitudes, longitudes)`, each shaped `(len(azimuths), len(ranges_m))`
  - `EFFECTIVE_EARTH_RADIUS_M: float`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_geometry.py
import numpy as np
import pyart
import pytest

from src.geometry import gate_coordinates

REFERENCE_VOLUME = "cache/KEMX/KEMX20260712_022646_V06"


@pytest.fixture(scope="module")
def radar():
    return pyart.io.read_nexrad_archive(REFERENCE_VOLUME)


def test_gate_coordinates_match_pyart_reference(radar):
    """Our georeferencing must agree with Py-ART's Doviak & Zrnic implementation."""
    sweep_start, sweep_end = radar.get_start_end(0)
    azimuths = radar.azimuth["data"][sweep_start:sweep_end + 1]
    ranges_m = radar.range["data"]
    elevation = float(radar.fixed_angle["data"][0])

    lat, lon = gate_coordinates(
        azimuths=azimuths,
        ranges_m=ranges_m,
        elevation_deg=elevation,
        radar_lat=float(radar.latitude["data"][0]),
        radar_lon=float(radar.longitude["data"][0]),
    )

    expected_lat = radar.gate_latitude["data"][sweep_start:sweep_end + 1]
    expected_lon = radar.gate_longitude["data"][sweep_start:sweep_end + 1]

    assert lat.shape == expected_lat.shape
    np.testing.assert_allclose(lat, expected_lat, atol=1e-6)
    np.testing.assert_allclose(lon, expected_lon, atol=1e-6)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_geometry.py::test_gate_coordinates_match_pyart_reference -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.geometry'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/geometry.py
"""Radar-to-geographic mathematics.

This is the only module that converts between radar antenna coordinates and
geographic coordinates. All beam propagation uses the 4/3 effective earth
radius model (Doviak and Zrnic, equations 2.28b and 2.28c), which is the
standard for WSR-88D data and what Py-ART implements internally.
"""

import numpy as np
from pyart.core.transforms import antenna_to_cartesian, cartesian_to_geographic_aeqd

EARTH_RADIUS_M = 6371000.0
EFFECTIVE_EARTH_RADIUS_M = EARTH_RADIUS_M * 4.0 / 3.0


def gate_coordinates(
    azimuths: np.ndarray,
    ranges_m: np.ndarray,
    elevation_deg: float,
    radar_lat: float,
    radar_lon: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Geographic coordinates of every gate in a sweep.

    Returns (latitudes, longitudes), each shaped (n_rays, n_gates).
    """
    ranges_km = np.asarray(ranges_m, dtype=float) / 1000.0
    ranges_2d, azimuths_2d = np.meshgrid(ranges_km, np.asarray(azimuths, dtype=float))
    x, y, _z = antenna_to_cartesian(ranges_2d, azimuths_2d, elevation_deg)
    lon, lat = cartesian_to_geographic_aeqd(x, y, radar_lon, radar_lat)
    return lat, lon
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_geometry.py -v`
Expected: PASS

- [ ] **Step 5: Add the regression test that documents the old error's magnitude**

The spec records the displacement of the old implementation. Lock it in so the
magnitude stays documented rather than becoming folklore.

```python
# append to tests/unit/test_geometry.py
import math

from src.detection import polar_to_latlon


def test_legacy_polar_to_latlon_displacement_is_documented(radar):
    """Records how far the pre-correction implementation was wrong.

    This test exists to document the magnitude of the error that motivated
    the correction. Delete it only when polar_to_latlon is removed.
    """
    sweep_start, sweep_end = radar.get_start_end(0)
    azimuths = radar.azimuth["data"][sweep_start:sweep_end + 1]
    ranges_m = radar.range["data"]
    gate_lat = radar.gate_latitude["data"][sweep_start:sweep_end + 1]
    gate_lon = radar.gate_longitude["data"][sweep_start:sweep_end + 1]

    expected_displacement_m = {200: 8, 400: 26, 800: 107, 1600: 546}

    for range_index, expected_m in expected_displacement_m.items():
        legacy_lat, legacy_lon = polar_to_latlon(
            float(radar.latitude["data"][0]),
            float(radar.longitude["data"][0]),
            float(azimuths[0]),
            float(ranges_m[range_index]),
        )
        reference_lat = float(gate_lat[0, range_index])
        reference_lon = float(gate_lon[0, range_index])
        delta_north_m = (legacy_lat - reference_lat) * 111195.0
        delta_east_m = (
            (legacy_lon - reference_lon) * 111195.0 * math.cos(math.radians(reference_lat))
        )
        displacement_m = math.hypot(delta_north_m, delta_east_m)
        assert displacement_m == pytest.approx(expected_m, abs=5)
```

- [ ] **Step 6: Run the full geometry test file**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_geometry.py -v`
Expected: PASS (2 tests)

- [ ] **Step 7: Commit**

```bash
git add src/geometry.py tests/unit/test_geometry.py
git commit -m "feat: add geometry module with Py-ART-conforming gate coordinates"
```

---

### Task 2: Beam height with the 4/3 effective earth radius

Corrects `sites.compute_beam_height_m`, which uses the true earth radius and
overestimates beam height by up to 1766 m at 300 km, wrongly rejecting radar
sites beyond 306 km where the correct model reaches 345 km.

**Files:**
- Modify: `src/geometry.py`
- Modify: `src/sites.py:204-228`
- Modify: `devspec/06_Radar_Site_Selection_Beam_Height.md`
- Test: `tests/unit/test_geometry.py`, `tests/unit/test_sites.py`

**Interfaces:**
- Consumes: `EFFECTIVE_EARTH_RADIUS_M` from Task 1
- Produces: `beam_height_m(range_m: float | np.ndarray, elevation_deg: float, site_alt_m: float = 0.0) -> float | np.ndarray`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/unit/test_geometry.py
from src.geometry import beam_height_m


def test_beam_height_uses_effective_earth_radius():
    """Beam height must use the 4/3 effective radius, not the true radius.

    Expected values computed from the Doviak and Zrnic exact form at 0.5
    degrees elevation with the antenna at sea level.
    """
    expected_m = {
        50_000: 583.5,
        100_000: 1461.1,
        200_000: 4098.7,
        300_000: 7911.7,
    }
    for range_m, expected in expected_m.items():
        assert beam_height_m(range_m, elevation_deg=0.5) == pytest.approx(expected, abs=1.0)


def test_beam_height_ten_km_cutoff_reaches_345km():
    """The 10 km rejection threshold must be crossed near 345 km, not 306 km."""
    assert beam_height_m(340_000, elevation_deg=0.5) < 10_000.0
    assert beam_height_m(350_000, elevation_deg=0.5) > 10_000.0


def test_beam_height_adds_site_altitude():
    at_sea_level = beam_height_m(100_000, elevation_deg=0.5)
    at_altitude = beam_height_m(100_000, elevation_deg=0.5, site_alt_m=400.0)
    assert at_altitude == pytest.approx(at_sea_level + 400.0, abs=0.1)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_geometry.py -k beam_height -v`
Expected: FAIL with `ImportError: cannot import name 'beam_height_m'`

- [ ] **Step 3: Write minimal implementation**

```python
# append to src/geometry.py

def beam_height_m(
    range_m: float | np.ndarray,
    elevation_deg: float,
    site_alt_m: float = 0.0,
) -> float | np.ndarray:
    """Height of the radar beam centre above sea level at a given slant range.

    Doviak and Zrnic equation 2.28b under the 4/3 effective earth radius model.
    """
    r = np.asarray(range_m, dtype=float)
    theta = np.radians(elevation_deg)
    R = EFFECTIVE_EARTH_RADIUS_M
    height = np.sqrt(r**2 + R**2 + 2.0 * r * R * np.sin(theta)) - R + site_alt_m
    return float(height) if np.isscalar(range_m) or height.ndim == 0 else height
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_geometry.py -k beam_height -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Write the failing test for site selection reach**

```python
# append to tests/unit/test_sites.py
from src.sites import compute_beam_height_m


def test_site_selection_reaches_345km():
    """Sites out to ~345 km must remain selectable under the corrected model."""
    assert compute_beam_height_m(distance_km=340.0, radar_elevation_m=0.0) < 10_000.0
    assert compute_beam_height_m(distance_km=350.0, radar_elevation_m=0.0) > 10_000.0


def test_compute_beam_height_delegates_to_geometry():
    """sites.py must not carry its own earth model."""
    from src.geometry import beam_height_m
    expected = beam_height_m(200_000.0, elevation_deg=0.5, site_alt_m=100.0)
    actual = compute_beam_height_m(distance_km=200.0, radar_elevation_m=100.0)
    assert actual == pytest.approx(expected, abs=0.1)
```

- [ ] **Step 6: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_sites.py -k beam_height -v`
Expected: FAIL — current implementation returns ~9681 m at 300 km and rejects beyond 306 km

- [ ] **Step 7: Delegate `sites.compute_beam_height_m` to geometry**

```python
# replace src/sites.py:204-228
def compute_beam_height_m(distance_km: float, radar_elevation_m: float) -> float:
    """Height of the lowest radar beam above sea level at a given distance.

    Delegates to src.geometry, which owns the 4/3 effective earth radius model.
    """
    from src.geometry import beam_height_m

    return beam_height_m(
        range_m=distance_km * 1000.0,
        elevation_deg=LOWEST_ELEVATION_DEG,
        site_alt_m=radar_elevation_m,
    )
```

- [ ] **Step 8: Run the sites test suite**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_sites.py -v`
Expected: PASS. If existing tests assert old beam heights, update their expected
values to the corrected model and note the change in the commit message.

- [ ] **Step 9: Correct the devspec**

Replace the Beam Height Formula section of
`devspec/06_Radar_Site_Selection_Beam_Height.md`:

```markdown
## Beam Height Formula

Doviak and Zrnic equation 2.28b, under the 4/3 effective earth radius model:

beam_height = sqrt(r^2 + R^2 + 2*r*R*sin(elevation)) - R + radar_elevation

where:
  r = slant range from the radar
  R = effective earth radius = 4/3 * 6371 km = 8494.7 km
  elevation = beam elevation angle, lowest tilt ~= 0.5 degrees

The effective radius accounts for standard atmospheric refraction bending the
beam downward relative to a straight line. Using the true earth radius of
6371 km overestimates beam height by up to 1766 m at 300 km and wrongly
rejects radar sites beyond 306 km.
```

- [ ] **Step 10: Commit**

```bash
git add src/geometry.py src/sites.py tests/unit/test_geometry.py tests/unit/test_sites.py devspec/06_Radar_Site_Selection_Beam_Height.md
git commit -m "fix: use 4/3 effective earth radius for beam height

Site selection now reaches 345 km rather than 306 km. Corrects devspec/06,
which specified the true earth radius."
```

---

### Task 3: Gate areas from ground range

`detection._range_bin_areas_km2` computes area from slant range and infers
azimuth spacing from the first two rays. Moves to `geometry.py` using ground
range and actual per-ray spacing.

**Files:**
- Modify: `src/geometry.py`
- Modify: `src/detection.py:89-110`
- Test: `tests/unit/test_geometry.py`

**Interfaces:**
- Consumes: `EFFECTIVE_EARTH_RADIUS_M`, `beam_height_m` from Tasks 1-2
- Produces:
  - `ground_range_m(slant_range_m, elevation_deg) -> np.ndarray`
  - `gate_areas_km2(azimuths: np.ndarray, ranges_m: np.ndarray, elevation_deg: float) -> np.ndarray` returning per-range-bin areas shaped `(len(ranges_m),)`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/unit/test_geometry.py
from src.geometry import gate_areas_km2, ground_range_m


def test_ground_range_is_shorter_than_slant_range():
    slant = np.array([100_000.0, 300_000.0])
    ground = ground_range_m(slant, elevation_deg=0.5)
    assert np.all(ground < slant)
    # At low elevation the difference is small but non-zero.
    assert ground[0] == pytest.approx(100_000.0, rel=1e-3)


def test_gate_areas_grow_with_range():
    azimuths = np.linspace(0.0, 359.5, 720)
    ranges_m = np.arange(2125.0, 460_000.0, 250.0)
    areas = gate_areas_km2(azimuths, ranges_m, elevation_deg=0.5)
    assert areas.shape == ranges_m.shape
    assert np.all(np.diff(areas) > 0)


def test_gate_areas_use_ground_range_not_slant():
    """Areas must be computed from ground range, so they are smaller than the
    slant-range calculation the legacy code used."""
    azimuths = np.linspace(0.0, 359.5, 720)
    ranges_m = np.array([300_000.0, 300_250.0])
    areas = gate_areas_km2(azimuths, ranges_m, elevation_deg=0.5)
    az_spacing_rad = np.radians(0.5)
    slant_area_km2 = (300_000.0 * az_spacing_rad * 250.0) / 1e6
    assert areas[0] < slant_area_km2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_geometry.py -k "ground_range or gate_areas" -v`
Expected: FAIL with `ImportError: cannot import name 'gate_areas_km2'`

- [ ] **Step 3: Write minimal implementation**

```python
# append to src/geometry.py

def ground_range_m(
    slant_range_m: float | np.ndarray,
    elevation_deg: float,
) -> np.ndarray:
    """Great-circle distance along the ground beneath each gate.

    Doviak and Zrnic equation 2.28c under the 4/3 effective earth radius model.
    """
    r = np.asarray(slant_range_m, dtype=float)
    theta = np.radians(elevation_deg)
    R = EFFECTIVE_EARTH_RADIUS_M
    height = np.sqrt(r**2 + R**2 + 2.0 * r * R * np.sin(theta)) - R
    return R * np.arcsin(r * np.cos(theta) / (R + height))


def gate_areas_km2(
    azimuths: np.ndarray,
    ranges_m: np.ndarray,
    elevation_deg: float,
) -> np.ndarray:
    """Ground-projected area of one gate at each range bin, in square km.

    Azimuth spacing is the median gap between consecutive rays, which is robust
    to the small irregularities in real NEXRAD azimuth sequences. The legacy
    implementation used the gap between the first two rays only.
    """
    ranges_m = np.asarray(ranges_m, dtype=float)
    if len(ranges_m) < 2 or len(azimuths) < 2:
        return np.zeros_like(ranges_m, dtype=float)

    range_spacing_m = float(np.median(np.abs(np.diff(ranges_m))))
    azimuth_gaps = np.abs(np.diff(np.unwrap(np.asarray(azimuths, dtype=float), period=360.0)))
    az_spacing_rad = np.radians(float(np.median(azimuth_gaps)))

    ground = ground_range_m(ranges_m, elevation_deg)
    return (ground * az_spacing_rad * range_spacing_m) / 1e6
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_geometry.py -v`
Expected: PASS

- [ ] **Step 5: Delegate `detection._range_bin_areas_km2` to geometry**

`detection.py` currently has no elevation angle available in
`_range_bin_areas_km2`. Add it as a keyword argument with a default of 0.5 so
existing call sites keep working, and thread the real elevation through in
Task 5 when `SweepData` carries it.

```python
# replace src/detection.py:104-110
def _range_bin_areas_km2(
    azimuths: np.ndarray,
    ranges_m: np.ndarray,
    elevation_deg: float = 0.5,
) -> np.ndarray:
    from src.geometry import gate_areas_km2

    return gate_areas_km2(azimuths, ranges_m, elevation_deg)
```

Delete `_compute_pixel_area_km2` (`detection.py:89-101`) — grep confirms it has
no callers.

- [ ] **Step 6: Verify `_compute_pixel_area_km2` really is unused before deleting**

Run: `grep -rn "_compute_pixel_area_km2" src/ tests/ scripts/`
Expected: no matches outside `src/detection.py`. If there are matches, migrate
them to `gate_areas_km2` in this step rather than deleting.

- [ ] **Step 7: Run the full suite**

Run: `.venv/Scripts/python.exe -m pytest -v`
Expected: PASS. Object areas shrink slightly at long range; update any test
asserting exact area values and note it in the commit.

- [ ] **Step 8: Commit**

```bash
git add src/geometry.py src/detection.py tests/unit/test_geometry.py
git commit -m "fix: compute gate areas from ground range with median azimuth spacing"
```

---

### Task 4: VCP-aware sweep selection in `src/sweeps.py`

`parser.extract_velocity` selects velocity sweeps by index `range(3)`. In split-cut
VCPs two of those three carry no velocity data. Selection must inspect file
contents.

**Files:**
- Create: `src/sweeps.py`
- Test: `tests/unit/test_sweeps.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `select_reflectivity_sweep(radar) -> int`
  - `select_velocity_sweeps(radar, max_sweeps: int = 3) -> list[int]`
  - `sweep_field_coverage(radar, sweep_index: int, field: str) -> float`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_sweeps.py
import numpy as np
import pytest

from src.sweeps import select_reflectivity_sweep, select_velocity_sweeps


class FakeRadar:
    """Minimal stand-in exposing only what sweep selection reads.

    Real NEXRAD split-cut volumes interleave surveillance cuts (reflectivity,
    low Nyquist, no usable velocity) with Doppler cuts (velocity, high Nyquist)
    at the same elevation. Tests must reproduce that structure — the previous
    synthetic fixtures gave every sweep velocity, which is why the defect
    survived.
    """

    def __init__(self, fixed_angles, reflectivity_valid, velocity_valid, rays_per_sweep=8, gates=10):
        self.nsweeps = len(fixed_angles)
        self.fixed_angle = {"data": np.array(fixed_angles, dtype=float)}
        self._rays = rays_per_sweep
        self._gates = gates
        self.fields = {
            "reflectivity": {"data": self._build(reflectivity_valid)},
            "velocity": {"data": self._build(velocity_valid)},
        }

    def _build(self, valid_flags):
        rows = []
        for is_valid in valid_flags:
            block = np.full((self._rays, self._gates), 1.0 if is_valid else np.nan)
            rows.append(block)
        return np.ma.masked_invalid(np.vstack(rows))

    def get_start_end(self, sweep_index):
        start = sweep_index * self._rays
        return start, start + self._rays - 1


def make_split_cut_radar():
    """Mirrors KEMX20260712_022646_V06: paired cuts, velocity only on odd sweeps."""
    return FakeRadar(
        fixed_angles=[0.48, 0.48, 0.88, 0.88, 1.27, 1.27, 1.80],
        reflectivity_valid=[True, True, True, True, True, True, True],
        velocity_valid=[False, True, False, True, False, True, True],
    )


def test_selects_surveillance_cut_for_reflectivity():
    radar = make_split_cut_radar()
    assert select_reflectivity_sweep(radar) == 0


def test_skips_empty_velocity_sweeps():
    """The defect: index-based selection returned [0, 1, 2], two of them empty."""
    radar = make_split_cut_radar()
    assert select_velocity_sweeps(radar, max_sweeps=3) == [1, 3, 5]


def test_legacy_volume_without_split_cuts():
    radar = FakeRadar(
        fixed_angles=[0.5, 1.5, 2.4],
        reflectivity_valid=[True, True, True],
        velocity_valid=[True, True, True],
    )
    assert select_reflectivity_sweep(radar) == 0
    assert select_velocity_sweeps(radar, max_sweeps=3) == [0, 1, 2]


def test_volume_with_no_velocity_at_all():
    radar = FakeRadar(
        fixed_angles=[0.5, 1.5],
        reflectivity_valid=[True, True],
        velocity_valid=[False, False],
    )
    assert select_velocity_sweeps(radar) == []


def test_respects_max_sweeps():
    radar = make_split_cut_radar()
    assert select_velocity_sweeps(radar, max_sweeps=2) == [1, 3]


def test_one_sweep_per_elevation_group():
    """Both cuts at 0.88 carry velocity — only the better one may be returned."""
    radar = FakeRadar(
        fixed_angles=[0.48, 0.48, 0.88, 0.88],
        reflectivity_valid=[True, True, True, True],
        velocity_valid=[False, True, True, True],
    )
    selected = select_velocity_sweeps(radar, max_sweeps=3)
    assert len(selected) == 2
    assert selected[0] == 1
    assert selected[1] in (2, 3)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_sweeps.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.sweeps'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/sweeps.py
"""VCP-aware sweep selection.

NEXRAD volume coverage patterns use split cuts at low elevations: each tilt is
scanned twice, once as a surveillance cut (long PRF, reflectivity, unambiguous
range to 460 km, low Nyquist) and once as a Doppler cut (short PRF, velocity,
range to 300 km, high Nyquist). Selecting sweeps by index silently picks
surveillance cuts for velocity, which carry no velocity data.

Selection here inspects what each sweep actually contains, so it is
VCP-independent and survives NWS scan strategy changes.
"""

import numpy as np

ELEVATION_GROUP_TOLERANCE_DEG = 0.05


def sweep_field_coverage(radar, sweep_index: int, field: str) -> float:
    """Fraction of gates in a sweep carrying valid data for a field."""
    if field not in radar.fields:
        return 0.0
    sweep_start, sweep_end = radar.get_start_end(sweep_index)
    data = radar.fields[field]["data"][sweep_start:sweep_end + 1]
    if data.size == 0:
        return 0.0
    valid = ~np.ma.getmaskarray(data)
    if not np.ma.isMaskedArray(data):
        valid = ~np.isnan(np.asarray(data, dtype=float))
    return float(np.count_nonzero(valid)) / float(data.size)


def _elevation_groups(radar) -> list[list[int]]:
    """Group sweep indices by elevation, collapsing split cuts into one group."""
    angles = np.asarray(radar.fixed_angle["data"], dtype=float)
    order = sorted(range(radar.nsweeps), key=lambda i: (angles[i], i))
    groups: list[list[int]] = []
    for sweep_index in order:
        if groups and abs(angles[sweep_index] - angles[groups[-1][0]]) <= ELEVATION_GROUP_TOLERANCE_DEG:
            groups[-1].append(sweep_index)
        else:
            groups.append([sweep_index])
    return groups


def _best_in_group(radar, group: list[int], field: str) -> int | None:
    """Highest-coverage sweep for a field within one elevation group."""
    scored = [(sweep_field_coverage(radar, i, field), -i, i) for i in group]
    coverage, _, best_index = max(scored)
    return best_index if coverage > 0.0 else None


def select_reflectivity_sweep(radar) -> int:
    """Index of the sweep to use for reflectivity: lowest tilt, best coverage."""
    groups = _elevation_groups(radar)
    for group in groups:
        best = _best_in_group(radar, group, "reflectivity")
        if best is not None:
            return best
    return 0


def select_velocity_sweeps(radar, max_sweeps: int = 3) -> list[int]:
    """Indices of the lowest sweeps carrying usable velocity, one per elevation."""
    selected: list[int] = []
    for group in _elevation_groups(radar):
        if len(selected) >= max_sweeps:
            break
        best = _best_in_group(radar, group, "velocity")
        if best is not None:
            selected.append(best)
    return selected
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_sweeps.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Add a live test against the reference volume**

```python
# append to tests/unit/test_sweeps.py
import pyart

REFERENCE_VOLUME = "cache/KEMX/KEMX20260712_022646_V06"


@pytest.fixture(scope="module")
def real_radar():
    return pyart.io.read_nexrad_archive(REFERENCE_VOLUME)


def test_real_volume_selects_doppler_cuts(real_radar):
    """On the reference volume, velocity lives on sweeps 1, 3, 5."""
    assert select_velocity_sweeps(real_radar, max_sweeps=3) == [1, 3, 5]


def test_real_volume_selected_sweeps_all_carry_velocity(real_radar):
    from src.sweeps import sweep_field_coverage

    for sweep_index in select_velocity_sweeps(real_radar, max_sweeps=3):
        assert sweep_field_coverage(real_radar, sweep_index, "velocity") > 0.0


def test_real_volume_reflectivity_sweep_unchanged(real_radar):
    """Sweep 0 was already correct. This must not change, so QC's effect on
    detection can be measured in isolation."""
    assert select_reflectivity_sweep(real_radar) == 0
```

- [ ] **Step 6: Run the live tests**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_sweeps.py -v`
Expected: PASS (9 tests)

- [ ] **Step 7: Commit**

```bash
git add src/sweeps.py tests/unit/test_sweeps.py
git commit -m "feat: add VCP-aware sweep selection by content inspection

Selecting velocity sweeps by index returns surveillance cuts, which carry no
velocity data. Selection now inspects per-sweep field coverage."
```

---

### Task 5: `SweepData` — rename and polarimetric field extraction

`ReflectivityData` becomes `SweepData` carrying co-registered polarimetric
fields. The eight consumer modules use almost exclusively `.reflectivity`,
`.timestamp` and the polar axes, so the change is mechanical.

**Files:**
- Modify: `src/parser.py`
- Modify: `src/buffer.py:6,17`, `src/preprocess.py:6,67-74`, `src/server.py:124-139`, `src/map_layer.py:121,171,208,238`, `src/tracking/association.py:155-156,200-201`, `src/tracking/motion_field.py:196-197,255-256`
- Test: `tests/unit/test_parser.py`

**Interfaces:**
- Consumes: `select_reflectivity_sweep` from Task 4
- Produces:
  - `SweepData` dataclass with fields per spec §6
  - `extract_sweep_data(radar) -> SweepData`
  - `ReflectivityData` name is removed entirely

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_parser.py — add to existing file
import numpy as np
import pyart
import pytest

from src.parser import SweepData, extract_sweep_data

REFERENCE_VOLUME = "cache/KEMX/KEMX20260712_022646_V06"


@pytest.fixture(scope="module")
def radar():
    return pyart.io.read_nexrad_archive(REFERENCE_VOLUME)


def test_extract_sweep_data_carries_polarimetric_fields(radar):
    sweep = extract_sweep_data(radar)
    assert isinstance(sweep, SweepData)
    assert sweep.reflectivity.shape == sweep.rhohv.shape
    assert sweep.reflectivity.shape == sweep.zdr.shape
    assert sweep.reflectivity.shape[0] == len(sweep.azimuths)
    assert sweep.reflectivity.shape[1] == len(sweep.ranges_m)


def test_extract_sweep_data_uses_masked_fill(radar):
    sweep = extract_sweep_data(radar)
    assert np.isnan(sweep.reflectivity).any()
    assert not np.ma.isMaskedArray(sweep.reflectivity)


def test_extract_sweep_data_records_site_altitude(radar):
    sweep = extract_sweep_data(radar)
    assert isinstance(sweep.radar_alt_m, float)


def test_extract_sweep_data_handles_absent_polarimetric_fields():
    """Pre-2013 volumes have no RhoHV. Absent must mean None, never zeros."""

    class NoDualPolRadar:
        nsweeps = 1
        fixed_angle = {"data": np.array([0.5])}
        latitude = {"data": np.array([35.0])}
        longitude = {"data": np.array([-97.0])}
        altitude = {"data": np.array([390.0])}
        azimuth = {"data": np.linspace(0, 359.5, 8)}
        range = {"data": np.arange(2125.0, 4625.0, 250.0)}
        time = {"units": "seconds since 2011-05-01T00:00:00Z"}
        fields = {"reflectivity": {"data": np.ma.masked_invalid(np.full((8, 10), 25.0))}}
        instrument_parameters = None

        def get_start_end(self, sweep_index):
            return 0, 7

    sweep = extract_sweep_data(NoDualPolRadar())
    assert sweep.rhohv is None
    assert sweep.zdr is None
    assert sweep.reflectivity is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_parser.py -k sweep_data -v`
Expected: FAIL with `ImportError: cannot import name 'SweepData'`

- [ ] **Step 3: Write the new dataclass and extractor**

```python
# replace the ReflectivityData dataclass and extract_reflectivity_from_radar
# in src/parser.py
from dataclasses import dataclass, field as dataclass_field

import numpy as np
import pyart

from src.sweeps import select_reflectivity_sweep

NEXRAD_FIELD_NAMES = {
    "reflectivity": "reflectivity",
    "velocity": "velocity",
    "rhohv": "cross_correlation_ratio",
    "zdr": "differential_reflectivity",
    "phidp": "differential_phase",
    "spectrum_width": "spectrum_width",
    "clutter_power_removed": "clutter_filter_power_removed",
}


@dataclass
class SweepData:
    """One radar sweep with all co-registered fields available for it."""

    reflectivity: np.ndarray
    azimuths: np.ndarray
    ranges_m: np.ndarray
    elevation_angle: float
    elevation_angles: list[float]
    radar_lat: float
    radar_lon: float
    radar_alt_m: float
    timestamp: str

    velocity: np.ndarray | None = None
    rhohv: np.ndarray | None = None
    zdr: np.ndarray | None = None
    phidp: np.ndarray | None = None
    spectrum_width: np.ndarray | None = None
    clutter_power_removed: np.ndarray | None = None
    nyquist_velocity: float | None = None
    gate_classification: np.ndarray | None = None


def _extract_field(radar, nexrad_name: str, sweep_start: int, sweep_end: int) -> np.ndarray | None:
    """Pull one field for one sweep, or None if the volume does not carry it."""
    if nexrad_name not in radar.fields:
        return None
    data = radar.fields[nexrad_name]["data"][sweep_start:sweep_end + 1]
    if hasattr(data, "filled"):
        data = data.filled(np.nan)
    return np.asarray(data, dtype=float)


def extract_sweep_data(radar) -> SweepData:
    """Extract the reflectivity sweep with every co-registered field it carries."""
    sweep_index = select_reflectivity_sweep(radar)
    sweep_start, sweep_end = radar.get_start_end(sweep_index)

    fields = {
        key: _extract_field(radar, nexrad_name, sweep_start, sweep_end)
        for key, nexrad_name in NEXRAD_FIELD_NAMES.items()
    }

    nyquist = None
    instrument = getattr(radar, "instrument_parameters", None)
    if instrument and "nyquist_velocity" in instrument:
        nyquist = float(instrument["nyquist_velocity"]["data"][sweep_start])

    elevation_angles = sorted(set(np.round(radar.fixed_angle["data"], 1)))

    return SweepData(
        reflectivity=fields["reflectivity"],
        velocity=fields["velocity"],
        rhohv=fields["rhohv"],
        zdr=fields["zdr"],
        phidp=fields["phidp"],
        spectrum_width=fields["spectrum_width"],
        clutter_power_removed=fields["clutter_power_removed"],
        azimuths=np.asarray(radar.azimuth["data"][sweep_start:sweep_end + 1], dtype=float),
        ranges_m=np.asarray(radar.range["data"], dtype=float),
        elevation_angle=float(radar.fixed_angle["data"][sweep_index]),
        elevation_angles=[float(a) for a in elevation_angles],
        radar_lat=float(radar.latitude["data"][0]),
        radar_lon=float(radar.longitude["data"][0]),
        radar_alt_m=float(radar.altitude["data"][0]),
        timestamp=str(radar.time["units"]).replace("seconds since ", ""),
        nyquist_velocity=nyquist,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_parser.py -k sweep_data -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Rename every consumer**

Rename `ReflectivityData` → `SweepData` and `extract_reflectivity_from_radar` →
`extract_sweep_data` across the codebase. `BufferedScan.reflectivity_data` keeps
its attribute name — it still holds the reflectivity sweep, and renaming it
would touch far more code for no clarity gain.

Run: `grep -rn "ReflectivityData\|extract_reflectivity" src/ tests/ scripts/`

Update each hit. Delete the old `extract_reflectivity` and
`extract_reflectivity_from_radar` functions once no callers remain.

- [ ] **Step 6: Thread the real elevation angle into gate areas**

`detection.compute_object_properties` and `detection.detect_objects_with_grid`
gained an `elevation_deg` default in Task 3. Pass the real value from
`SweepData.elevation_angle` at the call site in `src/server.py:127-133`:

```python
    result = detect_objects_with_grid(
        reflectivity=ref_data.reflectivity,
        azimuths=ref_data.azimuths,
        ranges_m=ref_data.ranges_m,
        radar_lat=ref_data.radar_lat,
        radar_lon=ref_data.radar_lon,
        elevation_deg=ref_data.elevation_angle,
    )
```

Add `elevation_deg: float = 0.5` to `detect_objects_with_grid`,
`detect_objects` and `compute_object_properties`, threading it to
`_range_bin_areas_km2`.

- [ ] **Step 7: Run the full suite**

Run: `.venv/Scripts/python.exe -m pytest -v`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add -A src/ tests/
git commit -m "refactor: ReflectivityData becomes SweepData with polarimetric fields

Carries RhoHV, ZDR, PhiDP, spectrum width and clutter filter diagnostics
alongside reflectivity. Absent fields are None, never zeros."
```

---

### Task 6: Velocity extraction uses selected Doppler sweeps

Fixes the confirmed defect. Also rewrites the two synthetic tests that passed
while the feature was broken.

**Files:**
- Modify: `src/parser.py:69-109` (`extract_velocity`)
- Modify: `tests/unit/test_velocity.py:60-73,139-153`
- Test: `tests/unit/test_parser.py`

**Interfaces:**
- Consumes: `select_velocity_sweeps` from Task 4
- Produces: `extract_velocity(radar, max_sweeps: int = 3) -> VelocityData | None` — unchanged signature, corrected selection

- [ ] **Step 1: Write the failing test**

```python
# append to tests/unit/test_parser.py
from src.parser import extract_velocity


def test_extract_velocity_skips_empty_surveillance_cuts(radar):
    """Every returned sweep must carry actual velocity data."""
    velocity_data = extract_velocity(radar, max_sweeps=3)
    assert velocity_data is not None
    assert len(velocity_data.sweeps) == 3
    for sweep in velocity_data.sweeps:
        finite_fraction = np.count_nonzero(~np.isnan(sweep.velocity)) / sweep.velocity.size
        assert finite_fraction > 0.0, "selected a sweep with no velocity data"


def test_extract_velocity_returns_distinct_elevations(radar):
    velocity_data = extract_velocity(radar, max_sweeps=3)
    elevations = [round(s.elevation_angle, 2) for s in velocity_data.sweeps]
    assert len(set(elevations)) == 3


def test_extract_velocity_uses_doppler_nyquist(radar):
    """Doppler cuts have a high Nyquist; surveillance cuts about 8 m/s."""
    velocity_data = extract_velocity(radar, max_sweeps=3)
    for sweep in velocity_data.sweeps:
        assert sweep.nyquist_velocity > 20.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_parser.py -k extract_velocity -v`
Expected: FAIL — sweeps 0 and 2 have zero finite velocity gates, and their
Nyquist is 8.3 m/s

- [ ] **Step 3: Rewrite `extract_velocity`**

```python
# replace src/parser.py:69-109
def extract_velocity(radar, max_sweeps: int = 3) -> VelocityData | None:
    """Extract velocity from the lowest sweeps that actually carry it.

    Split-cut VCPs pair a surveillance cut and a Doppler cut at each low
    elevation. Only the Doppler cut carries usable velocity, so sweeps are
    chosen by inspection rather than by index.

    Dealiasing runs after selection, so surveillance cuts with no velocity are
    not processed.
    """
    from src.sweeps import select_velocity_sweeps

    if "velocity" not in radar.fields:
        return None

    sweep_indices = select_velocity_sweeps(radar, max_sweeps=max_sweeps)
    if not sweep_indices:
        return None

    try:
        dealiased = pyart.correct.dealias_region_based(radar, field="velocity")
        radar.add_field("dealiased_velocity", dealiased, replace_existing=True)
        velocity_field = "dealiased_velocity"
    except Exception:
        velocity_field = "velocity"

    sweeps: list[VelocitySweep] = []
    for sweep_index in sweep_indices:
        sweep_start, sweep_end = radar.get_start_end(sweep_index)
        velocity = radar.fields[velocity_field]["data"][sweep_start:sweep_end + 1]
        if hasattr(velocity, "filled"):
            velocity = velocity.filled(np.nan)

        sweeps.append(VelocitySweep(
            velocity=np.asarray(velocity, dtype=float),
            azimuths=np.asarray(radar.azimuth["data"][sweep_start:sweep_end + 1], dtype=float),
            ranges_m=np.asarray(radar.range["data"], dtype=float),
            elevation_angle=float(radar.fixed_angle["data"][sweep_index]),
            nyquist_velocity=float(
                radar.instrument_parameters["nyquist_velocity"]["data"][sweep_start]
            ),
        ))

    return VelocityData(
        sweeps=sweeps,
        radar_lat=float(radar.latitude["data"][0]),
        radar_lon=float(radar.longitude["data"][0]),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_parser.py -k extract_velocity -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Rewrite the two misleading synthetic tests**

Replace `tests/unit/test_velocity.py:60-73` and `:139-153`. The originals gave
every sweep velocity data, a VCP structure that does not exist, so they passed
while production was broken.

```python
# replace test_detect_velocity_regions_multi_sweep_increases_sweep_count
def test_detect_velocity_regions_multi_sweep_increases_sweep_count():
    """Sweep count rises across Doppler cuts at distinct elevations.

    Elevations here match the Doppler cuts of a real split-cut VCP
    (0.48, 0.88, 1.27) rather than the invented 0.5/1.5 pair used previously.
    """
    grids = []
    for peak in (-20.0, -22.0, -21.0):
        grid = np.full((360, 500), np.nan)
        grid[50:70, 100:130] = peak
        grids.append(grid)
    vel_data = _make_velocity_data([
        _make_sweep(grids[0], elevation=0.48),
        _make_sweep(grids[1], elevation=0.88),
        _make_sweep(grids[2], elevation=1.27),
    ])
    regions = detect_velocity_regions(vel_data)
    inbound = [r for r in regions if r.region_type == "inbound"]
    assert len(inbound) == 1
    assert inbound[0].sweep_count == 3
    assert inbound[0].elevation_angles == [0.48, 0.88, 1.27]
```

Apply the same three-Doppler-cut structure to
`test_detect_rotation_multi_sweep_increases_sweep_count`, asserting
`sweep_count == 3`.

- [ ] **Step 6: Add a live test that multi-sweep confirmation actually engages**

```python
# append to tests/unit/test_velocity.py
import pyart
import pytest

from src.parser import extract_velocity

REFERENCE_VOLUME = "cache/KEMX/KEMX20260712_022646_V06"


def test_sweep_count_can_exceed_one_on_real_data():
    """Regression for the split-cut defect.

    Before the fix, two of three velocity sweeps were empty, so sweep_count was
    structurally pinned at 1 and multi-sweep confirmation never engaged.
    """
    radar = pyart.io.read_nexrad_archive(REFERENCE_VOLUME)
    vel_data = extract_velocity(radar, max_sweeps=3)
    regions = detect_velocity_regions(vel_data)
    if not regions:
        pytest.skip("reference volume has no velocity regions above threshold")
    assert max(r.sweep_count for r in regions) > 1
```

- [ ] **Step 7: Run the velocity suite**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_velocity.py -v`
Expected: PASS. If `test_sweep_count_can_exceed_one_on_real_data` skips, choose
a cached volume with active convection and use that instead — the test must
actually assert, not skip, before this task is complete.

- [ ] **Step 8: Record the rotation baseline change**

Run the existing replay harness against a cached window and record rotation
signature counts, strengths and `sweep_count` distribution before and after.
Save to `docs/test_reports/2026-08-23-sweep-selection-rotation-delta.md`.

This is spec §11 "rotation changes measured separately". It is evidence, not
decoration — Task 18 compares against it.

- [ ] **Step 9: Commit**

```bash
git add src/parser.py tests/unit/test_velocity.py tests/unit/test_parser.py docs/test_reports/
git commit -m "fix: select Doppler cuts for velocity instead of sweep indices

Two of three velocity sweeps were surveillance cuts carrying no data, so
sweep_count was pinned at 1 and multi-sweep rotation confirmation could never
engage. Rewrites the synthetic tests that passed while this was broken."
```

---

### Task 7: Texture measures in `src/qc/texture.py`

Non-meteorological echo is spatially incoherent; weather is smooth. Texture
needs no dual-pol, which is what keeps the pre-2013 degraded mode useful.

**Files:**
- Create: `src/qc/__init__.py`, `src/qc/texture.py`
- Test: `tests/unit/test_qc_texture.py`

**Interfaces:**
- Consumes: nothing
- Produces: `local_standard_deviation(field: np.ndarray, window: int = 3) -> np.ndarray`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_qc_texture.py
import numpy as np

from src.qc.texture import local_standard_deviation


def test_smooth_field_has_low_texture():
    field = np.full((20, 20), 30.0)
    texture = local_standard_deviation(field)
    assert np.nanmax(texture) < 1e-6


def test_noisy_field_has_high_texture():
    rng = np.random.default_rng(0)
    field = rng.normal(loc=30.0, scale=15.0, size=(20, 20))
    texture = local_standard_deviation(field)
    assert np.nanmean(texture) > 5.0


def test_gradient_field_has_moderate_texture():
    field = np.tile(np.arange(20, dtype=float), (20, 1))
    texture = local_standard_deviation(field)
    assert 0.5 < np.nanmean(texture) < 2.0


def test_nan_gates_stay_nan():
    field = np.full((10, 10), 30.0)
    field[5, 5] = np.nan
    texture = local_standard_deviation(field)
    assert np.isnan(texture[5, 5])


def test_all_nan_field_returns_all_nan():
    field = np.full((10, 10), np.nan)
    texture = local_standard_deviation(field)
    assert np.all(np.isnan(texture))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_qc_texture.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.qc'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/qc/__init__.py
"""Polarimetric quality control for radar sweeps."""
```

```python
# src/qc/texture.py
"""Local texture measures.

Meteorological echo varies smoothly from gate to gate. Ground clutter,
biological scatterers and interference do not. Local standard deviation
captures that difference and requires no polarimetric data, so it remains
available on pre-2013 volumes.
"""

import numpy as np
from scipy.ndimage import uniform_filter


def local_standard_deviation(field: np.ndarray, window: int = 3) -> np.ndarray:
    """Standard deviation of a field within a square window around each gate.

    NaN gates are excluded from their neighbours' statistics and stay NaN in
    the output.
    """
    values = np.asarray(field, dtype=float)
    valid = np.isfinite(values)
    filled = np.where(valid, values, 0.0)

    count = uniform_filter(valid.astype(float), size=window, mode="nearest")
    mean = uniform_filter(filled, size=window, mode="nearest")
    mean_square = uniform_filter(filled**2, size=window, mode="nearest")

    with np.errstate(invalid="ignore", divide="ignore"):
        true_mean = mean / count
        true_mean_square = mean_square / count
        variance = np.clip(true_mean_square - true_mean**2, 0.0, None)
        deviation = np.sqrt(variance)

    return np.where(valid & (count > 0), deviation, np.nan)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_qc_texture.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add src/qc/ tests/unit/test_qc_texture.py
git commit -m "feat: add local texture measures for quality control"
```

---

### Task 8: Membership functions and the parameter table

Trapezoidal fuzzy membership functions with mandatory provenance markers, per
spec §8 and §10.

**Files:**
- Create: `src/qc/membership.py`, `src/qc/parameters.py`
- Test: `tests/unit/test_qc_membership.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `trapezoid(values: np.ndarray, x1: float, x2: float, x3: float, x4: float) -> np.ndarray`
  - `MembershipParameter` dataclass with `x1, x2, x3, x4, weight, provenance`
  - `CLASS_PARAMETERS: dict[str, dict[str, MembershipParameter]]`
  - `GateClass` string constants

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_qc_membership.py
import numpy as np
import pytest

from src.qc.membership import trapezoid
from src.qc.parameters import CLASS_PARAMETERS, GateClass


def test_trapezoid_plateau_is_one():
    assert trapezoid(np.array([5.0]), 0.0, 2.0, 8.0, 10.0)[0] == pytest.approx(1.0)


def test_trapezoid_outside_support_is_zero():
    values = np.array([-1.0, 11.0])
    assert np.allclose(trapezoid(values, 0.0, 2.0, 8.0, 10.0), 0.0)


def test_trapezoid_ramps_linearly():
    assert trapezoid(np.array([1.0]), 0.0, 2.0, 8.0, 10.0)[0] == pytest.approx(0.5)
    assert trapezoid(np.array([9.0]), 0.0, 2.0, 8.0, 10.0)[0] == pytest.approx(0.5)


def test_trapezoid_handles_nan():
    assert np.isnan(trapezoid(np.array([np.nan]), 0.0, 2.0, 8.0, 10.0)[0])


def test_trapezoid_supports_open_left_edge():
    """x1 == x2 gives a hard left edge rather than a ramp."""
    assert trapezoid(np.array([0.0]), 0.0, 0.0, 8.0, 10.0)[0] == pytest.approx(1.0)


def test_every_parameter_has_provenance():
    """Global constraint: no membership parameter may lack a provenance marker."""
    for class_name, variables in CLASS_PARAMETERS.items():
        for variable_name, parameter in variables.items():
            assert parameter.provenance, f"{class_name}.{variable_name} lacks provenance"


def test_all_expected_classes_present():
    assert set(CLASS_PARAMETERS) == {
        GateClass.PRECIPITATION,
        GateClass.GROUND_CLUTTER,
        GateClass.BIOLOGICAL,
        GateClass.HAIL,
        GateClass.DEBRIS,
    }


def test_melting_layer_classes_are_absent():
    """Classes needing melting-layer height must not be present — spec 8."""
    forbidden = {"wet_snow", "dry_snow", "ice_crystals", "graupel", "big_drops"}
    assert forbidden.isdisjoint(CLASS_PARAMETERS)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_qc_membership.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.qc.membership'`

- [ ] **Step 3: Write the trapezoid function**

```python
# src/qc/membership.py
"""Trapezoidal fuzzy membership functions.

Each function rises from 0 at x1 to 1 at x2, holds 1 through x3, and falls to
0 at x4. Setting x1 == x2 or x3 == x4 produces a hard edge instead of a ramp.
"""

import numpy as np


def trapezoid(
    values: np.ndarray,
    x1: float,
    x2: float,
    x3: float,
    x4: float,
) -> np.ndarray:
    """Membership of each value in a trapezoidal set. NaN in, NaN out."""
    v = np.asarray(values, dtype=float)
    result = np.zeros_like(v, dtype=float)

    rising = (v > x1) & (v < x2)
    if x2 > x1:
        result[rising] = (v[rising] - x1) / (x2 - x1)

    result[(v >= x2) & (v <= x3)] = 1.0

    falling = (v > x3) & (v < x4)
    if x4 > x3:
        result[falling] = (x4 - v[falling]) / (x4 - x3)

    return np.where(np.isfinite(v), result, np.nan)
```

- [ ] **Step 4: Write the parameter table**

Initial values are derived from the class discriminator table in spec §8 and are
marked `arw-tuned-initial`. Task 19 replaces those obtainable from Park et al.
2009 and updates their provenance. **Do not label any value `park2009` until it
has actually been read from the paper.**

```python
# src/qc/parameters.py
"""Membership function parameters, with mandatory provenance.

Provenance values:
  "park2009"           value taken from Park, Ryzhkov, Zrnic and Kim,
                       Weather and Forecasting 2009
  "arw-tuned-initial"  derived from the class discriminator table in the design
                       spec, not yet validated against live cases
  "arw-tuned"          set by ARW against a named validation case, recorded in
                       the reason field
"""

from dataclasses import dataclass


class GateClass:
    PRECIPITATION = "precipitation"
    GROUND_CLUTTER = "ground_clutter"
    BIOLOGICAL = "biological"
    HAIL = "hail"
    DEBRIS = "debris"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class MembershipParameter:
    x1: float
    x2: float
    x3: float
    x4: float
    weight: float
    provenance: str
    reason: str = ""


_INITIAL = "arw-tuned-initial"
_SPEC = "derived from design spec section 8 class table"

CLASS_PARAMETERS: dict[str, dict[str, MembershipParameter]] = {
    GateClass.PRECIPITATION: {
        "rhohv": MembershipParameter(0.93, 0.96, 1.0, 1.0, 1.0, _INITIAL, _SPEC),
        "zdr": MembershipParameter(-1.0, 0.0, 4.0, 5.5, 0.8, _INITIAL, _SPEC),
        "texture_z": MembershipParameter(0.0, 0.0, 3.0, 6.0, 1.0, _INITIAL, _SPEC),
    },
    GateClass.GROUND_CLUTTER: {
        "rhohv": MembershipParameter(0.0, 0.0, 0.85, 0.92, 1.0, _INITIAL, _SPEC),
        "texture_z": MembershipParameter(4.0, 8.0, 100.0, 100.0, 1.0, _INITIAL, _SPEC),
        "abs_velocity": MembershipParameter(0.0, 0.0, 1.0, 3.0, 1.2, _INITIAL, _SPEC),
        "beam_height_km": MembershipParameter(0.0, 0.0, 1.0, 2.5, 0.8, _INITIAL, _SPEC),
    },
    GateClass.BIOLOGICAL: {
        "rhohv": MembershipParameter(0.0, 0.0, 0.85, 0.92, 1.0, _INITIAL, _SPEC),
        "zdr": MembershipParameter(3.0, 4.5, 8.0, 10.0, 1.2, _INITIAL, _SPEC),
        "reflectivity": MembershipParameter(0.0, 0.0, 25.0, 32.0, 1.0, _INITIAL, _SPEC),
        "texture_z": MembershipParameter(3.0, 6.0, 100.0, 100.0, 0.8, _INITIAL, _SPEC),
    },
    GateClass.HAIL: {
        "reflectivity": MembershipParameter(48.0, 55.0, 80.0, 80.0, 1.2, _INITIAL, _SPEC),
        "zdr": MembershipParameter(-1.5, -0.5, 1.0, 2.0, 1.0, _INITIAL, _SPEC),
        "rhohv": MembershipParameter(0.82, 0.87, 0.95, 0.97, 1.0, _INITIAL, _SPEC),
    },
    GateClass.DEBRIS: {
        "reflectivity": MembershipParameter(38.0, 45.0, 80.0, 80.0, 1.0, _INITIAL, _SPEC),
        "zdr": MembershipParameter(-4.0, -3.0, 0.0, 1.0, 1.2, _INITIAL, _SPEC),
        "rhohv": MembershipParameter(0.0, 0.0, 0.80, 0.87, 1.2, _INITIAL, _SPEC),
        "texture_z": MembershipParameter(3.0, 6.0, 100.0, 100.0, 0.6, _INITIAL, _SPEC),
    },
}
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_qc_membership.py -v`
Expected: PASS (8 tests)

- [ ] **Step 6: Commit**

```bash
git add src/qc/membership.py src/qc/parameters.py tests/unit/test_qc_membership.py
git commit -m "feat: add fuzzy membership functions with mandatory provenance"
```

---

### Task 9: Gate classification

Aggregates memberships across variables and picks the winning class.

**Files:**
- Create: `src/qc/classifier.py`
- Test: `tests/unit/test_qc_classifier.py`

**Interfaces:**
- Consumes: `trapezoid`, `CLASS_PARAMETERS`, `GateClass` (Task 8); `local_standard_deviation` (Task 7); `beam_height_m` (Task 2)
- Produces:
  - `ClassificationResult` dataclass with `classes: np.ndarray` (int8), `confidence: np.ndarray`, `class_names: list[str]`
  - `classify_gates(sweep: SweepData, velocity: np.ndarray | None = None) -> ClassificationResult`
  - `CLASS_CODES: dict[str, int]`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_qc_classifier.py
import numpy as np
import pytest

from src.parser import SweepData
from src.qc.classifier import CLASS_CODES, classify_gates
from src.qc.parameters import GateClass


def _sweep(**overrides) -> SweepData:
    shape = (36, 40)
    base = dict(
        reflectivity=np.full(shape, 30.0),
        rhohv=np.full(shape, 0.99),
        zdr=np.full(shape, 0.5),
        azimuths=np.linspace(0.0, 350.0, shape[0]),
        ranges_m=np.arange(2125.0, 2125.0 + 250.0 * shape[1], 250.0),
        elevation_angle=0.5,
        elevation_angles=[0.5],
        radar_lat=35.3331,
        radar_lon=-97.2778,
        radar_alt_m=390.0,
        timestamp="2026-08-23T00:00:00Z",
    )
    base.update(overrides)
    return SweepData(**base)


def test_uniform_rain_classifies_as_precipitation():
    result = classify_gates(_sweep())
    assert result.classes[10, 10] == CLASS_CODES[GateClass.PRECIPITATION]


def test_low_rhohv_stationary_noisy_classifies_as_ground_clutter():
    rng = np.random.default_rng(1)
    shape = (36, 40)
    sweep = _sweep(
        reflectivity=rng.normal(35.0, 12.0, shape),
        rhohv=np.full(shape, 0.6),
        zdr=rng.normal(0.0, 3.0, shape),
    )
    result = classify_gates(sweep, velocity=np.zeros(shape))
    clutter = CLASS_CODES[GateClass.GROUND_CLUTTER]
    assert (result.classes == clutter).mean() > 0.5


def test_high_zdr_weak_echo_classifies_as_biological():
    rng = np.random.default_rng(2)
    shape = (36, 40)
    sweep = _sweep(
        reflectivity=rng.normal(15.0, 8.0, shape),
        rhohv=np.full(shape, 0.6),
        zdr=np.full(shape, 6.0),
    )
    result = classify_gates(sweep, velocity=np.full(shape, 12.0))
    biological = CLASS_CODES[GateClass.BIOLOGICAL]
    assert (result.classes == biological).mean() > 0.4


def test_intense_low_zdr_core_classifies_as_hail():
    shape = (36, 40)
    sweep = _sweep(
        reflectivity=np.full(shape, 62.0),
        rhohv=np.full(shape, 0.91),
        zdr=np.full(shape, 0.0),
    )
    result = classify_gates(sweep)
    assert result.classes[10, 10] == CLASS_CODES[GateClass.HAIL]


def test_nan_gates_classify_as_unknown():
    shape = (36, 40)
    sweep = _sweep(reflectivity=np.full(shape, np.nan))
    result = classify_gates(sweep)
    assert (result.classes == CLASS_CODES[GateClass.UNKNOWN]).all()


def test_missing_polarimetric_fields_do_not_classify_everything_as_clutter():
    """Global constraint: absent is not low. A pre-2013 scan must not become
    wall-to-wall ground clutter."""
    shape = (36, 40)
    sweep = _sweep(rhohv=None, zdr=None)
    result = classify_gates(sweep)
    clutter = CLASS_CODES[GateClass.GROUND_CLUTTER]
    assert (result.classes == clutter).mean() < 0.2


def test_classes_array_is_int8():
    result = classify_gates(_sweep())
    assert result.classes.dtype == np.int8


def test_confidence_between_zero_and_one():
    result = classify_gates(_sweep())
    finite = result.confidence[np.isfinite(result.confidence)]
    assert finite.min() >= 0.0
    assert finite.max() <= 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_qc_classifier.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.qc.classifier'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/qc/classifier.py
"""Fuzzy-logic gate classification.

Structured after the operational Hydrometeor Classification Algorithm: each
class scores every gate as a weighted mean of trapezoidal memberships across
the available variables, and the highest-scoring class wins.

Deliberately not a threshold cascade. Low correlation coefficient is ambiguous
between ground clutter, biological scatterers, hail and tornado debris, so no
single variable may decide the outcome.

Classes requiring melting-layer height (wet snow, dry snow, ice crystals,
graupel, big drops) are absent by design — see the design spec, section 8.
"""

from dataclasses import dataclass

import numpy as np

from src.geometry import beam_height_m
from src.qc.membership import trapezoid
from src.qc.parameters import CLASS_PARAMETERS, GateClass
from src.qc.texture import local_standard_deviation

CLASS_CODES: dict[str, int] = {
    GateClass.UNKNOWN: 0,
    GateClass.PRECIPITATION: 1,
    GateClass.GROUND_CLUTTER: 2,
    GateClass.BIOLOGICAL: 3,
    GateClass.HAIL: 4,
    GateClass.DEBRIS: 5,
}
CODE_TO_CLASS: dict[int, str] = {code: name for name, code in CLASS_CODES.items()}

MIN_CONFIDENCE_TO_CLASSIFY = 0.2


def _build_variables(sweep, velocity: np.ndarray | None) -> dict[str, np.ndarray]:
    """Assemble the discriminator fields available for this sweep.

    Variables absent from the volume are simply omitted. They are never
    substituted with zeros, which would make a pre-2013 scan look like clutter.
    """
    reflectivity = np.asarray(sweep.reflectivity, dtype=float)
    variables: dict[str, np.ndarray] = {
        "reflectivity": reflectivity,
        "texture_z": local_standard_deviation(reflectivity),
    }

    if sweep.rhohv is not None:
        variables["rhohv"] = np.asarray(sweep.rhohv, dtype=float)
    if sweep.zdr is not None:
        variables["zdr"] = np.asarray(sweep.zdr, dtype=float)
    if velocity is not None:
        variables["abs_velocity"] = np.abs(np.asarray(velocity, dtype=float))

    heights_km = beam_height_m(sweep.ranges_m, sweep.elevation_angle, sweep.radar_alt_m) / 1000.0
    variables["beam_height_km"] = np.broadcast_to(
        heights_km[None, :], reflectivity.shape
    ).copy()

    return variables


def _score_class(class_name: str, variables: dict[str, np.ndarray], shape) -> np.ndarray:
    """Weighted mean membership across the variables this volume provides."""
    total = np.zeros(shape, dtype=float)
    weight_sum = 0.0
    for variable_name, parameter in CLASS_PARAMETERS[class_name].items():
        if variable_name not in variables:
            continue
        membership = trapezoid(
            variables[variable_name], parameter.x1, parameter.x2, parameter.x3, parameter.x4
        )
        total += np.nan_to_num(membership, nan=0.0) * parameter.weight
        weight_sum += parameter.weight
    if weight_sum == 0.0:
        return np.zeros(shape, dtype=float)
    return total / weight_sum


@dataclass
class ClassificationResult:
    classes: np.ndarray
    confidence: np.ndarray
    class_names: list[str]


def classify_gates(sweep, velocity: np.ndarray | None = None) -> ClassificationResult:
    """Classify every gate in a sweep."""
    reflectivity = np.asarray(sweep.reflectivity, dtype=float)
    shape = reflectivity.shape
    variables = _build_variables(sweep, velocity)

    class_names = list(CLASS_PARAMETERS)
    scores = np.stack([_score_class(name, variables, shape) for name in class_names])

    best_index = np.argmax(scores, axis=0)
    confidence = np.max(scores, axis=0)

    classes = np.full(shape, CLASS_CODES[GateClass.UNKNOWN], dtype=np.int8)
    for index, name in enumerate(class_names):
        classes[(best_index == index) & (confidence >= MIN_CONFIDENCE_TO_CLASSIFY)] = CLASS_CODES[name]

    invalid = ~np.isfinite(reflectivity)
    classes[invalid] = CLASS_CODES[GateClass.UNKNOWN]
    confidence = np.where(invalid, np.nan, confidence)

    return ClassificationResult(
        classes=classes, confidence=confidence, class_names=class_names
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_qc_classifier.py -v`
Expected: PASS (8 tests). If a discriminator test fails, adjust the
`arw-tuned-initial` parameters in `src/qc/parameters.py` and record the reason
in the parameter's `reason` field — do not weaken the test.

- [ ] **Step 5: Commit**

```bash
git add src/qc/classifier.py tests/unit/test_qc_classifier.py
git commit -m "feat: add fuzzy-logic gate classifier"
```

---

### Task 10: Geographic collocation with rotation signatures

The debris class requires rotation, detected on a different sweep with
different azimuth sampling. Matching by array index would assume a shared grid.

**Files:**
- Create: `src/qc/collocation.py`
- Test: `tests/unit/test_qc_collocation.py`

**Interfaces:**
- Consumes: `gate_coordinates` (Task 1)
- Produces: `rotation_proximity_mask(sweep, rotation_signatures, margin_km: float = ROTATION_COLLOCATION_MARGIN_KM) -> np.ndarray`, and `ROTATION_COLLOCATION_MARGIN_KM: float`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_qc_collocation.py
import numpy as np

from src.qc.collocation import rotation_proximity_mask
from src.velocity import RotationSignature


def _sweep():
    from src.parser import SweepData

    shape = (72, 60)
    return SweepData(
        reflectivity=np.full(shape, 40.0),
        azimuths=np.linspace(0.0, 355.0, shape[0]),
        ranges_m=np.arange(2125.0, 2125.0 + 250.0 * shape[1], 250.0),
        elevation_angle=0.5,
        elevation_angles=[0.5],
        radar_lat=35.3331,
        radar_lon=-97.2778,
        radar_alt_m=390.0,
        timestamp="2026-08-23T00:00:00Z",
    )


def _signature(lat: float, lon: float, diameter_km: float = 3.0) -> RotationSignature:
    return RotationSignature(
        centroid_lat=lat,
        centroid_lon=lon,
        distance_km=10.0,
        bearing_deg=0.0,
        max_shear_ms=30.0,
        max_inbound_ms=-20.0,
        max_outbound_ms=20.0,
        diameter_km=diameter_km,
        sweep_count=2,
        elevation_angles=[0.48, 0.88],
        strength="moderate",
    )


def test_no_signatures_gives_empty_mask():
    mask = rotation_proximity_mask(_sweep(), [])
    assert mask.shape == (72, 60)
    assert not mask.any()


def test_gates_near_signature_are_marked():
    sweep = _sweep()
    from src.geometry import gate_coordinates

    lat, lon = gate_coordinates(
        sweep.azimuths, sweep.ranges_m, sweep.elevation_angle,
        sweep.radar_lat, sweep.radar_lon,
    )
    target_lat, target_lon = float(lat[10, 30]), float(lon[10, 30])
    mask = rotation_proximity_mask(sweep, [_signature(target_lat, target_lon)])
    assert mask[10, 30]


def test_distant_gates_are_not_marked():
    sweep = _sweep()
    mask = rotation_proximity_mask(sweep, [_signature(40.0, -100.0)])
    assert not mask.any()


def test_margin_widens_the_mask():
    sweep = _sweep()
    from src.geometry import gate_coordinates

    lat, lon = gate_coordinates(
        sweep.azimuths, sweep.ranges_m, sweep.elevation_angle,
        sweep.radar_lat, sweep.radar_lon,
    )
    signature = _signature(float(lat[10, 30]), float(lon[10, 30]))
    narrow = rotation_proximity_mask(sweep, [signature], margin_km=0.0)
    wide = rotation_proximity_mask(sweep, [signature], margin_km=20.0)
    assert wide.sum() > narrow.sum()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_qc_collocation.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.qc.collocation'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/qc/collocation.py
"""Geographic collocation between sweeps.

Rotation signatures are detected on Doppler cuts; classification runs on the
surveillance cut. The two sweeps have different azimuth sampling, so matching
by array index would silently assume a shared grid. Everything here works in
geographic space instead.
"""

import numpy as np

from src.geometry import gate_coordinates

EARTH_RADIUS_KM = 6371.0

# ARW-tuned. A debris field extends beyond the circulation that lofted it, so
# the protection radius is the signature diameter plus this margin.
ROTATION_COLLOCATION_MARGIN_KM = 5.0


def _haversine_km(lat1, lon1, lat2: float, lon2: float) -> np.ndarray:
    """Great-circle distance from every gate to one point, in km."""
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = phi2 - phi1
    delta_lambda = np.radians(lon2 - lon1)
    a = np.sin(delta_phi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def rotation_proximity_mask(
    sweep,
    rotation_signatures,
    margin_km: float = ROTATION_COLLOCATION_MARGIN_KM,
) -> np.ndarray:
    """Gates lying within a rotation signature's radius plus margin."""
    shape = np.asarray(sweep.reflectivity).shape
    mask = np.zeros(shape, dtype=bool)
    if not rotation_signatures:
        return mask

    lat, lon = gate_coordinates(
        azimuths=sweep.azimuths,
        ranges_m=sweep.ranges_m,
        elevation_deg=sweep.elevation_angle,
        radar_lat=sweep.radar_lat,
        radar_lon=sweep.radar_lon,
    )

    for signature in rotation_signatures:
        radius_km = signature.diameter_km / 2.0 + margin_km
        distance_km = _haversine_km(lat, lon, signature.centroid_lat, signature.centroid_lon)
        mask |= distance_km <= radius_km

    return mask
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_qc_collocation.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/qc/collocation.py tests/unit/test_qc_collocation.py
git commit -m "feat: add geographic collocation between reflectivity and Doppler sweeps"
```

---

### Task 11: Protection rules

The safety layer. Hard overrides applied after classification, per spec §9.
Rule 3 is what stops a storm arriving at detection with a hole punched in it.

**Files:**
- Create: `src/qc/protection.py`
- Test: `tests/unit/test_qc_protection.py`

**Interfaces:**
- Consumes: `rotation_proximity_mask` (Task 10)
- Produces: `protected_mask(sweep, rotation_signatures, keep_dbz: float = PROTECTED_MIN_DBZ) -> np.ndarray`, and `PROTECTED_MIN_DBZ: float`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_qc_protection.py
import numpy as np

from src.parser import SweepData
from src.qc.protection import PROTECTED_MIN_DBZ, protected_mask


def _sweep(reflectivity: np.ndarray) -> SweepData:
    n_az, n_rng = reflectivity.shape
    return SweepData(
        reflectivity=reflectivity,
        azimuths=np.linspace(0.0, 355.0, n_az),
        ranges_m=np.arange(2125.0, 2125.0 + 250.0 * n_rng, 250.0),
        elevation_angle=0.5,
        elevation_angles=[0.5],
        radar_lat=35.3331,
        radar_lon=-97.2778,
        radar_alt_m=390.0,
        timestamp="2026-08-23T00:00:00Z",
    )


def test_intense_gates_are_protected():
    reflectivity = np.full((36, 40), 20.0)
    reflectivity[10, 10] = PROTECTED_MIN_DBZ + 5.0
    mask = protected_mask(_sweep(reflectivity), [])
    assert mask[10, 10]


def test_weak_isolated_gates_are_not_protected():
    reflectivity = np.full((36, 40), 20.0)
    mask = protected_mask(_sweep(reflectivity), [])
    assert not mask.any()


def test_rule_three_protects_the_whole_connected_component():
    """A storm containing an intense core must be protected entirely, so QC
    cannot punch a hole in it or amputate its edge."""
    reflectivity = np.full((36, 40), np.nan)
    reflectivity[8:16, 8:16] = 30.0       # storm body, individually unprotected
    reflectivity[11, 11] = 58.0            # intense core
    mask = protected_mask(_sweep(reflectivity), [])
    assert mask[8:16, 8:16].all(), "protection did not spread to the whole storm"


def test_component_without_a_core_is_not_protected():
    reflectivity = np.full((36, 40), np.nan)
    reflectivity[8:16, 8:16] = 30.0
    mask = protected_mask(_sweep(reflectivity), [])
    assert not mask.any()


def test_separate_components_are_independent():
    reflectivity = np.full((36, 40), np.nan)
    reflectivity[2:6, 2:6] = 30.0          # no core
    reflectivity[20:26, 20:26] = 30.0
    reflectivity[22, 22] = 58.0            # core in the second blob only
    mask = protected_mask(_sweep(reflectivity), [])
    assert not mask[2:6, 2:6].any()
    assert mask[20:26, 20:26].all()


def test_rotation_collocation_protects_weak_debris_gates():
    """The tornado case: debris is weak enough to look like clutter, and is
    saved only by sitting on top of a rotation couplet."""
    from src.geometry import gate_coordinates
    from src.velocity import RotationSignature

    reflectivity = np.full((72, 60), np.nan)
    reflectivity[10:14, 28:32] = 42.0
    sweep = _sweep(reflectivity)
    lat, lon = gate_coordinates(
        sweep.azimuths, sweep.ranges_m, sweep.elevation_angle,
        sweep.radar_lat, sweep.radar_lon,
    )
    signature = RotationSignature(
        centroid_lat=float(lat[12, 30]),
        centroid_lon=float(lon[12, 30]),
        distance_km=10.0,
        bearing_deg=0.0,
        max_shear_ms=40.0,
        max_inbound_ms=-25.0,
        max_outbound_ms=25.0,
        diameter_km=2.0,
        sweep_count=2,
        elevation_angles=[0.48, 0.88],
        strength="strong",
    )
    mask = protected_mask(sweep, [signature])
    assert mask[10:14, 28:32].all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_qc_protection.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.qc.protection'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/qc/protection.py
"""Hard protection rules applied after classification.

These exist because low correlation coefficient cannot distinguish a flock of
birds from a tornado lofting debris. A misclassification that discards weather
is silent and unrecoverable downstream, so three overrides guarantee that
anything plausibly severe survives quality control.

Erring toward over-reporting is deliberate: a false storm is recoverable, a
deleted tornado is not.
"""

import numpy as np
from scipy.ndimage import label

from src.qc.collocation import rotation_proximity_mask

# ARW-tuned. Above this, echo is too intense to be biological or ground clutter
# in any operationally meaningful case.
PROTECTED_MIN_DBZ = 50.0

_CONNECTIVITY = np.ones((3, 3), dtype=int)


def protected_mask(
    sweep,
    rotation_signatures,
    keep_dbz: float = PROTECTED_MIN_DBZ,
) -> np.ndarray:
    """Gates that quality control may never discard.

    Rule 1: reflectivity at or above keep_dbz.
    Rule 2: within a rotation signature's radius plus margin.
    Rule 3: any connected component containing a gate protected by 1 or 2.
    """
    reflectivity = np.asarray(sweep.reflectivity, dtype=float)
    finite = np.isfinite(reflectivity)

    seeds = (finite & (reflectivity >= keep_dbz))
    seeds |= rotation_proximity_mask(sweep, rotation_signatures) & finite

    if not seeds.any():
        return np.zeros_like(seeds, dtype=bool)

    labeled, component_count = label(finite, structure=_CONNECTIVITY)
    if component_count == 0:
        return seeds

    protected_component_ids = np.unique(labeled[seeds])
    protected_component_ids = protected_component_ids[protected_component_ids > 0]

    return np.isin(labeled, protected_component_ids)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_qc_protection.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add src/qc/protection.py tests/unit/test_qc_protection.py
git commit -m "feat: add protection rules preventing QC from discarding severe echo"
```

---

### Task 12: Rejected echo layer, quality report and degraded modes

Classification produces a filtered field plus a fully inspectable record of what
was removed and why. The rejected layer is a validation dependency, not a
convenience — Proof 2 cannot be written without it.

**Files:**
- Create: `src/qc/report.py`, `src/qc/apply.py`
- Test: `tests/unit/test_qc_apply.py`

**Interfaces:**
- Consumes: `classify_gates`, `CLASS_CODES`, `CODE_TO_CLASS` (Task 9); `protected_mask` (Task 11)
- Produces:
  - `QualityReport` dataclass: `class_fractions: dict[str, float]`, `rejected_fraction: float`, `degraded_modes: list[str]`, `mean_confidence: float`
  - `RejectedEcho` dataclass: `mask: np.ndarray`, `reasons: np.ndarray`
  - `apply_quality_control(sweep, rotation_signatures, velocity=None) -> tuple[SweepData, RejectedEcho, QualityReport]`
  - `DEGRADED_NO_DUAL_POL`, `DEGRADED_NO_VELOCITY`, `DEGRADED_LOW_CONFIDENCE`, `DEGRADED_BEYOND_VELOCITY_RANGE` string constants
  - `MAX_VELOCITY_RANGE_M: float`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_qc_apply.py
import numpy as np

from src.parser import SweepData
from src.qc.apply import (
    DEGRADED_LOW_CONFIDENCE,
    DEGRADED_NO_DUAL_POL,
    apply_quality_control,
)
from src.qc.parameters import GateClass


def _sweep(**overrides) -> SweepData:
    shape = (36, 40)
    base = dict(
        reflectivity=np.full(shape, 30.0),
        rhohv=np.full(shape, 0.99),
        zdr=np.full(shape, 0.5),
        azimuths=np.linspace(0.0, 350.0, shape[0]),
        ranges_m=np.arange(2125.0, 2125.0 + 250.0 * shape[1], 250.0),
        elevation_angle=0.5,
        elevation_angles=[0.5],
        radar_lat=35.3331,
        radar_lon=-97.2778,
        radar_alt_m=390.0,
        timestamp="2026-08-23T00:00:00Z",
    )
    base.update(overrides)
    return SweepData(**base)


def test_clean_precipitation_survives_quality_control():
    filtered, rejected, report = apply_quality_control(_sweep(), [])
    assert np.isfinite(filtered.reflectivity).all()
    assert not rejected.mask.any()
    assert report.rejected_fraction == 0.0


def test_clutter_is_removed_from_the_filtered_field():
    rng = np.random.default_rng(3)
    shape = (36, 40)
    sweep = _sweep(
        reflectivity=rng.normal(35.0, 12.0, shape),
        rhohv=np.full(shape, 0.6),
        zdr=rng.normal(0.0, 3.0, shape),
    )
    filtered, rejected, report = apply_quality_control(
        sweep, [], velocity=np.zeros(shape)
    )
    assert rejected.mask.any()
    assert report.rejected_fraction > 0.0
    assert np.isnan(filtered.reflectivity[rejected.mask]).all()


def test_rejected_gates_retain_reasons():
    rng = np.random.default_rng(4)
    shape = (36, 40)
    sweep = _sweep(
        reflectivity=rng.normal(35.0, 12.0, shape),
        rhohv=np.full(shape, 0.6),
        zdr=rng.normal(0.0, 3.0, shape),
    )
    _filtered, rejected, _report = apply_quality_control(
        sweep, [], velocity=np.zeros(shape)
    )
    reasons = set(rejected.reasons[rejected.mask].tolist())
    assert reasons
    assert GateClass.PRECIPITATION not in reasons


def test_protected_gates_are_never_rejected():
    shape = (36, 40)
    reflectivity = np.full(shape, 20.0)
    reflectivity[10, 10] = 62.0
    sweep = _sweep(reflectivity=reflectivity, rhohv=np.full(shape, 0.5))
    filtered, rejected, _report = apply_quality_control(
        sweep, [], velocity=np.zeros(shape)
    )
    assert not rejected.mask[10, 10]
    assert np.isfinite(filtered.reflectivity[10, 10])


def test_missing_dual_pol_reports_degraded_mode():
    _filtered, _rejected, report = apply_quality_control(
        _sweep(rhohv=None, zdr=None), []
    )
    assert DEGRADED_NO_DUAL_POL in report.degraded_modes


def test_gates_beyond_velocity_range_report_degraded_mode():
    """The surveillance cut reaches 460 km but Doppler cuts only 300 km, so
    debris protection is unavailable in the outer ring. That limit is a
    property of the radar and must be reported, not silently absorbed."""
    from src.qc.apply import DEGRADED_BEYOND_VELOCITY_RANGE

    long_ranges = np.arange(2125.0, 460_000.0, 250.0)
    shape = (36, len(long_ranges))
    sweep = _sweep(
        reflectivity=np.full(shape, 30.0),
        rhohv=np.full(shape, 0.99),
        zdr=np.full(shape, 0.5),
        ranges_m=long_ranges,
    )
    _filtered, _rejected, report = apply_quality_control(
        sweep, [], velocity=np.zeros(shape)
    )
    assert DEGRADED_BEYOND_VELOCITY_RANGE in report.degraded_modes


def test_short_range_sweep_does_not_report_range_degradation():
    from src.qc.apply import DEGRADED_BEYOND_VELOCITY_RANGE

    _filtered, _rejected, report = apply_quality_control(
        _sweep(), [], velocity=np.zeros((36, 40))
    )
    assert DEGRADED_BEYOND_VELOCITY_RANGE not in report.degraded_modes


def test_class_fractions_sum_to_one():
    _filtered, _rejected, report = apply_quality_control(_sweep(), [])
    assert sum(report.class_fractions.values()) == pytest.approx(1.0, abs=1e-6)


def test_gate_classification_is_attached_to_filtered_sweep():
    filtered, _rejected, _report = apply_quality_control(_sweep(), [])
    assert filtered.gate_classification is not None
    assert filtered.gate_classification.dtype == np.int8
```

Add `import pytest` at the top of the file.

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_qc_apply.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.qc.apply'`

- [ ] **Step 3: Write the report dataclasses**

```python
# src/qc/report.py
"""Quality control reporting.

Every removal and every degradation is recorded. Nothing about quality control
is allowed to be silent — a scan where QC quietly ate a storm must be
distinguishable from a scan that was genuinely clear.
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class RejectedEcho:
    """Everything quality control removed, retained for inspection.

    Required by the clutter-persistence proof, which checks that the same fixed
    gates are flagged across consecutive scans.
    """

    mask: np.ndarray
    reasons: np.ndarray


@dataclass
class QualityReport:
    class_fractions: dict[str, float] = field(default_factory=dict)
    rejected_fraction: float = 0.0
    mean_confidence: float = 0.0
    degraded_modes: list[str] = field(default_factory=list)
```

- [ ] **Step 4: Write the apply step**

```python
# src/qc/apply.py
"""Derive a quality-controlled reflectivity field from a classified sweep."""

from dataclasses import replace

import numpy as np

from src.qc.classifier import CLASS_CODES, CODE_TO_CLASS, classify_gates
from src.qc.parameters import GateClass
from src.qc.protection import protected_mask
from src.qc.report import QualityReport, RejectedEcho

DEGRADED_NO_DUAL_POL = "no_dual_pol"
DEGRADED_NO_VELOCITY = "no_velocity"
DEGRADED_LOW_CONFIDENCE = "low_classification_confidence"
DEGRADED_BEYOND_VELOCITY_RANGE = "beyond_velocity_range"

# Doppler cuts are range-folded beyond roughly 300 km while the surveillance
# cut reaches 460 km. Gates in that outer ring have no velocity, so protection
# rule 2 cannot apply to them. Rule 1 still does.
MAX_VELOCITY_RANGE_M = 300_000.0

# ARW-tuned. Below this mean confidence the classifier is not separating
# classes usefully and the speech layer should say so.
LOW_CONFIDENCE_THRESHOLD = 0.35

NON_METEOROLOGICAL_CLASSES = (GateClass.GROUND_CLUTTER, GateClass.BIOLOGICAL)


def apply_quality_control(sweep, rotation_signatures, velocity=None):
    """Classify a sweep, remove non-meteorological echo, and report what happened.

    Returns (filtered_sweep, rejected_echo, quality_report). The filtered sweep
    carries gate_classification; the rejected echo retains every removed gate
    with its reason.
    """
    classification = classify_gates(sweep, velocity=velocity)
    classes = classification.classes

    reject_codes = [CLASS_CODES[name] for name in NON_METEOROLOGICAL_CLASSES]
    candidate = np.isin(classes, reject_codes)

    protected = protected_mask(sweep, rotation_signatures)
    rejected = candidate & ~protected

    reasons = np.full(classes.shape, "", dtype=object)
    for code in reject_codes:
        reasons[rejected & (classes == code)] = CODE_TO_CLASS[code]

    filtered_reflectivity = np.array(sweep.reflectivity, dtype=float, copy=True)
    filtered_reflectivity[rejected] = np.nan

    total_gates = float(classes.size)
    class_fractions = {
        name: float(np.count_nonzero(classes == code)) / total_gates
        for code, name in CODE_TO_CLASS.items()
    }

    finite_confidence = classification.confidence[np.isfinite(classification.confidence)]
    mean_confidence = float(finite_confidence.mean()) if finite_confidence.size else 0.0

    degraded_modes: list[str] = []
    if sweep.rhohv is None or sweep.zdr is None:
        degraded_modes.append(DEGRADED_NO_DUAL_POL)
    if velocity is None:
        degraded_modes.append(DEGRADED_NO_VELOCITY)
    if mean_confidence < LOW_CONFIDENCE_THRESHOLD:
        degraded_modes.append(DEGRADED_LOW_CONFIDENCE)
    if float(np.max(sweep.ranges_m)) > MAX_VELOCITY_RANGE_M:
        degraded_modes.append(DEGRADED_BEYOND_VELOCITY_RANGE)

    report = QualityReport(
        class_fractions=class_fractions,
        rejected_fraction=float(np.count_nonzero(rejected)) / total_gates,
        mean_confidence=mean_confidence,
        degraded_modes=degraded_modes,
    )

    filtered_sweep = replace(
        sweep, reflectivity=filtered_reflectivity, gate_classification=classes
    )
    return filtered_sweep, RejectedEcho(mask=rejected, reasons=reasons), report
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_qc_apply.py -v`
Expected: PASS (9 tests)

- [ ] **Step 6: Commit**

```bash
git add src/qc/report.py src/qc/apply.py tests/unit/test_qc_apply.py
git commit -m "feat: add QC apply step with rejected echo layer and quality report"
```

---

### Task 13: Wire quality control into the pipeline

Speckle removal moves after classification. Running it first would delete small
hail cores while leaving large clutter fields intact.

**Files:**
- Modify: `src/preprocess.py`
- Modify: `src/server.py:120-150`
- Modify: `src/buffer.py`
- Test: `tests/unit/test_preprocess.py`, `tests/smoke/test_server_smoke.py`

**Interfaces:**
- Consumes: `apply_quality_control` (Task 12)
- Produces: `preprocess_sweep(sweep, rotation_signatures, velocity=None) -> tuple[SweepData, ScanQuality, RejectedEcho]`; `ScanQuality` gains `class_fractions`, `rejected_fraction`, `degraded_modes`, `mean_confidence`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/unit/test_preprocess.py
import numpy as np

from src.preprocess import preprocess_sweep


def _sweep(reflectivity, **overrides):
    from src.parser import SweepData

    n_az, n_rng = reflectivity.shape
    base = dict(
        reflectivity=reflectivity,
        rhohv=np.full(reflectivity.shape, 0.99),
        zdr=np.full(reflectivity.shape, 0.5),
        azimuths=np.linspace(0.0, 355.0, n_az),
        ranges_m=np.arange(2125.0, 2125.0 + 250.0 * n_rng, 250.0),
        elevation_angle=0.5,
        elevation_angles=[0.5],
        radar_lat=35.3331,
        radar_lon=-97.2778,
        radar_alt_m=390.0,
        timestamp="2026-08-23T00:00:00Z",
    )
    base.update(overrides)
    return SweepData(**base)


def test_preprocess_returns_quality_with_class_fractions():
    sweep = _sweep(np.full((36, 40), 30.0))
    _processed, quality, _rejected = preprocess_sweep(sweep, [])
    assert quality.class_fractions
    assert 0.0 <= quality.rejected_fraction <= 1.0


def test_quality_control_runs_before_speckle_removal():
    """A small intense core must survive. If speckle removal ran first it would
    delete the core before QC could protect it."""
    reflectivity = np.full((36, 40), np.nan)
    reflectivity[10, 10] = 58.0
    reflectivity[10, 11] = 56.0
    sweep = _sweep(reflectivity)
    processed, _quality, _rejected = preprocess_sweep(sweep, [])
    assert np.isfinite(processed.reflectivity[10, 10])


def test_preprocess_attaches_gate_classification():
    sweep = _sweep(np.full((36, 40), 30.0))
    processed, _quality, _rejected = preprocess_sweep(sweep, [])
    assert processed.gate_classification is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_preprocess.py -k preprocess_sweep -v`
Expected: FAIL with `ImportError: cannot import name 'preprocess_sweep'`

- [ ] **Step 3: Extend `ScanQuality` and add `preprocess_sweep`**

```python
# in src/preprocess.py — extend the ScanQuality dataclass
@dataclass
class ScanQuality:
    score: float
    finite_fraction: float
    removed_speckle_pixels: int
    removed_speckle_fraction: float
    flags: list[str]
    class_fractions: dict[str, float] = field(default_factory=dict)
    rejected_fraction: float = 0.0
    mean_confidence: float = 0.0
    degraded_modes: list[str] = field(default_factory=list)
```

Add `field` to the dataclasses import.

```python
# add to src/preprocess.py
def preprocess_sweep(sweep, rotation_signatures, velocity=None):
    """Quality control, then speckle removal, then scan quality assessment.

    Order matters. Quality control runs first so its protection rules can save
    small intense cores that speckle removal would otherwise delete.
    """
    from src.qc.apply import apply_quality_control

    original_reflectivity = sweep.reflectivity
    qc_sweep, rejected, qc_report = apply_quality_control(
        sweep, rotation_signatures, velocity=velocity
    )

    despeckled, removed_speckle_pixels = _remove_weak_speckle(qc_sweep.reflectivity)

    quality = assess_scan_quality(
        original_reflectivity=original_reflectivity,
        processed_reflectivity=despeckled,
        removed_speckle_pixels=removed_speckle_pixels,
    )
    quality.class_fractions = qc_report.class_fractions
    quality.rejected_fraction = qc_report.rejected_fraction
    quality.mean_confidence = qc_report.mean_confidence
    quality.degraded_modes = qc_report.degraded_modes

    return replace(qc_sweep, reflectivity=despeckled), quality, rejected
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_preprocess.py -v`
Expected: PASS

- [ ] **Step 5: Rewire `_ingest_to_buffer`**

Velocity analysis must run before quality control, because protection rule 2
needs the rotation signatures. Replace `src/server.py:120-150`:

```python
def _ingest_to_buffer(site_id: str, dt: datetime | None = None) -> BufferedScan:
    """Fetch a scan, quality control it, detect objects, and buffer the result."""
    filepath = fetch_scan(site_id.upper(), dt)
    radar = parse_radar_file(filepath)
    raw_sweep = extract_sweep_data(radar)
    vel_data = extract_velocity(radar)

    # Rotation is detected before quality control so protection rule 2 can use
    # it: a debris signature is only distinguishable from clutter by sitting on
    # top of a velocity couplet.
    preliminary_rotations = detect_rotation_signatures(vel_data) if vel_data else []
    lowest_velocity = vel_data.sweeps[0].velocity if vel_data and vel_data.sweeps else None

    ref_data, scan_quality, rejected_echo = preprocess_sweep(
        raw_sweep, preliminary_rotations, velocity=lowest_velocity
    )

    result = detect_objects_with_grid(
        reflectivity=ref_data.reflectivity,
        azimuths=ref_data.azimuths,
        ranges_m=ref_data.ranges_m,
        radar_lat=ref_data.radar_lat,
        radar_lon=ref_data.radar_lon,
        elevation_deg=ref_data.elevation_angle,
    )
    regions, rotations, annotated_objects = analyze_velocity(vel_data, result.objects)
    scan_timestamp = (
        datetime.fromisoformat(ref_data.timestamp)
        if isinstance(ref_data.timestamp, str)
        else ref_data.timestamp
    )
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
        rejected_echo=rejected_echo,
    )
    _buffer.add_scan(buffered)
    _tracker.update(buffered)
    return buffered
```

Add `rejected_echo: RejectedEcho | None = None` to `BufferedScan` in
`src/buffer.py`, and import `detect_rotation_signatures` and `preprocess_sweep`
in `src/server.py`.

- [ ] **Step 6: Run the full suite**

Run: `.venv/Scripts/python.exe -m pytest -v`
Expected: PASS. Object counts will change now that clutter is removed. Update
smoke tests asserting exact counts to assert ranges or invariants instead, and
record the change in the commit message.

- [ ] **Step 7: Commit**

```bash
git add src/preprocess.py src/server.py src/buffer.py tests/
git commit -m "feat: wire quality control into the ingest pipeline

Rotation detection runs before QC so protection rule 2 can use it. Speckle
removal now runs after QC so it cannot delete protected cores."
```

---

### Task 14: Phase-neutral precipitation labels

ARW currently reports "moderate rain" during snowstorms. Phase discrimination
needs melting-layer height, which is deferred to Spec 4, so the labels stop
asserting phase.

**Files:**
- Modify: `src/detection.py:13-19,51-58`
- Modify: `src/map_layer.py:26-33,54-61`
- Modify: `src/summary.py`
- Test: `tests/unit/test_detection.py`

**Interfaces:**
- Consumes: nothing
- Produces: `INTENSITY_THRESHOLDS` labels become "light precipitation", "moderate precipitation", "heavy precipitation", "intense precipitation", "severe core"; `classify_intensity` returns the same strings; threshold values are unchanged

- [ ] **Step 1: Write the failing test**

```python
# append to tests/unit/test_detection.py
from src.detection import classify_intensity


def test_intensity_labels_do_not_assert_precipitation_phase():
    """ARW cannot distinguish rain from snow without melting-layer height, so
    it must not claim either."""
    for dbz in (25.0, 35.0, 45.0, 55.0, 65.0):
        assert "rain" not in classify_intensity(dbz)


def test_intensity_thresholds_are_unchanged():
    assert classify_intensity(25.0) == "light precipitation"
    assert classify_intensity(35.0) == "moderate precipitation"
    assert classify_intensity(45.0) == "heavy precipitation"
    assert classify_intensity(55.0) == "intense precipitation"
    assert classify_intensity(65.0) == "severe core"
    assert classify_intensity(10.0) == "drizzle"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_detection.py -k intensity -v`
Expected: FAIL — `classify_intensity(25.0)` returns `"light rain"`

- [ ] **Step 3: Update the labels**

```python
# replace src/detection.py:13-19
INTENSITY_THRESHOLDS = [
    (20, 30, "light precipitation"),
    (30, 40, "moderate precipitation"),
    (40, 50, "heavy precipitation"),
    (50, 60, "intense precipitation"),
    (60, float("inf"), "severe core"),
]
```

- [ ] **Step 4: Update every consumer of the literal strings**

Run: `grep -rn "light rain\|moderate rain\|heavy rain\|intense rain" src/ tests/ scripts/`

Update `_intensity_fill_color` and `_intensity_rule_type` in
`src/map_layer.py:26-61`, keeping the GeoJSON `ruleType` values stable
(`radar_light_rain` and friends) so existing Audiom styling does not break —
only the human-readable labels change. Add a comment at each mapping saying so.

- [ ] **Step 5: Run the full suite**

Run: `.venv/Scripts/python.exe -m pytest -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/detection.py src/map_layer.py src/summary.py tests/
git commit -m "fix: use phase-neutral precipitation labels

ARW reported 'moderate rain' during snowstorms. Phase discrimination requires
melting-layer height, deferred to Spec 4. GeoJSON ruleType values are
unchanged so Audiom styling is unaffected."
```

---

### Task 15: Proof 2 — clutter persistence

Spec §11. Tests the classifier against physics rather than an eyeballed answer.

**Files:**
- Create: `tests/e2e/test_proof_clutter_persistence.py`

**Interfaces:**
- Consumes: `apply_quality_control` (Task 12), `extract_sweep_data`, `extract_velocity` (Tasks 5-6)
- Produces: nothing

- [ ] **Step 1: Confirm the clear-air scans are still cached**

Six consecutive KIWA scans on 2026-07-12 were measured at roughly 0.2% of valid
gates above 20 dBZ — essentially clear air, so persistent returns are dominated
by terrain and structures. Confirm they are present:

```bash
ls cache/KIWA/KIWA20260712_16*_V06 cache/KIWA/KIWA20260712_17*_V06
```

Expected: `163410`, `164255`, `165142`, `170029`, `170914`, `171800`.

If any are missing, fetch them through `src/ingest.py` — never by adding a
network call elsewhere.

- [ ] **Step 2: Write the test**

```python
# tests/e2e/test_proof_clutter_persistence.py
"""Proof 2 from the design spec: ground clutter sits still.

Buildings and terrain return echo from the same gates on every scan. A
classifier that works must flag substantially the same gate set each time, and
those gates must show near-zero radial velocity. This checks a property that is
true by physics, so it needs no human to eyeball an expected answer.
"""

import numpy as np
import pyart
import pytest

from src.parser import extract_sweep_data, extract_velocity
from src.qc.apply import apply_quality_control
from src.qc.classifier import CLASS_CODES
from src.qc.parameters import GateClass

# Consecutive KIWA scans on 2026-07-12. Measured at ~0.2% of valid gates above
# 20 dBZ, so what persists between them is terrain and structures rather than
# weather. Roughly 9 minutes apart.
CLEAR_AIR_SCANS = [
    "cache/KIWA/KIWA20260712_163410_V06",
    "cache/KIWA/KIWA20260712_164255_V06",
    "cache/KIWA/KIWA20260712_165142_V06",
    "cache/KIWA/KIWA20260712_170029_V06",
    "cache/KIWA/KIWA20260712_170914_V06",
    "cache/KIWA/KIWA20260712_171800_V06",
]

MIN_JACCARD_OVERLAP = 0.6
MAX_MEAN_ABS_VELOCITY_MS = 3.0


def _clutter_mask(path: str):
    radar = pyart.io.read_nexrad_archive(path)
    sweep = extract_sweep_data(radar)
    velocity_data = extract_velocity(radar)
    velocity = velocity_data.sweeps[0].velocity if velocity_data else None
    filtered, _rejected, _report = apply_quality_control(sweep, [], velocity=velocity)
    clutter = filtered.gate_classification == CLASS_CODES[GateClass.GROUND_CLUTTER]
    return clutter, velocity


def test_clutter_gates_persist_across_consecutive_scans():
    masks = [_clutter_mask(path)[0] for path in CLEAR_AIR_SCANS]
    assert any(m.any() for m in masks), "no clutter identified in any scan"

    for earlier, later in zip(masks, masks[1:]):
        intersection = np.count_nonzero(earlier & later)
        union = np.count_nonzero(earlier | later)
        assert union > 0
        jaccard = intersection / union
        assert jaccard >= MIN_JACCARD_OVERLAP, f"clutter set unstable: {jaccard:.2f}"


def test_clutter_gates_are_nearly_stationary():
    for path in CLEAR_AIR_SCANS:
        clutter, velocity = _clutter_mask(path)
        if velocity is None or not clutter.any():
            continue
        overlap = min(clutter.shape[0], velocity.shape[0])
        clutter_velocity = velocity[:overlap][clutter[:overlap]]
        finite = clutter_velocity[np.isfinite(clutter_velocity)]
        if finite.size == 0:
            continue
        assert np.abs(finite).mean() < MAX_MEAN_ABS_VELOCITY_MS
```

- [ ] **Step 3: Run the proof**

Run: `.venv/Scripts/python.exe -m pytest tests/e2e/test_proof_clutter_persistence.py -v`
Expected: PASS. If the Jaccard overlap falls short, tune the ground clutter
parameters in `src/qc/parameters.py`, updating their `provenance` to
`arw-tuned` and recording the case in `reason`.

- [ ] **Step 4: Commit**

```bash
git add tests/e2e/test_proof_clutter_persistence.py src/qc/parameters.py
git commit -m "test: add clutter persistence proof"
```

---

### Task 16: Proof 3 — Level III and MRMS cross-check

Spec §11. Independent algorithmic opinions on identical data.

**Files:**
- Create: `scripts/crosscheck_external.py`
- Create: `docs/test_reports/2026-08-23-external-crosscheck.md`
- Modify: `src/ingest.py`

**Interfaces:**
- Consumes: `apply_quality_control`, `detect_objects_with_grid`
- Produces: `fetch_level3_attributes(site_id, timestamp)` and `fetch_mrms_composite(timestamp)` in `src/ingest.py`

- [ ] **Step 1: Discover the bucket layouts before writing any code**

`src/ingest.py` uses `nexradaws`, which handles Level II only. Level III and
MRMS live in different public buckets and need `boto3` directly — already a
project dependency. Their key layouts must be read from the buckets, not
assumed. Run:

```bash
.venv/Scripts/python.exe -c "
import boto3
from botocore import UNSIGNED
from botocore.config import Config
s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
for bucket, prefix in [('unidata-nexrad-level3', ''), ('noaa-mrms-pds', 'CONUS/')]:
    r = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, Delimiter='/', MaxKeys=20)
    print(bucket)
    for p in r.get('CommonPrefixes', []): print('  dir ', p['Prefix'])
    for o in r.get('Contents', [])[:5]: print('  key ', o['Key'])
"
```

Record the actual layouts in a comment above the functions you write. Drill
down with further `Prefix=` calls until you can construct a key for a known
site and timestamp.

- [ ] **Step 2: Add the retrieval functions to the ingest layer**

The architectural rule is absolute: only the Ingest Manager makes network
calls. These belong in `src/ingest.py` alongside `fetch_scan`, never in
`src/qc/` or in the comparison script.

```python
# add to src/ingest.py
import boto3
from botocore import UNSIGNED
from botocore.config import Config

NEXRAD_LEVEL3_BUCKET = "unidata-nexrad-level3"
MRMS_BUCKET = "noaa-mrms-pds"

# Storm attributes: the Level III product carrying the storm tracking
# attributes RadarScope and RadarOmega display.
LEVEL3_STORM_ATTRIBUTES_PRODUCT = "NST"


def _anonymous_s3():
    """Public NOAA buckets need no credentials."""
    return boto3.client("s3", config=Config(signature_version=UNSIGNED))


def _download_key(bucket: str, key: str, subdirectory: str) -> str:
    """Download one S3 object into the cache, skipping if already present."""
    filename = key.rsplit("/", 1)[-1]
    cache_dir = os.path.join(os.path.abspath(CACHE_DIR), subdirectory)
    os.makedirs(cache_dir, exist_ok=True)
    local_path = os.path.join(cache_dir, filename)
    if os.path.isfile(local_path):
        return local_path
    _anonymous_s3().download_file(bucket, key, local_path)
    return local_path


def _nearest_key(bucket: str, prefix: str, timestamp: datetime) -> str:
    """Key under a prefix whose name sorts closest to the wanted timestamp.

    Both buckets embed the scan time in the object name, so lexical proximity
    on a zero-padded timestamp is chronological proximity.
    """
    paginator = _anonymous_s3().get_paginator("list_objects_v2")
    keys = [
        obj["Key"]
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix)
        for obj in page.get("Contents", [])
    ]
    if not keys:
        raise RuntimeError(f"No objects under s3://{bucket}/{prefix}")
    wanted = timestamp.strftime("%Y%m%d%H%M%S")
    return min(keys, key=lambda k: _timestamp_distance(k, wanted))


def _timestamp_distance(key: str, wanted: str) -> int:
    """Absolute difference between the digits embedded in a key and the target."""
    digits = "".join(c for c in key.rsplit("/", 1)[-1] if c.isdigit())
    if len(digits) < len(wanted):
        return 10**18
    return abs(int(digits[: len(wanted)]) - int(wanted))


def fetch_level3_attributes(site_id: str, timestamp: datetime) -> str:
    """Download the NWS Level III storm attributes product for a site and time.

    site_id is the four-letter ICAO identifier; the Level III bucket keys use
    the three-letter form, so the leading K is stripped.
    """
    short_site = site_id.upper().lstrip("K")
    prefix = _level3_prefix(short_site, LEVEL3_STORM_ATTRIBUTES_PRODUCT, timestamp)
    key = _nearest_key(NEXRAD_LEVEL3_BUCKET, prefix, timestamp)
    return _download_key(NEXRAD_LEVEL3_BUCKET, key, f"level3/{site_id.upper()}")


def fetch_mrms_composite(timestamp: datetime) -> str:
    """Download the MRMS quality-controlled composite reflectivity grid."""
    prefix = _mrms_prefix(timestamp)
    key = _nearest_key(MRMS_BUCKET, prefix, timestamp)
    return _download_key(MRMS_BUCKET, key, "mrms")
```

Write `_level3_prefix(short_site, product, timestamp)` and
`_mrms_prefix(timestamp)` from the layouts discovered in Step 1. Both return a
prefix string narrow enough that `_nearest_key` lists a single day rather than
the whole bucket.

- [ ] **Step 3: Add a grib2 reader if MRMS needs one**

MRMS composite products are grib2. Nothing in the current dependency list reads
grib2. If the comparison needs the grid values rather than just the file, add
`cfgrib` (or `pygrib`) to `pyproject.toml` dependencies and run
`uv sync`. If only the footprint extent is needed, note that and skip the
dependency — do not add a package the comparison does not use.

- [ ] **Step 4: Write the comparison script**

```python
# scripts/crosscheck_external.py
"""Proof 3: compare ARW's quality-controlled output against independent sources.

NWS Level III attributes and NOAA MRMS are independent algorithmic opinions on
the same radar data. Disagreement is a signal to investigate, not an automatic
failure — but unexplained disagreement blocks the spec.

Usage:
    python scripts/crosscheck_external.py KTLX 2026-04-13T22:30:00
"""

import argparse
from datetime import datetime

import pyart

from src.detection import detect_objects_with_grid
from src.ingest import fetch_level3_attributes, fetch_mrms_composite, fetch_scan
from src.parser import extract_sweep_data, extract_velocity
from src.preprocess import preprocess_sweep
from src.velocity import detect_rotation_signatures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("site_id")
    parser.add_argument("timestamp", type=datetime.fromisoformat)
    args = parser.parse_args()

    radar = pyart.io.read_nexrad_archive(fetch_scan(args.site_id, args.timestamp))
    sweep = extract_sweep_data(radar)
    velocity_data = extract_velocity(radar)
    rotations = detect_rotation_signatures(velocity_data) if velocity_data else []
    velocity = velocity_data.sweeps[0].velocity if velocity_data else None

    filtered, quality, rejected = preprocess_sweep(sweep, rotations, velocity=velocity)
    result = detect_objects_with_grid(
        reflectivity=filtered.reflectivity,
        azimuths=filtered.azimuths,
        ranges_m=filtered.ranges_m,
        radar_lat=filtered.radar_lat,
        radar_lon=filtered.radar_lon,
        elevation_deg=filtered.elevation_angle,
    )

    print(f"ARW objects: {len(result.objects)}")
    print(f"Rejected fraction: {quality.rejected_fraction:.4f}")
    print(f"Class fractions: {quality.class_fractions}")
    print(f"Degraded modes: {quality.degraded_modes}")
    print(f"Level III attributes: {fetch_level3_attributes(args.site_id, args.timestamp)}")
    print(f"MRMS composite: {fetch_mrms_composite(args.timestamp)}")
    print("Compare ARW object centroids against Level III storm attribute")
    print("positions, and ARW surviving echo footprint against the MRMS")
    print("quality-controlled composite. Record findings in docs/test_reports/.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run the cross-check on at least three live volumes**

Choose one dense convective case (the KTLX 2026-04-10 afternoon window in
`cache/KTLX/` has 55 scans), one clear-air case (the KIWA 2026-07-12 series
from Task 15), and one winter case. Run the script against each.

- [ ] **Step 6: Record findings**

Write `docs/test_reports/2026-08-23-external-crosscheck.md` containing, for each
volume: ARW object count, rejected fraction, class fractions, Level III storm
attribute count and positions, MRMS footprint comparison, and an explicit
statement of every disagreement found and whether it is explained.

**Unexplained disagreement blocks the spec.** If something cannot be explained,
say so plainly in the report rather than rationalising it.

- [ ] **Step 7: Commit**

```bash
git add src/ingest.py scripts/crosscheck_external.py pyproject.toml docs/test_reports/
git commit -m "test: add Level III and MRMS cross-check proof"
```

---

### Task 17: Proof 4 — SPC known severe cases

Spec §11. The proof that catches the catastrophic failure: quality control must
never delete the thing the app exists to report.

**Files:**
- Create: `tests/e2e/test_proof_known_severe_cases.py`

**Interfaces:**
- Consumes: everything from Tasks 5-13
- Produces: nothing

- [ ] **Step 1: Select the cases**

Choose at least two confirmed events from SPC storm reports
(https://www.spc.noaa.gov/climo/reports/) with cached or retrievable volumes:
one confirmed tornado with a debris signature, one confirmed large hail report.

Record for each: site, scan timestamp, event latitude and longitude, event type,
and the SPC report reference. These go in the test as module constants with
comments.

- [ ] **Step 2: Write the test**

```python
# tests/e2e/test_proof_known_severe_cases.py
"""Proof 4 from the design spec: quality control must not delete hazards.

Low correlation coefficient is ambiguous between ground clutter and tornado
debris. A classifier tuned only against clear-air clutter could silently
discard exactly the echo this application exists to report. These are real
events with confirmed SPC storm reports.
"""

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pyart
import pytest

from src.detection import detect_objects_with_grid
from src.geometry import gate_coordinates
from src.parser import extract_sweep_data, extract_velocity
from src.preprocess import preprocess_sweep
from src.qc.collocation import _haversine_km
from src.velocity import detect_rotation_signatures


@dataclass(frozen=True)
class SevereCase:
    name: str
    volume_path: str
    event_lat: float
    event_lon: float
    spc_reference: str


# Replace with real confirmed events before this task is complete.
KNOWN_CASES = [
    SevereCase(
        name="REPLACE: confirmed tornado with debris signature",
        volume_path="cache/REPLACE/REPLACE",
        event_lat=0.0,
        event_lon=0.0,
        spc_reference="REPLACE with SPC storm report reference",
    ),
    SevereCase(
        name="REPLACE: confirmed large hail",
        volume_path="cache/REPLACE/REPLACE",
        event_lat=0.0,
        event_lon=0.0,
        spc_reference="REPLACE with SPC storm report reference",
    ),
]

SEARCH_RADIUS_KM = 10.0


def _process(case: SevereCase):
    radar = pyart.io.read_nexrad_archive(case.volume_path)
    sweep = extract_sweep_data(radar)
    velocity_data = extract_velocity(radar)
    rotations = detect_rotation_signatures(velocity_data) if velocity_data else []
    velocity = velocity_data.sweeps[0].velocity if velocity_data else None
    filtered, quality, rejected = preprocess_sweep(sweep, rotations, velocity=velocity)
    return filtered, quality, rejected


@pytest.mark.parametrize("case", KNOWN_CASES, ids=lambda c: c.name)
def test_hazard_echo_survives_quality_control(case: SevereCase):
    """Echo at the confirmed event location must not be removed."""
    filtered, _quality, _rejected = _process(case)
    lat, lon = gate_coordinates(
        filtered.azimuths, filtered.ranges_m, filtered.elevation_angle,
        filtered.radar_lat, filtered.radar_lon,
    )
    near_event = _haversine_km(lat, lon, case.event_lat, case.event_lon) <= SEARCH_RADIUS_KM
    surviving = np.isfinite(filtered.reflectivity) & near_event
    assert surviving.any(), f"QC removed all echo at the {case.name} location"


@pytest.mark.parametrize("case", KNOWN_CASES, ids=lambda c: c.name)
def test_hazard_produces_a_detected_object(case: SevereCase):
    """The event must survive all the way through to a detected object."""
    filtered, _quality, _rejected = _process(case)
    result = detect_objects_with_grid(
        reflectivity=filtered.reflectivity,
        azimuths=filtered.azimuths,
        ranges_m=filtered.ranges_m,
        radar_lat=filtered.radar_lat,
        radar_lon=filtered.radar_lon,
        elevation_deg=filtered.elevation_angle,
    )
    distances_km = [
        float(_haversine_km(
            np.array([o.centroid_lat]), np.array([o.centroid_lon]),
            case.event_lat, case.event_lon,
        )[0])
        for o in result.objects
    ]
    assert distances_km, "no objects detected at all"
    assert min(distances_km) <= 30.0, f"nearest object {min(distances_km):.1f} km away"
```

- [ ] **Step 3: Replace every REPLACE placeholder with real case data**

A test parameterised over placeholder coordinates proves nothing. This task is
not complete until both cases reference confirmed SPC reports and real volumes.

- [ ] **Step 4: Run the proof**

Run: `.venv/Scripts/python.exe -m pytest tests/e2e/test_proof_known_severe_cases.py -v`
Expected: PASS

If a hazard is removed, **do not weaken the test.** Investigate which rule
should have protected it. The likely fixes are widening
`ROTATION_COLLOCATION_MARGIN_KM`, lowering `PROTECTED_MIN_DBZ`, or correcting
the debris membership parameters. Record whichever changed and why.

- [ ] **Step 5: Commit**

```bash
git add tests/e2e/test_proof_known_severe_cases.py src/qc/
git commit -m "test: add SPC known severe case proof"
```

---

### Task 18: Benchmark re-baselining

Spec §11. QC is the only change that moves detection and tracking, because
sweep selection leaves the reflectivity field untouched.

**Files:**
- Create: `docs/test_reports/2026-08-23-qc-rebaseline.md`
- Modify: `PROGRESS.md`

**Interfaces:**
- Consumes: the existing benchmark manifest harness in `scripts/`
- Produces: nothing

- [ ] **Step 1: Locate the benchmark harness**

Run: `ls scripts/` and read whichever script runs the benchmark manifests
described in PROGRESS.md. Record its exact invocation in the report.

- [ ] **Step 2: Capture the post-change baseline**

Run the benchmark manifests across the same windows PROGRESS.md names: the
dense KTLX window, the simpler window, and the merge/split-sensitive window.

Record for each: object counts, fragmentation proxy, focus switches, heading
flips, lineage and event totals, summary publishability counts.

- [ ] **Step 3: Compare against the pre-change baseline**

The pre-change numbers are in `docs/benchmarks/` and the existing reports in
`docs/test_reports/`. Produce a delta table per window.

- [ ] **Step 4: Check the expected direction**

Removing clutter should **reduce** fragmentation and heading instability. Some
Phase 2 suppression machinery likely exists to compensate for noise that QC
now eliminates at source.

**If fragmentation or heading instability increased, that is evidence QC is
removing weather, and it blocks the spec.** Investigate before proceeding —
the most likely cause is over-aggressive ground clutter or biological
membership parameters.

- [ ] **Step 5: Write the report**

`docs/test_reports/2026-08-23-qc-rebaseline.md` containing the delta tables, the
direction check, and an explicit statement of anything unexplained.

Include the rotation delta from Task 6 Step 8 as a separate section, since that
change originates in sweep selection rather than QC.

- [ ] **Step 6: Do NOT re-tune Phase 2 suppression parameters**

Global constraint. Record the deltas and leave the tuning alone. If the
suppression machinery now looks over-conservative, note it in the report as
input to Spec 3. Changing the input field and the tuning that responds to it in
one spec destroys attribution.

- [ ] **Step 7: Commit**

```bash
git add docs/test_reports/2026-08-23-qc-rebaseline.md
git commit -m "docs: record QC re-baseline against existing benchmark windows"
```

---

### Task 19: Obtain published parameters and correct project records

Closes the open dependency in spec §13 and corrects the Phase 3 record.

**Files:**
- Modify: `src/qc/parameters.py`
- Modify: `PROGRESS.md`

**Interfaces:**
- Consumes: nothing
- Produces: nothing

- [ ] **Step 1: Attempt to obtain the published membership parameters**

Source: Park, H. S., A. V. Ryzhkov, D. S. Zrnić and K. Kim, 2009: "The
Hydrometeor Classification Algorithm for the Polarimetric WSR-88D: Description
and Application to an MCS." *Weather and Forecasting*, 24, 730-748.

For each class ARW implements, extract the published trapezoid parameters for
reflectivity, ZDR, RhoHV and texture where the paper provides them.

- [ ] **Step 2: Update the parameter table**

For every parameter successfully sourced, replace the value and set
`provenance="park2009"` with the table or figure reference in `reason`.

**If a value cannot be obtained, leave it `arw-tuned` and say so.** Never
attribute an invented number to the paper. This is a global constraint.

- [ ] **Step 3: Re-run every proof after changing parameters**

Run: `.venv/Scripts/python.exe -m pytest -v`

Changing membership parameters changes classification, so Proofs 2 and 4 and
the Task 18 re-baseline must all be re-run. If the re-baseline moves, update
`docs/test_reports/2026-08-23-qc-rebaseline.md`.

- [ ] **Step 4: Verify the provenance constraint holds**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_qc_membership.py::test_every_parameter_has_provenance -v`
Expected: PASS

- [ ] **Step 5: Correct the Phase 3 record in PROGRESS.md**

The entry claiming multi-sweep rotation confirmation was validated is wrong.
Replace it with an accurate account:

```markdown
  - Rotation signature detection: gate-to-gate shear detection (NWS criteria:
    >= 15 m/s across < 5 km), strength classification (weak/moderate/strong),
    multi-sweep confirmation
    - CORRECTION 2026-08-23: multi-sweep confirmation did not function as
      described. Velocity sweeps were selected by index, which in split-cut
      VCPs returns surveillance cuts carrying no velocity data. Two of three
      selected sweeps were empty, so sweep_count was structurally pinned at 1.
      The unit tests passed because they used synthetic sweeps that all carried
      velocity — a VCP structure that does not occur. Fixed in Spec 1 by
      content-based sweep selection.
```

- [ ] **Step 6: Add the Spec 1 completion entry to PROGRESS.md**

Under Completed, record: content-based sweep selection, polarimetric quality
control with protection rules, corrected georeferencing and beam height,
phase-neutral labels, and the four proofs. Under Next, record Spec 2 (contour
shapes), Spec 3 (SCIT), Spec 4 (hail and mesocyclone).

- [ ] **Step 7: Run the full suite one final time**

Run: `.venv/Scripts/python.exe -m pytest -v`
Expected: PASS, all tests

- [ ] **Step 8: Commit**

```bash
git add src/qc/parameters.py PROGRESS.md docs/test_reports/
git commit -m "docs: source published QC parameters and correct Phase 3 record

Multi-sweep rotation confirmation was recorded as validated but could never
engage. PROGRESS.md now states what actually happened and why the tests
missed it."
```

---

## Completion Checklist

From spec §14. All must hold before Spec 1 is complete:

- [ ] All four proofs pass (Tasks 1, 15, 16, 17)
- [ ] Benchmark re-baseline shows fragmentation and heading instability unchanged or reduced (Task 18)
- [ ] Rotation `sweep_count` distribution shows multi-sweep confirmation engaging on real data (Task 6)
- [ ] Site selection reaches 345 km rather than 306 km (Task 2)
- [ ] No membership parameter lacks a provenance marker (Task 19)
- [ ] Full test suite green, both synthetic velocity tests rewritten (Task 6)
- [ ] `devspec/06` corrected (Task 2)
- [ ] PROGRESS.md updated, including the Phase 3 correction (Task 19)
