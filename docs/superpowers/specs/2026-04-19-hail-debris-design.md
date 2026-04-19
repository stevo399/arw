# Phase 4: Hail Detection and Debris Scoring — Design Spec

Date: 2026-04-19

## Goal

Add hail detection (dual-pol discrimination + NWS MESH diameter estimation) and tornado debris signature (TDS) scoring to ARW, giving blind users the same hail size and debris information available in mainstream radar applications.

## Architecture

A single `src/hazards.py` module handles all hail and debris analysis. Freezing level data is fetched by a new arm of the Ingest Manager (`src/ingest/`). Dual-pol fields are extracted by an extension to the existing parser. The hazard analysis plugs into the existing pipeline between velocity analysis and buffer storage.

Pipeline addition:

```
... → analyze_velocity() → extract_dual_pol() → fetch_freezing_level() → analyze_hazards() → BufferedScan
```

## 1. Dual-Pol Extraction

**Module:** `src/parser.py`

New function `extract_dual_pol(radar)` pulls ZDR and CC from the lowest dual-pol sweep (sweep 0 on NEXRAD Level II).

```python
@dataclass
class DualPolData:
    zdr: np.ndarray          # differential reflectivity (dB)
    cc: np.ndarray           # cross-correlation ratio (0-1)
    azimuths: np.ndarray
    ranges_m: np.ndarray
    radar_lat: float
    radar_lon: float
    elevation_angle: float
    timestamp: str
```

- Sweep 0 contains dual-pol fields; sweep 1 is Doppler-only. The function reads only sweep 0.
- If `differential_reflectivity` or `cross_correlation_ratio` is missing from the radar object, return `None`. The pipeline continues without hail/debris analysis.
- KDP is not extracted initially — ZDR and CC are sufficient for both hail and debris detection.

## 2. Freezing Level Ingest

**Module:** `src/ingest/` (new function, stays inside Ingest Manager per architecture rule)

`fetch_freezing_level(lat, lon)` downloads the 0 C isotherm height from NOAA's RAP (Rapid Refresh) model via NOMADS.

```python
@dataclass
class FreezingLevel:
    height_m_asl: float      # meters above sea level
    timestamp: str           # RAP analysis time
    source: str              # "RAP" or "fallback"
```

**Caching:**
- Called once per site switch, not per scan. Freezing level changes slowly.
- Cached in memory at module level in `server.py`, keyed by site ID.
- A site switch clears the cache and triggers a fresh fetch.

**Fallback:**
- If RAP is unavailable (network error, data gap, parse failure), MESH is skipped entirely.
- Dual-pol hail detection still works without MESH — it just produces no diameter estimate.
- The `source` field makes fallback state transparent to downstream consumers.

## 3. Hail Detection

**Module:** `src/hazards.py`

Two complementary methods combined into a single per-object hail assessment.

### 3a. Dual-Pol Discrimination

Applied to every detected object when dual-pol data is available:

| Criterion | Threshold |
|-----------|-----------|
| High reflectivity | >= 45 dBZ (object peak) |
| Low ZDR | < 1.0 dB (within object mask) |
| Reduced CC | 0.85-0.95 (within object mask) |

Classification:
- All three criteria met: `hail_likely`
- High reflectivity + one pol indicator: `hail_possible`
- Otherwise: `None`

### 3b. MESH (Maximum Estimated Size of Hail)

Applied when freezing level is available. For each object:
- Integrate reflectivity profile above the freezing level using the NWS SHI/MESH algorithm
- Produce a diameter estimate in inches (NWS convention)

Size classification:
- < 1.0 inches: `small`
- 1.0-1.75 inches: `moderate`
- >= 1.75 inches: `severe` (matches NWS severe hail criteria)

### 3c. Combined Output

```python
@dataclass
class HailAssessment:
    object_id: int
    dual_pol_flag: str | None      # "hail_likely", "hail_possible", or None
    mesh_diameter_in: float | None  # inches, None if no freezing level
    mesh_label: str | None          # "small", "moderate", "severe"
    max_zdr: float | None
    min_cc: float | None
    peak_dbz_above_fzl: float | None
```

## 4. Tornado Debris Signature (TDS) Detection

**Module:** `src/hazards.py`

Strict NWS criteria — all four must be met simultaneously:

1. **CC < 0.80** — debris causes dramatic decorrelation
2. **Reflectivity >= 20 dBZ** — enough signal to trust the CC reading
3. **Co-located rotation** — an existing rotation signature (from Phase 3) must overlap within 5 km
4. **Low elevation angle** — only the lowest sweep (< 1.0 degrees), since debris stays near the surface

Detection flow:
- For each gate meeting criteria 1+2+4, cluster into contiguous regions via connected-component labeling (same approach used in velocity region detection)
- For each region, check whether any Phase 3 rotation signature centroid is within 5 km (haversine)
- Only regions with co-located rotation produce a TDS

```python
@dataclass
class DebrisSignature:
    centroid_lat: float
    centroid_lon: float
    distance_km: float
    bearing_deg: float
    min_cc: float
    area_km2: float
    co_located_rotation_strength: str  # "weak", "moderate", "strong"
```

## 5. Pipeline Integration

### 5a. Entry Point

`src/hazards.py` exposes one main function:

```python
def analyze_hazards(
    dual_pol: DualPolData | None,
    objects: list[DetectedObject],
    rotation_signatures: list[RotationSignature],
    freezing_level: FreezingLevel | None,
    reflectivity_data: ReflectivityData,
) -> HazardAnalysis:
```

```python
@dataclass
class HazardAnalysis:
    hail_assessments: list[HailAssessment]
    debris_signatures: list[DebrisSignature]
```

Called from `_ingest_to_buffer` in `server.py` after `analyze_velocity`.

### 5b. Buffer

`BufferedScan` gains three new fields:
- `dual_pol_data: DualPolData | None = None`
- `hail_assessments: list[HailAssessment] = field(default_factory=list)`
- `debris_signatures: list[DebrisSignature] = field(default_factory=list)`

### 5c. Tracker

`Track` gains `hail_history: list` — recent hail assessments per scan, capped at 6 entries (same as rotation_history). Debris is not tracked per-track; it is a per-scan spatial alert.

### 5d. API

- `/objects/{site_id}` gains per-object fields: `hail_flag`, `mesh_diameter_in`, `mesh_label`
- New `/hazards/{site_id}` endpoint returning `HazardResponse` with hail assessments and debris signatures
- `/summary/{site_id}` integrates hail and debris language

### 5e. Freezing Level Management

Fetched in `_ingest_to_buffer` on first scan for a site (or after site switch). Stored module-level in `server.py` alongside `_buffer` and `_tracker`.

### 5f. Summary Language

**Priority order** (highest first):
1. Debris signatures — standalone alert, always spoken
2. Strongest object with hail/rotation/motion annotations
3. Standalone rotation reports (up to 2 non-strongest objects)
4. Scene coverage

**Hail speech** — per-object annotation:
- With MESH: "Hail up to 2.0 inches."
- Without MESH: "Possible hail signatures." or "Hail signatures detected."

**Debris speech** — standalone alert before other content:
- "Tornado debris signature detected 15 miles NNW of the radar, strong rotation."
- Up to 2 TDS regions reported, nearest first.

## 6. Testing Strategy

### Unit Tests (~25-30 new)

- `tests/unit/test_dual_pol_parser.py` — ZDR/CC extraction, missing field handling, sweep selection
- `tests/unit/test_hazards.py` — hail thresholds (likely/possible/none), MESH calculation, TDS criteria (all four NWS requirements), edge cases (no dual-pol, no freezing level, no co-located rotation)
- `tests/unit/test_summary.py` — hail language, debris alert language, priority ordering

### Smoke Tests

- Server endpoint patches updated for new pipeline arguments
- New `/hazards/{site_id}` smoke test

### E2E Tests

- Full pipeline patch updates for dual-pol + hazards flow

### Live Validation

- Replay against cached KTLX 2026-04-10 data (active severe weather)
- Confirm hail signatures appear on high-reflectivity objects
- Confirm debris signatures require co-located rotation
- Confirm speech reads naturally
- Confirm no regressions in tracking, focus, rotation persistence
