# Velocity Ingestion & Region Detection Design

Date: 2026-04-18

## Overview

Add velocity data extraction, inbound/outbound region detection, and rotation signature detection to the ARW pipeline. Velocity is extracted from multiple low-level sweeps, analyzed in a single new `src/velocity.py` module, and integrated into the existing buffer, tracker, and summary systems.

## Parser Changes

### Current state

`extract_reflectivity(filepath)` opens a NEXRAD file with `pyart.io.read_nexrad_archive()`, extracts reflectivity from sweep 0, and discards the radar object.

### New design

- `parse_radar_file(filepath) -> pyart Radar` — reads the file once and returns the radar object
- `extract_reflectivity(radar) -> ReflectivityData` — refactored to accept a radar object instead of a filepath
- `extract_velocity(radar, max_sweeps=3) -> VelocityData | None` — extracts velocity from the lowest N sweeps where velocity is available; returns `None` if no velocity data in the file

### Data structures

```python
@dataclass
class VelocitySweep:
    velocity: np.ndarray        # 2D (azimuths x range gates), m/s, NaN for missing
    azimuths: np.ndarray
    ranges_m: np.ndarray
    elevation_angle: float
    nyquist_velocity: float     # needed for dealiasing awareness

@dataclass
class VelocityData:
    sweeps: list[VelocitySweep]
    radar_lat: float
    radar_lon: float
```

Py-ART's dealiasing is applied during extraction to unwrap aliased velocity values.

## Velocity Module: `src/velocity.py`

A single module handling region detection, rotation detection, and rain object association.

### Region Detection

Detects spatially coherent areas of significant inbound or outbound flow.

**Algorithm:**
1. For each sweep, threshold velocity into inbound (<= -10 m/s) and outbound (>= 10 m/s) masks
2. Connected-component labeling on each mask (same approach as reflectivity detection)
3. Filter regions below minimum area (~4 km2)
4. Cross-sweep merging: match regions across sweeps by spatial overlap
5. Compute centroid lat/lon using polar-to-geographic conversion

**Output:**

```python
@dataclass
class VelocityRegion:
    region_type: str              # "inbound" or "outbound"
    peak_velocity_ms: float
    mean_velocity_ms: float
    area_km2: float
    centroid_lat: float
    centroid_lon: float
    distance_km: float
    bearing_deg: float
    sweep_count: int              # how many sweeps this region appears on
    elevation_angles: list[float]
```

The 10 m/s threshold is a tunable parameter, chosen to filter background wind while keeping meteorologically significant flow.

### Rotation Detection

Finds gate-to-gate shear signatures where strong inbound is adjacent to strong outbound.

**Algorithm:**
1. For each sweep, scan azimuthally for adjacent gates where velocity changes sign sharply
2. Compute shear: `|V_inbound - V_outbound|` across the gate-to-gate distance
3. Flag as rotation candidate when shear >= 15 m/s across < 5 km (NWS mesocyclone criteria)
4. Cluster nearby shear pixels into rotation signatures
5. Multi-sweep confirmation: match signatures across sweeps by spatial proximity

**Strength classification:**
- **weak:** 15-25 m/s shear
- **moderate:** 25-35 m/s shear
- **strong:** 35+ m/s shear (significant tornado potential)

A rotation appearing on 2+ sweeps at similar location gets higher confidence. Single-sweep detections are still reported but flagged as less certain.

**Output:**

```python
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
    strength: str                 # "weak", "moderate", "strong"
```

### Rain Object Association

Each rotation signature and velocity region is matched to the nearest overlapping `DetectedObject` by centroid proximity within the object's spatial extent. Unmatched signatures are reported as standalone features.

## Buffer & Tracker Integration

### BufferedScan additions

```python
velocity_data: VelocityData | None
velocity_regions: list[VelocityRegion]
rotation_signatures: list[RotationSignature]
```

`None` for `velocity_data` handles NEXRAD files without velocity.

### DetectedObject additions

```python
max_inbound_ms: float | None
max_outbound_ms: float | None
rotation: RotationSignature | None
```

### Tracker changes

Tracks inherit velocity annotations from `DetectedObject` through existing association. A lightweight `rotation_history` list on tracks records rotation detections per scan, enabling persistence summaries:
- "persistent rotation" (3+ consecutive scans)
- "new rotation detected" (first detection)
- "rotation weakening" (was present, now gone)

No changes to the core tracker association, motion, or focus logic.

### Pipeline change

```python
radar = parse_radar_file(filepath)
ref_data = extract_reflectivity(radar)
ref_data, quality = preprocess_reflectivity_data(ref_data)
vel_data = extract_velocity(radar)
objects = detect_objects_with_grid(...)
regions, rotations = analyze_velocity(vel_data, objects)
# annotate objects with velocity/rotation overlap
buffered = BufferedScan(..., velocity_data=vel_data,
    velocity_regions=regions, rotation_signatures=rotations)
```

## API Integration

### New endpoint

`GET /velocity/{site_id}` returns velocity regions and rotation signatures.

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
    associated_object_id: int | None

class VelocityResponse(BaseModel):
    site_id: str
    timestamp: str
    regions: list[VelocityRegionModel]
    rotation_signatures: list[RotationSignatureModel]
```

### Enhanced existing endpoints

- `/objects/{site_id}` — `RainObject` gains optional `max_inbound_ms`, `max_outbound_ms`, `rotation_strength` fields
- `/tracks/{site_id}` — `StormTrack` gains optional `rotation_history`

## Summary Integration

### Spoken patterns

**Object annotation:**
> "severe core, 31 miles E, moving SE at 8 mph, moderate rotation detected"

**Standalone rotation:**
> "Rotation signature detected 45 miles NW of the radar, strong shear of 40 m/s"

**Persistence language:**
- "persistent rotation" — 3+ consecutive scans
- "new rotation" — first detection on this track
- "rotation weakening" — was present, now absent

### Priority

Rotation signatures are always spoken — they are the most operationally critical information. When multiple rotations exist, strongest first.

## Testing Strategy

- **Unit tests:** velocity extraction from cached NEXRAD files, region detection on synthetic velocity grids, rotation detection on synthetic shear couplets, strength classification, cross-sweep merging, rain object association
- **Smoke tests:** full pipeline with velocity through the API endpoints
- **Live replay:** validate velocity regions and rotation detections against cached KTLX scans (which contain velocity data)
