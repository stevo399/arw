# Spec 1: Correct the Data Layer

**Date:** 2026-08-22
**Status:** Design — awaiting review
**Phase:** Precedes Phase 4. Blocks Spec 2 (shape), Spec 3 (SCIT), Spec 4 (hail/mesocyclone).

## 1. Purpose

ARW converts NEXRAD radar into speech and GeoJSON for blind users. Those users
cannot check the output against a visual radar map, so an error in ARW is an
error the user has no way to catch. Everything downstream — object detection,
tracking, shape, hazard analysis — inherits whatever the data layer hands it.

This spec makes the data layer correct and conforming before any interpretation
work is built on top of it.

### Governing principle

ARW conforms to established radar meteorology practice everywhere except its
output layer. The accessible presentation — speech, spatial audio, GeoJSON for
Audiom — is new work with no precedent. How data is retrieved, quality
controlled, and interpreted follows published convention.

This matters because the reference applications do not invent interpretation
either. RadarScope, RadarOmega and GRLevel3 render base data faithfully and
overlay algorithm output the National Weather Service already computed. The
convention ARW conforms to is therefore the NWS's published algorithms, not any
particular app's behaviour.

## 2. Scope

### In scope

- VCP-aware sweep selection (fixes a confirmed velocity defect)
- Dual-polarimetric field extraction
- A gate classification quality-control stage
- Georeferencing corrections: gate coordinates, beam height, gate areas
- Phase-neutral precipitation labelling
- Validation and re-baselining of existing tracking benchmarks

### Out of scope — deliberately deferred

| Deferred | Reason | Lands in |
|---|---|---|
| Contour-based polygon shapes | Own correctness surface | Spec 2 |
| SCIT cell identification | Reworks validated tracking code | Spec 3 |
| HDA/SHI/MESH hail, MDA mesocyclone | Need volumetric parsing | Spec 4 |
| Melting-layer-dependent hydrometeor classes | Need melting-layer height | Spec 4 |
| Re-tuning Phase 2 suppression parameters | Would destroy attribution — see §11 | Spec 3 |

## 3. Evidence

All measurements taken 2026-08-22 against `cache/KEMX/KEMX20260712_022646_V06`,
a real cached volume. They are reproducible and should be locked into tests.

### 3.1 Split-cut sweep defect (confirmed, functional)

Volume elevation angles: `[0.48, 0.48, 0.88, 0.88, 1.27, 1.27, 1.8, ...]`

Each low elevation is scanned twice — a surveillance cut for reflectivity and a
Doppler cut for velocity.

| Sweep | Elevation | Reflectivity valid | Velocity valid | Nyquist |
|---|---|---|---|---|
| 0 | 0.48° | 18.5% | **0.0%** | 8.3 m/s |
| 1 | 0.48° | 14.3% | 12.8% | 30.6 m/s |
| 2 | 0.88° | 19.7% | **0.0%** | 8.3 m/s |
| 3 | 0.88° | 16.3% | 14.9% | 30.6 m/s |
| 4 | 1.27° | 20.7% | **0.0%** | 9.1 m/s |
| 5 | 1.27° | 17.4% | 16.0% | 30.6 m/s |

`parser.py:83` selects velocity sweeps by index `range(3)`, taking sweeps 0, 1
and 2. Two of those three carry no velocity data whatsoever.

Consequence: `sweep_count` increments only on cross-sweep merge
(`velocity.py:306`), so it can never exceed 1 in production. The multi-sweep
rotation confirmation recorded in PROGRESS.md as validated Phase 3 work is
structurally incapable of confirming anything. `sweep_count` is published to API
consumers at `server.py:455` as though it were a confidence signal.
`RotationSignature.elevation_angles` likewise only ever sees one tilt, so there
is no vertical depth information — and vertical continuity is central to how the
operational mesocyclone algorithm confirms a circulation.

The unit tests at `test_velocity.py:60` and `test_velocity.py:139` pass because
they construct synthetic sweeps that all contain velocity — a VCP structure that
does not occur in real data.

Reflectivity is unaffected: sweep 0 is the correct surveillance cut.

### 3.2 Beam height error (functional)

`compute_beam_height_m` (`sites.py:204`) uses the true earth radius of 6371 km.
The standard is the 4/3 effective radius, 8494.7 km. `devspec/06` specifies
6371 km, so the specification is wrong as well as the code.

| Range | Current | Correct | Overestimate |
|---|---|---|---|
| 50 km | 633 m | 583 m | 49 m |
| 100 km | 1657 m | 1461 m | 196 m |
| 200 km | 4885 m | 4100 m | 785 m |
| 300 km | 9681 m | 7916 m | 1766 m |

Because `MAX_BEAM_HEIGHT_M` rejects sites above 10 km, site selection currently
rejects everything beyond **306 km** where the correct model reaches **345 km**.
A 39 km band of usable radar sites is wrongly excluded, and every surviving
site's ranking is computed from inflated heights.

### 3.3 Gate coordinate error (minor)

`polar_to_latlon` (`detection.py:67`) treats slant range as ground range and
ignores elevation angle. Measured against Py-ART's Doviak & Zrnić
implementation at 0.48° elevation:

| Range | Displacement |
|---|---|
| 52 km | 8 m |
| 102 km | 26 m |
| 202 km | 107 m |
| 302 km | 271 m |
| 460 km | 764 m |

Negligible at the ranges where most analysis happens. Corrected because it is
free once `geometry.py` exists, not because it is urgent.

### 3.4 No polarimetric quality control

`parser.py` extracts only `reflectivity` and `velocity`. Without RhoHV
screening, ground clutter, anomalous propagation, birds, insects, chaff and
interference are all counted as precipitation and promoted to detected objects.

`devspec/04` lists correlation coefficient and differential reflectivity as
required inputs and `devspec/05` names low correlation coefficient as a debris
scoring input. Neither is ingested.

**The fields are already present in cached files** and require no new ingest:
`cross_correlation_ratio`, `differential_reflectivity`, `differential_phase`,
`spectrum_width`, `clutter_filter_power_removed`.

The cached volumes also contain all 17 elevation sweeps, so the later volumetric
specs need parsing and memory work, not new network paths.

### 3.5 Precipitation phase asserted without basis

`classify_intensity` (`detection.py:51`) maps dBZ directly to "light rain",
"moderate rain", "heavy rain". In a snowstorm ARW reports *moderate rain*.
Phase discrimination requires the melting-layer height, which ARW does not have.

## 4. Architecture

Radar knowledge is currently scattered: `detection.py` owns coordinate
conversion and gate areas, `sites.py` owns beam height, `parser.py` owns sweep
selection implicitly by indexing. The split-cut defect exists because sweep
selection was never any module's stated responsibility.

### Pipeline

```
ingest
  → parser.parse_radar_file()                    pyart Radar object
  → sweeps.select()                              NEW — VCP-aware selection
  → parser.extract_sweep_data()                  CHANGED — multi-field
       ↓ SweepData
  → qc.classify_gates()                          NEW — gate classification
       ↓ SweepData + gate_classification + RejectedEcho
  → qc.apply()                                   NEW — derive filtered field
  → preprocess.remove_speckle()                  MOVED — now runs after QC
  → detection.detect_objects_with_grid()         unchanged interface
  → velocity.detect_regions() / detect_rotation()
  → buffer.BufferedScan
  → tracking / summary / map_layer / server
```

### New modules

**`src/geometry.py`** — all radar-to-geographic mathematics, and the only place
it lives.

| Function | Replaces |
|---|---|
| `gate_coordinates(sweep)` | `detection.polar_to_latlon` |
| `beam_height_m(range_km, elevation_deg, site_m)` | `sites.compute_beam_height_m` |
| `gate_areas_km2(sweep)` | `detection._range_bin_areas_km2` |
| `ground_range_km(slant_range_m, elevation_deg)` | new |

Backed internally by Py-ART's transforms rather than hand-rolled trigonometry.
Gate coordinates are computed on demand with a per-site cache, never stored per
scan — they are a pure function of site, elevation, azimuths and ranges.

**`src/sweeps.py`** — VCP-aware sweep selection. See §5.

**`src/qc/`** — package containing:
- `classifier.py` — membership functions and class decision
- `protection.py` — never-discard override rules
- `report.py` — `QualityReport`, degraded-mode reporting

### Changed modules

- `src/parser.py` — `ReflectivityData` becomes `SweepData` (§6)
- `src/preprocess.py` — speckle removal moves after classification; becomes a
  thin orchestrator

**Correction to the speckle ordering rationale, 2026-08-25.** The plan derived
from this spec justified running quality control before speckle removal on the
grounds that speckle-first "would delete small hail cores". That is false.
`_remove_weak_speckle` already preserves any component whose peak reaches
`MIN_SPECKLE_PEAK_DBZ_TO_KEEP` (35 dBZ), so hail cores at 50+ dBZ were never at
risk from it. The ordering is still correct — running speckle removal on the
cleaned field means it acts on weather speckle rather than clutter speckle — but
not for the reason originally given.

The ordering has one measured consequence in the opposite direction: QC-first can
strip the surroundings of a moderate 20–35 dBZ gate, leaving a one-gate island
that speckle removal then deletes, where speckle-first would have kept it inside
a larger component. This is accepted. A gate whose entire neighbourhood
classified as non-meteorological is itself likely non-meteorological, and the
difference cannot reach the output in any case: one gate spans 0.05–0.44 km²
depending on range, while `MIN_OBJECT_AREA_KM2` is 4.0 — 19 to 74 contiguous
gates are required before any object forms.
- `src/sites.py` — beam height delegates to `geometry.py`
- `src/detection.py` — coordinate and area helpers delegate to `geometry.py`;
  intensity labels become phase-neutral
- `devspec/06_Radar_Site_Selection_Beam_Height.md` — corrected to 4/3 model

## 5. Sweep selection

**Selection is by inspection of file contents, never by index.**

1. Group sweeps by `fixed_angle`, rounded to 0.05°, so split cuts fall in one
   group.
2. Within each group, score every candidate by the fraction of gates carrying
   valid data for the requested field.
3. **Reflectivity** — the highest-coverage reflectivity sweep in the lowest
   elevation group. Resolves to the surveillance cut, which also has
   unambiguous range to 460 km.
4. **Velocity** — the highest-coverage velocity sweep in each group, ascending,
   keeping the lowest 3 elevation groups that contain any usable velocity sweep.
   Resolves to the Doppler cuts at 0.48°, 0.88° and 1.27° in the reference
   volume. Three preserves the existing `max_sweeps=3` contract; the difference
   is that all three now carry data.

No VCP lookup table and no parity assumption. The rule survives NWS scan
strategy changes and handles legacy volumes without split cuts, where each group
holds a single sweep and selection is trivially correct.

### Consequences

- **Reflectivity is unchanged.** Sweep 0 is already the correct surveillance
  cut. Detection and tracking benchmarks therefore do not move from sweep
  selection. This isolation is deliberate and must be preserved — see §11.
- **Velocity changes substantially.** Rotation detection moves from one real
  tilt padded with two empty ones to three real tilts. `sweep_count` becomes
  meaningful, multi-sweep confirmation engages for the first time, and rotation
  signature counts and strengths will shift.

### Dealiasing

`dealias_region_based` currently runs across the whole radar object including
empty surveillance cuts (`parser.py:79`). It moves after selection and runs only
on selected sweeps, each against its own Nyquist velocity.

### Documented range limit

The surveillance cut sees reflectivity to 460 km; Doppler cuts see velocity to
300 km. Rotation detection, and therefore debris protection rule 2 (§8), is
unavailable beyond 300 km. This is a property of the radar and must be reported
in `QualityReport`, not left to be discovered.

## 6. Data model

`ReflectivityData` becomes `SweepData`:

```python
@dataclass
class SweepData:
    # co-registered gate arrays, all (n_rays, n_gates)
    reflectivity: np.ndarray
    velocity: np.ndarray | None
    rhohv: np.ndarray | None
    zdr: np.ndarray | None
    spectrum_width: np.ndarray | None
    clutter_power_removed: np.ndarray | None
    gate_classification: np.ndarray | None   # int8, written by qc

    # geometry
    azimuths: np.ndarray
    ranges_m: np.ndarray
    elevation_angle: float
    nyquist_velocity: float | None

    # site and time
    radar_lat: float
    radar_lon: float
    radar_alt_m: float
    timestamp: str
    elevation_angles: list[float]
```

A real rename, not an alias. `ReflectivityData` carrying correlation coefficient
would be a name that lies. The eight consumer modules use almost exclusively
`.reflectivity`, `.timestamp` and the polar axes, so the change is mechanical.

### Memory

Each `BufferedScan` currently holds one reflectivity sweep (5.3 MB) plus three
velocity sweeps (16 MB) ≈ 21 MB; a full two-hour buffer is roughly 525 MB.

Adding RhoHV and ZDR adds 10.6 MB per scan. Removing the two empty velocity
sweeps frees the same amount. **Net footprint is approximately unchanged while
carrying materially more information.**

Gate coordinates are not stored. `gate_classification` is `int8`, not float.

## 7. Georeferencing corrections

| Item | Change |
|---|---|
| Gate coordinates | Doviak & Zrnić 4/3 effective earth model via Py-ART |
| Beam height | effective radius 8494.7 km, not 6371 km |
| Gate areas | ground range, actual per-ray azimuth spacing |
| `devspec/06` | corrected alongside the code |

Gate areas will shrink slightly at long range; this appears in the re-baseline
and is expected.

## 8. Gate classifier

### Structure

Trapezoidal fuzzy membership functions per variable per class, aggregated by
weighted sum, highest aggregate score wins — following the structure of the
operational Hydrometeor Classification Algorithm (Park, Ryzhkov, Zrnić and Kim,
*Weather and Forecasting*, 2009).

Explicitly **not** a threshold cascade. A cascade is where the catastrophic
failure lives: one variable crossing one line decides everything, and low RhoHV
alone cannot distinguish a flock of birds from a tornado lofting debris.

### Input variables

| Variable | Source | Role |
|---|---|---|
| Z | `reflectivity` | intensity |
| ZDR | `differential_reflectivity` | particle shape |
| RhoHV | `cross_correlation_ratio` | particle uniformity |
| SD(Z) | computed, local window | spatial coherence |
| SD(PhiDP) | from `differential_phase` | phase coherence |
| Velocity | Doppler cut, matched geographically | motion — clutter does not move |
| Clutter power removed | `clutter_filter_power_removed` | radar's own clutter diagnostic |
| Beam height, range | `geometry.py` | clutter is low and near |

Texture measures carry significant weight. Non-meteorological echo is spatially
incoherent; weather is smooth. SD(Z) requires no dual-pol, which is what keeps
the pre-2013 degraded mode useful rather than useless.

### Classes

| Class | Z | ZDR | RhoHV | SD(Z) | Velocity | Other |
|---|---|---|---|---|---|---|
| Precipitation | any | 0 to +4 | > 0.95 | low | any | — |
| Ground clutter / AP | any | noisy | < 0.90 | high | ≈ 0 | near, low, clutter filter active |
| Biological | < 30 | > +4 | < 0.90 | high | non-zero | clear-air surroundings |
| Hail | > 50 | ≈ 0 | 0.85–0.95 | moderate | any | inside a storm |
| Tornado debris | > 40 | ≤ 0 | < 0.85 | high | any | collocated with rotation |
| Unknown | — | — | — | — | — | no class scores confidently |

Wet snow, dry snow, ice crystals, graupel and big drops are **absent by
design**. Each requires melting-layer height. Producing them without it yields
confident wrong answers, so the classifier declines instead.

### Cross-sweep collocation

The debris class requires rotation, which is detected on the Doppler cut — a
different sweep with different azimuth sampling from the surveillance cut being
classified. Matching by array index would silently assume a shared grid.

Collocation is therefore performed **in geographic space**: rotation signatures
carry a lat/lon centroid and diameter, and candidate gates are tested by
geographic distance using `geometry.py`. This removes any dependence on the two
sweeps having compatible shapes or azimuth ordering.

### Phase-neutral labelling

`classify_intensity` intensity labels change from "light rain" / "moderate
rain" / "heavy rain" to phase-neutral equivalents ("light precipitation", and so
on), and precipitation phase is reported as unknown. ARW stops asserting
something it cannot support.

Threshold *values* are unchanged; only the labels change. Speech and GeoJSON
consumers must be updated consistently — `summary.py`, `map_layer.py` rule
types and fill-colour lookups, and any test asserting the literal strings.

## 9. Protection rules

Hard overrides applied **after** classification. These are the reason a tornado
survives quality control.

A gate is never discarded if:

1. `Z >= 50 dBZ`. Too intense to be biological or clutter in any operationally
   meaningful case. If wrong, ARW over-reports, which is the survivable
   direction.
2. It lies within the collocation radius of a detected rotation signature,
   defined as the signature's `diameter_km` plus a fixed margin. The margin is
   an ARW-tuned parameter requiring a provenance marker under §10; it exists
   because a debris field extends beyond the circulation that lofted it.
3. It belongs to a connected component containing any gate protected by (1) or
   (2).

Rule 3 is not optional. Without it, a debris ball can be correctly retained
while surrounding gates are classified as clutter, delivering a storm to
detection with a hole punched in it or an edge amputated — which corrupts shape
in Spec 2 and centroids in tracking.

## 10. Output and degraded modes

### Classification, not deletion

The classifier writes `gate_classification` and a per-gate confidence. A
separate explicit step derives the filtered reflectivity field consumed by
detection.

**Everything rejected is retained as an inspectable `RejectedEcho` layer with
per-gate reasons.** This is a validation dependency, not a preference: the
clutter-persistence proof (§11) cannot be implemented if rejected gates are
discarded.

### Degraded modes

All reported in `QualityReport`, never silent:

| Mode | Trigger | Behaviour |
|---|---|---|
| No dual-pol | pre-2013 volume, RhoHV/ZDR absent | texture and velocity discriminators only |
| Beyond velocity range | gate > 300 km | no rotation, protection rule 2 inactive; rule 1 still applies |
| Low confidence | mean classifier confidence below an ARW-tuned threshold across the scan | flagged for the speech layer |

Missing polarimetric arrays must be detected as absent, never treated as low
values — doing so would classify an entire historical scan as clutter.

`ScanQuality` grows to carry per-class gate fractions, rejected fraction, and
active degraded modes.

### Provenance

Every membership function parameter carries a comment marking it as either a
published value with citation, or an ARW-tuned value with its reasoning and the
case it was tuned against. This distinction is the entire point of conforming to
published algorithms and is unrecoverable if not recorded during
implementation.

## 11. Validation

### The four proofs

**1. Py-ART agreement.** Assert `geometry.gate_coordinates()` matches Py-ART's
`gate_latitude` / `gate_longitude` to floating-point tolerance across several
elevations and the full range extent. Lock the §3.3 displacement table into a
test so the magnitude of the old error remains documented.

**2. Clutter persistence.** Over consecutive clear-air scans at a site with
known terrain returns, assert the classifier flags substantially the same gate
set each time and that those gates carry near-zero radial velocity. Tests
against physics rather than an eyeballed expected answer.

**3. Level III and MRMS cross-check.** For selected live volumes, compare
surviving echo footprint and object set against NWS Level III attributes and
NOAA's quality-controlled MRMS for the same time. Disagreement is a signal to
investigate, not an automatic failure — but unexplained disagreement blocks the
spec.

**4. SPC known cases.** Replay documented tornado and large-hail events with
confirmed SPC storm reports; assert the hazard survives quality control and
lands in the correct place. Real cases with confirmed reports, never synthetic.

### Re-baselining

Sweep selection leaves the reflectivity field untouched, so **QC is the only
change that moves detection and tracking.** Preserve that isolation
deliberately.

Run existing benchmark manifests before and after. Attribute every delta in
object counts, fragmentation, focus switches and heading flips to QC alone.

**Expected direction:** removing clutter should *reduce* fragmentation and
heading instability, since some Phase 2 suppression machinery likely exists to
compensate for noise QC eliminates at source. **If the numbers move the other
way, that is evidence QC is removing weather, and it blocks the spec.**

**Do not re-tune Phase 2 suppression parameters in this spec.** Record deltas,
leave tuning alone, address in Spec 3 where that code is reworked anyway.
Changing the input field and the tuning that responds to it in one spec destroys
attribution.

Rotation changes are measured separately, since they originate in sweep
selection rather than QC: signature counts, strengths and `sweep_count`
distribution before and after.

## 12. Test plan

### Unit

- Membership functions: shape, boundaries, saturation
- Protection rules, especially rule 3 connected-component propagation
- Sweep selection against synthetic VCPs: split-cut, legacy non-split,
  velocity-absent, single-sweep
- `geometry.py` against Py-ART reference values
- Degraded modes: absent RhoHV/ZDR, beyond-velocity-range gates
- Cross-sweep geographic collocation with deliberately mismatched azimuths

### Live

The four proofs of §11, against cached windows.

### Regression

- Existing 195 tests
- **`test_velocity.py:60` and `test_velocity.py:139` rewritten** against
  realistic split-cut structure. They currently pass while the feature is broken
  in production and will certify the next defect too.

## 13. Open dependencies

**Park et al. 2009 membership function parameters.** These are not currently in
hand. Obtaining the published values is a task in the implementation plan.
Parameters must not be invented and labelled as published. If the values cannot
be obtained, the honest fallback is ARW-tuned parameters explicitly marked as
such under §10 provenance.

**SPC storm report case selection.** Specific tornado and large-hail events with
confirmed reports and corresponding cached or retrievable volumes must be chosen
before proof 4 can be written.

**Level III and MRMS access.** Retrieval paths for cross-check data must respect
the architectural rule that only the Ingest Manager makes network calls.

## 14. Success criteria

The spec is complete when:

1. All four proofs pass.
2. Benchmark re-baseline shows fragmentation and heading instability unchanged
   or reduced.
3. Rotation `sweep_count` distribution demonstrates multi-sweep confirmation
   engaging on real data.
4. Site selection reaches 345 km rather than 306 km.
5. No membership function parameter lacks a provenance marker.
6. Full test suite green, with the two synthetic velocity tests rewritten.
7. `devspec/06` corrected.
8. PROGRESS.md updated, including correction of the Phase 3 multi-sweep
   confirmation claim.
