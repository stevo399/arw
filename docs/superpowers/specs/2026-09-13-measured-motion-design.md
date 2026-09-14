# Measured storm motion: design and plan (2026-09-13)

## Owner requirements and decisions

- Everything reported must be factual. Motion that has not been measured must not be stated as
  fact (2026-09-13).
- A storm with one tracked position reports nearby storms' measured motion, spoken as such, or
  "motion not yet known" when no nearby storm has one (owner decision).
- Test pattern matching before building (owner decision). Results are in
  `docs/test_reports/2026-09-13-pattern-motion-evaluation.md`; guarded pattern matching was the
  most accurate measurement tested.

## Defects this replaces

1. **The scene-wide motion estimate.** It quantizes shifts to 1 km and 2 degrees and derives
   heading from a centroid of ray indices, so its heading depends on beam order. Reported motion
   falls back to it.
2. **Motion from track centres.** Centres jitter about 2 km per scan and jump at merges and
   splits. A two-position velocity predicts worse than assuming no motion.
3. **A storm with one position is reported "stationary"** with high confidence.
4. **Per-storm local motion windows** are bounding boxes in ray and gate indices, which break for a
   storm straddling ray 0.
5. **The "nearly stationary" threshold (2 km/h)** lies inside the measurement noise.

## Design

### Measurement: `src/tracking/pattern_motion.py` (new)

- `GroundSampler`: nearest-gate lookup from radar-centred ground coordinates for one sweep, using
  that sweep's own azimuths, so beam order and ray 0 do not matter.
- `match_storm(current_sweep, storm_mask, previous_sweep, hours) -> PatternMatch | None`:
  - The template is every echo within 10 km of the storm in the current scan, on a 0.5 km ground
    grid.
  - It is searched for in the previous scan within 150 km/h, by zero-mean normalized
    cross-correlation with sub-pixel peak refinement.
  - Returns east and north velocity in km/h, the correlation, and whether the peak lies on the
    search edge.
- `guard_matches(matches, centres) -> dict[track_id, (east_kmh, north_kmh)]`:
  - drops edge peaks;
  - where a storm has at least 3 other matches within 60 km, drops it if it is more than 25 km/h
    from their vector median.
- `nearby_velocity(track_id, accepted, centres)`: the vector median of accepted velocities of other
  storms within 60 km.

### Reporting: rules in `src/tracking/motion.py`

Each active track seen in a scan gets its guarded velocity appended to
`Track.measured_velocities` (at most 3 kept). Its reported motion is chosen by the first rule that
applies:

1. **Measured.** The track has 2 or more positions and at least one accepted velocity. Report the
   component-wise median of its last up to 3 velocities.
   - Source `pattern_match`.
   - Confidence high (0.9) with 3 velocities, medium (0.6) with fewer.
2. **Nearby storms.** Report nearby storms' velocity.
   - Source `nearby_storms`, confidence medium (0.5).
   - Never used for storm-relative velocity, which requires 0.9.
3. **Unknown.** Heading label `unknown`, source `not_measured`, confidence low (0.0).

Speeds below 6 km/h are reported as "nearly stationary". A single guarded velocity changes between
consecutive scans by a median of 3.8 km/h, which bounds the per-component noise at about 2.3 km/h,
and 6 km/h is the 95% bound for a zero velocity under that noise. "Stationary" is never reported.

### Association

- Track prediction uses the track's reported velocity (zero when unknown).
- Advected overlap shifts the earlier mask by that displacement, converted to rays and gates at the
  track's centre.
- Reacquisition uses the track's velocity times the elapsed time.
- `motion_field.py` and the scene and local phase-correlation estimates are removed.

### Speech and API

- `unknown` -> "motion not yet known".
- `nearby_storms` -> "likely moving NE at 20 mph with nearby storms", or "likely nearly stationary
  like nearby storms".
- The API motion model reports the new sources and labels.
- `TRACKER_STATE_VERSION` becomes 2, so persisted state built by the old motion rules is rebuilt
  (the `continuity_rebuilt` path) rather than mixed.

## Plan

1. **Pattern measurement module with unit tests:**
   - synthetic shift;
   - storm straddling due north;
   - edge rejection;
   - neighbour consistency rejects an outlier;
   - nearby velocity.
2. **Motion rules, `VelocitySample` type, `Track.measured_velocities`, state version 2, with unit
   tests:**
   - each rule;
   - the nearly-stationary threshold;
   - records round trip.
3. **Tracker integration and removal of the old motion resolution, with tracker tests on synthetic
   moving scans:**
   - an established storm reports measured motion;
   - a new storm with neighbours reports nearby storms' motion;
   - a lone new storm reports unknown.
4. **Association guidance from measured motion; `motion_field.py` removed; association and
   reacquisition tests updated.**
5. **Speech and API wording, with tests.**
6. **Verification:**
   - full suite;
   - a real-data proof that reported motion predicts the next position better than assuming no
     motion on a cached window;
   - tracking benchmark with every difference explained;
   - reacquisition survey rerun;
   - live check;
   - report and `PROGRESS.md`.

## Amendments from verification (2026-09-13)

The benchmark and the all-windows check found false matches the evaluated guards let through. Each
rule below was added from evidence recorded in
`docs/test_reports/2026-09-13-isolated-storm-motion-evaluation.md`:

- Matches are within 150 km/h in any direction, not per axis (the square window reached 190 km/h).
- Velocities measured more than 20 minutes ago are not reported.
- A storm with fewer than 3 judging neighbours needs correlation of at least 0.6.
- A match rejected by these rules is accepted when it agrees within 10 km/h with the storm's
  previous valid match (a stationary echo among moving storms).
- No match beyond 350 km from the radar is computed or accepted.
- Any match faster than 80 km/h needs that corroboration.

Results: `docs/test_reports/2026-09-13-measured-motion.md`.
