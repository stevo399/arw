# Truthful rotation detection (#9) and a calibrated gate classifier (#8)

Date: 2026-09-14. GitHub issues #8 and #9.

**Status (2026-09-14):**
- **Part A (rotation):** implemented and merged. Owner decision 3 (live rotation "after validation")
  is only partly met: analysis runs live, but no evidence level is spoken. Persistent evidence passed
  the corpus but failed the live check. Details: `docs/test_reports/2026-09-14-rotation-corpus.md`.
- **Part B (classifier):** not started.

## Owner decisions (2026-09-14)

1. **Classifier truth:** fit to labels built from independent physical evidence, and cross-check
   against the NWS Level III hydrometeor classification on the same volumes.
2. **Debris:** stop reporting it until enough confirmed tornado-debris cases exist to calibrate it.
3. **Live rotation:** after validation, the live server runs velocity and rotation analysis so
   evidence reaches speech and the map.
4. **Rotation truth:**
   - confirmed events from SPC storm reports and NWS surveys (hits and misses);
   - NWS Level III mesocyclone (and TVS, where published) detections on the same volumes, as an
     operational cross-check;
   - clear-air scans for false alarms.

## Current state (measured 2026-09-14)

**Rotation:**
- **Raw candidates:** 120 to 130 per clear-air KIWA scan (755 over six scans); 59 on the Moore EF5
  volume, 8 of them "vertically confirmed". Nobody has checked whether those 8 are at the tornado.
- **Physics defect:** a candidate is any opposite-signed pair of neighbouring gates in all 8
  directions. A sign change along the beam (same ray, next gate) is convergence or divergence, not
  rotation. Rotation is azimuthal shear: opposite signs on adjacent rays at the same range.
- **Aliasing:** the live path uses velocity that has not been dealiased, so a fold from +25 to
  -25 m/s looks like 50 m/s of shear.
- **Weak vertical confirmation:** "vertically confirmed" means candidates on two tilts within
  10 km of each other.
- **Not live:** `_process_scan_file` sets regions and rotations to empty lists for speed. Speech
  and maps show no rotation evidence.
- **Benchmarks and replays are misleading:** `scripts/evaluate_tracking.py` and
  `scripts/live_replay.py` call `analyze_velocity` without masks, taking a legacy path that attaches
  unassessed candidates to any storm within 30 km. The resulting summary text is not what
  production says.

**Classifier:**
- **Unvalidated parameters:** all 18 membership parameters are `arw-tuned-initial`.
- **Debris condemned:** 27.4% of the confirmed Moore debris signature is labelled clutter or
  biological (strict xfail).
- **No separation without velocity:** biological and clutter separate by only 1.05× (strict xfail).
- **Where it reaches users:** object `class_fractions`, precipitation-layer band evidence, and the
  quality advisory.

**Data availability (checked):**
- **Level III on AWS** (`unidata-nexrad-level3`): HHC (hybrid hydrometeor classification) and N0H
  (0.5 degree classification) exist for the 2026 cached dates, and Py-ART reads both. NMD
  (mesocyclone detection) exists but is a binary product Py-ART does not parse. Nothing for 2013.
- **SPC reports** are reachable. On cached dates: 9 tornado reports, about 90 hail reports, about 440
  wind reports. None of the tornado reports is within 230 km and 30 minutes of a cached Level II
  volume, so validation volumes must be fetched.

## Principles

- **Labels never come from the thing being scored.** A variable used to build a class label is not
  a scoring input for that class in the evaluation that judges it, or the separation is circular
  (the defect `test_proof_clutter_persistence.py` documents).
- **Held-out evaluation.** Parameters are fitted on some dates and radars and judged on others. A
  result is reported per case, never only pooled.
- **Honest scope.** Every rule change records its population, source, and the false-alarm /
  missed-signal trade-off before it changes production (the existing `calibration_policy`).
- **Network calls only in the ingest manager** (CLAUDE.md). Level III and SPC fetches are ingest
  functions; evaluation reads the cache.

## Part A: rotation (#9)

### A1. Validation corpus v2 (`docs/validation/rotation-signal-corpus.json`)

- **Tornado events:** every SPC tornado report on the chosen dates, starting with the 9 on cached
  dates. For each:
  - the nearest WSR-88D within 150 km;
  - Level II volumes from 20 minutes before to 10 minutes after the report;
  - the matching Level III NMD (and TVS where published).

  The 2013 Moore volumes stay, without Level III.
- **Severe non-tornado events:** SPC hail of 2 inches or more, and wind of 65 kt or more, from the
  same dates. These show a detector does not become a hail detector.
- **Null cases:**
  - the six clear-air KIWA scans;
  - quiet-day volumes from other radars;
  - storm volumes with no tornado, mesocyclone or TVS within 50 km.
- **Ingest:** `src/ingest.py` gains Level III and SPC report fetchers; a corpus-builder script
  fetches into the cache and writes the manifest.

### A2. Metrics and baseline

**Evaluator:** `scripts/evaluate_rotation_signal_corpus.py`, extended.
- **Event hit:** an assessment at or above a given evidence level within 10 km and 15 minutes of a
  report.
- **Operational agreement:** an assessment within 5 km of an NMD or TVS detection in the same
  volume, reported both ways (ARW-only and NWS-only).
- **False alarms:** assessments per null volume, by evidence level, and what speech would say.
- **Baseline first:** the current detector is measured on the corpus before any change.

### A3. Detector correction (each change measured on the corpus before adoption)

- **Azimuthal shear only:** opposite-signed velocity on adjacent rays at the same gate (or one gate
  either side), with the shear computed from local azimuthal separation. Pairs along the beam are
  not candidates.
- **Aliasing:** dealias before detection, or reject pairs whose difference is consistent with a
  fold (close to twice Nyquist). The choice follows cost and accuracy on the corpus.
- **Couplet structure:** both inbound and outbound magnitudes above a minimum, and a diameter
  within mesocyclone and tornado-vortex scales, chosen from the corpus trade-off rather than a
  guessed number.
- **Vertical confirmation:** candidates on different tilts overlap on the ground within their
  diameters, not merely within 10 km.
- Cell association and persistence rules stay; they are re-measured.
- **Scripts:** the benchmark and replay scripts use the same assessed path as production, so
  summaries they print are what users would hear.

### A4. Live pipeline

- Velocity and rotation analysis runs for live scans. Cost is measured on dense volumes and
  reduced where needed (for example, candidate extraction limited to cell footprints) so a first
  map is not delayed.
- Speech and map expose only evidence levels that validate. A level that does not beat null
  cases on the corpus is not spoken.

## Part B: classifier (#8)

### B1. Label builder (partial, per gate; evidence independent of the classifier)

- **Ground clutter:** the signal processor removed clutter power at the gate
  (`clutter_filter_power_removed` at or above a threshold), the gate is near-stationary, and it
  recurs at the same ground location across scans. RhoHV, ZDR, reflectivity and texture are not
  used.
- **Biological:** clear-air volumes (no SPC report within 100 km, less than 1% of gates at 35 dBZ
  or more), echo below 1.5 km, drifting with the wind, with no echo at the same ground location on
  the next tilt up.
- **Precipitation:** echo of 20 dBZ or more that is vertically continuous into the second tilt at
  the same ground location and persists into the next scan, taken from the storm-day volumes.
  RhoHV, ZDR and texture are not used.
- **Hail:** gates of 50 dBZ or more within 10 km and 10 minutes of an SPC hail report of 1 inch or
  more. This is a weak label; its size and uncertainty are reported, and it may not support
  calibration.
- **Debris:** not labelled or reported (decision 2).

### B2. Fitting and evaluation

- **Split** by date and radar into fit and held-out sets.
- **Membership functions** are fitted per class and variable from the labelled distributions:
  trapezoid breakpoints at stated percentiles. Weights come from each variable's measured
  separability on the fit set.
- **Held-out reports:**
  - confusion matrix per case;
  - the existing clear-air velocity-separation proof run without velocity (the #8 strict xfail);
  - Moore hazard gates labelled clutter or biological (the other strict xfail).
- **NWS cross-check:** agreement with HHC/N0H on the same gates, after mapping NWS classes to ARW
  classes (biological 10, ground clutter 20, rain 60/70, big drops 80, graupel 90, hail 100, unknown
  140). Recorded as agreement, not truth.
- **Provenance:** each parameter moves from `arw-tuned-initial` to `arw-fitted`, with the fit set
  named.

### B3. Debris

The debris class is removed from classification and from every exposed class fraction. The
Moore-case proof keeps asserting that hazard echo is never condemned as clutter or biological.

## Verification and delivery

- Unit tests for each detector rule and label rule, on synthetic fields.
- Real-data proofs replace the strict xfails once they genuinely pass. Each xfail is removed only
  when the defect it documents is fixed.
- Corpus reports under `docs/test_reports/`, including the trade-off tables that justify every
  threshold.
- Tracking benchmark rerun, with differences explained.
- Live check with rotation evidence spoken only at validated levels.
- Order: A1 -> A2 -> A3 -> A4, then B1 -> B2 -> B3. Rotation first: it needs no labelled gates and
  its truth sources exist today.

## Risks

- **Tornado sample.** Nine tornado reports is a small, geographically scattered sample; it
  supports a false-alarm and missed-signal trade-off, not a probability. More events can be added
  by the corpus builder.
- **Weak labels.** Hail labels, and biology labels on a single clear-air site, may be too weak to
  calibrate. If so, the report says so and the parameters stay unfitted rather than being fitted
  to noise.
- **Cost.** Dealiasing is costly (the reason live velocity is not dealiased); A3 may need a cheaper
  local fold check.
