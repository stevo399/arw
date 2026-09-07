# Rotation Signal Integrity Design

## Problem

`src.velocity.detect_rotation_signatures` currently calls every connected
collection of adjacent, opposite-signed velocity gates with at least 15 m/s
difference a rotation signature. It runs on base radial velocity, has no storm
association, no vertical or temporal continuity requirement, and merges
unrelated results within 10 km.

On six local clear-air KIWA volumes this creates 755 signatures (121--132 per
scan). Protection rule 2 then shields about 97% of the classifier-flagged
non-meteorological echo, reducing the QC advisory rate for >=20 dBZ echo to
0.39%. The result is not a usable rotation product.

This redesign is deliberately not a warning algorithm and must never imply a
tornado. It produces conservative, inspectable radar-circulation evidence for
ARW's map, speech, and QC-advisory consumers.

## Operational basis

The implementation follows these operational principles rather than adopting a
single universal delta-velocity number:

- Doppler rotation is interpreted in storm-relative velocity and with radar
  sampling limits understood; a gate-to-gate couplet alone is insufficient.
- A detected circulation must be co-located with a storm cell. ROC tropical
  cyclone guidance explicitly uses SCIT-cell proximity to reduce false alarms.
- Circulation depth across elevation slices and persistence over volume scans
  increase confidence; NWS warning-forecaster task analysis identifies 2--3
  volume scans as meaningful persistence.
- Reflectivity and dual-polarimetric evidence must be co-located with rotation;
  low correlation coefficient on its own is not debris or a tornado.

Primary sources:

- [ROC WSR-88D Tropical Cyclone Operations Plan](https://www.roc.noaa.gov/public-documents/operations-branch/2023_WSR-88D_TCOP_Build_21x_Final.pdf)
- [NWS warning-forecaster cognitive task analysis](https://training.weather.gov/wdtd/courses/woc/human-factors/expertise/cta/story_content/external_files/Hahn%20et%20al.%202003%20-%20Cognitive%20Task%20Anlaysis%20of%20the%20Warning%20Forecaster%20Task.pdf)
- [NWS dual-polarization applications guidance](https://www.weather.gov/jan/dualpolupgrade-applications)

## Design

### 1. Separate observations from assessments

`RotationCandidate` is an unfiltered, single-sweep velocity-couplet
observation. It is diagnostic-only and never reaches protection, speech, or
the public map. Its provenance includes sweep/elevation, gate pair geometry,
delta velocity, local shear, range, and data-quality flags.

`RotationAssessment` is the public/internal product. It is one of:

- `unconfirmed`: a cell-associated candidate with one supporting dimension;
- `vertically_confirmed`: spatially co-located observations on at least two
  distinct elevations;
- `persistent`: an associated assessment observed in at least two volume scans;
- `corroborated`: an assessment with cell association plus co-located relevant
  reflectivity/dual-pol evidence.

These are evidence labels, not hazard labels. The API must expose the evidence
and reason for non-confirmation rather than silently discard it.

### 2. Use physical geometry and data quality

Candidate extraction must calculate azimuthal shear from the pair's local
range, gate spacing, and ray spacing; it must not use array-wide mean range or
assume uniform azimuth spacing. It rejects folded, missing, and implausibly
isolated pairs, and preserves the underlying radial velocity values for review.

The pipeline must distinguish base radial velocity from storm-relative
velocity. When no independently supportable storm-motion estimate exists, the
assessment reports `motion_reference="base_radial"` and may not be promoted on
strength alone.

### 3. Associate before promotion

Candidates can be promoted only when spatially associated with a detected
reflectivity cell. Association uses the cell footprint, not only centroid
distance, and records the distance/overlap. A candidate in clear air or ahead
of a cell remains diagnostic-only.

This requires moving assessment after reflectivity preprocessing/detection.
Raw candidates may still be collected earlier for diagnostics, but they may not
drive protection while no cell context exists. Since QC is advisory-only, this
ordering cannot delete or hide hazard echo.

### 4. Correlate elevations and time explicitly

Vertical support is counted by distinct physical elevation, never by the number
of nearby gate pairs. Cross-elevation matching uses projected geographic
distance scaled to the candidate's diameter and records every contributing
sweep. Temporal persistence is managed by a bounded per-site association state
and only compares consecutive/nearby scans.

The first scan remains useful: it may yield an `unconfirmed` assessment, but
cannot be described as persistent or vertically confirmed without the required
evidence.

### 5. Consumer contract

- `protected_mask` no longer treats every raw couplet as a reason to suppress
  a QC advisory. Only assessments meeting a documented evidence policy may be
  passed to it, and all protection remains advisory-only.
- Spoken/map language states the evidence level: for example, "unconfirmed
  rotation near a storm" rather than "rotation" for a one-sweep candidate.
- Tornado/debris language requires its own future hazard policy; this redesign
  never infers it from a velocity couplet.

## Validation contract

Before changing a production threshold, add tests and local-volume proofs for:

1. Clear-air KIWA: zero promoted assessments; raw candidates may be retained
   only in diagnostics with their rejection reason.
2. Moore tornado volume: a cell-associated low-level candidate is retained and
   its evidence is reported. It must not be lost merely because it lacks a
   second scan or because a higher sweep samples a different circulation size.
3. Multi-sweep reference volume: vertical-support count equals distinct
   elevations, never merged couplet count, and cannot exceed velocity sweeps.
4. A temporal replay window: persistence requires separate scan timestamps and
   expires when the scan gap exceeds the configured bound.
5. QC effectiveness: clear-air advisory measurements are reported after the
   rotation protection change, while both known severe cases remain present in
   the full pipeline.

Numerical strength/range thresholds are calibration parameters, not imported
from a warning product. They will be selected only from a labelled severe/null
case corpus and recorded with source, population, and failure trade-off.

## Non-goals

- Reimplementing the operational WSR-88D TDA/MDA or claiming equivalent
  probability-of-detection/false-alarm performance.
- Issuing a tornado warning or labeling a velocity couplet a tornado.
- Treating a low rhohv area as debris without co-located circulation and storm
  context.

## Implementation status

The first two evidence layers are implemented:

- Candidate geometry now uses local gate-pair range and actual ray spacing.
- Assessment follows reflectivity-cell detection and uses the detected cell
  footprint, not a 30 km centroid shortcut.
- `vertically_confirmed` requires both cell association and at least two
  distinct elevation slices; raw/unconfirmed candidates cannot affect the QC
  advisory.
- `persistent` requires a second, distinct volume within fifteen minutes on
  the same already-associated storm track, with a circulation displacement
  compatible with the tracker's existing 120 km/h physical motion bound.
- Every rotation product currently declares `motion_reference="base_radial"`.
  Speech and map output say "velocity couplet" for unconfirmed evidence and
  do not imply storm-relative interpretation.
- Cell-associated assessments expose their colocated object's peak
  reflectivity and whether RhoHV/ZDR were available. These are context and
  provenance only; no debris/tornado inference is made from them.

Remaining work is storm-relative velocity provenance, explicit dual-pol/
reflectivity corroboration, and a labelled case corpus for calibration. None
of those should be replaced with a stronger raw-shear cutoff.

Storm-relative processing is a separate future change: the tracker currently
offers a motion vector only after it has enough history, and its confidence
must be high and its heading non-null before it can be used as a reference.
Until then, every candidate remains explicitly `base_radial`; no inferred
motion may silently change the detection input.

The implementation now contains a tested per-sweep translation-removal
primitive gated at tracker confidence >=0.90. It is deliberately not yet
wired into volume-wide candidate detection: applying one storm's motion to
another storm in the same volume would create a false storm-relative product.
For a cell-associated signature, ARW now exposes trusted storm-relative
inbound/outbound context at the signature centroid while retaining the
explicit `base_radial_detection_with_storm_relative_context` provenance.

The initial evaluation manifest is
`docs/validation/rotation-signal-corpus.json`. It deliberately separates the
KIWA clear-air null sequence from the two documented Moore severe-context
volumes and records what each can and cannot support. It is a starting
evaluation population, not a claim that three contexts are enough to tune a
warning-like product.

`docs/test_reports/2026-09-07-rotation-corpus-baseline.json` records the
first evaluator baseline: all 755 clear-air KIWA candidates remained
unconfirmed, while the two documented severe contexts retained 8 and 7
vertically confirmed assessments respectively. These measurements verify the
promotion boundary; they do not calibrate a detection threshold or establish
warning performance.
