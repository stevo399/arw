# Tracking System Report

Date: 2026-04-11

## Purpose

This report explains how the current ARW tracking system works in code, what design choices it is using today, how those choices align with established storm/object-tracking methodology, and what evidence we have from live replay validation.

This tracker is intentionally generic. It does not contain radar-specific rule tables or site-specific hardcoded behavior. The working assumption is that any fix must survive across different scenes and radar domains rather than patching one problematic feed.

## Executive Summary

The current tracker is a staged reflectivity-object pipeline:

1. detect reflectivity objects and produce stable masks
2. estimate scene-level motion between scans
3. globally associate prior tracks to new objects
4. normalize merge and split events after primary assignment
5. compute reportable motion with confidence-aware fallbacks
6. maintain a primary-focus track for summary stability
7. render summaries and API payloads from track state

Architecturally, this is much closer to mature storm-tracking practice than the earlier monolithic overlap-first tracker. In particular, three choices matter:

- association is solved globally, not greedily
- split and merge handling happens after primary one-to-one assignment
- public motion is filtered through confidence and plausibility gates rather than speaking every raw displacement

Those choices line up well with published approaches used in radar and atmospheric feature tracking, especially ETITAN's multithreshold segmentation plus motion-field guidance, Raut et al.'s first-guess motion plus global assignment, and `tobac`'s modular detect/segment/track pipeline with explicit split/merge post-processing.

## Current Pipeline

### 1. Detection and segmentation

The detector in `src/detection.py` starts with a 20 dBZ connected-component mask and computes per-object centroid, area, peak intensity, and layer information. It then applies two additional controls intended to reduce false object structure:

- conservative multi-core splitting inside a parent low-threshold blob
- filtering of very small weak echoes before they become standalone tracked objects

Current thresholds:

- base detection threshold: `20.0` dBZ
- split-seed thresholds: `50.0`, `60.0` dBZ
- minimum seed size: `6` pixels
- weak-object suppression: drop objects smaller than `8.0 km^2` if peak intensity is below `40.0` dBZ

Operationally, the detector first identifies a parent blob, then checks whether it contains multiple intense internal cores. If yes, it partitions the blob around those cores. If no, it keeps the blob whole. That is a conservative multithreshold segmentation approach rather than a simple "every lobe becomes a storm" rule.

Implementation references:

- detection thresholds and object filters: `src/detection.py:6-11`
- small weak-object suppression: `src/detection.py:147-150`
- multi-core seed detection and parent partitioning: `src/detection.py:193-259`
- detection entry point returning masks and labeled grid: `src/detection.py:262-306`

Why this matters:

- crowded scenes often produce false merges if adjacent cores are never separated
- simpler scenes often produce false splits if weak fragments are allowed to survive as independent storms

### 2. Tracking-friendly segmentation adapter

`src/tracking/segmentation.py` wraps detector output into tracking-oriented objects with masks and geometry metadata. This keeps the tracking stages decoupled from the raw detector implementation and leaves room for future detector upgrades without rewriting the association layer.

### 3. Scene-level motion guidance

`src/tracking/motion_field.py` now prefers full-scan phase correlation over object-centroid drift when estimating background displacement between scans. The implementation:

- thresholds low signal out of the reflectivity grid
- downsamples the grid for efficiency
- computes FFT-based phase correlation between successive scans
- extracts a bulk row/column displacement
- converts that shift into geographic motion
- reports a quality score for downstream gating

If phase-correlation quality is poor, the system falls back to an object-weighted centroid displacement estimate. The motion field is used as association guidance and as a fallback motion source for low-confidence tracks.

This is an important design choice. A scene-level first guess is more robust in crowded reflectivity fields than relying only on a track's last centroid jump, especially when segmentation changes from scan to scan.

Implementation references:

- preprocessing and downsampling: `src/tracking/motion_field.py:31-41`
- FFT phase-correlation estimate: `src/tracking/motion_field.py:75-116`
- object-centroid fallback: `src/tracking/motion_field.py:119-140`
- preferred scan-to-scan geographic estimate: `src/tracking/motion_field.py:143-202`

### 4. Global association

`src/tracking/association.py` builds a cost matrix between active tracks and newly detected objects. Each candidate pair is scored using:

- mask overlap
- raw centroid distance
- distance from the predicted location after applying the motion field
- relative area change
- relative intensity change

The tracker then uses Hungarian assignment (`linear_sum_assignment`) to solve the primary one-to-one matching globally. This is a major upgrade over greedy matching because it prevents one early local decision from corrupting the rest of the scan.

The association layer also tracks:

- `primary_matches`
- `merge_candidates`
- `split_candidates`
- unmatched tracks and objects
- the motion-field estimate and scan time delta used for prediction

Implementation references:

- association result contract: `src/tracking/association.py:19-28`
- candidate scoring inputs and weights: `src/tracking/association.py:39-85`
- motion-guided association flow: `src/tracking/association.py:88-172`
- merge and split candidate derivation after primary assignment: `src/tracking/association.py:174-203`

### 5. Merge and split handling

Merge and split logic is intentionally not part of the primary assignment itself.

The system first solves one-to-one continuity, then derives structural events:

- merge candidate: multiple prior tracks are highly compatible with one new object
- split candidate: one prior track is highly compatible with multiple new objects

`src/tracking/events.py` normalizes these events so that:

- duplicate IDs are removed
- survivor IDs do not appear in their own merged list
- self-merges cannot be emitted

This is the right failure boundary. The previous class of duplicate/self-merge bugs came from allowing track reuse and event generation to get entangled.

Implementation references:

- merge event normalization call site: `src/tracker.py:111-115`
- split handling in tracker update: `src/tracker.py:148-169`
- merge handling in tracker update: `src/tracker.py:171-191`

### 6. Track-state updates

`src/tracker.py` owns track lifecycle and scan-to-scan state. It:

- creates tracks on first appearance
- updates active tracks from the association result
- marks merged tracks as absorbed
- creates child tracks for split products
- degrades confidence for missed tracks
- marks tracks lost after repeated misses

It also stores:

- `identity_confidence`
- `motion_confidence`
- `last_motion`
- `diagnostic_motion`
- `is_primary_focus`

That separation between identity confidence and motion confidence is important. A track can exist without being trustworthy enough to publish an exact heading and speed.

Implementation references:

- track creation and identity initialization: `src/tracker.py:45-57`
- motion refresh for active tracks: `src/tracker.py:59-72`
- missed-scan decay and loss logic: `src/tracker.py:210-217`

### 7. Confidence-aware reported motion

`src/tracking/motion.py` computes motion from track history, but it does not blindly publish it.

The motion resolver currently combines three sources:

- track-history motion
- motion-field motion
- suppression when neither is trustworthy enough

The resolver uses:

- a plausibility cap of `160 km/h`
- scan-to-scan jump consistency checks
- regression residual checks across the track history
- identity-confidence thresholds
- history-versus-field disagreement checks

Decision flow:

- if history strongly disagrees with a strong motion field, prefer the field
- if track identity and history quality are both good, publish history motion
- if identity is weak but the motion field is strong enough, publish field motion
- otherwise suppress motion and report `tracking uncertain`

This is the correct bias for a radar monitoring app. Wrong motion is more damaging than withheld motion.

Implementation references:

- motion plausibility and confidence thresholds: `src/tracking/motion.py:11-23`
- history-motion confidence logic: `src/tracking/motion.py:60-99`
- public motion from track history: `src/tracking/motion.py:102-166`
- field-derived motion and suppression: `src/tracking/motion.py:169-217`
- history-versus-field disagreement and final resolution: `src/tracking/motion.py:227-291`

### 8. Focus selection and summary stability

`src/tracker.py` and `src/summary.py` now maintain a primary-focus track so the summary does not re-elect a "strongest object" from scratch each scan. Focus scoring combines:

- track persistence
- identity confidence
- object area
- peak intensity
- prior-focus bonus
- distance penalty

The tracker also applies a handoff margin (`FOCUS_SWITCH_MARGIN = 2.0`) so minor score fluctuations do not cause focus flapping. The summary layer follows the primary-focus track directly when present.

This does not make the summary "stick forever." It makes the focus stateful and harder to steal on weak evidence.

Implementation references:

- focus switch margin: `src/tracker.py:9-10`
- focus scoring inputs: `src/tracker.py:74-84`
- hysteretic focus update: `src/tracker.py:86-109`
- summary selection preferring primary focus: `src/summary.py:42-68`

### 9. Summary and API rendering

The public outputs still come from the same top-level API surfaces:

- `/summary/{site_id}`
- `/tracks/{site_id}`
- `/motion/{site_id}/{track_id}`

The summary renderer in `src/summary.py` now:

- reports scene-wide coverage, not just focal-object area
- uses the primary-focus track when available
- says `tracking uncertain`, `stationary`, or `nearly stationary` when appropriate
- includes merge and split counts from the latest scan

Implementation references:

- motion formatting rules: `src/summary.py:71-81`
- scene-wide coverage and summary rendering: `src/summary.py:84-134`

## How This Maps To Published Practice

### Modular detect/segment/track pipeline

The current architecture closely matches the modular design described in the `tobac` literature: feature detection, segmentation, and tracking are treated as separate but connected stages. That matters because split/merge behavior, motion quality, and object identity are easier to reason about when those concerns are not fused into one large tracker function.

Our code reflects that directly:

- detection in `src/detection.py`
- segmentation adapter in `src/tracking/segmentation.py`
- motion field in `src/tracking/motion_field.py`
- association in `src/tracking/association.py`
- event normalization in `src/tracking/events.py`
- orchestration in `src/tracker.py`

### Multithreshold segmentation instead of pure single-threshold blobs

ETITAN explicitly improves on centroid-only storm tracking by using multithreshold identification and morphology to isolate storm cells inside crowded regions. Our detector is not a full ETITAN reproduction, but the direction is aligned:

- detect a broad reflectivity object at a lower threshold
- look for stronger internal cores at higher thresholds
- split only when multiple meaningful cores are present

That is exactly the class of segmentation needed to reduce false mergers caused by adjacent intense cells living inside one low-threshold blob.

### First-guess motion plus global assignment

Raut et al. describe a tracking pattern built around:

- a first-guess motion estimate
- candidate generation around the predicted position
- a disparity or cost metric
- Hungarian assignment for globally optimal primary matching
- split/merge processing after the primary linking stage

That is very close to the current ARW association design. We use a scene-level motion estimate to predict the likely next location, score multiple compatibility features, solve the one-to-one match globally, then derive structural events afterward.

### Split/merge handling as post-processing

The newer `tobac` split/merge work is explicit that split and merger processing is a distinct post-processing step applied after initial feature linking. That is also the current ARW design, and it is the right design if the goal is to avoid tracker-state corruption. The primary association result should establish continuity first; event logic should annotate structural change afterward.

### Conservative public motion

Published storm/object tracking systems routinely distinguish between internal guidance and externally reported certainty. ETITAN, for example, uses a motion-vector field for forecasting while still constraining how storms are linked. ARW follows the same principle:

- the motion field helps guide association
- the reportable motion depends on track identity confidence and plausibility checks
- low-confidence tracks can fall back to field motion or suppression

That is preferable to a naive "every centroid delta becomes spoken motion" approach.

## Source-Backed Limits And Recommendations

The literature supports the current staged direction, but it also points to several areas where the present implementation is still a simplified version of stronger practice.

### 1. Preprocessing and quality control are still thin

The current ARW pipeline moves from reflectivity ingest into object detection without a dedicated QC stage for non-meteorological echoes, edge artifacts, or explicit scan-quality penalties. WDSS-II-oriented operational practice places more emphasis on input conditioning and quality control before higher-level identification and tracking products are produced.

That does not mean ARW is architecturally wrong. It means the next upgrade path should add a real preprocessing stage ahead of detection rather than trying to patch poor-quality input later in the tracker.

### 2. Segmentation should probably evolve toward a fuller hierarchy

The current detector uses a low-threshold parent object plus conservative high-threshold core splitting. That is materially better than a single threshold alone, but it is still simpler than the richer multithreshold or hierarchical segmentation approaches described in ETITAN-style work and in later modular tracking systems.

This is a source-backed recommendation for future work:

- keep the current staged segmentation architecture
- move the detector toward a multi-level object hierarchy rather than a one-off parent-split heuristic

The literature supports the direction, but it does not dictate one exact threshold ladder or one exact hierarchy policy for this codebase. Those choices still need empirical tuning against live replay behavior.

### 3. One bulk scene-motion estimate is a reasonable baseline, not an endpoint

The current motion-field stage uses one full-scene phase-correlation estimate. That is a reasonable first-guess motion model and is much better than publishing raw centroid jumps from unstable tracks.

However, published operational and research tracking work also makes clear that tracking becomes harder when storms in different parts of the domain do not share one common motion regime. The strongest defensible conclusion here is:

- a single scene-level motion prior is a sound baseline
- it should not be assumed to be universally sufficient in spatially heterogeneous convection

It is reasonable to consider local or regional motion guidance later, but that should be described as an engineering recommendation informed by operational limitations, not as something this report claims is already mandated by the cited papers.

### 4. Geometry should continue to matter more than centroid continuity alone

One of the clearer lessons from improved storm-tracking literature is that area and shape continuity matter, especially in crowded or irregular convection. ARW already preserves masks and uses overlap in the cost model, which is good. The likely next improvement is to make advected geometry a stronger part of the association decision, rather than relying mainly on raw overlap plus centroid-style distances.

That recommendation is grounded in the literature's general direction, but the exact feature set for ARW, such as advected IoU or other shape descriptors, remains an implementation choice to be validated empirically.

### 5. Validation needs quantitative metrics, not only replays

The live replay harness is the right operational guardrail and should stay. It catches failure modes synthetic tests missed, which was a major lesson of this refactor.

But replay sanity alone is not enough to claim best practice. Stronger practice would add quantitative tracking evaluation, for example:

- identity-switch counts
- false merge rates
- false split rates
- fragmentation rates
- motion error on adjudicated cases

That recommendation is methodological rather than ARW-specific. The tracker should continue using live replay for realism, but it should eventually pair that with explicit scoring.

## What The Tracker Is Not Doing

The current implementation is materially stronger than the earlier tracker, but it is still not a full research-grade storm analysis system. In particular:

- the motion field is a bulk scene estimate, not dense optical flow
- segmentation is still reflectivity-only and still threshold-based
- split/merge logic is based on candidate compatibility around the primary assignment, not a full lineage graph
- there is no site-specific tuning layer, by design

Those are acceptable limits for the current stage because the architecture can absorb better segmentation or motion models later without another major rewrite.

## Validation Evidence In This Repo

Recent commits show a coherent tracker progression rather than ad hoc patching:

- `c22fb66` `refactor: scaffold tracking package and design docs`
- `a45bace` `refactor: add tracking segmentation adapter`
- `408f657` `feat: add tracking motion field baseline`
- `bd84770` `feat: add global track association`
- `771e945` `feat: add confidence-aware track motion`
- `f56b33b` `feat: expose confidence-aware tracking via API`
- `e2cef22` `feat: add live replay regression harness`
- `51493a0` `fix: stabilize summary coverage and finalize tracker validation`
- `e5c76e4` `feat: add field-guided reported motion`
- `f1406cc` `feat: add fast local replay mode`
- `eacd0c6` `feat: add conservative multi-core segmentation`
- `2172e57` `refine: filter weak fragmented echoes`
- `8afa8f2` `feat: stabilize summary focus across scans`
- `1559476` `refine: add focus handoff hysteresis`
- `bf34a62` `improve: use recent cached scans for local replays`

Live replay evidence recorded in `docs/test_reports/2026-04-10-live-replay-harness.md` and `PROGRESS.md` shows:

- duplicate and self-merge regressions remain fixed on live replay
- dense-scene spoken motion no longer exposes the earlier absurd outliers
- simpler scenes produce less object-count noise after weak-fragment filtering
- summary focus is materially more stable across crowded replay windows
- cached local-only replays remain usable for regression testing when partial historical data is already on disk

That validation pattern matters as much as the code structure. The tracker is being checked against real radar scans, not only synthetic unit tests.

Repo references:

- completed tracker capabilities and remaining validation work: `PROGRESS.md:3-39`
- generic staged-pipeline design requirements: `docs/superpowers/specs/2026-04-10-phase2-tracker-refactor-design.md:3-396`
- live replay validation history and representative outputs: `docs/test_reports/2026-04-10-live-replay-harness.md:1-423`

## Assessment

The current tracking system is not a one-off patchwork anymore. It is now a generic staged tracker with:

- conservative multithreshold segmentation
- scene-level motion guidance
- global association
- post-assignment split/merge normalization
- confidence-aware motion publication
- stateful summary focus selection

That architecture is well aligned with established storm/object tracking practice and is a reasonable foundation for later upgrades. The most source-defensible next steps are:

- add a real preprocessing and quality-control stage ahead of detection
- evolve segmentation toward a fuller multilevel hierarchy
- strengthen geometry-aware association features
- move event bookkeeping toward richer lineage state over time
- add quantitative evaluation alongside live replay

Those are staged upgrades to the existing architecture, not an argument to scrap it and restart.

## Sources

### Code and repo sources

- `PROGRESS.md`
- `docs/superpowers/specs/2026-04-10-phase2-tracker-refactor-design.md`
- `docs/test_reports/2026-04-10-live-replay-harness.md`
- `src/detection.py`
- `src/tracking/motion_field.py`
- `src/tracking/association.py`
- `src/tracking/motion.py`
- `src/tracker.py`
- `src/summary.py`

### External methodology sources

1. Raut, B. A., et al. (2021). "An Adaptive Tracking Algorithm for Convection in Simulated and Remote Sensing Data." *Journal of Applied Meteorology and Climatology*.
   - DOI: https://doi.org/10.1175/JAMC-D-20-0119.1
   - OSTI entry: https://www.osti.gov/pages/biblio/1863221

2. Han, L., et al. (2009). "3D Convective Storm Identification, Tracking, and Forecasting: An Enhanced TITAN Algorithm." *Journal of Atmospheric and Oceanic Technology*.
   - DOI: https://doi.org/10.1175/2008JTECHA1084.1
   - Repository summary: https://ir.pku.edu.cn/handle/20.500.11897/396606

3. Sokolowsky, G. A., et al. (2024). "`tobac` v1.5: introducing fast 3D tracking, splits and mergers, and other enhancements for identifying and analysing meteorological phenomena." *Geoscientific Model Development*.
   - Article: https://gmd.copernicus.org/articles/17/5309/2024/gmd-17-5309-2024.html
