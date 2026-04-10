# ARW Phase 2 Tracker Refactor Design Spec

## Overview

This design replaces the current single-module tracker with a staged tracking pipeline that can support dense convective scenes, motion-field guidance, global association, confidence-aware motion, and live-regression validation without repeated structural rewrites.

The immediate trigger for this work is live validation:

- one live merge/split regression case exposed duplicate merge events and self-merges in the prior tracker
- one live dense-scene case exposed implausible motion values even after merge/split bookkeeping improved

This refactor is explicitly generic. No radar-site-specific rules are permitted.

## Goals

- separate tracking concerns into stable modules
- support crowded-scene association better than overlap-first greedy matching
- add scene-level motion guidance
- make motion output confidence-aware
- preserve current API contracts where practical
- institutionalize live validation as part of development

## Non-Goals

- full dual-pol nowcasting
- site-specific tuning tables
- changing the public API unless required for correctness
- replacing the current reflectivity detector outright in the first pass

## Design Principles

- Favor architecture that survives later upgrades over short-term patching
- When live data contradicts mocks, fix the design/tests around the live behavior
- Suppress uncertain motion rather than speaking a wrong speed/direction
- Treat segmentation, association, motion estimation, and event generation as separate stages

## Pipeline

```text
Buffered scans
  -> segmentation adapter
  -> motion-field estimation
  -> global association
  -> track state update
  -> merge/split event normalization
  -> track-level motion + confidence
  -> API / speech summary rendering
```

## Module Responsibilities

### `src/tracking/types.py`

Shared internal dataclasses and helper enums for:

- segmented storm objects
- track state
- association candidates and scores
- motion-field outputs
- motion-confidence outputs
- normalized merge/split events

### `src/tracking/segmentation.py`

Adapter between `src.detection` and tracking.

Responsibilities:

- wrap current connected-component detection output in a tracking-friendly structure
- expose stable masks, centroid, area, intensity, and geometry metadata
- leave room for future multi-threshold or hierarchical storm objects

Initial implementation:

- use the existing detector as the source of truth
- do not change endpoint-visible object output in this stage

### `src/tracking/motion_field.py`

Scene-level background motion estimator.

Responsibilities:

- consume two or more successive scans
- estimate bulk displacement guidance for reflectivity features
- provide predicted object displacement for association
- expose quality/confidence metrics

Initial implementation target:

- interface designed so the underlying method can evolve from simple cross-correlation to optical-flow-like methods without changing callers

### `src/tracking/association.py`

Global track/object association engine.

Responsibilities:

- build the cost matrix for prior tracks vs new objects
- score candidate matches using multiple features
- solve primary one-to-one assignment globally
- identify merge/split candidates after primary assignment
- return normalized assignment decisions for the tracker state machine

Scoring inputs:

- mask overlap
- centroid distance
- predicted position from motion field
- area change
- peak intensity change
- optional heading continuity if track motion confidence is sufficient

### `src/tracking/events.py`

Merge/split event normalization and validation.

Responsibilities:

- generate normalized event payloads
- prevent duplicate IDs in `involved_track_ids`
- prevent self-merges
- centralize event description formatting

### `src/tracking/motion.py`

Track-level motion and confidence estimation.

Responsibilities:

- estimate motion from track history
- score confidence from history length, jump consistency, residual error, and assignment quality
- classify outputs as moving, stationary, nearly stationary, or uncertain

Rules:

- absurd output is worse than no output
- if motion is uncertain, it must be downgraded or withheld

### `src/tracking/tracker.py`

Top-level orchestrator.

Responsibilities:

- maintain tracker state across scans
- reset on site change
- call segmentation, motion-field, association, state update, event normalization, and motion modules
- expose active tracks, all tracks, and recent events

## Data Contracts

### Segmented Storm Object

Minimum required fields:

- `object_id`
- `mask`
- `centroid_lat`
- `centroid_lon`
- `distance_km`
- `bearing_deg`
- `area_km2`
- `peak_dbz`
- `peak_label`
- `bbox` or equivalent geometry summary
- optional parent/child threshold structure placeholder

### Track State

Minimum required fields:

- `track_id`
- `status`
- `positions`
- `peak_history`
- `current_object`
- `first_seen`
- `last_seen`
- `identity_confidence`
- `motion_confidence`
- `merged_into`
- `split_from`
- assignment-quality metadata for the most recent update

### Association Candidate

Minimum required fields:

- `track_id`
- `object_id`
- `overlap_score`
- `distance_score`
- `predicted_position_score`
- `area_change_score`
- `intensity_change_score`
- `total_cost`

## Association Design

### Primary Assignment

Primary matching is strict one-to-one track-to-object assignment.

Why:

- this avoids the core corruption pattern that caused duplicate/self-merge failures
- many-to-one and one-to-many should be handled after primary assignment as explicit structural events

### Merge Detection

After primary assignment:

- if one new object has strong compatibility with multiple prior tracks, and at least one prior track remains unassigned because of the one-to-one constraint, classify the situation as a merge candidate
- pick a surviving track based on best continuity score, not just raw overlap
- mark the others merged
- emit one normalized merge event

### Split Detection

After primary assignment:

- if one prior track has strong compatibility with multiple new objects, classify as a split candidate
- keep the parent with the highest continuity child
- create child tracks for the remaining pieces
- emit one normalized split event

### Cost Model

The total association score should be a weighted combination of:

- geometric overlap
- distance from predicted location
- raw centroid distance
- relative area change
- relative intensity change
- optional heading consistency

The design must permit changing weights without changing the calling interface.

## Motion-Field Design

The motion field is guidance for association, not the only public motion estimate.

It should provide:

- background displacement vector or field
- confidence / quality score
- helper for projecting a track’s expected next position

The motion-field module should be replaceable later if a better method is adopted.

## Motion Confidence Design

Track-level motion must not be spoken or treated as reliable when any of the following are true:

- history length is too short
- recent assignments are low-confidence
- displacement is inconsistent across scans
- heading oscillates sharply
- computed speed exceeds configured physical plausibility thresholds

Output states:

- `stationary`
- `nearly stationary`
- `moving`
- `uncertain`

The summary layer should not speak a direction/speed when motion is `uncertain`.

## API and Summary Behavior

### Preserve if possible

- `/summary/{site_id}`
- `/tracks/{site_id}`
- `/motion/{site_id}/{track_id}`

### Internal additions

Internally, track models should carry:

- identity confidence
- motion confidence
- most recent association quality

Public exposure decision:

- confidence may remain internal initially if keeping API compatibility is more important
- summary generation must still use it even if clients do not see it yet

### Summary Rules

- speak motion only when confidence is acceptable
- if uncertain, say `tracking uncertain` or omit motion text
- continue speaking count, strongest intensity, location, and event notes

## Migration Strategy

### Stage 1

- create tracking package and shared types
- keep current behavior as much as possible

### Stage 2

- move tracking orchestration into new package
- preserve current endpoints

### Stage 3

- replace greedy association with global association
- add motion-field guidance

### Stage 4

- switch summaries to confidence-aware motion
- add live replay harness

## Live Validation Requirements

These cases must be rerun during the refactor:

### Known Merge/Split Regression Case

Purpose:

- validate that duplicate/self-merge failures stay fixed

Must hold:

- no duplicate merge event IDs
- no self-merges
- plausible active-track counts

### Dense Convective Scene

Purpose:

- validate that dense-scene motion is no longer absurd

Must hold:

- no spoken/returned motion with implausible storm speeds
- dense scenes still produce summaries and tracks
- merge/split behavior remains sane under crowding

### Lower-Complexity Site

Purpose:

- ensure changes do not degrade simple scenes

Suggested behavior:

- clear or lightly populated site should still produce stable summaries with low noise

## Testing Strategy

### Unit Tests

- cost component tests
- global assignment tests
- merge/split normalization tests
- motion confidence tests
- segmentation contract tests

### Smoke Tests

- server startup
- `/summary`
- `/tracks`
- `/motion`

### End-to-End Tests

- synthetic crowded scene
- synthetic split/merge scene
- uncertain-motion summary behavior

### Live Tests

- replay helper for a known merge/split regression case
- replay helper for a dense convective scene
- current-scan summary checks on a lower-complexity site

## Risks

- dense segmentation itself may still be too unstable for perfect object tracking
- background motion-field quality may be weak if scan quality is noisy
- a stricter motion-confidence policy may reduce spoken motion frequency before it improves quality

## Decision

Proceed with the refactor toward a staged tracking pipeline, global association, motion-field-guided matching, and confidence-aware motion. This is the least rewrite-heavy path because it directly targets the architecture used by more mature radar tracking systems instead of extending the current monolith further.
