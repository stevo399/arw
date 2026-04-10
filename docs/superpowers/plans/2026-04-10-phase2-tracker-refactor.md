# ARW Phase 2 Tracker Refactor Plan

> **For agentic workers:** Execute this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Do not skip the completion protocol at the end of each task.

**Goal:** Replace the current monolithic overlap-first tracker with a tracking architecture that can survive dense convective scenes without requiring repeated rewrites. The target is a generic radar-object tracker that separates segmentation, association, motion-field guidance, motion estimation, and event generation, while preserving the current API shape where practical.

**Why this refactor exists:** Live validation showed that the current tracker can still produce unreliable motion in crowded scenes. The current design mixes object matching, event generation, and motion estimation inside one module, which makes later improvements expensive and fragile.

**Key requirement:** This refactor must be generic. No site-specific logic, thresholds keyed to one radar, or one-off behavior tuned to any individual site.

## Mandatory Completion Protocol

Every task in this plan is only complete when all of the following are done:

- [ ] Run the relevant unit tests
- [ ] Run smoke tests for the affected API/server behavior
- [ ] Run live data fetches / API rendering / output analysis when appropriate for that task
- [ ] Confirm results are green or document the exact blocker
- [ ] Commit the changes once green
- [ ] Mark the task as done in this file

Use live validation whenever the task affects any of:

- ingest behavior
- object detection / segmentation
- track association
- motion estimation
- summary generation
- `/summary`, `/tracks`, or `/motion` endpoint behavior

## Target Architecture

Refactor tracking into explicit stages:

```text
Reflectivity scans
  -> segmentation
  -> motion-field estimation
  -> association / global assignment
  -> track state update
  -> merge/split event generation
  -> motion estimation + confidence
  -> speech/API summaries
```

## Proposed Source Layout

```text
src/
  tracking/
    __init__.py
    types.py           # Track state, association candidates, confidence models
    segmentation.py    # Adapter around detection output and future multi-threshold objects
    motion_field.py    # Background motion field estimation from successive scans
    association.py     # Global assignment / cost matrix / merge-split candidate detection
    events.py          # Merge/split event normalization and validation
    motion.py          # Track-level motion estimate + confidence + sanity gating
    tracker.py         # Orchestrates the stages and maintains tracker state
```

Compatibility goal:

- keep the existing FastAPI response contracts unless a task explicitly changes them
- allow the old `src/tracker.py` import path to delegate to the new package during migration if that reduces churn

## Success Criteria

- No duplicate merge events
- No self-merge events
- No physically absurd spoken motion values in dense scenes
- Motion output is confidence-aware and can be withheld when unreliable
- Live replay of a known merge/split regression case remains clean
- Live replay of a dense convective scene stops producing extreme false speeds

---

## Task 1: Write the Refactor Design Spec

**Goal:** Freeze the intended architecture before code churn starts.

**Files:**
- Create: `docs/superpowers/specs/2026-04-10-phase2-tracker-refactor-design.md`

- [x] Define the new tracking stages and responsibilities
- [x] Document the data contracts passed between segmentation, motion-field estimation, association, events, and motion modules
- [x] Define how global assignment should work, including merge/split candidate handling
- [x] Define confidence outputs for track identity and motion
- [x] Define live-validation cases that must pass before the refactor is considered complete
- [x] Run smoke tests if any code changed while preparing the design
- [x] Run live API/data checks if any code changed while preparing the design
- [x] Commit when green
- [x] Mark this task done in this file

---

## Task 2: Introduce Tracking Package and Shared Types

**Goal:** Decouple the current monolith without changing behavior yet.

**Files:**
- Create: `src/tracking/__init__.py`
- Create: `src/tracking/types.py`
- Create: `src/tracking/events.py`
- Create: `tests/unit/test_tracking_types.py`
- Modify: `src/tracker.py` or replace it with a compatibility shim

- [x] Create shared dataclasses / models for track state, match candidates, association scores, motion confidence, and normalized events
- [x] Move event formatting/validation rules into `src/tracking/events.py`
- [x] Leave existing runtime behavior unchanged except for import location changes
- [x] Add unit tests for the new types and event normalization
- [x] Run targeted unit tests
- [x] Run smoke tests
- [x] Run a live replay of a known merge/split regression case to confirm behavior parity after the module move
- [x] Commit when green
- [x] Mark this task done in this file

---

## Task 3: Build Segmentation Adapter for Future Multi-Threshold Objects

**Goal:** Stop coupling association directly to raw connected components.

**Files:**
- Create: `src/tracking/segmentation.py`
- Modify: `src/detection.py`
- Create: `tests/unit/test_tracking_segmentation.py`

- [x] Introduce a segmentation adapter that wraps current detection output into tracking-friendly storm objects
- [x] Preserve current detection behavior as the baseline implementation
- [x] Include hooks for future multi-threshold / hierarchical storm structure without changing endpoint outputs yet
- [x] Ensure segmentation returns stable geometry, masks, centroids, area, and peak metadata required by association
- [x] Add unit tests for segmentation contracts
- [x] Run targeted unit tests
- [x] Run smoke tests
- [x] Run live object extraction checks for at least one dense scene and one lower-complexity scene
- [x] Review live output for obviously unstable segmentation artifacts
- [x] Commit when green
- [x] Mark this task done in this file

---

## Task 4: Add Background Motion-Field Estimation

**Goal:** Introduce scene-level motion guidance so dense storms are not matched by overlap alone.

**Files:**
- Create: `src/tracking/motion_field.py`
- Create: `tests/unit/test_tracking_motion_field.py`

- [x] Implement background motion-field estimation from successive scans
- [x] Start with a design that can later support optical flow / cross-correlation without changing the tracker interface
- [x] Expose predicted displacement for track association
- [x] Define confidence/quality metrics for the motion field
- [x] Add unit tests for simple motion-field cases and degenerate/noisy cases
- [x] Run targeted unit tests
- [x] Run smoke tests if any API-visible behavior changes
- [x] Run live replay checks on one known merge/split regression case and one dense scene to inspect raw motion-field reasonableness
- [x] Commit when green
- [x] Mark this task done in this file

---

## Task 5: Replace Greedy Matching with Global Association

**Goal:** Make crowded-scene identity decisions globally instead of locally.

**Files:**
- Create: `src/tracking/association.py`
- Create: `tests/unit/test_tracking_association.py`
- Modify: `src/tracking/tracker.py` or equivalent orchestrator

- [x] Build a cost matrix for prior tracks vs new segmented objects
- [x] Include overlap, centroid distance, predicted position from the motion field, area change, and intensity change in the score
- [x] Solve 1:1 assignments globally
- [x] Detect merge and split candidates after primary assignment rather than by ad hoc reuse of the same tracks
- [x] Guarantee one prior track cannot be claimed by multiple new objects in the same primary assignment pass
- [x] Guarantee event lists cannot contain duplicate IDs or self-merges
- [x] Add regression tests for known duplicate/self-merge failures
- [x] Add crowded-scene synthetic tests with ambiguous neighbors
- [x] Run targeted unit tests
- [x] Run smoke tests
- [x] Run live replay checks on a known merge/split regression case and a dense convective scene
- [x] Inspect output for assignment stability, merge/split sanity, and active-track plausibility
- [x] Commit when green
- [x] Mark this task done in this file

---

## Task 6: Replace Raw Linear Regression Output with Confidence-Aware Motion

**Goal:** Stop speaking or exposing absurd motion values when identity is weak or history is noisy.

**Files:**
- Create: `src/tracking/motion.py`
- Create: `tests/unit/test_tracking_motion.py`
- Modify: `src/motion.py` or replace it with compatibility imports

- [x] Compute track-level motion from history with confidence scoring
- [x] Add sanity gating for unrealistic displacement, inconsistent headings, and low-quality histories
- [x] Return explicit states such as `stationary`, `nearly stationary`, and `uncertain`
- [x] Ensure dense-scene identity churn cannot produce 400+ mph spoken outputs
- [x] Add unit tests for stable motion, noisy motion, identity-swap-like jumps, and uncertain cases
- [x] Run targeted unit tests
- [x] Run smoke tests
- [x] Run live replay checks on a known merge/split regression case and a dense convective scene
- [x] Confirm that implausible speeds are suppressed or downgraded appropriately
- [x] Commit when green
- [x] Mark this task done in this file

---

## Task 7: Wire the New Tracker Pipeline into the Server

**Goal:** Put the refactored tracker behind the existing API.

**Files:**
- Create: `src/tracking/tracker.py`
- Modify: `src/server.py`
- Modify: `src/models.py`
- Modify: `src/summary.py`
- Create/Modify tests in `tests/smoke/` and `tests/e2e/`

- [x] Replace direct use of the legacy tracker logic with the new pipeline
- [x] Preserve `/tracks` and `/motion` response shapes where possible
- [x] Surface motion confidence in internal models; decide whether to expose it publicly now or keep it internal
- [x] Make summary generation confidence-aware so uncertain motion is not spoken as fact
- [x] Add smoke tests for `/summary`, `/tracks`, and `/motion`
- [x] Add e2e tests covering dense scenes, merge/split reporting, and uncertain-motion summaries
- [x] Run targeted tests
- [x] Run smoke tests
- [x] Run live API fetches for a known merge/split regression case, a dense convective scene, and at least one lower-complexity site
- [x] Inspect rendered output, not just status codes
- [x] Commit when green
- [x] Mark this task done in this file

---

## Task 8: Add Live Regression Harness

**Goal:** Make live validation a repeatable part of development instead of an ad hoc manual exercise.

**Files:**
- Create: `scripts/` helper or `tests/live/` helper if appropriate
- Create: `docs/test_reports/2026-04-10-*.md` updates as needed
- Create: `tests/unit/test_live_replay_contracts.py` if a pure unit wrapper is helpful

- [x] Add a reusable command/script for replaying multiple scans from a site and printing summary/track diagnostics
- [x] Support at least one known merge/split regression case and one dense convective scene
- [x] Ensure output includes object counts, strongest object summary, merge/split events, and motion sanity flags
- [x] Document how and when to run it
- [x] Run the live regression harness for a known merge/split regression case
- [x] Run the live regression harness for a dense convective scene
- [x] Run smoke tests if script/server changes affected runtime behavior (not required for this script-only task)
- [x] Commit when green
- [x] Mark this task done in this file

---

## Task 9: Final Validation and Documentation Update

**Goal:** Close the refactor cleanly and leave accurate operational notes.

**Files:**
- Modify: `PROGRESS.md`
- Modify or add: `docs/test_reports/*.md`
- Modify: this plan file

- [ ] Run the full test suite
- [ ] Run smoke tests explicitly
- [ ] Run live validation on a known merge/split regression case, a dense convective scene, and one lower-complexity site
- [ ] Review current summaries for motion plausibility and event sanity
- [ ] Update `PROGRESS.md` with completed work, current state, next work, and any remaining blockers
- [ ] Update test-report docs with live validation outcomes and dates
- [ ] Commit when green
- [ ] Mark this task done in this file

---

## Notes for Implementation

- Prefer architecture that survives later improvement over minimal patching
- Do not bake pySTEPS-specific assumptions into public interfaces yet; keep the motion-field module swappable
- When motion is uncertain, suppress it rather than speak nonsense
- If live data contradicts mock-based expectations, trust the live data and update the tests/specs accordingly
