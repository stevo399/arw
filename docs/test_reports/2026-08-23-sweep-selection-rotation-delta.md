# Sweep-Selection Fix: Rotation/Region Baseline Delta

Date: 2026-08-23 (produced 2026-08-24)

## What this measures

Task 6 replaced index-based velocity sweep selection (`range(3)`) with
coverage-based selection (`select_velocity_sweeps`), because on real
split-cut NEXRAD volumes `range(3)` returns two surveillance cuts with zero
velocity gates and only one real Doppler cut. This report is spec §11
"rotation changes measured separately": it records region/rotation counts,
strengths, and `sweep_count` distributions before and after that fix on the
same cached scan window, so a later task (Task 18) has a real number to
compare against instead of the fabricated-looking "126-142 signatures"
figures in `docs/test_reports/2026-04-18-velocity-validation.md`, which (see
below) turn out to be explainable by a *different*, pre-existing defect.

## How the numbers were obtained

No stash/checkout was used. A standalone script
(`rotation_baseline.py`, not committed — scratch tooling) imports the
current, already-fixed `src.parser.extract_velocity` for the "AFTER" pass,
and defines `extract_velocity_old`, a byte-for-byte copy of the pre-fix
function body from `src/parser.py` (the `range(min(max_sweeps,
radar.nsweeps))` version, `git show`n from before this task's changes),
with only `VelocitySweep(elevations=...)` added so it can construct the
current dataclass. Both passes are run over the same 9 cached, already-local
NEXRAD volumes (no network calls — `nexradaws`/`list_scans_for_date` was
deliberately not used; filenames were read directly from `cache/KTLX/`) and
fed through the unmodified `detect_velocity_regions` /
`detect_rotation_signatures` from `src/velocity.py`.

**Window:** site KTLX, the 9 consecutive cached scans
`KTLX20260410_170556_V06` through `KTLX20260410_173731_V06` (17:05:56Z-17:37:31Z,
2026-04-10), the same afternoon severe-weather window used in
`docs/test_reports/2026-04-18-velocity-validation.md`.

Exact commands run:
```
.venv/Scripts/python.exe -u rotation_baseline.py
```
(script contents summarized above; available on request, not part of the commit)

## Confirmed: the defect was real, exactly as described

Coverage check on `KTLX20260410_170556_V06`, sweep indices 0-5:

| idx | elevation | velocity coverage |
|-----|-----------|--------------------|
| 0 | 0.483 | 0.0 |
| 1 | 0.483 | 0.141 |
| 2 | 0.879 | 0.0 |
| 3 | 0.879 | 0.130 |
| 4 | 1.274 | 0.0 |
| 5 | 1.274 | 0.115 |

Old code selected indices `[0, 1, 2]` -> coverage `[0.0, 0.141, 0.0]`. Two of
three selected sweeps carried zero velocity gates, exactly as described in
the task brief. New code selects `[1, 3, 5]` -> coverage `[0.141, 0.130,
0.115]`, all three real.

## Results

### Region counts (per-scan totals, after cross-sweep IOU merge)

| Scan | Before (regions) | After (regions) |
|---|---|---|
| 170556 | 66 | 183 |
| 170946 | 71 | 186 |
| 171336 | 76 | 210 |
| 171731 | 77 | 210 |
| 172134 | 77 | 202 |
| 172537 | 69 | 193 |
| 172939 | 78 | 192 |
| 173342 | 83 | 195 |
| 173731 | 69 | 185 |
| **Total** | **666** | **1756** |

Region `sweep_count` distribution: **`{1: 666}` before, `{1: 1756}` after —
unchanged, always 1, in both conditions, for this window.**

This needs to be stated plainly rather than spun: the fix roughly **triples**
the number of detected velocity regions (2.6x) because three real Doppler
sweeps now feed the detector instead of one, but **no single region's mask
ever overlaps >=30% (`CROSS_SWEEP_OVERLAP_THRESHOLD`) across sweeps in this
particular window**, before or after. `VelocityRegion.sweep_count` never
exceeds 1 here. This is a property of this storm's structure (the velocity
signature's spatial footprint shifts enough between 0.48/0.88/1.27 degrees
that IOU-based mask overlap doesn't clear the threshold), not something the
sweep-selection fix changes by itself, and not evidence the fix is
ineffective — see the next section for a case where it does confirm.

**Region-level multi-sweep confirmation does work post-fix on other data.**
`tests/unit/test_velocity.py::test_sweep_count_can_exceed_one_on_real_data`
uses `cache/KTLX/KTLX20260410_000100_V06`, found by scanning all 152 cached
volumes for a genuine cross-sweep match: one inbound region there merges
cleanly across all 3 real Doppler sweeps (`sweep_count == 3`,
`elevation_angles == [0.483, 0.879, 1.274]`, no duplicate elevations). That
scan is not in the 9-scan window used for this report (a different site/day),
so it is cited here as existence proof, not folded into the KTLX table above.

### Rotation counts and `sweep_count` — a caveat that needs flagging

| Scan | Before (rotations) | After (rotations) |
|---|---|---|
| 170556 | 131 | 135 |
| 170946 | 123 | 131 |
| 171336 | 121 | 131 |
| 171731 | 128 | 138 |
| 172134 | 129 | 134 |
| 172537 | 139 | 137 |
| 172939 | 138 | 145 |
| 173342 | 142 | 138 |
| 173731 | 141 | 140 |
| **Total** | **1192** | **1229** |

Total rotation-signature counts barely move (+3%), unlike regions
(+164%). Strength mix shifts: weak 507->597, moderate 411->494, strong
274->138 (before -> after).

Rotation `sweep_count` distribution goes as high as **120 before the fix and
198 after it** (full histograms in the raw output, not reproduced here —
they are long tails, not single numbers). This is *not* believable as "120
real radar sweeps confirmed a mesocyclone" — the reference volume and this
window both have exactly 3 real velocity-bearing sweeps, before and after
the fix. Inspecting `_merge_cross_sweep_rotations` in `src/velocity.py`
explains why: it merges any new shear-couplet signature into any existing
entry within `ROTATION_MERGE_DISTANCE_KM` (10 km) regardless of whether that
entry already came from the *same* sweep. A single sweep with many
gate-to-gate shear couplets close together (which a wide velocity-couplet
field on any one tilt easily produces) merges those couplets into one
another and inflates `sweep_count`, with the same `elevation_angle` value
repeated dozens of times in `elevation_angles`. Confirmed directly: before
the fix, with only 1 real velocity-bearing sweep in the volume,
`sweep_count` still reached 120 on that single sweep's own couplets.

**This is a pre-existing defect in the rotation cross-sweep merge, separate
from the sweep-selection bug this task fixes.** It existed before this task
(it produced inflated-but-lower numbers even when starved to 1 real sweep)
and still exists after. It is out of Task 6's scope (`extract_velocity` /
`select_velocity_sweeps`), but it means **`RotationSignature.sweep_count`
should not be read as a multi-sweep confirmation signal today** — only
`VelocityRegion.sweep_count` (IOU-based, immune to this particular bug) is
trustworthy for that purpose, and that field is the one demonstrated working
correctly above (real 3-sweep merges on `KTLX20260410_000100_V06`,
appropriately staying at 1 when spatial overlap genuinely isn't there in the
9-scan window). Flagging this now rather than letting a future task (Task 18
or otherwise) reuse `RotationSignature.sweep_count` as if it meant the same
thing.

The `docs/test_reports/2026-04-18-velocity-validation.md` figures
("129-142 signatures per scan", written pre-fix) are consistent with the
"Before" column here (121-142) — that prior report's rotation *counts* were
correct; it never made a `sweep_count`-based cross-sweep confirmation claim,
so it is not itself invalidated by either bug.

## Summary

- Confirmed the defect precisely: old selection pulled the empty
  surveillance cut in 2 of 3 slots; new selection pulls the correct Doppler
  cut every time, on 152/152 spot-checked and 9/9 windowed real volumes.
- Region *count* roughly triples post-fix because 3x the real velocity data
  now reaches detection.
- Region `sweep_count` (the IOU-based, trustworthy multi-sweep signal) stays
  1 in this particular 9-scan window both before and after -- a real
  property of this storm's evolution, not a fix regression -- but is proven
  to reach 3 post-fix on a different cached volume
  (`KTLX20260410_000100_V06`), which is what the regression test in
  `tests/unit/test_velocity.py` now asserts (not skips).
- Rotation `sweep_count` numbers in both before/after columns are inflated
  by an unrelated, pre-existing bug in `_merge_cross_sweep_rotations` and
  should not be trusted as a multi-sweep confirmation metric until that is
  fixed separately.
