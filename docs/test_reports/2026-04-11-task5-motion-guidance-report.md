# Task 5 Motion Guidance Report

Date: 2026-04-11

Purpose: document the Task 5 upgrade from one global motion prior to a conservative local-plus-global motion-guidance model, along with the clean validation data for later comparison.

## What Changed

- kept the existing global full-scan phase-correlation estimate
- added a local ROI phase-correlation estimate around each tracked object's prior bbox
- blended local and global geographic motion by quality
- rejected local guidance when it was too weak or too inconsistent with the broader scene motion
- fed per-track blended motion guidance into:
  - association prediction
  - track-level reported-motion fallback logic

This keeps the architecture generic and moves the motion stage beyond one bulk scene estimate without introducing radar-specific rules.

## Verification

Command:

```powershell
uv run pytest tests/unit/test_tracking_motion_field.py tests/unit/test_motion.py tests/unit/test_tracking_association.py tests/unit/test_tracker.py tests/unit/test_summary.py tests/smoke/test_server_smoke.py -q
```

Result:

- `64 passed in 3.14s`

## Live Replay Validation

### Dense cached replay

Command:

```powershell
uv run python scripts/live_replay.py KTLX --date 2026-04-10 --quick --local-only
```

Observed behavior:

- dense replay stayed stable after the motion-guidance upgrade
- focal summary remained anchored on the same northern storm
- spoken motion stayed plausible at `0`, `7`, and `20` mph
- no uncertain-motion regression remained after the conservative local/global consistency gate was added

Clean output:

- `2026-04-10T23:46:05Z objects=48 active=48 uncertain_tracks=0 max_speed_mph=0 scan_quality=0.43 quality_flags=high_missing_fraction,speckle_filtered merges=0 splits=0`
- `2026-04-10T23:51:18Z objects=50 active=56 uncertain_tracks=0 max_speed_mph=7 scan_quality=0.43 quality_flags=high_missing_fraction,speckle_filtered merges=8 splits=3`
- `2026-04-10T23:56:21Z objects=50 active=61 uncertain_tracks=0 max_speed_mph=20 scan_quality=0.43 quality_flags=high_missing_fraction,speckle_filtered merges=6 splits=6`

Clean summaries:

- `Oklahoma City: 48 rain objects detected. Strongest: intense rain, 98 miles NNE of the radar, stationary. Covering approximately 3634 square miles.`
- `Oklahoma City: 50 rain objects detected. Strongest: intense rain, 97 miles NNE of the radar, moving NE at 7 mph. Note: 8 storms merged in the last scan. Note: 3 storms split in the last scan. Covering approximately 3714 square miles.`
- `Oklahoma City: 50 rain objects detected. Strongest: intense rain, 98 miles NNE of the radar, moving SE at 20 mph. Note: 6 storms merged in the last scan. Note: 6 storms split in the last scan. Covering approximately 3815 square miles.`

### Lower-complexity replay

Command:

```powershell
uv run python scripts/live_replay.py KSOX --date 2026-04-10 --quick
```

Observed behavior:

- simple-scene replay stayed stable after the initial local-motion regression was corrected
- the earlier false fast-motion fire was removed by the conservative blend gate
- spoken motion returned to `stationary` or `nearly stationary`

Clean output:

- `2026-04-10T23:35:42Z objects=3 active=3 uncertain_tracks=0 max_speed_mph=0 scan_quality=0.37 quality_flags=high_missing_fraction,speckle_filtered merges=0 splits=0`
- `2026-04-10T23:44:23Z objects=2 active=2 uncertain_tracks=0 max_speed_mph=0 scan_quality=0.37 quality_flags=high_missing_fraction,speckle_filtered merges=1 splits=0`
- `2026-04-10T23:53:02Z objects=7 active=7 uncertain_tracks=0 max_speed_mph=0 scan_quality=0.37 quality_flags=high_missing_fraction,speckle_filtered merges=1 splits=1`

Clean summaries:

- `Santa Ana Mountains: 3 rain objects detected. Strongest: intense rain, 34 miles NNE of the radar, stationary. Covering approximately 10 square miles.`
- `Santa Ana Mountains: 2 rain objects detected. Strongest: intense rain, 35 miles NE of the radar, nearly stationary. Note: 1 storm merged in the last scan. Covering approximately 9 square miles.`
- `Santa Ana Mountains: 7 rain objects detected. Strongest: heavy rain, 35 miles NE of the radar, nearly stationary. Note: 1 storm merged in the last scan. Note: 1 storm split in the last scan. Covering approximately 30 square miles.`

## Development Note

The first Task 5 pass allowed local guidance to influence reported motion too aggressively in a simpler scene, which surfaced a false fast-motion report. That regression was fixed by making the local/global blend conservative:

- local guidance must meet a minimum quality threshold
- local guidance must remain reasonably consistent with the global field
- local pixel guidance is only used when the blended geographic field actually accepts local input

This is the kind of clean data trail we want to keep per task so later behavior changes can be compared directly against a known-good baseline.
