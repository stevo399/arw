# Earlier-Window Selector Validation

Date: 2026-04-14

Purpose: validate that replay and benchmark manifests can target earlier same-day cached windows generically, without relying on tail-of-day selection or any site-specific exception logic.

## What changed

- `scripts/live_replay.py` now supports an optional `--end-filename` selector
- replay window selection can now end on a specific scan filename and take the preceding `N` scans
- local-only replay fallback now honors that same target when reconstructing windows from cache
- `scripts/evaluate_tracking.py` now accepts manifest entries that include `end_filename`

## Verification

Unit tests:

- `uv run pytest tests/unit/test_live_replay_contracts.py -q`
- result: `7 passed in 10.41s`
- `uv run pytest tests/unit/test_tracking_evaluation.py -q`
- result: `1 passed in 2.57s`

Smoke tests:

- `uv run pytest tests/smoke/test_server_smoke.py -q`
- result: `9 passed in 9.67s`

## Targeted Replay Checks

Command:

```powershell
uv run python scripts/live_replay.py KTLX --date 2026-04-10 --scans 2 --local-only --end-filename KTLX20260410_214445_V06
```

Observed replay output:

```text
SITE KTLX (Oklahoma City)
REPLAY_SCANS 2
MODE local-only
2026-04-10T21:40:23Z objects=68 active=68 uncertain_tracks=0 max_speed_mph=0 focus_track=11 focus_identity=medium:0.6 focus_continuity=high:0.9 focus_selection_margin=0.07 focus_runner_up=1 scan_quality=0.44 quality_flags=high_missing_fraction,speckle_filtered merges=0 splits=0
  Oklahoma City: 68 rain objects detected. Strongest: intense rain, 90 miles NE of the radar, stationary. Covering approximately 7519 square miles.
2026-04-10T21:44:45Z objects=77 active=83 uncertain_tracks=8 max_speed_mph=26 focus_track=11 focus_identity=high:0.89 focus_continuity=medium:0.7 focus_selection_margin=4.64 focus_runner_up=3 scan_quality=0.44 quality_flags=high_missing_fraction,speckle_filtered merges=12 splits=7
  Oklahoma City: 77 rain objects detected. Strongest: severe core, 95 miles ENE of the radar, moving NE at 9 mph. Note: 12 storms merged in the last scan. Note: 7 storms split in the last scan. Covering approximately 7479 square miles.
```

Command:

```powershell
uv run python scripts/live_replay.py KEYX --date 2026-04-10 --scans 2 --local-only --end-filename KEYX20260410_212646_V06
```

Observed replay output:

```text
SITE KEYX (Edwards AFB)
REPLAY_SCANS 2
MODE local-only
2026-04-10T21:19:45Z objects=6 active=6 uncertain_tracks=0 max_speed_mph=0 focus_track=1 focus_identity=medium:0.58 focus_continuity=high:0.9 focus_selection_margin=0.32 focus_runner_up=2 scan_quality=0.38 quality_flags=high_missing_fraction,speckle_filtered merges=0 splits=0
  Edwards AFB: 6 rain objects detected. Strongest: intense rain, 39 miles WNW of the radar, stationary. Covering approximately 26 square miles.
2026-04-10T21:26:46Z objects=6 active=8 uncertain_tracks=1 max_speed_mph=5 focus_track=3 focus_identity=high:0.8 focus_continuity=high:1.0 focus_selection_margin=1.29 focus_runner_up=4 scan_quality=0.38 quality_flags=high_missing_fraction,speckle_filtered merges=1 splits=1
  Edwards AFB: 6 rain objects detected. Strongest: intense rain, 40 miles WNW of the radar, moving NW at 5 mph. Note: 1 storm merged in the last scan. Note: 1 storm split in the last scan. Covering approximately 31 square miles.
```

## Conclusions

- earlier cached windows are now selectable directly rather than only by taking the last scans of a day
- the same selector worked in both a dense window and a lower-object merge/split-sensitive window
- this is a generic replay-harness capability, not a site-specific rule
- full broader-manifest evaluation remains computationally expensive on the current replay path, so this report serves as the clean operational proof for the selector addition while broader quantitative runs remain a runtime-management problem rather than a correctness problem
