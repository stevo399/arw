# Motion Agreement Follow-Up

Date: 2026-04-12

Purpose: verify that publishable motion is now suppressed when the selected motion source disagrees sharply with the focus track's recent short-horizon trajectory, without regressing simpler low-motion scenes.

## Validation Commands

```powershell
uv run pytest tests/unit/test_motion.py tests/unit/test_tracking_focus.py tests/unit/test_summary.py tests/unit/test_live_replay_contracts.py -q
uv run python scripts/live_replay.py KTLX --date 2026-04-10 --scans 8 --local-only
uv run python scripts/live_replay.py KSOX --date 2026-04-10 --scans 6 --local-only
```

## Dense Extended Window

Raw replay lines of interest:

- `2026-04-10T23:46:05Z objects=48 active=56 uncertain_tracks=13 max_speed_mph=20 focus_track=1 focus_identity=high:0.79 focus_continuity=medium:0.7 ... merges=5 splits=5`
- `2026-04-10T23:51:18Z objects=50 active=56 uncertain_tracks=9 max_speed_mph=20 focus_track=1 focus_identity=high:0.81 focus_continuity=medium:0.7 ... merges=8 splits=3`
- `2026-04-10T23:56:21Z objects=50 active=61 uncertain_tracks=10 max_speed_mph=21 focus_track=1 focus_identity=high:0.81 focus_continuity=medium:0.7 ... merges=6 splits=6`

Clean summary lines:

- `Oklahoma City: 48 rain objects detected. Strongest: severe core, 26 miles E of the radar, moving WNW at 20 mph. Note: 5 storms merged in the last scan. Note: 5 storms split in the last scan. Covering approximately 3634 square miles.`
- `Oklahoma City: 50 rain objects detected. Strongest: severe core, 25 miles E of the radar, tracking uncertain. Note: 8 storms merged in the last scan. Note: 3 storms split in the last scan. Covering approximately 3714 square miles.`
- `Oklahoma City: 50 rain objects detected. Strongest: severe core, 25 miles ESE of the radar, tracking uncertain. Note: 6 storms merged in the last scan. Note: 6 storms split in the last scan. Covering approximately 3815 square miles.`

Observed behavior:

- the late dense-scene `NE -> SE` reversal pair is no longer spoken as clean publishable motion
- the suppression is generic: it is driven by disagreement between the chosen motion source and the recent short-horizon track trajectory
- the earlier `WNW` scan still publishes, which means this change narrowed the failure mode instead of blanket-suppressing the whole dense window

## Simpler Extended Window

Representative summary lines:

- `Santa Ana Mountains: 4 rain objects detected. Strongest: intense rain, 35 miles NE of the radar, stationary. Covering approximately 12 square miles.`
- `Santa Ana Mountains: 3 rain objects detected. Strongest: intense rain, 34 miles NNE of the radar, nearly stationary. Note: 1 storm merged in the last scan. Covering approximately 10 square miles.`
- `Santa Ana Mountains: 7 rain objects detected. Strongest: heavy rain, 35 miles NE of the radar, nearly stationary. Note: 1 storm merged in the last scan. Note: 1 storm split in the last scan. Covering approximately 30 square miles.`

Observed behavior:

- no new uncertain-motion output was introduced in the simpler low-motion scene
- stationary and nearly-stationary summaries remained intact across the window

## Conclusion

- the remaining dense-scene problem is reduced from a three-scan reversal-heavy spoken sequence to a single earlier directional outlier followed by two suppressed scans
- this is a generic motion-publication safeguard, not a radar-specific rule or a per-site exception
