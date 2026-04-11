# Dense Confidence Follow-Up

Date: 2026-04-11

Purpose: extend dense-scene live validation beyond the 3-scan benchmark and capture how the new confidence diagnostics behave over a broader cached replay window.

## Command

```powershell
uv run python scripts/live_replay.py KTLX --date 2026-04-10 --scans 5 --local-only
```

## Runtime Notes

- a 12-scan cached replay did not finish within a 10-minute timeout in this environment
- an 8-scan cached replay also exceeded the available timeout budget
- the 5-scan cached replay completed successfully and is the current practical broader-window follow-up artifact

## Raw Replay Lines

- `2026-04-10T23:35:25Z objects=39 active=39 uncertain_tracks=0 max_speed_mph=0 scan_quality=0.43 quality_flags=high_missing_fraction,speckle_filtered merges=0 splits=0`
- `2026-04-10T23:40:53Z objects=48 active=54 uncertain_tracks=0 max_speed_mph=30 scan_quality=0.43 quality_flags=high_missing_fraction,speckle_filtered merges=7 splits=2`
- `2026-04-10T23:46:05Z objects=48 active=56 uncertain_tracks=0 max_speed_mph=34 scan_quality=0.43 quality_flags=high_missing_fraction,speckle_filtered merges=5 splits=5`
- `2026-04-10T23:51:18Z objects=50 active=56 uncertain_tracks=0 max_speed_mph=20 scan_quality=0.43 quality_flags=high_missing_fraction,speckle_filtered merges=8 splits=3`
- `2026-04-10T23:56:21Z objects=50 active=61 uncertain_tracks=0 max_speed_mph=31 scan_quality=0.43 quality_flags=high_missing_fraction,speckle_filtered merges=6 splits=6`

## Clean Summary Lines

- `Oklahoma City: 39 rain objects detected. Strongest: severe core, 93 miles NNE of the radar, stationary. Covering approximately 3571 square miles.`
- `Oklahoma City: 48 rain objects detected. Strongest: heavy rain, 90 miles NE of the radar, moving SE at 18 mph. Note: 7 storms merged in the last scan. Note: 2 storms split in the last scan. Covering approximately 3620 square miles.`
- `Oklahoma City: 48 rain objects detected. Strongest: heavy rain, 90 miles NE of the radar, moving WNW at 20 mph. Note: 5 storms merged in the last scan. Note: 5 storms split in the last scan. Covering approximately 3634 square miles.`
- `Oklahoma City: 50 rain objects detected. Strongest: intense rain, 97 miles NNE of the radar, moving NE at 7 mph. Note: 8 storms merged in the last scan. Note: 3 storms split in the last scan. Covering approximately 3714 square miles.`
- `Oklahoma City: 50 rain objects detected. Strongest: intense rain, 98 miles NNE of the radar, moving SE at 20 mph. Note: 6 storms merged in the last scan. Note: 6 storms split in the last scan. Covering approximately 3815 square miles.`

## Observations

- the broader dense replay still kept `uncertain_tracks=0` across the window, so the current confidence calibration is not yet suppressing this evolving focus-motion sequence
- the focus storm motion swung across adjacent scans as `stationary -> SE 18 mph -> WNW 20 mph -> NE 7 mph -> SE 20 mph`
- the strongest-object description also changed from `severe core` to `heavy rain` to `intense rain`, which suggests the focus object identity is evolving at the same time motion is turning sharply
- merge and split pressure remained elevated throughout the window, especially from `23:40:53Z` onward
- this is a useful validation target for the next iteration because the short benchmark catches one heading flip, but the 5-scan replay shows repeated directional reversals rather than a single isolated event

## Conclusion

- Task 7 improved calibration for simpler and merge/split-sensitive scenes, but this denser 5-scan follow-up shows that abrupt focus-motion reversals can still pass through as publishable motion in crowded evolving scenes
- the next work item should target dense-scene focus-motion stability over multi-scan windows, using this report and `docs/test_reports/2026-04-11-task7-confidence-evaluation.md` together
