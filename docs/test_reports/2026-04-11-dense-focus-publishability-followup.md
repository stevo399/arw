# Dense Focus Publishability Follow-Up

Date: 2026-04-11

Purpose: verify that the summary layer now downgrades spoken motion during unstable dense-scene focus continuity instead of publishing rapid direction reversals as clean motion.

## Command

```powershell
uv run python scripts/live_replay.py KTLX --date 2026-04-10 --scans 5 --local-only
```

## Raw Replay Lines

- `2026-04-10T23:35:25Z objects=39 active=39 uncertain_tracks=0 max_speed_mph=0 scan_quality=0.43 quality_flags=high_missing_fraction,speckle_filtered merges=0 splits=0`
- `2026-04-10T23:40:53Z objects=48 active=54 uncertain_tracks=0 max_speed_mph=30 scan_quality=0.43 quality_flags=high_missing_fraction,speckle_filtered merges=7 splits=2`
- `2026-04-10T23:46:05Z objects=48 active=56 uncertain_tracks=0 max_speed_mph=34 scan_quality=0.43 quality_flags=high_missing_fraction,speckle_filtered merges=5 splits=5`
- `2026-04-10T23:51:18Z objects=50 active=56 uncertain_tracks=0 max_speed_mph=20 scan_quality=0.43 quality_flags=high_missing_fraction,speckle_filtered merges=8 splits=3`
- `2026-04-10T23:56:21Z objects=50 active=61 uncertain_tracks=0 max_speed_mph=31 scan_quality=0.43 quality_flags=high_missing_fraction,speckle_filtered merges=6 splits=6`

## Clean Summary Lines

- `Oklahoma City: 39 rain objects detected. Strongest: severe core, 93 miles NNE of the radar, stationary. Covering approximately 3571 square miles.`
- `Oklahoma City: 48 rain objects detected. Strongest: heavy rain, 90 miles NE of the radar, tracking uncertain. Note: 7 storms merged in the last scan. Note: 2 storms split in the last scan. Covering approximately 3620 square miles.`
- `Oklahoma City: 48 rain objects detected. Strongest: heavy rain, 90 miles NE of the radar, tracking uncertain. Note: 5 storms merged in the last scan. Note: 5 storms split in the last scan. Covering approximately 3634 square miles.`
- `Oklahoma City: 50 rain objects detected. Strongest: intense rain, 97 miles NNE of the radar, tracking uncertain. Note: 8 storms merged in the last scan. Note: 3 storms split in the last scan. Covering approximately 3714 square miles.`
- `Oklahoma City: 50 rain objects detected. Strongest: intense rain, 98 miles NNE of the radar, moving SE at 20 mph. Note: 6 storms merged in the last scan. Note: 6 storms split in the last scan. Covering approximately 3815 square miles.`

## Observations

- the dense 5-scan follow-up no longer speaks the mid-window `SE -> WNW -> NE` directional swings as if they were equally trustworthy
- the summary now suppresses motion during the three unstable middle scans while merge/split pressure is elevated and focus continuity is weaker
- the final scan still speaks `SE at 20 mph`, which is acceptable under the current generic gate because the focus track is back to high identity confidence
- this change is product-layer gating only; it does not hardcode behavior for any individual radar and does not alter the underlying tracker metrics

## Conclusion

- the remaining dense-scene issue is narrowed from "spoken motion is obviously unstable" to "determine whether the final regained publishable motion is strict enough"
- this is a materially better live behavior for users because unstable focus-motion sequences are no longer spoken as clean directional facts
