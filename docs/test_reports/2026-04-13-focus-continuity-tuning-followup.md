# Focus Continuity Tuning Follow-Up

Date: 2026-04-13

Purpose: verify that focus continuity scoring now penalizes low-confidence motion under structural pressure while avoiding false heading-instability penalties in stationary or nearly-stationary scenes.

## Validation Commands

```powershell
uv run pytest tests/unit/test_tracking_focus.py tests/unit/test_motion.py tests/unit/test_summary.py tests/unit/test_live_replay_contracts.py -q
uv run python scripts/evaluate_tracking.py --manifest docs/benchmarks/tracking_benchmark_manifest_broader_validation.json --output-json docs/test_reports/2026-04-13-continuity-tuning-evaluation.json --output-md docs/test_reports/2026-04-13-continuity-tuning-evaluation.md
uv run python scripts/live_replay.py KTLX --date 2026-04-10 --scans 8 --local-only
uv run python scripts/live_replay.py KSOX --date 2026-04-10 --scans 6 --local-only
uv run python scripts/live_replay.py KEYX --date 2026-04-10 --scans 7 --local-only
```

## Key Metric Changes

- dense extended window:
  - focus low-continuity scans moved from `0` to `2`
  - mean focus continuity moved from `0.72` to `0.62`
  - the two late `tracking uncertain` dense scans now also carry low focus continuity (`0.3`)
- lower-complexity extended window:
  - focus low-continuity scans moved from `3` to `0`
  - mean focus continuity moved from `0.72` to `0.89`
  - stationary and nearly-stationary summaries remained unchanged
- merge/split-heavy extended window:
  - focus low-continuity scans moved from `3` to `1`
  - mean focus continuity moved from `0.66` to `0.76`
  - the remaining low-continuity case is the late-window focus handoff that already renders `tracking uncertain`

## Representative Live Output

Dense late-window scans:

- `2026-04-10T23:51:18Z ... focus_identity=high:0.81 focus_continuity=low:0.3 ...`
- `2026-04-10T23:56:21Z ... focus_identity=high:0.81 focus_continuity=low:0.3 ...`

Clean summary lines:

- `Oklahoma City: 50 rain objects detected. Strongest: severe core, 25 miles E of the radar, tracking uncertain. Note: 8 storms merged in the last scan. Note: 3 storms split in the last scan. Covering approximately 3714 square miles.`
- `Oklahoma City: 50 rain objects detected. Strongest: severe core, 25 miles ESE of the radar, tracking uncertain. Note: 6 storms merged in the last scan. Note: 6 storms split in the last scan. Covering approximately 3815 square miles.`

Lower-complexity scan samples:

- `2026-04-10T23:35:42Z ... focus_identity=medium:0.73 focus_continuity=high:0.9 ...`
- `2026-04-10T23:44:23Z ... focus_identity=medium:0.74 focus_continuity=high:0.9 ...`
- `2026-04-10T23:53:02Z ... focus_identity=medium:0.71 focus_continuity=high:0.9 ...`

Merge/split-heavy late-window scan:

- `2026-04-10T23:56:44Z ... focus_track=25 focus_identity=high:0.83 focus_continuity=low:0.3 ...`

## Conclusion

- focus continuity is now better aligned with the windows that are genuinely risky for published motion
- stationary and nearly-stationary scenes are no longer being downgraded just because short-step headings were noisy under low-motion conditions
- the remaining low-continuity merge/split-heavy case corresponds to a real late-window focus handoff rather than a broad false-positive pattern
