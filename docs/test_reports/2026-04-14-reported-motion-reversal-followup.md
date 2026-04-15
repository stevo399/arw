# Reported Motion Reversal Follow-Up

Date: 2026-04-14

Purpose: verify that focus continuity now penalizes abrupt reversals in reported focus motion under structural pressure, even when raw centroid-step diagnostics are too noisy to expose the problem directly.

## Validation Commands

```powershell
uv run pytest tests/unit/test_tracking_focus.py tests/unit/test_models.py tests/unit/test_live_replay_contracts.py tests/unit/test_summary.py tests/smoke/test_server_smoke.py -q
uv run python scripts/evaluate_tracking.py --manifest docs/benchmarks/tracking_benchmark_manifest_broader_validation.json --output-json docs/test_reports/2026-04-14-motion-reversal-continuity-evaluation.json --output-md docs/test_reports/2026-04-14-motion-reversal-continuity-evaluation.md
uv run python scripts/live_replay.py KTLX --date 2026-04-10 --scans 8 --local-only
uv run python scripts/live_replay.py KSOX --date 2026-04-10 --scans 6 --local-only
```

## Key Result

The remaining dense-window `WNW` outlier at `2026-04-10T23:46:05Z` is no longer published as clean motion.

Before:

- `Oklahoma City: 48 rain objects detected. Strongest: severe core, 26 miles E of the radar, moving WNW at 20 mph. ...`

After:

- `Oklahoma City: 48 rain objects detected. Strongest: severe core, 26 miles E of the radar, tracking uncertain. ...`

The relevant diagnostics at that scan are now:

- `focus_continuity=low:0.35`
- `focus_selection_margin=2.88`
- `focus_runner_up=4`

This confirms the downgrade is not coming from a forced focus handoff. The same focus track still wins, but its reported motion sequence is no longer treated as stable enough to publish under heavy structural pressure.

## Benchmark Changes

- dense extended window:
  - focus low-continuity scans moved from `2` to `3`
  - summary tracking-uncertain count moved from `2` to `3`
  - mean focus continuity moved from `0.62` to `0.51`
- lower-complexity extended window:
  - focus low-continuity scans remained `0`
  - summary tracking-uncertain count remained `0`
- merge/split-heavy extended window:
  - focus low-continuity scans remained `1`
  - summary tracking-uncertain count remained `1`

## Conclusion

- the earlier dense-window `WNW` outlier was a reported-motion stability problem, not primarily a focus-selection problem
- continuity now captures that case using reported-motion reversal history under structural pressure
- the simpler and merge/split-heavy benchmark windows remained acceptably stable
