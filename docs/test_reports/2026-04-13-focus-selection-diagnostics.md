# Focus Selection Diagnostics

Date: 2026-04-13

Purpose: verify whether the remaining dense-window directional outlier is caused by choosing the wrong focus track or by motion behavior on the correct focus track.

## Validation Commands

```powershell
uv run pytest tests/unit/test_live_replay_contracts.py tests/unit/test_tracking_focus.py tests/unit/test_models.py tests/smoke/test_server_smoke.py -q
uv run python scripts/live_replay.py KTLX --date 2026-04-10 --scans 8 --local-only
```

## Dense Live Output

Representative diagnostics:

- `2026-04-10T23:24:50Z ... focus_track=1 focus_continuity=medium:0.7 focus_selection_margin=1.87 focus_runner_up=5 ...`
- `2026-04-10T23:30:07Z ... focus_track=1 focus_continuity=medium:0.7 focus_selection_margin=1.23 focus_runner_up=5 ...`
- `2026-04-10T23:35:25Z ... focus_track=1 focus_continuity=medium:0.7 focus_selection_margin=1.25 focus_runner_up=5 ...`
- `2026-04-10T23:40:53Z ... focus_track=1 focus_continuity=medium:0.7 focus_selection_margin=3.45 focus_runner_up=5 ...`
- `2026-04-10T23:46:05Z ... focus_track=1 focus_continuity=medium:0.7 focus_selection_margin=2.88 focus_runner_up=4 ...`
- `2026-04-10T23:51:18Z ... focus_track=1 focus_continuity=low:0.3 focus_selection_margin=2.49 focus_runner_up=119 ...`
- `2026-04-10T23:56:21Z ... focus_track=1 focus_continuity=low:0.3 focus_selection_margin=1.11 focus_runner_up=119 ...`

## Findings

- the dense-window outlier is not explained by an obviously wrong focus handoff
- the same focus track remains selected across the whole checked dense window
- challenger pressure is real and now visible, but it is diagnostic context rather than sufficient evidence that the focus should have switched
- the late-window uncertainty case is stronger than the earlier `WNW` outlier:
  - at `23:46:05Z`, the focus still wins by `2.88`
  - at `23:51:18Z`, the margin narrows to `2.49`
  - at `23:56:21Z`, the margin narrows further to `1.11`

## Conclusion

- focus selection is not the primary remaining defect in the dense live window
- the new `focus_selection_margin` and `focus_runner_up` diagnostics make that conclusion reviewable in normal live replay output without another ad hoc script
- if the earlier `WNW` outlier is still worth addressing, the next change should target motion-source behavior or publication policy, not a generic focus-handoff rule
