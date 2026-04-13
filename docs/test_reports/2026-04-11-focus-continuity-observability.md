# Focus Continuity Observability

Date: 2026-04-11

Purpose: verify that live replay output now exposes focus continuity directly and that the printed diagnostics align with spoken publishability behavior in both dense and simpler scenes.

## Validation Commands

```powershell
uv run pytest tests/unit/test_live_replay_contracts.py tests/smoke/test_server_smoke.py tests/unit/test_models.py tests/unit/test_tracking_evaluation.py -q
uv run python scripts/live_replay.py KTLX --date 2026-04-10 --scans 5 --local-only
uv run python scripts/live_replay.py KSOX --date 2026-04-10 --quick
```

## Observed Dense Window

- `2026-04-10T23:35:25Z ... focus_identity=medium:0.6 focus_continuity=high:0.9 ...`
- `2026-04-10T23:40:53Z ... focus_identity=medium:0.71 focus_continuity=medium:0.6 ...`
- `2026-04-10T23:46:05Z ... focus_identity=medium:0.51 focus_continuity=medium:0.6 ...`
- `2026-04-10T23:51:18Z ... focus_identity=low:0.4 focus_continuity=medium:0.45 ...`
- `2026-04-10T23:56:21Z ... focus_identity=high:0.8 focus_continuity=medium:0.7 ...`

Dense summary behavior:

- the three scans with degraded focus continuity at `0.6`, `0.6`, and `0.45` all produced `tracking uncertain`
- the final scan at `0.7` published `moving SE at 20 mph`

## Observed Simpler Window

- `2026-04-10T23:35:42Z ... focus_identity=medium:0.58 focus_continuity=high:0.9 ...`
- `2026-04-10T23:44:23Z ... focus_identity=high:0.83 focus_continuity=high:1.0 ...`
- `2026-04-10T23:53:02Z ... focus_identity=high:0.77 focus_continuity=high:1.0 ...`

Simpler summary behavior:

- focus continuity stayed high throughout the window
- no uncertain-motion output was introduced
- stationary and nearly-stationary summaries remained plausible

## Conclusion

- focus continuity is now visible in the same operational output used for live review
- the new field is informative rather than decorative: it tracks the same dense-scene window that currently triggers spoken-motion suppression
- the simpler window stayed clean, which suggests the new diagnostic is distinguishing unstable focus continuity from ordinary low-motion scenes
