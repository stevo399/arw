# Reported Heading Observability Follow-Up

Date: 2026-04-15

Purpose: expose the recent reported heading sequence directly in replay, benchmark, and API diagnostics so dense-scene suppressions can be judged from the actual motion-history pattern rather than inferred from aggregate flip counters.

## What changed

- focus continuity now carries `recent_reported_heading_sequence`
- live replay output now prints `focus_reported_sequence=...`
- benchmark snapshots now include `focus_reported_sequence=...`
- `/tracks/{site_id}` and `/motion/{site_id}/{track_id}` now expose the same sequence in the `focus` payload

Format:

- directional samples with a heading: `SE@126:motion_field`
- stationaryish/uncertain samples: `nearly stationary:motion_field`, `uncertain:suppressed`

## Verification

Unit tests:

- `uv run pytest tests/unit/test_models.py -q`
- result: `14 passed in 0.20s`
- `uv run pytest tests/unit/test_live_replay_contracts.py -q`
- result: `7 passed in 2.72s`
- `uv run pytest tests/unit/test_tracking_evaluation.py -q`
- result: `1 passed in 2.78s`

Smoke tests:

- `uv run pytest tests/smoke/test_server_smoke.py -q`
- result: `9 passed in 3.15s`

Live validation:

- `uv run python scripts/evaluate_tracking.py --manifest docs/benchmarks/tracking_benchmark_manifest_broader_validation.json --output-json docs/test_reports/2026-04-14-reported-heading-observability-evaluation.json --output-md docs/test_reports/2026-04-14-reported-heading-observability-evaluation.md`
- `uv run python scripts/live_replay.py KTLX --date 2026-04-10 --scans 8 --local-only --end-filename KTLX20260410_214445_V06`

## Key Dense-Window Evidence

The earlier dense evening window is now directly interpretable from the recent reported heading sequence.

Middle scans that are allowed to publish:

- `2026-04-10T21:26:35Z`
  - `focus_reported_sequence=stationary:track_history|NE@48:motion_field|NE@51:motion_field|SE@126:motion_field`
  - summary: `moving SE at 21 mph`
- `2026-04-10T21:30:56Z`
  - `focus_reported_sequence=NE@48:motion_field|NE@51:motion_field|SE@126:motion_field|S@182:motion_field`
  - summary: `moving S at 22 mph`

Late scans that remain suppressed:

- `2026-04-10T21:35:40Z`
  - `focus_reported_sequence=NE@51:motion_field|SE@126:motion_field|S@182:motion_field|NNE@33:motion_field`
  - summary: `tracking uncertain`
- `2026-04-10T21:40:23Z`
  - `focus_reported_sequence=SE@126:motion_field|S@182:motion_field|NNE@33:motion_field|ESE@117:motion_field`
  - summary: `tracking uncertain`
- `2026-04-10T21:44:45Z`
  - `focus_reported_sequence=S@182:motion_field|NNE@33:motion_field|ESE@117:motion_field|NE@48:motion_field`
  - summary: `tracking uncertain`

That makes the remaining suppression rationale explicit:

- the late window is not merely one isolated reversal
- it is a multi-step unstable heading sequence under heavy merge/split pressure

## Why this matters

- the remaining dense-scene suppressions can now be reviewed from first-order evidence instead of post-hoc interpretation
- future tuning can target the actual sequence pattern instead of only counting flips
- the same observability is available in both offline benchmark reports and runtime API/live replay diagnostics

## Conclusion

- the current remaining late-window uncertainty looks justified by unstable reported-heading history
- the next refinement, if any, should be based on sequence-pattern methodology, not guesswork about whether a single flip counter is too strict
