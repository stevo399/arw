# Reported Heading Stability Follow-Up

Date: 2026-04-15

Purpose: replace the blunt reported-heading reversal check with a generic sequence-pattern classifier so the tracker can separate coherent one-direction turning motion from oscillatory or reversal-prone heading behavior in dense live windows.

## What changed

- focus continuity now classifies recent reported-heading history as:
  - `insufficient`
  - `stable`
  - `coherent_turn`
  - `mixed`
  - `unstable`
- the classifier uses the recent reported heading sequence itself, not any site-specific rule table
- replay, benchmark, and API diagnostics now expose:
  - `reported_heading_stability_label`
  - `reported_heading_stability_score`
  - `reported_heading_stability_reason`

Method summary:

- fewer than two directional samples stays `insufficient`
- low-delta same-direction sequences are `stable`
- one-direction turning sequences stay `coherent_turn`
- a single abrupt reversal is `mixed`
- repeated sign changes or reversal-heavy oscillation becomes `unstable`

## Verification

Unit tests:

- `uv run pytest tests/unit/test_tracking_focus.py -q`
- result: `10 passed in 2.85s`
- `uv run pytest tests/unit/test_live_replay_contracts.py -q`
- result: `7 passed in 3.00s`
- `uv run pytest tests/unit/test_models.py -q`
- result: `14 passed in 0.15s`
- `uv run pytest tests/unit/test_tracking_evaluation.py -q`
- result: `1 passed in 2.93s`

Smoke tests:

- `uv run pytest tests/smoke/test_server_smoke.py -q`
- result: `9 passed in 3.55s`

Live validation:

- `uv run python scripts/live_replay.py KTLX --date 2026-04-10 --scans 8 --local-only --end-filename KTLX20260410_214445_V06`
- `uv run python scripts/evaluate_tracking.py --manifest docs/benchmarks/tracking_benchmark_manifest_broader_validation.json --output-json docs/test_reports/2026-04-15-reported-heading-stability-evaluation.json --output-md docs/test_reports/2026-04-15-reported-heading-stability-evaluation.md`

## Key Dense-Window Evidence

The dense evening window now separates coherent turning from true instability directly in the live replay output.

Publishable middle-window scans:

- `2026-04-10T21:26:35Z`
  - `focus_reported_sequence=stationary:track_history|NE@48:motion_field|NE@51:motion_field|SE@126:motion_field`
  - `focus_reported_stability=coherent_turn:0.85`
  - summary: `moving SE at 21 mph`
- `2026-04-10T21:30:56Z`
  - `focus_reported_sequence=NE@48:motion_field|NE@51:motion_field|SE@126:motion_field|S@182:motion_field`
  - `focus_reported_stability=coherent_turn:0.85`
  - summary: `moving S at 22 mph`

Suppressed late-window scans:

- `2026-04-10T21:35:40Z`
  - `focus_reported_sequence=NE@51:motion_field|SE@126:motion_field|S@182:motion_field|NNE@33:motion_field`
  - `focus_reported_stability=unstable:0.2`
  - summary: `tracking uncertain`
- `2026-04-10T21:40:23Z`
  - `focus_reported_sequence=SE@126:motion_field|S@182:motion_field|NNE@33:motion_field|ESE@117:motion_field`
  - `focus_reported_stability=unstable:0.1`
  - summary: `tracking uncertain`
- `2026-04-10T21:44:45Z`
  - `focus_reported_sequence=S@182:motion_field|NNE@33:motion_field|ESE@117:motion_field|NE@48:motion_field`
  - `focus_reported_stability=unstable:0.1`
  - summary: `tracking uncertain`

## Broader Replay Evidence

From `docs/test_reports/2026-04-15-reported-heading-stability-evaluation.md`:

- the dense evening validation window kept `3` publishable motion summaries and `4` `tracking uncertain` summaries
- the lower-complexity replay stayed stable, with `0` reported-heading reversal scans and `0` low-continuity scans
- the merge/split-sensitive evening window also stayed stable, with `0` reported-heading reversal scans and `0` low-continuity scans
- a later dense extended window now exposes a useful `mixed` category for single abrupt reversals, which is the main remaining calibration space

## Conclusion

- the tracker is no longer treating every large heading change as the same failure mode
- coherent turning motion now remains publishable in the dense live window
- genuinely oscillatory or reversal-prone sequences still collapse to low continuity and `tracking uncertain`
- the remaining tuning question is the generic handling of `mixed` one-off reversal cases, not any site-specific exception
