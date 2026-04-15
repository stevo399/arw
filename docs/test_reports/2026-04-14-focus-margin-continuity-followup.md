# Focus-Margin Continuity Follow-Up

Date: 2026-04-14

Purpose: refine focus continuity so dense-scene structural pressure does not automatically suppress motion when the same focus track is still winning clearly against challengers, while preserving suppression for actual reported-motion reversals and low-motion-confidence cases.

## Problem

The broader evening dense replay window exposed a gap in the continuity model:

- the focus track was not switching
- the selection margin remained strong
- reported motion was still internally stable in the middle of the window
- but raw centroid-step heading-flip penalties were still collapsing focus continuity and forcing `tracking uncertain`

That produced false suppression in the middle portion of the earlier dense window.

## Change

Focus continuity now treats strong challenger separation as evidence that dense structural chaos does not automatically imply focus ambiguity.

Specifically:

- if structural pressure is elevated
- and the focus selection margin is strong
- and reported motion remains high-confidence without reported reversals

then the raw track-position heading-flip penalty is suppressed and a small focus-margin bonus is applied.

This does **not** override:

- reported-motion reversal penalties
- low motion-confidence penalties
- weak identity penalties

## Verification

Unit tests:

- `uv run pytest tests/unit/test_tracking_focus.py -q`
- result: `9 passed in 2.44s`
- `uv run pytest tests/unit/test_summary.py -q`
- result: `15 passed in 2.37s`
- `uv run pytest tests/unit/test_live_replay_contracts.py -q`
- result: `7 passed in 2.50s`

Smoke tests:

- `uv run pytest tests/smoke/test_server_smoke.py -q`
- result: `9 passed in 2.96s`

Live validation:

- `uv run python scripts/evaluate_tracking.py --manifest docs/benchmarks/tracking_benchmark_manifest_broader_validation.json --output-json docs/test_reports/2026-04-14-focus-margin-continuity-evaluation.json --output-md docs/test_reports/2026-04-14-focus-margin-continuity-evaluation.md`
- `uv run python scripts/live_replay.py KTLX --date 2026-04-10 --scans 8 --local-only --end-filename KTLX20260410_214445_V06`

## Key Dense-Window Result

Earlier evening dense-window replay now behaves as follows:

- `2026-04-10T21:21:58Z`
  - remains suppressed
  - `focus_identity=low:0.4`
  - `focus_continuity=medium:0.6`
  - summary: `tracking uncertain`
- `2026-04-10T21:26:35Z`
  - now republished
  - `focus_identity=medium:0.63`
  - `focus_continuity=high:0.75`
  - `focus_selection_margin=4.76`
  - summary: `moving SE at 21 mph`
- `2026-04-10T21:30:56Z`
  - now republished
  - `focus_identity=high:0.78`
  - `focus_continuity=high:0.85`
  - `focus_selection_margin=4.09`
  - summary: `moving S at 22 mph`
- `2026-04-10T21:35:40Z`
  - still suppressed
  - `focus_reported_flips=1`
  - `focus_continuity=low:0.0`
  - summary: `tracking uncertain`

This is the intended distinction:

- strong focus dominance plus stable reported motion can publish
- true reported-motion reversal still cannot

## Benchmark Deltas

Compared with the prior broader-window report:

- `dense_cached_evening_window`
  - mean focus continuity: `0.31 -> 0.49`
  - focus low-continuity scans: `6 -> 4`
  - summary tracking-uncertain count: `6 -> 4`
  - summary moving-motion count: `1 -> 3`
- `dense_cached_extended`
  - effectively unchanged at the decision level
  - mean focus continuity: `0.51 -> 0.52`
  - summary tracking-uncertain count remained `3`
- `lower_complexity_extended`
  - unchanged
  - summary tracking-uncertain count remained `0`
- `merge_split_evening_window`
  - unchanged
  - summary tracking-uncertain count remained `0`
- `merge_split_extended`
  - unchanged at the decision level
  - summary tracking-uncertain count remained `1`

## Conclusion

- the earlier dense-window false suppressions were caused by over-weighting raw centroid-step reversals in a case where focus dominance was still clear
- focus continuity is now more aligned with the observed product need: suppress real instability, not merely dense morphology
- the refinement improved the earlier dense window without reopening the already-fixed late dense reversal cases or the simpler validation windows
