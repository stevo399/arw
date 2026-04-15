# Detection Runtime Optimization

Date: 2026-04-14

Purpose: reduce dense-scene replay and benchmark runtime enough that broader cached-window validation can run to completion as a routine engineering task instead of timing out.

## Root Cause

Dense replay runtime was dominated by detection, not by Py-ART ingest or tracker update.

Profiled dense single-scan breakdown before the change:

```text
parse_s=1.689
preprocess_s=1.065
detect_s=184.942
track_s=0.001
objects=77
active_tracks=77
```

Measured dense single-scan replay wall time before the change:

```text
193.174261 seconds
```

Measured lower-complexity single-scan replay wall time before the change:

```text
9.416506 seconds
```

This isolated the problem to dense-scene detection behavior, not generic replay overhead.

## Code Changes

- vectorized object-area calculation by precomputing per-range-bin pixel areas instead of summing per pixel in Python
- vectorized intensity-layer area calculation using per-object pixel arrays instead of rebuilding full-grid masks for each layer
- replaced hierarchy parent assignment by full-grid mask-overlap scans with labeled-component overlap lookup from the prior threshold level
- vectorized remaining-pixel assignment during multilevel core splitting instead of iterating one pixel at a time in Python

These changes preserve the generic multilevel segmentation design and avoid any site-specific or case-specific tuning.

## Verification

Unit tests:

- `uv run pytest tests/unit/test_detection.py -q`
- result: `16 passed in 0.45s`
- `uv run pytest tests/unit/test_tracking_segmentation.py -q`
- result: `5 passed in 2.40s`

Smoke tests:

- `uv run pytest tests/smoke/test_server_smoke.py -q`
- result: `9 passed in 2.83s`

Live validation:

- `uv run python scripts/evaluate_tracking.py --manifest docs/benchmarks/tracking_benchmark_manifest_broader_validation.json --output-json docs/test_reports/2026-04-14-broader-window-validation-evaluation.json --output-md docs/test_reports/2026-04-14-broader-window-validation-evaluation.md`
- result: completed successfully

## Runtime Impact

Profiled dense single-scan breakdown after the change:

```text
parse_s=1.754
preprocess_s=1.127
detect_s=33.288
track_s=0.001
objects=77
active_tracks=77
```

Measured dense single-scan replay wall time after the change:

```text
40.535603 seconds
```

Observed speedup on the dense single-scan replay:

- wall time improved from about `193.17s` to `40.54s`
- dense detection stage improved from about `184.94s` to `33.29s`

## Broader Validation Outcome

The full broader benchmark manifest now runs to completion and produces reviewable clean data:

- [2026-04-14-broader-window-validation-evaluation.md](/C:/Users/steve/Documents/arw/docs/test_reports/2026-04-14-broader-window-validation-evaluation.md)
- [2026-04-14-broader-window-validation-evaluation.json](/C:/Users/steve/Documents/arw/docs/test_reports/2026-04-14-broader-window-validation-evaluation.json)

Notable benchmark results from the newly completed broader run:

- `dense_cached_evening_window`
  - `mean focus continuity: 0.31`
  - `focus low-continuity scans: 6`
  - `summary tracking-uncertain count: 6`
  - `focus switches: 0`
- `dense_cached_extended`
  - remains the calmer dense late window
  - `mean focus continuity: 0.51`
  - `focus low-continuity scans: 3`
- `merge_split_evening_window`
  - `mean focus continuity: 0.97`
  - `focus switches: 2`
  - no heading-reversal hits

## Conclusions

- the broader validation bottleneck was real and was rooted in dense-scene detection, not in tracker state or report generation
- the optimized detection path keeps the same architecture and behavior class while materially reducing dense-scene replay cost
- broader cached-window evaluation is now operational again and is already surfacing additional dense-scene continuity pressure that was previously blocked by runtime
