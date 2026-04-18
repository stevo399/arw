# Mixed Heading Stability Suppression Follow-Up

Date: 2026-04-18

Purpose: suppress `mixed` reported heading stability under dense structural pressure so single-reversal cases no longer publish uncertain motion to blind users.

## What changed

- focus continuity now applies a -0.3 penalty to `mixed` heading stability under high structural pressure (>= 6 merge/split events), increased from -0.2
- focus continuity now also applies a -0.15 penalty to `mixed` heading stability under elevated structural pressure (>= 4 events), previously no penalty
- the penalty now matches on `label == "mixed"` (all three variants: 0.45, 0.6, 0.65) instead of `score <= 0.45` (only the worst variant)
- `mixed` stability in low-structural-pressure scenes is unaffected

## Verification

Unit tests:

- `uv run pytest tests/unit/test_tracking_focus.py -q`
- result: `10 passed in 10.24s`

Full test suite:

- `uv run pytest tests/ -q`
- result: `164 passed in 4.24s`

## Benchmark Evidence

From `docs/test_reports/2026-04-18-mixed-suppression-evaluation.md`:

### dense_cached_evening_window (KTLX, 8 scans, ~75 objects)

- No change from prior evaluation
- 3 publishable motion summaries, 4 "tracking uncertain"
- coherent turn (NE→SE→S) still publishes; late unstable reversals still suppress

### dense_cached_extended (KTLX, 8 scans, ~52 objects)

- Previous: `23:46:05Z` had `continuity=high:0.75`, spoke "moving WNW"
- Now: `23:46:05Z` has `continuity=medium:0.45`, says "tracking uncertain"
- The `mixed:0.65` single reversal (SE→WNW) is now correctly suppressed
- Summary counts changed from 2 uncertain / 5 moving to 3 uncertain / 4 moving

### lower_complexity_extended (KSOX, 6 scans, ~4 objects)

- No change — 0 low-continuity, 0 uncertain, all stationary

### merge_split_evening_window (KEYX, 7 scans, ~4 objects)

- No change — 0 low-continuity, 0 false uncertainty

### merge_split_extended (KEYX, 7 scans, ~17 objects)

- Final scan `mixed:0.6` continuity dropped from 0.3 to 0.15 (already was "tracking uncertain")
- No behavior change, just stronger suppression of already-suppressed case

## Broader Live Replay Validation

Four additional replay windows tested with no regressions:

### KTLX afternoon (17:05-17:37, 52-80 objects)

- ESE→N reversal at 17:17: `mixed:0.45` → "tracking uncertain" (correct)
- Heading stabilizes to consistent E by 17:33 → publishes "moving E at 30 mph"
- Next scan suppresses again due to low identity — appropriate conservative behavior

### KTLX earlier evening (20:14-20:49, 52-73 objects)

- Coherent ESE→SE turning motion publishes across 4 consecutive scans
- Final focus switch with `mixed:0.65` correctly suppresses
- No false uncertainty on well-tracked scans

### KEYX later window (23:10-23:56, 10-21 objects)

- Mostly stationary objects, all publish correctly
- Final scan WNW→NNE reversal (`mixed:0.6`) correctly suppresses
- No false uncertainty on stationary scans

### KTLX late window (23:19-23:56, 39-64 objects)

- Confirms benchmark: stable SE motion publishes, late WNW reversal suppresses

## Conclusion

- all `mixed` heading stability cases are now suppressed under dense structural pressure
- no regressions in any simpler or lower-complexity scene
- coherent turning motion remains publishable
- the heading stability classifier is no longer a remaining calibration gap
