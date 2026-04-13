# Tracking Evaluation Report

Date: YYYY-MM-DD

Purpose: summarize quantitative tracking metrics across the benchmark replay set so behavior changes can be compared over time without relying only on anecdotes.

## Inputs

- manifest: `docs/benchmarks/tracking_benchmark_manifest.json`
- command:

```powershell
uv run python scripts/evaluate_tracking.py --manifest docs/benchmarks/tracking_benchmark_manifest.json --output-json <JSON_PATH> --output-md <MD_PATH>
```

## Benchmarks

### <benchmark id>

- category:
- site:
- scans:
- mean objects:
- mean active tracks:
- max speed mph:
- total merges:
- total splits:
- mean uncertain tracks:
- mean focus identity confidence:
- mean focus continuity:
- mean focus motion confidence:
- focus low-identity scans:
- focus low-continuity scans:
- focus low-motion scans:
- focus switches:
- focus heading flips >=90 deg:
- focus flips with low motion confidence:
- summary tracking-uncertain count:
- summary moving-motion count:
- summary stationary/nearly-stationary count:
- total new tracks after first scan:
- merged tracks total:
- lost tracks total:
- split children total:
- absorbed links total:
- fragmentation proxy:

Representative snapshots:

- `<timestamp> ...`

## Notes

- call out regressions explicitly
- note whether any heading-flip or focus-switch metrics worsened
- note whether focus continuity tracked the same windows that were downgraded in summary output
- compare against the prior clean report when one exists
