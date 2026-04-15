# Diagnostics Consolidation Follow-Up

Date: 2026-04-14

Purpose: make the key focus-tracking diagnostics reviewable from the standard benchmark report instead of relying on ad hoc live replay inspection.

## Added Benchmark Diagnostics

The benchmark evaluation now includes:

- `mean focus selection margin`
- `focus reported-heading-reversal scans`
- `focus motion-field source scans`
- `focus suppressed-motion source scans`

Representative snapshots now also print:

- `focus_margin`
- `runner_up`
- `focus_reported_flips`
- `focus_motion_source`

## Why This Matters

The recent dense-window work relied on three distinct classes of evidence:

- whether the focus track was actually weakly selected
- whether published focus motion had reversed sharply
- whether motion was coming from track history, field fallback, or suppression

Those are now visible in the normal benchmark report. This means the next review cycle can compare tracker behavior across commits without another one-off diagnostic script.

## Example From The Dense Extended Window

- `2026-04-10T23:46:05Z ... focus_margin=2.88 runner_up=4 focus_reported_flips=1 focus_motion_source=motion_field ...`
- `2026-04-10T23:51:18Z ... focus_margin=2.49 runner_up=119 focus_reported_flips=1 focus_motion_source=suppressed ...`
- `2026-04-10T23:56:21Z ... focus_margin=1.11 runner_up=119 focus_reported_flips=1 focus_motion_source=suppressed ...`

These lines make the dense-window progression directly legible:

- the focus still wins
- reported motion has become reversal-prone
- the motion source transitions from field fallback to suppression late in the sequence

## Conclusion

- the benchmark report is now a better operational review artifact, not just a metric dump
- future tracker changes can be judged against the same motion-source and focus-pressure context that drove the recent live fixes
