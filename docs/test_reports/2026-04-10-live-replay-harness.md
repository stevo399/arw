# Live Replay Harness

Purpose: provide a repeatable command for replaying multiple live scans through the current tracking pipeline and printing summary + motion sanity diagnostics.

## Command

Run from the repo root:

```powershell
uv run python scripts/live_replay.py <SITE_ID> --scans 5
```

Fast development replay:

```powershell
uv run python scripts/live_replay.py <SITE_ID> --quick
```

Optional historical replay for a specific day:

```powershell
uv run python scripts/live_replay.py <SITE_ID> --date 2026-04-10 --scans 5
```

Cached-only replay with no downloads:

```powershell
uv run python scripts/live_replay.py <SITE_ID> --date 2026-04-10 --quick --local-only
```

## Output

For each replayed scan the harness prints:

- timestamp
- object count
- active track count
- uncertain track count
- maximum reported track speed in mph
- merge count
- split count
- generated speech summary

This makes it easy to detect:

- duplicate/self-merge regressions
- dense-scene motion blowups
- summaries that should say `tracking uncertain` but do not

The harness also prints a `MODE` line so you can tell whether it is replaying cached scans only or fetching missing files.

## When To Use

Run the harness whenever a change affects:

- ingest behavior
- segmentation
- association
- motion estimation
- summary generation
- `/summary`, `/tracks`, or `/motion`

For fast iteration:

- use `--quick` during development
- use `--local-only` when you want cached replay without waiting on downloads
- reserve longer historical windows for final validation

## Notes

- A single latest-scan fetch will often show `stationary` because there is not yet enough history.
- Replaying multiple scans is the intended way to validate motion behavior.

## Validation Runs

### Historical merge/split regression replay

Command:

```powershell
uv run python scripts/live_replay.py KEYX --date 2026-04-10 --scans 5
```

Observed behavior:

- replay completed across five scans with no duplicate or self-merge symptoms
- merge/split counts stayed plausible for a low-complexity scene
- max reported speed stayed between `0` and `22` mph
- summaries remained stable and low-noise

Representative output:

- `2026-04-10T20:57:32Z objects=5 active=5 uncertain_tracks=0 max_speed_mph=0 merges=0 splits=0`
- `2026-04-10T21:26:46Z objects=11 active=12 uncertain_tracks=0 max_speed_mph=22 merges=1 splits=2`

### Dense live replay

Command:

```powershell
uv run python scripts/live_replay.py KTLX --scans 12
```

Observed behavior:

- replay completed across roughly one hour of recent scans
- dense-scene motion no longer blew up into absurd triple-digit outputs for spoken summaries
- uncertain tracks were surfaced explicitly and summaries used `tracking uncertain` when needed
- merge/split counts remained high, which is expected in a crowded scene
- strongest-object identity and area still showed instability in some scans and need follow-up validation

Representative output:

- `2026-04-10T20:44:53Z objects=89 active=95 uncertain_tracks=7 max_speed_mph=49 merges=16 splits=7`
- `2026-04-10T21:12:30Z objects=111 active=113 uncertain_tracks=17 max_speed_mph=42 merges=21 splits=15`
- `2026-04-10T21:21:58Z objects=115 active=125 uncertain_tracks=19 max_speed_mph=69 merges=29 splits=25`

Representative summary lines:

- `Oklahoma City: 89 rain objects detected. Strongest: severe core, 27 miles NE of the radar, tracking uncertain. Note: 16 storms merged in the last scan. Note: 7 storms split in the last scan. Covering approximately 170 square miles.`
- `Oklahoma City: 115 rain objects detected. Strongest: severe core, 117 miles E of the radar, tracking uncertain. Note: 29 storms merged in the last scan. Note: 25 storms split in the last scan. Covering approximately 22 square miles.`

## Completion Notes

- Unit verification: `uv run pytest tests/unit/test_live_replay_contracts.py -q`
- Smoke tests: not required for this task because no server runtime or API behavior changed

## Follow-Up Validation After Summary Stability Fix

### Dense live replay after summary fix

Command:

```powershell
uv run python scripts/live_replay.py KTLX --scans 12
```

Observed behavior:

- total coverage now stays scene-level instead of collapsing to the currently selected focal object
- coverage stayed in a consistent `7236` to `8582` square-mile band across the replay
- summaries continued to suppress unreliable motion with `tracking uncertain`
- dense scenes can still produce very large raw track speeds in diagnostics, but those tracks are being downgraded in spoken output rather than announced as fact

Representative output:

- `2026-04-10T21:12:30Z objects=111 active=113 uncertain_tracks=10 max_speed_mph=42 merges=20 splits=15`
- `2026-04-10T21:49:29Z objects=123 active=133 uncertain_tracks=14 max_speed_mph=289 merges=21 splits=21`
- `2026-04-10T21:54:00Z objects=130 active=149 uncertain_tracks=16 max_speed_mph=222 merges=27 splits=23`

Representative summary lines:

- `Oklahoma City: 111 rain objects detected. Strongest: severe core, 26 miles NE of the radar, tracking uncertain. Note: 20 storms merged in the last scan. Note: 15 storms split in the last scan. Covering approximately 7184 square miles.`
- `Oklahoma City: 123 rain objects detected. Strongest: severe core, 135 miles E of the radar, tracking uncertain. Note: 21 storms merged in the last scan. Note: 21 storms split in the last scan. Covering approximately 7354 square miles.`

### Lower-complexity live replay

Command:

```powershell
uv run python scripts/live_replay.py KSOX --scans 5
```

Observed behavior:

- low-complexity scene remained stable across five live scans
- strongest-object summaries stayed geographically consistent
- coverage remained small and plausible at `14` to `39` square miles
- max reported speed remained low at `0` to `36` mph

Representative output:

- `2026-04-10T22:35:03Z objects=4 active=4 uncertain_tracks=0 max_speed_mph=0 merges=0 splits=0`
- `2026-04-10T22:52:26Z objects=6 active=6 uncertain_tracks=0 max_speed_mph=36 merges=2 splits=3`
- `2026-04-10T23:09:40Z objects=5 active=9 uncertain_tracks=1 max_speed_mph=5 merges=1 splits=2`

## Follow-Up Validation After Motion Methodology Upgrade

### What changed

- reported motion now carries provenance
- weak-identity tracks use field-guided motion or suppression instead of raw centroid-history motion
- scene guidance now prefers scan-based phase correlation instead of weighted object centroids
- field-derived motion is also subject to plausibility suppression

### Dense historical replay after motion methodology upgrade

Command:

```powershell
uv run python scripts/live_replay.py KTLX --date 2026-04-10 --scans 12
```

Observed behavior:

- dense-scene focal summaries stayed in low double-digit speeds rather than surfacing triple-digit false motion
- the previous field-driven extreme outlier was suppressed and counted as uncertain instead of being exposed as a 600+ mph report
- scene coverage remained stable in the `3635` to `4927` square-mile range
- merge/split counts remained high, which is expected for the scene

Representative output:

- `2026-04-10T23:03:32Z objects=74 active=86 uncertain_tracks=1 max_speed_mph=71 merges=16 splits=12`
- `2026-04-10T23:24:50Z objects=95 active=108 uncertain_tracks=0 max_speed_mph=36 merges=18 splits=17`
- `2026-04-10T23:51:18Z objects=77 active=81 uncertain_tracks=0 max_speed_mph=21 merges=12 splits=10`

Representative summary lines:

- `Oklahoma City: 74 rain objects detected. Strongest: severe core, 28 miles E of the radar, moving SW at 8 mph. Note: 16 storms merged in the last scan. Note: 12 storms split in the last scan. Covering approximately 4805 square miles.`
- `Oklahoma City: 95 rain objects detected. Strongest: severe core, 89 miles NNE of the radar, moving ESE at 18 mph. Note: 18 storms merged in the last scan. Note: 17 storms split in the last scan. Covering approximately 3904 square miles.`

### Verification

- targeted regression set:
  - `uv run pytest tests/unit/test_tracking_motion_field.py tests/unit/test_motion.py tests/unit/test_tracker.py tests/unit/test_summary.py tests/smoke/test_server_smoke.py -q`
  - `52 passed in 2.94s`

## Follow-Up Validation After Segmentation Upgrade

### What changed

- low-threshold parent blobs can now be partitioned around multiple intense internal cores
- the split trigger is intentionally conservative: it requires multiple intense cores rather than splitting on every moderate pocket
- endpoint contracts stay the same; only mask/object partitioning changed

### Verification

- targeted regression set:
  - `uv run pytest tests/unit/test_detection.py tests/unit/test_tracking_segmentation.py tests/unit/test_tracking_association.py tests/unit/test_motion.py tests/unit/test_summary.py tests/smoke/test_server_smoke.py tests/e2e/test_full_pipeline.py -q`
  - `57 passed in 3.29s`

### Dense replay after segmentation upgrade

Command:

```powershell
uv run python scripts/live_replay.py KTLX --date 2026-04-10 --quick --local-only
```

Observed behavior:

- crowded-scene summaries remained plausible
- scene coverage remained stable in the `3692` to `3863` square-mile range
- object counts and merge/split counts remained high, but no absurd summary motion returned

Representative output:

- `2026-04-10T23:46:05Z objects=80 active=80 uncertain_tracks=0 max_speed_mph=0 merges=0 splits=0`
- `2026-04-10T23:51:18Z objects=96 active=102 uncertain_tracks=0 max_speed_mph=23 merges=15 splits=16`
- `2026-04-10T23:56:21Z objects=86 active=103 uncertain_tracks=0 max_speed_mph=37 merges=19 splits=15`

### Lower-complexity replay after segmentation upgrade

Command:

```powershell
uv run python scripts/live_replay.py KSOX --date 2026-04-10 --quick
```

Observed behavior:

- the stricter split trigger avoided the earlier extreme over-segmentation on the first two scans
- first two scans stayed in a plausible `5` to `6` object range with low speeds
- the third scan still rose to `20` objects, which is acceptable for now but should be watched as part of future segmentation tuning

Representative output:

- `2026-04-10T23:35:42Z objects=5 active=5 uncertain_tracks=0 max_speed_mph=0 merges=0 splits=0`
- `2026-04-10T23:44:23Z objects=6 active=6 uncertain_tracks=0 max_speed_mph=2 merges=1 splits=1`
- `2026-04-10T23:53:02Z objects=20 active=18 uncertain_tracks=0 max_speed_mph=2 merges=1 splits=2`
