# Live Replay Harness

Purpose: provide a repeatable command for replaying multiple live scans through the current tracking pipeline and printing summary + motion sanity diagnostics.

## Command

Run from the repo root:

```powershell
uv run python scripts/live_replay.py <SITE_ID> --scans 5
```

Optional historical replay for a specific day:

```powershell
uv run python scripts/live_replay.py <SITE_ID> --date 2026-04-10 --scans 5
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

## When To Use

Run the harness whenever a change affects:

- ingest behavior
- segmentation
- association
- motion estimation
- summary generation
- `/summary`, `/tracks`, or `/motion`

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
