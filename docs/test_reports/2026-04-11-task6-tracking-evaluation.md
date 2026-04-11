# Tracking Evaluation Report

Date: 2026-04-11

## Benchmarks

### dense_cached_quick

- category: `dense`
- site: `KTLX`
- scans: `3`
- mean objects: `49.33`
- mean active tracks: `55`
- max speed mph: `20`
- total merges: `14`
- total splits: `9`
- mean uncertain tracks: `0`
- focus switches: `0`
- focus heading flips >=90 deg: `1`
- total new tracks after first scan: `36`
- fragmentation proxy: `0.243`

Representative snapshots:

- `2026-04-10T23:46:05Z objects=48 active=48 uncertain=0 max_speed_mph=0 merges=0 splits=0 focus_track=4 focus_heading=stationary`
- `2026-04-10T23:51:18Z objects=50 active=56 uncertain=0 max_speed_mph=7 merges=8 splits=3 focus_track=4 focus_heading=NE`
- `2026-04-10T23:56:21Z objects=50 active=61 uncertain=0 max_speed_mph=20 merges=6 splits=6 focus_track=4 focus_heading=SE`

### lower_complexity_quick

- category: `lower_complexity`
- site: `KSOX`
- scans: `3`
- mean objects: `4`
- mean active tracks: `4`
- max speed mph: `0`
- total merges: `2`
- total splits: `1`
- mean uncertain tracks: `0`
- focus switches: `0`
- focus heading flips >=90 deg: `0`
- total new tracks after first scan: `7`
- fragmentation proxy: `0.583`

Representative snapshots:

- `2026-04-10T23:35:42Z objects=3 active=3 uncertain=0 max_speed_mph=0 merges=0 splits=0 focus_track=1 focus_heading=stationary`
- `2026-04-10T23:44:23Z objects=2 active=2 uncertain=0 max_speed_mph=0 merges=1 splits=0 focus_track=1 focus_heading=nearly stationary`
- `2026-04-10T23:53:02Z objects=7 active=7 uncertain=0 max_speed_mph=0 merges=1 splits=1 focus_track=1 focus_heading=nearly stationary`

### merge_split_regression

- category: `merge_split`
- site: `KEYX`
- scans: `5`
- mean objects: `19`
- mean active tracks: `20.4`
- max speed mph: `29`
- total merges: `12`
- total splits: `6`
- mean uncertain tracks: `0`
- focus switches: `1`
- focus heading flips >=90 deg: `0`
- total new tracks after first scan: `23`
- fragmentation proxy: `0.242`

Representative snapshots:

- `2026-04-10T23:37:55Z objects=18 active=18 uncertain=0 max_speed_mph=0 merges=0 splits=0 focus_track=1 focus_heading=stationary`
- `2026-04-10T23:42:37Z objects=21 active=24 uncertain=0 max_speed_mph=13 merges=3 splits=2 focus_track=1 focus_heading=nearly stationary`
- `2026-04-10T23:47:20Z objects=18 active=19 uncertain=0 max_speed_mph=0 merges=5 splits=1 focus_track=1 focus_heading=nearly stationary`
- `2026-04-10T23:52:02Z objects=18 active=22 uncertain=0 max_speed_mph=13 merges=1 splits=2 focus_track=1 focus_heading=nearly stationary`
- `2026-04-10T23:56:44Z objects=20 active=19 uncertain=0 max_speed_mph=29 merges=3 splits=1 focus_track=6 focus_heading=NE`
