# Tracking Evaluation Report

Date: 2026-04-11

## Benchmarks

### dense_cached_quick

- category: `dense`
- site: `KTLX`
- scans: `3`
- mean objects: `49.33`
- mean active tracks: `55.0`
- max speed mph: `37`
- total merges: `14`
- total splits: `9`
- mean uncertain tracks: `0.0`
- mean focus identity confidence: `0.6`
- mean focus motion confidence: `0.99`
- focus low-identity scans: `1`
- focus low-motion scans: `0`
- focus switches: `0`
- focus heading flips >=90 deg: `1`
- focus flips with low motion confidence: `0`
- total new tracks after first scan: `36`
- fragmentation proxy: `0.243`

Representative snapshots:

- `2026-04-10T23:46:05Z objects=48 active=48 uncertain=0 max_speed_mph=0 merges=0 splits=0 focus_track=4 focus_identity=medium:0.6 focus_heading=stationary focus_motion_conf=high:1.0`
- `2026-04-10T23:51:18Z objects=50 active=56 uncertain=0 max_speed_mph=23 merges=8 splits=3 focus_track=4 focus_identity=low:0.4 focus_heading=NE focus_motion_conf=high:0.98`
- `2026-04-10T23:56:21Z objects=50 active=61 uncertain=0 max_speed_mph=37 merges=6 splits=6 focus_track=4 focus_identity=high:0.8 focus_heading=SE focus_motion_conf=high:0.98`

### lower_complexity_quick

- category: `lower_complexity`
- site: `KSOX`
- scans: `3`
- mean objects: `4.0`
- mean active tracks: `4.0`
- max speed mph: `2`
- total merges: `2`
- total splits: `1`
- mean uncertain tracks: `0.0`
- mean focus identity confidence: `0.73`
- mean focus motion confidence: `0.86`
- focus low-identity scans: `0`
- focus low-motion scans: `0`
- focus switches: `0`
- focus heading flips >=90 deg: `0`
- focus flips with low motion confidence: `0`
- total new tracks after first scan: `7`
- fragmentation proxy: `0.583`

Representative snapshots:

- `2026-04-10T23:35:42Z objects=3 active=3 uncertain=0 max_speed_mph=0 merges=0 splits=0 focus_track=1 focus_identity=medium:0.58 focus_heading=stationary focus_motion_conf=high:1.0`
- `2026-04-10T23:44:23Z objects=2 active=2 uncertain=0 max_speed_mph=2 merges=1 splits=0 focus_track=1 focus_identity=high:0.83 focus_heading=E focus_motion_conf=medium:0.6`
- `2026-04-10T23:53:02Z objects=7 active=7 uncertain=0 max_speed_mph=0 merges=1 splits=1 focus_track=1 focus_identity=high:0.77 focus_heading=nearly stationary focus_motion_conf=medium:0.99`

### merge_split_regression

- category: `merge_split`
- site: `KEYX`
- scans: `5`
- mean objects: `19.0`
- mean active tracks: `20.4`
- max speed mph: `35`
- total merges: `12`
- total splits: `6`
- mean uncertain tracks: `0.0`
- mean focus identity confidence: `0.74`
- mean focus motion confidence: `0.88`
- focus low-identity scans: `0`
- focus low-motion scans: `0`
- focus switches: `1`
- focus heading flips >=90 deg: `1`
- focus flips with low motion confidence: `0`
- total new tracks after first scan: `23`
- fragmentation proxy: `0.242`

Representative snapshots:

- `2026-04-10T23:37:55Z objects=18 active=18 uncertain=0 max_speed_mph=0 merges=0 splits=0 focus_track=1 focus_identity=medium:0.57 focus_heading=stationary focus_motion_conf=high:1.0`
- `2026-04-10T23:42:37Z objects=21 active=24 uncertain=0 max_speed_mph=24 merges=3 splits=2 focus_track=1 focus_identity=high:0.83 focus_heading=N focus_motion_conf=medium:0.6`
- `2026-04-10T23:47:20Z objects=18 active=19 uncertain=0 max_speed_mph=18 merges=5 splits=1 focus_track=1 focus_identity=high:0.83 focus_heading=S focus_motion_conf=high:0.9`
- `2026-04-10T23:52:02Z objects=18 active=22 uncertain=0 max_speed_mph=13 merges=1 splits=2 focus_track=1 focus_identity=medium:0.63 focus_heading=nearly stationary focus_motion_conf=medium:0.99`
- `2026-04-10T23:56:44Z objects=20 active=19 uncertain=0 max_speed_mph=35 merges=3 splits=1 focus_track=3 focus_identity=high:0.83 focus_heading=nearly stationary focus_motion_conf=high:0.9`
