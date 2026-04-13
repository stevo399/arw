# Tracking Evaluation Report

Date: 2026-04-12

## Benchmarks

### dense_cached_extended

- category: `dense_extended`
- site: `KTLX`
- scans: `8`
- mean objects: `51.88`
- mean active tracks: `61.25`
- max speed mph: `87`
- total merges: `52`
- total splits: `35`
- mean uncertain tracks: `8.0`
- mean focus identity confidence: `0.81`
- mean focus continuity: `0.67`
- mean focus motion confidence: `0.92`
- focus low-identity scans: `0`
- focus low-continuity scans: `1`
- focus low-motion scans: `0`
- focus switches: `0`
- focus heading flips >=90 deg: `3`
- focus flips with low motion confidence: `0`
- summary tracking-uncertain count: `1`
- summary moving-motion count: `6`
- summary stationary/nearly-stationary count: `1`
- total new tracks after first scan: `130`
- merged tracks total: `60`
- lost tracks total: `68`
- split children total: `39`
- absorbed links total: `60`
- fragmentation proxy: `0.313`

Representative snapshots:

- `2026-04-10T23:19:24Z objects=59 active=59 uncertain=0 max_speed_mph=0 merges=0 splits=0 focus_track=1 focus_identity=medium:0.6 focus_continuity=high:0.9 focus_heading=stationary focus_motion_conf=high:1.0`
- `2026-04-10T23:24:50Z objects=64 active=74 uncertain=6 max_speed_mph=25 merges=10 splits=7 focus_track=1 focus_identity=high:0.85 focus_continuity=medium:0.7 focus_heading=SSE focus_motion_conf=medium:0.6`
- `2026-04-10T23:30:07Z objects=57 active=69 uncertain=17 max_speed_mph=87 merges=11 splits=7 focus_track=1 focus_identity=high:0.84 focus_continuity=medium:0.7 focus_heading=SE focus_motion_conf=high:0.9`
- `2026-04-10T23:35:25Z objects=39 active=61 uncertain=14 max_speed_mph=18 merges=5 splits=5 focus_track=1 focus_identity=high:0.88 focus_continuity=medium:0.7 focus_heading=SE focus_motion_conf=high:0.9`
- `2026-04-10T23:40:53Z objects=48 active=54 uncertain=8 max_speed_mph=33 merges=7 splits=2 focus_track=1 focus_identity=high:0.88 focus_continuity=medium:0.7 focus_heading=SE focus_motion_conf=high:0.98`
- `2026-04-10T23:46:05Z objects=48 active=56 uncertain=4 max_speed_mph=20 merges=5 splits=5 focus_track=1 focus_identity=high:0.79 focus_continuity=medium:0.7 focus_heading=WNW focus_motion_conf=high:0.98`
- `2026-04-10T23:51:18Z objects=50 active=56 uncertain=6 max_speed_mph=20 merges=8 splits=3 focus_track=1 focus_identity=high:0.81 focus_continuity=low:0.3 focus_heading=NE focus_motion_conf=high:0.98`
- `2026-04-10T23:56:21Z objects=50 active=61 uncertain=9 max_speed_mph=21 merges=6 splits=6 focus_track=1 focus_identity=high:0.81 focus_continuity=medium:0.7 focus_heading=SE focus_motion_conf=high:0.98`

### lower_complexity_extended

- category: `lower_complexity_extended`
- site: `KSOX`
- scans: `6`
- mean objects: `4.0`
- mean active tracks: `4.0`
- max speed mph: `0`
- total merges: `5`
- total splits: `1`
- mean uncertain tracks: `0.0`
- mean focus identity confidence: `0.65`
- mean focus continuity: `0.89`
- mean focus motion confidence: `1.0`
- focus low-identity scans: `1`
- focus low-continuity scans: `0`
- focus low-motion scans: `0`
- focus switches: `0`
- focus heading flips >=90 deg: `0`
- focus flips with low motion confidence: `0`
- summary tracking-uncertain count: `0`
- summary moving-motion count: `0`
- summary stationary/nearly-stationary count: `6`
- total new tracks after first scan: `15`
- merged tracks total: `12`
- lost tracks total: `0`
- split children total: `2`
- absorbed links total: `12`
- fragmentation proxy: `0.625`

Representative snapshots:

- `2026-04-10T23:09:40Z objects=4 active=4 uncertain=0 max_speed_mph=0 merges=0 splits=0 focus_track=1 focus_identity=medium:0.58 focus_continuity=high:0.9 focus_heading=stationary focus_motion_conf=high:1.0`
- `2026-04-10T23:18:21Z objects=4 active=4 uncertain=0 max_speed_mph=0 merges=1 splits=0 focus_track=1 focus_identity=high:0.75 focus_continuity=high:1.0 focus_heading=nearly stationary focus_motion_conf=medium:1.0`
- `2026-04-10T23:27:02Z objects=4 active=4 uncertain=0 max_speed_mph=0 merges=1 splits=0 focus_track=1 focus_identity=low:0.4 focus_continuity=high:0.75 focus_heading=nearly stationary focus_motion_conf=medium:1.0`
- `2026-04-10T23:35:42Z objects=3 active=3 uncertain=0 max_speed_mph=0 merges=1 splits=0 focus_track=1 focus_identity=medium:0.73 focus_continuity=high:0.9 focus_heading=nearly stationary focus_motion_conf=medium:1.0`
- `2026-04-10T23:44:23Z objects=2 active=2 uncertain=0 max_speed_mph=0 merges=1 splits=0 focus_track=1 focus_identity=medium:0.74 focus_continuity=high:0.9 focus_heading=nearly stationary focus_motion_conf=medium:1.0`
- `2026-04-10T23:53:02Z objects=7 active=7 uncertain=0 max_speed_mph=0 merges=1 splits=1 focus_track=1 focus_identity=medium:0.71 focus_continuity=high:0.9 focus_heading=nearly stationary focus_motion_conf=medium:0.99`

### merge_split_extended

- category: `merge_split_extended`
- site: `KEYX`
- scans: `7`
- mean objects: `16.57`
- mean active tracks: `17.86`
- max speed mph: `35`
- total merges: `14`
- total splits: `11`
- mean uncertain tracks: `1.57`
- mean focus identity confidence: `0.74`
- mean focus continuity: `0.84`
- mean focus motion confidence: `0.94`
- focus low-identity scans: `0`
- focus low-continuity scans: `0`
- focus low-motion scans: `0`
- focus switches: `2`
- focus heading flips >=90 deg: `0`
- focus flips with low motion confidence: `0`
- summary tracking-uncertain count: `0`
- summary moving-motion count: `1`
- summary stationary/nearly-stationary count: `6`
- total new tracks after first scan: `41`
- merged tracks total: `18`
- lost tracks total: `14`
- split children total: `18`
- absorbed links total: `18`
- fragmentation proxy: `0.353`

Representative snapshots:

- `2026-04-10T23:14:46Z objects=10 active=10 uncertain=0 max_speed_mph=0 merges=0 splits=0 focus_track=2 focus_identity=medium:0.57 focus_continuity=high:0.9 focus_heading=stationary focus_motion_conf=high:1.0`
- `2026-04-10T23:19:28Z objects=11 active=12 uncertain=0 max_speed_mph=26 merges=2 splits=0 focus_track=2 focus_identity=high:0.83 focus_continuity=high:1.0 focus_heading=nearly stationary focus_motion_conf=medium:0.6`
- `2026-04-10T23:37:55Z objects=18 active=19 uncertain=4 max_speed_mph=0 merges=2 splits=4 focus_track=11 focus_identity=high:0.79 focus_continuity=medium:0.7 focus_heading=nearly stationary focus_motion_conf=medium:0.99`
- `2026-04-10T23:42:37Z objects=21 active=24 uncertain=1 max_speed_mph=29 merges=1 splits=3 focus_track=11 focus_identity=high:0.8 focus_continuity=high:0.85 focus_heading=nearly stationary focus_motion_conf=medium:0.99`
- `2026-04-10T23:47:20Z objects=18 active=19 uncertain=4 max_speed_mph=9 merges=5 splits=1 focus_track=11 focus_identity=high:0.79 focus_continuity=medium:0.7 focus_heading=nearly stationary focus_motion_conf=medium:0.99`
- `2026-04-10T23:52:02Z objects=18 active=22 uncertain=1 max_speed_mph=13 merges=1 splits=2 focus_track=11 focus_identity=medium:0.59 focus_continuity=high:0.9 focus_heading=nearly stationary focus_motion_conf=medium:0.99`
- `2026-04-10T23:56:44Z objects=20 active=19 uncertain=1 max_speed_mph=35 merges=3 splits=1 focus_track=25 focus_identity=high:0.83 focus_continuity=high:0.85 focus_heading=NNE focus_motion_conf=high:0.99`
