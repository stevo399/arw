"""Verification (measured motion): does the motion ARW reports predict where storms go?

Runs every cached back-to-back window through the production history and
tracker.  For every storm seen in two consecutive scans whose reported motion
in the first had a velocity, predicts the second position and outline from that
motion and from no motion, and records both (centre error km, outline IoU).
Also counts reported motion by source and checks no storm is called stationary.

Writes a JSON summary to argv[1].
"""
import json
import math
import os
import statistics
import sys
from collections import Counter, defaultdict

sys.path.insert(0, ".")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import src.server as server
from pattern_motion_eval import windows
from src.geometry import gate_ground_xy_m
from src.history.store import RadarHistory
from src.sites import haversine_distance_km
from src.tracking.motion import KM_PER_DEGREE_LAT
from src.tracking.pattern_motion import GroundSampler


def outline_iou(mask_a, sweep_a, mask_b, sweep_b, east_m, north_m, grid_m=500.0):
    """IoU on a ground grid of outline A moved by (east, north) with outline B."""
    points = []
    for mask, sweep in ((mask_a, sweep_a), (mask_b, sweep_b)):
        rays, gates = np.nonzero(mask)
        points.append(gate_ground_xy_m(rays, gates, sweep.azimuths, sweep.ranges_m, sweep.elevations))
    x0 = min(points[0][0].min() + east_m, points[1][0].min()) - 2000.0
    x1 = max(points[0][0].max() + east_m, points[1][0].max()) + 2000.0
    y0 = min(points[0][1].min() + north_m, points[1][1].min()) - 2000.0
    y1 = max(points[0][1].max() + north_m, points[1][1].max()) + 2000.0
    gx, gy = np.meshgrid(np.arange(x0, x1, grid_m), np.arange(y0, y1, grid_m))
    moved = GroundSampler.from_sweep(sweep_a).sample(mask_a.astype(float), gx - east_m, gy - north_m, 0.0) > 0.5
    target = GroundSampler.from_sweep(sweep_b).sample(mask_b.astype(float), gx, gy, 0.0) > 0.5
    union = np.count_nonzero(moved | target)
    return float(np.count_nonzero(moved & target)) / union if union else 0.0


def predicted(lat, lon, east_km, north_km):
    return lat + north_km / KM_PER_DEGREE_LAT, lon + east_km / (KM_PER_DEGREE_LAT * math.cos(math.radians(lat)))

STRUCTURAL = {"merge_survivor", "split_child", "reacquired"}
rows = []
sources = Counter()
labels = Counter()
fastest = []

for site, run in windows():
    history = RadarHistory.open(site, None)
    previous = None
    for stamp, path in run:
        try:
            buffered = server._process_scan_file(site, os.path.abspath(path))
        except Exception as exc:
            print("skip", path, exc, flush=True)
            continue
        if history.add_live_scan(buffered) is None:
            continue
        seen = {}
        for track in history.tracker.active_tracks:
            if track.last_seen != buffered.timestamp:
                continue
            motion = track.get_motion()
            if motion.heading_label == "stationary":
                raise AssertionError(f"{site} {buffered.timestamp} track {track.track_id} reported stationary")
            sources[motion.source] += 1
            labels[motion.heading_label] += 1
            if motion.speed_kmh is not None:
                fastest.append((motion.speed_kmh, site, str(buffered.timestamp), track.track_id, motion.source))
            obj = track.current_object
            seen[track.track_id] = {
                "obj": obj, "mask": np.asarray(buffered.object_masks[obj.object_id]), "motion": motion,
                "event": track.identity_diagnostics.event_context if track.identity_diagnostics else None,
            }
        if previous is not None:
            hours = (buffered.timestamp - previous[0].timestamp).total_seconds() / 3600.0
            for track_id, target in seen.items():
                last = previous[1].get(track_id)
                if last is None or last["motion"].east_kmh is None:
                    continue
                lo, to = last["obj"], target["obj"]
                result = {"site": site, "source": last["motion"].source, "clean": target["event"] not in STRUCTURAL}
                for name, (east, north) in (("reported", (last["motion"].east_kmh, last["motion"].north_kmh)), ("stationary", (0.0, 0.0))):
                    p = predicted(lo.centroid_lat, lo.centroid_lon, east * hours, north * hours)
                    result[f"{name}_error_km"] = haversine_distance_km(*p, to.centroid_lat, to.centroid_lon)
                    result[f"{name}_iou"] = outline_iou(
                        last["mask"], previous[0].reflectivity_data, target["mask"], buffered.reflectivity_data,
                        east * hours * 1000.0, north * hours * 1000.0,
                    )
                rows.append(result)
        previous = (buffered, seen)
    print("window done", site, run[0][0], len(run), flush=True)


def summary(subset):
    if not subset:
        return {"n": 0}
    return {
        "n": len(subset),
        "median_error_km": {k: round(statistics.median(r[f"{k}_error_km"] for r in subset), 3) for k in ("reported", "stationary")},
        "p90_error_km": {k: round(float(np.percentile([r[f"{k}_error_km"] for r in subset], 90)), 3) for k in ("reported", "stationary")},
        "mean_iou": {k: round(statistics.fmean(r[f"{k}_iou"] for r in subset), 4) for k in ("reported", "stationary")},
    }


by_source = defaultdict(list)
for row in rows:
    by_source[row["source"]].append(row)
out = {
    "reported_motion_sources": dict(sources),
    "reported_heading_labels": dict(labels.most_common()),
    "fastest_reported": [list(item) for item in sorted(fastest, reverse=True)[:10]],
    "all": summary(rows),
    "clean_targets": summary([r for r in rows if r["clean"]]),
    "by_source": {source: summary(subset) for source, subset in sorted(by_source.items())},
    "by_site": {site: summary([r for r in rows if r["site"] == site]) for site in sorted({r["site"] for r in rows})},
}
with open(sys.argv[1], "w", encoding="utf-8") as handle:
    json.dump(out, handle, indent=2)
print(json.dumps(out, indent=1))
