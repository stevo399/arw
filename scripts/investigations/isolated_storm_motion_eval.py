"""Investigation (measured motion): how to accept a pattern match for a storm without neighbours.

Ran against commit e07e528 (measured motion, before guards for isolated storms).

The production neighbour-consistency guard needs three other matches within
60 km.  In the tracking benchmark, isolated weak storms at long range were
accepted at 120-190 km/h.  For every raw match the production tracker makes
on every cached back-to-back window, this records the match, how many
neighbours could judge it, and its correlation, speed and agreement with the
same storm's previous raw match.  At the next scan it records how well the
raw velocity, the reported motion and no motion predict the storm (centre
error and outline IoU), so candidate rules can be compared afterwards.

Writes one JSON line per scored target to argv[1].
"""
import json
import math
import os
import sys

sys.path.insert(0, ".")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import src.server as server
import src.tracker as tracker_module
from pattern_motion_eval import windows
from src.geometry import gate_ground_xy_m
from src.history.store import RadarHistory
from src.sites import haversine_distance_km
from src.tracking.motion import KM_PER_DEGREE_LAT
from src.tracking.pattern_motion import NEIGHBOUR_RADIUS_KM, GroundSampler

OUT = sys.argv[1]
captured = {}
real_match = tracker_module.match_storm


def capturing_match(current_sweep, mask, previous_sweep, hours):
    found = real_match(current_sweep, mask, previous_sweep, hours)
    captured.setdefault("matches", []).append((mask, found))
    return found


tracker_module.match_storm = capturing_match


def outline_iou(mask_a, sweep_a, mask_b, sweep_b, east_m, north_m, grid_m=500.0):
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


for window_index, (site, run) in enumerate(windows()):
    history = RadarHistory.open(site, None)
    previous = None  # (buffered, {track_id: entry})
    last_raw = {}  # track id -> previous raw (east, north)
    for stamp, path in run:
        try:
            buffered = server._process_scan_file(site, os.path.abspath(path))
        except Exception as exc:
            print("skip", path, exc, flush=True)
            continue
        captured["matches"] = []
        if history.add_live_scan(buffered) is None:
            continue
        by_mask = {id(mask): found for mask, found in captured["matches"]}
        seen = {}
        for track in history.tracker.active_tracks:
            if track.last_seen != buffered.timestamp:
                continue
            obj = track.current_object
            mask = np.asarray(buffered.object_masks[obj.object_id])
            seen[track.track_id] = {"obj": obj, "mask": mask, "motion": track.get_motion(), "positions": len(track.positions),
                                    "event": track.identity_diagnostics.event_context if track.identity_diagnostics else None,
                                    "accepted": bool(track.measured_velocities and track.measured_velocities[-1].timestamp == buffered.timestamp)}
        # The tracker passes each storm's own mask array, so identity pairs raw matches with tracks.
        raw_by_track = {track_id: by_mask[id(entry["mask"])] for track_id, entry in seen.items() if id(entry["mask"]) in by_mask}
        assert len(raw_by_track) == len(captured["matches"]), (len(raw_by_track), len(captured["matches"]))
        candidates = {tid: f for tid, f in raw_by_track.items() if f is not None and not f.at_edge}
        for track_id, entry in seen.items():
            found = raw_by_track.get(track_id)
            entry["raw"] = None if found is None else (found.east_kmh, found.north_kmh, found.correlation, found.at_edge)
            entry["neighbours"] = sum(
                1 for other in candidates
                if other != track_id and haversine_distance_km(seen[other]["obj"].centroid_lat, seen[other]["obj"].centroid_lon, entry["obj"].centroid_lat, entry["obj"].centroid_lon) <= NEIGHBOUR_RADIUS_KM
            )
            entry["previous_raw"] = last_raw.get(track_id)

        if previous is not None:
            hours = (buffered.timestamp - previous[0].timestamp).total_seconds() / 3600.0
            with open(OUT, "a", encoding="utf-8") as handle:
                for track_id, target in seen.items():
                    last = previous[1].get(track_id)
                    if last is None or last["raw"] is None:
                        continue
                    lo, to = last["obj"], target["obj"]
                    prev_sweep, sweep = previous[0].reflectivity_data, buffered.reflectivity_data
                    results = {}
                    options = {"stationary": (0.0, 0.0), "raw": last["raw"][:2]}
                    if last["motion"].east_kmh is not None:
                        options["reported"] = (last["motion"].east_kmh, last["motion"].north_kmh)
                    for name, (east, north) in options.items():
                        p = predicted(lo.centroid_lat, lo.centroid_lon, east * hours, north * hours)
                        results[name] = {
                            "centre_error_km": round(haversine_distance_km(*p, to.centroid_lat, to.centroid_lon), 3),
                            "iou": round(outline_iou(last["mask"], prev_sweep, target["mask"], sweep, east * hours * 1000.0, north * hours * 1000.0), 4),
                        }
                    handle.write(json.dumps({
                        "site": site, "window": window_index, "t": str(buffered.timestamp), "track": track_id,
                        "hours": round(hours, 4), "positions_before": last["positions"],
                        "raw": [round(v, 3) if isinstance(v, float) else v for v in last["raw"]],
                        "raw_speed_kmh": round(math.hypot(*last["raw"][:2]), 1),
                        "neighbours": last["neighbours"], "accepted": last["accepted"],
                        "previous_raw": None if last["previous_raw"] is None else [round(v, 2) for v in last["previous_raw"]],
                        "reported_source": last["motion"].source,
                        "area_km2": round(lo.area_km2, 1), "peak_dbz": lo.peak_dbz, "range_km": round(lo.distance_km, 1),
                        "target_event": target.get("event"),
                        "results": results,
                    }) + "\n")

        for track_id, entry in seen.items():
            if entry["raw"] is not None and not entry["raw"][3]:
                last_raw[track_id] = entry["raw"][:2]
        previous = (buffered, seen)
    print("window done", site, run[0][0], len(run), flush=True)
print("done", flush=True)
