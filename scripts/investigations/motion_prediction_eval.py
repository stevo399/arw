"""Investigation (motion redesign): which motion estimate predicts where a storm will be?

Runs every cached back-to-back window through the production tracker
(RadarHistory, reacquisition on) and records every active track's position
and identity event at each scan.  Then, for each observed position of each
track, predicts it from the track's earlier positions with each candidate
estimator and records the error in km.  Candidates:

  stationary            last position, no motion
  lifetime              least-squares fit over all earlier positions
  last_N                fit over the last N earlier positions (N = 2..5)
  last_M_min            fit over earlier positions within M minutes of the last
  clean_steps_K         median velocity of the last K earlier steps that do not
                        end in a structural event (merge, split, reacquired)
  neighbours            median last-3 velocity of other tracks within 60 km that
                        have at least 3 positions at that scan
  scene_field           the tracker's own blended field for the track (as reported today)

Targets are split by how many earlier positions the track had, and by whether
the step into the target was itself structural (its centroid then reflects a
reshaped object, not motion).  Writes a JSON summary to argv[1].
"""
import json
import math
import os
import statistics
import sys
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, ".")

import numpy as np

import src.server as server
import src.tracker as tracker_module
from src.history.store import RadarHistory
from src.sites import haversine_distance_km
from src.tracking.motion import KM_PER_DEGREE_LAT

OUT = sys.argv[1]
STRUCTURAL = {"merge_survivor", "split_child", "reacquired"}
NEIGHBOUR_RADIUS_KM = 60.0


def windows():
    found = []
    for site in sorted(os.listdir("cache")):
        folder = os.path.join("cache", site)
        if not os.path.isdir(folder):
            continue
        stamps = sorted(
            (datetime.strptime(name[4:19], "%Y%m%d_%H%M%S"), os.path.join(folder, name))
            for name in os.listdir(folder)
            if name.endswith("_V06") and len(name) >= 23
        )
        run = []
        for stamp, path in stamps:
            if run and (stamp - run[-1][0]).total_seconds() > 20 * 60:
                if len(run) >= 3:
                    found.append((site, run))
                run = []
            run.append((stamp, path))
        if len(run) >= 3:
            found.append((site, run))
    return found


def to_km(lat, lon, lat0, lon0):
    return (lat - lat0) * KM_PER_DEGREE_LAT, (lon - lon0) * KM_PER_DEGREE_LAT * math.cos(math.radians(lat0))


def fit_velocity(points):
    """Least-squares velocity (north, east km/h) over (t_seconds, north_km, east_km) points."""
    if len(points) < 2:
        return None
    t = np.array([p[0] for p in points])
    if t[-1] - t[0] <= 0:
        return None
    n = np.polyfit(t, [p[1] for p in points], 1)[0] * 3600.0
    e = np.polyfit(t, [p[2] for p in points], 1)[0] * 3600.0
    return n, e


captured = {}
real_associate = tracker_module.associate_tracks


def capturing(*args, **kwargs):
    result = real_associate(*args, **kwargs)
    captured["result"] = result
    return result


tracker_module.associate_tracks = capturing

# track key -> list of dicts (t, lat, lon, event, field_velocity)
histories = defaultdict(list)
# (window, scan time) -> track keys observed
by_scan = defaultdict(list)

for window_index, (site, run) in enumerate(windows()):
    history = RadarHistory.open(site, None)
    for stamp, path in run:
        try:
            buffered = server._process_scan_file(site, os.path.abspath(path))
        except Exception as exc:
            print("skip", path, exc, flush=True)
            continue
        captured.pop("result", None)
        if history.add_live_scan(buffered) is None:
            continue
        result = captured.get("result")
        for track in history.tracker.active_tracks:
            if track.last_seen != buffered.timestamp or track.current_object is None:
                continue
            obj = track.current_object
            field_velocity = None
            if result is not None and track.track_id in result.track_geo_motion and result.dt_hours > 0:
                g = result.track_geo_motion[track.track_id]
                field_velocity = (
                    g.delta_lat * KM_PER_DEGREE_LAT / result.dt_hours,
                    g.delta_lon * KM_PER_DEGREE_LAT * math.cos(math.radians(obj.centroid_lat)) / result.dt_hours,
                )
            event = track.identity_diagnostics.event_context if track.identity_diagnostics else None
            key = (window_index, track.track_id)
            histories[key].append({"t": buffered.timestamp, "lat": obj.centroid_lat, "lon": obj.centroid_lon, "event": event, "field": field_velocity})
            by_scan[(window_index, buffered.timestamp)].append(key)
    print("window done", site, run[0][0], len(run), flush=True)

errors = defaultdict(list)  # (group, candidate) -> errors


def recent_velocity(entries, count):
    base = entries[-1]
    points = [((e["t"] - base["t"]).total_seconds(), *to_km(e["lat"], e["lon"], base["lat"], base["lon"])) for e in entries[-count:]]
    return fit_velocity(points)


for key, entries in histories.items():
    for k in range(1, len(entries)):
        earlier, target = entries[:k], entries[k]
        last = earlier[-1]
        hours = (target["t"] - last["t"]).total_seconds() / 3600.0
        if hours <= 0:
            continue
        prior = "1" if k == 1 else "2" if k == 2 else "3" if k == 3 else "4+"
        clean_target = target["event"] not in STRUCTURAL
        groups = [f"prior_{prior}", f"prior_{prior}_{'clean' if clean_target else 'structural'}_target", "all"]

        def record(name, velocity):
            if velocity is None:
                return
            north = velocity[0] * hours
            east = velocity[1] * hours
            actual_n, actual_e = to_km(target["lat"], target["lon"], last["lat"], last["lon"])
            err = math.hypot(actual_n - north, actual_e - east)
            for group in groups:
                errors[(group, name)].append(err)

        record("stationary", (0.0, 0.0))
        if k >= 2:
            record("lifetime", recent_velocity(earlier, len(earlier)))
            for n in (2, 3, 4, 5):
                if k >= n:
                    record(f"last_{n}", recent_velocity(earlier, n))
            for minutes in (10, 15, 20, 30):
                window = [e for e in earlier if (last["t"] - e["t"]).total_seconds() <= minutes * 60]
                if len(window) >= 2:
                    record(f"last_{minutes}_min", recent_velocity(window, len(window)))
            steps = []
            for a, b in zip(earlier, earlier[1:]):
                h = (b["t"] - a["t"]).total_seconds() / 3600.0
                if h > 0 and b["event"] not in STRUCTURAL:
                    dn, de = to_km(b["lat"], b["lon"], a["lat"], a["lon"])
                    steps.append((dn / h, de / h))
            for count in (1, 2, 3):
                if len(steps) >= count:
                    recent = steps[-count:]
                    record(f"clean_steps_{count}", (statistics.median(s[0] for s in recent), statistics.median(s[1] for s in recent)))
        record("scene_field", last["field"])
        neighbour_velocities = []
        for other in by_scan[(key[0], last["t"])]:
            if other == key:
                continue
            other_entries = [e for e in histories[other] if e["t"] <= last["t"]]
            if len(other_entries) < 3:
                continue
            o = other_entries[-1]
            if haversine_distance_km(o["lat"], o["lon"], last["lat"], last["lon"]) > NEIGHBOUR_RADIUS_KM:
                continue
            v = recent_velocity(other_entries, 3)
            if v is not None:
                neighbour_velocities.append(v)
        if neighbour_velocities:
            record("neighbours", (statistics.median(v[0] for v in neighbour_velocities), statistics.median(v[1] for v in neighbour_velocities)))

summary = defaultdict(dict)
for (group, name), values in sorted(errors.items()):
    summary[group][name] = {
        "n": len(values),
        "median_km": round(statistics.median(values), 2),
        "mean_km": round(statistics.fmean(values), 2),
        "p90_km": round(float(np.percentile(values, 90)), 2),
    }
with open(OUT, "w", encoding="utf-8") as handle:
    json.dump(summary, handle, indent=2)
print(json.dumps(summary, indent=2))
