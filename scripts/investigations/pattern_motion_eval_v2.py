"""Investigation (motion redesign), second pass: guarded pattern matching.

The first pass (pattern_motion_eval.py) showed pattern matching beats following
the centre when its match is plausible, but in dense scenes it often locked onto
a different echo near the edge of its search window (median implied speed of bad
matches: 129 km/h).  This pass measures three guards, alone and together:

* edge rejection -- a correlation peak on the border of the search window is
  not a match;
* context template -- the template holds every echo within 10 km of the storm,
  not only the storm itself, so a lone blob cannot match any similar blob;
* neighbour consistency -- a storm's pattern velocity is accepted only when it is
  within 25 km/h of the median of other storms' velocities within 60 km (when at
  least three such neighbours exist), the vector-median check used for
  tracking-radar-echoes-by-correlation methods.

Candidates scored (centre error km and outline IoU, predicting scan k from
information up to k-1): stationary, clean_steps_3, mask_last (first-pass
method), mask_edge_last, ctx_edge_last, ctx_guarded_last, ctx_guarded_median_3,
neighbours_guarded (median guarded velocity of neighbours), young_storm_rule
(own guarded velocity, else neighbours_guarded), and full_rule (clean_steps_3,
else own guarded velocity, else neighbours_guarded).

Writes one JSON line per target to argv[1].
"""
import json
import math
import os
import statistics
import sys

sys.path.insert(0, ".")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from pyart.core.transforms import geographic_to_cartesian_aeqd
from scipy.signal import correlate

from pattern_motion_eval import (
    GRID_M,
    MAX_SPEED_KMH,
    MIN_TEMPLATE_DBZ,
    NEIGHBOUR_RADIUS_KM,
    STRUCTURAL,
    GroundSampler,
    _subpixel,
    advected_iou,
    grid,
    mask_extent_m,
    windows,
)

CONTEXT_MARGIN_M = 10000.0
MASK_MARGIN_M = 2000.0
CONSISTENCY_KMH = 25.0
MIN_NEIGHBOURS = 3


def match(template_field, template_mask, template_sampler, other_field, other_sampler, extent, max_disp_m, context):
    """(east m, north m, ncc, at_edge) carrying the storm from the other scan to the template scan, or None."""
    margin = CONTEXT_MARGIN_M if context else MASK_MARGIN_M
    x0, x1, y0, y1 = extent[0] - margin, extent[1] + margin, extent[2] - margin, extent[3] + margin
    tx, ty = grid(x0, x1, y0, y1)
    t_dbz = np.nan_to_num(template_sampler.sample(template_field, tx, ty, np.nan), nan=0.0)
    values = np.clip(t_dbz - MIN_TEMPLATE_DBZ + 5.0, 0.0, None)
    if context:
        template = values
    else:
        inside = template_sampler.sample(template_mask.astype(float), tx, ty, 0.0) > 0.5
        template = np.where(inside, values, 0.0)
    if np.count_nonzero(template) < 8:
        return None
    pad = int(math.ceil(max_disp_m / GRID_M))
    sx, sy = grid(x0 - pad * GRID_M, x1 + pad * GRID_M, y0 - pad * GRID_M, y1 + pad * GRID_M)
    search = np.clip(np.nan_to_num(other_sampler.sample(other_field, sx, sy, np.nan), nan=0.0) - MIN_TEMPLATE_DBZ + 5.0, 0.0, None)
    template = template[: sy.shape[0] - 2 * pad, : sx.shape[1] - 2 * pad]
    template = template - (template.mean() if context else 0.0)
    numerator = correlate(search, template, mode="valid", method="fft")
    if context:
        # Zero-mean template: normalise by the search window's local variance.
        ones = np.ones_like(template)
        count = float(template.size)
        local_sum = correlate(search, ones, mode="valid", method="fft")
        local_sq = correlate(search ** 2, ones, mode="valid", method="fft")
        variance = np.clip(local_sq - local_sum ** 2 / count, 1e-9, None)
        denom = np.sqrt(variance * float(np.sum(template ** 2)))
    else:
        energy = correlate(search ** 2, np.ones_like(template), mode="valid", method="fft")
        denom = np.sqrt(np.clip(energy, 1e-9, None) * float(np.sum(template ** 2)))
    ncc = numerator / denom
    i, j = np.unravel_index(int(np.argmax(ncc)), ncc.shape)
    at_edge = i in (0, ncc.shape[0] - 1) or j in (0, ncc.shape[1] - 1)
    di = _subpixel(ncc[:, j], i) - pad
    dj = _subpixel(ncc[i, :], j) - pad
    return -dj * GRID_M, -di * GRID_M, float(ncc[i, j]), bool(at_edge)


def median_vector(vectors):
    return statistics.median(v[0] for v in vectors), statistics.median(v[1] for v in vectors)


def main(out_path):
    import src.server as server
    from src.history.store import RadarHistory
    from src.sites import haversine_distance_km

    for window_index, (site, run) in enumerate(windows()):
        history = RadarHistory.open(site, None)
        previous = None
        tracks = {}
        for stamp, path in run:
            try:
                buffered = server._process_scan_file(site, os.path.abspath(path))
            except Exception as exc:
                print("skip", path, exc, flush=True)
                continue
            if history.add_live_scan(buffered) is None:
                continue
            sweep = buffered.reflectivity_data
            sampler = GroundSampler(sweep.azimuths, sweep.ranges_m, sweep.elevation_angle)
            seen = {}
            hours_since = (buffered.timestamp - previous[0].timestamp).total_seconds() / 3600.0 if previous else 0.0
            for track in history.tracker.active_tracks:
                if track.last_seen != buffered.timestamp or track.current_object is None:
                    continue
                obj = track.current_object
                mask = np.asarray(buffered.object_masks[obj.object_id])
                x, y = geographic_to_cartesian_aeqd(np.asarray([obj.centroid_lon]), np.asarray([obj.centroid_lat]), sweep.radar_lon, sweep.radar_lat)
                entry = {
                    "t": buffered.timestamp, "x": float(np.ravel(x)[0]), "y": float(np.ravel(y)[0]),
                    "lat": obj.centroid_lat, "lon": obj.centroid_lon,
                    "event": track.identity_diagnostics.event_context if track.identity_diagnostics else None,
                    "mask": mask, "extent": mask_extent_m(mask, sweep),
                    "mask_v": None, "mask_edge": None, "ctx_v": None, "ctx_edge": None, "ctx_ncc": None, "guarded": None,
                }
                if previous is not None and hours_since > 0:
                    max_disp = MAX_SPEED_KMH * 1000.0 * hours_since
                    for key, context in (("mask", False), ("ctx", True)):
                        found = match(sweep.reflectivity, mask, sampler, previous[0].reflectivity_data.reflectivity, previous[1], entry["extent"], max_disp, context)
                        if found is not None:
                            entry[f"{key}_v"] = (found[0] / 1000.0 / hours_since, found[1] / 1000.0 / hours_since)
                            entry[f"{key}_edge"] = found[3]
                            if context:
                                entry["ctx_ncc"] = found[2]
                seen[track.track_id] = entry

            # Neighbour consistency on this scan's context matches.
            candidates_now = {tid: e["ctx_v"] for tid, e in seen.items() if e["ctx_v"] is not None and not e["ctx_edge"]}
            for tid, entry in seen.items():
                v = candidates_now.get(tid)
                if v is None:
                    continue
                neighbours = [
                    ov for other, ov in candidates_now.items()
                    if other != tid and haversine_distance_km(seen[other]["lat"], seen[other]["lon"], entry["lat"], entry["lon"]) <= NEIGHBOUR_RADIUS_KM
                ]
                if len(neighbours) >= MIN_NEIGHBOURS:
                    m = median_vector(neighbours)
                    if math.hypot(v[0] - m[0], v[1] - m[1]) > CONSISTENCY_KMH:
                        continue
                entry["guarded"] = v

            if previous is not None:
                hours = hours_since
                last_entries = {tid: hist[-1] for tid, hist in tracks.items() if hist[-1]["t"] == previous[0].timestamp}
                with open(out_path, "a", encoding="utf-8") as handle:
                    for tid, target in seen.items():
                        if tid not in last_entries:
                            continue
                        entries = tracks[tid]
                        last = entries[-1]
                        c = {"stationary": (0.0, 0.0)}
                        steps = []
                        for a, b in zip(entries, entries[1:]):
                            h = (b["t"] - a["t"]).total_seconds() / 3600.0
                            if h > 0 and b["event"] not in STRUCTURAL:
                                steps.append(((b["x"] - a["x"]) / 1000.0 / h, (b["y"] - a["y"]) / 1000.0 / h))
                        if len(steps) >= 3:
                            c["clean_steps_3"] = median_vector(steps[-3:])
                        if last["mask_v"] is not None:
                            c["mask_last"] = last["mask_v"]
                            if not last["mask_edge"]:
                                c["mask_edge_last"] = last["mask_v"]
                        if last["ctx_v"] is not None and not last["ctx_edge"]:
                            c["ctx_edge_last"] = last["ctx_v"]
                        if last["guarded"] is not None:
                            c["ctx_guarded_last"] = last["guarded"]
                        guarded_history = [e["guarded"] for e in entries if e["guarded"] is not None]
                        if len(guarded_history) >= 3:
                            c["ctx_guarded_median_3"] = median_vector(guarded_history[-3:])
                        neighbours = [
                            e["guarded"] for other, e in last_entries.items()
                            if other != tid and e["guarded"] is not None
                            and haversine_distance_km(e["lat"], e["lon"], last["lat"], last["lon"]) <= NEIGHBOUR_RADIUS_KM
                        ]
                        if neighbours:
                            c["neighbours_guarded"] = median_vector(neighbours)
                        young = c.get("ctx_guarded_last", c.get("neighbours_guarded"))
                        if young is not None:
                            c["young_storm_rule"] = young
                        full = c.get("clean_steps_3", young)
                        if full is not None:
                            c["full_rule"] = full
                        x0 = min(last["extent"][0], target["extent"][0]) - 5000.0
                        x1 = max(last["extent"][1], target["extent"][1]) + 5000.0
                        y0 = min(last["extent"][2], target["extent"][2]) - 5000.0
                        y1 = max(last["extent"][3], target["extent"][3]) + 5000.0
                        record = {
                            "window": window_index, "site": site, "t": str(buffered.timestamp), "track": tid,
                            "prior_positions": len(entries), "hours": round(hours, 4),
                            "target_event": target["event"], "clean_target": target["event"] not in STRUCTURAL,
                            "last_pattern_ncc": None if last["ctx_ncc"] is None else round(last["ctx_ncc"], 3),
                            "results": {},
                        }
                        for name, (ve, vn) in c.items():
                            dx, dy = ve * hours * 1000.0, vn * hours * 1000.0
                            error_km = math.hypot(target["x"] - (last["x"] + dx), target["y"] - (last["y"] + dy)) / 1000.0
                            iou = advected_iou(last["mask"], previous[1], target["mask"], sampler, (x0, x1, y0, y1), dx, dy)
                            record["results"][name] = {"east_kmh": round(ve, 2), "north_kmh": round(vn, 2), "centre_error_km": round(error_km, 3), "iou": round(iou, 4)}
                        handle.write(json.dumps(record) + "\n")

            for tid, entry in seen.items():
                tracks.setdefault(tid, []).append(entry)
                if len(tracks[tid]) > 6:
                    tracks[tid][-7]["mask"] = None
            previous = (buffered, sampler)
        print("window done", site, run[0][0], len(run), flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
