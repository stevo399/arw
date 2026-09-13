"""Proof (2026-09-13): reported storm motion predicts where storms go.

Runs a cached KTLX window through the production history and tracker.  For
every storm seen in two consecutive scans whose motion was reported in the
first, the reported motion predicts where the storm is in the second.  On the
same storms, that prediction must beat assuming no motion, both in centre
error and in how well the moved outline overlaps the next outline.

Also: no storm is ever reported "stationary", and a storm with unknown motion
has no speed.  Results are written to
docs/test_reports/2026-09-13-measured-motion-proof.json.
"""

import json
import math
import statistics
from pathlib import Path

import numpy as np

import src.server as server
from src.geometry import gate_ground_xy_m
from src.history.store import RadarHistory
from src.sites import haversine_distance_km
from src.tracking.motion import KM_PER_DEGREE_LAT

WINDOW = [
    f"cache/KTLX/KTLX20260410_{stamp}_V06"
    for stamp in ("224209", "224737", "225255", "225805", "230332", "230850", "231407")
]
REPORT = Path("docs/test_reports/2026-09-13-measured-motion-proof.json")


def _outline_iou(mask_a, sweep_a, mask_b, sweep_b, east_m, north_m, grid_m=500.0):
    """IoU on a ground grid of outline A moved by (east, north) with outline B."""
    from src.tracking.pattern_motion import GroundSampler

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
    return np.count_nonzero(moved & target) / union if union else 0.0


def test_reported_motion_predicts_storm_positions_better_than_no_motion(tmp_path):
    history = RadarHistory.open("KTLX", tmp_path)
    previous = None  # (buffered, {track_id: (object, mask, motion)})
    rows = []
    sources = {}
    for path in WINDOW:
        buffered = server._process_scan_file("KTLX", str(Path(path).resolve()))
        history.add_live_scan(buffered)
        seen = {}
        for track in history.tracker.active_tracks:
            if track.last_seen != buffered.timestamp:
                continue
            motion = track.get_motion()
            assert motion.heading_label != "stationary"
            if motion.heading_label == "unknown":
                assert motion.speed_kmh is None and motion.speed_mph is None
            sources[motion.source] = sources.get(motion.source, 0) + 1
            obj = track.current_object
            seen[track.track_id] = (obj, np.asarray(buffered.object_masks[obj.object_id]), motion)
        if previous is not None:
            hours = (buffered.timestamp - previous[0].timestamp).total_seconds() / 3600.0
            for track_id, (obj, mask, _) in seen.items():
                if track_id not in previous[1]:
                    continue
                last_obj, last_mask, last_motion = previous[1][track_id]
                if last_motion.east_kmh is None:
                    continue
                east_km, north_km = last_motion.east_kmh * hours, last_motion.north_kmh * hours
                predicted = (
                    last_obj.centroid_lat + north_km / KM_PER_DEGREE_LAT,
                    last_obj.centroid_lon + east_km / (KM_PER_DEGREE_LAT * math.cos(math.radians(last_obj.centroid_lat))),
                )
                rows.append({
                    "track": track_id,
                    "source": last_motion.source,
                    "reported_error_km": haversine_distance_km(*predicted, obj.centroid_lat, obj.centroid_lon),
                    "stationary_error_km": haversine_distance_km(last_obj.centroid_lat, last_obj.centroid_lon, obj.centroid_lat, obj.centroid_lon),
                    "reported_iou": _outline_iou(last_mask, previous[0].reflectivity_data, mask, buffered.reflectivity_data, east_km * 1000.0, north_km * 1000.0),
                    "stationary_iou": _outline_iou(last_mask, previous[0].reflectivity_data, mask, buffered.reflectivity_data, 0.0, 0.0),
                })
        previous = (buffered, seen)

    def summary(subset):
        return {
            "n": len(subset),
            "median_error_km": {k: round(statistics.median(r[f"{k}_error_km"] for r in subset), 3) for k in ("reported", "stationary")},
            "p90_error_km": {k: round(float(np.percentile([r[f"{k}_error_km"] for r in subset], 90)), 3) for k in ("reported", "stationary")},
            "mean_iou": {k: round(statistics.fmean(r[f"{k}_iou"] for r in subset), 4) for k in ("reported", "stationary")},
        }

    result = {
        "window": [Path(p).name for p in WINDOW],
        "reported_motion_sources": sources,
        "all": summary(rows),
        "by_source": {source: summary([r for r in rows if r["source"] == source]) for source in sorted({r["source"] for r in rows})},
    }
    REPORT.write_text(json.dumps(result, indent=2))

    overall = result["all"]
    assert overall["n"] >= 100
    assert overall["median_error_km"]["reported"] < overall["median_error_km"]["stationary"]
    assert overall["mean_iou"]["reported"] > overall["mean_iou"]["stationary"]
    measured = result["by_source"]["pattern_match"]
    assert measured["median_error_km"]["reported"] < measured["median_error_km"]["stationary"]
    assert measured["mean_iou"]["reported"] > measured["mean_iou"]["stationary"]
