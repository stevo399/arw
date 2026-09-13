"""Investigation (motion redesign): does matching a storm's reflectivity pattern between
scans measure motion better than following its centre?

For every cached back-to-back window, runs the production tracker and, at each
scan k, for every track observed at both k-1 and k:

* measures a **pattern velocity** for each track at each scan j: the storm's
  reflectivity at scan j (inside its outline), resampled to a 0.5 km ground grid,
  is matched by normalized cross-correlation against scan j-1's field, giving the
  displacement from j-1 to j.  A storm first detected at j still has one, because
  scan j-1's field exists whether or not the storm was detected in it;
* predicts the storm at k from information up to k-1 with each candidate, and
  scores the prediction by (a) centre error in km and (b) the overlap (IoU) of
  the storm's outline at k-1, moved by the predicted displacement, with its
  outline at k, both on the ground grid.

Candidates: stationary; clean_steps_3 (median of the last three centre steps not
ending in a merge, split or reacquisition); pattern_last; pattern_median_2 and _3
(this track's last pattern velocities); neighbours_pattern (median last pattern
velocity of other tracks within 60 km).

Writes one JSON line per scored target to argv[1].  `--self-test` checks the
matcher on a synthetic shifted storm and exits.
"""
import json
import math
import os
import statistics
import sys
from datetime import datetime

sys.path.insert(0, ".")

import numpy as np
from pyart.core.transforms import geographic_to_cartesian_aeqd
from scipy.signal import correlate

GRID_M = 500.0
MAX_SPEED_KMH = 150.0
TEMPLATE_MARGIN_M = 2000.0
MIN_TEMPLATE_DBZ = 20.0
NEIGHBOUR_RADIUS_KM = 60.0
STRUCTURAL = {"merge_survivor", "split_child", "reacquired"}


class GroundSampler:
    """Nearest-gate lookup from radar-centred ground coordinates (metres)."""

    def __init__(self, azimuths, ranges_m, elevation_deg):
        from src.geometry import ground_range_m

        azimuths = np.asarray(azimuths, dtype=float) % 360.0
        self.order = np.argsort(azimuths)
        self.sorted_az = azimuths[self.order]
        self.ground = np.asarray(ground_range_m(np.asarray(ranges_m, dtype=float), float(elevation_deg)), dtype=float)
        self.gate_spacing = float(np.median(np.diff(self.ground)))

    def indices(self, x, y):
        az = np.degrees(np.arctan2(x, y)) % 360.0
        r = np.hypot(x, y)
        n = len(self.sorted_az)
        pos = np.searchsorted(self.sorted_az, az) % n
        before = (pos - 1) % n
        gap_after = np.abs((self.sorted_az[pos] - az + 180.0) % 360.0 - 180.0)
        gap_before = np.abs((self.sorted_az[before] - az + 180.0) % 360.0 - 180.0)
        sorted_index = np.where(gap_after <= gap_before, pos, before)
        rays = self.order[sorted_index]
        gates = np.rint(np.interp(r, self.ground, np.arange(len(self.ground)))).astype(int)
        valid = (r >= self.ground[0] - self.gate_spacing / 2) & (r <= self.ground[-1] + self.gate_spacing / 2)
        return rays, gates, valid

    def sample(self, field, x, y, fill):
        rays, gates, valid = self.indices(x, y)
        out = np.full(x.shape, fill, dtype=float)
        out[valid] = field[rays[valid], gates[valid]]
        return out


def grid(x0, x1, y0, y1):
    xs = np.arange(x0, x1 + GRID_M, GRID_M)
    ys = np.arange(y0, y1 + GRID_M, GRID_M)
    return np.meshgrid(xs, ys)  # arrays shaped (ny, nx); axis 0 is north


def mask_extent_m(mask, sweep):
    from src.geometry import gate_ground_xy_m

    rays, gates = np.nonzero(mask)
    x, y = gate_ground_xy_m(rays, gates, sweep.azimuths, sweep.ranges_m, sweep.elevations)
    return float(x.min()), float(x.max()), float(y.min()), float(y.max())


def _subpixel(values, index):
    if 0 < index < len(values) - 1:
        a, b, c = values[index - 1], values[index], values[index + 1]
        denom = a - 2 * b + c
        if denom != 0:
            return index + 0.5 * (a - c) / denom
    return float(index)


def pattern_displacement(template_field, template_mask, template_sampler, other_field, other_sampler, extent, max_disp_m):
    """Displacement (east, north metres) that carries the storm from the other scan to the template scan.

    The template is the storm (inside its outline) at the template scan; it is
    searched for in the other scan.  Returns (dx, dy, peak_ncc) or None.
    """
    x0, x1, y0, y1 = extent
    x0 -= TEMPLATE_MARGIN_M; x1 += TEMPLATE_MARGIN_M; y0 -= TEMPLATE_MARGIN_M; y1 += TEMPLATE_MARGIN_M
    tx, ty = grid(x0, x1, y0, y1)
    t_dbz = template_sampler.sample(template_field, tx, ty, np.nan)
    t_in = template_sampler.sample(template_mask.astype(float), tx, ty, 0.0) > 0.5
    template = np.where(t_in & np.isfinite(t_dbz), np.clip(np.nan_to_num(t_dbz) - MIN_TEMPLATE_DBZ + 5.0, 0.0, None), 0.0)
    if np.count_nonzero(template) < 8:
        return None
    pad = int(math.ceil(max_disp_m / GRID_M))
    sx, sy = grid(x0 - pad * GRID_M, x1 + pad * GRID_M, y0 - pad * GRID_M, y1 + pad * GRID_M)
    s_dbz = other_sampler.sample(other_field, sx, sy, np.nan)
    search = np.clip(np.nan_to_num(s_dbz, nan=0.0) - MIN_TEMPLATE_DBZ + 5.0, 0.0, None)
    template = template[: sy.shape[0] - 2 * pad, : sx.shape[1] - 2 * pad]
    numerator = correlate(search, template, mode="valid", method="fft")
    energy = correlate(search ** 2, np.ones_like(template), mode="valid", method="fft")
    denom = np.sqrt(np.clip(energy, 1e-9, None) * float(np.sum(template ** 2)))
    ncc = numerator / denom
    i, j = np.unravel_index(int(np.argmax(ncc)), ncc.shape)
    di = _subpixel(ncc[:, j], i) - pad
    dj = _subpixel(ncc[i, :], j) - pad
    # The storm sat at offset (di, dj) in the other scan relative to where it is in the template scan.
    return -dj * GRID_M, -di * GRID_M, float(ncc[i, j])


def advected_iou(mask_a, sampler_a, mask_b, sampler_b, extent, dx, dy):
    """IoU of outline A moved by (dx, dy) with outline B, on the ground grid."""
    x0, x1, y0, y1 = extent
    gx, gy = grid(x0, x1, y0, y1)
    moved = sampler_a.sample(mask_a.astype(float), gx - dx, gy - dy, 0.0) > 0.5
    target = sampler_b.sample(mask_b.astype(float), gx, gy, 0.0) > 0.5
    union = np.count_nonzero(moved | target)
    return float(np.count_nonzero(moved & target)) / union if union else 0.0


def self_test():
    """A synthetic storm moved 3.0 km east and 2.0 km south between scans must be recovered."""
    from types import SimpleNamespace

    azimuths = np.arange(0.25, 360.0, 0.5)
    ranges_m = np.arange(2125.0, 300000.0, 250.0)
    elevations = np.full(len(azimuths), 0.5)
    sweep = SimpleNamespace(azimuths=azimuths, ranges_m=ranges_m, elevations=elevations)
    sampler = GroundSampler(azimuths, ranges_m, 0.5)
    from src.geometry import gate_ground_xy_m

    rays, gates = np.meshgrid(np.arange(len(azimuths)), np.arange(len(ranges_m)), indexing="ij")
    x, y = gate_ground_xy_m(rays.ravel(), gates.ravel(), azimuths, ranges_m, elevations)
    x = x.reshape(rays.shape); y = y.reshape(rays.shape)

    def storm(cx, cy):
        d2 = ((x - cx) / 6000.0) ** 2 + ((y - cy) / 4000.0) ** 2
        cells = 55.0 * np.exp(-d2) + 8.0 * np.sin((x - cx) / 1700.0) * np.exp(-d2 / 4)
        return np.where(cells >= 20.0, cells, np.nan)

    earlier = storm(80000.0, 60000.0)
    later = storm(83000.0, 58000.0)
    later_mask = np.isfinite(later)
    extent = mask_extent_m(later_mask, sweep)
    result = pattern_displacement(later, later_mask, sampler, earlier, sampler, extent, 20000.0)
    print("self-test displacement (east, north, ncc):", [round(v, 3) for v in result])
    assert abs(result[0] - 3000.0) < 300 and abs(result[1] + 2000.0) < 300, result
    iou_moved = advected_iou(np.isfinite(earlier), sampler, later_mask, sampler, (40000, 120000, 30000, 90000), 3000.0, -2000.0)
    iou_still = advected_iou(np.isfinite(earlier), sampler, later_mask, sampler, (40000, 120000, 30000, 90000), 0.0, 0.0)
    print("self-test IoU moved", round(iou_moved, 3), "stationary", round(iou_still, 3))
    assert iou_moved > 0.9 > iou_still
    print("self-test passed")


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


def main(out_path):
    import src.server as server
    from src.history.store import RadarHistory
    from src.sites import haversine_distance_km
    from src.tracking.motion import KM_PER_DEGREE_LAT

    for window_index, (site, run) in enumerate(windows()):
        history = RadarHistory.open(site, None)
        previous = None  # (buffered, sampler)
        tracks = {}  # track id -> list of per-scan entries
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
                    "mask": mask, "extent": mask_extent_m(mask, sweep), "pattern": None,
                }
                if previous is not None:
                    hours = (buffered.timestamp - previous[0].timestamp).total_seconds() / 3600.0
                    measured = pattern_displacement(
                        sweep.reflectivity, mask, sampler, previous[0].reflectivity_data.reflectivity, previous[1],
                        entry["extent"], MAX_SPEED_KMH * 1000.0 * hours,
                    )
                    if measured is not None and hours > 0:
                        entry["pattern"] = (measured[0] / 1000.0 / hours, measured[1] / 1000.0 / hours, measured[2])
                seen[track.track_id] = entry

            # Score predictions of scan k (this scan) from scans up to k-1.
            if previous is not None:
                hours = (buffered.timestamp - previous[0].timestamp).total_seconds() / 3600.0
                last_entries = {tid: hist[-1] for tid, hist in tracks.items() if hist[-1]["t"] == previous[0].timestamp}
                with open(out_path, "a", encoding="utf-8") as handle:
                    for tid, target in seen.items():
                        if tid not in last_entries:
                            continue
                        history_entries = tracks[tid]
                        last = history_entries[-1]
                        candidates = {"stationary": (0.0, 0.0)}
                        steps = []
                        for a, b in zip(history_entries, history_entries[1:]):
                            h = (b["t"] - a["t"]).total_seconds() / 3600.0
                            if h > 0 and b["event"] not in STRUCTURAL:
                                steps.append(((b["x"] - a["x"]) / 1000.0 / h, (b["y"] - a["y"]) / 1000.0 / h))
                        if len(steps) >= 3:
                            recent = steps[-3:]
                            candidates["clean_steps_3"] = (statistics.median(s[0] for s in recent), statistics.median(s[1] for s in recent))
                        patterns = [e["pattern"] for e in history_entries if e["pattern"] is not None]
                        if last["pattern"] is not None:
                            candidates["pattern_last"] = last["pattern"][:2]
                        for count in (2, 3):
                            if len(patterns) >= count:
                                recent = patterns[-count:]
                                candidates[f"pattern_median_{count}"] = (statistics.median(p[0] for p in recent), statistics.median(p[1] for p in recent))
                        neighbour = [
                            e["pattern"] for other, e in last_entries.items()
                            if other != tid and e["pattern"] is not None
                            and haversine_distance_km(e["lat"], e["lon"], last["lat"], last["lon"]) <= NEIGHBOUR_RADIUS_KM
                        ]
                        if neighbour:
                            candidates["neighbours_pattern"] = (statistics.median(p[0] for p in neighbour), statistics.median(p[1] for p in neighbour))
                        x0 = min(last["extent"][0], target["extent"][0]) - 5000.0
                        x1 = max(last["extent"][1], target["extent"][1]) + 5000.0
                        y0 = min(last["extent"][2], target["extent"][2]) - 5000.0
                        y1 = max(last["extent"][3], target["extent"][3]) + 5000.0
                        record = {
                            "window": window_index, "site": site, "t": str(buffered.timestamp), "track": tid,
                            "prior_positions": len(history_entries), "hours": round(hours, 4),
                            "target_event": target["event"], "clean_target": target["event"] not in STRUCTURAL,
                            "last_pattern_ncc": None if last["pattern"] is None else round(last["pattern"][2], 3),
                            "results": {},
                        }
                        for name, (ve, vn) in candidates.items():
                            dx, dy = ve * hours * 1000.0, vn * hours * 1000.0
                            error_km = math.hypot(target["x"] - (last["x"] + dx), target["y"] - (last["y"] + dy)) / 1000.0
                            iou = advected_iou(last["mask"], previous[1], target["mask"], sampler, (x0, x1, y0, y1), dx, dy)
                            record["results"][name] = {"east_kmh": round(ve, 2), "north_kmh": round(vn, 2), "centre_error_km": round(error_km, 3), "iou": round(iou, 4)}
                        handle.write(json.dumps(record) + "\n")

            for tid, entry in seen.items():
                tracks.setdefault(tid, []).append(entry)
                if len(tracks[tid]) > 6:
                    tracks[tid][-7]["mask"] = None  # only the last position's outline is needed
            previous = (buffered, sampler)
        print("window done", site, run[0][0], len(run), flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "--self-test":
        self_test()
    else:
        main(sys.argv[1])
