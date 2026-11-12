"""Replay cached KIWA volumes through the server scan path and the tracker.

Prints identity events per scan and a SUMMARY line with area-weighted track
life and merge inversions (docs/test_reports/2026-11-11-phoenix-identity-stability.md).

Usage: python scripts/investigations/identity_stability_eval.py START END [--sig-area 20] [--out file.jsonl]
START and END are UTC, for example 20261030_2200.
"""
import argparse
import json
import math
import os
import sys
from datetime import datetime

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

from src.server import _process_scan_file  # noqa: E402
from src.tracker import StormTracker  # noqa: E402
from src.sites import haversine_distance_km  # noqa: E402

PHX = (33.4484, -112.0740)
CACHE = os.path.join(REPO_ROOT, "cache", "KIWA")


def scan_files(start: str, end: str) -> list[str]:
    names = sorted(
        n for n in os.listdir(CACHE)
        if n.startswith("KIWA") and n.endswith("_V06") and start <= n[4:17] <= end
    )
    return [os.path.join(CACHE, n) for n in names]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("start")
    ap.add_argument("end")
    ap.add_argument("--sig-area", type=float, default=20.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    store: dict[datetime, object] = {}
    tracker = StormTracker(scan_loader=lambda site, ts: store.get(ts), reacquire=True)
    out = open(args.out, "w") if args.out else None
    prev_sig: dict[int, object] = {}  # track_id -> last significant object
    prev_status: dict[int, str] = {}
    inversions = 0
    merges = 0
    life: dict[int, list] = {}
    scans_alive: dict[int, int] = {}  # track_id -> scans with an object of any size
    last_ts = None
    phx_src: dict[str, int] = {}

    for path in scan_files(args.start, args.end):
        scan = _process_scan_file("KIWA", path)
        store[scan.timestamp] = scan
        tracker.update(scan)
        events = tracker.recent_events
        obj_to_track = dict(tracker._obj_to_track)
        for tid in obj_to_track.values():
            scans_alive[tid] = scans_alive.get(tid, 0) + 1
        last_ts = scan.timestamp
        tracks = {t.track_id: t for t in tracker.all_tracks}
        for t in tracks.values():
            if t.status == "merged" and prev_status.get(t.track_id) not in (None, "merged"):
                merges += 1
                sv = tracks.get(t.merged_into)
                if sv is not None and len(sv.positions) < len(t.positions):
                    inversions += 1
        prev_status = {t.track_id: t.status for t in tracks.values()}
        tracks_last_seen = {t.track_id: t.last_seen for t in tracks.values()}
        sig = [o for o in scan.detected_objects if o.area_km2 >= args.sig_area]
        rows = []
        new_ids, reacq, motion_known = [], [], 0
        for o in sig:
            t = tracks[obj_to_track[o.object_id]]
            age = len(t.positions)
            src = t.last_motion.source if t.last_motion else "none"
            if src != "not_measured":
                motion_known += 1
            state = "new" if age == 1 else "cont"
            if t.identity_diagnostics and t.identity_diagnostics.event_context == "reacquired" and t.last_seen == scan.timestamp:
                state = "reacq"
            if age == 1 and t.split_from:
                state = "split"
            rows.append(dict(
                track=t.track_id, obj=o.object_id, state=state, age=age,
                area=round(o.area_km2, 1), peak=round(o.peak_dbz, 1),
                lat=round(o.centroid_lat, 3), lon=round(o.centroid_lon, 3),
                phx_km=round(haversine_distance_km(o.centroid_lat, o.centroid_lon, *PHX), 1),
                motion=src, heading=t.last_motion.heading_label if t.last_motion else None,
                speed=t.last_motion.speed_mph if t.last_motion else None,
                ident=t.identity_confidence,
            ))
            life.setdefault(t.track_id, []).append(o.area_km2)
            if rows[-1]["phx_km"] <= 60:
                phx_src[src] = phx_src.get(src, 0) + 1
            if state == "new":
                new_ids.append(t.track_id)
            if state == "reacq":
                reacq.append(t.track_id)
        # Significant tracks from last scan that are gone now
        vanished = []
        for tid, last in prev_sig.items():
            t = tracks.get(tid)
            if t is None or t.last_seen != scan.timestamp:
                # did a new significant object appear near where it was?
                near = [r for r in rows if r["state"] in ("new", "split") and haversine_distance_km(
                    last.centroid_lat, last.centroid_lon, r["lat"], r["lon"]) < 15]
                vanished.append(dict(track=tid, status=t.status if t else "?", area=round(last.area_km2, 1),
                                     replaced_by=[r["track"] for r in near]))
        prev_sig = {r["track"]: next(o for o in sig if o.object_id == r["obj"]) for r in rows}
        rec = dict(
            ts=scan.timestamp.isoformat(), objects=len(scan.detected_objects), sig=len(sig),
            new=new_ids, reacq=reacq, vanished=vanished, motion_known=motion_known,
            events=[e["event_type"] for e in events], rows=rows,
        )
        if out:
            out.write(json.dumps(rec) + "\n")
        swaps = sum(1 for v in vanished if v["replaced_by"])
        print(f"{scan.timestamp:%H%M%S} obj={len(scan.detected_objects):3d} sig={len(sig):2d} "
              f"new={len(new_ids)} reacq={len(reacq)} vanished={len(vanished)} swaps={swaps} "
              f"motion={motion_known}/{len(sig)} ev={','.join(sorted(set(rec['events']))) or '-'}", flush=True)
    if out:
        out.close()
    lens = [len(v) for v in life.values()]
    wsum = sum(sum(v) for v in life.values())
    wlife = sum(len(v) * sum(v) for v in life.values()) / wsum
    tot = sum(phx_src.values())
    # seen_once counts scans at >= sig-area only.  one_scan_life counts tracks
    # whose whole life (any size) is one scan, without those born on the last scan.
    one_scan = sum(
        1 for tid in life
        if scans_alive.get(tid, 0) == 1 and tracks_last_seen.get(tid) != last_ts
    )
    print(f"SUMMARY merges={merges} inversions={inversions} sig_tracks={len(lens)} "
          f"seen_once={sum(1 for x in lens if x == 1)} one_scan_life={one_scan} median_life={sorted(lens)[len(lens)//2]} "
          f"area_weighted_life={wlife:.1f} phx_pattern={phx_src.get('pattern_match',0)}/{tot} "
          f"phx_unknown={phx_src.get('not_measured',0)}/{tot}", flush=True)


if __name__ == "__main__":
    main()
