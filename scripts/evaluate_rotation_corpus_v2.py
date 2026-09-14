"""Evaluate rotation detector configurations on corpus v2 (plan 2026-09-14, Tasks 6-7).

Each case's volumes are processed in time order through the production
analysis path once. Every detector configuration then runs velocity analysis
on the same inputs. Persistence is promoted from that configuration's own
per-track rotation history, keyed by one identity tracker per case.
Only cell-associated assessments are scored: unassociated candidates never
reach speech or the map.

Usage: evaluate_rotation_corpus_v2.py OUT.json [--sweep]
"""
import itertools
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import src.server as server
from src.buffer import BufferedScan
from src.detection import detect_objects_with_grid
from src.parser import extract_sweep_data, extract_velocity, parse_radar_file
from src.preprocess import preprocess_sweep
from src.tracker import StormTracker
from src.validation.nmd import parse_nmd
from src.validation.rotation_metrics import Point, VolumeResult, summarize
from src.validation.spc import StormReport
from src.velocity import (
    RotationDetectorConfig,
    _associate_rotation_candidates,
    detect_rotation_signatures,
    promote_persistent_rotation_assessments,
)

CORPUS = ROOT / "docs/validation/rotation-signal-corpus-v2.json"
NMD_MATCH_SECONDS = 180


def sweep_configs():
    grid = itertools.product(
        (15.0, 20.0, 25.0),      # min_shear_ms
        (0.0, 10.0),             # min_side_ms
        (None, 10.0),            # max_diameter_km
        (False, True),           # fold_rejection
    )
    for min_shear, min_side, max_diameter, fold in grid:
        yield RotationDetectorConfig(
            min_shear_ms=min_shear, min_side_ms=min_side, max_diameter_km=max_diameter,
            fold_rejection=fold, azimuthal_pairs_only=True, ground_overlap_merge=True,
        )


def load_volume(path):
    radar = parse_radar_file(str(ROOT / path), scans=server.LIVE_MAP_SCAN_INDICES, include_fields=server.LIVE_MAP_FIELDS)
    raw = extract_sweep_data(radar)
    vel = extract_velocity(radar, dealias=False)
    velocity = server._velocity_aligned_to_reflectivity(raw, vel)
    ref, quality, _ = preprocess_sweep(raw, [], velocity=velocity)
    detection = detect_objects_with_grid(
        reflectivity=ref.reflectivity, azimuths=ref.azimuths, ranges_m=ref.ranges_m, radar_lat=ref.radar_lat,
        radar_lon=ref.radar_lon, elevation_deg=ref.elevation_angle, elevations=ref.elevations,
        gate_classification=ref.gate_classification,
    )
    return ref, quality, vel, detection


def volume_timestamp(ref) -> datetime:
    stamp = ref.timestamp if isinstance(ref.timestamp, str) else ref.timestamp.isoformat()
    return datetime.fromisoformat(stamp.replace("Z", "+00:00")).replace(tzinfo=None)


def main(out_path: str, sweep: bool) -> None:
    corpus = json.loads(CORPUS.read_text(encoding="utf-8"))
    reports = [StormReport(**{**r, "time_utc": datetime.fromisoformat(r["time_utc"])}) for r in corpus["reports"]]
    configs = [RotationDetectorConfig.baseline()] + (list(sweep_configs()) if sweep else [])
    results: dict[int, list[VolumeResult]] = {i: [] for i in range(len(configs))}
    seconds = {i: 0.0 for i in range(len(configs))}

    for case in corpus["cases"]:
        tracker = StormTracker()
        histories = {i: {} for i in range(len(configs))}  # track id -> (timestamp, assessment)
        nmd_products = []
        for path in case["nmd"]:
            when, detections = parse_nmd((ROOT / path).read_bytes(), case["site_id"])
            if when is not None:
                nmd_products.append((when, detections))
        for path in case["volumes"]:
            try:
                ref, quality, vel, detection = load_volume(path)
            except Exception as exc:  # recorded, never silently dropped
                print("unreadable", case["id"], path, repr(exc), flush=True)
                continue
            timestamp = volume_timestamp(ref)
            tracker.update(BufferedScan(
                timestamp=timestamp, site_id=case["site_id"], reflectivity_data=ref,
                detected_objects=detection.objects, labeled_grid=detection.labeled_grid,
                object_masks=detection.object_masks, scan_quality=quality,
            ))
            nearest = min(nmd_products, key=lambda item: abs((item[0] - timestamp).total_seconds()), default=None)
            nmd = nearest[1] if nearest is not None and abs((nearest[0] - timestamp).total_seconds()) <= NMD_MATCH_SECONDS else []
            centroids = [Point(o.centroid_lat, o.centroid_lon) for o in detection.objects]
            for index, config in enumerate(configs):
                started = time.perf_counter()
                # Exactly the rotation list analyze_velocity returns, without the
                # configuration-independent velocity regions.
                assessments = _associate_rotation_candidates(
                    detect_rotation_signatures(vel, config), detection.object_masks, ref,
                    {obj.object_id: obj for obj in detection.objects},
                )
                seconds[index] += time.perf_counter() - started
                track_history = {}
                for object_id, track_id in tracker._obj_to_track.items():
                    earlier = histories[index].get(track_id)
                    if earlier is not None:
                        track_history[object_id] = [SimpleNamespace(timestamp=earlier[0], rotation=earlier[1])]
                assessments = promote_persistent_rotation_assessments(assessments, track_history, timestamp)
                for assessment in assessments:
                    track_id = tracker._obj_to_track.get(assessment.associated_object_id)
                    if track_id is not None:
                        histories[index][track_id] = (timestamp, assessment)
                results[index].append(VolumeResult(
                    case_id=case["id"], label=case["label"], volume_time=timestamp,
                    assessments=[(Point(a.centroid_lat, a.centroid_lon), a.evidence_level)
                                 for a in assessments if a.associated_object_id is not None],
                    storm_centroids=centroids, nmd=nmd,
                ))
            print("done", case["id"], Path(path).name, flush=True)

    output = []
    for index, config in enumerate(configs):
        output.append({
            "config": asdict(config),
            "analysis_seconds": round(seconds[index], 1),
            "summary": summarize(results[index], corpus["cases"], reports),
            "per_case": {
                case["id"]: summarize([r for r in results[index] if r.case_id == case["id"]], [case], reports)[1]
                for case in corpus["cases"]
            },
        })
    Path(out_path).write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")
    print("wrote", out_path, flush=True)


if __name__ == "__main__":
    main(sys.argv[1], "--sweep" in sys.argv)
