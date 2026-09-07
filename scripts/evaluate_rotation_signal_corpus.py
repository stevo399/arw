"""Measure rotation-signal evidence on the documented local corpus.

This is an evaluator, not a trainer and not a threshold tuner. It reads only
the cached volumes listed in docs/validation/rotation-signal-corpus.json and
writes one JSON report to stdout for archival or comparison.
"""

from __future__ import annotations

import json
import sys
from contextlib import redirect_stdout
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
with redirect_stdout(sys.stderr):
    import pyart

from src.detection import detect_objects_with_grid
from src.buffer import BufferedScan
from src.geometry import align_field_by_azimuth
from src.parser import extract_sweep_data, extract_velocity
from src.preprocess import preprocess_sweep, refresh_quality_advisory
from src.tracker import StormTracker
from src.velocity import analyze_velocity, promote_persistent_rotation_assessments


CORPUS_PATH = ROOT / "docs/validation/rotation-signal-corpus.json"
EARTH_RADIUS_KM = 6371.0


def _aligned_velocity(sweep, vel_data):
    if vel_data is None or not vel_data.sweeps:
        return None
    velocity_sweep = vel_data.sweeps[0]
    if not np.array_equal(sweep.ranges_m, velocity_sweep.ranges_m):
        return None
    return align_field_by_azimuth(
        velocity_sweep.velocity, velocity_sweep.azimuths, sweep.azimuths,
    )


def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2.0) ** 2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2.0) ** 2
    return float(2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0))))


def evaluate_volume(path: str, event: dict | None = None, tracker: StormTracker | None = None) -> dict:
    radar = pyart.io.read_nexrad_archive(path)
    raw_sweep = extract_sweep_data(radar)
    vel_data = extract_velocity(radar)
    velocity = _aligned_velocity(raw_sweep, vel_data)
    ref_data, quality, _ = preprocess_sweep(raw_sweep, [], velocity=velocity)
    result = detect_objects_with_grid(
        reflectivity=ref_data.reflectivity,
        azimuths=ref_data.azimuths,
        ranges_m=ref_data.ranges_m,
        radar_lat=ref_data.radar_lat,
        radar_lon=ref_data.radar_lon,
        elevation_deg=ref_data.elevation_angle,
        elevations=ref_data.elevations,
        gate_classification=ref_data.gate_classification,
    )
    _regions, assessments, _objects = analyze_velocity(
        vel_data, result.objects, result.object_masks, ref_data,
    )
    if tracker is not None:
        timestamp = datetime.fromisoformat(ref_data.timestamp)
        scan = BufferedScan(
            timestamp=timestamp,
            site_id=path.split("/")[1],
            reflectivity_data=ref_data,
            detected_objects=_objects,
            labeled_grid=result.labeled_grid,
            object_masks=result.object_masks,
            scan_quality=quality,
            velocity_data=vel_data,
            rotation_signatures=assessments,
        )
        tracker.update(scan)
        histories = {
            obj.object_id: tracker.rotation_history_for_current_object(obj.object_id)
            for obj in _objects
        }
        assessments = promote_persistent_rotation_assessments(
            assessments, histories, timestamp,
        )
    _refreshed, refreshed_quality, _advisory = refresh_quality_advisory(
        ref_data, quality, assessments, velocity=velocity,
    )
    evidence_counts = Counter(item.evidence_level for item in assessments)
    row = {
        "volume": path,
        "timestamp": str(ref_data.timestamp),
        "object_count": len(result.objects),
        "rotation_candidate_count": len(assessments),
        "rotation_evidence_counts": dict(sorted(evidence_counts.items())),
        "quality_advisory_fraction": refreshed_quality.advisory_fraction,
        "dual_pol_available": bool(ref_data.rhohv is not None and ref_data.zdr is not None),
    }
    if event is not None:
        distances = [
            _haversine_km(obj.centroid_lat, obj.centroid_lon, event["latitude"], event["longitude"])
            for obj in result.objects
        ]
        row["nearest_object_to_event_km"] = min(distances) if distances else None
        row["event_search_radius_km"] = event["search_radius_km"]
    return row


def evaluate_corpus(corpus: dict) -> dict:
    cases = []
    for case in corpus["cases"]:
        volumes = case["volumes"] if "volumes" in case else [case["volume"]]
        tracker = StormTracker() if case.get("evaluation_mode") == "tracked_sequence" else None
        rows = [evaluate_volume(path, case.get("event"), tracker) for path in volumes]
        cases.append({"id": case["id"], "label": case["label"], "measurements": rows})
    return {"corpus_schema_version": corpus["schema_version"], "cases": cases}


if __name__ == "__main__":
    corpus = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    print(json.dumps(evaluate_corpus(corpus), indent=2, sort_keys=True))
