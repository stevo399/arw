"""Proof (2026-09-14): live rotation evidence is spoken only where the corpus supports it.

The full corpus sweep takes about an hour, so its recorded result
(docs/test_reports/2026-09-14-rotation-sweep.json) is checked against the
code's constants instead:
- The detector's defaults are the configuration the pre-registered rule selects.
- Every spoken evidence level is among the speakable ranks.  Passing the corpus
  is necessary, not sufficient: "persistent" passed it but was withdrawn after
  the live check (docs/test_reports/2026-09-14-rotation-corpus.md), so the
  spoken set is currently empty.

The production live path (`_process_scan_file` then `RadarHistory.add_live_scan`,
which promotes persistence) is then run on real corpus volumes:
- Clear air (KIWA, six consecutive volumes): nothing at a spoken level, and no
  rotation in any summary or map description.
- A tornado (Spring Grove VA, KAKQ, 2026-09-12 20:46Z): persistent evidence
  within 10 km of the report, 20 minutes before to 10 minutes after it, as in
  the corpus evaluation.
"""

import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import src.server as server
from src.history.store import RadarHistory
from src.map_layer import build_storm_geojson
from src.sites import haversine_distance_km
from src.summary import SPOKEN_ROTATION_EVIDENCE, generate_summary
from src.validation.rotation_metrics import EVIDENCE_RANK, HIT_AFTER, HIT_BEFORE, HIT_RADIUS_KM
from src.velocity import RotationDetectorConfig

ROOT = Path(__file__).resolve().parents[2]
SWEEP = ROOT / "docs/test_reports/2026-09-14-rotation-sweep.json"
CORPUS = ROOT / "docs/validation/rotation-signal-corpus-v2.json"


def _case(case_id):
    corpus = json.loads(CORPUS.read_text(encoding="utf-8"))
    return next(case for case in corpus["cases"] if case["id"] == case_id)


def _run_live(case, tmp_path):
    history = RadarHistory.open(case["site_id"], tmp_path)
    scans = []
    for path in case["volumes"]:
        buffered = server._process_scan_file(case["site_id"], str((ROOT / path).resolve()))
        assert history.add_live_scan(buffered) is not None
        scans.append(buffered)
    return scans


def test_recorded_selection_matches_the_detector_defaults_and_bounds_the_spoken_levels():
    sys.path.insert(0, str(ROOT / "scripts"))
    from select_rotation_config import main as select

    chosen = select(str(SWEEP))
    assert chosen is not None
    assert chosen["config"] == asdict(RotationDetectorConfig())

    speakable_ranks = [int(rank) for rank, summary in chosen["summary"].items() if summary["speakable"]]
    corpus_speakable = frozenset(
        level for level, rank in EVIDENCE_RANK.items() if speakable_ranks and rank >= min(speakable_ranks)
    )
    assert SPOKEN_ROTATION_EVIDENCE <= corpus_speakable


def test_clear_air_live_scans_speak_no_rotation(tmp_path):
    case = _case("clear-air-KIWA-20260712")
    for buffered in _run_live(case, tmp_path):
        spoken = [r for r in buffered.rotation_signatures if r.evidence_level in SPOKEN_ROTATION_EVIDENCE]
        assert spoken == [], (buffered.timestamp, spoken)
        tracking = buffered.tracking
        summary = generate_summary(  # as GET /summary calls it
            site_id=case["site_id"], site_name="Tucson", timestamp=buffered.reflectivity_data.timestamp,
            objects=buffered.detected_objects, tracks=tracking.active_tracks, events=tracking.recent_events,
        ).lower()
        assert "rotation" not in summary and "couplet" not in summary, summary
        for feature in build_storm_geojson(buffered)["features"]:
            description = feature["properties"]["description"].lower()
            assert "rotation" not in description and "couplet" not in description


def test_live_path_finds_persistent_rotation_at_the_spring_grove_tornado(tmp_path):
    case = _case("tornado-20260912-2046-8")
    report = case["report"]
    reported_at = datetime.fromisoformat(report["time_utc"])
    hits = []
    for buffered in _run_live(case, tmp_path):
        scanned_at = buffered.timestamp.replace(tzinfo=None)  # both UTC; the report time is naive
        if not (reported_at - HIT_BEFORE <= scanned_at <= reported_at + HIT_AFTER):
            continue
        for rotation in buffered.rotation_signatures:
            if rotation.associated_object_id is None or rotation.evidence_level != "persistent":
                continue
            distance = haversine_distance_km(
                rotation.centroid_lat, rotation.centroid_lon, report["latitude"], report["longitude"],
            )
            if distance <= HIT_RADIUS_KM:
                storm = next(o for o in buffered.detected_objects if o.object_id == rotation.associated_object_id)
                hits.append((buffered.timestamp, round(distance, 1), storm.rotation))
    assert hits, "no persistent rotation within 10 km of the Spring Grove tornado report"
    # The storm the hit belongs to carries the persistent assessment in its data.
    assert all(rotation is not None and rotation.evidence_level == "persistent" for _, _, rotation in hits), hits
