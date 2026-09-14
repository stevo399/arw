"""Pre-registered rotation validation metrics.

Defined in docs/superpowers/plans/2026-09-14-rotation-detection-truth.md
(Global Constraints) before any measurement, so no threshold here is chosen
from the results it scores.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta

from src.sites import haversine_distance_km
from src.validation.nmd import MesocycloneDetection
from src.validation.spc import StormReport

EVIDENCE_RANK = {"unconfirmed": 1, "vertically_confirmed": 2, "corroborated": 2, "persistent": 3}
HIT_RADIUS_KM = 10.0
HIT_BEFORE = timedelta(minutes=20)
HIT_AFTER = timedelta(minutes=10)
NMD_MATCH_KM = 5.0
NULL_EXCLUSION_KM = 50.0
NULL_REPORT_WINDOW = timedelta(minutes=60)
SPEAK_MIN_HITS = 2
SPEAK_MIN_LIFT = 3.0


@dataclass(frozen=True)
class Point:
    latitude: float
    longitude: float


@dataclass
class VolumeResult:
    case_id: str
    label: str
    volume_time: datetime
    assessments: list[tuple[Point, str]]
    storm_centroids: list[Point]
    nmd: list[MesocycloneDetection]


def _distance(point: Point, latitude: float, longitude: float) -> float:
    return haversine_distance_km(point.latitude, point.longitude, latitude, longitude)


def _at_rank(result: VolumeResult, min_rank: int) -> list[Point]:
    return [point for point, level in result.assessments if EVIDENCE_RANK.get(level, 0) >= min_rank]


def event_hit(results: list[VolumeResult], report: StormReport, min_rank: int) -> bool:
    """Any assessment at `min_rank` within 10 km of the report, 20 min before to 10 min after."""
    for result in results:
        if not (report.time_utc - HIT_BEFORE <= result.volume_time <= report.time_utc + HIT_AFTER):
            continue
        if any(_distance(p, report.latitude, report.longitude) <= HIT_RADIUS_KM for p in _at_rank(result, min_rank)):
            return True
    return False


def nmd_agreement(result: VolumeResult, min_rank: int) -> tuple[int, int, int]:
    """(matched assessments, ARW-only assessments, NMD-only detections) within 5 km."""
    points = _at_rank(result, min_rank)
    matched = sum(1 for p in points if any(_distance(p, d.latitude, d.longitude) <= NMD_MATCH_KM for d in result.nmd))
    detected = sum(1 for d in result.nmd if any(_distance(p, d.latitude, d.longitude) <= NMD_MATCH_KM for p in points))
    return matched, len(points) - matched, len(result.nmd) - detected


def null_storm_probes(result: VolumeResult, reports: list[StormReport], min_rank: int) -> tuple[int, int]:
    """(probes, probes with an assessment within 10 km).

    A probe is a storm centroid more than 50 km from every SPC report within
    60 minutes and from every NMD detection in the volume.
    """
    if result.label == "clear_air":
        return 0, 0
    nearby = [r for r in reports if abs(r.time_utc - result.volume_time) <= NULL_REPORT_WINDOW]
    points = _at_rank(result, min_rank)
    probes = alarms = 0
    for centroid in result.storm_centroids:
        if any(_distance(centroid, r.latitude, r.longitude) <= NULL_EXCLUSION_KM for r in nearby):
            continue
        if any(_distance(centroid, d.latitude, d.longitude) <= NULL_EXCLUSION_KM for d in result.nmd):
            continue
        probes += 1
        alarms += any(_distance(centroid, p.latitude, p.longitude) <= HIT_RADIUS_KM for p in points)
    return probes, alarms


def clear_air_false_alarms(result: VolumeResult, min_rank: int) -> int:
    return len(_at_rank(result, min_rank)) if result.label == "clear_air" else 0


def _case_report(case: dict) -> StormReport:
    report = case["report"]
    return StormReport(
        kind=case["label"],
        time_utc=datetime.fromisoformat(report["time_utc"]),
        latitude=report["latitude"],
        longitude=report["longitude"],
        magnitude=report.get("magnitude"),
        location=report.get("location", ""),
        state=report.get("state", ""),
    )


def summarize(results: list[VolumeResult], cases: list[dict], reports: list[StormReport]) -> dict[int, dict]:
    by_case: dict[str, list[VolumeResult]] = {}
    for result in results:
        by_case.setdefault(result.case_id, []).append(result)
    tornado_cases = [c for c in cases if c["label"] == "tornado" and c["id"] in by_case]
    summary: dict[int, dict] = {}
    for rank in (1, 2, 3):
        agreement = [nmd_agreement(r, rank) for r in results]
        probes = [null_storm_probes(r, reports, rank) for r in results]
        summary[rank] = {
            "tornado_events": len(tornado_cases),
            "tornado_hits": sum(event_hit(by_case[c["id"]], _case_report(c), rank) for c in tornado_cases),
            "nmd_matched": sum(a[0] for a in agreement),
            "arw_only": sum(a[1] for a in agreement),
            "nmd_only": sum(a[2] for a in agreement),
            "null_probes": sum(p[0] for p in probes),
            "null_false_alarms": sum(p[1] for p in probes),
            "clear_air_false_alarms": sum(clear_air_false_alarms(r, rank) for r in results),
            "clear_air_volumes": sum(1 for r in results if r.label == "clear_air"),
        }
        summary[rank]["speakable"] = speakable(summary[rank])
    return summary


def speakable(rank_summary: dict) -> bool:
    """Pre-registered speak rule: zero clear-air false alarms, at least 2 tornado
    hits, and a hit fraction at least 3x the null-storm false-alarm fraction."""
    if rank_summary["clear_air_false_alarms"] != 0 or rank_summary["tornado_hits"] < SPEAK_MIN_HITS:
        return False
    hit_fraction = rank_summary["tornado_hits"] / max(rank_summary["tornado_events"], 1)
    null_fraction = rank_summary["null_false_alarms"] / max(rank_summary["null_probes"], 1)
    return hit_fraction >= SPEAK_MIN_LIFT * null_fraction
