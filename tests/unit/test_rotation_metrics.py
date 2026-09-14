from datetime import datetime, timedelta

from src.validation.nmd import MesocycloneDetection
from src.validation.rotation_metrics import (
    Point, VolumeResult, clear_air_false_alarms, event_hit, nmd_agreement, null_storm_probes, speakable, summarize,
)
from src.validation.spc import StormReport

T = datetime(2026, 9, 12, 19, 40)
REPORT = StormReport("tornado", T, 29.60, -81.28, 1.0, "Palm Coast", "FL")


def volume(minutes, assessments=(), centroids=(), nmd=(), label="tornado", case_id="case"):
    return VolumeResult(case_id, label, T + timedelta(minutes=minutes), list(assessments), list(centroids), list(nmd))


def test_a_hit_needs_rank_distance_and_time():
    near = (Point(29.65, -81.28), "unconfirmed")          # 5.6 km away
    assert event_hit([volume(-5, [near])], REPORT, 1)
    assert not event_hit([volume(-5, [near])], REPORT, 2)
    assert not event_hit([volume(-25, [near])], REPORT, 1)  # before the window
    assert not event_hit([volume(-5, [(Point(29.75, -81.28), "persistent")])], REPORT, 1)  # 16.7 km


def test_nmd_agreement_counts_matches_within_5_km():
    detection = MesocycloneDetection("1", 0.0, 0.0, 5, "A1", False, 29.60, -81.28)
    result = volume(0, [(Point(29.62, -81.28), "unconfirmed"), (Point(30.5, -81.0), "unconfirmed")], nmd=[detection])
    assert nmd_agreement(result, 1) == (1, 1, 0)


def test_null_storm_probes_exclude_storms_near_reports_or_detections():
    far_storm, near_storm = Point(31.0, -83.0), Point(29.7, -81.3)
    result = volume(0, [(Point(31.02, -83.0), "unconfirmed")], centroids=[far_storm, near_storm])
    assert null_storm_probes(result, [REPORT], 1) == (1, 1)
    assert null_storm_probes(result, [REPORT], 2) == (1, 0)


def test_clear_air_false_alarms_count_only_clear_air_volumes():
    assessments = [(Point(33.0, -112.0), "unconfirmed")]
    assert clear_air_false_alarms(volume(0, assessments, label="clear_air"), 1) == 1
    assert clear_air_false_alarms(volume(0, assessments, label="tornado"), 1) == 0


def test_speak_rule():
    assert speakable({"clear_air_false_alarms": 0, "tornado_hits": 3, "tornado_events": 9, "null_probes": 100, "null_false_alarms": 5})
    assert not speakable({"clear_air_false_alarms": 1, "tornado_hits": 9, "tornado_events": 9, "null_probes": 100, "null_false_alarms": 0})
    assert not speakable({"clear_air_false_alarms": 0, "tornado_hits": 1, "tornado_events": 9, "null_probes": 100, "null_false_alarms": 0})
    assert not speakable({"clear_air_false_alarms": 0, "tornado_hits": 3, "tornado_events": 9, "null_probes": 100, "null_false_alarms": 20})


def test_summarize_scores_each_tornado_case_against_its_own_report():
    cases = [
        {"id": "hit", "label": "tornado", "report": {"time_utc": T.isoformat(), "latitude": 29.60, "longitude": -81.28}},
        {"id": "miss", "label": "tornado", "report": {"time_utc": T.isoformat(), "latitude": 35.0, "longitude": -97.0}},
    ]
    other = StormReport("tornado", T, 35.0, -97.0, None, "Elsewhere", "OK")
    results = [
        volume(0, [(Point(29.61, -81.28), "unconfirmed")], case_id="hit"),
        volume(0, [], case_id="miss"),
    ]
    summary = summarize(results, cases, [REPORT, other])
    assert summary[1]["tornado_events"] == 2 and summary[1]["tornado_hits"] == 1
    assert summary[2]["tornado_hits"] == 0
