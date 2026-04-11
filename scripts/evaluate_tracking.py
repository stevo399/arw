import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
import statistics
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.buffer import BufferedScan
from src.detection import detect_objects_with_grid
from src.ingest import fetch_scan, get_cache_path
from src.parser import extract_reflectivity
from src.preprocess import preprocess_reflectivity_data
from src.tracker import StormTracker
from scripts.live_replay import (
    _local_only_scans,
    _scan_filename,
    _scan_timestamp,
    _select_scans,
    _site_name,
    summarize_scan,
)


@dataclass
class ReplaySnapshot:
    timestamp: str
    object_count: int
    active_count: int
    uncertain_tracks: int
    max_speed_mph: int
    merge_count: int
    split_count: int
    focus_track_id: int | None
    focus_identity_label: str | None
    focus_identity_score: float | None
    focus_continuity_label: str | None
    focus_continuity_score: float | None
    focus_heading_deg: float | None
    focus_heading_label: str | None
    focus_speed_mph: int | None
    focus_motion_confidence_label: str | None
    focus_motion_confidence_score: float | None
    new_tracks: int
    total_tracks_seen: int
    summary: str


@dataclass
class BenchmarkResult:
    benchmark_id: str
    category: str
    site_id: str
    date: str | None
    local_only: bool
    scan_count: int
    mean_objects: float
    mean_active_tracks: float
    max_speed_mph: int
    total_merges: int
    total_splits: int
    mean_uncertain_tracks: float
    mean_focus_identity_confidence: float
    mean_focus_continuity: float
    mean_focus_motion_confidence: float
    focus_low_identity_scans: int
    focus_low_continuity_scans: int
    focus_low_motion_scans: int
    focus_switches: int
    focus_heading_flips_ge_90: int
    focus_flips_with_low_motion_confidence: int
    focus_track_distance_changes: int
    total_new_tracks_after_first_scan: int
    merged_tracks_total: int
    lost_tracks_total: int
    split_children_total: int
    absorbed_links_total: int
    fragmentation_proxy: float
    snapshots: list[ReplaySnapshot]


def _heading_delta_deg(a: float | None, b: float | None) -> float:
    if a is None or b is None:
        return 0.0
    delta = abs(a - b) % 360.0
    return min(delta, 360.0 - delta)


def _mean(values) -> float:
    normalized = [float(value) for value in values]
    if not normalized:
        return 0.0
    return float(statistics.mean(normalized))


def _normalize_json_value(value):
    if isinstance(value, dict):
        return {key: _normalize_json_value(inner) for key, inner in value.items()}
    if isinstance(value, list):
        return [_normalize_json_value(inner) for inner in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except ValueError:
            return value
    return value


def _load_manifest(path: str) -> list[dict]:
    return json.loads(Path(path).read_text())


def _focus_track(tracker: StormTracker):
    return next((track for track in tracker.active_tracks if getattr(track, "is_primary_focus", False)), None)


def _snapshot(site_name: str, buffered: BufferedScan, tracker: StormTracker, seen_track_ids: set[int]) -> ReplaySnapshot:
    diagnostics = summarize_scan(site_name, buffered, tracker)
    focus_track = _focus_track(tracker)
    focus_motion = focus_track.get_motion() if focus_track is not None else None
    current_track_ids = {track.track_id for track in tracker.all_tracks}
    new_track_ids = current_track_ids - seen_track_ids
    seen_track_ids.update(current_track_ids)
    return ReplaySnapshot(
        timestamp=diagnostics.timestamp,
        object_count=diagnostics.object_count,
        active_count=diagnostics.active_count,
        uncertain_tracks=diagnostics.uncertain_tracks,
        max_speed_mph=diagnostics.max_speed_mph,
        merge_count=diagnostics.merge_count,
        split_count=diagnostics.split_count,
        focus_track_id=focus_track.track_id if focus_track is not None else None,
        focus_identity_label=(
            focus_track.identity_diagnostics.label
            if focus_track is not None and getattr(focus_track, "identity_diagnostics", None) is not None
            else None
        ),
        focus_identity_score=(
            round(float(focus_track.identity_diagnostics.score), 2)
            if focus_track is not None and getattr(focus_track, "identity_diagnostics", None) is not None
            else (round(float(focus_track.identity_confidence), 2) if focus_track is not None else None)
        ),
        focus_continuity_label=(
            focus_track.focus_continuity.label
            if focus_track is not None and getattr(focus_track, "focus_continuity", None) is not None
            else None
        ),
        focus_continuity_score=(
            round(float(focus_track.focus_continuity.score), 2)
            if focus_track is not None and getattr(focus_track, "focus_continuity", None) is not None
            else None
        ),
        focus_heading_deg=focus_motion.heading_deg if focus_motion is not None else None,
        focus_heading_label=focus_motion.heading_label if focus_motion is not None else None,
        focus_speed_mph=focus_motion.speed_mph if focus_motion is not None else None,
        focus_motion_confidence_label=(
            focus_motion.confidence.label if focus_motion is not None and focus_motion.confidence is not None else None
        ),
        focus_motion_confidence_score=(
            round(float(focus_motion.confidence.score), 2)
            if focus_motion is not None and focus_motion.confidence is not None
            else None
        ),
        new_tracks=len(new_track_ids),
        total_tracks_seen=len(seen_track_ids),
        summary=diagnostics.summary,
    )


def run_benchmark(entry: dict) -> BenchmarkResult:
    site_id = entry["site_id"].upper()
    date_str = entry.get("date")
    local_only = bool(entry.get("local_only", False))
    scan_count = int(entry.get("scans", 5))
    scans = _select_scans(site_id, date_str, scan_count)
    if local_only:
        scans = _local_only_scans(site_id, date_str, scans, scan_count)
        if not scans:
            raise RuntimeError(f"No cached scans available for benchmark {entry['id']}")

    tracker = StormTracker()
    seen_track_ids: set[int] = set()
    snapshots: list[ReplaySnapshot] = []
    site_name = _site_name(site_id)
    for scan in scans:
        scan_dt = _scan_timestamp(scan)
        filepath = get_cache_path(site_id, _scan_filename(scan)) if local_only else fetch_scan(site_id, scan_dt)
        raw_reflectivity = extract_reflectivity(filepath)
        reflectivity, scan_quality = preprocess_reflectivity_data(raw_reflectivity)
        detection = detect_objects_with_grid(
            reflectivity=reflectivity.reflectivity,
            azimuths=reflectivity.azimuths,
            ranges_m=reflectivity.ranges_m,
            radar_lat=reflectivity.radar_lat,
            radar_lon=reflectivity.radar_lon,
        )
        buffered = BufferedScan(
            timestamp=datetime.fromisoformat(reflectivity.timestamp),
            site_id=site_id,
            reflectivity_data=reflectivity,
            detected_objects=detection.objects,
            labeled_grid=detection.labeled_grid,
            object_masks=detection.object_masks,
            scan_quality=scan_quality,
        )
        tracker.update(buffered)
        snapshots.append(_snapshot(site_name, buffered, tracker, seen_track_ids))

    focus_switches = 0
    focus_heading_flips_ge_90 = 0
    focus_flips_with_low_motion_confidence = 0
    focus_track_distance_changes = 0
    previous_focus_track_id = None
    previous_focus_heading_deg = None
    for snapshot in snapshots:
        if previous_focus_track_id is not None and snapshot.focus_track_id is not None and snapshot.focus_track_id != previous_focus_track_id:
            focus_switches += 1
        if previous_focus_heading_deg is not None and snapshot.focus_heading_deg is not None:
            if _heading_delta_deg(previous_focus_heading_deg, snapshot.focus_heading_deg) >= 90.0:
                focus_heading_flips_ge_90 += 1
                focus_track_distance_changes += 1
                if snapshot.focus_motion_confidence_score is not None and snapshot.focus_motion_confidence_score < 0.45:
                    focus_flips_with_low_motion_confidence += 1
        previous_focus_track_id = snapshot.focus_track_id
        previous_focus_heading_deg = snapshot.focus_heading_deg

    all_tracks = tracker.all_tracks
    merged_tracks_total = sum(1 for track in all_tracks if track.status == "merged")
    lost_tracks_total = sum(1 for track in all_tracks if track.status == "lost")
    split_children_total = sum(len(track.child_track_ids) for track in all_tracks)
    absorbed_links_total = sum(len(track.absorbed_track_ids) for track in all_tracks)
    total_new_tracks_after_first_scan = sum(snapshot.new_tracks for snapshot in snapshots[1:])
    fragmentation_proxy = round(total_new_tracks_after_first_scan / max(sum(snapshot.object_count for snapshot in snapshots), 1), 3)

    return BenchmarkResult(
        benchmark_id=entry["id"],
        category=entry["category"],
        site_id=site_id,
        date=date_str,
        local_only=local_only,
        scan_count=len(snapshots),
        mean_objects=round(_mean(snapshot.object_count for snapshot in snapshots), 2),
        mean_active_tracks=round(_mean(snapshot.active_count for snapshot in snapshots), 2),
        max_speed_mph=max(snapshot.max_speed_mph for snapshot in snapshots),
        total_merges=sum(snapshot.merge_count for snapshot in snapshots),
        total_splits=sum(snapshot.split_count for snapshot in snapshots),
        mean_uncertain_tracks=round(_mean(snapshot.uncertain_tracks for snapshot in snapshots), 2),
        mean_focus_identity_confidence=round(
            _mean(snapshot.focus_identity_score for snapshot in snapshots if snapshot.focus_identity_score is not None),
            2,
        ),
        mean_focus_continuity=round(
            _mean(snapshot.focus_continuity_score for snapshot in snapshots if snapshot.focus_continuity_score is not None),
            2,
        ),
        mean_focus_motion_confidence=round(
            _mean(
                snapshot.focus_motion_confidence_score
                for snapshot in snapshots
                if snapshot.focus_motion_confidence_score is not None
            ),
            2,
        ),
        focus_low_identity_scans=sum(
            1 for snapshot in snapshots if snapshot.focus_identity_score is not None and snapshot.focus_identity_score < 0.45
        ),
        focus_low_continuity_scans=sum(
            1 for snapshot in snapshots if snapshot.focus_continuity_score is not None and snapshot.focus_continuity_score < 0.45
        ),
        focus_low_motion_scans=sum(
            1
            for snapshot in snapshots
            if snapshot.focus_motion_confidence_score is not None and snapshot.focus_motion_confidence_score < 0.45
        ),
        focus_switches=focus_switches,
        focus_heading_flips_ge_90=focus_heading_flips_ge_90,
        focus_flips_with_low_motion_confidence=focus_flips_with_low_motion_confidence,
        focus_track_distance_changes=focus_track_distance_changes,
        total_new_tracks_after_first_scan=total_new_tracks_after_first_scan,
        merged_tracks_total=merged_tracks_total,
        lost_tracks_total=lost_tracks_total,
        split_children_total=split_children_total,
        absorbed_links_total=absorbed_links_total,
        fragmentation_proxy=fragmentation_proxy,
        snapshots=snapshots,
    )


def render_markdown(results: list[BenchmarkResult]) -> str:
    lines = [
        "# Tracking Evaluation Report",
        "",
        f"Date: {datetime.now().date().isoformat()}",
        "",
        "## Benchmarks",
        "",
    ]
    for result in results:
        lines.extend([
            f"### {result.benchmark_id}",
            "",
            f"- category: `{result.category}`",
            f"- site: `{result.site_id}`",
            f"- scans: `{result.scan_count}`",
            f"- mean objects: `{result.mean_objects}`",
            f"- mean active tracks: `{result.mean_active_tracks}`",
            f"- max speed mph: `{result.max_speed_mph}`",
            f"- total merges: `{result.total_merges}`",
            f"- total splits: `{result.total_splits}`",
            f"- mean uncertain tracks: `{result.mean_uncertain_tracks}`",
            f"- mean focus identity confidence: `{result.mean_focus_identity_confidence}`",
            f"- mean focus continuity: `{result.mean_focus_continuity}`",
            f"- mean focus motion confidence: `{result.mean_focus_motion_confidence}`",
            f"- focus low-identity scans: `{result.focus_low_identity_scans}`",
            f"- focus low-continuity scans: `{result.focus_low_continuity_scans}`",
            f"- focus low-motion scans: `{result.focus_low_motion_scans}`",
            f"- focus switches: `{result.focus_switches}`",
            f"- focus heading flips >=90 deg: `{result.focus_heading_flips_ge_90}`",
            f"- focus flips with low motion confidence: `{result.focus_flips_with_low_motion_confidence}`",
            f"- total new tracks after first scan: `{result.total_new_tracks_after_first_scan}`",
            f"- fragmentation proxy: `{result.fragmentation_proxy}`",
            "",
            "Representative snapshots:",
            "",
        ])
        for snapshot in result.snapshots:
            lines.append(
                f"- `{snapshot.timestamp} objects={snapshot.object_count} active={snapshot.active_count} "
                f"uncertain={snapshot.uncertain_tracks} max_speed_mph={snapshot.max_speed_mph} "
                f"merges={snapshot.merge_count} splits={snapshot.split_count} focus_track={snapshot.focus_track_id} "
                f"focus_identity={snapshot.focus_identity_label}:{snapshot.focus_identity_score} "
                f"focus_continuity={snapshot.focus_continuity_label}:{snapshot.focus_continuity_score} "
                f"focus_heading={snapshot.focus_heading_label} "
                f"focus_motion_conf={snapshot.focus_motion_confidence_label}:{snapshot.focus_motion_confidence_score}`"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run quantitative tracking evaluation across benchmark replay windows.")
    parser.add_argument("--manifest", default="docs/benchmarks/tracking_benchmark_manifest.json")
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-md", default=None)
    args = parser.parse_args()

    manifest = _load_manifest(args.manifest)
    results = [run_benchmark(entry) for entry in manifest]
    payload = [_normalize_json_value(asdict(result)) for result in results]

    if args.output_json:
        Path(args.output_json).write_text(json.dumps(payload, indent=2))
    if args.output_md:
        Path(args.output_md).write_text(render_markdown(results))
    if not args.output_json and not args.output_md:
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
