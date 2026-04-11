import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.buffer import BufferedScan
from src.detection import detect_objects_with_grid
from src.ingest import get_cache_path, list_latest_scans, list_scans_for_date, fetch_scan, scan_is_cached
from src.parser import extract_reflectivity
from src.sites import NEXRAD_SITES
from src.summary import generate_summary
from src.tracker import StormTracker


@dataclass
class ReplayDiagnostics:
    timestamp: str
    object_count: int
    active_count: int
    uncertain_tracks: int
    max_speed_mph: int
    merge_count: int
    split_count: int
    summary: str


def _site_name(site_id: str) -> str:
    for site in NEXRAD_SITES:
        if site["site_id"] == site_id.upper():
            return site["name"]
    return site_id.upper()


def summarize_scan(site_name: str, buffered: BufferedScan, tracker: StormTracker) -> ReplayDiagnostics:
    merge_count = sum(1 for event in tracker.recent_events if event["event_type"] == "merge")
    split_count = sum(1 for event in tracker.recent_events if event["event_type"] == "split")
    uncertain_tracks = 0
    max_speed_mph = 0
    for track in tracker.active_tracks:
        motion = track.get_motion()
        if motion.heading_label == "uncertain":
            uncertain_tracks += 1
        max_speed_mph = max(max_speed_mph, motion.speed_mph)

    summary = generate_summary(
        site_id=buffered.site_id,
        site_name=site_name,
        timestamp=buffered.reflectivity_data.timestamp,
        objects=buffered.detected_objects,
        tracks=tracker.active_tracks,
        events=tracker.recent_events,
    )
    return ReplayDiagnostics(
        timestamp=buffered.reflectivity_data.timestamp,
        object_count=len(buffered.detected_objects),
        active_count=len(tracker.active_tracks),
        uncertain_tracks=uncertain_tracks,
        max_speed_mph=max_speed_mph,
        merge_count=merge_count,
        split_count=split_count,
        summary=summary,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay multiple live radar scans and print tracking diagnostics.")
    parser.add_argument("site_id", help="Radar site ID, e.g. KTLX")
    parser.add_argument("--date", help="Date to replay in YYYY-MM-DD format. Defaults to latest available scans.")
    parser.add_argument("--scans", type=int, default=None, help="Number of scans to replay. Defaults to 5, or 3 with --quick.")
    parser.add_argument("--quick", action="store_true", help="Use a short development replay window. Defaults to 3 scans.")
    parser.add_argument("--local-only", action="store_true", help="Replay only scans already present in the local cache.")
    return parser.parse_args()


def _select_scan_count(args: argparse.Namespace) -> int:
    if args.scans is not None:
        return args.scans
    if args.quick:
        return 3
    return 5


def _select_scans(site_id: str, date_str: str | None, count: int):
    if date_str:
        scans = list_scans_for_date(site_id, date_str)
    else:
        scans = list_latest_scans(site_id)
    if not scans:
        raise RuntimeError(f"No scans available for {site_id}")
    return scans[-count:]


def _scan_filename(scan) -> str:
    return scan.filename


def _cached_scans(site_id: str, scans: list) -> list:
    return [scan for scan in scans if scan_is_cached(site_id, _scan_filename(scan))]


def _scan_timestamp(scan) -> datetime:
    return datetime.strptime(_scan_filename(scan).split("_V")[0][-15:], "%Y%m%d_%H%M%S")


def main() -> None:
    args = _parse_args()
    site_id = args.site_id.upper()
    site_name = _site_name(site_id)
    scan_count = _select_scan_count(args)
    scans = _select_scans(site_id, args.date, scan_count)
    if args.local_only:
        scans = _cached_scans(site_id, scans)
        if not scans:
            raise RuntimeError(f"No cached scans available for {site_id} in the selected window")
    tracker = StormTracker()

    print(f"SITE {site_id} ({site_name})")
    print(f"REPLAY_SCANS {len(scans)}")
    print(f"MODE {'local-only' if args.local_only else 'fetch-missing'}")

    for scan in scans:
        scan_dt = _scan_timestamp(scan)
        filepath = get_cache_path(site_id, _scan_filename(scan)) if args.local_only else fetch_scan(site_id, scan_dt)
        reflectivity = extract_reflectivity(filepath)
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
        )
        tracker.update(buffered)
        diagnostics = summarize_scan(site_name, buffered, tracker)
        print(
            f"{diagnostics.timestamp} "
            f"objects={diagnostics.object_count} "
            f"active={diagnostics.active_count} "
            f"uncertain_tracks={diagnostics.uncertain_tracks} "
            f"max_speed_mph={diagnostics.max_speed_mph} "
            f"merges={diagnostics.merge_count} "
            f"splits={diagnostics.split_count}"
        )
        print(f"  {diagnostics.summary}")


if __name__ == "__main__":
    main()
