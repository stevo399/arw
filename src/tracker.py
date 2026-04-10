from datetime import datetime
import numpy as np
from src.buffer import BufferedScan
from src.detection import DetectedObject
from src.motion import compute_motion, MotionVector
from src.sites import haversine_distance_km
from src.tracking.events import normalize_merge_event, normalize_split_event
from src.tracking.types import PeakEntry, Track, TrackPosition

MIN_OVERLAP_PCT = 0.30
MAX_STORM_SPEED_KMH = 120.0
MAX_MISSED_SCANS = 2


def _get_track_motion(track: Track) -> MotionVector:
    pos_tuples = [
        (p.timestamp, p.latitude, p.longitude)
        for p in track.positions
    ]
    return compute_motion(pos_tuples)


def _compute_overlap(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Compute overlap percentage between two boolean masks.
    Returns the fraction of mask_a's pixels that overlap with mask_b.
    """
    if not np.any(mask_a):
        return 0.0
    intersection = np.sum(mask_a & mask_b)
    return float(intersection) / float(np.sum(mask_a))


class StormTracker:
    """Tracks storms across multiple radar scans."""

    def __init__(self):
        self._tracks: list[Track] = []
        self._next_id: int = 1
        self._recent_events: list[dict] = []
        self._prev_scan: BufferedScan | None = None
        self._obj_to_track: dict[int, int] = {}  # object_id -> track_id for current scan

    def _reset_for_site(self) -> None:
        """Clear tracker state when switching radar sites."""
        self._tracks.clear()
        self._next_id = 1
        self._recent_events.clear()
        self._prev_scan = None
        self._obj_to_track.clear()

    def _create_track(self, timestamp: datetime, obj: DetectedObject) -> Track:
        track = Track(track_id=self._next_id, status="active")
        self._next_id += 1
        track.add_position(timestamp, obj)
        self._tracks.append(track)
        return track

    def _append_merge_event(self, timestamp: datetime, surviving_track_id: int, merged_track_ids: list[int]) -> None:
        """Record a merge event with deduplicated track ids."""
        event = normalize_merge_event(timestamp, surviving_track_id, merged_track_ids)
        if event is not None:
            self._recent_events.append(event)

    def update(self, scan: BufferedScan) -> None:
        """Process a new scan: match objects to tracks, detect merges/splits."""
        self._recent_events.clear()
        timestamp = scan.timestamp

        if self._prev_scan is not None and self._prev_scan.site_id != scan.site_id:
            self._reset_for_site()

        if self._prev_scan is None:
            # First scan: create a track for each object
            self._obj_to_track.clear()
            for obj in scan.detected_objects:
                track = self._create_track(timestamp, obj)
                self._obj_to_track[obj.object_id] = track.track_id
            self._prev_scan = scan
            return

        prev_masks = self._prev_scan.object_masks
        new_masks = scan.object_masks
        prev_objects = {obj.object_id: obj for obj in self._prev_scan.detected_objects}
        new_objects = {obj.object_id: obj for obj in scan.detected_objects}

        # Compute scan interval for centroid fallback
        dt_hours = (timestamp - self._prev_scan.timestamp).total_seconds() / 3600.0
        max_distance_km = MAX_STORM_SPEED_KMH * dt_hours if dt_hours > 0 else 50.0

        # Build match candidates: new_obj_id -> list of (prev_obj_id, overlap_pct)
        match_candidates: dict[int, list[tuple[int, float]]] = {
            nid: [] for nid in new_objects
        }
        # Also track reverse: prev_obj_id -> list of (new_obj_id, overlap_pct)
        reverse_candidates: dict[int, list[tuple[int, float]]] = {
            pid: [] for pid in prev_objects
        }

        for new_id, new_mask in new_masks.items():
            for prev_id, prev_mask in prev_masks.items():
                overlap = _compute_overlap(prev_mask, new_mask)
                if overlap >= MIN_OVERLAP_PCT:
                    match_candidates[new_id].append((prev_id, overlap))
                    reverse_candidates.setdefault(prev_id, []).append((new_id, overlap))

        # Centroid fallback for unmatched new objects
        for new_id in list(match_candidates.keys()):
            if not match_candidates[new_id] and new_id in new_objects:
                new_obj = new_objects[new_id]
                for prev_id, prev_obj in prev_objects.items():
                    dist = haversine_distance_km(
                        prev_obj.centroid_lat, prev_obj.centroid_lon,
                        new_obj.centroid_lat, new_obj.centroid_lon,
                    )
                    if dist <= max_distance_km:
                        match_candidates[new_id].append((prev_id, 0.0))
                        reverse_candidates.setdefault(prev_id, []).append((new_id, 0.0))

        # Sort candidates once so merge/split/simple passes make consistent choices.
        all_pairs = []
        for new_id, candidates in match_candidates.items():
            for prev_id, overlap in candidates:
                all_pairs.append((overlap, prev_id, new_id))
        all_pairs.sort(reverse=True)

        assignments: dict[int, list[int]] = {nid: [] for nid in new_objects}
        reverse_assignments: dict[int, list[int]] = {pid: [] for pid in prev_objects}
        overlap_lookup: dict[tuple[int, int], float] = {}
        for overlap, prev_id, new_id in all_pairs:
            if prev_id not in assignments[new_id]:
                assignments[new_id].append(prev_id)
            if new_id not in reverse_assignments[prev_id]:
                reverse_assignments[prev_id].append(new_id)
            overlap_lookup[(new_id, prev_id)] = overlap

        # Process assignments: detect merges, splits, and simple updates
        new_obj_to_track: dict[int, int] = {}
        claimed_new: set[int] = set()
        claimed_prev: set[int] = set()

        # Handle merges: new object matched to multiple previous objects
        merge_candidates = [
            (
                sum(overlap_lookup[(new_id, prev_id)] for prev_id in prev_ids),
                max(overlap_lookup[(new_id, prev_id)] for prev_id in prev_ids),
                new_id,
            )
            for new_id, prev_ids in assignments.items()
            if len(prev_ids) > 1
        ]
        merge_candidates.sort(reverse=True)
        for _, _, new_id in merge_candidates:
            prev_ids = assignments[new_id]
            available_prev_ids = [prev_id for prev_id in prev_ids if prev_id not in claimed_prev]
            if len(available_prev_ids) <= 1:
                continue
            survivor_prev_id = max(
                available_prev_ids,
                key=lambda prev_id: (overlap_lookup[(new_id, prev_id)], -self._obj_to_track.get(prev_id, 10**9)),
            )
            surviving_track_id = self._obj_to_track.get(survivor_prev_id)
            if surviving_track_id is None:
                continue
            surviving = self.get_track(surviving_track_id)
            if surviving is None or surviving.status != "active":
                continue

            surviving.add_position(timestamp, new_objects[new_id])
            new_obj_to_track[new_id] = surviving_track_id
            claimed_new.add(new_id)
            claimed_prev.update(available_prev_ids)

            merged_track_ids = []
            for prev_id in available_prev_ids:
                if prev_id == survivor_prev_id:
                    continue
                merged_track_id = self._obj_to_track.get(prev_id)
                if merged_track_id is None:
                    continue
                merged_track = self.get_track(merged_track_id)
                if merged_track is not None and merged_track.status == "active":
                    merged_track.status = "merged"
                    merged_track.merged_into = surviving_track_id
                    merged_track_ids.append(merged_track_id)
            self._append_merge_event(timestamp, surviving_track_id, merged_track_ids)

        # Handle splits: one previous object matched to multiple new objects
        split_candidates = [
            (
                max(overlap_lookup[(new_id, prev_id)] for new_id in new_ids),
                len(new_ids),
                prev_id,
            )
            for prev_id, new_ids in reverse_assignments.items()
            if len(new_ids) > 1
        ]
        split_candidates.sort(reverse=True)
        for _, _, prev_id in split_candidates:
            if prev_id in claimed_prev:
                continue
            new_ids = reverse_assignments[prev_id]
            available_new_ids = [new_id for new_id in new_ids if new_id not in claimed_new]
            if len(available_new_ids) <= 1:
                continue
            parent_tid = self._obj_to_track.get(prev_id)
            if parent_tid is None:
                continue
            parent_track = self.get_track(parent_tid)
            if parent_track is None or parent_track.status != "active":
                continue
            # Parent continues with largest piece
            largest = max(available_new_ids, key=lambda nid: new_objects[nid].area_km2)
            parent_track.add_position(timestamp, new_objects[largest])
            new_obj_to_track[largest] = parent_tid
            claimed_prev.add(prev_id)
            claimed_new.add(largest)
            child_ids = []
            for nid in available_new_ids:
                if nid == largest:
                    continue
                child = self._create_track(timestamp, new_objects[nid])
                child.split_from = parent_tid
                new_obj_to_track[nid] = child.track_id
                claimed_new.add(nid)
                child_ids.append(child.track_id)
            event = normalize_split_event(timestamp, parent_tid, child_ids)
            if event is not None:
                self._recent_events.append(event)

        # Handle simple 1:1 matches
        for overlap, prev_id, new_id in all_pairs:
            if new_id in claimed_new or prev_id in claimed_prev:
                continue
            tid = self._obj_to_track.get(prev_id)
            if tid is None:
                continue
            track = self.get_track(tid)
            if track is None or track.status != "active":
                continue
            track.add_position(timestamp, new_objects[new_id])
            new_obj_to_track[new_id] = tid
            claimed_new.add(new_id)
            claimed_prev.add(prev_id)

        # Create new tracks for unmatched new objects
        for new_id, obj in new_objects.items():
            if new_id not in claimed_new:
                track = self._create_track(timestamp, obj)
                new_obj_to_track[new_id] = track.track_id

        # Increment missed scans for unmatched active tracks
        matched_track_ids = set(new_obj_to_track.values())
        for track in self._tracks:
            if track.status == "active" and track.track_id not in matched_track_ids:
                track._missed_scans += 1
                if track._missed_scans >= MAX_MISSED_SCANS:
                    track.status = "lost"

        self._obj_to_track = new_obj_to_track
        self._prev_scan = scan

    @property
    def active_tracks(self) -> list[Track]:
        return [t for t in self._tracks if t.status == "active"]

    @property
    def all_tracks(self) -> list[Track]:
        return list(self._tracks)

    @property
    def recent_events(self) -> list[dict]:
        return list(self._recent_events)

    def get_track(self, track_id: int) -> Track | None:
        for t in self._tracks:
            if t.track_id == track_id:
                return t
        return None


Track.get_motion = _get_track_motion
