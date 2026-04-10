# src/tracker.py
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
from src.buffer import BufferedScan
from src.detection import DetectedObject
from src.motion import compute_motion, MotionVector
from src.sites import haversine_distance_km

MIN_OVERLAP_PCT = 0.30
MAX_STORM_SPEED_KMH = 120.0
MAX_MISSED_SCANS = 2


@dataclass
class TrackPosition:
    timestamp: datetime
    latitude: float
    longitude: float
    distance_km: float
    bearing_deg: float


@dataclass
class PeakEntry:
    timestamp: datetime
    peak_dbz: float
    peak_label: str


@dataclass
class Track:
    track_id: int
    status: str  # "active", "merged", "split", "lost"
    positions: list[TrackPosition] = field(default_factory=list)
    peak_history: list[PeakEntry] = field(default_factory=list)
    current_object: DetectedObject | None = None
    merged_into: int | None = None
    split_from: int | None = None
    first_seen: datetime | None = None
    last_seen: datetime | None = None
    _missed_scans: int = 0

    def add_position(self, timestamp: datetime, obj: DetectedObject) -> None:
        self.positions.append(TrackPosition(
            timestamp=timestamp,
            latitude=obj.centroid_lat,
            longitude=obj.centroid_lon,
            distance_km=obj.distance_km,
            bearing_deg=obj.bearing_deg,
        ))
        self.peak_history.append(PeakEntry(
            timestamp=timestamp,
            peak_dbz=obj.peak_dbz,
            peak_label=obj.peak_label,
        ))
        self.current_object = obj
        self.last_seen = timestamp
        self._missed_scans = 0
        if self.first_seen is None:
            self.first_seen = timestamp

    def get_motion(self) -> MotionVector:
        pos_tuples = [
            (p.timestamp, p.latitude, p.longitude)
            for p in self.positions
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

    def _create_track(self, timestamp: datetime, obj: DetectedObject) -> Track:
        track = Track(track_id=self._next_id, status="active")
        self._next_id += 1
        track.add_position(timestamp, obj)
        self._tracks.append(track)
        return track

    def update(self, scan: BufferedScan) -> None:
        """Process a new scan: match objects to tracks, detect merges/splits."""
        self._recent_events.clear()
        timestamp = scan.timestamp

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

        # Greedy assignment: sort all candidate pairs by overlap desc
        all_pairs = []
        for new_id, candidates in match_candidates.items():
            for prev_id, overlap in candidates:
                all_pairs.append((overlap, prev_id, new_id))
        all_pairs.sort(reverse=True)

        assigned_new: set[int] = set()
        assigned_prev: set[int] = set()
        assignments: dict[int, list[int]] = {}  # new_obj_id -> [prev_obj_ids]
        reverse_assignments: dict[int, list[int]] = {}  # prev_obj_id -> [new_obj_ids]

        for overlap, prev_id, new_id in all_pairs:
            # Allow many-to-one for merges and one-to-many for splits
            if new_id not in assignments:
                assignments[new_id] = []
            if prev_id not in reverse_assignments:
                reverse_assignments[prev_id] = []
            assignments[new_id].append(prev_id)
            reverse_assignments[prev_id].append(new_id)
            assigned_new.add(new_id)
            assigned_prev.add(prev_id)

        # Deduplicate assignments
        for key in assignments:
            assignments[key] = list(dict.fromkeys(assignments[key]))
        for key in reverse_assignments:
            reverse_assignments[key] = list(dict.fromkeys(reverse_assignments[key]))

        # Process assignments: detect merges, splits, and simple updates
        new_obj_to_track: dict[int, int] = {}
        processed_new: set[int] = set()

        # Handle merges: new object matched to multiple previous objects
        for new_id, prev_ids in assignments.items():
            if len(prev_ids) <= 1:
                continue
            # Merge: multiple previous -> one new
            surviving_track_id = None
            merged_track_ids = []
            for pid in prev_ids:
                tid = self._obj_to_track.get(pid)
                if tid is not None:
                    if surviving_track_id is None:
                        surviving_track_id = tid
                    else:
                        merged_track_ids.append(tid)

            if surviving_track_id is not None:
                surviving = self.get_track(surviving_track_id)
                if surviving is not None:
                    surviving.add_position(timestamp, new_objects[new_id])
                    new_obj_to_track[new_id] = surviving_track_id
                    for mtid in merged_track_ids:
                        mt = self.get_track(mtid)
                        if mt is not None and mt.status == "active":
                            mt.status = "merged"
                            mt.merged_into = surviving_track_id
                    self._recent_events.append({
                        "event_type": "merge",
                        "timestamp": timestamp.isoformat(),
                        "description": f"Tracks {', '.join(str(t) for t in merged_track_ids)} merged into track {surviving_track_id}",
                        "involved_track_ids": [surviving_track_id] + merged_track_ids,
                    })
            processed_new.add(new_id)

        # Handle splits: one previous object matched to multiple new objects
        for prev_id, new_ids in reverse_assignments.items():
            if len(new_ids) <= 1:
                continue
            unprocessed = [nid for nid in new_ids if nid not in processed_new]
            if len(unprocessed) <= 1:
                continue
            parent_tid = self._obj_to_track.get(prev_id)
            if parent_tid is None:
                continue
            parent_track = self.get_track(parent_tid)
            if parent_track is None:
                continue
            # Parent continues with largest piece
            largest = max(unprocessed, key=lambda nid: new_objects[nid].area_km2)
            parent_track.add_position(timestamp, new_objects[largest])
            new_obj_to_track[largest] = parent_tid
            processed_new.add(largest)
            child_ids = []
            for nid in unprocessed:
                if nid == largest:
                    continue
                child = self._create_track(timestamp, new_objects[nid])
                child.split_from = parent_tid
                new_obj_to_track[nid] = child.track_id
                processed_new.add(nid)
                child_ids.append(child.track_id)
            self._recent_events.append({
                "event_type": "split",
                "timestamp": timestamp.isoformat(),
                "description": f"Track {parent_tid} split into tracks {', '.join(str(c) for c in child_ids)}",
                "involved_track_ids": [parent_tid] + child_ids,
            })

        # Handle simple 1:1 matches
        for new_id, prev_ids in assignments.items():
            if new_id in processed_new:
                continue
            if len(prev_ids) == 1:
                prev_id = prev_ids[0]
                tid = self._obj_to_track.get(prev_id)
                if tid is not None:
                    track = self.get_track(tid)
                    if track is not None and track.status == "active":
                        track.add_position(timestamp, new_objects[new_id])
                        new_obj_to_track[new_id] = tid
                        processed_new.add(new_id)

        # Create new tracks for unmatched new objects
        for new_id, obj in new_objects.items():
            if new_id not in processed_new:
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
