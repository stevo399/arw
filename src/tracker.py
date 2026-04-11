from datetime import datetime
from src.buffer import BufferedScan
from src.detection import DetectedObject
from src.motion import resolve_reported_motion, MotionVector
from src.tracking.association import associate_tracks
from src.tracking.events import normalize_merge_event, normalize_split_event
from src.tracking.types import Track

MAX_MISSED_SCANS = 2
FOCUS_SWITCH_MARGIN = 2.0


def _scan_quality_factor(scan: BufferedScan | None) -> float:
    if scan is None or scan.scan_quality is None:
        return 1.0
    return max(0.25, min(scan.scan_quality.score, 1.0))


def _get_track_motion(track: Track) -> MotionVector:
    if track.last_motion is not None:
        return track.last_motion
    pos_tuples = [(p.timestamp, p.latitude, p.longitude) for p in track.positions]
    reported_motion, diagnostic_motion = resolve_reported_motion(
        pos_tuples,
        identity_confidence=track.identity_confidence,
    )
    track.last_motion = reported_motion
    track.diagnostic_motion = diagnostic_motion
    track.motion_confidence = reported_motion.confidence
    return reported_motion


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
        track.identity_confidence = 0.3
        self._tracks.append(track)
        return track

    def _score_confidence(self, association, new_object_id: int, track_id: int, scan: BufferedScan) -> float:
        for score in association.candidate_scores:
            if score.object_id == new_object_id and score.track_id == track_id:
                base_confidence = max(0.0, min(1.0, 1.0 - (score.total_cost / 10.0)))
                return round(base_confidence * _scan_quality_factor(scan), 2)
        return round(0.3 * _scan_quality_factor(scan), 2)

    def _refresh_track_motions(self, field_estimate, field_dt_hours: float) -> None:
        for track in self._tracks:
            if track.status != "active":
                continue
            positions = [(p.timestamp, p.latitude, p.longitude) for p in track.positions]
            reported_motion, diagnostic_motion = resolve_reported_motion(
                positions,
                identity_confidence=track.identity_confidence,
                field_estimate=field_estimate,
                field_dt_hours=field_dt_hours,
            )
            track.last_motion = reported_motion
            track.diagnostic_motion = diagnostic_motion
            track.motion_confidence = reported_motion.confidence

    def _focus_score(self, track: Track, previous_focus_track_id: int | None = None) -> float:
        if track.current_object is None:
            return float("-inf")
        current = track.current_object
        persistence = min(len(track.positions), 6) * 0.7
        confidence = track.identity_confidence * 2.0
        area_bonus = min(current.area_km2 / 150.0, 8.0)
        peak_bonus = current.peak_dbz / 10.0
        prior_focus_bonus = 2.5 if previous_focus_track_id is not None and track.track_id == previous_focus_track_id else 0.0
        distance_penalty = min(current.distance_km / 40.0, 6.0)
        return peak_bonus + area_bonus + persistence + confidence + prior_focus_bonus - distance_penalty

    def _update_primary_focus(self) -> None:
        active_tracks = [track for track in self._tracks if track.status == "active" and track.current_object is not None]
        previous_focus_track_id = next((track.track_id for track in active_tracks if track.is_primary_focus), None)
        previous_focus_track = next((track for track in active_tracks if track.track_id == previous_focus_track_id), None)
        for track in active_tracks:
            track.is_primary_focus = False
        if not active_tracks:
            return
        ranked_tracks = sorted(
            active_tracks,
            key=lambda track: (
                self._focus_score(track, previous_focus_track_id),
                track.current_object.peak_dbz if track.current_object is not None else 0.0,
                track.current_object.area_km2 if track.current_object is not None else 0.0,
            ),
            reverse=True,
        )
        primary = ranked_tracks[0]
        if previous_focus_track is not None:
            previous_score = self._focus_score(previous_focus_track, previous_focus_track_id)
            challenger_score = self._focus_score(primary, previous_focus_track_id)
            if primary.track_id != previous_focus_track.track_id and challenger_score < previous_score + FOCUS_SWITCH_MARGIN:
                primary = previous_focus_track
        primary.is_primary_focus = True

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
                track.identity_confidence = round(_scan_quality_factor(scan), 2)
                self._obj_to_track[obj.object_id] = track.track_id
            self._refresh_track_motions(field_estimate=None, field_dt_hours=0.0)
            self._update_primary_focus()
            self._prev_scan = scan
            return

        new_objects = {obj.object_id: obj for obj in scan.detected_objects}
        association = associate_tracks(
            previous_scan=self._prev_scan,
            current_scan=scan,
            tracks=self._tracks,
            obj_to_track=self._obj_to_track,
        )

        new_obj_to_track: dict[int, int] = {}
        claimed_new: set[int] = set(association.primary_matches.keys())

        handled_split_parents: set[int] = set()
        for parent_tid, related_new_ids in association.split_candidates.items():
            if not related_new_ids:
                continue
            parent_track = self.get_track(parent_tid)
            if parent_track is None or parent_track.status != "active":
                continue
            primary_new_id = related_new_ids[0]
            parent_track.add_position(timestamp, new_objects[primary_new_id])
            parent_track.identity_confidence = self._score_confidence(association, primary_new_id, parent_tid, scan)
            new_obj_to_track[primary_new_id] = parent_tid
            child_ids = []
            for new_id in related_new_ids[1:]:
                child = self._create_track(timestamp, new_objects[new_id])
                child.split_from = parent_tid
                child.identity_confidence = round(0.35 * _scan_quality_factor(scan), 2)
                new_obj_to_track[new_id] = child.track_id
                child_ids.append(child.track_id)
            event = normalize_split_event(timestamp, parent_tid, child_ids)
            if event is not None:
                self._recent_events.append(event)
            handled_split_parents.add(parent_tid)

        handled_merge_new_ids: set[int] = set()
        for new_id, related_track_ids in association.merge_candidates.items():
            if not related_track_ids:
                continue
            surviving_track_id = related_track_ids[0]
            surviving = self.get_track(surviving_track_id)
            if surviving is None or surviving.status != "active":
                continue
            if surviving_track_id not in handled_split_parents:
                surviving.add_position(timestamp, new_objects[new_id])
                surviving.identity_confidence = self._score_confidence(association, new_id, surviving_track_id, scan)
                new_obj_to_track[new_id] = surviving_track_id
            merged_track_ids = []
            for merged_track_id in related_track_ids[1:]:
                merged_track = self.get_track(merged_track_id)
                if merged_track is not None and merged_track.status == "active":
                    merged_track.status = "merged"
                    merged_track.merged_into = surviving_track_id
                    merged_track_ids.append(merged_track_id)
            self._append_merge_event(timestamp, surviving_track_id, merged_track_ids)
            handled_merge_new_ids.add(new_id)

        for new_id, track_id in association.primary_matches.items():
            if new_id in new_obj_to_track:
                continue
            track = self.get_track(track_id)
            if track is None or track.status != "active":
                continue
            track.add_position(timestamp, new_objects[new_id])
            track.identity_confidence = self._score_confidence(association, new_id, track_id, scan)
            new_obj_to_track[new_id] = track_id

        # Create new tracks for unmatched new objects
        for new_id, obj in new_objects.items():
            if new_id not in new_obj_to_track:
                track = self._create_track(timestamp, obj)
                track.identity_confidence = round(0.3 * _scan_quality_factor(scan), 2)
                new_obj_to_track[new_id] = track.track_id

        # Increment missed scans for unmatched active tracks
        matched_track_ids = set(new_obj_to_track.values())
        for track in self._tracks:
            if track.status == "active" and track.track_id not in matched_track_ids:
                track._missed_scans += 1
                track.identity_confidence = max(0.0, round(track.identity_confidence - 0.2, 2))
                if track._missed_scans >= MAX_MISSED_SCANS:
                    track.status = "lost"

        self._obj_to_track = new_obj_to_track
        self._refresh_track_motions(field_estimate=association.geo_motion, field_dt_hours=association.dt_hours)
        self._update_primary_focus()
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
