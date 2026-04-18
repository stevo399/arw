from datetime import datetime
from src.buffer import BufferedScan
from src.detection import DetectedObject
from src.motion import resolve_reported_motion, MotionVector, recent_heading_flip_count
from src.tracking.motion import MotionContinuityContext
from src.tracking.association import associate_tracks
from src.tracking.events import normalize_merge_event, normalize_split_event
from src.tracking.types import FocusContinuity, IdentityConfidence, MotionSample, Track

MAX_MISSED_SCANS = 2
FOCUS_SWITCH_MARGIN = 2.0
HIGH_CONFIDENCE = 0.75
MEDIUM_CONFIDENCE = 0.45


def _scan_quality_factor(scan: BufferedScan | None) -> float:
    if scan is None or scan.scan_quality is None:
        return 1.0
    return max(0.25, min(scan.scan_quality.score, 1.0))


def _confidence_label(score: float) -> str:
    if score >= HIGH_CONFIDENCE:
        return "high"
    if score >= MEDIUM_CONFIDENCE:
        return "medium"
    return "low"


def _heading_delta_deg(a: float | None, b: float | None) -> float:
    if a is None or b is None:
        return 0.0
    delta = abs(a - b) % 360.0
    return min(delta, 360.0 - delta)


def _signed_heading_delta_deg(a: float | None, b: float | None) -> float:
    if a is None or b is None:
        return 0.0
    return ((b - a + 540.0) % 360.0) - 180.0


def _recent_reported_heading_flip_count(track: Track, *, max_samples: int = 4) -> int:
    heading_samples = _recent_reported_heading_samples(track, max_samples=max_samples)
    if len(heading_samples) < 2:
        return 0
    return sum(
        1
        for previous_heading, current_heading in zip(heading_samples, heading_samples[1:])
        if _heading_delta_deg(previous_heading, current_heading) >= 90.0
    )


def _recent_reported_heading_samples(track: Track, *, max_samples: int = 4) -> list[float]:
    return [
        sample.heading_deg
        for sample in track.motion_history[-max_samples:]
        if sample.heading_deg is not None and sample.heading_label not in {"uncertain", "stationary", "nearly stationary"}
    ]


def _recent_reported_heading_sequence(track: Track, *, max_samples: int = 4) -> list[str]:
    sequence: list[str] = []
    for sample in track.motion_history[-max_samples:]:
        if sample.heading_deg is not None:
            sequence.append(f"{sample.heading_label}@{round(sample.heading_deg)}:{sample.source}")
        else:
            sequence.append(f"{sample.heading_label}:{sample.source}")
    return sequence


def _classify_reported_heading_stability(
    track: Track, *, max_samples: int = 4
) -> tuple[str, float, str]:
    directional_samples = [
        sample
        for sample in track.motion_history[-max_samples:]
        if sample.heading_deg is not None and sample.heading_label not in {"uncertain", "stationary", "nearly stationary"}
    ]
    if len(directional_samples) < 2:
        return ("insufficient", 1.0, "insufficient directional heading history")

    signed_deltas = [
        _signed_heading_delta_deg(previous.heading_deg, current.heading_deg)
        for previous, current in zip(directional_samples, directional_samples[1:])
    ]
    abs_deltas = [abs(delta) for delta in signed_deltas]
    reversal_count = sum(1 for delta in abs_deltas if delta >= 90.0)
    turn_signs = [1 if delta >= 20.0 else -1 if delta <= -20.0 else 0 for delta in signed_deltas]
    nonzero_turn_signs = [sign for sign in turn_signs if sign != 0]
    sign_change_count = sum(
        1 for previous_sign, current_sign in zip(nonzero_turn_signs, nonzero_turn_signs[1:]) if previous_sign != current_sign
    )
    max_abs_delta = max(abs_deltas, default=0.0)
    mean_abs_delta = sum(abs_deltas) / len(abs_deltas)

    if reversal_count >= 2 or sign_change_count >= 2:
        return ("unstable", 0.1, "oscillating reported heading sequence")
    if reversal_count >= 1 and sign_change_count >= 1:
        return ("unstable", 0.2, "reversal-prone reported heading sequence")
    if max_abs_delta <= 35.0:
        return ("stable", 0.95, "consistent reported heading sequence")
    if sign_change_count == 0 and max_abs_delta <= 75.0:
        return ("coherent_turn", 0.85, "coherent turning reported heading sequence")
    if sign_change_count == 0 and mean_abs_delta <= 60.0:
        return ("mixed", 0.65, "broad but one-directional reported heading changes")
    if reversal_count >= 1:
        return ("mixed", 0.45, "single abrupt reported heading reversal")
    return ("mixed", 0.6, "mixed reported heading sequence")


def _focus_margin_bonus(selection_margin: float | None, structural_event_count: int) -> float:
    if selection_margin is None or structural_event_count < 4:
        return 0.0
    if selection_margin >= 4.0:
        return 0.15
    if selection_margin >= 2.5:
        return 0.05
    return 0.0


def _get_track_motion(track: Track) -> MotionVector:
    if track.last_motion is not None:
        return track.last_motion
    pos_tuples = [(p.timestamp, p.latitude, p.longitude) for p in track.positions]
    reported_motion, diagnostic_motion = resolve_reported_motion(
        pos_tuples,
        identity_confidence=track.identity_confidence,
        continuity=MotionContinuityContext(
            identity_score=track.identity_confidence,
            event_context=track.identity_diagnostics.event_context if track.identity_diagnostics is not None else None,
            ambiguity_margin=track.identity_diagnostics.ambiguity_margin if track.identity_diagnostics is not None else None,
        ),
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
        self._focus_history: list[int | None] = []

    def _reset_for_site(self) -> None:
        """Clear tracker state when switching radar sites."""
        self._tracks.clear()
        self._next_id = 1
        self._recent_events.clear()
        self._prev_scan = None
        self._obj_to_track.clear()
        self._focus_history.clear()

    def _create_track(self, timestamp: datetime, obj: DetectedObject) -> Track:
        track = Track(track_id=self._next_id, status="active")
        self._next_id += 1
        track.add_position(timestamp, obj)
        track.identity_confidence = 0.3
        track.identity_diagnostics = self._build_identity_diagnostics(
            score_value=0.3,
            scan=None,
            track=track,
            reason="new track initialized before scan-quality adjustment",
            event_context="new_track",
        )
        self._tracks.append(track)
        return track

    @staticmethod
    def _append_unique(track_ids: list[int], track_id: int) -> None:
        if track_id not in track_ids:
            track_ids.append(track_id)

    def _record_split_lineage(self, parent_track: Track, child_track: Track) -> None:
        self._append_unique(parent_track.child_track_ids, child_track.track_id)
        self._append_unique(child_track.parent_track_ids, parent_track.track_id)

    def _record_merge_lineage(self, surviving_track: Track, merged_track: Track) -> None:
        self._append_unique(surviving_track.absorbed_track_ids, merged_track.track_id)
        self._append_unique(merged_track.parent_track_ids, surviving_track.track_id)

    @staticmethod
    def _lineage_complexity(track: Track) -> int:
        return len(track.parent_track_ids) + len(track.child_track_ids) + len(track.absorbed_track_ids)

    @staticmethod
    def _match_quality(score) -> float:
        return round(max(0.0, min(1.0, 1.0 - (score.total_cost / 10.0))), 2)

    def _association_ambiguity_margin(self, association, new_object_id: int, track_id: int) -> float | None:
        object_scores = sorted(
            score.total_cost for score in association.candidate_scores if score.object_id == new_object_id
        )
        track_scores = sorted(
            score.total_cost for score in association.candidate_scores if score.track_id == track_id
        )
        margins = []
        if len(object_scores) > 1:
            margins.append(object_scores[1] - object_scores[0])
        if len(track_scores) > 1:
            margins.append(track_scores[1] - track_scores[0])
        if not margins:
            return None
        return round(min(margins), 2)

    def _build_identity_diagnostics(
        self,
        *,
        score_value: float,
        scan: BufferedScan | None,
        track: Track,
        reason: str,
        match_quality: float | None = None,
        ambiguity_margin: float | None = None,
        event_context: str | None = None,
    ) -> IdentityConfidence:
        return IdentityConfidence(
            label=_confidence_label(score_value),
            score=round(max(0.0, min(1.0, score_value)), 2),
            reason=reason,
            match_quality=match_quality,
            ambiguity_margin=ambiguity_margin,
            scan_quality=round(_scan_quality_factor(scan), 2) if scan is not None else None,
            missed_scans=track._missed_scans,
            lineage_complexity=self._lineage_complexity(track),
            event_context=event_context,
        )

    def _score_confidence(self, association, new_object_id: int, track_id: int, scan: BufferedScan, track: Track) -> float:
        for score in association.candidate_scores:
            if score.object_id == new_object_id and score.track_id == track_id:
                match_quality = self._match_quality(score)
                ambiguity_margin = self._association_ambiguity_margin(association, new_object_id, track_id)
                ambiguity_reason = "well-separated association candidate"
                if ambiguity_margin is None:
                    ambiguity_score = 0.7
                    ambiguity_reason = "single viable association candidate"
                else:
                    ambiguity_score = max(0.0, min(1.0, ambiguity_margin / 0.75))
                if ambiguity_margin is not None and ambiguity_margin < 0.15:
                    ambiguity_reason = "ambiguous association margin"
                elif ambiguity_margin is not None and ambiguity_margin < 0.4:
                    ambiguity_reason = "moderately ambiguous association margin"
                lineage_score = max(0.55, 1.0 - (0.08 * self._lineage_complexity(track)))
                score_value = round(
                    (match_quality * 0.45)
                    + (ambiguity_score * 0.2)
                    + (_scan_quality_factor(scan) * 0.15)
                    + (lineage_score * 0.2),
                    2,
                )
                if ambiguity_margin is not None and ambiguity_margin < 0.15:
                    score_value = min(score_value, 0.4)
                track.identity_diagnostics = self._build_identity_diagnostics(
                    score_value=score_value,
                    scan=scan,
                    track=track,
                    reason=ambiguity_reason,
                    match_quality=match_quality,
                    ambiguity_margin=ambiguity_margin,
                    event_context="matched",
                )
                return score_value
        fallback = round(0.3 * _scan_quality_factor(scan), 2)
        track.identity_diagnostics = self._build_identity_diagnostics(
            score_value=fallback,
            scan=scan,
            track=track,
            reason="no scored association candidate",
            event_context="fallback",
        )
        return fallback

    def _refresh_track_motions(self, field_estimates, field_dt_hours: float) -> None:
        structural_event_count = sum(
            1 for event in self._recent_events if event["event_type"] in {"merge", "split"}
        )
        for track in self._tracks:
            if track.status != "active":
                continue
            positions = [(p.timestamp, p.latitude, p.longitude) for p in track.positions]
            field_estimate = None
            if isinstance(field_estimates, dict):
                field_estimate = field_estimates.get(track.track_id)
            reported_motion, diagnostic_motion = resolve_reported_motion(
                positions,
                identity_confidence=track.identity_confidence,
                field_estimate=field_estimate,
                field_dt_hours=field_dt_hours,
                continuity=MotionContinuityContext(
                    identity_score=track.identity_confidence,
                    event_context=track.identity_diagnostics.event_context if track.identity_diagnostics is not None else None,
                    ambiguity_margin=track.identity_diagnostics.ambiguity_margin if track.identity_diagnostics is not None else None,
                    structural_event_count=structural_event_count,
                ),
            )
            track.last_motion = reported_motion
            track.diagnostic_motion = diagnostic_motion
            track.motion_confidence = reported_motion.confidence
            track.motion_history.append(
                MotionSample(
                    timestamp=track.last_seen or (self._prev_scan.timestamp if self._prev_scan is not None else datetime.min),
                    heading_deg=reported_motion.heading_deg,
                    heading_label=reported_motion.heading_label,
                    source=reported_motion.source,
                    confidence_score=reported_motion.confidence.score if reported_motion.confidence is not None else None,
                )
            )
            if len(track.motion_history) > 6:
                track.motion_history = track.motion_history[-6:]

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

    def _build_focus_continuity(
        self,
        track: Track,
        previous_focus_track_id: int | None,
        structural_event_count: int,
        selection_margin: float | None = None,
        runner_up_track_id: int | None = None,
    ) -> FocusContinuity:
        identity_score = track.identity_diagnostics.score if track.identity_diagnostics is not None else track.identity_confidence
        motion = track.last_motion
        motion_confidence_score = (
            motion.confidence.score
            if motion is not None and motion.confidence is not None and motion.confidence.score is not None
            else 1.0
        )
        motion_is_stationaryish = motion is None or motion.heading_label in {"stationary", "nearly stationary"}
        recent_heading_flip_total = recent_heading_flip_count(
            [(p.timestamp, p.latitude, p.longitude) for p in track.positions],
            max_steps=4,
        )
        recent_reported_heading_flip_total = _recent_reported_heading_flip_count(track, max_samples=4)
        recent_reported_heading_sequence = _recent_reported_heading_sequence(track, max_samples=4)
        (
            reported_heading_stability_label,
            reported_heading_stability_score,
            reported_heading_stability_reason,
        ) = _classify_reported_heading_stability(track, max_samples=4)
        strong_focus_margin = selection_margin is not None and selection_margin >= 3.0
        reliable_reported_motion = motion_confidence_score >= 0.9 and reported_heading_stability_score >= 0.85
        suppress_raw_heading_flip_penalty = (
            not motion_is_stationaryish
            and structural_event_count >= 4
            and strong_focus_margin
            and reliable_reported_motion
        )
        effective_heading_flip_total = 0 if motion_is_stationaryish or suppress_raw_heading_flip_penalty else recent_heading_flip_total

        recent_focus_switch_count = 1 if previous_focus_track_id is not None and previous_focus_track_id != track.track_id else 0
        score = 1.0
        reason = "stable focus continuity"
        if recent_focus_switch_count:
            score -= 0.35
            reason = "recent focus handoff"
        if structural_event_count >= 6 and reported_heading_stability_score <= 0.2:
            score -= 0.4
            reason = "unstable reported focus heading sequence under structural pressure"
        elif structural_event_count >= 4 and reported_heading_stability_score <= 0.2:
            score -= 0.25
            reason = "unstable reported focus heading sequence"
        elif structural_event_count >= 6 and reported_heading_stability_label == "mixed":
            score -= 0.3
            reason = "mixed reported focus heading sequence under structural pressure"
        elif structural_event_count >= 4 and reported_heading_stability_label == "mixed":
            score -= 0.15
            reason = "mixed reported focus heading sequence under elevated pressure"
        if effective_heading_flip_total >= 2:
            score -= 0.55
            reason = "repeated focus heading reversals"
        elif effective_heading_flip_total == 1:
            score -= 0.35
            reason = "recent focus heading reversal"
        if structural_event_count >= 6:
            score -= 0.3
            reason = "high structural event pressure around focus"
        elif structural_event_count >= 4:
            score -= 0.15
            reason = "elevated structural event pressure around focus"
        if structural_event_count >= 6 and motion_confidence_score < 0.45:
            score -= 0.4
            reason = "high structural pressure with low motion confidence"
        elif structural_event_count >= 4 and motion_confidence_score < 0.45:
            score -= 0.2
            reason = "elevated structural pressure with low motion confidence"
        if identity_score < 0.45:
            score -= 0.25
            reason = "weak focus identity continuity"
        elif identity_score < 0.75:
            score -= 0.1
            reason = "moderate focus identity continuity"
        score += _focus_margin_bonus(selection_margin, structural_event_count)
        score = round(max(0.0, min(1.0, score)), 2)
        if score > 0.6 and suppress_raw_heading_flip_penalty and structural_event_count >= 4:
            reason = "stable focus winner despite structural pressure"
        return FocusContinuity(
            label=_confidence_label(score),
            score=score,
            reason=reason,
            selection_margin=selection_margin,
            runner_up_track_id=runner_up_track_id,
            recent_heading_flip_count=effective_heading_flip_total,
            recent_reported_heading_flip_count=recent_reported_heading_flip_total,
            recent_reported_heading_sequence=recent_reported_heading_sequence,
            reported_heading_stability_label=reported_heading_stability_label,
            reported_heading_stability_score=round(reported_heading_stability_score, 2),
            reported_heading_stability_reason=reported_heading_stability_reason,
            recent_focus_switch_count=recent_focus_switch_count,
            recent_structural_event_count=structural_event_count,
        )

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
        primary_score = self._focus_score(primary, previous_focus_track_id)
        runner_up_track_id = ranked_tracks[1].track_id if len(ranked_tracks) > 1 else None
        selection_margin = (
            round(primary_score - self._focus_score(ranked_tracks[1], previous_focus_track_id), 2)
            if len(ranked_tracks) > 1
            else None
        )
        if previous_focus_track is not None:
            previous_score = self._focus_score(previous_focus_track, previous_focus_track_id)
            challenger_score = primary_score
            if primary.track_id != previous_focus_track.track_id and challenger_score < previous_score + FOCUS_SWITCH_MARGIN:
                primary = previous_focus_track
                primary_score = previous_score
                other_scores = [
                    (self._focus_score(track, previous_focus_track_id), track.track_id)
                    for track in ranked_tracks
                    if track.track_id != primary.track_id
                ]
                if other_scores:
                    best_other_score, runner_up_track_id = max(other_scores)
                    selection_margin = round(primary_score - best_other_score, 2)
                else:
                    runner_up_track_id = None
                    selection_margin = None
        primary.is_primary_focus = True
        self._focus_history.append(primary.track_id)
        if len(self._focus_history) > 6:
            self._focus_history = self._focus_history[-6:]
        structural_event_count = sum(1 for event in self._recent_events if event["event_type"] in {"merge", "split"})
        for track in active_tracks:
            track.focus_continuity = None
        primary.focus_continuity = self._build_focus_continuity(
            primary,
            previous_focus_track_id,
            structural_event_count,
            selection_margin=selection_margin,
            runner_up_track_id=runner_up_track_id,
        )

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
                track.identity_confidence = round(0.45 + (_scan_quality_factor(scan) * 0.35), 2)
                track.identity_diagnostics = self._build_identity_diagnostics(
                    score_value=track.identity_confidence,
                    scan=scan,
                    track=track,
                    reason="first-scan track initialization",
                    event_context="initial",
                )
                self._obj_to_track[obj.object_id] = track.track_id
            self._refresh_track_motions(field_estimates=None, field_dt_hours=0.0)
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
            parent_track.identity_confidence = self._score_confidence(association, primary_new_id, parent_tid, scan, parent_track)
            if parent_track.identity_diagnostics is not None:
                parent_track.identity_diagnostics.event_context = "split_parent"
            new_obj_to_track[primary_new_id] = parent_tid
            child_ids = []
            for new_id in related_new_ids[1:]:
                child = self._create_track(timestamp, new_objects[new_id])
                child.split_from = parent_tid
                self._record_split_lineage(parent_track, child)
                child.identity_confidence = round(0.25 + (_scan_quality_factor(scan) * 0.2), 2)
                child.identity_diagnostics = self._build_identity_diagnostics(
                    score_value=child.identity_confidence,
                    scan=scan,
                    track=child,
                    reason="new split child inherits limited confidence until track history forms",
                    event_context="split_child",
                )
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
                surviving.identity_confidence = self._score_confidence(association, new_id, surviving_track_id, scan, surviving)
                if surviving.identity_diagnostics is not None:
                    surviving.identity_diagnostics.event_context = "merge_survivor"
                new_obj_to_track[new_id] = surviving_track_id
            merged_track_ids = []
            for merged_track_id in related_track_ids[1:]:
                merged_track = self.get_track(merged_track_id)
                if merged_track is not None and merged_track.status == "active":
                    merged_track.status = "merged"
                    merged_track.merged_into = surviving_track_id
                    self._record_merge_lineage(surviving, merged_track)
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
            track.identity_confidence = self._score_confidence(association, new_id, track_id, scan, track)
            new_obj_to_track[new_id] = track_id

        # Create new tracks for unmatched new objects
        for new_id, obj in new_objects.items():
            if new_id not in new_obj_to_track:
                track = self._create_track(timestamp, obj)
                track.identity_confidence = round(0.2 + (_scan_quality_factor(scan) * 0.2), 2)
                track.identity_diagnostics = self._build_identity_diagnostics(
                    score_value=track.identity_confidence,
                    scan=scan,
                    track=track,
                    reason="unmatched object created a new track",
                    event_context="new_track",
                )
                new_obj_to_track[new_id] = track.track_id

        # Increment missed scans for unmatched active tracks
        matched_track_ids = set(new_obj_to_track.values())
        for track in self._tracks:
            if track.status == "active" and track.track_id not in matched_track_ids:
                track._missed_scans += 1
                track.identity_confidence = max(0.0, round(track.identity_confidence - 0.2, 2))
                track.identity_diagnostics = self._build_identity_diagnostics(
                    score_value=track.identity_confidence,
                    scan=scan,
                    track=track,
                    reason="track missed a scan",
                    event_context="missed_scan",
                )
                if track._missed_scans >= MAX_MISSED_SCANS:
                    track.status = "lost"

        self._obj_to_track = new_obj_to_track
        self._refresh_track_motions(field_estimates=association.track_geo_motion, field_dt_hours=association.dt_hours)
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
