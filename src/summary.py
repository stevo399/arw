# src/summary.py
import math

from src.detection import DetectedObject, degrees_to_bearing
from src.motion import MotionVector

KM_PER_MILE = 1.60934


def km_to_miles(km: float) -> int:
    """Convert kilometers to miles, rounded to nearest whole number."""
    return round(km / KM_PER_MILE)


def km2_to_mi2(km2: float) -> int:
    """Convert square kilometers to square miles, rounded to nearest whole number."""
    return round(km2 / (KM_PER_MILE ** 2))


def _get_motion_for_object(obj: DetectedObject, tracks) -> MotionVector | None:
    """Find the motion vector for a detected object by matching to its track."""
    if tracks is None:
        return None
    for track in tracks:
        if track.current_object is not None and track.current_object.object_id == obj.object_id:
            if hasattr(track, '_motion_override'):
                return track._motion_override
            return track.get_motion()
    return None


def _get_track_for_object(obj: DetectedObject, tracks):
    """Find the active track currently associated with a detected object."""
    if tracks is None:
        return None
    for track in tracks:
        if track.current_object is not None and track.current_object.object_id == obj.object_id:
            return track
    return None


def _summary_strength_score(obj: DetectedObject, track) -> float:
    """Prefer intense storms, but damp tiny-core jitter with area and track continuity."""
    area_bonus = min(math.log1p(max(obj.area_km2, 0.0)), 6.0)
    track_bonus = 0.0
    if track is not None:
        history_bonus = min(len(track.positions), 6) * 0.5
        confidence_bonus = max(min(track.identity_confidence, 1.0), 0.0)
        primary_focus_bonus = 3.0 if getattr(track, "is_primary_focus", False) else 0.0
        track_bonus = history_bonus + confidence_bonus + primary_focus_bonus
    return obj.peak_dbz + area_bonus + track_bonus


def _pick_summary_object(objects: list[DetectedObject], tracks) -> DetectedObject:
    """Choose a summary focal object with less scan-to-scan jitter than raw peak ordering."""
    if tracks is not None:
        for track in tracks:
            if getattr(track, "is_primary_focus", False) and track.current_object is not None:
                return track.current_object
    return max(
        objects,
        key=lambda obj: (
            _summary_strength_score(obj, _get_track_for_object(obj, tracks)),
            obj.peak_dbz,
            obj.area_km2,
            -obj.distance_km,
        ),
    )


def _format_motion(motion: MotionVector | None) -> str:
    """Format motion info for speech."""
    if motion is None:
        return ""
    if motion.heading_label == "uncertain":
        return ", tracking uncertain"
    if motion.heading_label == "stationary":
        return ", stationary"
    if motion.heading_label == "nearly stationary":
        return ", nearly stationary"
    return f", moving {motion.heading_label} at {motion.speed_mph} mph"


def generate_summary(
    site_id: str,
    site_name: str,
    timestamp: str,
    objects: list[DetectedObject],
    tracks=None,
    events: list[dict] | None = None,
) -> str:
    """Generate a speech-ready text summary of detected rain objects.

    Args:
        site_id: Radar site ID.
        site_name: Radar site display name.
        timestamp: Scan timestamp.
        objects: Detected objects sorted by peak_dbz descending.
        tracks: Optional list of Track objects for motion info.
        events: Optional list of recent merge/split event dicts.
    """
    if not objects:
        return f"{site_name}: No significant precipitation detected."

    count = len(objects)
    obj_word = "rain object" if count == 1 else "rain objects"
    strongest = _pick_summary_object(objects, tracks)
    distance_mi = km_to_miles(strongest.distance_km)
    bearing = degrees_to_bearing(strongest.bearing_deg)
    area_mi2 = km2_to_mi2(sum(obj.area_km2 for obj in objects))

    motion = _get_motion_for_object(strongest, tracks)
    motion_str = _format_motion(motion)

    parts = [
        f"{site_name}: {count} {obj_word} detected. "
        f"Strongest: {strongest.peak_label}, "
        f"{distance_mi} miles {bearing} of the radar{motion_str}."
    ]

    # Add merge/split events
    if events:
        merge_count = sum(1 for e in events if e["event_type"] == "merge")
        split_count = sum(1 for e in events if e["event_type"] == "split")
        if merge_count > 0:
            storms_word = "storm" if merge_count == 1 else "storms"
            parts.append(f" Note: {merge_count} {storms_word} merged in the last scan.")
        if split_count > 0:
            storms_word = "storm" if split_count == 1 else "storms"
            parts.append(f" Note: {split_count} {storms_word} split in the last scan.")

    parts.append(f" Covering approximately {area_mi2} square miles.")

    return "".join(parts)
