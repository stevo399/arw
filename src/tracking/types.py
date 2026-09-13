from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from src.detection import DetectedObject
from src.velocity import RotationSignature


# Reflectivity at or above this marks a storm's intense core (the
# "intense precipitation" and "severe core" intensity bands).
CORE_MIN_DBZ = 50.0
MAX_TREND_SAMPLES = 6
# Reported motion is the median of at most this many measured velocities.
MAX_MEASURED_VELOCITIES = 3


@dataclass
class VelocitySample:
    """A storm's velocity measured by pattern matching against the previous scan."""

    timestamp: datetime
    east_kmh: float
    north_kmh: float


@dataclass
class TrendSample:
    timestamp: datetime
    area_km2: float
    core_area_km2: float
    peak_dbz: float
    # True when this sample came from a reacquired match; trends spanning it
    # are not reported, since the storm was unobserved for at least a scan.
    reacquired: bool = False


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
class AssociationScore:
    track_id: int
    object_id: int
    overlap_score: float
    advected_overlap_score: float
    distance_score: float
    predicted_position_score: float
    area_change_score: float
    intensity_change_score: float
    total_cost: float


@dataclass
class SegmentedStormObject:
    object_id: int
    detected_object: DetectedObject
    mask: object
    bbox: tuple[int, int, int, int]
    pixel_count: int
    threshold_parent_id: int | None = None
    threshold_level: float | None = None
    threshold_path: tuple[float, ...] = field(default_factory=tuple)


@dataclass
class MotionConfidence:
    label: str
    score: float
    reason: str | None = None


@dataclass
class MotionSample:
    timestamp: datetime
    heading_deg: float | None
    heading_label: str
    source: str
    confidence_score: float | None = None


@dataclass
class IdentityConfidence:
    label: str
    score: float
    reason: str | None = None
    match_quality: float | None = None
    ambiguity_margin: float | None = None
    scan_quality: float | None = None
    missed_scans: int = 0
    lineage_complexity: int = 0
    event_context: str | None = None


@dataclass
class FocusContinuity:
    label: str
    score: float
    reason: str | None = None
    selection_margin: float | None = None
    runner_up_track_id: int | None = None
    recent_heading_flip_count: int = 0
    recent_reported_heading_flip_count: int = 0
    recent_reported_heading_sequence: list[str] = field(default_factory=list)
    reported_heading_stability_label: str | None = None
    reported_heading_stability_score: float | None = None
    reported_heading_stability_reason: str | None = None
    recent_focus_switch_count: int = 0
    recent_structural_event_count: int = 0


@dataclass
class RotationHistoryEntry:
    timestamp: datetime
    rotation: RotationSignature | None


@dataclass
class Track:
    track_id: int
    status: str  # "active", "missing", "merged", "split", "lost"
    positions: list[TrackPosition] = field(default_factory=list)
    peak_history: list[PeakEntry] = field(default_factory=list)
    current_object: DetectedObject | None = None
    merged_into: int | None = None
    split_from: int | None = None
    parent_track_ids: list[int] = field(default_factory=list)
    child_track_ids: list[int] = field(default_factory=list)
    absorbed_track_ids: list[int] = field(default_factory=list)
    first_seen: datetime | None = None
    last_seen: datetime | None = None
    identity_confidence: float = 1.0
    identity_diagnostics: IdentityConfidence | None = None
    focus_continuity: FocusContinuity | None = None
    motion_confidence: MotionConfidence | None = None
    last_motion: Any | None = None
    motion_history: list[MotionSample] = field(default_factory=list)
    rotation_history: list[RotationHistoryEntry] = field(default_factory=list)
    trend_samples: list[TrendSample] = field(default_factory=list)
    measured_velocities: list[VelocitySample] = field(default_factory=list)
    is_primary_focus: bool = False
    # (scan timestamp, object id) of the last scan this track was matched in.
    # Reacquisition rebuilds the track's mask from that scan.
    last_seen_ref: tuple[datetime, int] | None = None
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
        self.last_seen_ref = (timestamp, obj.object_id)
        self.trend_samples.append(TrendSample(
            timestamp=timestamp,
            area_km2=obj.area_km2,
            core_area_km2=sum(layer.area_km2 for layer in obj.layers if layer.min_dbz >= CORE_MIN_DBZ),
            peak_dbz=obj.peak_dbz,
        ))
        if len(self.trend_samples) > MAX_TREND_SAMPLES:
            self.trend_samples = self.trend_samples[-MAX_TREND_SAMPLES:]
        self._missed_scans = 0
        if self.first_seen is None:
            self.first_seen = timestamp
