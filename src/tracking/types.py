from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from src.detection import DetectedObject


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


@dataclass
class MotionConfidence:
    label: str
    score: float
    reason: str | None = None


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
    identity_confidence: float = 1.0
    motion_confidence: MotionConfidence | None = None
    last_motion: Any | None = None
    diagnostic_motion: Any | None = None
    is_primary_focus: bool = False
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
