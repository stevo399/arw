from pydantic import BaseModel


class RadarSite(BaseModel):
    site_id: str
    name: str
    latitude: float
    longitude: float
    elevation_m: float
    distance_km: float
    beam_height_m: float


class ScanMeta(BaseModel):
    site_id: str
    timestamp: str
    elevation_angles: list[float]


class IntensityLayer(BaseModel):
    label: str
    min_dbz: float
    max_dbz: float
    area_km2: float


class RainObject(BaseModel):
    object_id: int
    centroid_lat: float
    centroid_lon: float
    distance_km: float
    bearing_deg: float
    peak_dbz: float
    peak_label: str
    area_km2: float
    layers: list[IntensityLayer]
    max_inbound_ms: float | None = None
    max_outbound_ms: float | None = None
    rotation_strength: str | None = None


class VelocityRegionModel(BaseModel):
    region_type: str
    peak_velocity_ms: float
    mean_velocity_ms: float
    area_km2: float
    centroid_lat: float
    centroid_lon: float
    distance_km: float
    bearing_deg: float
    sweep_count: int
    elevation_angles: list[float]


class RotationSignatureModel(BaseModel):
    centroid_lat: float
    centroid_lon: float
    distance_km: float
    bearing_deg: float
    max_shear_ms: float
    max_inbound_ms: float
    max_outbound_ms: float
    diameter_km: float
    sweep_count: int
    elevation_angles: list[float]
    strength: str
    associated_object_id: int | None = None


class VelocityResponse(BaseModel):
    site_id: str
    timestamp: str
    regions: list[VelocityRegionModel]
    rotation_signatures: list[RotationSignatureModel]


class RotationHistoryEntryModel(BaseModel):
    timestamp: str
    strength: str | None = None
    max_shear_ms: float | None = None


class ObjectsResponse(BaseModel):
    site_id: str
    timestamp: str
    object_count: int
    objects: list[RainObject]


class SummaryResponse(BaseModel):
    site_id: str
    timestamp: str
    text: str


class TrackPosition(BaseModel):
    timestamp: str
    latitude: float
    longitude: float
    distance_km: float
    bearing_deg: float


class TrackMotion(BaseModel):
    speed_kmh: float
    speed_mph: int
    heading_deg: float | None
    heading_label: str
    source: str | None = None
    confidence_label: str | None = None
    confidence_score: float | None = None
    confidence_reason: str | None = None


class TrackIdentity(BaseModel):
    label: str | None = None
    score: float | None = None
    reason: str | None = None
    match_quality: float | None = None
    ambiguity_margin: float | None = None
    scan_quality: float | None = None
    missed_scans: int | None = None
    lineage_complexity: int | None = None
    event_context: str | None = None


class TrackFocus(BaseModel):
    label: str | None = None
    score: float | None = None
    reason: str | None = None
    selection_margin: float | None = None
    runner_up_track_id: int | None = None
    recent_heading_flip_count: int | None = None
    recent_reported_heading_flip_count: int | None = None
    recent_reported_heading_sequence: list[str] | None = None
    reported_heading_stability_label: str | None = None
    reported_heading_stability_score: float | None = None
    reported_heading_stability_reason: str | None = None
    recent_focus_switch_count: int | None = None
    recent_structural_event_count: int | None = None


class PeakHistoryEntry(BaseModel):
    timestamp: str
    peak_dbz: float
    peak_label: str


class StormTrack(BaseModel):
    track_id: int
    status: str
    positions: list[TrackPosition]
    motion: TrackMotion
    identity: TrackIdentity | None = None
    focus: TrackFocus | None = None
    peak_dbz: float
    peak_label: str
    merged_into: int | None
    split_from: int | None
    first_seen: str
    last_seen: str
    rotation_history: list[RotationHistoryEntryModel] = []


class TrackEvent(BaseModel):
    event_type: str
    timestamp: str
    description: str
    involved_track_ids: list[int]


class TracksResponse(BaseModel):
    site_id: str
    timestamp: str
    active_count: int
    tracks: list[StormTrack]
    recent_events: list[TrackEvent]


class TrackDetailResponse(BaseModel):
    track_id: int
    status: str
    positions: list[TrackPosition]
    motion: TrackMotion
    identity: TrackIdentity | None = None
    focus: TrackFocus | None = None
    peak_history: list[PeakHistoryEntry]
    merged_into: int | None
    split_from: int | None
    first_seen: str
    last_seen: str
