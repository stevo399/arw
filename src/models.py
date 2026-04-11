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


class PeakHistoryEntry(BaseModel):
    timestamp: str
    peak_dbz: float
    peak_label: str


class StormTrack(BaseModel):
    track_id: int
    status: str
    positions: list[TrackPosition]
    motion: TrackMotion
    peak_dbz: float
    peak_label: str
    merged_into: int | None
    split_from: int | None
    first_seen: str
    last_seen: str


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
    peak_history: list[PeakHistoryEntry]
    merged_into: int | None
    split_from: int | None
    first_seen: str
    last_seen: str
