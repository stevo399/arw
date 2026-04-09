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
