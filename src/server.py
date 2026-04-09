from datetime import datetime
from fastapi import FastAPI, Query
from src.models import RadarSite, ScanMeta, ObjectsResponse, SummaryResponse, RainObject, IntensityLayer
from src.sites import geocode_city_state, rank_sites, NEXRAD_SITES
from src.ingest import fetch_scan
from src.parser import extract_reflectivity
from src.detection import detect_objects
from src.summary import generate_summary

app = FastAPI(title="ARW - Accessible Radar Workstation", version="0.1.0")


def _find_site_name(site_id: str) -> str:
    """Look up the display name for a NEXRAD site."""
    for site in NEXRAD_SITES:
        if site["site_id"] == site_id.upper():
            return site["name"]
    return site_id


def _parse_datetime(dt_str: str | None) -> datetime | None:
    """Parse an optional datetime query parameter."""
    if dt_str is None:
        return None
    return datetime.fromisoformat(dt_str)


@app.get("/")
def root():
    return {"name": "ARW - Accessible Radar Workstation", "version": "0.1.0"}


@app.get("/sites", response_model=list[RadarSite])
def get_sites(city: str = Query(...), state: str = Query(...)):
    lat, lon = geocode_city_state(city, state)
    ranked = rank_sites(lat, lon)
    return [RadarSite(**site) for site in ranked]


@app.get("/scan/{site_id}", response_model=ScanMeta)
def get_scan(site_id: str, datetime: str | None = Query(None)):
    dt = _parse_datetime(datetime)
    filepath = fetch_scan(site_id.upper(), dt)
    ref_data = extract_reflectivity(filepath)
    return ScanMeta(
        site_id=site_id.upper(),
        timestamp=ref_data.timestamp,
        elevation_angles=ref_data.elevation_angles,
    )


@app.get("/objects/{site_id}", response_model=ObjectsResponse)
def get_objects(site_id: str, datetime: str | None = Query(None)):
    dt = _parse_datetime(datetime)
    filepath = fetch_scan(site_id.upper(), dt)
    ref_data = extract_reflectivity(filepath)
    detected = detect_objects(
        reflectivity=ref_data.reflectivity,
        azimuths=ref_data.azimuths,
        ranges_m=ref_data.ranges_m,
        radar_lat=ref_data.radar_lat,
        radar_lon=ref_data.radar_lon,
    )
    rain_objects = [
        RainObject(
            object_id=obj.object_id,
            centroid_lat=obj.centroid_lat,
            centroid_lon=obj.centroid_lon,
            distance_km=obj.distance_km,
            bearing_deg=obj.bearing_deg,
            peak_dbz=obj.peak_dbz,
            peak_label=obj.peak_label,
            area_km2=obj.area_km2,
            layers=[
                IntensityLayer(
                    label=layer.label,
                    min_dbz=layer.min_dbz,
                    max_dbz=layer.max_dbz,
                    area_km2=layer.area_km2,
                )
                for layer in obj.layers
            ],
        )
        for obj in detected
    ]
    return ObjectsResponse(
        site_id=site_id.upper(),
        timestamp=ref_data.timestamp,
        object_count=len(rain_objects),
        objects=rain_objects,
    )


@app.get("/summary/{site_id}", response_model=SummaryResponse)
def get_summary(site_id: str, datetime: str | None = Query(None)):
    dt = _parse_datetime(datetime)
    filepath = fetch_scan(site_id.upper(), dt)
    ref_data = extract_reflectivity(filepath)
    detected = detect_objects(
        reflectivity=ref_data.reflectivity,
        azimuths=ref_data.azimuths,
        ranges_m=ref_data.ranges_m,
        radar_lat=ref_data.radar_lat,
        radar_lon=ref_data.radar_lon,
    )
    site_name = _find_site_name(site_id)
    text = generate_summary(
        site_id=site_id.upper(),
        site_name=site_name,
        timestamp=ref_data.timestamp,
        objects=detected,
    )
    return SummaryResponse(
        site_id=site_id.upper(),
        timestamp=ref_data.timestamp,
        text=text,
    )
