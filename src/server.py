# src/server.py
from datetime import datetime, date as date_type
from dataclasses import replace
import math
import os

import numpy as np
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from src.models import (
    RadarSite, ScanMeta, ObjectsResponse, SummaryResponse, RainObject, IntensityLayer,
    TracksResponse, StormTrack, TrackPosition, TrackMotion, TrackEvent, TrackIdentity, TrackFocus,
    TrackDetailResponse, PeakHistoryEntry,
    VelocityResponse, VelocityRegionModel, RotationSignatureModel, RotationHistoryEntryModel,
    MapLocation, StormMapLayerResponse,
)
from src.sites import geocode_city_state, geocode_zipcode, rank_sites, NEXRAD_SITES
from src.ingest import fetch_scan
from src.parser import parse_radar_file, extract_sweep_data, extract_velocity, SweepData, VelocityData
from src.velocity import analyze_velocity, promote_persistent_rotation_assessments
from src.detection import detect_objects_with_grid
from src.preprocess import preprocess_sweep, refresh_quality_advisory
from src.geometry import align_field_by_azimuth
from src.summary import generate_summary
from src.map_layer import (
    build_precipitation_field_geojson,
    build_storm_audiom_geojson,
    build_storm_centroid_geojson,
    build_storm_geojson,
    build_storm_intensity_geojson,
    storm_layer_drawing_info,
    storm_layer_fields,
)
from src.buffer import ReplayBuffer, BufferedScan
from src.radar_page import radar_page_html
from src.tracker import StormTracker

app = FastAPI(title="ARW - Accessible Radar Workstation", version="0.2.0")


def _cors_origins() -> list[str]:
    configured = os.getenv("ARW_CORS_ORIGINS")
    if configured:
        return [origin.strip() for origin in configured.split(",") if origin.strip()]
    return [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001",
        "https://www.audiom.net",
        "https://audiom.net",
        "https://audiom-staging.herokuapp.com",
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_methods=["GET"],
    allow_headers=["*"],
)

# Module-level state for buffer and tracker
_buffer = ReplayBuffer()
_tracker = StormTracker()


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


def _parse_layer_datetime(
    datetime_value: str | None,
    date_value: date_type | None,
    time_value: str | None,
) -> datetime | None:
    if datetime_value is not None:
        return _parse_datetime(datetime_value)
    if date_value is None:
        return None
    if time_value is None:
        return datetime.combine(date_value, datetime.min.time())
    return datetime.fromisoformat(f"{date_value.isoformat()}T{time_value}")


def _json_safe(value):
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _resolve_map_location(
    city: str | None,
    state: str | None,
    zipcode: str | None,
) -> tuple[float, float, str]:
    if zipcode:
        lat, lon = geocode_zipcode(zipcode)
        return lat, lon, zipcode
    if city and state:
        lat, lon = geocode_city_state(city, state)
        return lat, lon, f"{city}, {state}"
    raise HTTPException(
        status_code=422,
        detail="Provide either zipcode or both city and state.",
    )


def _velocity_aligned_to_reflectivity(
    raw_sweep: SweepData, vel_data: VelocityData | None
) -> np.ndarray | None:
    """Remap the lowest velocity sweep onto the reflectivity sweep's azimuths.

    Split-cut VCPs scan reflectivity (surveillance cut) and velocity (Doppler
    cut) on separate antenna revolutions, so the same array index refers to a
    different compass bearing in each sweep. Both arrays happen to share a
    shape, so pairing them by raw index never raises -- it silently compares
    gates that can be tens of kilometres apart at longer range. Only the
    azimuth axis differs between the two cuts; range gates are identical, and
    that assumption is checked here rather than trusted: if a VCP ever
    produces cuts with different ranges_m, a wrong discriminator is worse than
    an absent one (an absent variable is correctly omitted from
    classification scoring, per src/qc/classifier.py), so this returns None
    instead of aligning across mismatched range axes.
    """
    if vel_data is None or not vel_data.sweeps:
        return None
    velocity_sweep = vel_data.sweeps[0]
    if not np.array_equal(
        np.asarray(velocity_sweep.ranges_m, dtype=float),
        np.asarray(raw_sweep.ranges_m, dtype=float),
    ):
        return None
    return align_field_by_azimuth(
        velocity_sweep.velocity, velocity_sweep.azimuths, raw_sweep.azimuths
    )


def _ingest_to_buffer(site_id: str, dt: datetime | None = None) -> BufferedScan:
    """Fetch a scan, quality control it, detect objects, and buffer the result."""
    filepath = fetch_scan(site_id.upper(), dt)
    radar = parse_radar_file(filepath)
    raw_sweep = extract_sweep_data(radar)
    vel_data = extract_velocity(radar)

    # The Doppler cut's rays do not share the surveillance cut's azimuth
    # sampling -- align by nearest azimuth before this reaches classify_gates,
    # which pairs abs_velocity with reflectivity elementwise by index.
    lowest_velocity = _velocity_aligned_to_reflectivity(raw_sweep, vel_data)

    # Raw velocity couplets cannot affect QC.  First classify/despeckle and
    # identify reflectivity cells; velocity analysis then associates evidence
    # to those exact footprints.  QC is advisory-only, so this ordering never
    # removes or hides hazard echo.
    ref_data, scan_quality, echo_advisory = preprocess_sweep(
        raw_sweep, [], velocity=lowest_velocity
    )

    result = detect_objects_with_grid(
        reflectivity=ref_data.reflectivity,
        azimuths=ref_data.azimuths,
        ranges_m=ref_data.ranges_m,
        radar_lat=ref_data.radar_lat,
        radar_lon=ref_data.radar_lon,
        elevation_deg=ref_data.elevation_angle,
        elevations=ref_data.elevations,
        gate_classification=ref_data.gate_classification,
    )
    regions, rotations, annotated_objects = analyze_velocity(
        vel_data, result.objects, result.object_masks, ref_data,
    )
    scan_timestamp = (
        datetime.fromisoformat(ref_data.timestamp)
        if isinstance(ref_data.timestamp, str)
        else ref_data.timestamp
    )
    buffered = BufferedScan(
        timestamp=scan_timestamp,
        site_id=site_id.upper(),
        reflectivity_data=ref_data,
        detected_objects=annotated_objects,
        labeled_grid=result.labeled_grid,
        object_masks=result.object_masks,
        scan_quality=scan_quality,
        velocity_data=vel_data,
        velocity_regions=regions,
        rotation_signatures=rotations,
        echo_advisory=echo_advisory,
    )
    _buffer.add_scan(buffered)
    _tracker.update(buffered)
    histories = {
        obj.object_id: _tracker.rotation_history_for_current_object(obj.object_id)
        for obj in annotated_objects
    }
    rotations = promote_persistent_rotation_assessments(
        rotations, histories, scan_timestamp,
    )
    annotated_objects = [
        replace(
            obj,
            rotation=next(
                (rotation for rotation in rotations
                 if rotation.associated_object_id == obj.object_id),
                None,
            ),
        )
        for obj in annotated_objects
    ]
    ref_data, scan_quality, echo_advisory = refresh_quality_advisory(
        ref_data, scan_quality, rotations, velocity=lowest_velocity,
    )
    buffered.reflectivity_data = ref_data
    buffered.detected_objects = annotated_objects
    buffered.scan_quality = scan_quality
    buffered.rotation_signatures = rotations
    buffered.echo_advisory = echo_advisory
    return buffered


def _motion_to_model(motion) -> TrackMotion:
    confidence = getattr(motion, "confidence", None)
    return TrackMotion(
        speed_kmh=motion.speed_kmh,
        speed_mph=motion.speed_mph,
        heading_deg=motion.heading_deg,
        heading_label=motion.heading_label,
        source=getattr(motion, "source", None),
        confidence_label=confidence.label if confidence is not None else None,
        confidence_score=confidence.score if confidence is not None else None,
        confidence_reason=confidence.reason if confidence is not None else None,
    )


def _track_to_model(track) -> StormTrack:
    """Convert internal Track to Pydantic StormTrack model."""
    motion = track.get_motion()
    identity = getattr(track, "identity_diagnostics", None)
    focus = getattr(track, "focus_continuity", None)
    return StormTrack(
        track_id=track.track_id,
        status=track.status,
        positions=[
            TrackPosition(
                timestamp=p.timestamp.isoformat() if isinstance(p.timestamp, datetime) else p.timestamp,
                latitude=p.latitude,
                longitude=p.longitude,
                distance_km=p.distance_km,
                bearing_deg=p.bearing_deg,
            )
            for p in track.positions
        ],
        motion=_motion_to_model(motion),
        identity=TrackIdentity(
            label=identity.label if identity is not None else None,
            score=identity.score if identity is not None else track.identity_confidence,
            reason=identity.reason if identity is not None else None,
            match_quality=identity.match_quality if identity is not None else None,
            ambiguity_margin=identity.ambiguity_margin if identity is not None else None,
            scan_quality=identity.scan_quality if identity is not None else None,
            missed_scans=identity.missed_scans if identity is not None else None,
            lineage_complexity=identity.lineage_complexity if identity is not None else None,
            event_context=identity.event_context if identity is not None else None,
        ),
        focus=TrackFocus(
            label=focus.label if focus is not None else None,
            score=focus.score if focus is not None else None,
            reason=focus.reason if focus is not None else None,
            selection_margin=focus.selection_margin if focus is not None else None,
            runner_up_track_id=focus.runner_up_track_id if focus is not None else None,
            recent_heading_flip_count=focus.recent_heading_flip_count if focus is not None else None,
            recent_reported_heading_flip_count=focus.recent_reported_heading_flip_count if focus is not None else None,
            recent_reported_heading_sequence=focus.recent_reported_heading_sequence if focus is not None else None,
            reported_heading_stability_label=focus.reported_heading_stability_label if focus is not None else None,
            reported_heading_stability_score=focus.reported_heading_stability_score if focus is not None else None,
            reported_heading_stability_reason=focus.reported_heading_stability_reason if focus is not None else None,
            recent_focus_switch_count=focus.recent_focus_switch_count if focus is not None else None,
            recent_structural_event_count=focus.recent_structural_event_count if focus is not None else None,
        ),
        peak_dbz=track.peak_history[-1].peak_dbz if track.peak_history else 0.0,
        peak_label=track.peak_history[-1].peak_label if track.peak_history else "unknown",
        merged_into=track.merged_into,
        split_from=track.split_from,
        first_seen=track.first_seen.isoformat() if track.first_seen else "",
        last_seen=track.last_seen.isoformat() if track.last_seen else "",
        rotation_history=[
            RotationHistoryEntryModel(
                timestamp=entry.timestamp.isoformat() if isinstance(entry.timestamp, datetime) else entry.timestamp,
                strength=entry.rotation.strength if entry.rotation else None,
                max_shear_ms=entry.rotation.max_shear_ms if entry.rotation else None,
            )
            for entry in getattr(track, 'rotation_history', [])
        ],
    )


@app.get("/")
def root():
    return {"name": "ARW - Accessible Radar Workstation", "version": "0.2.0"}


@app.get("/radar", response_class=HTMLResponse)
def radar_page():
    return HTMLResponse(radar_page_html())


@app.get("/radar/config")
def radar_config():
    return {
        "audiom_map_url": os.getenv("AUDIOM_MAP_URL", "https://www.audiom.net/map"),
        "audiom_rules_path": os.getenv("AUDIOM_STORM_RULES_PATH", "/rules/arw-storms.json"),
        "has_audiom_api_key": bool(os.getenv("AUDIOM_API_KEY", "")),
    }


@app.get("/sites", response_model=list[RadarSite])
def get_sites(city: str = Query(...), state: str = Query(...)):
    lat, lon = geocode_city_state(city, state)
    ranked = rank_sites(lat, lon)
    return [RadarSite(**site) for site in ranked]


@app.get("/scan/{site_id}", response_model=ScanMeta)
def get_scan(site_id: str, datetime: str | None = Query(None)):
    dt = _parse_datetime(datetime)
    buffered = _ingest_to_buffer(site_id, dt)
    return ScanMeta(
        site_id=site_id.upper(),
        timestamp=buffered.reflectivity_data.timestamp,
        elevation_angles=buffered.reflectivity_data.elevation_angles,
    )


@app.get("/objects/{site_id}", response_model=ObjectsResponse)
def get_objects(site_id: str, datetime: str | None = Query(None)):
    dt = _parse_datetime(datetime)
    buffered = _ingest_to_buffer(site_id, dt)
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
            max_inbound_ms=getattr(obj, 'max_inbound_ms', None),
            max_outbound_ms=getattr(obj, 'max_outbound_ms', None),
            rotation_strength=obj.rotation.strength if getattr(obj, 'rotation', None) is not None else None,
            rotation_evidence_level=obj.rotation.evidence_level if getattr(obj, 'rotation', None) is not None else None,
            rotation_motion_reference=obj.rotation.motion_reference if getattr(obj, 'rotation', None) is not None else None,
            class_fractions=getattr(obj, 'class_fractions', {}) or {},
        )
        for obj in buffered.detected_objects
    ]
    return ObjectsResponse(
        site_id=site_id.upper(),
        timestamp=buffered.reflectivity_data.timestamp,
        object_count=len(rain_objects),
        objects=rain_objects,
    )


@app.get("/map/storms", response_model=StormMapLayerResponse)
def get_storm_map_layer(
    city: str | None = Query(None),
    state: str | None = Query(None),
    zipcode: str | None = Query(None),
    datetime: str | None = Query(None),
    date: date_type | None = Query(None),
    time: str | None = Query(None),
):
    try:
        lat, lon, label = _resolve_map_location(city, state, zipcode)
        dt = _parse_layer_datetime(datetime, date, time)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    ranked_sites = rank_sites(lat, lon)
    if not ranked_sites:
        raise HTTPException(status_code=404, detail="No radar site found for that location")

    site = ranked_sites[0]
    buffered = _ingest_to_buffer(site["site_id"], dt)
    geojson = build_storm_geojson(buffered)
    intensity_geojson = build_storm_intensity_geojson(buffered)
    audiom_geojson = build_storm_audiom_geojson(buffered)
    centroid_geojson = build_storm_centroid_geojson(buffered)
    return StormMapLayerResponse(
        layer_name="ARW storm polygons",
        layer_type="FeatureLayer",
        geometryType="esriGeometryPolygon",
        objectIdField="object_id",
        displayFieldName="peak_label",
        spatialReference={"wkid": 4326},
        fields=storm_layer_fields(),
        drawingInfo=storm_layer_drawing_info(),
        site=RadarSite(**site),
        location=MapLocation(latitude=lat, longitude=lon, label=label),
        timestamp=buffered.reflectivity_data.timestamp,
        feature_count=len(geojson["features"]),
        geojson=geojson,
        intensity_geojson=intensity_geojson,
        audiom_geojson=audiom_geojson,
        centroid_geojson=centroid_geojson,
    )


@app.get("/map/storms.geojson")
def get_storm_map_geojson(
    city: str | None = Query(None),
    state: str | None = Query(None),
    zipcode: str | None = Query(None),
    datetime: str | None = Query(None),
    date: date_type | None = Query(None),
    time: str | None = Query(None),
    mode: str = Query("audiom"),
):
    try:
        lat, lon, _label = _resolve_map_location(city, state, zipcode)
        dt = _parse_layer_datetime(datetime, date, time)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    ranked_sites = rank_sites(lat, lon)
    if not ranked_sites:
        raise HTTPException(status_code=404, detail="No radar site found for that location")

    buffered = _ingest_to_buffer(ranked_sites[0]["site_id"], dt)
    if mode == "footprints":
        geojson = build_storm_geojson(buffered)
    elif mode == "intensity":
        geojson = build_storm_intensity_geojson(buffered)
    elif mode == "audiom":
        geojson = build_storm_audiom_geojson(buffered)
    else:
        raise HTTPException(
            status_code=422,
            detail="mode must be one of: audiom, intensity, footprints",
        )
    return JSONResponse(_json_safe(geojson))


@app.get("/map/precipitation.geojson")
def get_precipitation_map_geojson(
    city: str | None = Query(None),
    state: str | None = Query(None),
    zipcode: str | None = Query(None),
    datetime: str | None = Query(None),
    date: date_type | None = Query(None),
    time: str | None = Query(None),
):
    """The whole precipitation field, contoured to 15 dBZ with neither the
    20 dBZ nor the 4 km2 threshold object detection applies -- see
    build_precipitation_field_geojson. Mirrors get_storm_map_geojson's
    location/datetime parameter handling exactly; there is no `mode` here
    since this layer has only one representation.
    """
    try:
        lat, lon, _label = _resolve_map_location(city, state, zipcode)
        dt = _parse_layer_datetime(datetime, date, time)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    ranked_sites = rank_sites(lat, lon)
    if not ranked_sites:
        raise HTTPException(status_code=404, detail="No radar site found for that location")

    buffered = _ingest_to_buffer(ranked_sites[0]["site_id"], dt)
    geojson = build_precipitation_field_geojson(buffered)
    return JSONResponse(_json_safe(geojson))


@app.get("/summary/{site_id}", response_model=SummaryResponse)
def get_summary(site_id: str, datetime: str | None = Query(None)):
    dt = _parse_datetime(datetime)
    buffered = _ingest_to_buffer(site_id, dt)
    site_name = _find_site_name(site_id)
    text = generate_summary(
        site_id=site_id.upper(),
        site_name=site_name,
        timestamp=buffered.reflectivity_data.timestamp,
        objects=buffered.detected_objects,
        tracks=_tracker.active_tracks,
        events=_tracker.recent_events,
    )
    return SummaryResponse(
        site_id=site_id.upper(),
        timestamp=buffered.reflectivity_data.timestamp,
        text=text,
    )


@app.get("/tracks/{site_id}", response_model=TracksResponse)
def get_tracks(site_id: str, datetime: str | None = Query(None)):
    dt = _parse_datetime(datetime)
    buffered = _ingest_to_buffer(site_id, dt)
    active = _tracker.active_tracks
    events = _tracker.recent_events
    return TracksResponse(
        site_id=site_id.upper(),
        timestamp=buffered.reflectivity_data.timestamp,
        active_count=len(active),
        tracks=[_track_to_model(t) for t in active],
        recent_events=[
            TrackEvent(
                event_type=e["event_type"],
                timestamp=e["timestamp"],
                description=e["description"],
                involved_track_ids=e["involved_track_ids"],
            )
            for e in events
        ],
    )


@app.get("/velocity/{site_id}", response_model=VelocityResponse)
def get_velocity(site_id: str, datetime: str | None = Query(None)):
    dt = _parse_datetime(datetime)
    buffered = _ingest_to_buffer(site_id, dt)
    regions = [
        VelocityRegionModel(
            region_type=r.region_type,
            peak_velocity_ms=r.peak_velocity_ms,
            mean_velocity_ms=r.mean_velocity_ms,
            area_km2=r.area_km2,
            centroid_lat=r.centroid_lat,
            centroid_lon=r.centroid_lon,
            distance_km=r.distance_km,
            bearing_deg=r.bearing_deg,
            sweep_count=r.sweep_count,
            elevation_angles=r.elevation_angles,
        )
        for r in buffered.velocity_regions
    ]
    rotation_sigs = [
        RotationSignatureModel(
            centroid_lat=s.centroid_lat,
            centroid_lon=s.centroid_lon,
            distance_km=s.distance_km,
            bearing_deg=s.bearing_deg,
            max_shear_ms=s.max_shear_ms,
            max_inbound_ms=s.max_inbound_ms,
            max_outbound_ms=s.max_outbound_ms,
            diameter_km=s.diameter_km,
            sweep_count=s.sweep_count,
            elevation_angles=s.elevation_angles,
            strength=s.strength,
            associated_object_id=s.associated_object_id,
            evidence_level=s.evidence_level,
            motion_reference=s.motion_reference,
        )
        for s in buffered.rotation_signatures
    ]
    return VelocityResponse(
        site_id=site_id.upper(),
        timestamp=buffered.reflectivity_data.timestamp,
        regions=regions,
        rotation_signatures=rotation_sigs,
    )


@app.get("/motion/{site_id}/{track_id}", response_model=TrackDetailResponse)
def get_motion(site_id: str, track_id: int):
    track = _tracker.get_track(track_id)
    if track is None:
        raise HTTPException(status_code=404, detail=f"Track {track_id} not found")
    motion = track.get_motion()
    identity = getattr(track, "identity_diagnostics", None)
    focus = getattr(track, "focus_continuity", None)
    return TrackDetailResponse(
        track_id=track.track_id,
        status=track.status,
        positions=[
            TrackPosition(
                timestamp=p.timestamp.isoformat() if isinstance(p.timestamp, datetime) else p.timestamp,
                latitude=p.latitude,
                longitude=p.longitude,
                distance_km=p.distance_km,
                bearing_deg=p.bearing_deg,
            )
            for p in track.positions
        ],
        motion=_motion_to_model(motion),
        identity=TrackIdentity(
            label=identity.label if identity is not None else None,
            score=identity.score if identity is not None else track.identity_confidence,
            reason=identity.reason if identity is not None else None,
            match_quality=identity.match_quality if identity is not None else None,
            ambiguity_margin=identity.ambiguity_margin if identity is not None else None,
            scan_quality=identity.scan_quality if identity is not None else None,
            missed_scans=identity.missed_scans if identity is not None else None,
            lineage_complexity=identity.lineage_complexity if identity is not None else None,
            event_context=identity.event_context if identity is not None else None,
        ),
        focus=TrackFocus(
            label=focus.label if focus is not None else None,
            score=focus.score if focus is not None else None,
            reason=focus.reason if focus is not None else None,
            selection_margin=focus.selection_margin if focus is not None else None,
            runner_up_track_id=focus.runner_up_track_id if focus is not None else None,
            recent_heading_flip_count=focus.recent_heading_flip_count if focus is not None else None,
            recent_reported_heading_flip_count=focus.recent_reported_heading_flip_count if focus is not None else None,
            recent_reported_heading_sequence=focus.recent_reported_heading_sequence if focus is not None else None,
            reported_heading_stability_label=focus.reported_heading_stability_label if focus is not None else None,
            reported_heading_stability_score=focus.reported_heading_stability_score if focus is not None else None,
            reported_heading_stability_reason=focus.reported_heading_stability_reason if focus is not None else None,
            recent_focus_switch_count=focus.recent_focus_switch_count if focus is not None else None,
            recent_structural_event_count=focus.recent_structural_event_count if focus is not None else None,
        ),
        peak_history=[
            PeakHistoryEntry(
                timestamp=p.timestamp.isoformat() if isinstance(p.timestamp, datetime) else p.timestamp,
                peak_dbz=p.peak_dbz,
                peak_label=p.peak_label,
            )
            for p in track.peak_history
        ],
        merged_into=track.merged_into,
        split_from=track.split_from,
        first_seen=track.first_seen.isoformat() if track.first_seen else "",
        last_seen=track.last_seen.isoformat() if track.last_seen else "",
    )
