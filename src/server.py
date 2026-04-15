# src/server.py
from datetime import datetime
from fastapi import FastAPI, Query, HTTPException
from src.models import (
    RadarSite, ScanMeta, ObjectsResponse, SummaryResponse, RainObject, IntensityLayer,
    TracksResponse, StormTrack, TrackPosition, TrackMotion, TrackEvent, TrackIdentity, TrackFocus,
    TrackDetailResponse, PeakHistoryEntry,
)
from src.sites import geocode_city_state, rank_sites, NEXRAD_SITES
from src.ingest import fetch_scan
from src.parser import extract_reflectivity
from src.detection import detect_objects_with_grid
from src.preprocess import preprocess_reflectivity_data
from src.summary import generate_summary
from src.buffer import ReplayBuffer, BufferedScan
from src.tracker import StormTracker

app = FastAPI(title="ARW - Accessible Radar Workstation", version="0.2.0")

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


def _ingest_to_buffer(site_id: str, dt: datetime | None = None) -> BufferedScan:
    """Fetch a scan, detect objects, and add to buffer + tracker."""
    filepath = fetch_scan(site_id.upper(), dt)
    raw_ref_data = extract_reflectivity(filepath)
    ref_data, scan_quality = preprocess_reflectivity_data(raw_ref_data)
    result = detect_objects_with_grid(
        reflectivity=ref_data.reflectivity,
        azimuths=ref_data.azimuths,
        ranges_m=ref_data.ranges_m,
        radar_lat=ref_data.radar_lat,
        radar_lon=ref_data.radar_lon,
    )
    scan_timestamp = datetime.fromisoformat(ref_data.timestamp) if isinstance(ref_data.timestamp, str) else ref_data.timestamp
    buffered = BufferedScan(
        timestamp=scan_timestamp,
        site_id=site_id.upper(),
        reflectivity_data=ref_data,
        detected_objects=result.objects,
        labeled_grid=result.labeled_grid,
        object_masks=result.object_masks,
        scan_quality=scan_quality,
    )
    _buffer.add_scan(buffered)
    _tracker.update(buffered)
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
            recent_focus_switch_count=focus.recent_focus_switch_count if focus is not None else None,
            recent_structural_event_count=focus.recent_structural_event_count if focus is not None else None,
        ),
        peak_dbz=track.peak_history[-1].peak_dbz if track.peak_history else 0.0,
        peak_label=track.peak_history[-1].peak_label if track.peak_history else "unknown",
        merged_into=track.merged_into,
        split_from=track.split_from,
        first_seen=track.first_seen.isoformat() if track.first_seen else "",
        last_seen=track.last_seen.isoformat() if track.last_seen else "",
    )


@app.get("/")
def root():
    return {"name": "ARW - Accessible Radar Workstation", "version": "0.2.0"}


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
        )
        for obj in buffered.detected_objects
    ]
    return ObjectsResponse(
        site_id=site_id.upper(),
        timestamp=buffered.reflectivity_data.timestamp,
        object_count=len(rain_objects),
        objects=rain_objects,
    )


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
