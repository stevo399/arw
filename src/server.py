# src/server.py
from datetime import datetime, date as date_type, timedelta
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
import logging
import math
import os
from pathlib import Path
from threading import RLock, Timer

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
from src.detection import detect_objects_with_grid
from src.preprocess import preprocess_sweep, refresh_quality_advisory
from src.geometry import align_field_by_azimuth
from src.summary import generate_summary
from src.map_layer import (
    build_precipitation_field_geojson,
    build_storm_audiom_centroid_geojson,
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

# A Level II volume is expensive to turn into an accessibility-ready map: it
# must be parsed, quality-controlled, segmented, and paired with velocity.
# Keep that completed work in memory.  Raw files in ``cache/`` only avoid a
# download; without this cache every visitor still repeated all of that work.
_processed_scans: dict[tuple[str, str], BufferedScan] = {}
_map_layers: dict[tuple[str, str], dict[str, dict]] = {}
_live_scans: dict[str, BufferedScan] = {}
_refreshing_sites: set[str] = set()
_refresh_started_at: dict[str, datetime] = {}
_refresh_errors: dict[str, str] = {}
_state_lock = RLock()
_ingest_lock = RLock()
# Rendering an already-completed immutable GeoJSON layer must never wait for a
# different site's slow Level II ingest/tracker update.
_map_build_lock = RLock()
_refresh_interval_seconds = int(os.getenv("ARW_REFRESH_INTERVAL_SECONDS", "120"))
_refresh_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="arw-radar")
_prewarm_sites = tuple(
    site.strip().upper()
    for site in os.getenv("ARW_PREWARM_SITES", "").split(",")
    if site.strip()
)
_refresh_timers: dict[str, Timer] = {}
_background_refresh_enabled = False
_logger = logging.getLogger(__name__)

# A location-led radar request should describe conditions near that location,
# not every small echo in the selected radar's full coverage. Clients can
# deliberately widen this, but the default keeps an accessible map from
# announcing an unrelated echo hundreds of miles away.
DEFAULT_MAP_RELEVANCE_RADIUS_MILES = 75.0
ELEVATED_BEAM_HEIGHT_KM = 3.0
LIVE_REFRESH_FAILURE_RETRY_SECONDS = 5
# A split-cut Level II volume normally places the surveillance reflectivity
# cut and its companion Doppler cut among the first several scans.  ARW's
# current-map pipeline selects its reflectivity sweep and at most three
# low-level velocity sweeps, so reading farther elevations only delays the
# first accessible map and greatly expands memory use.
LIVE_MAP_SCAN_INDICES = list(range(6))
# These are the only moments consumed by the rapid current-map pipeline:
# reflectivity for objects, rhohv/zdr for dual-polarization QC, and velocity
# for base-radial context.  Excluding unused archive moments prevents a
# 10 MB compressed Level II volume becoming several GB in memory.
LIVE_MAP_FIELDS = [
    "reflectivity",
    "velocity",
    "cross_correlation_ratio",
    "differential_reflectivity",
]


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
    latitude: float | None,
    longitude: float | None,
) -> tuple[float, float, str]:
    if latitude is not None or longitude is not None:
        if latitude is None or longitude is None:
            raise HTTPException(
                status_code=422,
                detail="Provide both latitude and longitude.",
            )
        if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
            raise HTTPException(status_code=422, detail="Latitude or longitude is out of range.")
        label = f"{city}, {state}" if city and state else f"{latitude:.4f}, {longitude:.4f}"
        return latitude, longitude, label
    if zipcode:
        lat, lon = geocode_zipcode(zipcode)
        return lat, lon, zipcode
    if city and state:
        lat, lon = geocode_city_state(city, state)
        return lat, lon, f"{city}, {state}"
    raise HTTPException(
        status_code=422,
        detail="Provide latitude and longitude, a zipcode, or both city and state.",
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


def _process_scan_file(site_id: str, filepath: str) -> BufferedScan:
    """Run the full analysis pipeline for one already-selected volume."""
    radar = parse_radar_file(
        filepath,
        scans=LIVE_MAP_SCAN_INDICES,
        include_fields=LIVE_MAP_FIELDS,
    )
    raw_sweep = extract_sweep_data(radar)
    # Py-ART's region-based dealiaser expands the *entire* Level II volume,
    # although the interactive map needs only three low-level Doppler cuts.
    # On recent full-resolution volumes that one operation consumed several
    # gigabytes and kept a first map in "Preparing" for minutes.  Use the
    # observed base-radial velocities for the live map so reflectivity and QC
    # can become available promptly; a later analysis path can request
    # dealiased velocity when its extra precision is material.
    vel_data = extract_velocity(radar, dealias=False)

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
    # Velocity-region and rotational-couplet detection is a separate,
    # high-cost diagnostic product.  Do not hold the first reflectivity map
    # hostage to it: the live snapshot has already used the nearest Doppler
    # cut for QC, while these advanced velocity diagnostics can be added by a
    # dedicated analysis request.  This avoids multi-minute cold-map waits on
    # high-resolution volumes without pretending a rotation assessment exists.
    regions: list = []
    rotations: list = []
    annotated_objects = result.objects
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
        source_path=str(Path(filepath).resolve()),
        velocity_data=vel_data,
        velocity_regions=regions,
        rotation_signatures=rotations,
        echo_advisory=echo_advisory,
    )
    _buffer.add_scan(buffered)
    _tracker.update(buffered)
    for obj in annotated_objects:
        obj.temporal_status = _tracker.temporal_status_for_current_object(obj.object_id)
    # Keep the raw, aligned velocity evidence in QC only.  `velocity_data`
    # would retain three large Doppler arrays after that use, so do not pin it
    # in the completed live-map cache.
    vel_data = None
    ref_data, scan_quality, echo_advisory = refresh_quality_advisory(
        ref_data, scan_quality, rotations, velocity=lowest_velocity,
    )
    buffered.reflectivity_data = ref_data
    buffered.detected_objects = annotated_objects
    buffered.scan_quality = scan_quality
    buffered.rotation_signatures = rotations
    buffered.echo_advisory = echo_advisory
    return buffered


def _map_layer_cache_key(buffered: BufferedScan) -> tuple[str, str] | None:
    source_path = getattr(buffered, "source_path", None)
    if not source_path or not Path(source_path).is_file():
        # Synthetic scans and test doubles are deliberately not shared.
        return None
    return buffered.site_id.upper(), source_path


def _prepared_map_layers(
    buffered: BufferedScan, requested_layers: set[str] | None = None
) -> dict[str, dict]:
    """Build requested public map representations once per completed scan.

    The lightweight Audiom source is the only layer needed to make a live map
    usable.  Do not also contour the detailed footprints, intensity bands, and
    centroids until an endpoint actually asks for them.
    """
    requested_layers = requested_layers or {"footprints", "intensity", "audiom", "centroids"}
    cache_key = _map_layer_cache_key(buffered)
    cached: dict[str, dict] | None = None
    if cache_key is not None:
        with _state_lock:
            cached = _map_layers.get(cache_key)
        if cached is not None and requested_layers.issubset(cached):
            return {name: cached[name] for name in requested_layers}

    # Only one caller builds uncached contour layers.  This is intentionally
    # separate from _ingest_lock: completed map layers are immutable and safe
    # to serve while another site's radar volume is being parsed and tracked.
    with _map_build_lock:
        if cache_key is not None:
            with _state_lock:
                cached = _map_layers.get(cache_key)
            if cached is not None and requested_layers.issubset(cached):
                return {name: cached[name] for name in requested_layers}
        layers = dict(cached or {})
        if "footprints" in requested_layers and "footprints" not in layers:
            layers["footprints"] = build_storm_geojson(buffered)
        if "intensity" in requested_layers and "intensity" not in layers:
            layers["intensity"] = build_storm_intensity_geojson(buffered)
        if "audiom" in requested_layers and "audiom" not in layers:
            layers["audiom"] = build_storm_audiom_centroid_geojson(buffered)
        if "centroids" in requested_layers and "centroids" not in layers:
            layers["centroids"] = build_storm_centroid_geojson(buffered)
        if cache_key is not None:
            with _state_lock:
                _map_layers[cache_key] = layers
        return {name: layers[name] for name in requested_layers}


def _prepared_precipitation_layer(buffered: BufferedScan) -> dict:
    """Build the much larger whole-field layer only when it is requested."""
    cache_key = _map_layer_cache_key(buffered)
    if cache_key is not None:
        with _state_lock:
            cached = _map_layers.get(cache_key, {}).get("precipitation")
        if cached is not None:
            return cached

    with _map_build_lock:
        if cache_key is not None:
            with _state_lock:
                cached = _map_layers.get(cache_key, {}).get("precipitation")
            if cached is not None:
                return cached
        precipitation = build_precipitation_field_geojson(buffered)
        if cache_key is not None:
            with _state_lock:
                _map_layers.setdefault(cache_key, {})["precipitation"] = precipitation
        return precipitation


def _ingest_to_buffer(
    site_id: str, dt: datetime | None = None, *, publish_live: bool = True,
) -> BufferedScan:
    """Return a completed interpretation, analyzing each local volume once.

    The lock also protects the shared replay buffer and storm tracker.  It is
    deliberately held while selecting the scan so two simultaneous map
    requests cannot both choose, parse, and track the same new volume.
    """
    normalized_site = site_id.upper()
    with _ingest_lock:
        filepath = fetch_scan(normalized_site, dt)
        cache_key = (normalized_site, str(Path(filepath).resolve()))
        # Test doubles do not point to files.  Requiring a real file here
        # prevents one mocked scan from leaking into a later test/request.
        if Path(filepath).is_file():
            with _state_lock:
                cached = _processed_scans.get(cache_key)
            if cached is not None:
                if dt is None:
                    with _state_lock:
                        if publish_live:
                            _live_scans[normalized_site] = cached
                return cached

        buffered = _process_scan_file(normalized_site, filepath)
        if Path(filepath).is_file():
            with _state_lock:
                _processed_scans[cache_key] = buffered
                if dt is None and publish_live:
                    _live_scans[normalized_site] = buffered
        return buffered


def _refresh_live_scan(site_id: str) -> None:
    """Refresh a live site off the request path; errors leave prior data live."""
    normalized_site = site_id.upper()
    try:
        buffered = _ingest_to_buffer(normalized_site, publish_live=False)
        _prepared_map_layers(buffered, {"audiom"})
        with _state_lock:
            _live_scans[normalized_site] = buffered
            _refresh_errors.pop(normalized_site, None)
    except Exception as exc:  # pragma: no cover - exercised by deployment failures
        _logger.exception("Unable to refresh live radar scan for %s", normalized_site)
        with _state_lock:
            _refresh_errors[normalized_site] = str(exc)
            # Do not leave a cold site rate-limited for the full normal refresh
            # interval after one transient ingest failure. The readiness client
            # may retry in a few seconds and recover without a new session.
            _refresh_started_at[normalized_site] = datetime.now() - timedelta(
                seconds=_refresh_interval_seconds - LIVE_REFRESH_FAILURE_RETRY_SECONDS
            )
    finally:
        with _state_lock:
            _refreshing_sites.discard(normalized_site)
        _schedule_recurring_refresh(normalized_site)


def _schedule_live_refresh(site_id: str) -> bool:
    """Queue at most one, rate-limited latest-scan check for a radar site."""
    normalized_site = site_id.upper()
    now = datetime.now()
    with _state_lock:
        if normalized_site in _refreshing_sites:
            return True
        last_started = _refresh_started_at.get(normalized_site)
        if last_started is not None and (
            now - last_started
        ).total_seconds() < _refresh_interval_seconds:
            return False
        _refreshing_sites.add(normalized_site)
        _refresh_started_at[normalized_site] = now
    _refresh_executor.submit(_refresh_live_scan, normalized_site)
    return True


def _schedule_recurring_refresh(site_id: str) -> None:
    """Keep configured local sites warm between visitors' map requests."""
    normalized_site = site_id.upper()
    with _state_lock:
        if not _background_refresh_enabled or normalized_site not in _prewarm_sites:
            return
        prior_timer = _refresh_timers.get(normalized_site)
        if prior_timer is not None:
            prior_timer.cancel()
        timer = Timer(_refresh_interval_seconds, _schedule_live_refresh, (normalized_site,))
        timer.daemon = True
        _refresh_timers[normalized_site] = timer
        timer.start()


def _live_or_ingest(site_id: str, dt: datetime | None = None) -> tuple[BufferedScan, bool]:
    """Serve the latest completed live scan and refresh it in the background.

    Historical queries remain exact and synchronous.  A first-ever live
    request has no honest completed result to serve, so it performs one ingest;
    every later live request returns immediately while a refresh is queued.
    """
    normalized_site = site_id.upper()
    if dt is not None:
        return _ingest_to_buffer(normalized_site, dt), False
    with _state_lock:
        completed = _live_scans.get(normalized_site)
    if completed is None:
        return _ingest_to_buffer(normalized_site), False
    return completed, _schedule_live_refresh(normalized_site)


def _live_scan_state(site_id: str) -> str:
    with _state_lock:
        return "updating" if site_id.upper() in _refreshing_sites else "ready"


def _map_geojson_response(geojson: dict, buffered: BufferedScan) -> JSONResponse:
    """Expose scan freshness without changing the GeoJSON feature contract."""
    metadata = geojson.setdefault("metadata", {})
    timestamp = buffered.reflectivity_data.timestamp
    metadata["scanTimestamp"] = (
        timestamp.isoformat() if isinstance(timestamp, datetime) else str(timestamp)
    )
    metadata["refreshState"] = _live_scan_state(buffered.site_id)
    return JSONResponse(
        _json_safe(geojson),
        headers={
            "Cache-Control": "no-store",
            "X-ARW-Scan-Timestamp": metadata["scanTimestamp"],
            "X-ARW-Refresh-State": metadata["refreshState"],
        },
    )


def _great_circle_distance_miles(
    latitude_a: float, longitude_a: float, latitude_b: float, longitude_b: float
) -> float:
    """Great-circle distance without making a geocoder/network call."""
    lat_a, lon_a, lat_b, lon_b = map(
        math.radians, (latitude_a, longitude_a, latitude_b, longitude_b)
    )
    haversine = (
        math.sin((lat_b - lat_a) / 2) ** 2
        + math.cos(lat_a) * math.cos(lat_b) * math.sin((lon_b - lon_a) / 2) ** 2
    )
    return 3958.7613 * 2 * math.asin(math.sqrt(haversine))


def _beam_height_above_radar_km(distance_km: float, elevation_deg: float) -> float:
    """Approximate the center of a 4/3-Earth radar beam above its antenna."""
    distance_m = max(0.0, distance_km) * 1000.0
    effective_earth_radius_m = 4.0 / 3.0 * 6_371_000.0
    elevation_rad = math.radians(elevation_deg)
    height_m = (
        math.sqrt(
            distance_m ** 2
            + effective_earth_radius_m ** 2
            + 2 * distance_m * effective_earth_radius_m * math.sin(elevation_rad)
        )
        - effective_earth_radius_m
    )
    return height_m / 1000.0


def _location_bbox(
    latitude: float, longitude: float, radius_miles: float
) -> list[float]:
    """A geographic request extent, including when no storms are drawable.

    Audiom uses a GeoJSON bbox as the projection anchor. Without one, an empty
    local result falls back to [0, 0], which turns otherwise valid Phoenix
    camera coordinates into nonsensical walking coordinates.
    """
    latitude_delta = radius_miles / 69.0
    longitude_scale = max(69.172 * math.cos(math.radians(latitude)), 0.001)
    longitude_delta = radius_miles / longitude_scale
    return [
        max(-180.0, longitude - longitude_delta),
        max(-90.0, latitude - latitude_delta),
        min(180.0, longitude + longitude_delta),
        min(90.0, latitude + latitude_delta),
    ]


def _empty_local_result_anchor(latitude: float, longitude: float) -> dict:
    """A truthful geographic anchor for an otherwise empty local map.

    Audiom's empty-source path uses a non-geographic world origin. A point at
    the requested location keeps its projection geographic and tells the user
    why the storm interpretation layer has no weather objects.
    """
    return {
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [longitude, latitude]},
        "properties": {
            "id": "arw-request-location",
            "name": "Requested radar location: no nearby interpreted echoes",
            "ruleName": "Radar request location",
            "ruleType": "radar_echo",
            "description": (
                "No interpreted storm features are within the selected local radar range. "
                "Remote and raw radar data remain available separately."
            ),
            "passable": True,
            "soundPriority": 100,
            "centroid_lat": latitude,
            "centroid_lon": longitude,
            "isLocationAnchor": True,
        },
    }


def _filter_map_features_by_relevance(
    geojson: dict,
    request_latitude: float,
    request_longitude: float,
    radius_miles: float,
    elevation_deg: float,
) -> dict:
    """Limit interpreted map features to the requested location's vicinity.

    This works on a request-local copy of the completed layer. It never
    removes a scan, detected object, or whole-field reflectivity from ARW's
    underlying data; it only keeps remote interpretations out of this map.
    """
    kept_features = []
    omitted_object_ids: set[str] = set()
    omitted_feature_count = 0
    unlocated_feature_count = 0

    for feature in geojson.get("features", []):
        properties = feature.get("properties", {})
        centroid_lat = properties.get("centroid_lat")
        centroid_lon = properties.get("centroid_lon")
        if centroid_lat is None or centroid_lon is None:
            # Do not make an unlocated record silently disappear. This should
            # be impossible for detected storms, but keeping it is safer.
            unlocated_feature_count += 1
            kept_features.append(feature)
            continue

        distance_miles = _great_circle_distance_miles(
            request_latitude, request_longitude, centroid_lat, centroid_lon
        )
        if distance_miles > radius_miles:
            omitted_feature_count += 1
            omitted_object_ids.add(str(properties.get("object_id", properties.get("id", "unknown"))))
            continue

        properties = dict(properties)
        radar_distance_km = properties.get("distance_km")
        if radar_distance_km is not None:
            beam_height_km = _beam_height_above_radar_km(radar_distance_km, elevation_deg)
            properties["beamHeightKmAboveRadar"] = round(beam_height_km, 1)
            if beam_height_km >= ELEVATED_BEAM_HEIGHT_KM:
                properties["surfacePrecipitationObservable"] = False
                properties["soundPriority"] = min(properties.get("soundPriority", 500), 250)
                properties["description"] = (
                    f"{properties.get('description', '')} At this range the radar beam is about "
                    f"{beam_height_km:.1f} km above the radar; this is elevated evidence and cannot "
                    "determine precipitation at the surface."
                ).strip()
            else:
                properties["surfacePrecipitationObservable"] = True
        feature = dict(feature)
        feature["properties"] = properties
        kept_features.append(feature)

    metadata = geojson.setdefault("metadata", {})
    metadata["relevanceRadiusMiles"] = radius_miles
    metadata["relevanceOmittedObjectCount"] = len(omitted_object_ids)
    metadata["relevanceOmittedFeatureCount"] = omitted_feature_count
    metadata["relevanceUnlocatedFeatureCount"] = unlocated_feature_count
    # This is not a claim about storm geometry: it is the requested local map
    # extent. It keeps an empty, dry local result centered where the user asked.
    geojson["bbox"] = _location_bbox(
        request_latitude, request_longitude, radius_miles
    )
    if not kept_features:
        kept_features.append(
            _empty_local_result_anchor(request_latitude, request_longitude)
        )
        metadata["relevanceEmpty"] = True
        metadata["relevanceLocationAnchor"] = True
    else:
        metadata["relevanceEmpty"] = False
        metadata["relevanceLocationAnchor"] = False
    geojson["features"] = kept_features
    return geojson


def _map_layer_for_location(
    layer: dict,
    buffered: BufferedScan,
    latitude: float,
    longitude: float,
    radius_miles: float,
) -> dict:
    return _filter_map_features_by_relevance(
        layer,
        latitude,
        longitude,
        radius_miles,
        float(buffered.reflectivity_data.elevation_angle),
    )


@app.on_event("startup")
def start_live_refresh_workers() -> None:
    """Begin prewarming explicitly configured radar sites after server start."""
    global _background_refresh_enabled
    _background_refresh_enabled = True
    for site_id in _prewarm_sites:
        _schedule_live_refresh(site_id)


@app.on_event("shutdown")
def stop_live_refresh_workers() -> None:
    """Cancel timers and allow the process to release its refresh worker."""
    global _background_refresh_enabled
    with _state_lock:
        _background_refresh_enabled = False
        for timer in _refresh_timers.values():
            timer.cancel()
        _refresh_timers.clear()
    _refresh_executor.shutdown(wait=False, cancel_futures=True)


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


@app.get("/live/{site_id}/status")
def get_live_scan_status(site_id: str):
    """Report whether a completed live interpretation is ready or updating."""
    normalized_site = site_id.upper()
    with _state_lock:
        completed = _live_scans.get(normalized_site)
        refresh_started = _refresh_started_at.get(normalized_site)
        error = _refresh_errors.get(normalized_site)
        refreshing = normalized_site in _refreshing_sites
    timestamp = None if completed is None else completed.reflectivity_data.timestamp
    return {
        "site_id": normalized_site,
        "available": completed is not None,
        "scan_timestamp": (
            timestamp.isoformat() if isinstance(timestamp, datetime) else timestamp
        ),
        "refresh_state": "updating" if refreshing else "ready",
        "refresh_started_at": (
            refresh_started.isoformat() if refresh_started is not None else None
        ),
        # An old completed scan remains usable if a refresh fails.  The error
        # is explicit for accessible clients instead of becoming a blank map.
        "last_refresh_error": error,
    }


@app.get("/map/status")
def get_map_status(
    city: str | None = Query(None),
    state: str | None = Query(None),
    zipcode: str | None = Query(None),
    latitude: float | None = Query(None),
    longitude: float | None = Query(None),
):
    """Queue and report readiness for a location-led latest radar map.

    This endpoint deliberately never performs a cold ingest in the HTTP
    request. A caller can poll it before constructing an Audiom source URL,
    avoiding a source timeout that would otherwise persist in the viewer.
    """
    lat, lon, label = _resolve_map_location(city, state, zipcode, latitude, longitude)
    ranked_sites = rank_sites(lat, lon)
    if not ranked_sites:
        raise HTTPException(status_code=404, detail="No radar site found for that location")
    site_id = ranked_sites[0]["site_id"].upper()
    queued = _schedule_live_refresh(site_id)
    with _state_lock:
        completed = _live_scans.get(site_id)
        refreshing = site_id in _refreshing_sites
        error = _refresh_errors.get(site_id)
    timestamp = None if completed is None else completed.reflectivity_data.timestamp
    return {
        "site_id": site_id,
        "location": {"latitude": lat, "longitude": lon, "label": label},
        "available": completed is not None,
        "refresh_state": "updating" if refreshing else "ready",
        "queued": queued,
        "scan_timestamp": timestamp.isoformat() if isinstance(timestamp, datetime) else timestamp,
        "last_refresh_error": error,
        "poll_after_seconds": LIVE_REFRESH_FAILURE_RETRY_SECONDS,
    }


@app.get("/sites", response_model=list[RadarSite])
def get_sites(city: str = Query(...), state: str = Query(...)):
    lat, lon = geocode_city_state(city, state)
    ranked = rank_sites(lat, lon)
    return [RadarSite(**site) for site in ranked]


@app.get("/scan/{site_id}", response_model=ScanMeta)
def get_scan(site_id: str, datetime: str | None = Query(None)):
    dt = _parse_datetime(datetime)
    buffered, _ = _live_or_ingest(site_id, dt)
    return ScanMeta(
        site_id=site_id.upper(),
        timestamp=buffered.reflectivity_data.timestamp,
        elevation_angles=buffered.reflectivity_data.elevation_angles,
    )


@app.get("/objects/{site_id}", response_model=ObjectsResponse)
def get_objects(site_id: str, datetime: str | None = Query(None)):
    dt = _parse_datetime(datetime)
    buffered, _ = _live_or_ingest(site_id, dt)
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
    latitude: float | None = Query(None),
    longitude: float | None = Query(None),
    datetime: str | None = Query(None),
    date: date_type | None = Query(None),
    time: str | None = Query(None),
    radius_miles: float = Query(DEFAULT_MAP_RELEVANCE_RADIUS_MILES, ge=1, le=250),
):
    try:
        lat, lon, label = _resolve_map_location(city, state, zipcode, latitude, longitude)
        dt = _parse_layer_datetime(datetime, date, time)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    ranked_sites = rank_sites(lat, lon)
    if not ranked_sites:
        raise HTTPException(status_code=404, detail="No radar site found for that location")

    site = ranked_sites[0]
    buffered, _ = _live_or_ingest(site["site_id"], dt)
    layers = _prepared_map_layers(buffered)
    geojson = _map_layer_for_location(deepcopy(layers["footprints"]), buffered, lat, lon, radius_miles)
    intensity_geojson = _map_layer_for_location(deepcopy(layers["intensity"]), buffered, lat, lon, radius_miles)
    audiom_geojson = _map_layer_for_location(deepcopy(layers["audiom"]), buffered, lat, lon, radius_miles)
    centroid_geojson = _map_layer_for_location(deepcopy(layers["centroids"]), buffered, lat, lon, radius_miles)
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
    latitude: float | None = Query(None),
    longitude: float | None = Query(None),
    datetime: str | None = Query(None),
    date: date_type | None = Query(None),
    time: str | None = Query(None),
    mode: str = Query("audiom"),
    radius_miles: float = Query(DEFAULT_MAP_RELEVANCE_RADIUS_MILES, ge=1, le=250),
):
    try:
        lat, lon, _label = _resolve_map_location(city, state, zipcode, latitude, longitude)
        dt = _parse_layer_datetime(datetime, date, time)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    ranked_sites = rank_sites(lat, lon)
    if not ranked_sites:
        raise HTTPException(status_code=404, detail="No radar site found for that location")

    buffered, _ = _live_or_ingest(ranked_sites[0]["site_id"], dt)
    if mode not in {"footprints", "intensity", "audiom"}:
        raise HTTPException(
            status_code=422,
            detail="mode must be one of: audiom, intensity, footprints",
        )
    layers = _prepared_map_layers(buffered, {mode})
    if mode == "footprints":
        geojson = deepcopy(layers["footprints"])
    elif mode == "intensity":
        geojson = deepcopy(layers["intensity"])
    else:
        geojson = deepcopy(layers["audiom"])
    return _map_geojson_response(
        _map_layer_for_location(geojson, buffered, lat, lon, radius_miles), buffered
    )


@app.get("/map/precipitation.geojson")
def get_precipitation_map_geojson(
    city: str | None = Query(None),
    state: str | None = Query(None),
    zipcode: str | None = Query(None),
    latitude: float | None = Query(None),
    longitude: float | None = Query(None),
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
        lat, lon, _label = _resolve_map_location(city, state, zipcode, latitude, longitude)
        dt = _parse_layer_datetime(datetime, date, time)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    ranked_sites = rank_sites(lat, lon)
    if not ranked_sites:
        raise HTTPException(status_code=404, detail="No radar site found for that location")

    buffered, _ = _live_or_ingest(ranked_sites[0]["site_id"], dt)
    geojson = deepcopy(_prepared_precipitation_layer(buffered))
    return _map_geojson_response(geojson, buffered)


@app.get("/summary/{site_id}", response_model=SummaryResponse)
def get_summary(site_id: str, datetime: str | None = Query(None)):
    dt = _parse_datetime(datetime)
    buffered, _ = _live_or_ingest(site_id, dt)
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
    buffered, _ = _live_or_ingest(site_id, dt)
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
    buffered, _ = _live_or_ingest(site_id, dt)
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
            associated_object_peak_dbz=s.associated_object_peak_dbz,
            dual_pol_available=s.dual_pol_available,
            evidence_level=s.evidence_level,
            motion_reference=s.motion_reference,
            storm_relative_max_inbound_ms=s.storm_relative_max_inbound_ms,
            storm_relative_max_outbound_ms=s.storm_relative_max_outbound_ms,
            storm_motion_speed_kmh=s.storm_motion_speed_kmh,
            storm_motion_heading_deg=s.storm_motion_heading_deg,
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
