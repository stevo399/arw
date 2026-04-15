from src.models import RadarSite, ScanMeta, IntensityLayer, RainObject, ObjectsResponse, SummaryResponse
from src.models import TrackPosition, TrackMotion, TrackFocus, StormTrack, TracksResponse, TrackDetailResponse, TrackEvent


def test_radar_site_model():
    site = RadarSite(
        site_id="KTLX",
        name="Oklahoma City",
        latitude=35.3331,
        longitude=-97.2778,
        elevation_m=370.0,
        distance_km=50.0,
        beam_height_m=1200.0,
    )
    assert site.site_id == "KTLX"
    assert site.distance_km == 50.0


def test_scan_meta_model():
    meta = ScanMeta(
        site_id="KTLX",
        timestamp="2026-04-08T18:30:00Z",
        elevation_angles=[0.5, 1.5, 2.4],
    )
    assert meta.site_id == "KTLX"
    assert len(meta.elevation_angles) == 3


def test_intensity_layer_model():
    layer = IntensityLayer(
        label="heavy rain",
        min_dbz=40.0,
        max_dbz=50.0,
        area_km2=12.5,
    )
    assert layer.label == "heavy rain"


def test_rain_object_model():
    obj = RainObject(
        object_id=1,
        centroid_lat=35.5,
        centroid_lon=-97.3,
        distance_km=25.0,
        bearing_deg=270.0,
        peak_dbz=55.0,
        peak_label="intense rain",
        area_km2=80.0,
        layers=[
            IntensityLayer(label="light rain", min_dbz=20.0, max_dbz=30.0, area_km2=80.0),
            IntensityLayer(label="moderate rain", min_dbz=30.0, max_dbz=40.0, area_km2=40.0),
            IntensityLayer(label="heavy rain", min_dbz=40.0, max_dbz=50.0, area_km2=15.0),
            IntensityLayer(label="intense rain", min_dbz=50.0, max_dbz=60.0, area_km2=5.0),
        ],
    )
    assert obj.peak_label == "intense rain"
    assert len(obj.layers) == 4


def test_objects_response_model():
    resp = ObjectsResponse(
        site_id="KTLX",
        timestamp="2026-04-08T18:30:00Z",
        object_count=0,
        objects=[],
    )
    assert resp.object_count == 0


def test_summary_response_model():
    resp = SummaryResponse(
        site_id="KTLX",
        timestamp="2026-04-08T18:30:00Z",
        text="KTLX: No significant precipitation detected.",
    )
    assert "No significant" in resp.text


def test_track_position_model():
    pos = TrackPosition(
        timestamp="2026-04-08T18:30:00Z",
        latitude=35.5,
        longitude=-97.3,
        distance_km=40.2,
        bearing_deg=270.0,
    )
    assert pos.latitude == 35.5


def test_track_motion_model():
    motion = TrackMotion(
        speed_kmh=56.3,
        speed_mph=35,
        heading_deg=45.0,
        heading_label="NE",
    )
    assert motion.speed_mph == 35
    assert motion.heading_label == "NE"


def test_track_motion_model_stationary():
    motion = TrackMotion(
        speed_kmh=0.0,
        speed_mph=0,
        heading_deg=None,
        heading_label="stationary",
    )
    assert motion.heading_deg is None


def test_track_focus_model():
    focus = TrackFocus(
        label="medium",
        score=0.58,
        reason="recent focus handoff",
        selection_margin=1.8,
        runner_up_track_id=7,
        recent_heading_flip_count=1,
        recent_reported_heading_flip_count=1,
        recent_reported_heading_sequence=["SE@140:motion_field", "WNW@290:motion_field"],
        recent_focus_switch_count=1,
        recent_structural_event_count=6,
    )
    assert focus.label == "medium"
    assert focus.selection_margin == 1.8
    assert focus.recent_reported_heading_flip_count == 1
    assert focus.recent_heading_flip_count == 1
    assert focus.recent_reported_heading_sequence == ["SE@140:motion_field", "WNW@290:motion_field"]


def test_storm_track_model():
    track = StormTrack(
        track_id=1,
        status="active",
        positions=[
            TrackPosition(timestamp="2026-04-08T18:30:00Z", latitude=35.5, longitude=-97.3, distance_km=40.2, bearing_deg=270.0),
        ],
        motion=TrackMotion(speed_kmh=56.3, speed_mph=35, heading_deg=45.0, heading_label="NE"),
        focus=TrackFocus(label="high", score=0.9),
        peak_dbz=55.0,
        peak_label="intense rain",
        merged_into=None,
        split_from=None,
        first_seen="2026-04-08T18:20:00Z",
        last_seen="2026-04-08T18:30:00Z",
    )
    assert track.track_id == 1
    assert track.status == "active"


def test_track_event_model():
    event = TrackEvent(
        event_type="merge",
        timestamp="2026-04-08T18:30:00Z",
        description="Tracks 2, 3 merged into track 1",
        involved_track_ids=[1, 2, 3],
    )
    assert event.event_type == "merge"


def test_tracks_response_model():
    resp = TracksResponse(
        site_id="KTLX",
        timestamp="2026-04-08T18:30:00Z",
        active_count=0,
        tracks=[],
        recent_events=[],
    )
    assert resp.active_count == 0


def test_track_detail_response_model():
    resp = TrackDetailResponse(
        track_id=1,
        status="active",
        positions=[],
        motion=TrackMotion(speed_kmh=0.0, speed_mph=0, heading_deg=None, heading_label="stationary"),
        focus=TrackFocus(label="medium", score=0.6),
        peak_history=[],
        merged_into=None,
        split_from=None,
        first_seen="2026-04-08T18:20:00Z",
        last_seen="2026-04-08T18:30:00Z",
    )
    assert resp.track_id == 1
