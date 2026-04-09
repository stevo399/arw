from src.models import RadarSite, ScanMeta, IntensityLayer, RainObject, ObjectsResponse, SummaryResponse


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
