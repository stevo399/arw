from src.summary import generate_summary, km_to_miles
from src.detection import DetectedObject, IntensityLayerData, degrees_to_bearing


def test_km_to_miles():
    assert km_to_miles(1.60934) == 1
    assert km_to_miles(0.0) == 0
    assert km_to_miles(100.0) == 62


def test_generate_summary_no_objects():
    text = generate_summary(
        site_id="KTLX",
        site_name="Oklahoma City",
        timestamp="2026-04-08T18:30:00Z",
        objects=[],
    )
    assert text == "Oklahoma City: No significant precipitation detected."


def test_generate_summary_single_object():
    obj = DetectedObject(
        object_id=1,
        centroid_lat=35.5,
        centroid_lon=-97.3,
        distance_km=40.2,
        bearing_deg=270.0,
        peak_dbz=45.0,
        peak_label="heavy rain",
        area_km2=120.5,
        layers=[
            IntensityLayerData(label="light rain", min_dbz=20, max_dbz=30, area_km2=120.5),
            IntensityLayerData(label="moderate rain", min_dbz=30, max_dbz=40, area_km2=60.0),
            IntensityLayerData(label="heavy rain", min_dbz=40, max_dbz=50, area_km2=15.0),
        ],
    )
    text = generate_summary(
        site_id="KTLX",
        site_name="Oklahoma City",
        timestamp="2026-04-08T18:30:00Z",
        objects=[obj],
    )
    assert "Oklahoma City" in text
    assert "1 rain object" in text
    assert "heavy rain" in text
    assert "25 miles" in text
    assert "W" in text
    assert "75 square miles" in text


def test_generate_summary_multiple_objects():
    obj1 = DetectedObject(
        object_id=1,
        centroid_lat=35.5, centroid_lon=-97.3,
        distance_km=40.2, bearing_deg=270.0,
        peak_dbz=55.0, peak_label="intense rain",
        area_km2=200.0, layers=[],
    )
    obj2 = DetectedObject(
        object_id=2,
        centroid_lat=35.8, centroid_lon=-97.1,
        distance_km=60.0, bearing_deg=0.0,
        peak_dbz=30.0, peak_label="moderate rain",
        area_km2=50.0, layers=[],
    )
    text = generate_summary(
        site_id="KTLX",
        site_name="Oklahoma City",
        timestamp="2026-04-08T18:30:00Z",
        objects=[obj1, obj2],
    )
    assert "2 rain objects" in text
    assert "intense rain" in text  # Strongest object
