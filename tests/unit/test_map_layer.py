from datetime import datetime

import numpy as np

from src.buffer import BufferedScan
from src.detection import DetectedObject, IntensityLayerData
from src.map_layer import (
    build_storm_audiom_geojson,
    build_storm_centroid_geojson,
    build_storm_geojson,
    build_storm_intensity_geojson,
)
from src.parser import SweepData


def test_build_storm_geojson_returns_polygon_features():
    reflectivity = np.full((12, 12), np.nan)
    reflectivity[4:8, 5:9] = 50.0
    mask = ~np.isnan(reflectivity)
    scan = BufferedScan(
        timestamp=datetime(2026, 4, 10, 20, 0),
        site_id="KTLX",
        reflectivity_data=SweepData(
            reflectivity=reflectivity,
            azimuths=np.linspace(80, 100, 12),
            ranges_m=np.linspace(20000, 40000, 12),
            radar_lat=35.3331,
            radar_lon=-97.2778,
            elevation_angle=0.5,
            elevations=np.full(12, 0.5),
            elevation_angles=[0.5],
            radar_alt_m=390.0,
            timestamp="2026-04-10T20:00:00Z",
        ),
        detected_objects=[
            DetectedObject(
                object_id=1,
                centroid_lat=35.2,
                centroid_lon=-96.9,
                distance_km=30.0,
                bearing_deg=90.0,
                peak_dbz=50.0,
                peak_label="intense precipitation",
                area_km2=24.0,
                layers=[
                    IntensityLayerData("intense precipitation", 50, 60, 24.0),
                ],
            )
        ],
        labeled_grid=mask.astype(int),
        object_masks={1: mask},
    )

    geojson = build_storm_geojson(scan)

    assert geojson["type"] == "FeatureCollection"
    assert len(geojson["features"]) == 1
    feature = geojson["features"][0]
    assert feature["geometry"]["type"] == "Polygon"
    assert feature["properties"]["heat_value"] == 50.0
    assert feature["properties"]["name"] == "Intense precipitation storm 19 miles E"
    assert feature["properties"]["ruleName"] == "Storm polygon"
    assert feature["properties"]["ruleType"] == "storm_intense"
    assert feature["properties"]["passable"] is True
    assert "Peak reflectivity 50.0 dBZ." in feature["properties"]["description"]
    assert feature["properties"]["fill"] == "#e69f00"
    assert geojson["metadata"]["mapType"] == "heatmap"
    assert geojson["metadata"]["suggestedMapType"] == "heatmap"
    assert geojson["metadata"]["currentStat"] == "heat_value"
    assert geojson["metadata"]["dataProperties"][0] == {
        "field": "heat_value",
        "displayName": "Reflectivity dBZ",
        "type": "number",
    }
    assert geojson["metadata"]["coordinateSystem"] == "standard"
    ring = feature["geometry"]["coordinates"][0]
    assert len(ring) >= 4
    assert ring[0] == ring[-1]


def test_build_storm_intensity_geojson_returns_radar_band_features():
    reflectivity = np.full((12, 12), np.nan)
    reflectivity[4:8, 5:9] = 45.0
    reflectivity[5:7, 6:8] = 62.0
    mask = ~np.isnan(reflectivity)
    scan = BufferedScan(
        timestamp=datetime(2026, 4, 10, 20, 0),
        site_id="KTLX",
        reflectivity_data=SweepData(
            reflectivity=reflectivity,
            azimuths=np.linspace(80, 100, 12),
            ranges_m=np.linspace(20000, 40000, 12),
            radar_lat=35.3331,
            radar_lon=-97.2778,
            elevation_angle=0.5,
            elevations=np.full(12, 0.5),
            elevation_angles=[0.5],
            radar_alt_m=390.0,
            timestamp="2026-04-10T20:00:00Z",
        ),
        detected_objects=[
            DetectedObject(
                object_id=1,
                centroid_lat=35.2,
                centroid_lon=-96.9,
                distance_km=30.0,
                bearing_deg=90.0,
                peak_dbz=62.0,
                peak_label="severe core",
                area_km2=24.0,
                layers=[
                    IntensityLayerData("heavy precipitation", 40, 50, 18.0),
                    IntensityLayerData("severe core", 60, float("inf"), 6.0),
                ],
            )
        ],
        labeled_grid=mask.astype(int),
        object_masks={1: mask},
    )

    geojson = build_storm_intensity_geojson(scan)

    assert geojson["metadata"]["name"] == "ARW radar intensity bands"
    assert [feature["properties"]["ruleType"] for feature in geojson["features"]] == [
        "radar_heavy_rain",
        "radar_severe_core",
    ]
    assert geojson["features"][0]["properties"]["heat_value"] == 45.0
    assert geojson["features"][1]["properties"]["heat_value"] == 65.0
    assert geojson["features"][1]["properties"]["max_dbz"] is None


def test_build_storm_audiom_geojson_combines_footprints_and_intensity_bands():
    reflectivity = np.full((12, 12), np.nan)
    reflectivity[4:8, 5:9] = 45.0
    mask = ~np.isnan(reflectivity)
    scan = BufferedScan(
        timestamp=datetime(2026, 4, 10, 20, 0),
        site_id="KTLX",
        reflectivity_data=SweepData(
            reflectivity=reflectivity,
            azimuths=np.linspace(80, 100, 12),
            ranges_m=np.linspace(20000, 40000, 12),
            radar_lat=35.3331,
            radar_lon=-97.2778,
            elevation_angle=0.5,
            elevations=np.full(12, 0.5),
            elevation_angles=[0.5],
            radar_alt_m=390.0,
            timestamp="2026-04-10T20:00:00Z",
        ),
        detected_objects=[
            DetectedObject(
                object_id=1,
                centroid_lat=35.2,
                centroid_lon=-96.9,
                distance_km=30.0,
                bearing_deg=90.0,
                peak_dbz=45.0,
                peak_label="heavy precipitation",
                area_km2=24.0,
                layers=[IntensityLayerData("heavy precipitation", 40, 50, 24.0)],
            )
        ],
        labeled_grid=mask.astype(int),
        object_masks={1: mask},
    )

    geojson = build_storm_audiom_geojson(scan)

    assert len(geojson["features"]) == 2
    assert geojson["features"][0]["properties"]["ruleName"] == "Storm footprint"
    assert geojson["features"][1]["properties"]["ruleName"] == "Radar intensity band"


def test_build_storm_centroid_geojson_returns_point_features():
    reflectivity = np.full((12, 12), np.nan)
    mask = np.zeros_like(reflectivity, dtype=bool)
    scan = BufferedScan(
        timestamp=datetime(2026, 4, 10, 20, 0),
        site_id="KTLX",
        reflectivity_data=SweepData(
            reflectivity=reflectivity,
            azimuths=np.linspace(80, 100, 12),
            ranges_m=np.linspace(20000, 40000, 12),
            radar_lat=35.3331,
            radar_lon=-97.2778,
            elevation_angle=0.5,
            elevations=np.full(12, 0.5),
            elevation_angles=[0.5],
            radar_alt_m=390.0,
            timestamp="2026-04-10T20:00:00Z",
        ),
        detected_objects=[
            DetectedObject(
                object_id=1,
                centroid_lat=35.2,
                centroid_lon=-96.9,
                distance_km=30.0,
                bearing_deg=90.0,
                peak_dbz=50.0,
                peak_label="intense precipitation",
                area_km2=24.0,
            )
        ],
        labeled_grid=mask.astype(int),
        object_masks={},
    )

    geojson = build_storm_centroid_geojson(scan)

    feature = geojson["features"][0]
    assert feature["geometry"] == {
        "type": "Point",
        "coordinates": [-96.9, 35.2],
    }
    assert feature["properties"]["object_id"] == 1
    assert geojson["metadata"]["sourceName"] == "arw-storm-centroids"


def test_intensity_rule_type_stability():
    """Verify all ruleType values are stable external contracts with Audiom.

    These identifiers (radar_light_rain, etc.) must never be renamed, as they
    are consumed by Audiom's accessible-mapping tool for user styling
    configuration. Renaming them silently breaks existing user styling without
    any error or warning. This test protects against silent breakage by asserting
    the complete mapping.
    """
    from src.map_layer import _intensity_rule_type
    from src.detection import INTENSITY_THRESHOLDS

    # Build expected mapping from INTENSITY_THRESHOLDS
    expected = {
        "light precipitation": "radar_light_rain",
        "moderate precipitation": "radar_moderate_rain",
        "heavy precipitation": "radar_heavy_rain",
        "intense precipitation": "radar_intense_rain",
        "severe core": "radar_severe_core",
    }

    # Verify each label maps to the expected ruleType
    for label, expected_rule_type in expected.items():
        actual_rule_type = _intensity_rule_type(label)
        assert actual_rule_type == expected_rule_type, (
            f"ruleType for '{label}' changed from '{expected_rule_type}' to '{actual_rule_type}'. "
            f"This breaks Audiom user styling."
        )


def test_intensity_fill_color_complete_coverage():
    """Verify _intensity_fill_color returns a color for each intensity label.

    A missing label mapping that falls through to the default color silently
    changes every polygon's appearance. This test ensures each label has an
    explicit color mapping.
    """
    from src.map_layer import _intensity_fill_color
    from src.detection import INTENSITY_THRESHOLDS

    # Extract labels from INTENSITY_THRESHOLDS
    labels = [label for _, _, label in INTENSITY_THRESHOLDS]

    # Define expected colors
    expected_colors = {
        "light precipitation": "#2ca25f",
        "moderate precipitation": "#ffff66",
        "heavy precipitation": "#ffcc33",
        "intense precipitation": "#ff6600",
        "severe core": "#cc00cc",
    }

    # Verify each label has a color (not the default)
    default_color = "#56b4e9"
    for label in labels:
        actual_color = _intensity_fill_color(label)
        assert actual_color == expected_colors[label], (
            f"Color for '{label}' is '{actual_color}', expected '{expected_colors[label]}'"
        )
        assert actual_color != default_color, (
            f"'{label}' fell through to default color '{default_color}'. "
            f"This silently changes polygon appearance."
        )
