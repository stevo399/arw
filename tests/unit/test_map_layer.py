from datetime import datetime

import numpy as np
from shapely.geometry import shape as shapely_shape

from src.buffer import BufferedScan
from src.detection import DetectedObject, IntensityLayerData
from src.map_layer import (
    PRECIP_FIELD_LEVELS,
    build_precipitation_field_geojson,
    build_storm_audiom_geojson,
    build_storm_centroid_geojson,
    build_storm_geojson,
    build_storm_intensity_geojson,
    object_geometry,
)
from src.parser import SweepData
from src.qc.classifier import CLASS_CODES
from src.qc.parameters import GateClass


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
    # contour_mask always wraps its result in a MultiPolygon (src/contours.py),
    # even for a single contiguous blob, so this is never bare "Polygon" now
    # that the shape comes from contouring rather than a convex hull.
    assert feature["geometry"]["type"] == "MultiPolygon"
    assert feature["properties"]["heat_value"] == 50.0
    assert feature["properties"]["name"] == "Intense precipitation storm 19 miles E"
    assert feature["properties"]["ruleName"] == "Storm polygon"
    assert feature["properties"]["ruleType"] == "storm_intense"
    assert feature["properties"]["passable"] is True
    assert "Peak reflectivity 50.0 dBZ." in feature["properties"]["description"]
    assert feature["properties"]["fill"] == "#e69f00"
    assert feature["properties"]["class_fractions"] == {}
    assert geojson["metadata"]["mapType"] == "heatmap"
    assert geojson["metadata"]["suggestedMapType"] == "heatmap"
    assert geojson["metadata"]["currentStat"] == "heat_value"
    assert geojson["metadata"]["dataProperties"][0] == {
        "field": "heat_value",
        "displayName": "Reflectivity dBZ",
        "type": "number",
    }
    assert geojson["metadata"]["coordinateSystem"] == "standard"
    assert geojson["metadata"]["omittedObjectCount"] == 0
    # MultiPolygon coordinates are [polygon][ring][point]; a single blob still
    # contains exactly one polygon component.
    polygons = feature["geometry"]["coordinates"]
    assert len(polygons) == 1
    ring = polygons[0][0]
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


def test_storm_feature_carries_class_composition():
    """Audiom and the speech layer must be able to tell a mostly-clutter
    object apart from a mostly-precipitation one by reading the GeoJSON
    feature alone (2026-08-26 amendment: QC no longer deletes echo, so this
    is the only place that information survives to the output)."""
    reflectivity = np.full((12, 12), np.nan)
    reflectivity[4:8, 5:9] = 50.0
    mask = ~np.isnan(reflectivity)
    class_fractions = {"precipitation": 0.2, "ground_clutter": 0.8, "biological": 0.0}
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
                layers=[IntensityLayerData("intense precipitation", 50, 60, 24.0)],
                class_fractions=class_fractions,
            )
        ],
        labeled_grid=mask.astype(int),
        object_masks={1: mask},
    )

    geojson = build_storm_geojson(scan)
    feature = geojson["features"][0]
    assert feature["properties"]["class_fractions"] == class_fractions
    # Mutating the returned property dict must not alias the object's own
    # field.
    feature["properties"]["class_fractions"]["precipitation"] = 999.0
    assert scan.detected_objects[0].class_fractions["precipitation"] == 0.2


def test_intensity_rule_type_stability():
    """Verify all ruleType values are stable external contracts with Audiom.

    These identifiers (radar_light_rain, etc.) must never be renamed, as they
    are consumed by Audiom's accessible-mapping tool for user styling
    configuration. Renaming them silently breaks existing user styling without
    any error or warning. This test protects against silent breakage by asserting
    the complete mapping.

    "drizzle" -> "radar_drizzle" is included here too: it is a NEW identifier
    (added for the precipitation-field layer's 15-20 dBZ band, which
    `classify_intensity` labels "drizzle"), but once it ships it becomes an
    external contract exactly like the other five, so it belongs in this same
    pinning test rather than a separate one that could silently drift.
    """
    from src.map_layer import _intensity_rule_type
    from src.detection import INTENSITY_THRESHOLDS

    # Build expected mapping from INTENSITY_THRESHOLDS
    expected = {
        "drizzle": "radar_drizzle",
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


def _scan_with_mask(mask: np.ndarray, peak_dbz: float = 50.0) -> BufferedScan:
    """A BufferedScan whose single object occupies exactly `mask`."""
    n_rays, n_gates = mask.shape
    reflectivity = np.where(mask, peak_dbz, np.nan)
    return BufferedScan(
        timestamp=datetime(2026, 4, 10, 20, 0),
        site_id="KTLX",
        reflectivity_data=SweepData(
            reflectivity=reflectivity,
            azimuths=np.linspace(0.0, 359.0, n_rays),
            ranges_m=np.linspace(20000.0, 60000.0, n_gates),
            radar_lat=35.3331,
            radar_lon=-97.2778,
            elevation_angle=0.5,
            elevations=np.full(n_rays, 0.5),
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
                peak_dbz=peak_dbz,
                peak_label="intense precipitation",
                area_km2=100.0,
            )
        ],
        labeled_grid=mask.astype(int),
        object_masks={1: mask},
    )


def test_object_geometry_emits_a_polygon_type():
    mask = np.zeros((60, 60), dtype=bool)
    mask[20:40, 20:40] = True
    geometry = object_geometry(_scan_with_mask(mask), _scan_with_mask(mask).detected_objects[0])
    assert geometry["type"] in ("Polygon", "MultiPolygon")


def test_object_geometry_carries_an_interior_ring_for_a_donut():
    """The convex hull fills this middle in. Walking into the gap would then
    report a storm that is not there."""
    mask = np.zeros((60, 60), dtype=bool)
    mask[15:45, 15:45] = True
    mask[25:35, 25:35] = False

    scan = _scan_with_mask(mask)
    geometry = object_geometry(scan, scan.detected_objects[0])

    if geometry["type"] == "Polygon":
        rings = [geometry["coordinates"]]
    else:
        rings = geometry["coordinates"]
    assert any(len(part) > 1 for part in rings), "no interior ring emitted"


def test_object_geometry_keeps_two_blobs_separate():
    mask = np.zeros((60, 60), dtype=bool)
    mask[5:15, 5:15] = True
    mask[40:50, 40:50] = True

    scan = _scan_with_mask(mask)
    geometry = object_geometry(scan, scan.detected_objects[0])

    assert geometry["type"] == "MultiPolygon"
    assert len(geometry["coordinates"]) == 2


def test_object_geometry_returns_none_rather_than_inventing_a_shape():
    """The previous implementation drew a square at the centroid when hulling
    failed, presenting a fabricated shape as measurement.

    An implementation that unconditionally returns None would pass this
    assertion alone -- so this test also checks, in the same function, that a
    real mask still comes back as real geometry. That makes the "returns
    None" claim mean "returns None *only* for the empty case," not "returns
    None, full stop," without relying on test execution order or on the
    other three tests in this file to catch a stubbed-out implementation.
    """
    empty_mask = np.zeros((60, 60), dtype=bool)
    empty_scan = _scan_with_mask(empty_mask)
    assert object_geometry(empty_scan, empty_scan.detected_objects[0]) is None

    real_mask = np.zeros((60, 60), dtype=bool)
    real_mask[20:40, 20:40] = True
    real_scan = _scan_with_mask(real_mask)
    assert object_geometry(real_scan, real_scan.detected_objects[0]) is not None


def _sweep_for_omission_tests(n: int, reflectivity: np.ndarray) -> SweepData:
    return SweepData(
        reflectivity=reflectivity,
        azimuths=np.linspace(0.0, 359.0, n),
        ranges_m=np.linspace(20000.0, 60000.0, n),
        radar_lat=35.3331,
        radar_lon=-97.2778,
        elevation_angle=0.5,
        elevations=np.full(n, 0.5),
        elevation_angles=[0.5],
        radar_alt_m=390.0,
        timestamp="2026-04-10T20:00:00Z",
    )


def _scan_with_one_missing_object() -> BufferedScan:
    """Two detected objects: object 1 has real echo backing its mask, object
    2's mask is entirely False (its supporting gates vanished after
    detection, or it was never given one). object_geometry must return None
    for object 2 without touching object 1 -- this is the exact "a storm
    disappears from the map with no trace" failure the omission-count
    metadata exists to make visible instead of silent."""
    n = 20
    reflectivity = np.full((n, n), np.nan)
    reflectivity[4:8, 4:8] = 45.0
    mask1 = ~np.isnan(reflectivity)
    mask2 = np.zeros((n, n), dtype=bool)
    return BufferedScan(
        timestamp=datetime(2026, 4, 10, 20, 0),
        site_id="KTLX",
        reflectivity_data=_sweep_for_omission_tests(n, reflectivity),
        detected_objects=[
            DetectedObject(
                object_id=1,
                centroid_lat=35.2,
                centroid_lon=-96.9,
                distance_km=30.0,
                bearing_deg=90.0,
                peak_dbz=45.0,
                peak_label="heavy precipitation",
                area_km2=50.0,
                layers=[IntensityLayerData("heavy precipitation", 40, 50, 50.0)],
            ),
            DetectedObject(
                object_id=2,
                centroid_lat=35.5,
                centroid_lon=-97.1,
                distance_km=40.0,
                bearing_deg=180.0,
                peak_dbz=45.0,
                peak_label="heavy precipitation",
                area_km2=50.0,
                layers=[IntensityLayerData("heavy precipitation", 40, 50, 50.0)],
            ),
        ],
        labeled_grid=mask1.astype(int),
        object_masks={1: mask1, 2: mask2},
    )


def test_build_storm_geojson_omits_object_with_no_valid_contour():
    """Pins the skip-and-count path in build_storm_geojson. Without this,
    an implementation that reverted to a bare list comprehension over
    storm_object_to_feature results -- restoring the vanishing-storm
    behaviour object_geometry was built to prevent -- would stay green."""
    scan = _scan_with_one_missing_object()
    geojson = build_storm_geojson(scan)

    assert len(geojson["features"]) == 1
    assert geojson["features"][0]["properties"]["object_id"] == 1
    assert geojson["metadata"]["omittedObjectCount"] == 1


def test_build_storm_audiom_geojson_omits_object_and_records_both_counts():
    """Audiom is the layer the accessible mapping tool actually consumes, so
    the omission count must survive being combined from the footprint and
    intensity-band builders into this layer's own metadata."""
    scan = _scan_with_one_missing_object()
    geojson = build_storm_audiom_geojson(scan)

    footprints = [
        f for f in geojson["features"] if f["properties"]["ruleName"] == "Storm footprint"
    ]
    assert len(footprints) == 1
    assert footprints[0]["properties"]["object_id"] == 1
    assert geojson["metadata"]["omittedObjectCount"] == 1
    # Object 2's own "heavy precipitation" layer is also omitted, for the
    # same reason its footprint is: its mask is entirely False, so its band
    # contour comes back empty too. Both counts must independently survive
    # into this layer's combined metadata.
    assert geojson["metadata"]["omittedBandCount"] == 1


def _scan_with_one_empty_band() -> BufferedScan:
    """One object with two intensity layers: 'heavy precipitation' (40-50
    dBZ) matches real gates in the mask; 'severe core' (60+ dBZ) does not --
    no gate in this object ever reaches 60 dBZ, so that band's mask is
    entirely False and its contour must come back empty. The object's own
    footprint must still be reported; only the empty band should vanish from
    the intensity layer, and only with its omission counted."""
    n = 20
    reflectivity = np.full((n, n), np.nan)
    reflectivity[4:8, 4:8] = 45.0
    mask = ~np.isnan(reflectivity)
    return BufferedScan(
        timestamp=datetime(2026, 4, 10, 20, 0),
        site_id="KTLX",
        reflectivity_data=_sweep_for_omission_tests(n, reflectivity),
        detected_objects=[
            DetectedObject(
                object_id=1,
                centroid_lat=35.2,
                centroid_lon=-96.9,
                distance_km=30.0,
                bearing_deg=90.0,
                peak_dbz=45.0,
                peak_label="heavy precipitation",
                area_km2=50.0,
                layers=[
                    IntensityLayerData("heavy precipitation", 40, 50, 50.0),
                    IntensityLayerData("severe core", 60, float("inf"), 0.0),
                ],
            ),
        ],
        labeled_grid=mask.astype(int),
        object_masks={1: mask},
    )


def test_build_storm_intensity_geojson_omits_band_with_no_valid_contour():
    """Pins the equivalent skip-and-count path for intensity bands
    (omittedBandCount), added beyond the brief but under the same
    no-invented-shape contract as object_geometry."""
    scan = _scan_with_one_empty_band()
    geojson = build_storm_intensity_geojson(scan)

    assert len(geojson["features"]) == 1
    assert geojson["features"][0]["properties"]["intensity_label"] == "heavy precipitation"
    assert geojson["metadata"]["omittedBandCount"] == 1


def _scan_with_core() -> BufferedScan:
    """A 25 dBZ object wrapped around a 45 dBZ core."""
    reflectivity = np.full((60, 60), np.nan)
    reflectivity[15:45, 15:45] = 25.0
    reflectivity[25:35, 25:35] = 45.0
    mask = np.isfinite(reflectivity)
    scan = _scan_with_mask(mask)
    scan.reflectivity_data.reflectivity = reflectivity
    scan.detected_objects[0].layers = [
        IntensityLayerData("light precipitation", 20.0, 30.0, 60.0),
        IntensityLayerData("heavy precipitation", 40.0, 50.0, 10.0),
    ]
    return scan


def test_intensity_bands_do_not_overlap():
    """Walking must place the user in exactly one band.

    Today each band is its own convex hull, filled to its own middle, so
    walking inward crosses several overlapping claims about the same ground.
    """
    geojson = build_storm_intensity_geojson(_scan_with_core())
    geoms = [shapely_shape(f["geometry"]) for f in geojson["features"]]

    # A non-emptiness check first: nothing overlaps nothing, so a
    # no-overlap assertion alone would also pass an implementation that
    # (bugfully) emitted no bands at all. This pins that both real bands
    # from _scan_with_core are actually present before checking they don't
    # overlap each other.
    assert len(geoms) == 2
    for geom in geoms:
        assert geom.area > 0

    for i, a in enumerate(geoms):
        for b in geoms[i + 1:]:
            assert a.intersection(b).area < 1e-12


def test_weaker_band_has_a_hole_where_the_core_sits():
    geojson = build_storm_intensity_geojson(_scan_with_core())
    weaker = next(
        shapely_shape(f["geometry"])
        for f in geojson["features"]
        if f["properties"]["min_dbz"] == 20.0
    )
    assert weaker.area > 0
    parts = list(weaker.geoms) if weaker.geom_type == "MultiPolygon" else [weaker]
    assert sum(len(p.interiors) for p in parts) >= 1


def test_band_rule_types_are_unchanged():
    """These identifiers are an external contract with Audiom's styling.

    Spec 1 established that renaming them breaks the user's map silently.
    """
    geojson = build_storm_intensity_geojson(_scan_with_core())
    rule_types = {f["properties"]["ruleType"] for f in geojson["features"]}
    assert rule_types <= {
        "radar_light_rain", "radar_moderate_rain", "radar_heavy_rain",
        "radar_intense_rain", "radar_severe_core",
    }
    # `<=` alone would also pass an empty set, which proves nothing about
    # this scan's actual bands -- pin that both bands from _scan_with_core
    # are present and that their ruleTypes are the expected two, not just
    # "a subset of the five".
    assert rule_types == {"radar_light_rain", "radar_heavy_rain"}


def _scan_with_two_adjacent_objects() -> BufferedScan:
    """Two separate objects, same intensity, disjoint at the raster level
    but separated by only a single gate along a deliberately wavy (two
    superimposed sine frequencies) shared boundary.

    A plain rectangle will NOT reproduce the bug this guards: a rectangle's
    corners are exactly the points Douglas-Peucker keeps, so independently
    simplifying two rectangular footprints doesn't bulge them into each
    other (verified by hand before landing on this fixture). Real storm
    object boundaries are irregular like this wavy one, not rectangular --
    on a real KTLX volume (2013-05-20 19:55:27, objects 29 and 30) this
    exact failure mode measured 4.03e-06 deg^2 of cross-object band overlap
    before the fix this test pins.
    """
    n_rays, n_gates = 60, 80
    cols = np.arange(n_gates)
    boundary = np.round(30 + 28 * np.sin(cols * 1.3) + 14 * np.sin(cols * 3.7)).astype(int)
    gap = 1  # rows of empty space between the two objects at every column
    mask_upper = np.zeros((n_rays, n_gates), dtype=bool)
    mask_lower = np.zeros((n_rays, n_gates), dtype=bool)
    for c in range(n_gates):
        top = boundary[c]
        mask_upper[0:max(top, 0), c] = True
        mask_lower[min(top + gap, n_rays):n_rays, c] = True
    assert not np.any(mask_upper & mask_lower), "fixture must be disjoint at the raster level"
    assert mask_upper.any() and mask_lower.any()

    reflectivity = np.full((n_rays, n_gates), np.nan)
    reflectivity[mask_upper] = 45.0
    reflectivity[mask_lower] = 45.0
    labeled_grid = np.zeros((n_rays, n_gates), dtype=int)
    labeled_grid[mask_upper] = 1
    labeled_grid[mask_lower] = 2

    return BufferedScan(
        timestamp=datetime(2026, 4, 10, 20, 0),
        site_id="KTLX",
        reflectivity_data=SweepData(
            reflectivity=reflectivity,
            azimuths=np.linspace(0.0, 29.5, n_rays),
            ranges_m=np.linspace(200.0, 3200.0, n_gates),
            radar_lat=35.3331,
            radar_lon=-97.2778,
            elevation_angle=0.5,
            elevations=np.full(n_rays, 0.5),
            elevation_angles=[0.5],
            radar_alt_m=390.0,
            timestamp="2026-04-10T20:00:00Z",
        ),
        detected_objects=[
            DetectedObject(
                object_id=1, centroid_lat=35.30, centroid_lon=-97.25, distance_km=1.0,
                bearing_deg=5.0, peak_dbz=45.0, peak_label="heavy precipitation",
                area_km2=1.0, layers=[IntensityLayerData("heavy precipitation", 40, 50, 1.0)],
            ),
            DetectedObject(
                object_id=2, centroid_lat=35.20, centroid_lon=-97.25, distance_km=1.0,
                bearing_deg=25.0, peak_dbz=45.0, peak_label="heavy precipitation",
                area_km2=1.0, layers=[IntensityLayerData("heavy precipitation", 40, 50, 1.0)],
            ),
        ],
        labeled_grid=labeled_grid,
        object_masks={1: mask_upper, 2: mask_lower},
    )


def test_intensity_bands_do_not_overlap_across_different_objects():
    """Two separate storms must never claim overlapping ground, even when
    their raster masks are genuinely disjoint and their footprints are
    independently simplified.

    Measured directly on this fixture against the pre-fix code (clipping
    against each object's independently-simplified display footprint):
    7.834e-07 deg^2 of real cross-object overlap -- twelve orders of
    magnitude above float noise, so this is not a precision artefact. The
    area assertions below also guard against a stub that returns empty
    bands passing this vacuously (nothing overlaps nothing).
    """
    geojson = build_storm_intensity_geojson(_scan_with_two_adjacent_objects())
    by_object: dict[int, list] = {}
    for feature in geojson["features"]:
        oid = feature["properties"]["object_id"]
        by_object.setdefault(oid, []).append(shapely_shape(feature["geometry"]))

    assert set(by_object) == {1, 2}
    for geoms in by_object.values():
        assert sum(g.area for g in geoms) > 0

    for a in by_object[1]:
        for b in by_object[2]:
            assert a.intersection(b).area < 1e-12


def _scan_with_light_rain(rhohv_value: float = 0.99) -> BufferedScan:
    """Echo at 17 dBZ -- below the 20 dBZ object threshold -- plus a tiny patch.

    `gate_classification` is populated here (code for "precipitation" on
    every echoing gate) because that mirrors production: `_ingest_to_buffer`
    in src/server.py always runs `preprocess_sweep` -> `apply_quality_control`
    before a scan's reflectivity_data reaches `build_precipitation_field_geojson`,
    so `sweep.gate_classification` is never None by the time this builder
    actually runs. Leaving it None here (its BufferedScan default) would
    exercise a code path -- `dominant_class` computed with no class_fractions
    at all -- that production never hits, and would make
    `test_high_correlation_still_names_a_class` pass or fail for the wrong
    reason (an always-empty `class_fractions`) rather than genuinely
    exercising the `UNCERTAIN_RHOHV` threshold it is meant to pin.
    """
    reflectivity = np.full((60, 60), np.nan)
    reflectivity[10:30, 10:30] = 17.0        # light rain, forms no object
    reflectivity[50:52, 50:52] = 35.0        # patch far below 4 km2
    mask = np.isfinite(reflectivity)
    scan = _scan_with_mask(mask)
    scan.reflectivity_data.reflectivity = reflectivity
    scan.reflectivity_data.rhohv = np.where(mask, rhohv_value, np.nan)
    scan.reflectivity_data.zdr = np.where(mask, 1.4, np.nan)
    scan.reflectivity_data.gate_classification = np.where(
        mask, CLASS_CODES[GateClass.PRECIPITATION], CLASS_CODES[GateClass.UNKNOWN]
    ).astype(np.int8)
    return scan


def test_precipitation_layer_shows_rain_below_the_object_threshold():
    """17 dBZ forms no detected object today, so it is invisible on the map."""
    geojson = build_precipitation_field_geojson(_scan_with_light_rain())
    bands = [f["properties"]["min_dbz"] for f in geojson["features"]]
    assert 15.0 in bands


def test_precipitation_layer_shows_patches_below_the_area_threshold():
    geojson = build_precipitation_field_geojson(_scan_with_light_rain())
    strong = [f for f in geojson["features"] if f["properties"]["min_dbz"] == 30.0]
    assert strong, "a 2x2 gate patch at 35 dBZ produced no feature"


def test_every_band_carries_its_evidence():
    geojson = build_precipitation_field_geojson(_scan_with_light_rain())
    for feature in geojson["features"]:
        properties = feature["properties"]
        assert "median_rhohv" in properties
        assert "median_zdr" in properties
        assert "dominant_class" in properties
        assert "class_fractions" in properties


def test_low_correlation_reports_uncertain_rather_than_guessing():
    """On the clear-air KIWA volume both precipitation and biological classes
    sit at correlation 0.58-0.64 -- indistinguishable, and far too low to be
    rain. No label is supportable there, so the band must say so."""
    geojson = build_precipitation_field_geojson(_scan_with_light_rain(rhohv_value=0.60))
    classes = {f["properties"]["dominant_class"] for f in geojson["features"]}
    assert classes == {"uncertain"}


def test_high_correlation_still_names_a_class():
    geojson = build_precipitation_field_geojson(_scan_with_light_rain(rhohv_value=0.99))
    classes = {f["properties"]["dominant_class"] for f in geojson["features"]}
    assert "uncertain" not in classes


def test_precipitation_layer_skips_a_band_with_no_supporting_gate():
    """`_scan_with_light_rain`'s reflectivity has two disjoint blobs (17 dBZ
    and 35 dBZ) with nothing in between, so `exclusive_bands` (src/contours.py)
    produces a genuinely non-empty [20, 30) geometry -- a floating-point
    sliver left by its coverage-simplify/aeqd round-trip on two contour
    levels that coincide before simplification -- even though no gate's own
    reflectivity ever falls in [20, 30). Confirmed directly against
    `exclusive_bands` on this fixture's field: that band's geometry has
    area ~1.5e-6 sq deg and `is_empty` is False, so the ordinary
    `geometry.is_empty` check alone would let it through. This pins that
    `build_precipitation_field_geojson` skips it (via `_band_gate_mask`)
    rather than emitting a shape with no real precipitation behind it,
    while the two real bands either side of it still come through --
    ruling out a stub that (bugfully) drops every band.
    """
    geojson = build_precipitation_field_geojson(_scan_with_light_rain())
    band_keys = {
        (f["properties"]["min_dbz"], f["properties"]["max_dbz"])
        for f in geojson["features"]
    }
    assert band_keys == {(15.0, 20.0), (30.0, 40.0)}
    assert (20.0, 30.0) not in band_keys


def test_light_rain_band_gets_its_own_ruletype_not_generic_echo():
    """15-20 dBZ is exactly the band this layer exists to surface (it is
    below the storm layer's 20 dBZ floor). `classify_intensity(15.0)`
    returns "drizzle", which had no entry in `_intensity_rule_type`'s
    mapping -- so this band fell through to the generic "radar_echo",
    which the Audiom accessible-mapping tool leaves unstyled. Styled or
    not, the band is present in the GeoJSON, so a test asserting only
    "some ruleType string exists" would pass either way; this pins the
    specific identifier so an unstyled regression is caught.
    """
    geojson = build_precipitation_field_geojson(_scan_with_light_rain())
    drizzle_band = next(
        f for f in geojson["features"] if f["properties"]["min_dbz"] == 15.0
    )
    assert drizzle_band["properties"]["ruleType"] == "radar_drizzle"
    assert drizzle_band["properties"]["ruleType"] != "radar_echo"


def _scan_with_fragment_sizes() -> BufferedScan:
    """One 25 dBZ band containing two disjoint pieces at genuinely different
    ground sizes, at real NEXRAD-like resolution (360 rays at ~1 degree,
    250 m gate spacing) so contouring produces realistic piece geometry
    rather than the coarse, oversized single-gate footprints the other
    fixtures in this file use (which would make every piece huge).

    - A 2x2 block of gates (rays 0-1, gates 40-41, ~30 km range) -- confirmed
      directly against `exclusive_bands` on this exact field: contouring it
      at DEFAULT_SIMPLIFY_M produces a real piece of about 0.1315 km2 --
      below MIN_PRECIP_FRAGMENT_AREA_KM2 (0.5), i.e. the speckle the filter
      exists to drop, but not so close to zero that its area would round
      away to nothing when reported.
    - A 6x6 block of gates (rays 90-95, gates 40-45) -- confirmed directly
      at about 3.34 km2, comfortably above the 0.5 km2 threshold, i.e. a
      genuine small shower that must survive.
    """
    n_rays, n_gates = 360, 60
    azimuths = np.linspace(0.0, 359.0, n_rays)
    ranges_m = np.arange(20000.0, 20000.0 + 250.0 * n_gates, 250.0)
    reflectivity = np.full((n_rays, n_gates), np.nan)
    reflectivity[0:2, 40:42] = 25.0     # tiny: 2x2 gate block, ~0.13 km2
    reflectivity[90:96, 40:46] = 25.0   # big: 6x6 gate block, ~3.34 km2
    return BufferedScan(
        timestamp=datetime(2026, 4, 10, 20, 0),
        site_id="KTLX",
        reflectivity_data=SweepData(
            reflectivity=reflectivity,
            azimuths=azimuths,
            ranges_m=ranges_m,
            radar_lat=35.3331,
            radar_lon=-97.2778,
            elevation_angle=0.5,
            elevations=np.full(n_rays, 0.5),
            elevation_angles=[0.5],
            radar_alt_m=390.0,
            timestamp="2026-04-10T20:00:00Z",
        ),
        detected_objects=[],
        labeled_grid=np.zeros((n_rays, n_gates), dtype=int),
        object_masks={},
    )


def test_small_fragment_is_dropped_and_counted():
    geojson = build_precipitation_field_geojson(_scan_with_fragment_sizes())
    band = next(f for f in geojson["features"] if f["properties"]["min_dbz"] == 20.0)
    geometry = shapely_shape(band["geometry"])
    pieces = list(geometry.geoms) if geometry.geom_type == "MultiPolygon" else [geometry]
    # Only the 6x6 piece should remain -- the isolated single-gate piece is
    # below MIN_PRECIP_FRAGMENT_AREA_KM2 and must be gone from the geometry
    # actually served, not just noted in metadata.
    assert len(pieces) == 1
    assert geojson["metadata"]["omittedFragmentCount"] == 1


def test_large_fragment_survives():
    geojson = build_precipitation_field_geojson(_scan_with_fragment_sizes())
    band = next(f for f in geojson["features"] if f["properties"]["min_dbz"] == 20.0)
    geometry = shapely_shape(band["geometry"])
    # A stub that dropped every fragment (not just the small one) would also
    # leave omittedFragmentCount == 1 if it happened to report the count
    # correctly but return empty geometry -- so this checks the surviving
    # shape actually has real area, not just that the feature exists.
    assert geometry.area > 0


def test_dropped_fragment_area_is_reported_and_nonzero():
    geojson = build_precipitation_field_geojson(_scan_with_fragment_sizes())
    assert geojson["metadata"]["omittedFragmentCount"] == 1
    assert geojson["metadata"]["omittedFragmentAreaKm2"] > 0.0
    # The dropped piece is the 2x2 gate block (~0.13 km2), not the
    # surviving 6x6 block (~3.34 km2) -- pin that the reported area is on
    # the scale of the dropped piece, not the whole band.
    assert geojson["metadata"]["omittedFragmentAreaKm2"] < 0.5


def test_storm_layer_is_unaffected_by_the_fragment_filter():
    """The fragment-size filter (MIN_PRECIP_FRAGMENT_AREA_KM2) applies only
    to build_precipitation_field_geojson. A tiny object footprint -- well
    under the 0.5 km2 threshold -- must still be reported in full by the
    storm layer, which is governed by object detection's own 4 km2 area
    floor, not this constant.
    """
    n_rays, n_gates = 360, 60
    azimuths = np.linspace(0.0, 359.0, n_rays)
    ranges_m = np.arange(20000.0, 20000.0 + 250.0 * n_gates, 250.0)
    reflectivity = np.full((n_rays, n_gates), np.nan)
    reflectivity[0:1, 40:41] = 45.0  # a single isolated gate, ~3.3e-8 km2
    mask = np.isfinite(reflectivity)
    scan = BufferedScan(
        timestamp=datetime(2026, 4, 10, 20, 0),
        site_id="KTLX",
        reflectivity_data=SweepData(
            reflectivity=reflectivity,
            azimuths=azimuths,
            ranges_m=ranges_m,
            radar_lat=35.3331,
            radar_lon=-97.2778,
            elevation_angle=0.5,
            elevations=np.full(n_rays, 0.5),
            elevation_angles=[0.5],
            radar_alt_m=390.0,
            timestamp="2026-04-10T20:00:00Z",
        ),
        detected_objects=[
            DetectedObject(
                object_id=1,
                centroid_lat=35.30,
                centroid_lon=-97.20,
                distance_km=3.0,
                bearing_deg=0.0,
                peak_dbz=45.0,
                peak_label="intense precipitation",
                area_km2=0.00000003,
                layers=[IntensityLayerData("intense precipitation", 40, 50, 0.00000003)],
            )
        ],
        labeled_grid=mask.astype(int),
        object_masks={1: mask},
    )

    geojson = build_storm_geojson(scan)
    assert len(geojson["features"]) == 1
    assert geojson["metadata"]["omittedObjectCount"] == 0


def _scan_with_uniform_field(value: float) -> BufferedScan:
    """A uniform block of exactly `value` dBZ, at real NEXRAD-like geometry.

    360 rays at ~1 degree and 250 m gate spacing, so a 20x20 gate block is
    about 23 km2 -- a real severe core's worth of ground, not the ~1550 km2
    the 60x60 fixtures elsewhere in this file would make of the same block.
    """
    n_rays, n_gates = 360, 60
    reflectivity = np.full((n_rays, n_gates), np.nan)
    reflectivity[100:120, 20:40] = value
    mask = np.isfinite(reflectivity)
    return BufferedScan(
        timestamp=datetime(2026, 4, 10, 20, 0),
        site_id="KTLX",
        reflectivity_data=SweepData(
            reflectivity=reflectivity,
            azimuths=np.linspace(0.0, 359.0, n_rays),
            ranges_m=np.arange(20000.0, 20000.0 + 250.0 * n_gates, 250.0),
            radar_lat=35.3331,
            radar_lon=-97.2778,
            elevation_angle=0.5,
            elevations=np.full(n_rays, 0.5),
            elevation_angles=[0.5],
            radar_alt_m=390.0,
            timestamp="2026-04-10T20:00:00Z",
            rhohv=np.where(mask, 0.99, np.nan),
            zdr=np.where(mask, 1.4, np.nan),
            gate_classification=np.where(
                mask, CLASS_CODES[GateClass.PRECIPITATION], CLASS_CODES[GateClass.UNKNOWN]
            ).astype(np.int8),
        ),
        detected_objects=[],
        labeled_grid=np.zeros((n_rays, n_gates), dtype=int),
        object_masks={},
    )


def test_uniform_severe_core_is_drawn_not_silently_dropped():
    """A 23 km2 region of uniform 60.0 dBZ must not come out as nothing.

    This exact case produced ZERO features and every metadata counter
    reading zero: the geometry used contourpy's strict `>` so a uniformly
    60.0 field had no gate above 60, and the guard meant to suppress
    phantom slivers used `>=` and deleted the band whose geometry was real.
    A severe core, rendered as total silence, with the metadata saying
    nothing had been omitted.
    """
    geojson = build_precipitation_field_geojson(_scan_with_uniform_field(60.0))
    bands = {f["properties"]["min_dbz"] for f in geojson["features"]}
    assert 60.0 in bands, (
        f"a uniform 60 dBZ severe core produced no 60+ feature; "
        f"omittedBands={geojson['metadata']['omittedBands']}"
    )
    core = next(f for f in geojson["features"] if f["properties"]["min_dbz"] == 60.0)
    assert shapely_shape(core["geometry"]).area > 0


def test_uniform_moderate_region_is_drawn_not_silently_dropped():
    geojson = build_precipitation_field_geojson(_scan_with_uniform_field(20.0))
    bands = {f["properties"]["min_dbz"] for f in geojson["features"]}
    assert 20.0 in bands, (
        f"a uniform 20 dBZ region produced no 20-30 feature; "
        f"omittedBands={geojson['metadata']['omittedBands']}"
    )


def test_every_band_that_produces_no_feature_is_named_with_its_reason():
    """No band may leave this builder without a trace.

    Two paths out of the feature loop used to `continue` without
    incrementing anything, so a band could disappear entirely and the
    metadata would say nothing was omitted. Every band the layer does not
    draw must now appear in `omittedBands` with a reason, the number of
    gates that supported it, and the ground those gates covered -- because
    a bare count cannot tell the project owner whether what vanished was
    speckle or a severe core.
    """
    geojson = build_precipitation_field_geojson(_scan_with_uniform_field(60.0))
    metadata = geojson["metadata"]

    drawn = {f["properties"]["min_dbz"] for f in geojson["features"]}
    omitted = {b["min_dbz"] for b in metadata["omittedBands"]}
    assert drawn | omitted == set(PRECIP_FIELD_LEVELS), (
        "some band neither drew a feature nor recorded an omission"
    )
    assert not (drawn & omitted), "a band cannot be both drawn and omitted"
    assert metadata["omittedBandCount"] == len(metadata["omittedBands"])

    reasons = {b["reason"] for b in metadata["omittedBands"]}
    assert reasons <= {
        "empty_geometry", "no_supporting_gate", "all_fragments_below_area_floor",
    }
    for band in metadata["omittedBands"]:
        assert set(band) == {
            "band", "min_dbz", "max_dbz", "reason", "gateCount", "gateAreaKm2",
        }
        # Nothing was actually lost in this fixture: every omitted band is
        # omitted because there is no such weather, which the record says.
        assert band["gateCount"] == 0
        assert band["gateAreaKm2"] == 0.0


def test_a_band_wiped_out_by_the_fragment_filter_is_named_in_metadata():
    """The aggregate fragment counters cannot say WHICH band disappeared.

    On KIWA the 30-40 and 40-50 dBZ bands both vanish this way, leaving
    that volume's precipitation layer showing nothing at all above 30 dBZ,
    with only an aggregate fragment count to show for it. A band that loses
    every piece to the area floor is now named, with the ground its own
    gates covered.
    """
    scan = _scan_with_fragment_sizes()
    # Add an isolated 45 dBZ gate: a whole band whose only piece is far
    # below MIN_PRECIP_FRAGMENT_AREA_KM2.
    scan.reflectivity_data.reflectivity[200, 30] = 45.0

    geojson = build_precipitation_field_geojson(scan)
    drawn = {f["properties"]["min_dbz"] for f in geojson["features"]}
    assert 40.0 not in drawn, "fixture no longer exercises the area floor"

    wiped = [b for b in geojson["metadata"]["omittedBands"] if b["min_dbz"] == 40.0]
    assert wiped, "a band wiped out by the fragment filter left no record"
    assert wiped[0]["reason"] == "all_fragments_below_area_floor"
    assert wiped[0]["gateCount"] == 1
    assert wiped[0]["gateAreaKm2"] > 0.0
