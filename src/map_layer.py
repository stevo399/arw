from typing import Any

from shapely.geometry import MultiPolygon, mapping

from src.buffer import BufferedScan
from src.contours import contour_mask, exclusive_bands
from src.detection import DetectedObject, IntensityLayerData, degrees_to_bearing
from src.summary import km2_to_mi2, km_to_miles


def _storm_fill_color(peak_dbz: float) -> str:
    if peak_dbz >= 60:
        return "#d55e00"
    if peak_dbz >= 50:
        return "#e69f00"
    if peak_dbz >= 40:
        return "#f0e442"
    if peak_dbz >= 30:
        return "#009e73"
    return "#56b4e9"


def _intensity_fill_color(label: str) -> str:
    return {
        "light precipitation": "#2ca25f",
        "moderate precipitation": "#ffff66",
        "heavy precipitation": "#ffcc33",
        "intense precipitation": "#ff6600",
        "severe core": "#cc00cc",
    }.get(label, "#56b4e9")


def _intensity_heat_value(layer: IntensityLayerData) -> float:
    if layer.max_dbz == float("inf"):
        return max(layer.min_dbz, 65.0)
    return round((layer.min_dbz + layer.max_dbz) / 2.0, 1)


def _storm_rule_type(peak_dbz: float) -> str:
    if peak_dbz >= 60:
        return "storm_severe_core"
    if peak_dbz >= 50:
        return "storm_intense"
    if peak_dbz >= 40:
        return "storm_heavy"
    if peak_dbz >= 30:
        return "storm_moderate"
    return "storm_light"


def _intensity_rule_type(label: str) -> str:
    # The ruleType values (radar_*_rain) are machine identifiers consumed by the
    # Audiom accessible-mapping tool for styling. They remain unchanged despite
    # the human-readable labels becoming phase-neutral (rain -> precipitation).
    # ARW cannot distinguish rain from snow without melting-layer height (deferred
    # to Spec 4), so labels stop asserting phase. The ruleType values stay stable
    # to preserve existing Audiom map styling (renaming them would silently break
    # the user's styling configuration).
    return {
        "light precipitation": "radar_light_rain",
        "moderate precipitation": "radar_moderate_rain",
        "heavy precipitation": "radar_heavy_rain",
        "intense precipitation": "radar_intense_rain",
        "severe core": "radar_severe_core",
    }.get(label, "radar_echo")


def _storm_name(obj: DetectedObject) -> str:
    distance_mi = km_to_miles(obj.distance_km)
    bearing = degrees_to_bearing(obj.bearing_deg)
    return f"{obj.peak_label.capitalize()} storm {distance_mi} miles {bearing}"


def _storm_description(obj: DetectedObject, rotation_strength: str | None) -> str:
    area_mi2 = km2_to_mi2(obj.area_km2)
    parts = [
        f"Peak reflectivity {obj.peak_dbz} dBZ.",
        f"Covers about {area_mi2} square miles.",
        f"Centroid is {km_to_miles(obj.distance_km)} miles {degrees_to_bearing(obj.bearing_deg)} of the radar.",
    ]
    if rotation_strength:
        parts.append(f"{rotation_strength.capitalize()} rotation detected.")
    return " ".join(parts)


def _timestamp_to_str(timestamp) -> str:
    if hasattr(timestamp, "isoformat"):
        return timestamp.isoformat()
    return str(timestamp)


def _object_footprint(scan: BufferedScan, obj: DetectedObject):
    """Shapely geometry of this object's own footprint, or None.

    Shared by `object_geometry` (which maps it to GeoJSON) and
    `_object_bands` (which uses it to clip each intensity band to this
    object's own silhouette so a band cannot bleed into a neighbouring
    object at the same intensity).
    """
    mask = scan.object_masks.get(obj.object_id)
    if mask is None:
        return None
    geom = contour_mask(mask, scan.reflectivity_data)
    if geom.is_empty:
        return None
    return geom


def _object_footprint_raw(scan: BufferedScan, obj: DetectedObject):
    """Unsimplified shapely geometry of this object's own footprint, or None.

    Used only to clip intensity bands (`_object_bands`), never for display.

    Object masks are disjoint at the raster level: `detect_objects_with_grid`
    labels each gate to at most one object, so two objects' boolean masks
    never share a True gate. Contouring a mask at `simplify_m=0.0` produces
    the exact (gate-hugging) boundary with no Douglas-Peucker displacement,
    so two objects' raw footprints stay exactly as disjoint as their masks
    were -- they can share an edge (zero-width, zero-area) but never enclose
    overlapping area.

    `object_geometry`/`_object_footprint` simplify each object's footprint
    independently (`DEFAULT_SIMPLIFY_M`) for a nicer displayed outline, and
    that is exactly where cross-object overlap was coming from: independent
    Douglas-Peucker on two adjacent, disjoint boundaries can bulge each one
    toward the other by up to the tolerance, the same failure mode Task 2
    fixed for same-object bands by simplifying them jointly
    (`coverage_simplify`). There is no such joint pass across *different*
    objects' footprints, so clipping uses the exact, unsimplified contour
    instead, which carries no bulge to begin with rather than needing one
    cancelled out.
    """
    mask = scan.object_masks.get(obj.object_id)
    if mask is None:
        return None
    geom = contour_mask(mask, scan.reflectivity_data, simplify_m=0.0)
    if geom.is_empty:
        return None
    return geom


def object_geometry(scan: BufferedScan, obj: DetectedObject) -> dict[str, Any] | None:
    """GeoJSON geometry for a detected object, or None if it has no valid shape.

    Returns None rather than inventing a placeholder. The previous
    implementation drew a square at the centroid when hulling failed, which
    presented a fabricated shape as measurement to a user who explores it by
    walking its surface.
    """
    geom = _object_footprint(scan, obj)
    if geom is None:
        return None
    return mapping(geom)


def _as_multipolygon(geom) -> MultiPolygon:
    """Coerce a shapely geometry to a MultiPolygon.

    `exclusive_bands` output intersected with an object footprint is
    ordinarily still Polygon/MultiPolygon, but a boundary-hugging
    intersection can in principle degenerate into a GeometryCollection
    holding stray points or lines along with the real area. Those
    lower-dimensional slivers are dropped here rather than left to trip up
    downstream GeoJSON consumers that expect only polygonal geometry.
    """
    if geom is None or geom.is_empty:
        return MultiPolygon()
    if geom.geom_type == "Polygon":
        return MultiPolygon([geom])
    if geom.geom_type == "MultiPolygon":
        return geom
    if geom.geom_type == "GeometryCollection":
        parts: list = []
        for part in geom.geoms:
            if part.geom_type == "Polygon":
                parts.append(part)
            elif part.geom_type == "MultiPolygon":
                parts.extend(part.geoms)
        return MultiPolygon(parts)
    return MultiPolygon()


def _object_bands(
    scan: BufferedScan, obj: DetectedObject
) -> dict[tuple[float, float], MultiPolygon]:
    """This object's intensity bands: exclusive, and clipped to its own footprint.

    `exclusive_bands` (src/contours.py) contours the whole scan's
    reflectivity field at once, so a raw band's geometry can include other
    objects elsewhere in the sweep that happen to sit at the same
    intensity. Intersecting each band with this object's own footprint
    confines most of that leakage to this storm before it becomes a
    feature. The footprint used here is the *unsimplified* one
    (`_object_footprint_raw`), not the display footprint `object_geometry`
    draws -- see that function's docstring for why: an independently
    simplified footprint can bulge toward a neighbouring object and let two
    different objects' clipped bands overlap.

    What this guarantees, precisely:

    - Two bands of the SAME object never overlap. `exclusive_bands` already
      guarantees bands are interior-disjoint (area(A intersect B) == 0), and
      for any one fixed footprint C, area((A^C) intersect (B^C)) ==
      area(A intersect B intersect C) <= area(A intersect B) == 0.
      Intersecting with a shared footprint can only shrink each band, never
      make two disjoint bands overlap. (Proved algebraically; not dependent
      on which footprint -- raw or simplified -- is used.)
    - Two bands belonging to DIFFERENT objects also do not overlap, because
      each is clipped to its own object's raw, unsimplified footprint, and
      those footprints are themselves disjoint (see
      `_object_footprint_raw`). This is NOT true of the simplified
      footprint `object_geometry` draws for display -- do not swap this
      call for that one.
    """
    footprint = _object_footprint_raw(scan, obj)
    if footprint is None or not obj.layers:
        return {}

    levels: set[float] = set()
    for layer in obj.layers:
        levels.add(float(layer.min_dbz))
        if layer.max_dbz != float("inf"):
            levels.add(float(layer.max_dbz))

    raw_bands = exclusive_bands(scan.reflectivity_data.reflectivity, levels, scan.reflectivity_data)
    return {
        key: _as_multipolygon(geom.intersection(footprint))
        for key, geom in raw_bands.items()
    }


def _layer_geometry(
    bands: dict[tuple[float, float], MultiPolygon], layer: IntensityLayerData
) -> dict[str, Any] | None:
    """GeoJSON geometry for one intensity band within an object, or None.

    Same no-invented-shape contract as `object_geometry`: a band that yields
    no valid contour (e.g. its mask is empty or the parent object has no
    mask) is omitted rather than drawn as a fallback shape.
    """
    geom = bands.get((float(layer.min_dbz), float(layer.max_dbz)))
    if geom is None or geom.is_empty:
        return None
    return mapping(geom)


def _storm_properties(scan: BufferedScan, obj: DetectedObject) -> dict[str, Any]:
    rotation = getattr(obj, "rotation", None)
    rotation_strength = rotation.strength if rotation is not None else None
    return {
        "object_id": obj.object_id,
        "id": f"{scan.site_id}-{obj.object_id}",
        "name": _storm_name(obj),
        "ruleName": "Storm polygon",
        "ruleType": _storm_rule_type(obj.peak_dbz),
        "description": _storm_description(obj, rotation_strength),
        "passable": True,
        "soundPriority": 500,
        "fill": _storm_fill_color(obj.peak_dbz),
        "stroke": "#202020",
        "stroke-contrast": "#000000",
        "stroke-width": 2,
        "fill-opacity": 0.55,
        "minstep": "100m",
        "maxstep": "300mi",
        "site_id": scan.site_id,
        "timestamp": _timestamp_to_str(scan.reflectivity_data.timestamp),
        "centroid_lat": obj.centroid_lat,
        "centroid_lon": obj.centroid_lon,
        "distance_km": obj.distance_km,
        "distance_mi": km_to_miles(obj.distance_km),
        "bearing_deg": obj.bearing_deg,
        "bearing_label": degrees_to_bearing(obj.bearing_deg),
        "peak_dbz": obj.peak_dbz,
        "peak_label": obj.peak_label,
        "area_km2": obj.area_km2,
        "area_mi2": km2_to_mi2(obj.area_km2),
        "heat_value": obj.peak_dbz,
        "rotation_strength": rotation_strength,
        "max_inbound_ms": getattr(obj, "max_inbound_ms", None),
        "max_outbound_ms": getattr(obj, "max_outbound_ms", None),
        "layers": [layer.__dict__ for layer in obj.layers],
        # QC class composition (precipitation/ground_clutter/biological/hail/
        # debris/unknown fractions). Quality control no longer deletes echo
        # (2026-08-26 amendment), so this is how Audiom and the speech layer
        # tell a mostly-clutter polygon apart from a mostly-precipitation one.
        "class_fractions": dict(getattr(obj, "class_fractions", None) or {}),
    }


def storm_object_to_feature(scan: BufferedScan, obj: DetectedObject) -> dict[str, Any] | None:
    """A GeoJSON Feature for this object, or None if it has no valid shape.

    Returning None (rather than a fabricated placeholder) means callers must
    filter -- see `build_storm_geojson` and `build_storm_audiom_geojson`,
    which count what they skip so the omission is visible in the layer
    metadata instead of silently vanishing.
    """
    geometry = object_geometry(scan, obj)
    if geometry is None:
        return None
    return {
        "type": "Feature",
        "id": obj.object_id,
        "geometry": geometry,
        "properties": _storm_properties(scan, obj),
    }


def storm_intensity_layer_to_feature(
    scan: BufferedScan,
    obj: DetectedObject,
    layer: IntensityLayerData,
    bands: dict[tuple[float, float], MultiPolygon],
) -> dict[str, Any] | None:
    """A GeoJSON Feature for this intensity band, or None if it has no valid shape.

    `bands` is this object's own exclusive-band geometry (see
    `_object_bands`), precomputed once per object rather than once per layer
    since `exclusive_bands` contours the whole field in a single pass.
    """
    geometry = _layer_geometry(bands, layer)
    if geometry is None:
        return None

    heat_value = _intensity_heat_value(layer)
    area_mi2 = km2_to_mi2(layer.area_km2)
    max_dbz = None if layer.max_dbz == float("inf") else layer.max_dbz
    properties = {
        "object_id": obj.object_id,
        "parent_object_id": obj.object_id,
        "id": f"{scan.site_id}-{obj.object_id}-{layer.label.replace(' ', '-')}",
        "name": f"{layer.label.capitalize()} band in {_storm_name(obj)}",
        "ruleName": "Radar intensity band",
        "ruleType": _intensity_rule_type(layer.label),
        "description": (
            f"{layer.label.capitalize()} reflectivity band from {layer.min_dbz} "
            f"to {max_dbz if max_dbz is not None else '60+'} dBZ. "
            f"Covers about {area_mi2} square miles."
        ),
        "passable": True,
        "soundPriority": 650,
        "fill": _intensity_fill_color(layer.label),
        "stroke": _intensity_fill_color(layer.label),
        "stroke-contrast": "#000000",
        "stroke-width": 1,
        "fill-opacity": 0.64,
        "minstep": "100m",
        "maxstep": "300mi",
        "site_id": scan.site_id,
        "timestamp": _timestamp_to_str(scan.reflectivity_data.timestamp),
        "centroid_lat": obj.centroid_lat,
        "centroid_lon": obj.centroid_lon,
        "distance_km": obj.distance_km,
        "distance_mi": km_to_miles(obj.distance_km),
        "bearing_deg": obj.bearing_deg,
        "bearing_label": degrees_to_bearing(obj.bearing_deg),
        "peak_dbz": obj.peak_dbz,
        "peak_label": obj.peak_label,
        "intensity_label": layer.label,
        "min_dbz": layer.min_dbz,
        "max_dbz": max_dbz,
        "area_km2": layer.area_km2,
        "area_mi2": area_mi2,
        "heat_value": heat_value,
    }
    return {
        "type": "Feature",
        "id": properties["id"],
        "geometry": geometry,
        "properties": properties,
    }


def storm_object_to_centroid_feature(scan: BufferedScan, obj: DetectedObject) -> dict[str, Any]:
    # Deliberately independent of object_geometry: a centroid is known even
    # for an object whose footprint has no valid contour, and the centroid
    # layer must not lose that object just because its shape did.
    return {
        "type": "Feature",
        "id": obj.object_id,
        "geometry": {
            "type": "Point",
            "coordinates": [obj.centroid_lon, obj.centroid_lat],
        },
        "properties": _storm_properties(scan, obj),
    }


def _storm_metadata(name: str = "ARW storm polygons") -> dict[str, Any]:
    return {
        "mapType": "heatmap",
        "suggestedMapType": "heatmap",
        "coordinateSystem": "standard",
        "name": name,
        "sourceName": "arw-storms",
        "currentStat": "heat_value",
        "displayField": "name",
        "dataProperties": [
            {"field": "heat_value", "displayName": "Reflectivity dBZ", "type": "number"},
            {"field": "peak_dbz", "displayName": "Parent storm peak dBZ", "type": "number"},
            {"field": "area_mi2", "displayName": "Area square miles", "type": "number"},
            {"field": "distance_mi", "displayName": "Distance miles", "type": "number"},
        ],
        "suggestedRefreshInterval": 300,
    }


def build_storm_geojson(scan: BufferedScan) -> dict[str, Any]:
    features = []
    omitted = 0
    for obj in scan.detected_objects:
        feature = storm_object_to_feature(scan, obj)
        if feature is None:
            omitted += 1
            continue
        features.append(feature)
    metadata = _storm_metadata()
    # A detected object with no valid contour (empty or degenerate mask) is
    # left out of `features` entirely rather than drawn as a placeholder --
    # this count is how that omission stays visible instead of silent.
    metadata["omittedObjectCount"] = omitted
    return {
        "type": "FeatureCollection",
        "metadata": metadata,
        "features": features,
    }


def build_storm_intensity_geojson(scan: BufferedScan) -> dict[str, Any]:
    features = []
    omitted = 0
    for obj in scan.detected_objects:
        bands = _object_bands(scan, obj)
        for layer in obj.layers:
            feature = storm_intensity_layer_to_feature(scan, obj, layer, bands)
            if feature is None:
                omitted += 1
                continue
            features.append(feature)
    metadata = _storm_metadata("ARW radar intensity bands")
    metadata["omittedBandCount"] = omitted
    return {
        "type": "FeatureCollection",
        "metadata": metadata,
        "features": features,
    }


def build_storm_audiom_geojson(scan: BufferedScan) -> dict[str, Any]:
    intensity_geojson = build_storm_intensity_geojson(scan)
    footprint_geojson = build_storm_geojson(scan)
    footprint_features = []
    for feature in footprint_geojson["features"]:
        feature = dict(feature)
        feature["properties"] = {
            **feature["properties"],
            "ruleName": "Storm footprint",
            "ruleType": f"{feature['properties']['ruleType']}_footprint",
            "fill-opacity": 0.12,
            "stroke-width": 3,
            "soundPriority": 450,
        }
        footprint_features.append(feature)
    metadata = _storm_metadata("ARW radar reflectivity")
    metadata["omittedObjectCount"] = footprint_geojson["metadata"]["omittedObjectCount"]
    metadata["omittedBandCount"] = intensity_geojson["metadata"]["omittedBandCount"]
    return {
        "type": "FeatureCollection",
        "metadata": metadata,
        "features": footprint_features + intensity_geojson["features"],
    }


def build_storm_centroid_geojson(scan: BufferedScan) -> dict[str, Any]:
    return {
        "type": "FeatureCollection",
        "metadata": {
            "mapType": "travel",
            "coordinateSystem": "standard",
            "name": "ARW storm centroids",
            "sourceName": "arw-storm-centroids",
            "suggestedRefreshInterval": 300,
        },
        "features": [storm_object_to_centroid_feature(scan, obj) for obj in scan.detected_objects],
    }


def storm_layer_fields() -> list[dict[str, str]]:
    return [
        {"name": "object_id", "type": "esriFieldTypeInteger", "alias": "Object ID"},
        {"name": "site_id", "type": "esriFieldTypeString", "alias": "Radar Site"},
        {"name": "timestamp", "type": "esriFieldTypeString", "alias": "Scan Time"},
        {"name": "peak_dbz", "type": "esriFieldTypeDouble", "alias": "Peak dBZ"},
        {"name": "peak_label", "type": "esriFieldTypeString", "alias": "Intensity"},
        {"name": "area_km2", "type": "esriFieldTypeDouble", "alias": "Area sq km"},
        {"name": "heat_value", "type": "esriFieldTypeDouble", "alias": "Heat Value"},
        {"name": "rotation_strength", "type": "esriFieldTypeString", "alias": "Rotation"},
    ]


def storm_layer_drawing_info() -> dict[str, Any]:
    return {
        "renderer": {
            "type": "classBreaks",
            "field": "heat_value",
            "classBreakInfos": [
                {"classMaxValue": 30, "label": "Light precipitation", "symbol": {"color": [86, 180, 233, 120]}},
                {"classMaxValue": 40, "label": "Moderate precipitation", "symbol": {"color": [0, 158, 115, 140]}},
                {"classMaxValue": 50, "label": "Heavy precipitation", "symbol": {"color": [240, 228, 66, 160]}},
                {"classMaxValue": 60, "label": "Intense precipitation", "symbol": {"color": [230, 159, 0, 180]}},
                {"classMaxValue": 90, "label": "Severe core", "symbol": {"color": [213, 94, 0, 200]}},
            ],
        }
    }
