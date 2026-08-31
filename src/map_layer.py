from typing import Any

import numpy as np
from shapely.geometry import mapping

from src.buffer import BufferedScan
from src.contours import contour_mask
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


def object_geometry(scan: BufferedScan, obj: DetectedObject) -> dict[str, Any] | None:
    """GeoJSON geometry for a detected object, or None if it has no valid shape.

    Returns None rather than inventing a placeholder. The previous
    implementation drew a square at the centroid when hulling failed, which
    presented a fabricated shape as measurement to a user who explores it by
    walking its surface.
    """
    mask = scan.object_masks.get(obj.object_id)
    if mask is None:
        return None
    geom = contour_mask(mask, scan.reflectivity_data)
    if geom.is_empty:
        return None
    return mapping(geom)


def _layer_geometry(
    scan: BufferedScan, obj: DetectedObject, layer: IntensityLayerData
) -> dict[str, Any] | None:
    """GeoJSON geometry for one intensity band within an object, or None.

    Same no-invented-shape contract as `object_geometry`: a band that yields
    no valid contour (e.g. its mask is empty or the parent object has no
    mask) is omitted rather than drawn as a fallback shape.
    """
    object_mask = scan.object_masks.get(obj.object_id)
    if object_mask is None:
        return None
    reflectivity = scan.reflectivity_data.reflectivity
    layer_mask = object_mask & ~np.isnan(reflectivity) & (reflectivity >= layer.min_dbz)
    if layer.max_dbz != float("inf"):
        layer_mask = layer_mask & (reflectivity < layer.max_dbz)
    geom = contour_mask(layer_mask, scan.reflectivity_data)
    if geom.is_empty:
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
) -> dict[str, Any] | None:
    """A GeoJSON Feature for this intensity band, or None if it has no valid shape."""
    geometry = _layer_geometry(scan, obj, layer)
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
        for layer in obj.layers:
            feature = storm_intensity_layer_to_feature(scan, obj, layer)
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
