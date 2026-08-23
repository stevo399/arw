from typing import Any

import numpy as np
from scipy.spatial import ConvexHull, QhullError

from src.buffer import BufferedScan
from src.detection import DetectedObject, IntensityLayerData, degrees_to_bearing
from src.geometry import gate_latlon
from src.summary import km2_to_mi2, km_to_miles


MAX_HULL_POINTS = 240


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
        "light rain": "#2ca25f",
        "moderate rain": "#ffff66",
        "heavy rain": "#ffcc33",
        "intense rain": "#ff6600",
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
    return {
        "light rain": "radar_light_rain",
        "moderate rain": "radar_moderate_rain",
        "heavy rain": "radar_heavy_rain",
        "intense rain": "radar_intense_rain",
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


def _sample_mask_points(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    az_indices, range_indices = np.where(mask)
    if len(az_indices) <= MAX_HULL_POINTS:
        return az_indices, range_indices
    step = max(1, len(az_indices) // MAX_HULL_POINTS)
    return az_indices[::step], range_indices[::step]


def _fallback_square(lon: float, lat: float, area_km2: float) -> list[list[float]]:
    half_km = max(area_km2 ** 0.5 / 2.0, 0.5)
    lat_delta = half_km / 111.0
    lon_delta = half_km / max(111.0 * np.cos(np.radians(lat)), 1.0)
    return [
        [round(lon - lon_delta, 6), round(lat - lat_delta, 6)],
        [round(lon + lon_delta, 6), round(lat - lat_delta, 6)],
        [round(lon + lon_delta, 6), round(lat + lat_delta, 6)],
        [round(lon - lon_delta, 6), round(lat + lat_delta, 6)],
        [round(lon - lon_delta, 6), round(lat - lat_delta, 6)],
    ]


def mask_to_polygon(
    scan: BufferedScan,
    mask: np.ndarray,
    fallback_lon: float,
    fallback_lat: float,
    fallback_area_km2: float,
) -> list[list[float]]:
    if mask is None or not np.any(mask):
        return _fallback_square(fallback_lon, fallback_lat, fallback_area_km2)

    az_indices, range_indices = _sample_mask_points(mask)
    coordinates = []
    ref = scan.reflectivity_data
    for az_idx, range_idx in zip(az_indices, range_indices):
        lat, lon = gate_latlon(
            azimuth_deg=float(ref.azimuths[az_idx]),
            range_m=float(ref.ranges_m[range_idx]),
            elevation_deg=float(ref.elevations[az_idx]),
            radar_lat=ref.radar_lat,
            radar_lon=ref.radar_lon,
        )
        coordinates.append([float(lon), float(lat)])

    unique = np.unique(np.array(coordinates), axis=0)
    if len(unique) < 3:
        return _fallback_square(fallback_lon, fallback_lat, fallback_area_km2)

    try:
        hull = ConvexHull(unique)
    except QhullError:
        return _fallback_square(fallback_lon, fallback_lat, fallback_area_km2)

    ring = [[round(float(unique[idx][0]), 6), round(float(unique[idx][1]), 6)] for idx in hull.vertices]
    if ring[0] != ring[-1]:
        ring.append(ring[0])
    return ring


def object_mask_to_polygon(scan: BufferedScan, obj: DetectedObject) -> list[list[float]]:
    mask = scan.object_masks.get(obj.object_id)
    return mask_to_polygon(scan, mask, obj.centroid_lon, obj.centroid_lat, obj.area_km2)


def storm_object_to_feature(scan: BufferedScan, obj: DetectedObject) -> dict[str, Any]:
    rotation = getattr(obj, "rotation", None)
    rotation_strength = rotation.strength if rotation is not None else None
    properties = {
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
    }
    return {
        "type": "Feature",
        "id": obj.object_id,
        "geometry": {
            "type": "Polygon",
            "coordinates": [object_mask_to_polygon(scan, obj)],
        },
        "properties": properties,
    }


def storm_intensity_layer_to_feature(
    scan: BufferedScan,
    obj: DetectedObject,
    layer: IntensityLayerData,
) -> dict[str, Any]:
    object_mask = scan.object_masks.get(obj.object_id)
    if object_mask is None:
        layer_mask = None
    else:
        reflectivity = scan.reflectivity_data.reflectivity
        layer_mask = object_mask & ~np.isnan(reflectivity) & (reflectivity >= layer.min_dbz)
        if layer.max_dbz != float("inf"):
            layer_mask = layer_mask & (reflectivity < layer.max_dbz)

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
        "geometry": {
            "type": "Polygon",
            "coordinates": [
                mask_to_polygon(
                    scan,
                    layer_mask,
                    obj.centroid_lon,
                    obj.centroid_lat,
                    layer.area_km2,
                )
            ],
        },
        "properties": properties,
    }


def storm_object_to_centroid_feature(scan: BufferedScan, obj: DetectedObject) -> dict[str, Any]:
    polygon_feature = storm_object_to_feature(scan, obj)
    return {
        "type": "Feature",
        "id": obj.object_id,
        "geometry": {
            "type": "Point",
            "coordinates": [obj.centroid_lon, obj.centroid_lat],
        },
        "properties": polygon_feature["properties"],
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
    return {
        "type": "FeatureCollection",
        "metadata": _storm_metadata(),
        "features": [storm_object_to_feature(scan, obj) for obj in scan.detected_objects],
    }


def build_storm_intensity_geojson(scan: BufferedScan) -> dict[str, Any]:
    features = []
    for obj in scan.detected_objects:
        for layer in obj.layers:
            features.append(storm_intensity_layer_to_feature(scan, obj, layer))
    return {
        "type": "FeatureCollection",
        "metadata": _storm_metadata("ARW radar intensity bands"),
        "features": features,
    }


def build_storm_audiom_geojson(scan: BufferedScan) -> dict[str, Any]:
    intensity_features = build_storm_intensity_geojson(scan)["features"]
    footprint_features = []
    for feature in build_storm_geojson(scan)["features"]:
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
    return {
        "type": "FeatureCollection",
        "metadata": _storm_metadata("ARW radar reflectivity"),
        "features": footprint_features + intensity_features,
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
                {"classMaxValue": 30, "label": "Light rain", "symbol": {"color": [86, 180, 233, 120]}},
                {"classMaxValue": 40, "label": "Moderate rain", "symbol": {"color": [0, 158, 115, 140]}},
                {"classMaxValue": 50, "label": "Heavy rain", "symbol": {"color": [240, 228, 66, 160]}},
                {"classMaxValue": 60, "label": "Intense rain", "symbol": {"color": [230, 159, 0, 180]}},
                {"classMaxValue": 90, "label": "Severe core", "symbol": {"color": [213, 94, 0, 200]}},
            ],
        }
    }
