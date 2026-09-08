from typing import Any

import numpy as np
from pyart.core.transforms import geographic_to_cartesian_aeqd
from shapely.geometry import MultiPolygon, mapping
from shapely.ops import transform

from src.buffer import BufferedScan
from src.contours import contour_mask, exclusive_bands
from src.detection import DetectedObject, IntensityLayerData, classify_intensity, degrees_to_bearing
from src.geometry import gate_areas_km2
from src.qc.classifier import CODE_TO_CLASS
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
        "drizzle": "radar_drizzle",
        "light precipitation": "radar_light_rain",
        "moderate precipitation": "radar_moderate_rain",
        "heavy precipitation": "radar_heavy_rain",
        "intense precipitation": "radar_intense_rain",
        "severe core": "radar_severe_core",
    }.get(label, "radar_echo")


def _storm_name(obj: DetectedObject) -> str:
    distance_mi = km_to_miles(obj.distance_km)
    bearing = degrees_to_bearing(obj.bearing_deg)
    phenomenon = _object_phenomenon(obj)
    intensity = _display_intensity_label(obj.peak_label, phenomenon)
    return f"{intensity.capitalize()} {phenomenon} {distance_mi} miles {bearing}"


def _object_phenomenon(obj: DetectedObject) -> str:
    """Return the strongest supportable phenomenon label for an object.

    The classifier operates on radar gates, not observations at the surface.
    Keep every echo available for analysis, but make the spoken map label
    describe the confidence and scope of the classification rather than state
    that precipitation is occurring at the ground.
    """
    fractions = obj.class_fractions or {}
    precipitation = fractions.get("precipitation", 0.0)
    dominant = max(fractions, key=fractions.get) if fractions else None
    if dominant == "ground_clutter" and fractions[dominant] >= 0.5:
        return "ground clutter echo"
    if dominant == "biological" and fractions[dominant] >= 0.5:
        return "biological echo"
    # A large, internally consistent majority supports a radar-gate
    # classification, but it still does not establish phase or that hydrometeors
    # are reaching the surface.  Any appreciable competing classification keeps
    # the wording explicitly unconfirmed.
    competing = 1.0 - precipitation
    if precipitation >= 0.9 and competing < 0.1:
        return "QC-classified precipitation echo"
    if precipitation >= 0.7:
        return "unconfirmed precipitation-classified echo"
    return "uncertain reflectivity echo"


def _display_intensity_label(label: str, phenomenon: str) -> str:
    # The intensity buckets were historically named for precipitation.  They
    # are reflectivity ranges, so spoken labels must remain phase-neutral even
    # when QC finds precipitation-like gates.
    return label.replace(" precipitation", " reflectivity").replace("severe core", "severe reflectivity core")


def _storm_description(obj: DetectedObject, rotation) -> str:
    area_mi2 = km2_to_mi2(obj.area_km2)
    phenomenon = _object_phenomenon(obj)
    parts = [
        f"QC interpretation: {phenomenon}.",
        f"Peak reflectivity {obj.peak_dbz} dBZ.",
        f"Covers about {area_mi2} square miles.",
        f"Centroid is {km_to_miles(obj.distance_km)} miles {degrees_to_bearing(obj.bearing_deg)} of the radar.",
    ]
    if "precipitation" in phenomenon:
        precipitation = (obj.class_fractions or {}).get("precipitation", 0.0)
        parts.append(
            f"{round(precipitation * 100)} percent of classified radar gates were precipitation-like; "
            "this does not confirm precipitation at the surface."
        )
    if rotation is not None:
        reference = "base-radial velocity" if rotation.motion_reference == "base_radial" else rotation.motion_reference
        if rotation.evidence_level == "persistent":
            parts.append(f"Persistent {rotation.strength} rotation evidence in {reference}.")
        elif rotation.evidence_level == "vertically_confirmed":
            parts.append(f"{rotation.strength.capitalize()} rotation evidence vertically confirmed in {reference}.")
        elif rotation.evidence_level == "corroborated":
            parts.append(f"{rotation.strength.capitalize()} corroborated rotation evidence in {reference}.")
        else:
            parts.append(f"Unconfirmed {rotation.strength} velocity couplet in {reference}.")
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
        "description": _storm_description(obj, rotation),
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
        "rotation_evidence_level": rotation.evidence_level if rotation is not None else None,
        "rotation_motion_reference": rotation.motion_reference if rotation is not None else None,
        "rotation_context_peak_dbz": rotation.associated_object_peak_dbz if rotation is not None else None,
        "rotation_dual_pol_available": rotation.dual_pol_available if rotation is not None else None,
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
    display_label = _display_intensity_label(layer.label, _object_phenomenon(obj))
    properties = {
        "object_id": obj.object_id,
        "parent_object_id": obj.object_id,
        "id": f"{scan.site_id}-{obj.object_id}-{layer.label.replace(' ', '-')}",
        "name": f"{display_label.capitalize()} band in {_storm_name(obj)}",
        "ruleName": "Radar intensity band",
        "ruleType": _intensity_rule_type(layer.label),
        "description": (
            f"{display_label.capitalize()} band from {layer.min_dbz} "
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


PRECIP_FIELD_LEVELS = (15.0, 20.0, 30.0, 40.0, 50.0, 60.0)

# ARW-tuned. Below this median correlation coefficient no class label is
# supportable: on clear-air volumes both precipitation and biological classes
# collapse to 0.58-0.64, indistinguishable and far too low to be rain.
UNCERTAIN_RHOHV = 0.85

# Measured on cache/KTLX/KTLX20130520_195527_V06.gz (Newcastle-Moore,
# 2013-05-20 19:55:27), pre-filter (this constant disabled): this layer
# contoured 7,535 separate polygon pieces (127,049 vertices total, counting
# len(exterior.coords) plus every interior ring's coords per piece, no
# closure-duplicate subtracted). Re-derived 2026-09-01 against this same
# commit's own code via docs/test_reports/2026-08-31-spec2-vertex-measurement.py
# (see docs/test_reports/2026-08-31-spec2-shape-truth.md, Finding 4) --
# the previous figure here (7,876 pieces / 131,472 vertices) came from a
# measurement taken at an earlier commit and does not reproduce; this one
# does. The size distribution is extremely lopsided --
# 93% of pieces carry only 2.6% of the total area, and are mostly under
# ~700 m across: isolated single/few-gate speckle, not real weather.
# Raising DEFAULT_SIMPLIFY_M does not help (still 7,535 pieces at 2000 m):
# simplification smooths a piece's outline, it does not merge pieces
# together, so the complexity here is fragment COUNT, not smoothness.
# 0.5 km2 is roughly 700 m across -- eight times smaller than the 4 km2
# object-detection threshold this layer exists to bypass -- chosen so a
# genuine small shower still survives while isolated speckle does not.
# Project-owner decision, 2026-08-31; do not raise or lower this thinking
# it arbitrary without re-measuring the fragment-size distribution first.
# Applies to the precipitation-field layer only -- the storm layer's own
# polygons are already well-behaved (median 40 vertices, p95 823 on the
# same volume) and are governed by a different contract (object detection's
# 4 km2 area floor), so they are never touched by this constant.
MIN_PRECIP_FRAGMENT_AREA_KM2 = 0.5


def _fragment_area_km2(polygon, radar_lat: float, radar_lon: float) -> float:
    """A single polygon piece's ground area in square kilometres.

    Projects to the same radar-centred azimuthal-equidistant frame
    `src/contours.py` uses (`geographic_to_cartesian_aeqd`), so this agrees
    with that module's own notion of ground distance rather than computing
    area in degrees^2, which is anisotropic and means nothing on its own.
    """
    def to_metres(x, y, z=None):
        return geographic_to_cartesian_aeqd(np.asarray(x), np.asarray(y), radar_lon, radar_lat)

    return transform(to_metres, polygon).area / 1e6


def _drop_small_fragments(
    geometry: MultiPolygon, radar_lat: float, radar_lon: float
) -> tuple[MultiPolygon, int, float]:
    """Split a band's geometry into its separate polygon pieces and drop any
    piece below `MIN_PRECIP_FRAGMENT_AREA_KM2`.

    Returns (kept_geometry, dropped_count, dropped_area_km2) -- the caller
    accumulates the latter two into layer metadata so a dropped fragment is
    counted, never silently discarded (the same principle the storm layer
    already applies via `omittedObjectCount`/`omittedBandCount`).
    """
    if geometry.is_empty:
        return geometry, 0, 0.0
    pieces = list(geometry.geoms) if geometry.geom_type == "MultiPolygon" else [geometry]

    kept = []
    dropped_count = 0
    dropped_area_km2 = 0.0
    for piece in pieces:
        area_km2 = _fragment_area_km2(piece, radar_lat, radar_lon)
        if area_km2 < MIN_PRECIP_FRAGMENT_AREA_KM2:
            dropped_count += 1
            dropped_area_km2 += area_km2
        else:
            kept.append(piece)
    return MultiPolygon(kept), dropped_count, dropped_area_km2


def _gate_area_grid(sweep) -> np.ndarray:
    """Ground area of every gate in the sweep, shaped like the field.

    Every gate in a given range ring has the same ground footprint whatever
    its azimuth, so `gate_areas_km2`'s one-value-per-range-bin result
    broadcasts across the rays -- the same way `src.shape_truth` uses it.
    """
    per_range_bin = gate_areas_km2(sweep.azimuths, sweep.ranges_m, sweep.elevation_angle)
    return np.broadcast_to(
        per_range_bin[None, :], np.asarray(sweep.reflectivity).shape
    )


def _band_label(lower: float, upper: float) -> str:
    return f"{lower:g}+" if upper == float("inf") else f"{lower:g}-{upper:g}"


def _band_gate_mask(sweep, lower: float, upper: float) -> np.ndarray:
    """Boolean mask of gates whose reflectivity genuinely falls in [lower, upper).

    Shared by `build_precipitation_field_geojson` (to decide whether a band's
    contoured geometry is backed by any real gate at all) and `_band_evidence`
    (to compute that band's evidence from exactly those gates), so the two
    can never disagree about which gates support a band.

    HALF-OPEN, CLOSED AT THE BOTTOM -- `lower <= v < upper` -- which is the
    convention `src.detection.classify_intensity` and
    `src.shape_truth.measure_walk_truth` already use, and which
    `src.contours` was made to match (see `LEVEL_MEMBERSHIP_EPSILON` there
    for why the geometry was the side that moved and what it cost). Before
    that, the geometry used contourpy's strict `>` while this used `>=`, so
    a gate sitting exactly on a level was in the evidence a band reported
    and outside the polygon that evidence was attached to: 5,912 such gates
    on KTLX, 8,748 on KEMX, 6.5-13.6% of each volume's >= 15 dBZ ground.
    NEXRAD reflectivity is quantized to 0.5 dBZ, so exact hits on integer
    band levels are guaranteed, not incidental. Do not change one side of
    this without the other.
    """
    field = np.asarray(sweep.reflectivity, dtype=float)
    in_band = np.isfinite(field) & (field >= lower)
    if upper != float("inf"):
        in_band &= field < upper
    return in_band


def _band_evidence(sweep, lower: float, upper: float) -> dict[str, Any]:
    """Evidence attributes for one intensity band.

    Within a real storm, polarimetry separates rain from insects cleanly:
    precipitation sits near rhohv 0.99 with zdr around +1.4, biological near
    rhohv 0.89 with zdr above +4. In clear air both collapse to rhohv ~0.6,
    where no label is supportable -- so the band reports uncertainty rather
    than guessing, and always carries the raw numbers so the call can be
    checked.
    """
    in_band = _band_gate_mask(sweep, lower, upper)

    evidence: dict[str, Any] = {
        "median_rhohv": None,
        "median_zdr": None,
        "dominant_class": "uncertain",
        "class_fractions": {},
    }
    if not in_band.any():
        return evidence

    if sweep.rhohv is not None:
        values = np.asarray(sweep.rhohv, dtype=float)[in_band]
        values = values[np.isfinite(values)]
        if values.size:
            evidence["median_rhohv"] = round(float(np.median(values)), 4)
    if sweep.zdr is not None:
        values = np.asarray(sweep.zdr, dtype=float)[in_band]
        values = values[np.isfinite(values)]
        if values.size:
            evidence["median_zdr"] = round(float(np.median(values)), 4)

    if sweep.gate_classification is not None:
        codes = np.asarray(sweep.gate_classification)[in_band]
        total = float(codes.size)
        fractions = {
            name: round(float(np.count_nonzero(codes == code)) / total, 4)
            for code, name in CODE_TO_CLASS.items()
        }
        evidence["class_fractions"] = {k: v for k, v in fractions.items() if v > 0.0}

    median_rhohv = evidence["median_rhohv"]
    if median_rhohv is not None and median_rhohv >= UNCERTAIN_RHOHV and evidence["class_fractions"]:
        evidence["dominant_class"] = max(
            evidence["class_fractions"], key=evidence["class_fractions"].get
        )
    return evidence


def build_precipitation_field_geojson(scan: BufferedScan) -> dict[str, Any]:
    """The whole precipitation field, independent of object detection.

    Detection ignores echo below 20 dBZ and discards anything under 4 km2, so
    light rain and small showers are invisible on the storm layer. This layer
    has neither threshold.
    """
    sweep = scan.reflectivity_data
    bands = exclusive_bands(sweep.reflectivity, PRECIP_FIELD_LEVELS, sweep)
    gate_area = _gate_area_grid(sweep)

    features = []
    omitted_fragment_count = 0
    omitted_fragment_area_km2 = 0.0
    omitted_bands: list[dict[str, Any]] = []

    def _record_omission(lower, upper, reason, support):
        """Name a band that produced no feature, and say what it cost.

        Every path out of this loop that does not append a feature comes
        through here. A band can carry real weather and still be dropped --
        by an empty geometry, by having no supporting gate, or by losing
        every piece to the small-fragment filter -- and the standing rule
        for this layer is that a dropped shape is counted and discoverable,
        never silent. The gate count and gate-summed ground area travel with
        the record because a bare count cannot tell the project owner
        whether what vanished was a speckle or a severe core.
        """
        omitted_bands.append({
            "band": _band_label(lower, upper),
            "min_dbz": lower,
            "max_dbz": None if upper == float("inf") else upper,
            "reason": reason,
            "gateCount": int(support.sum()),
            "gateAreaKm2": round(float(gate_area[support].sum()), 4),
        })

    for (lower, upper), geometry in bands.items():
        support = _band_gate_mask(sweep, lower, upper)
        if geometry.is_empty:
            # No contour at all for this band. Usually it means there is no
            # such weather, in which case `support` is empty too and the
            # record below says so -- but if `support` is NOT empty, real
            # gates in this band produced no drawable shape, and that is a
            # thing the metadata has to be able to say out loud.
            _record_omission(lower, upper, "empty_geometry", support)
            continue
        if not support.any():
            # Geometric difference between two contour levels can leave a
            # floating-point sliver (observed directly: ~1.5e-6 sq deg on a
            # synthetic two-blob field, from the coverage-simplify/aeqd
            # round-trip in src/contours.py) even where no gate's own
            # reflectivity actually falls in this band. That sliver is not
            # precipitation -- reporting it as a shape would be exactly the
            # fabricated-shape failure this codebase already rejects
            # elsewhere (see object_geometry's docstring), just from a
            # different source. A band with no supporting gate is skipped
            # rather than emitted with an empty/"uncertain" evidence block.
            _record_omission(lower, upper, "no_supporting_gate", support)
            continue
        geometry, fragment_count, fragment_area_km2 = _drop_small_fragments(
            geometry, sweep.radar_lat, sweep.radar_lon
        )
        omitted_fragment_count += fragment_count
        omitted_fragment_area_km2 += fragment_area_km2
        if geometry.is_empty:
            # Every piece of this band was below MIN_PRECIP_FRAGMENT_AREA_KM2,
            # so the whole band disappears. The fragment counters alone
            # cannot say that: they aggregate across bands and cannot tell
            # you WHICH band went (measured on KIWA, where the 30-40 and
            # 40-50 dBZ bands both vanish this way, leaving that volume's
            # precipitation layer showing nothing at all above 30 dBZ).
            _record_omission(lower, upper, "all_fragments_below_area_floor", support)
            continue
        properties = {
            "id": f"{scan.site_id}-precip-{int(lower)}",
            "ruleName": "Precipitation field",
            "ruleType": _intensity_rule_type(classify_intensity(lower)),
            "min_dbz": lower,
            "max_dbz": None if upper == float("inf") else upper,
            "heat_value": lower,
            "site_id": scan.site_id,
            "timestamp": _timestamp_to_str(sweep.timestamp),
            "passable": True,
            "soundPriority": 700,
            "minstep": "100m",
            "maxstep": "300mi",
        }
        properties.update(_band_evidence(sweep, lower, upper))
        features.append({
            "type": "Feature",
            "id": properties["id"],
            "geometry": mapping(geometry),
            "properties": properties,
        })

    metadata = _storm_metadata("ARW precipitation field")
    # A fragment below MIN_PRECIP_FRAGMENT_AREA_KM2 is dropped rather than
    # displayed -- see that constant's comment for why. Both counts survive
    # into metadata rather than vanishing: a count alone would not say
    # whether the filter is behaving, so the area is reported too.
    metadata["omittedFragmentCount"] = omitted_fragment_count
    metadata["omittedFragmentAreaKm2"] = round(omitted_fragment_area_km2, 4)
    # And every whole band that produced no feature, with the reason it
    # produced none and the ground its own gates covered. Before this, two
    # paths out of the feature loop incremented nothing at all: a uniform
    # 60.0 dBZ severe core of 23 km2 came through this builder as zero
    # features with every counter reading zero.
    metadata["omittedBands"] = omitted_bands
    metadata["omittedBandCount"] = len(omitted_bands)
    metadata["omittedBandAreaKm2"] = round(
        sum(band["gateAreaKm2"] for band in omitted_bands), 4
    )
    return {
        "type": "FeatureCollection",
        "metadata": metadata,
        "features": features,
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
        {"name": "rotation_evidence_level", "type": "esriFieldTypeString", "alias": "Rotation Evidence"},
        {"name": "rotation_motion_reference", "type": "esriFieldTypeString", "alias": "Velocity Reference"},
        {"name": "rotation_context_peak_dbz", "type": "esriFieldTypeDouble", "alias": "Rotation Cell Peak dBZ"},
        {"name": "rotation_dual_pol_available", "type": "esriFieldTypeString", "alias": "Rotation Dual-Pol Available"},
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
