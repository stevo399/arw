import math
from dataclasses import dataclass, field
import numpy as np
from scipy.ndimage import label

MIN_OBJECT_AREA_KM2 = 4.0
MIN_DBZ_THRESHOLD = 20.0

INTENSITY_THRESHOLDS = [
    (20, 30, "light rain"),
    (30, 40, "moderate rain"),
    (40, 50, "heavy rain"),
    (50, 60, "intense rain"),
    (60, float("inf"), "severe core"),
]

BEARING_LABELS = [
    "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
    "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW",
]


@dataclass
class IntensityLayerData:
    label: str
    min_dbz: float
    max_dbz: float
    area_km2: float


@dataclass
class DetectedObject:
    object_id: int
    centroid_lat: float
    centroid_lon: float
    distance_km: float
    bearing_deg: float
    peak_dbz: float
    peak_label: str
    area_km2: float
    layers: list[IntensityLayerData] = field(default_factory=list)


def classify_intensity(dbz: float) -> str:
    """Classify a dBZ value into an intensity label."""
    if dbz < 20:
        return "drizzle"
    for min_dbz, max_dbz, label_str in INTENSITY_THRESHOLDS:
        if min_dbz <= dbz < max_dbz:
            return label_str
    return "severe core"


def degrees_to_bearing(deg: float) -> str:
    """Convert compass degrees (0=N, 90=E) to a 16-point cardinal direction."""
    idx = round(deg / 22.5) % 16
    return BEARING_LABELS[idx]


def polar_to_latlon(
    radar_lat: float, radar_lon: float,
    azimuth_deg: float, range_m: float,
) -> tuple[float, float]:
    """Convert a polar coordinate (azimuth, range) relative to a radar to lat/lon."""
    earth_radius_m = 6371000.0
    az_rad = math.radians(azimuth_deg)
    lat1 = math.radians(radar_lat)
    lon1 = math.radians(radar_lon)
    angular_dist = range_m / earth_radius_m

    lat2 = math.asin(
        math.sin(lat1) * math.cos(angular_dist)
        + math.cos(lat1) * math.sin(angular_dist) * math.cos(az_rad)
    )
    lon2 = lon1 + math.atan2(
        math.sin(az_rad) * math.sin(angular_dist) * math.cos(lat1),
        math.cos(angular_dist) - math.sin(lat1) * math.sin(lat2),
    )
    return (math.degrees(lat2), math.degrees(lon2))


def _compute_pixel_area_km2(
    azimuths: np.ndarray, ranges_m: np.ndarray,
    az_idx: int, rng_idx: int,
) -> float:
    """Approximate the area of a single polar pixel in km²."""
    if len(ranges_m) < 2 or len(azimuths) < 2:
        return 0.0
    range_spacing_m = abs(ranges_m[1] - ranges_m[0])
    az_spacing_deg = abs(azimuths[1] - azimuths[0]) if len(azimuths) > 1 else 1.0
    az_spacing_rad = math.radians(az_spacing_deg)
    r = ranges_m[rng_idx]
    area_m2 = r * az_spacing_rad * range_spacing_m
    return area_m2 / 1e6


def compute_object_properties(
    obj_mask: np.ndarray,
    reflectivity: np.ndarray,
    azimuths: np.ndarray,
    ranges_m: np.ndarray,
    radar_lat: float,
    radar_lon: float,
    object_id: int,
) -> "DetectedObject | None":
    """Compute properties for a single detected object. Returns None if too small."""
    az_indices, rng_indices = np.where(obj_mask)
    if len(az_indices) == 0:
        return None

    total_area_km2 = sum(
        _compute_pixel_area_km2(azimuths, ranges_m, int(az), int(rng))
        for az, rng in zip(az_indices, rng_indices)
    )

    if total_area_km2 < MIN_OBJECT_AREA_KM2:
        return None

    obj_dbz = reflectivity[obj_mask]
    valid = ~np.isnan(obj_dbz)
    if not np.any(valid):
        return None

    weights = np.where(valid, obj_dbz, 0)
    weight_sum = weights.sum()
    if weight_sum == 0:
        return None

    centroid_az_idx = np.average(az_indices[valid], weights=weights[valid])
    centroid_rng_idx = np.average(rng_indices[valid], weights=weights[valid])
    centroid_az = float(np.interp(centroid_az_idx, range(len(azimuths)), azimuths))
    centroid_range = float(np.interp(centroid_rng_idx, range(len(ranges_m)), ranges_m))

    centroid_lat, centroid_lon = polar_to_latlon(
        radar_lat, radar_lon, centroid_az, centroid_range,
    )
    distance_km = centroid_range / 1000.0
    bearing_deg = centroid_az % 360

    peak_dbz = float(np.nanmax(obj_dbz))
    peak_label = classify_intensity(peak_dbz)

    layers = []
    for min_dbz, max_dbz, layer_label in INTENSITY_THRESHOLDS:
        layer_mask = obj_mask & (reflectivity >= min_dbz)
        if max_dbz != float("inf"):
            layer_mask = layer_mask & (reflectivity < max_dbz)
        layer_pixels = np.where(layer_mask)
        if len(layer_pixels[0]) == 0:
            continue
        layer_area = sum(
            _compute_pixel_area_km2(azimuths, ranges_m, int(az), int(rng))
            for az, rng in zip(layer_pixels[0], layer_pixels[1])
        )
        if layer_area > 0:
            layers.append(IntensityLayerData(
                label=layer_label,
                min_dbz=min_dbz,
                max_dbz=max_dbz,
                area_km2=round(layer_area, 2),
            ))

    return DetectedObject(
        object_id=object_id,
        centroid_lat=round(centroid_lat, 4),
        centroid_lon=round(centroid_lon, 4),
        distance_km=round(distance_km, 1),
        bearing_deg=round(bearing_deg, 1),
        peak_dbz=round(peak_dbz, 1),
        peak_label=peak_label,
        area_km2=round(total_area_km2, 2),
        layers=layers,
    )


@dataclass
class DetectionResult:
    """Result of object detection including labeled grid for tracking."""
    objects: list[DetectedObject]
    labeled_grid: np.ndarray
    object_masks: dict[int, np.ndarray]


def detect_objects_with_grid(
    reflectivity: np.ndarray,
    azimuths: np.ndarray,
    ranges_m: np.ndarray,
    radar_lat: float,
    radar_lon: float,
) -> DetectionResult:
    """Detect rain objects and return labeled grid + masks for tracking.

    Same as detect_objects but also returns the scipy labeled grid and
    per-object boolean masks needed for overlap-based tracking.
    """
    valid = ~np.isnan(reflectivity) & (reflectivity >= MIN_DBZ_THRESHOLD)
    labeled, num_features = label(valid)

    objects = []
    object_masks = {}
    for i in range(1, num_features + 1):
        obj_mask = labeled == i
        obj = compute_object_properties(
            obj_mask=obj_mask,
            reflectivity=reflectivity,
            azimuths=azimuths,
            ranges_m=ranges_m,
            radar_lat=radar_lat,
            radar_lon=radar_lon,
            object_id=i,
        )
        if obj is not None:
            objects.append(obj)
            object_masks[obj.object_id] = obj_mask

    objects.sort(key=lambda o: o.peak_dbz, reverse=True)
    return DetectionResult(
        objects=objects,
        labeled_grid=labeled,
        object_masks=object_masks,
    )


def detect_objects(
    reflectivity: np.ndarray,
    azimuths: np.ndarray,
    ranges_m: np.ndarray,
    radar_lat: float,
    radar_lon: float,
) -> list[DetectedObject]:
    """Detect rain objects from reflectivity data.
    Returns list of DetectedObject sorted by peak_dbz descending.
    """
    result = detect_objects_with_grid(
        reflectivity=reflectivity,
        azimuths=azimuths,
        ranges_m=ranges_m,
        radar_lat=radar_lat,
        radar_lon=radar_lon,
    )
    return result.objects
