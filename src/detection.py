import math
from dataclasses import dataclass, field
import numpy as np
from scipy.ndimage import label

MIN_OBJECT_AREA_KM2 = 4.0
MIN_SIGNIFICANT_WEAK_OBJECT_AREA_KM2 = 8.0
MIN_SMALL_OBJECT_PEAK_DBZ = 40.0
MIN_DBZ_THRESHOLD = 20.0
SEGMENTATION_HIERARCHY_THRESHOLDS = (20.0, 30.0, 40.0, 50.0, 60.0)
MIN_SEED_PIXELS = 6

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

    # Filter out very small weak echoes so simple scenes do not fragment into
    # many low-significance objects while compact intense cores still survive.
    if total_area_km2 < MIN_SIGNIFICANT_WEAK_OBJECT_AREA_KM2 and peak_dbz < MIN_SMALL_OBJECT_PEAK_DBZ:
        return None

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
    object_hierarchy: dict[int, list["ThresholdHierarchyNode"]] = field(default_factory=dict)


@dataclass
class ThresholdHierarchyNode:
    node_id: int
    threshold: float
    parent_node_id: int | None
    pixel_count: int
    peak_dbz: float
    mask: np.ndarray = field(repr=False)


def _assign_parent_node_id(
    candidate_mask: np.ndarray,
    previous_level_nodes: list[ThresholdHierarchyNode],
) -> int | None:
    best_parent_id = None
    best_overlap = 0
    for node in previous_level_nodes:
        overlap = int(np.count_nonzero(candidate_mask & node.mask))
        if overlap > best_overlap:
            best_overlap = overlap
            best_parent_id = node.node_id
    return best_parent_id


def _build_threshold_hierarchy(
    parent_mask: np.ndarray,
    reflectivity: np.ndarray,
) -> list[ThresholdHierarchyNode]:
    """Build nested threshold components inside a low-threshold parent blob."""
    nodes: list[ThresholdHierarchyNode] = []
    previous_level_nodes: list[ThresholdHierarchyNode] = []
    next_node_id = 1
    for threshold in SEGMENTATION_HIERARCHY_THRESHOLDS:
        threshold_grid = parent_mask & ~np.isnan(reflectivity) & (reflectivity >= threshold)
        labeled_grid, component_count = label(threshold_grid)
        current_level_nodes: list[ThresholdHierarchyNode] = []
        for component_id in range(1, component_count + 1):
            component_mask = labeled_grid == component_id
            pixel_count = int(np.count_nonzero(component_mask))
            if pixel_count <= 0:
                continue
            current_level_nodes.append(ThresholdHierarchyNode(
                node_id=next_node_id,
                threshold=threshold,
                parent_node_id=_assign_parent_node_id(component_mask, previous_level_nodes),
                pixel_count=pixel_count,
                peak_dbz=float(np.nanmax(reflectivity[component_mask])),
                mask=component_mask,
            ))
            next_node_id += 1
        nodes.extend(current_level_nodes)
        previous_level_nodes = current_level_nodes
    return nodes


def _hierarchy_children(nodes: list[ThresholdHierarchyNode], parent_id: int) -> list[ThresholdHierarchyNode]:
    return [node for node in nodes if node.parent_node_id == parent_id]


def _hierarchy_leaves(nodes: list[ThresholdHierarchyNode]) -> list[ThresholdHierarchyNode]:
    child_parent_ids = {node.parent_node_id for node in nodes if node.parent_node_id is not None}
    return [node for node in nodes if node.node_id not in child_parent_ids]


def _branch_threshold_path(nodes_by_id: dict[int, ThresholdHierarchyNode], leaf: ThresholdHierarchyNode) -> tuple[float, ...]:
    path: list[float] = []
    current = leaf
    while current is not None:
        path.append(current.threshold)
        current = nodes_by_id.get(current.parent_node_id) if current.parent_node_id is not None else None
    return tuple(sorted(path))


def _select_hierarchy_split_masks(
    parent_mask: np.ndarray,
    reflectivity: np.ndarray,
    hierarchy_nodes: list[ThresholdHierarchyNode],
) -> list[np.ndarray]:
    """Choose split branches from a multilevel threshold hierarchy."""
    if not hierarchy_nodes:
        return [parent_mask]

    nodes_by_id = {node.node_id: node for node in hierarchy_nodes}
    candidate_leaves = []
    for leaf in _hierarchy_leaves(hierarchy_nodes):
        if leaf.pixel_count < MIN_SEED_PIXELS:
            continue
        threshold_path = _branch_threshold_path(nodes_by_id, leaf)
        if leaf.threshold < 50.0:
            continue
        if 40.0 not in threshold_path:
            continue
        candidate_leaves.append(leaf)

    if len(candidate_leaves) < 2:
        return [parent_mask]

    # Collapse candidates that belong to the same immediate 40+ ancestor branch.
    branch_groups: dict[int, ThresholdHierarchyNode] = {}
    for leaf in candidate_leaves:
        current = leaf
        branch_anchor = leaf.node_id
        while current.parent_node_id is not None:
            parent = nodes_by_id[current.parent_node_id]
            if parent.threshold >= 40.0:
                branch_anchor = parent.node_id
            current = parent
        existing = branch_groups.get(branch_anchor)
        if existing is None or (leaf.threshold, leaf.pixel_count, leaf.peak_dbz) > (
            existing.threshold,
            existing.pixel_count,
            existing.peak_dbz,
        ):
            branch_groups[branch_anchor] = leaf

    selected_leaves = list(branch_groups.values())
    if len(selected_leaves) < 2:
        return [parent_mask]

    seed_masks = [leaf.mask for leaf in sorted(selected_leaves, key=lambda node: (node.threshold, node.pixel_count, node.peak_dbz), reverse=True)]
    centroids = _seed_centroids(seed_masks, reflectivity)
    child_masks = [np.zeros_like(parent_mask, dtype=bool) for _ in seed_masks]
    claimed_seed_pixels = np.zeros_like(parent_mask, dtype=bool)
    for index, seed_mask in enumerate(seed_masks):
        child_masks[index][seed_mask] = True
        claimed_seed_pixels |= seed_mask

    remaining_mask = parent_mask & ~claimed_seed_pixels
    rows, cols = np.where(remaining_mask)
    for row, col in zip(rows, cols):
        distances = [
            (row - seed_row) ** 2 + (col - seed_col) ** 2
            for seed_row, seed_col in centroids
        ]
        child_masks[int(np.argmin(distances))][row, col] = True

    non_empty_children = [mask for mask in child_masks if np.any(mask)]
    return non_empty_children if len(non_empty_children) >= 2 else [parent_mask]


def _seed_centroids(
    seed_masks: list[np.ndarray],
    reflectivity: np.ndarray,
) -> list[tuple[float, float]]:
    """Compute weighted centroids for seed masks."""
    centroids: list[tuple[float, float]] = []
    for mask in seed_masks:
        rows, cols = np.where(mask)
        weights = np.nan_to_num(reflectivity[mask], nan=0.0, posinf=0.0, neginf=0.0)
        if float(np.sum(weights)) <= 0.0:
            centroids.append((float(rows.mean()), float(cols.mean())))
            continue
        centroids.append((
            float(np.average(rows, weights=weights)),
            float(np.average(cols, weights=weights)),
        ))
    return centroids


def _split_parent_mask(
    parent_mask: np.ndarray,
    reflectivity: np.ndarray,
) -> tuple[list[np.ndarray], list[ThresholdHierarchyNode]]:
    """Partition a low-threshold blob around persistent multilevel branches."""
    hierarchy_nodes = _build_threshold_hierarchy(parent_mask, reflectivity)
    return _select_hierarchy_split_masks(parent_mask, reflectivity, hierarchy_nodes), hierarchy_nodes


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
    object_hierarchy: dict[int, list[ThresholdHierarchyNode]] = {}
    next_object_id = 1
    for i in range(1, num_features + 1):
        parent_mask = labeled == i
        split_masks, hierarchy_nodes = _split_parent_mask(parent_mask, reflectivity)
        for obj_mask in split_masks:
            obj = compute_object_properties(
                obj_mask=obj_mask,
                reflectivity=reflectivity,
                azimuths=azimuths,
                ranges_m=ranges_m,
                radar_lat=radar_lat,
                radar_lon=radar_lon,
                object_id=next_object_id,
            )
            if obj is None:
                continue
            objects.append(obj)
            object_masks[obj.object_id] = obj_mask
            object_hierarchy[obj.object_id] = hierarchy_nodes
            next_object_id += 1

    objects.sort(key=lambda o: (o.peak_dbz, o.area_km2), reverse=True)
    final_labeled = np.zeros_like(labeled)
    for obj in objects:
        final_labeled[object_masks[obj.object_id]] = obj.object_id
    return DetectionResult(
        objects=objects,
        labeled_grid=final_labeled,
        object_masks=object_masks,
        object_hierarchy=object_hierarchy,
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
