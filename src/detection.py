import math
from dataclasses import dataclass, field
import numpy as np
from scipy.ndimage import label, maximum

from src.geometry import gate_ground_xy_m, label_periodic_azimuth, weighted_geographic_centroid

MIN_OBJECT_AREA_KM2 = 4.0
MIN_SIGNIFICANT_WEAK_OBJECT_AREA_KM2 = 8.0
MIN_SMALL_OBJECT_PEAK_DBZ = 40.0
MIN_DBZ_THRESHOLD = 20.0
SEGMENTATION_HIERARCHY_THRESHOLDS = (20.0, 30.0, 40.0, 50.0, 60.0)
MIN_SEED_PIXELS = 6

INTENSITY_THRESHOLDS = [
    (20, 30, "light precipitation"),
    (30, 40, "moderate precipitation"),
    (40, 50, "heavy precipitation"),
    (50, 60, "intense precipitation"),
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
    max_inbound_ms: float | None = None
    max_outbound_ms: float | None = None
    rotation: "RotationSignature | None" = None
    # Fraction of this object's gates in each QC class (precipitation,
    # ground_clutter, biological, hail, debris, unknown). Since the
    # 2026-08-26 amendment, quality control no longer deletes echo, so an
    # object can be made mostly of clutter or biological gates and still
    # form -- this is how a consumer tells that object apart from one that is
    # mostly precipitation. Empty when gate_classification was not supplied
    # to compute_object_properties (e.g. legacy/synthetic callers).
    class_fractions: dict[str, float] = field(default_factory=dict)
    # Set by live cross-scan association. This describes repeat detection,
    # never confirmation of precipitation at the surface.
    temporal_status: str = "not_checked"


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
    """Convert a polar coordinate (azimuth, range) relative to a radar to lat/lon.

    Deprecated: uses a spherical-earth great-circle approximation that
    disagrees with Py-ART's antenna-coordinate georeferencing by tens to
    hundreds of meters at typical storm ranges (see
    tests/unit/test_geometry.py::test_legacy_polar_to_latlon_displacement_is_documented).
    Use `src.geometry.gate_coordinates` / `src.geometry.gate_latlon` instead.
    Retained only for that regression test and for any legacy callers not yet
    migrated; do not use in new code.
    """
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


def _range_bin_areas_km2(
    azimuths: np.ndarray,
    ranges_m: np.ndarray,
    elevation_deg: float = 0.5,
) -> np.ndarray:
    from src.geometry import gate_areas_km2

    return gate_areas_km2(azimuths, ranges_m, elevation_deg)


def _object_class_fractions(
    obj_mask: np.ndarray, gate_classification: np.ndarray | None
) -> dict[str, float]:
    """Fraction of an object's gates in each QC class.

    Returns {} when gate_classification is not available, so a consumer can
    tell "no classification data" apart from an (impossible) all-zero
    fraction map, rather than silently reporting misleading zeros.
    """
    if gate_classification is None:
        return {}
    from src.qc.classifier import CODE_TO_CLASS

    codes = np.asarray(gate_classification)[obj_mask]
    total = float(codes.size)
    if total == 0:
        return {}
    return {
        name: float(np.count_nonzero(codes == code)) / total
        for code, name in CODE_TO_CLASS.items()
    }


def compute_object_properties(
    obj_mask: np.ndarray,
    reflectivity: np.ndarray,
    azimuths: np.ndarray,
    ranges_m: np.ndarray,
    radar_lat: float,
    radar_lon: float,
    object_id: int,
    elevation_deg: float = 0.5,
    elevations: np.ndarray | None = None,
    gate_classification: np.ndarray | None = None,
    range_bin_areas_km2: np.ndarray | None = None,
) -> "DetectedObject | None":
    """Compute properties for a single detected object. Returns None if too small.

    `elevations`, if given, is the per-ray elevation array for this sweep
    (`SweepData.elevations`) and is preferred over the scalar
    `elevation_deg` for georeferencing the centroid -- see
    `geometry.gate_coordinates`'s docstring. When absent, `elevation_deg`
    (the nominal sweep angle) is used for every centroid instead.

    `gate_classification`, if given, is the sweep's QC classification array
    (co-registered with `reflectivity`) and is used to populate the returned
    object's `class_fractions`. Absent by default so existing/synthetic
    callers that never ran quality control keep working unchanged.
    """
    az_indices, rng_indices = np.where(obj_mask)
    if len(az_indices) == 0:
        return None

    # All components in one sweep share the same range geometry.  The live
    # detector supplies this precomputed vector rather than deriving it once
    # per candidate echo (which is especially wasteful on clear-air scans
    # containing hundreds of tiny candidates).
    if range_bin_areas_km2 is None:
        range_bin_areas_km2 = _range_bin_areas_km2(
            azimuths, ranges_m, elevation_deg
        )
    total_area_km2 = float(np.sum(range_bin_areas_km2[rng_indices]))

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

    # Owner decision (2026-09-13): a storm's centre is its reflectivity-
    # weighted centre on the ground, each gate counted by the area it covers.
    centroid = weighted_geographic_centroid(
        az_indices,
        rng_indices,
        weights * range_bin_areas_km2[rng_indices],
        azimuths,
        ranges_m,
        elevations if elevations is not None else elevation_deg,
        radar_lat,
        radar_lon,
    )
    if centroid is None:
        return None
    centroid_lat, centroid_lon, distance_km, bearing_deg = centroid
    peak_dbz = float(np.nanmax(obj_dbz))
    peak_label = classify_intensity(peak_dbz)

    # Filter out very small weak echoes so simple scenes do not fragment into
    # many low-significance objects while compact intense cores still survive.
    if total_area_km2 < MIN_SIGNIFICANT_WEAK_OBJECT_AREA_KM2 and peak_dbz < MIN_SMALL_OBJECT_PEAK_DBZ:
        return None

    layers = []
    obj_range_areas = range_bin_areas_km2[rng_indices]
    for min_dbz, max_dbz, layer_label in INTENSITY_THRESHOLDS:
        layer_valid = obj_dbz >= min_dbz
        if max_dbz != float("inf"):
            layer_valid = layer_valid & (obj_dbz < max_dbz)
        if not np.any(layer_valid):
            continue
        layer_area = float(np.sum(obj_range_areas[layer_valid]))
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
        class_fractions=_object_class_fractions(obj_mask, gate_classification),
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


@dataclass
class _HierarchyLevels:
    """Component labels for each threshold, within the parent blob's bounding box.

    Nodes used to carry one full-grid mask each.  A single real storm complex
    (KEMX, 36,222 gates) produced 1,329 nodes, a 1.77 GB peak, and 2.2-3.5 GB
    held until detection returned.  A node's mask is now derived from its
    level's labels, and only for the few seed leaves a split uses.
    """

    grid_shape: tuple[int, int]
    window: tuple[slice, slice]
    labels: dict[float, np.ndarray] = field(default_factory=dict)
    components: dict[int, tuple[float, int]] = field(default_factory=dict)  # node_id -> (threshold, label)

    def node_mask(self, node_id: int) -> np.ndarray:
        threshold, component_id = self.components[node_id]
        mask = np.zeros(self.grid_shape, dtype=bool)
        mask[self.window] = self.labels[threshold] == component_id
        return mask


def _mask_window(mask: np.ndarray) -> tuple[slice, slice]:
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    return slice(int(rows[0]), int(rows[-1]) + 1), slice(int(cols[0]), int(cols[-1]) + 1)


def _dominant_previous_components(labels: np.ndarray, previous_labels: np.ndarray) -> dict[int, int]:
    """For each component, the previous-level component it overlaps most.

    Ties go to the smaller previous label, matching the former per-component
    `np.unique` + `argmax` selection.
    """
    both = (labels > 0) & (previous_labels > 0)
    if not both.any():
        return {}
    current = labels[both].astype(np.int64)
    previous = previous_labels[both].astype(np.int64)
    base = int(previous.max()) + 1
    pairs, counts = np.unique(current * base + previous, return_counts=True)
    pair_current = pairs // base
    pair_previous = pairs % base
    dominant: dict[int, int] = {}
    for index in np.lexsort((pair_previous, -counts, pair_current)):
        dominant.setdefault(int(pair_current[index]), int(pair_previous[index]))
    return dominant


def _build_hierarchy_levels(
    parent_mask: np.ndarray,
    reflectivity: np.ndarray,
) -> tuple[list[ThresholdHierarchyNode], _HierarchyLevels]:
    """Build nested threshold components inside a low-threshold parent blob."""
    window = _mask_window(parent_mask)
    # A window spanning every ray has ray 0 and the last ray, which are
    # adjacent, as its edges; a core straddling them is one core.
    spans_all_rays = window[0].start == 0 and window[0].stop == parent_mask.shape[0]
    label_components = label_periodic_azimuth if spans_all_rays else label
    parent = parent_mask[window]
    field_values = reflectivity[window]
    finite = ~np.isnan(field_values)
    levels = _HierarchyLevels(grid_shape=parent_mask.shape, window=window)
    nodes: list[ThresholdHierarchyNode] = []
    previous_labels: np.ndarray | None = None
    previous_component_to_node_id: dict[int, int] = {}
    next_node_id = 1
    for threshold in SEGMENTATION_HIERARCHY_THRESHOLDS:
        labels, component_count = label_components(parent & finite & (field_values >= threshold))
        levels.labels[threshold] = labels
        current_component_to_node_id: dict[int, int] = {}
        if component_count:
            component_ids = np.arange(1, component_count + 1)
            pixel_counts = np.bincount(labels.ravel(), minlength=component_count + 1)
            peaks = np.asarray(maximum(field_values, labels=labels, index=component_ids))
            dominant = (
                _dominant_previous_components(labels, previous_labels)
                if previous_component_to_node_id
                else {}
            )
            for component_id in component_ids.tolist():
                pixel_count = int(pixel_counts[component_id])
                if pixel_count <= 0:
                    continue
                previous_component = dominant.get(component_id)
                nodes.append(ThresholdHierarchyNode(
                    node_id=next_node_id,
                    threshold=threshold,
                    parent_node_id=(
                        previous_component_to_node_id.get(previous_component)
                        if previous_component is not None
                        else None
                    ),
                    pixel_count=pixel_count,
                    peak_dbz=float(peaks[component_id - 1]),
                ))
                levels.components[next_node_id] = (threshold, component_id)
                current_component_to_node_id[component_id] = next_node_id
                next_node_id += 1
        previous_labels = labels
        previous_component_to_node_id = current_component_to_node_id
    return nodes, levels


def _build_threshold_hierarchy(
    parent_mask: np.ndarray,
    reflectivity: np.ndarray,
) -> list[ThresholdHierarchyNode]:
    """Build nested threshold components inside a low-threshold parent blob."""
    return _build_hierarchy_levels(parent_mask, reflectivity)[0]


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


@dataclass(frozen=True)
class _SweepGeometry:
    """What a split needs to measure distances between gates on the ground."""

    azimuths: np.ndarray
    ranges_m: np.ndarray
    elevation_deg: float | np.ndarray
    range_bin_areas_km2: np.ndarray

    def ground_xy_m(self, rays: np.ndarray, gates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return gate_ground_xy_m(rays, gates, self.azimuths, self.ranges_m, self.elevation_deg)


def _select_hierarchy_split_masks(
    parent_mask: np.ndarray,
    reflectivity: np.ndarray,
    hierarchy_nodes: list[ThresholdHierarchyNode],
    levels: _HierarchyLevels,
    geometry: _SweepGeometry,
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

    seed_masks = [levels.node_mask(leaf.node_id) for leaf in sorted(selected_leaves, key=lambda node: (node.threshold, node.pixel_count, node.peak_dbz), reverse=True)]
    centroids = _seed_centroids(seed_masks, reflectivity, geometry)
    child_masks = [np.zeros_like(parent_mask, dtype=bool) for _ in seed_masks]
    claimed_seed_pixels = np.zeros_like(parent_mask, dtype=bool)
    for index, seed_mask in enumerate(seed_masks):
        child_masks[index][seed_mask] = True
        claimed_seed_pixels |= seed_mask

    remaining_mask = parent_mask & ~claimed_seed_pixels
    rows, cols = np.where(remaining_mask)
    if len(rows) > 0:
        # Nearest core on the ground.  Ray and gate numbers are not distances:
        # a ray spans kilometres at long range where a gate spans 250 m, and
        # ray 0 is adjacent to the last ray.
        x, y = geometry.ground_xy_m(rows, cols)
        centroid_x = np.array([cx for cx, _ in centroids], dtype=float)
        centroid_y = np.array([cy for _, cy in centroids], dtype=float)
        distances = (x[:, None] - centroid_x[None, :]) ** 2 + (y[:, None] - centroid_y[None, :]) ** 2
        assignments = np.argmin(distances, axis=1)
        for index in range(len(child_masks)):
            assigned = assignments == index
            if np.any(assigned):
                child_masks[index][rows[assigned], cols[assigned]] = True

    non_empty_children = [mask for mask in child_masks if np.any(mask)]
    return non_empty_children if len(non_empty_children) >= 2 else [parent_mask]


def _seed_centroids(
    seed_masks: list[np.ndarray],
    reflectivity: np.ndarray,
    geometry: _SweepGeometry,
) -> list[tuple[float, float]]:
    """Ground centre (east, north metres from the radar) of each seed.

    Weighted like a storm's centre: reflectivity times gate area.
    """
    centroids: list[tuple[float, float]] = []
    for mask in seed_masks:
        rows, cols = np.where(mask)
        x, y = geometry.ground_xy_m(rows, cols)
        weights = np.nan_to_num(reflectivity[mask], nan=0.0, posinf=0.0, neginf=0.0) * geometry.range_bin_areas_km2[cols]
        if float(np.sum(weights)) <= 0.0:
            centroids.append((float(x.mean()), float(y.mean())))
            continue
        centroids.append((float(np.average(x, weights=weights)), float(np.average(y, weights=weights))))
    return centroids


def _split_parent_mask(
    parent_mask: np.ndarray,
    reflectivity: np.ndarray,
    geometry: _SweepGeometry,
) -> tuple[list[np.ndarray], list[ThresholdHierarchyNode]]:
    """Partition a low-threshold blob around persistent multilevel branches."""
    hierarchy_nodes, levels = _build_hierarchy_levels(parent_mask, reflectivity)
    return _select_hierarchy_split_masks(parent_mask, reflectivity, hierarchy_nodes, levels, geometry), hierarchy_nodes


def detect_objects_with_grid(
    reflectivity: np.ndarray,
    azimuths: np.ndarray,
    ranges_m: np.ndarray,
    radar_lat: float,
    radar_lon: float,
    elevation_deg: float = 0.5,
    elevations: np.ndarray | None = None,
    gate_classification: np.ndarray | None = None,
) -> DetectionResult:
    """Detect precipitation objects and return labeled grid + masks for tracking.

    Same as detect_objects but also returns the scipy labeled grid and
    per-object boolean masks needed for overlap-based tracking.

    `gate_classification`, if given, is forwarded to `compute_object_properties`
    so each returned object carries its QC class composition (see
    `DetectedObject.class_fractions`). Detection itself is unaffected by
    classification -- since the 2026-08-26 amendment `reflectivity` here is
    always the full, unfiltered field.
    """
    valid = ~np.isnan(reflectivity) & (reflectivity >= MIN_DBZ_THRESHOLD)
    # No structure argument => 4-connectivity, preserved from the original
    # scipy.ndimage.label call. label_periodic_azimuth additionally merges
    # components that touch across the row 0 / row N-1 azimuth seam.
    labeled, num_features = label_periodic_azimuth(valid)

    objects = []
    object_masks = {}
    object_hierarchy: dict[int, list[ThresholdHierarchyNode]] = {}
    next_object_id = 1
    range_bin_areas_km2 = _range_bin_areas_km2(
        azimuths, ranges_m, elevation_deg
    )
    geometry = _SweepGeometry(
        azimuths=azimuths,
        ranges_m=ranges_m,
        elevation_deg=elevations if elevations is not None else elevation_deg,
        range_bin_areas_km2=range_bin_areas_km2,
    )
    for i in range(1, num_features + 1):
        parent_mask = labeled == i
        # A split requires at least two >= 50 dBZ seed branches.  A component
        # whose peak never reaches that threshold cannot split under the
        # documented hierarchy rules, so running five whole-grid connected
        # component passes for it cannot change the result.  This short-circuit
        # is crucial for quiet/clear-air scans, which can contain hundreds of
        # small weak candidates.
        parent_peak = float(np.nanmax(reflectivity[parent_mask]))
        if parent_peak >= 50.0:
            split_masks, hierarchy_nodes = _split_parent_mask(
                parent_mask, reflectivity, geometry
            )
        else:
            split_masks, hierarchy_nodes = [parent_mask], []
        for obj_mask in split_masks:
            obj = compute_object_properties(
                obj_mask=obj_mask,
                reflectivity=reflectivity,
                azimuths=azimuths,
                ranges_m=ranges_m,
                radar_lat=radar_lat,
                radar_lon=radar_lon,
                object_id=next_object_id,
                elevation_deg=elevation_deg,
                elevations=elevations,
                gate_classification=gate_classification,
                range_bin_areas_km2=range_bin_areas_km2,
            )
            if obj is None:
                continue
            objects.append(obj)
            object_masks[obj.object_id] = obj_mask
            # Tracking metadata still needs the threshold path of an accepted
            # unsplit object.  Delay that bookkeeping until after the area/
            # significance filter: it is useful for real objects, but doing
            # it for every discarded weak speckle was the full-grid hot path.
            object_hierarchy[obj.object_id] = (
                hierarchy_nodes
                if hierarchy_nodes
                else _build_threshold_hierarchy(obj_mask, reflectivity)
            )
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
    elevation_deg: float = 0.5,
    elevations: np.ndarray | None = None,
    gate_classification: np.ndarray | None = None,
) -> list[DetectedObject]:
    """Detect precipitation objects from reflectivity data.
    Returns list of DetectedObject sorted by peak_dbz descending.
    """
    result = detect_objects_with_grid(
        reflectivity=reflectivity,
        azimuths=azimuths,
        ranges_m=ranges_m,
        radar_lat=radar_lat,
        radar_lon=radar_lon,
        elevation_deg=elevation_deg,
        elevations=elevations,
        gate_classification=gate_classification,
    )
    return result.objects
