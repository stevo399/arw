from dataclasses import dataclass

import numpy as np

from src.buffer import BufferedScan
from src.detection import DetectionResult, ThresholdHierarchyNode, detect_objects_with_grid
from src.tracking.types import SegmentedStormObject


@dataclass
class SegmentationResult:
    objects: list[SegmentedStormObject]
    labeled_grid: np.ndarray
    object_masks: dict[int, np.ndarray]


def compute_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Return (min_row, min_col, max_row, max_col) for a boolean mask."""
    rows, cols = np.where(mask)
    if len(rows) == 0:
        raise ValueError("cannot compute bbox for empty mask")
    return (int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max()))


def adapt_detection_result(result: DetectionResult) -> SegmentationResult:
    """Wrap detection output in tracking-friendly segmented storm objects."""
    segmented_objects: list[SegmentedStormObject] = []
    for detected in result.objects:
        mask = result.object_masks[detected.object_id]
        hierarchy = result.object_hierarchy.get(detected.object_id, [])
        hierarchy_by_id = {node.node_id: node for node in hierarchy}
        strongest_node = max(hierarchy, key=lambda node: (node.threshold, node.pixel_count, node.peak_dbz), default=None)
        threshold_path: tuple[float, ...] = ()
        threshold_parent_id: int | None = None
        threshold_level: float | None = None
        if strongest_node is not None:
            threshold_level = strongest_node.threshold
            threshold_parent_id = strongest_node.parent_node_id
            path: list[float] = []
            current: ThresholdHierarchyNode | None = strongest_node
            while current is not None:
                path.append(current.threshold)
                current = hierarchy_by_id.get(current.parent_node_id) if current.parent_node_id is not None else None
            threshold_path = tuple(sorted(path))
        segmented_objects.append(SegmentedStormObject(
            object_id=detected.object_id,
            detected_object=detected,
            mask=mask,
            bbox=compute_bbox(mask),
            pixel_count=int(np.count_nonzero(mask)),
            threshold_parent_id=threshold_parent_id,
            threshold_level=threshold_level,
            threshold_path=threshold_path,
        ))
    return SegmentationResult(
        objects=segmented_objects,
        labeled_grid=result.labeled_grid,
        object_masks=result.object_masks,
    )


def segment_buffered_scan(scan: BufferedScan) -> SegmentationResult:
    """Wrap a buffered scan's detected objects in segmentation metadata."""
    detection = DetectionResult(
        objects=scan.detected_objects,
        labeled_grid=scan.labeled_grid,
        object_masks=scan.object_masks,
    )
    return adapt_detection_result(detection)


def segment_storm_objects(
    reflectivity: np.ndarray,
    azimuths: np.ndarray,
    ranges_m: np.ndarray,
    radar_lat: float,
    radar_lon: float,
) -> SegmentationResult:
    """Create tracking-friendly segmented storm objects from reflectivity data."""
    detection = detect_objects_with_grid(
        reflectivity=reflectivity,
        azimuths=azimuths,
        ranges_m=ranges_m,
        radar_lat=radar_lat,
        radar_lon=radar_lon,
    )
    return adapt_detection_result(detection)
