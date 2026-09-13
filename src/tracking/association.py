from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
import math

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.buffer import BufferedScan
from src.sites import haversine_distance_km
from src.tracking.motion_field import (
    GeographicMotionFieldEstimate,
    blend_geographic_motion_fields,
    estimate_geographic_motion_field,
    estimate_motion_field,
    estimate_local_motion_field,
    estimate_local_scan_geographic_motion_field,
    estimate_scan_geographic_motion_field,
    predict_latlon_position,
)
from src.tracking.segmentation import segment_buffered_scan
from src.tracking.types import AssociationScore, Track

MIN_OVERLAP_PCT = 0.30
MIN_ADVECTED_OVERLAP_PCT = 0.15
MAX_STORM_SPEED_KMH = 120.0
UNMATCHED_COST = 10.0
# Must equal src.tracker.MAX_TEMPORAL_CONTINUITY_MINUTES (asserted in tests):
# a storm unseen for longer than that is not evidence of the same storm.
MAX_REACQUISITION_MINUTES = 20.0


@dataclass
class AssociationResult:
    primary_matches: dict[int, int] = field(default_factory=dict)  # new_obj_id -> track_id
    merge_candidates: dict[int, list[int]] = field(default_factory=dict)  # new_obj_id -> track_ids
    split_candidates: dict[int, list[int]] = field(default_factory=dict)  # track_id -> new_obj_ids
    unmatched_new_ids: set[int] = field(default_factory=set)
    unmatched_track_ids: set[int] = field(default_factory=set)
    candidate_scores: list[AssociationScore] = field(default_factory=list)
    geo_motion: object | None = None
    track_geo_motion: dict[int, object] = field(default_factory=dict)
    dt_hours: float = 0.0
    # Stage 2: missing tracks matched to objects stage 1 left unclaimed.  Kept
    # apart from candidate_scores so stage-1 ambiguity margins and identity
    # confidence are unaffected.
    reacquired_matches: dict[int, int] = field(default_factory=dict)  # new_obj_id -> track_id
    reacquisition_scores: list[AssociationScore] = field(default_factory=list)
    # Missing tracks that can never be reacquired (too old, or last-seen scan gone).
    unreacquirable_track_ids: set[int] = field(default_factory=set)


def compute_overlap(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Compute the fraction of mask_a covered by mask_b."""
    if not np.any(mask_a):
        return 0.0
    intersection = np.sum(mask_a & mask_b)
    return float(intersection) / float(np.sum(mask_a))


def _shift_mask(mask: np.ndarray, shift_rows: float, shift_cols: float) -> np.ndarray:
    """Translate a boolean mask by rounded pixel shifts."""
    row_shift = int(round(shift_rows))
    col_shift = int(round(shift_cols))
    shifted = np.zeros_like(mask, dtype=bool)

    src_row_start = max(0, -row_shift)
    src_row_end = mask.shape[0] - max(0, row_shift)
    src_col_start = max(0, -col_shift)
    src_col_end = mask.shape[1] - max(0, col_shift)
    dst_row_start = max(0, row_shift)
    dst_row_end = dst_row_start + max(0, src_row_end - src_row_start)
    dst_col_start = max(0, col_shift)
    dst_col_end = dst_col_start + max(0, src_col_end - src_col_start)

    if src_row_start >= src_row_end or src_col_start >= src_col_end:
        return shifted

    shifted[dst_row_start:dst_row_end, dst_col_start:dst_col_end] = mask[src_row_start:src_row_end, src_col_start:src_col_end]
    return shifted


def compute_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Compute intersection-over-union between two masks."""
    intersection = int(np.count_nonzero(mask_a & mask_b))
    union = int(np.count_nonzero(mask_a | mask_b))
    if union <= 0:
        return 0.0
    return float(intersection) / float(union)


@dataclass(frozen=True)
class MaskExtent:
    """A mask with its bounding window and gate count.

    Overlap between two masks can only occur inside both windows, so counting
    within the windows' intersection gives exactly the full-grid counts.
    Scoring every track against every object on full grids took 207 of 209
    seconds for a 218-by-225-storm KJAX update (2026-09-13).
    """

    mask: np.ndarray
    rows: slice
    cols: slice
    count: int


def mask_extent(mask: np.ndarray) -> MaskExtent:
    count = int(np.count_nonzero(mask))
    if count == 0:
        return MaskExtent(mask, slice(0, 0), slice(0, 0), 0)
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    return MaskExtent(mask, slice(int(rows[0]), int(rows[-1]) + 1), slice(int(cols[0]), int(cols[-1]) + 1), count)


def _intersection_count(a: MaskExtent, b: MaskExtent) -> int:
    row_start, row_stop = max(a.rows.start, b.rows.start), min(a.rows.stop, b.rows.stop)
    col_start, col_stop = max(a.cols.start, b.cols.start), min(a.cols.stop, b.cols.stop)
    if row_start >= row_stop or col_start >= col_stop:
        return 0
    return int(np.count_nonzero(
        a.mask[row_start:row_stop, col_start:col_stop] & b.mask[row_start:row_stop, col_start:col_stop]
    ))


def _overlap_from_extents(a: MaskExtent, b: MaskExtent) -> float:
    """Exactly `compute_overlap(a.mask, b.mask)`."""
    if a.count == 0:
        return 0.0
    return float(_intersection_count(a, b)) / float(a.count)


def _iou_from_extents(a: MaskExtent, b: MaskExtent) -> float:
    """Exactly `compute_iou(a.mask, b.mask)`."""
    intersection = _intersection_count(a, b)
    union = a.count + b.count - intersection
    if union <= 0:
        return 0.0
    return float(intersection) / float(union)


def compute_advected_iou(prev_mask: np.ndarray, new_mask: np.ndarray, shift_rows: float, shift_cols: float) -> float:
    """Compute IoU after advecting the previous mask with the scene motion."""
    return compute_iou(_shift_mask(prev_mask, shift_rows, shift_cols), new_mask)


def _candidate_score(
    track: Track,
    new_object,
    prev_extent: MaskExtent,
    shifted_extent: MaskExtent,
    new_extent: MaskExtent,
    max_distance_km: float,
    predicted_lat: float,
    predicted_lon: float,
) -> AssociationScore | None:
    """Score one track against one object.

    `shifted_extent` is the track's previous mask already shifted by its
    motion (computed once per track); overlaps are counted exactly within
    the masks' windows.
    """
    current_object = track.current_object
    if current_object is None:
        return None

    overlap = _overlap_from_extents(prev_extent, new_extent)
    advected_overlap = _iou_from_extents(shifted_extent, new_extent)
    centroid_distance = haversine_distance_km(
        current_object.centroid_lat,
        current_object.centroid_lon,
        new_object.centroid_lat,
        new_object.centroid_lon,
    )
    predicted_distance = haversine_distance_km(
        predicted_lat,
        predicted_lon,
        new_object.centroid_lat,
        new_object.centroid_lon,
    )
    plausible_distance = max(max_distance_km, 5.0)
    if overlap < MIN_OVERLAP_PCT and advected_overlap < MIN_ADVECTED_OVERLAP_PCT and centroid_distance > plausible_distance:
        return None

    overlap_cost = 1.0 - max(overlap, advected_overlap)
    distance_score = min(centroid_distance / plausible_distance, 5.0)
    predicted_score = min(predicted_distance / plausible_distance, 5.0)
    area_change = abs(new_object.area_km2 - current_object.area_km2) / max(current_object.area_km2, 1.0)
    intensity_change = abs(new_object.peak_dbz - current_object.peak_dbz) / 60.0
    total_cost = (overlap_cost * 0.65) + (predicted_score * 0.18) + (distance_score * 0.07) + (area_change * 0.07) + (intensity_change * 0.03)

    return AssociationScore(
        track_id=track.track_id,
        object_id=new_object.object_id,
        overlap_score=round(overlap, 4),
        advected_overlap_score=round(advected_overlap, 4),
        distance_score=round(distance_score, 4),
        predicted_position_score=round(predicted_score, 4),
        area_change_score=round(area_change, 4),
        intensity_change_score=round(intensity_change, 4),
        total_cost=round(total_cost, 4),
    )


def associate_tracks(
    previous_scan: BufferedScan,
    current_scan: BufferedScan,
    tracks: list[Track],
    obj_to_track: dict[int, int],
    reacquisition_loader: Callable[[datetime], BufferedScan | None] | None = None,
) -> AssociationResult:
    """Associate active tracks to new objects using a global cost matrix.

    With a reacquisition loader, missing tracks then compete for the objects
    left unclaimed (stage 2).
    """
    result = AssociationResult()
    active_tracks = [track for track in tracks if track.status == "active" and track.current_object is not None]
    missing_tracks = (
        [track for track in tracks if track.status == "missing" and track.current_object is not None]
        if reacquisition_loader is not None
        else []
    )
    if not active_tracks and not missing_tracks:
        result.unmatched_new_ids = {obj.object_id for obj in current_scan.detected_objects}
        return result

    previous_segmentation = segment_buffered_scan(previous_scan)
    current_segmentation = segment_buffered_scan(current_scan)
    geo_motion = estimate_scan_geographic_motion_field(previous_scan, current_scan)
    pixel_motion = estimate_motion_field(
        previous_scan.reflectivity_data.reflectivity,
        current_scan.reflectivity_data.reflectivity,
    )
    if geo_motion.quality <= 0.0:
        geo_motion = estimate_geographic_motion_field(previous_segmentation.objects, current_segmentation.objects)
    result.geo_motion = geo_motion

    new_objects = {obj.object_id: obj for obj in current_scan.detected_objects}
    prev_masks = previous_scan.object_masks
    new_masks = current_scan.object_masks

    dt_hours = (current_scan.timestamp - previous_scan.timestamp).total_seconds() / 3600.0
    result.dt_hours = dt_hours
    max_distance_km = MAX_STORM_SPEED_KMH * dt_hours if dt_hours > 0 else 50.0

    track_ids = [track.track_id for track in active_tracks]
    new_ids = [obj.object_id for obj in current_scan.detected_objects]
    track_index = {track_id: idx for idx, track_id in enumerate(track_ids)}
    new_index = {obj_id: idx for idx, obj_id in enumerate(new_ids)}
    cost_matrix = np.full((len(track_ids), len(new_ids)), UNMATCHED_COST, dtype=float)
    scores_by_track: dict[int, list[AssociationScore]] = {track_id: [] for track_id in track_ids}
    scores_by_object: dict[int, list[AssociationScore]] = {obj_id: [] for obj_id in new_ids}

    new_extents: dict[int, MaskExtent] = {}

    def new_extent(object_id: int) -> MaskExtent:
        if object_id not in new_extents:
            new_extents[object_id] = mask_extent(new_masks[object_id])
        return new_extents[object_id]

    for track in active_tracks:
        current_object = track.current_object
        if current_object is None:
            continue
        prev_object_id = None
        for object_id, track_id in obj_to_track.items():
            if track_id == track.track_id:
                prev_object_id = object_id
                break
        if prev_object_id is None or prev_object_id not in prev_masks:
            continue
        prev_mask = prev_masks[prev_object_id]
        prev_segment = next((seg for seg in previous_segmentation.objects if seg.object_id == prev_object_id), None)
        local_geo_motion = None
        local_pixel_motion = None
        if prev_segment is not None:
            local_geo_motion = estimate_local_scan_geographic_motion_field(
                previous_scan,
                current_scan,
                prev_segment.bbox,
            )
            local_pixel_motion = estimate_local_motion_field(
                previous_scan.reflectivity_data.reflectivity,
                current_scan.reflectivity_data.reflectivity,
                prev_segment.bbox,
                downsample=1,
            )
        blended_geo_motion = blend_geographic_motion_fields(geo_motion, local_geo_motion)
        result.track_geo_motion[track.track_id] = blended_geo_motion
        predicted_lat, predicted_lon = predict_latlon_position(
            current_object.centroid_lat,
            current_object.centroid_lon,
            blended_geo_motion,
        )
        blended_shift_rows = -pixel_motion.shift_rows
        blended_shift_cols = -pixel_motion.shift_cols
        use_local_pixel_guidance = (
            local_pixel_motion is not None
            and local_pixel_motion.quality > 0.0
            and blended_geo_motion.source.startswith("blended:")
        ) or (
            local_pixel_motion is not None
            and local_pixel_motion.quality > 0.0
            and blended_geo_motion.source == "local_phase_correlation"
        )
        if use_local_pixel_guidance:
            local_weight = max(local_pixel_motion.quality, 0.0)
            global_weight = max(pixel_motion.quality * 0.6, 0.0)
            total_weight = local_weight + global_weight
            if total_weight > 0.0:
                blended_shift_rows = -(((local_pixel_motion.shift_rows * local_weight) + (pixel_motion.shift_rows * global_weight)) / total_weight)
                blended_shift_cols = -(((local_pixel_motion.shift_cols * local_weight) + (pixel_motion.shift_cols * global_weight)) / total_weight)
        prev_extent = mask_extent(prev_mask)
        shifted_extent = mask_extent(_shift_mask(prev_mask, blended_shift_rows, blended_shift_cols))
        for new_id, new_object in new_objects.items():
            score = _candidate_score(
                track=track,
                new_object=new_object,
                prev_extent=prev_extent,
                shifted_extent=shifted_extent,
                new_extent=new_extent(new_id),
                max_distance_km=max_distance_km,
                predicted_lat=predicted_lat,
                predicted_lon=predicted_lon,
            )
            if score is None:
                continue
            result.candidate_scores.append(score)
            scores_by_track[track.track_id].append(score)
            scores_by_object[new_id].append(score)
            cost_matrix[track_index[track.track_id], new_index[new_id]] = score.total_cost

    if cost_matrix.size:
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        for row, col in zip(row_ind, col_ind):
            total_cost = float(cost_matrix[row, col])
            if total_cost >= UNMATCHED_COST:
                continue
            track_id = track_ids[row]
            new_id = new_ids[col]
            result.primary_matches[new_id] = track_id

    matched_track_ids = set(result.primary_matches.values())
    matched_new_ids = set(result.primary_matches.keys())
    result.unmatched_track_ids = set(track_ids) - matched_track_ids
    result.unmatched_new_ids = set(new_ids) - matched_new_ids

    best_track_for_object: dict[int, int] = {}
    for object_id, scores in scores_by_object.items():
        if scores:
            best_track_for_object[object_id] = min(scores, key=lambda score: score.total_cost).track_id

    for new_id, scores in scores_by_object.items():
        related_track_ids = [score.track_id for score in sorted(scores, key=lambda score: score.total_cost)]
        if len(related_track_ids) <= 1:
            continue
        surviving_track_id = result.primary_matches.get(new_id)
        if surviving_track_id is None:
            continue
        # Only a track left without its own match can merge.  A track matched
        # to another object continues as that storm; merging it away would
        # turn the same storm into a different one.
        merge_track_ids = [
            track_id
            for track_id in related_track_ids
            if track_id != surviving_track_id and track_id not in matched_track_ids
        ]
        if merge_track_ids:
            result.merge_candidates[new_id] = [surviving_track_id] + merge_track_ids

    for track_id, scores in scores_by_track.items():
        related_new_ids = [score.object_id for score in sorted(scores, key=lambda score: score.total_cost)]
        if len(related_new_ids) <= 1:
            continue
        primary_new_id = None
        for new_id, matched_track_id in result.primary_matches.items():
            if matched_track_id == track_id:
                primary_new_id = new_id
                break
        if primary_new_id is None:
            continue
        # Only an object left without its own match can be a split child.  An
        # object matched to another track continues that storm.
        split_new_ids = [
            new_id
            for new_id in related_new_ids
            if new_id != primary_new_id
            and new_id not in matched_new_ids
            and best_track_for_object.get(new_id) == track_id
        ]
        if split_new_ids:
            result.split_candidates[track_id] = [primary_new_id] + split_new_ids

    if reacquisition_loader is not None:
        _reacquire_missing_tracks(
            result=result,
            missing_tracks=missing_tracks,
            current_scan=current_scan,
            new_objects=new_objects,
            new_masks=new_masks,
            pixel_motion=pixel_motion,
            geo_motion=geo_motion,
            dt_hours=dt_hours,
            reacquisition_loader=reacquisition_loader,
        )
    return result


def _reacquire_missing_tracks(
    *,
    result: AssociationResult,
    missing_tracks: list[Track],
    current_scan: BufferedScan,
    new_objects: dict,
    new_masks,
    pixel_motion,
    geo_motion: GeographicMotionFieldEstimate,
    dt_hours: float,
    reacquisition_loader: Callable[[datetime], BufferedScan | None],
) -> None:
    """Stage 2: missing tracks compete for the objects stage 1 left unclaimed.

    Runs only after stage 1 is final, so a continuously tracked storm always
    keeps its match.  A missing track's mask comes from the scan where it was
    last seen, shifted by the scene motion scaled to the full elapsed time,
    and a match must pass the advected-overlap threshold and the maximum
    storm speed over that elapsed time.  A candidate that is too old, or whose
    last-seen scan can no longer be loaded, is reported as unreacquirable.
    """
    claimed = set(result.primary_matches)
    for split_ids in result.split_candidates.values():
        claimed.update(split_ids)
    unclaimed = [object_id for object_id in new_objects if object_id not in claimed]
    candidates = [track for track in missing_tracks if track.last_seen_ref is not None]
    if not unclaimed or not candidates or dt_hours <= 0:
        return

    seen_scans: dict[datetime, BufferedScan | None] = {}
    new_mask_cache: dict[int, MaskExtent] = {}
    cost_matrix = np.full((len(candidates), len(unclaimed)), UNMATCHED_COST, dtype=float)
    for row, track in enumerate(candidates):
        seen_at, seen_object_id = track.last_seen_ref
        elapsed_hours = (current_scan.timestamp - seen_at).total_seconds() / 3600.0
        if elapsed_hours <= 0 or elapsed_hours * 60.0 > MAX_REACQUISITION_MINUTES:
            result.unreacquirable_track_ids.add(track.track_id)
            continue
        if seen_at not in seen_scans:
            seen_scans[seen_at] = reacquisition_loader(seen_at)
        seen_scan = seen_scans[seen_at]
        prev_mask = seen_scan.object_masks.get(seen_object_id) if seen_scan is not None else None
        if prev_mask is None:
            result.unreacquirable_track_ids.add(track.track_id)
            continue
        scale = elapsed_hours / dt_hours
        scaled_motion = GeographicMotionFieldEstimate(
            delta_lat=geo_motion.delta_lat * scale,
            delta_lon=geo_motion.delta_lon * scale,
            quality=geo_motion.quality,
            source=f"reacquisition:{geo_motion.source}",
        )
        last_object = track.current_object
        predicted_lat, predicted_lon = predict_latlon_position(
            last_object.centroid_lat, last_object.centroid_lon, scaled_motion
        )
        max_distance_km = MAX_STORM_SPEED_KMH * elapsed_hours
        prev_extent = mask_extent(prev_mask)
        shifted_extent = mask_extent(
            _shift_mask(prev_mask, -pixel_motion.shift_rows * scale, -pixel_motion.shift_cols * scale)
        )
        for col, object_id in enumerate(unclaimed):
            if object_id not in new_mask_cache:
                new_mask_cache[object_id] = mask_extent(new_masks[object_id])
            candidate_extent = new_mask_cache[object_id]
            if candidate_extent.mask.shape != prev_mask.shape:
                continue
            new_object = new_objects[object_id]
            score = _candidate_score(
                track=track,
                new_object=new_object,
                prev_extent=prev_extent,
                shifted_extent=shifted_extent,
                new_extent=candidate_extent,
                max_distance_km=max_distance_km,
                predicted_lat=predicted_lat,
                predicted_lon=predicted_lon,
            )
            if score is None or score.advected_overlap_score < MIN_ADVECTED_OVERLAP_PCT:
                continue
            centroid_km = haversine_distance_km(
                last_object.centroid_lat, last_object.centroid_lon,
                new_object.centroid_lat, new_object.centroid_lon,
            )
            if centroid_km > max(max_distance_km, 5.0):
                continue
            result.reacquisition_scores.append(score)
            cost_matrix[row, col] = score.total_cost

    for row, col in zip(*linear_sum_assignment(cost_matrix)):
        if cost_matrix[row, col] >= UNMATCHED_COST:
            continue
        object_id, track_id = unclaimed[col], candidates[row].track_id
        result.reacquired_matches[object_id] = track_id
        result.unmatched_new_ids.discard(object_id)
