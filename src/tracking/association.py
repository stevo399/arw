from dataclasses import dataclass, field
from datetime import datetime
import math

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.buffer import BufferedScan
from src.sites import haversine_distance_km
from src.tracking.motion_field import estimate_geographic_motion_field, estimate_scan_geographic_motion_field, predict_latlon_position
from src.tracking.segmentation import segment_buffered_scan
from src.tracking.types import AssociationScore, Track

MIN_OVERLAP_PCT = 0.30
MAX_STORM_SPEED_KMH = 120.0
UNMATCHED_COST = 10.0


@dataclass
class AssociationResult:
    primary_matches: dict[int, int] = field(default_factory=dict)  # new_obj_id -> track_id
    merge_candidates: dict[int, list[int]] = field(default_factory=dict)  # new_obj_id -> track_ids
    split_candidates: dict[int, list[int]] = field(default_factory=dict)  # track_id -> new_obj_ids
    unmatched_new_ids: set[int] = field(default_factory=set)
    unmatched_track_ids: set[int] = field(default_factory=set)
    candidate_scores: list[AssociationScore] = field(default_factory=list)
    geo_motion: object | None = None
    dt_hours: float = 0.0


def compute_overlap(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Compute the fraction of mask_a covered by mask_b."""
    if not np.any(mask_a):
        return 0.0
    intersection = np.sum(mask_a & mask_b)
    return float(intersection) / float(np.sum(mask_a))


def _candidate_score(
    track: Track,
    new_object,
    prev_mask: np.ndarray,
    new_mask: np.ndarray,
    max_distance_km: float,
    predicted_lat: float,
    predicted_lon: float,
) -> AssociationScore | None:
    current_object = track.current_object
    if current_object is None:
        return None

    overlap = compute_overlap(prev_mask, new_mask)
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
    if overlap < MIN_OVERLAP_PCT and centroid_distance > plausible_distance:
        return None

    overlap_cost = 1.0 - overlap
    distance_score = min(centroid_distance / plausible_distance, 5.0)
    predicted_score = min(predicted_distance / plausible_distance, 5.0)
    area_change = abs(new_object.area_km2 - current_object.area_km2) / max(current_object.area_km2, 1.0)
    intensity_change = abs(new_object.peak_dbz - current_object.peak_dbz) / 60.0
    total_cost = overlap_cost + (predicted_score * 0.6) + (distance_score * 0.2) + (area_change * 0.15) + (intensity_change * 0.05)

    return AssociationScore(
        track_id=track.track_id,
        object_id=new_object.object_id,
        overlap_score=round(overlap, 4),
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
) -> AssociationResult:
    """Associate active tracks to new objects using a global cost matrix."""
    result = AssociationResult()
    active_tracks = [track for track in tracks if track.status == "active" and track.current_object is not None]
    if not active_tracks:
        result.unmatched_new_ids = {obj.object_id for obj in current_scan.detected_objects}
        return result

    previous_segmentation = segment_buffered_scan(previous_scan)
    current_segmentation = segment_buffered_scan(current_scan)
    geo_motion = estimate_scan_geographic_motion_field(previous_scan, current_scan)
    if geo_motion.quality <= 0.0:
        geo_motion = estimate_geographic_motion_field(previous_segmentation.objects, current_segmentation.objects)
    result.geo_motion = geo_motion

    previous_objects = {obj.object_id: obj for obj in previous_scan.detected_objects}
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

    for track in active_tracks:
        current_object = track.current_object
        if current_object is None:
            continue
        predicted_lat, predicted_lon = predict_latlon_position(
            current_object.centroid_lat,
            current_object.centroid_lon,
            geo_motion,
        )
        prev_object_id = None
        for object_id, track_id in obj_to_track.items():
            if track_id == track.track_id:
                prev_object_id = object_id
                break
        if prev_object_id is None or prev_object_id not in prev_masks:
            continue
        prev_mask = prev_masks[prev_object_id]
        for new_id, new_object in new_objects.items():
            score = _candidate_score(
                track=track,
                new_object=new_object,
                prev_mask=prev_mask,
                new_mask=new_masks[new_id],
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
        merge_track_ids = [track_id for track_id in related_track_ids if track_id != surviving_track_id]
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
        split_new_ids = [new_id for new_id in related_new_ids if new_id != primary_new_id and best_track_for_object.get(new_id) == track_id]
        if split_new_ids:
            result.split_candidates[track_id] = [primary_new_id] + split_new_ids

    return result
