import math
from dataclasses import dataclass, replace

import numpy as np
from scipy.ndimage import label

from src.detection import DetectedObject, polar_to_latlon, _range_bin_areas_km2
from src.parser import VelocityData

MIN_VELOCITY_MS = 10.0
MIN_REGION_AREA_KM2 = 4.0
CROSS_SWEEP_OVERLAP_THRESHOLD = 0.3


@dataclass
class VelocityRegion:
    region_type: str
    peak_velocity_ms: float
    mean_velocity_ms: float
    area_km2: float
    centroid_lat: float
    centroid_lon: float
    distance_km: float
    bearing_deg: float
    sweep_count: int
    elevation_angles: list[float]


def _detect_regions_single_sweep(
    velocity: np.ndarray,
    azimuths: np.ndarray,
    ranges_m: np.ndarray,
    radar_lat: float,
    radar_lon: float,
    elevation_angle: float,
) -> list[tuple[VelocityRegion, np.ndarray]]:
    """Detect inbound/outbound regions on a single sweep. Returns (region, mask) pairs."""
    range_bin_areas = _range_bin_areas_km2(azimuths, ranges_m)
    results: list[tuple[VelocityRegion, np.ndarray]] = []

    for region_type, condition in [
        ("inbound", velocity <= -MIN_VELOCITY_MS),
        ("outbound", velocity >= MIN_VELOCITY_MS),
    ]:
        valid = ~np.isnan(velocity) & condition
        labeled_grid, count = label(valid, structure=np.ones((3, 3), dtype=int))

        for component_id in range(1, count + 1):
            mask = labeled_grid == component_id
            az_indices, rng_indices = np.where(mask)
            area_km2 = float(np.sum(range_bin_areas[rng_indices]))

            if area_km2 < MIN_REGION_AREA_KM2:
                continue

            region_velocities = velocity[mask]
            if region_type == "inbound":
                peak = float(np.nanmin(region_velocities))
            else:
                peak = float(np.nanmax(region_velocities))
            mean = float(np.nanmean(region_velocities))

            weights = np.abs(region_velocities)
            weight_sum = float(np.sum(weights))
            if weight_sum == 0:
                continue
            centroid_az_idx = float(np.average(az_indices, weights=weights))
            centroid_rng_idx = float(np.average(rng_indices, weights=weights))
            centroid_az = float(np.interp(centroid_az_idx, range(len(azimuths)), azimuths))
            centroid_range = float(np.interp(centroid_rng_idx, range(len(ranges_m)), ranges_m))

            centroid_lat, centroid_lon = polar_to_latlon(
                radar_lat, radar_lon, centroid_az, centroid_range,
            )

            results.append((VelocityRegion(
                region_type=region_type,
                peak_velocity_ms=round(peak, 1),
                mean_velocity_ms=round(mean, 1),
                area_km2=round(area_km2, 2),
                centroid_lat=round(centroid_lat, 4),
                centroid_lon=round(centroid_lon, 4),
                distance_km=round(centroid_range / 1000.0, 1),
                bearing_deg=round(centroid_az % 360, 1),
                sweep_count=1,
                elevation_angles=[elevation_angle],
            ), mask))

    return results


def _merge_cross_sweep_regions(
    all_sweep_results: list[list[tuple[VelocityRegion, np.ndarray]]],
) -> list[VelocityRegion]:
    """Merge regions from multiple sweeps by spatial overlap."""
    if not all_sweep_results:
        return []

    merged: list[tuple[VelocityRegion, np.ndarray]] = []

    for sweep_results in all_sweep_results:
        for region, mask in sweep_results:
            matched = False
            for i, (existing_region, existing_mask) in enumerate(merged):
                if existing_region.region_type != region.region_type:
                    continue
                overlap = np.count_nonzero(mask & existing_mask)
                union = np.count_nonzero(mask | existing_mask)
                if union > 0 and overlap / union >= CROSS_SWEEP_OVERLAP_THRESHOLD:
                    if region.region_type == "inbound":
                        new_peak = min(existing_region.peak_velocity_ms, region.peak_velocity_ms)
                    else:
                        new_peak = max(existing_region.peak_velocity_ms, region.peak_velocity_ms)
                    merged[i] = (VelocityRegion(
                        region_type=existing_region.region_type,
                        peak_velocity_ms=new_peak,
                        mean_velocity_ms=round(
                            (existing_region.mean_velocity_ms + region.mean_velocity_ms) / 2, 1
                        ),
                        area_km2=max(existing_region.area_km2, region.area_km2),
                        centroid_lat=existing_region.centroid_lat,
                        centroid_lon=existing_region.centroid_lon,
                        distance_km=existing_region.distance_km,
                        bearing_deg=existing_region.bearing_deg,
                        sweep_count=existing_region.sweep_count + 1,
                        elevation_angles=existing_region.elevation_angles + region.elevation_angles,
                    ), existing_mask | mask)
                    matched = True
                    break
            if not matched:
                merged.append((region, mask))

    return [region for region, _ in merged]


MIN_SHEAR_MS = 15.0
MAX_COUPLET_DISTANCE_KM = 5.0
ROTATION_MERGE_DISTANCE_KM = 10.0


@dataclass
class RotationSignature:
    centroid_lat: float
    centroid_lon: float
    distance_km: float
    bearing_deg: float
    max_shear_ms: float
    max_inbound_ms: float
    max_outbound_ms: float
    diameter_km: float
    sweep_count: int
    elevation_angles: list[float]
    strength: str


def _classify_rotation_strength(shear_ms: float) -> str:
    if shear_ms >= 35.0:
        return "strong"
    if shear_ms >= 25.0:
        return "moderate"
    return "weak"


def _detect_shear_single_sweep(
    velocity: np.ndarray,
    azimuths: np.ndarray,
    ranges_m: np.ndarray,
    radar_lat: float,
    radar_lon: float,
    elevation_angle: float,
) -> list[tuple[RotationSignature, float, float]]:
    """Find gate-to-gate shear couplets on one sweep.

    Returns (signature, centroid_az, centroid_range_m) tuples for cross-sweep merging.
    """
    n_az, n_rng = velocity.shape
    range_spacing_m = float(ranges_m[1] - ranges_m[0]) if len(ranges_m) > 1 else 250.0
    az_spacing_deg = float(azimuths[1] - azimuths[0]) if len(azimuths) > 1 else 1.0

    shear_mask = np.zeros_like(velocity, dtype=bool)
    shear_values = np.full_like(velocity, np.nan)
    inbound_values = np.full_like(velocity, np.nan)
    outbound_values = np.full_like(velocity, np.nan)

    # Check azimuthal neighbors for sign changes
    for delta_az in [-1, 0, 1]:
        for delta_rng in [-1, 0, 1]:
            if delta_az == 0 and delta_rng == 0:
                continue
            shifted_az = np.roll(velocity, delta_az, axis=0)
            shifted_rng = np.roll(shifted_az, delta_rng, axis=1)

            gate_distance_km = math.sqrt(
                (delta_az * az_spacing_deg * math.pi / 180 * ranges_m.mean()) ** 2
                + (delta_rng * range_spacing_m) ** 2
            ) / 1000.0

            if gate_distance_km > MAX_COUPLET_DISTANCE_KM:
                continue

            v1 = velocity
            v2 = shifted_rng
            both_valid = ~np.isnan(v1) & ~np.isnan(v2)
            sign_change = both_valid & ((v1 < 0) != (v2 < 0))
            shear = np.where(sign_change, np.abs(v1 - v2), np.nan)
            strong_shear = ~np.isnan(shear) & (shear >= MIN_SHEAR_MS)

            new_shear = strong_shear & (~shear_mask | (shear > shear_values))
            shear_mask |= strong_shear
            shear_values = np.where(new_shear, shear, shear_values)
            inbound_values = np.where(
                new_shear,
                np.minimum(v1, v2),
                inbound_values,
            )
            outbound_values = np.where(
                new_shear,
                np.maximum(v1, v2),
                outbound_values,
            )

    labeled_shear, count = label(shear_mask, structure=np.ones((3, 3), dtype=int))
    results: list[tuple[RotationSignature, float, float]] = []

    for component_id in range(1, count + 1):
        component_mask = labeled_shear == component_id
        if np.count_nonzero(component_mask) < 2:
            continue

        az_indices, rng_indices = np.where(component_mask)
        component_shear = shear_values[component_mask]
        max_shear = float(np.nanmax(component_shear))
        max_inbound = float(np.nanmin(inbound_values[component_mask]))
        max_outbound = float(np.nanmax(outbound_values[component_mask]))

        weights = np.nan_to_num(component_shear, nan=0.0)
        weight_sum = float(np.sum(weights))
        if weight_sum == 0:
            continue
        centroid_az_idx = float(np.average(az_indices, weights=weights))
        centroid_rng_idx = float(np.average(rng_indices, weights=weights))
        centroid_az = float(np.interp(centroid_az_idx, range(len(azimuths)), azimuths))
        centroid_range = float(np.interp(centroid_rng_idx, range(len(ranges_m)), ranges_m))

        az_extent = (float(np.max(az_indices)) - float(np.min(az_indices))) * az_spacing_deg
        rng_extent = (float(np.max(rng_indices)) - float(np.min(rng_indices))) * range_spacing_m
        diameter_km = math.sqrt(
            (az_extent * math.pi / 180 * centroid_range) ** 2
            + rng_extent ** 2
        ) / 1000.0

        centroid_lat, centroid_lon = polar_to_latlon(
            radar_lat, radar_lon, centroid_az, centroid_range,
        )

        results.append((RotationSignature(
            centroid_lat=round(centroid_lat, 4),
            centroid_lon=round(centroid_lon, 4),
            distance_km=round(centroid_range / 1000.0, 1),
            bearing_deg=round(centroid_az % 360, 1),
            max_shear_ms=round(max_shear, 1),
            max_inbound_ms=round(max_inbound, 1),
            max_outbound_ms=round(max_outbound, 1),
            diameter_km=round(diameter_km, 1),
            sweep_count=1,
            elevation_angles=[elevation_angle],
            strength=_classify_rotation_strength(max_shear),
        ), centroid_az, centroid_range))

    return results


def _merge_cross_sweep_rotations(
    all_sweep_results: list[list[tuple[RotationSignature, float, float]]],
    ranges_m: np.ndarray,
) -> list[RotationSignature]:
    """Merge rotation signatures from multiple sweeps by proximity."""
    if not all_sweep_results:
        return []

    merged: list[tuple[RotationSignature, float, float]] = []

    for sweep_results in all_sweep_results:
        for sig, az, rng in sweep_results:
            matched = False
            for i, (existing_sig, existing_az, existing_rng) in enumerate(merged):
                az_dist_deg = abs(az - existing_az)
                if az_dist_deg > 180:
                    az_dist_deg = 360 - az_dist_deg
                rng_dist_m = abs(rng - existing_rng)
                approx_dist_km = math.sqrt(
                    (az_dist_deg * math.pi / 180 * rng) ** 2
                    + rng_dist_m ** 2
                ) / 1000.0

                if approx_dist_km <= ROTATION_MERGE_DISTANCE_KM:
                    merged[i] = (RotationSignature(
                        centroid_lat=existing_sig.centroid_lat,
                        centroid_lon=existing_sig.centroid_lon,
                        distance_km=existing_sig.distance_km,
                        bearing_deg=existing_sig.bearing_deg,
                        max_shear_ms=max(existing_sig.max_shear_ms, sig.max_shear_ms),
                        max_inbound_ms=min(existing_sig.max_inbound_ms, sig.max_inbound_ms),
                        max_outbound_ms=max(existing_sig.max_outbound_ms, sig.max_outbound_ms),
                        diameter_km=max(existing_sig.diameter_km, sig.diameter_km),
                        sweep_count=existing_sig.sweep_count + 1,
                        elevation_angles=existing_sig.elevation_angles + sig.elevation_angles,
                        strength=_classify_rotation_strength(
                            max(existing_sig.max_shear_ms, sig.max_shear_ms)
                        ),
                    ), existing_az, existing_rng)
                    matched = True
                    break
            if not matched:
                merged.append((sig, az, rng))

    return [sig for sig, _, _ in merged]


def detect_velocity_regions(vel_data: VelocityData) -> list[VelocityRegion]:
    """Detect inbound/outbound velocity regions across all sweeps."""
    all_sweep_results: list[list[tuple[VelocityRegion, np.ndarray]]] = []

    for sweep in vel_data.sweeps:
        sweep_results = _detect_regions_single_sweep(
            velocity=sweep.velocity,
            azimuths=sweep.azimuths,
            ranges_m=sweep.ranges_m,
            radar_lat=vel_data.radar_lat,
            radar_lon=vel_data.radar_lon,
            elevation_angle=sweep.elevation_angle,
        )
        all_sweep_results.append(sweep_results)

    return _merge_cross_sweep_regions(all_sweep_results)


def detect_rotation_signatures(vel_data: VelocityData) -> list[RotationSignature]:
    """Detect rotation signatures across all sweeps."""
    all_sweep_results: list[list[tuple[RotationSignature, float, float]]] = []

    for sweep in vel_data.sweeps:
        sweep_results = _detect_shear_single_sweep(
            velocity=sweep.velocity,
            azimuths=sweep.azimuths,
            ranges_m=sweep.ranges_m,
            radar_lat=vel_data.radar_lat,
            radar_lon=vel_data.radar_lon,
            elevation_angle=sweep.elevation_angle,
        )
        all_sweep_results.append(sweep_results)

    ranges_m = vel_data.sweeps[0].ranges_m if vel_data.sweeps else np.array([])
    return _merge_cross_sweep_rotations(all_sweep_results, ranges_m)


MAX_ASSOCIATION_DISTANCE_KM = 30.0


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def analyze_velocity(
    vel_data: "VelocityData | None",
    objects: list[DetectedObject],
) -> tuple[list[VelocityRegion], list[RotationSignature], list[DetectedObject]]:
    """Run full velocity analysis and associate results with detected objects.

    Returns (regions, rotation_signatures, annotated_objects).
    """
    if vel_data is None:
        return [], [], objects

    regions = detect_velocity_regions(vel_data)
    rotations = detect_rotation_signatures(vel_data)

    annotated = list(objects)
    for i, obj in enumerate(annotated):
        best_inbound: float | None = None
        best_outbound: float | None = None
        best_rotation: RotationSignature | None = None
        best_rotation_dist = float("inf")

        for region in regions:
            dist = _haversine_km(
                obj.centroid_lat, obj.centroid_lon,
                region.centroid_lat, region.centroid_lon,
            )
            if dist > MAX_ASSOCIATION_DISTANCE_KM:
                continue
            if region.region_type == "inbound":
                if best_inbound is None or region.peak_velocity_ms < best_inbound:
                    best_inbound = region.peak_velocity_ms
            else:
                if best_outbound is None or region.peak_velocity_ms > best_outbound:
                    best_outbound = region.peak_velocity_ms

        for rotation in rotations:
            dist = _haversine_km(
                obj.centroid_lat, obj.centroid_lon,
                rotation.centroid_lat, rotation.centroid_lon,
            )
            if dist < best_rotation_dist and dist <= MAX_ASSOCIATION_DISTANCE_KM:
                best_rotation_dist = dist
                best_rotation = rotation

        annotated[i] = replace(
            obj,
            max_inbound_ms=best_inbound,
            max_outbound_ms=best_outbound,
            rotation=best_rotation,
        )

    return regions, rotations, annotated
