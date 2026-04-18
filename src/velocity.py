import math
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import label

from src.detection import polar_to_latlon, _range_bin_areas_km2
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
