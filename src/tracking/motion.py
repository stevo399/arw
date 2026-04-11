import math
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from src.detection import degrees_to_bearing
from src.tracking.motion_field import GeographicMotionFieldEstimate
from src.tracking.types import MotionConfidence

KM_PER_DEGREE_LAT = 111.32
KM_PER_MILE = 1.60934
NEARLY_STATIONARY_KMH = 2.0
MAX_REASONABLE_SPEED_KMH = 160.0
HIGH_JUMP_RATIO = 2.5
HIGH_IDENTITY_CONFIDENCE = 0.75
MEDIUM_IDENTITY_CONFIDENCE = 0.45
MIN_FIELD_QUALITY = 0.35
STRONG_FIELD_QUALITY = 0.5
MAX_HISTORY_FIELD_SPEED_DELTA_KMH = 30.0
MAX_HISTORY_FIELD_RATIO = 1.8
MAX_HISTORY_FIELD_HEADING_DELTA_DEG = 75.0


@dataclass
class MotionVector:
    """Computed motion for a storm track."""
    speed_kmh: float
    speed_mph: int
    heading_deg: float | None
    heading_label: str
    confidence: MotionConfidence | None = None
    source: str = "track_history"


def _speed_components_kmh(times_s: np.ndarray, lats: np.ndarray, lons: np.ndarray) -> tuple[float, float]:
    lat_slope = np.polyfit(times_s, lats, 1)[0]
    lon_slope = np.polyfit(times_s, lons, 1)[0]
    mean_lat = np.mean(lats)
    km_per_degree_lon = KM_PER_DEGREE_LAT * math.cos(math.radians(mean_lat))
    lat_kmh = lat_slope * 3600 * KM_PER_DEGREE_LAT
    lon_kmh = lon_slope * 3600 * km_per_degree_lon
    return lat_kmh, lon_kmh


def _step_speeds_kmh(positions: list[tuple[datetime, float, float]]) -> list[float]:
    speeds = []
    for (t1, lat1, lon1), (t2, lat2, lon2) in zip(positions, positions[1:]):
        hours = (t2 - t1).total_seconds() / 3600.0
        if hours <= 0:
            continue
        mean_lat = (lat1 + lat2) / 2.0
        delta_lat_km = (lat2 - lat1) * KM_PER_DEGREE_LAT
        delta_lon_km = (lon2 - lon1) * KM_PER_DEGREE_LAT * math.cos(math.radians(mean_lat))
        distance = math.sqrt(delta_lat_km ** 2 + delta_lon_km ** 2)
        speeds.append(distance / hours)
    return speeds


def _motion_confidence(
    positions: list[tuple[datetime, float, float]],
    speed_kmh: float,
    times_s: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
) -> MotionConfidence:
    if len(positions) < 2:
        return MotionConfidence(label="low", score=0.0, reason="single position")

    if times_s[-1] == 0.0:
        return MotionConfidence(label="low", score=0.0, reason="zero elapsed time")

    if speed_kmh > MAX_REASONABLE_SPEED_KMH:
        return MotionConfidence(label="low", score=0.0, reason="speed exceeds plausibility threshold")

    step_speeds = _step_speeds_kmh(positions)
    if step_speeds:
        max_step = max(step_speeds)
        min_step = min(step_speeds)
        if min_step > 0 and (max_step / min_step) > HIGH_JUMP_RATIO:
            return MotionConfidence(label="low", score=0.2, reason="step speeds are inconsistent")
        if max_step > MAX_REASONABLE_SPEED_KMH:
            return MotionConfidence(label="low", score=0.0, reason="step speed exceeds plausibility threshold")

    if len(positions) == 2:
        return MotionConfidence(label="medium", score=0.6, reason="two-point estimate")

    lat_fit = np.polyfit(times_s, lats, 1)
    lon_fit = np.polyfit(times_s, lons, 1)
    lat_pred = np.polyval(lat_fit, times_s)
    lon_pred = np.polyval(lon_fit, times_s)
    lat_resid = float(np.sqrt(np.mean((lats - lat_pred) ** 2)))
    lon_resid = float(np.sqrt(np.mean((lons - lon_pred) ** 2)))
    residual = lat_resid + lon_resid
    if residual > 0.05:
        return MotionConfidence(label="low", score=0.25, reason="high regression residual")
    if residual > 0.01:
        return MotionConfidence(label="medium", score=0.55, reason="moderate regression residual")
    return MotionConfidence(label="high", score=0.9, reason="stable multi-point history")


def compute_motion(positions: list[tuple[datetime, float, float]]) -> MotionVector:
    """Compute a confidence-aware motion vector from timestamped positions."""
    stationary_confidence = MotionConfidence(label="high", score=1.0, reason="insufficient movement history")
    if len(positions) < 2:
        return MotionVector(
            speed_kmh=0.0,
            speed_mph=0,
            heading_deg=None,
            heading_label="stationary",
            confidence=stationary_confidence,
            source="track_history",
        )

    t0 = positions[0][0]
    times_s = np.array([(p[0] - t0).total_seconds() for p in positions])
    lats = np.array([p[1] for p in positions])
    lons = np.array([p[2] for p in positions])

    if times_s[-1] == 0.0:
        return MotionVector(
            speed_kmh=0.0,
            speed_mph=0,
            heading_deg=None,
            heading_label="stationary",
            confidence=MotionConfidence(label="low", score=0.0, reason="zero elapsed time"),
            source="track_history",
        )

    lat_kmh, lon_kmh = _speed_components_kmh(times_s, lats, lons)
    speed_kmh = round(math.sqrt(lat_kmh ** 2 + lon_kmh ** 2), 1)
    confidence = _motion_confidence(positions, speed_kmh, times_s, lats, lons)

    if confidence.label == "low" and speed_kmh >= NEARLY_STATIONARY_KMH:
        return MotionVector(
            speed_kmh=speed_kmh,
            speed_mph=round(speed_kmh / KM_PER_MILE),
            heading_deg=None,
            heading_label="uncertain",
            confidence=confidence,
            source="track_history",
        )

    if speed_kmh < NEARLY_STATIONARY_KMH:
        return MotionVector(
            speed_kmh=speed_kmh,
            speed_mph=round(speed_kmh / KM_PER_MILE),
            heading_deg=None,
            heading_label="nearly stationary",
            confidence=confidence,
            source="track_history",
        )

    heading_rad = math.atan2(lon_kmh, lat_kmh)
    heading_deg = round(math.degrees(heading_rad) % 360, 1)
    heading_label = degrees_to_bearing(heading_deg)
    speed_mph = round(speed_kmh / KM_PER_MILE)

    return MotionVector(
        speed_kmh=speed_kmh,
        speed_mph=speed_mph,
        heading_deg=heading_deg,
        heading_label=heading_label,
        confidence=confidence,
        source="track_history",
    )


def motion_from_field(
    estimate: GeographicMotionFieldEstimate | None,
    dt_hours: float,
) -> MotionVector | None:
    """Convert a bulk geographic motion-field estimate into a reportable vector."""
    if estimate is None or dt_hours <= 0:
        return None

    delta_lat_km = estimate.delta_lat * KM_PER_DEGREE_LAT
    mean_lat = 35.0
    km_per_degree_lon = KM_PER_DEGREE_LAT * math.cos(math.radians(mean_lat))
    delta_lon_km = estimate.delta_lon * km_per_degree_lon
    speed_kmh = round(math.sqrt(delta_lat_km ** 2 + delta_lon_km ** 2) / dt_hours, 1)
    quality = max(min(estimate.quality, 1.0), 0.0)

    if speed_kmh > MAX_REASONABLE_SPEED_KMH:
        return suppress_motion("motion field exceeds plausibility threshold")

    if speed_kmh < NEARLY_STATIONARY_KMH:
        confidence_label = "medium" if quality >= MIN_FIELD_QUALITY else "low"
        return MotionVector(
            speed_kmh=speed_kmh,
            speed_mph=round(speed_kmh / KM_PER_MILE),
            heading_deg=None,
            heading_label="nearly stationary",
            confidence=MotionConfidence(
                label=confidence_label,
                score=round(quality, 2),
                reason=f"scene motion field from {estimate.source}",
            ),
            source="motion_field",
        )

    heading_rad = math.atan2(delta_lon_km, delta_lat_km)
    heading_deg = round(math.degrees(heading_rad) % 360, 1)
    heading_label = degrees_to_bearing(heading_deg)
    confidence_label = "high" if quality >= 0.75 else "medium" if quality >= MIN_FIELD_QUALITY else "low"
    return MotionVector(
        speed_kmh=speed_kmh,
        speed_mph=round(speed_kmh / KM_PER_MILE),
        heading_deg=heading_deg,
        heading_label=heading_label,
        confidence=MotionConfidence(
            label=confidence_label,
            score=round(quality, 2),
            reason=f"scene motion field from {estimate.source}",
        ),
        source="motion_field",
    )


def _heading_delta_deg(a: float | None, b: float | None) -> float:
    if a is None or b is None:
        return 0.0
    delta = abs(a - b) % 360.0
    return min(delta, 360.0 - delta)


def _history_disagrees_with_field(history_motion: MotionVector, field_motion: MotionVector | None) -> bool:
    if field_motion is None or field_motion.confidence is None:
        return False
    if field_motion.confidence.score < STRONG_FIELD_QUALITY:
        return False
    if history_motion.heading_label in {"stationary", "nearly stationary", "uncertain"}:
        return False

    speed_delta = abs(history_motion.speed_kmh - field_motion.speed_kmh)
    field_speed_floor = max(field_motion.speed_kmh, 5.0)
    speed_ratio = history_motion.speed_kmh / field_speed_floor
    heading_delta = _heading_delta_deg(history_motion.heading_deg, field_motion.heading_deg)

    return (
        speed_delta >= MAX_HISTORY_FIELD_SPEED_DELTA_KMH
        and speed_ratio >= MAX_HISTORY_FIELD_RATIO
    ) or (
        field_motion.speed_kmh >= 5.0
        and history_motion.speed_kmh >= 20.0
        and heading_delta >= MAX_HISTORY_FIELD_HEADING_DELTA_DEG
    )


def suppress_motion(reason: str) -> MotionVector:
    """Return a deliberately non-publishable motion estimate."""
    return MotionVector(
        speed_kmh=0.0,
        speed_mph=0,
        heading_deg=None,
        heading_label="uncertain",
        confidence=MotionConfidence(label="low", score=0.0, reason=reason),
        source="suppressed",
    )


def resolve_reported_motion(
    positions: list[tuple[datetime, float, float]],
    *,
    identity_confidence: float,
    field_estimate: GeographicMotionFieldEstimate | None = None,
    field_dt_hours: float = 0.0,
) -> tuple[MotionVector, MotionVector]:
    """Return (reported_motion, diagnostic_history_motion) using proven-practice fallbacks."""
    history_motion = compute_motion(positions)
    field_motion = motion_from_field(field_estimate, field_dt_hours)

    if _history_disagrees_with_field(history_motion, field_motion):
        return field_motion, history_motion

    if identity_confidence >= HIGH_IDENTITY_CONFIDENCE and history_motion.confidence is not None:
        if history_motion.confidence.label != "low":
            return history_motion, history_motion

    if identity_confidence >= MEDIUM_IDENTITY_CONFIDENCE and history_motion.confidence is not None:
        if history_motion.confidence.label == "medium":
            return history_motion, history_motion

    if field_motion is not None and field_motion.confidence is not None:
        if field_motion.confidence.score >= MIN_FIELD_QUALITY:
            return field_motion, history_motion

    if history_motion.heading_label in {"stationary", "nearly stationary"}:
        return history_motion, history_motion

    return suppress_motion("track identity too weak for publishable motion"), history_motion
