"""Storm motion as reported: measured velocities and the rules that choose among them.

Velocities are measured by pattern matching (`src.tracking.pattern_motion`).
Motion is never inferred from a storm's centre track: centres jitter by about
2 km per scan and jump when storms merge or split, and a velocity from two
centres predicted a storm's next position worse than assuming no motion
(docs/test_reports/2026-09-13-motion-prediction-evaluation.md).
"""

import math
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from src.detection import degrees_to_bearing
from src.tracking.types import MAX_MEASURED_VELOCITIES, MotionConfidence, VelocitySample

KM_PER_DEGREE_LAT = 111.32
KM_PER_MILE = 1.60934
# A measured speed below this cannot be told from no motion.  A storm's single
# pattern velocity changes between consecutive scans by a median of 3.8 km/h,
# bounding per-component noise at about 2.3 km/h; 6 km/h is the 95% bound on
# the measured speed of a motionless storm under that noise
# (docs/test_reports/2026-09-13-pattern-motion-evaluation.md).
NEARLY_STATIONARY_KMH = 6.0
MIN_HEADING_CHECK_SPEED_KMH = 5.0
MAX_STEP_HEADING_DELTA_DEG = 90.0


@dataclass
class MotionVector:
    """Computed motion for a storm track.  Speed is None when motion is unknown."""
    speed_kmh: float | None
    speed_mph: int | None
    heading_deg: float | None
    heading_label: str
    confidence: MotionConfidence | None = None
    source: str = "track_history"


def _step_headings_deg(positions: list[tuple[datetime, float, float]]) -> list[float]:
    headings = []
    for (t1, lat1, lon1), (t2, lat2, lon2) in zip(positions, positions[1:]):
        hours = (t2 - t1).total_seconds() / 3600.0
        if hours <= 0:
            continue
        mean_lat = (lat1 + lat2) / 2.0
        delta_lat_km = (lat2 - lat1) * KM_PER_DEGREE_LAT
        delta_lon_km = (lon2 - lon1) * KM_PER_DEGREE_LAT * math.cos(math.radians(mean_lat))
        speed_kmh = math.sqrt(delta_lat_km ** 2 + delta_lon_km ** 2) / hours
        if speed_kmh < MIN_HEADING_CHECK_SPEED_KMH:
            continue
        heading_rad = math.atan2(delta_lon_km, delta_lat_km)
        headings.append(math.degrees(heading_rad) % 360)
    return headings


def recent_heading_flip_count(positions: list[tuple[datetime, float, float]], *, max_steps: int = 4) -> int:
    """Count large recent per-step heading reversals over a short history window."""
    if max_steps < 2:
        return 0
    step_headings = _step_headings_deg(positions)
    if len(step_headings) < 2:
        return 0
    recent_headings = step_headings[-max_steps:]
    return sum(
        1
        for previous_heading, current_heading in zip(recent_headings, recent_headings[1:])
        if _heading_delta_deg(previous_heading, current_heading) >= MAX_STEP_HEADING_DELTA_DEG
    )


def _heading_delta_deg(a: float | None, b: float | None) -> float:
    if a is None or b is None:
        return 0.0
    delta = abs(a - b) % 360.0
    return min(delta, 360.0 - delta)


MEASURED_SOURCE = "pattern_match"
NEARBY_SOURCE = "nearby_storms"
UNKNOWN_SOURCE = "not_measured"


def _vector_motion(east_kmh: float, north_kmh: float, source: str, confidence: MotionConfidence) -> MotionVector:
    speed_kmh = round(math.hypot(east_kmh, north_kmh), 1)
    if speed_kmh < NEARLY_STATIONARY_KMH:
        heading_deg = None
        heading_label = "nearly stationary"
    else:
        heading_deg = round(math.degrees(math.atan2(east_kmh, north_kmh)) % 360.0, 1)
        heading_label = degrees_to_bearing(heading_deg)
    return MotionVector(
        speed_kmh=speed_kmh,
        speed_mph=round(speed_kmh / KM_PER_MILE),
        heading_deg=heading_deg,
        heading_label=heading_label,
        confidence=confidence,
        source=source,
    )


def unknown_motion(reason: str = "no measured motion for this storm or nearby storms") -> MotionVector:
    return MotionVector(
        speed_kmh=None,
        speed_mph=None,
        heading_deg=None,
        heading_label="unknown",
        confidence=MotionConfidence(label="low", score=0.0, reason=reason),
        source=UNKNOWN_SOURCE,
    )


def report_motion(
    position_count: int,
    measured: list[VelocitySample],
    nearby: tuple[float, float] | None,
) -> MotionVector:
    """The motion a storm reports, by the first rule that applies.

    1. A storm seen in at least two scans with a measured velocity reports the
       median of its last three measured velocities.
    2. Otherwise nearby storms' measured velocity, reported as theirs.  A storm
       seen once always takes this rule: nearby storms' motion predicted its
       next position better than its own first measurement.
    3. Otherwise motion is unknown.
    """
    recent = measured[-MAX_MEASURED_VELOCITIES:]
    if position_count >= 2 and recent:
        east = float(np.median([sample.east_kmh for sample in recent]))
        north = float(np.median([sample.north_kmh for sample in recent]))
        if len(recent) >= MAX_MEASURED_VELOCITIES:
            confidence = MotionConfidence(label="high", score=0.9, reason="median of three pattern-matched scans")
        else:
            scans = "scan" if len(recent) == 1 else "scans"
            confidence = MotionConfidence(label="medium", score=0.6, reason=f"pattern matched over {len(recent)} {scans}")
        return _vector_motion(east, north, MEASURED_SOURCE, confidence)
    if nearby is not None:
        return _vector_motion(
            nearby[0], nearby[1], NEARBY_SOURCE,
            MotionConfidence(label="medium", score=0.5, reason="measured motion of storms within 60 km"),
        )
    return unknown_motion()
