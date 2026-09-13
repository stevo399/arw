"""Storm growth and decay from a track's recent per-scan samples.

Area comes from one low-elevation reflectivity sweep, so a trend is at best
medium confidence.  A window that includes a reacquired match is not
reported: the storm was unobserved for a scan inside it.
"""

from dataclasses import dataclass

from src.tracking.types import TrendSample

MIN_TREND_SAMPLES = 3
# Relative change from the first to the last sample that counts as growth or decay.
TREND_CHANGE_FRACTION = 0.15


@dataclass
class StormTrend:
    area: str  # "growing", "steady", "decaying" or "insufficient"
    core_area: str  # the same labels, or "none" when no >= 50 dBZ core was present
    confidence: str  # "medium", "low" or "none"
    reason: str
    sample_count: int


def _classify(first: float, last: float) -> str:
    change = (last - first) / max(first, 1.0)
    if change > TREND_CHANGE_FRACTION:
        return "growing"
    if change < -TREND_CHANGE_FRACTION:
        return "decaying"
    return "steady"


def _consistent(values: list[float], label: str) -> bool:
    steps = [later - earlier for earlier, later in zip(values, values[1:])]
    if label == "growing":
        return all(step >= 0 for step in steps)
    if label == "decaying":
        return all(step <= 0 for step in steps)
    return True


def compute_trend(samples: list[TrendSample]) -> StormTrend:
    window = list(samples)
    count = len(window)
    if count < MIN_TREND_SAMPLES:
        return StormTrend("insufficient", "insufficient", "none", f"needs {MIN_TREND_SAMPLES} scans, has {count}", count)
    if any(sample.reacquired for sample in window):
        return StormTrend("insufficient", "insufficient", "none", "window includes a reacquired match", count)
    areas = [sample.area_km2 for sample in window]
    cores = [sample.core_area_km2 for sample in window]
    area = _classify(areas[0], areas[-1])
    core = "none" if max(cores) == 0.0 else _classify(cores[0], cores[-1])
    consistent = _consistent(areas, area) and (core == "none" or _consistent(cores, core))
    return StormTrend(
        area=area,
        core_area=core,
        confidence="medium" if consistent else "low",
        reason="consistent across the window" if consistent else "direction changes within the window",
        sample_count=count,
    )
