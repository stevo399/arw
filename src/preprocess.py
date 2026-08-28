from dataclasses import dataclass, field, replace

import numpy as np

from src.geometry import label_periodic_azimuth
from src.parser import SweepData

MIN_DBZ_FOR_OBJECTS = 20.0
MAX_SPECKLE_PIXELS = 3
MIN_SPECKLE_PEAK_DBZ_TO_KEEP = 35.0
HIGH_MISSING_FRACTION = 0.35
NOTICEABLE_SPECKLE_FRACTION = 0.005

# Every other SweepData field that shares reflectivity's (n_rays, n_gates)
# grid. Speckle removal must blank these gates too, or SweepData's
# "co-registered fields" promise is broken -- a gate reflectivity dropped
# would still report a RhoHV/ZDR/PhiDP reading.
CO_REGISTERED_FIELD_NAMES = (
    "velocity",
    "rhohv",
    "zdr",
    "phidp",
    "spectrum_width",
    "clutter_power_removed",
    "gate_classification",
)


@dataclass
class ScanQuality:
    score: float
    finite_fraction: float
    removed_speckle_pixels: int
    removed_speckle_fraction: float
    flags: list[str]
    class_fractions: dict[str, float] = field(default_factory=dict)
    # Advisory only -- fraction of gates quality control WOULD flag as
    # non-meteorological, not a fraction actually removed. See the
    # 2026-08-26 amendment note in src/qc/report.py.
    advisory_fraction: float = 0.0
    mean_confidence: float = 0.0
    degraded_modes: list[str] = field(default_factory=list)


def _remove_weak_speckle(reflectivity: np.ndarray) -> tuple[np.ndarray, int]:
    processed = np.array(reflectivity, copy=True)
    storm_like = ~np.isnan(processed) & (processed >= MIN_DBZ_FOR_OBJECTS)
    labeled, count = label_periodic_azimuth(storm_like, structure=np.ones((3, 3), dtype=int))
    removed_pixels = 0
    for component_id in range(1, count + 1):
        component_mask = labeled == component_id
        component_pixels = int(np.count_nonzero(component_mask))
        if component_pixels > MAX_SPECKLE_PIXELS:
            continue
        peak_dbz = float(np.nanmax(processed[component_mask]))
        if peak_dbz >= MIN_SPECKLE_PEAK_DBZ_TO_KEEP:
            continue
        processed[component_mask] = np.nan
        removed_pixels += component_pixels
    return processed, removed_pixels


def assess_scan_quality(
    original_reflectivity: np.ndarray,
    processed_reflectivity: np.ndarray,
    removed_speckle_pixels: int,
) -> ScanQuality:
    total_pixels = int(original_reflectivity.size)
    finite_fraction = float(np.count_nonzero(~np.isnan(original_reflectivity))) / float(total_pixels)
    removed_fraction = float(removed_speckle_pixels) / float(total_pixels)
    score = max(0.0, min(1.0, 1.0 - ((1.0 - finite_fraction) * 0.7) - min(removed_fraction * 20.0, 0.3)))
    flags: list[str] = []
    if (1.0 - finite_fraction) >= HIGH_MISSING_FRACTION:
        flags.append("high_missing_fraction")
    if removed_speckle_pixels > 0:
        flags.append("speckle_filtered")
    if not np.any(~np.isnan(processed_reflectivity) & (processed_reflectivity >= MIN_DBZ_FOR_OBJECTS)):
        flags.append("no_object_scale_echo")
    return ScanQuality(
        score=round(score, 3),
        finite_fraction=round(finite_fraction, 3),
        removed_speckle_pixels=removed_speckle_pixels,
        removed_speckle_fraction=round(removed_fraction, 5),
        flags=flags,
    )


def _blank_at_mask(field: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return a copy of `field` with `mask` positions set to NaN."""
    updated = np.array(field, dtype=float, copy=True)
    updated[mask] = np.nan
    return updated


def preprocess_reflectivity_data(reflectivity_data: SweepData) -> tuple[SweepData, ScanQuality]:
    processed_reflectivity, removed_speckle_pixels = _remove_weak_speckle(reflectivity_data.reflectivity)
    quality = assess_scan_quality(
        original_reflectivity=reflectivity_data.reflectivity,
        processed_reflectivity=processed_reflectivity,
        removed_speckle_pixels=removed_speckle_pixels,
    )

    updates: dict[str, np.ndarray] = {"reflectivity": processed_reflectivity}
    removed_mask = np.isnan(processed_reflectivity) & ~np.isnan(reflectivity_data.reflectivity)
    if np.any(removed_mask):
        for field_name in CO_REGISTERED_FIELD_NAMES:
            value = getattr(reflectivity_data, field_name)
            if value is not None:
                updates[field_name] = _blank_at_mask(value, removed_mask)

    return replace(reflectivity_data, **updates), quality


def preprocess_sweep(sweep: SweepData, rotation_signatures, velocity=None):
    """Quality control (classify only), then speckle removal, then scan
    quality assessment.

    Amendment, 2026-08-26 ("quality control flags, it does not delete"):
    `apply_quality_control` no longer removes any echo, so its position ahead
    of speckle removal no longer protects anything from being deleted before
    QC can save it -- `qc_sweep.reflectivity` is byte-for-byte the input
    reflectivity. Speckle removal is therefore the only step in this function
    that can ever modify reflectivity, and it always sees the full,
    unfiltered field regardless of ordering. QC still has to run first only
    because `gate_classification` must be attached before the co-registered
    speckle blanking in `_remove_weak_speckle`/the update loop below can
    blank it consistently alongside the other fields.
    """
    from src.qc.apply import apply_quality_control

    original_reflectivity = sweep.reflectivity
    qc_sweep, advisory, qc_report = apply_quality_control(
        sweep, rotation_signatures, velocity=velocity
    )

    despeckled, removed_speckle_pixels = _remove_weak_speckle(qc_sweep.reflectivity)

    quality = assess_scan_quality(
        original_reflectivity=original_reflectivity,
        processed_reflectivity=despeckled,
        removed_speckle_pixels=removed_speckle_pixels,
    )
    quality.class_fractions = qc_report.class_fractions
    quality.advisory_fraction = qc_report.advisory_fraction
    quality.mean_confidence = qc_report.mean_confidence
    quality.degraded_modes = qc_report.degraded_modes

    return replace(qc_sweep, reflectivity=despeckled), quality, advisory
