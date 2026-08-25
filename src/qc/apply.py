"""Derive a quality-controlled reflectivity field from a classified sweep."""

from dataclasses import replace

import numpy as np

from src.qc.classifier import CLASS_CODES, CODE_TO_CLASS, classify_gates
from src.qc.parameters import GateClass
from src.qc.protection import protected_mask
from src.qc.report import QualityReport, RejectedEcho

DEGRADED_NO_DUAL_POL = "no_dual_pol"
DEGRADED_NO_VELOCITY = "no_velocity"
DEGRADED_LOW_CONFIDENCE = "low_classification_confidence"
DEGRADED_BEYOND_VELOCITY_RANGE = "beyond_velocity_range"

# Doppler cuts are range-folded beyond roughly 300 km while the surveillance
# cut reaches 460 km. Gates in that outer ring have no velocity, so protection
# rule 2 cannot apply to them. Rule 1 still does.
MAX_VELOCITY_RANGE_M = 300_000.0

# ARW-tuned. Below this mean confidence the classifier is not separating
# classes usefully and the speech layer should say so.
LOW_CONFIDENCE_THRESHOLD = 0.35

# Only ground clutter and biological scatterers are non-meteorological. Hail
# and debris are the hazards this application exists to report and must never
# be rejected here.
NON_METEOROLOGICAL_CLASSES = (GateClass.GROUND_CLUTTER, GateClass.BIOLOGICAL)


def apply_quality_control(sweep, rotation_signatures, velocity=None):
    """Classify a sweep, remove non-meteorological echo, and report what happened.

    Returns (filtered_sweep, rejected_echo, quality_report). The filtered sweep
    carries gate_classification; the rejected echo retains every removed gate
    with its reason.
    """
    classification = classify_gates(sweep, velocity=velocity)
    classes = classification.classes

    reject_codes = [CLASS_CODES[name] for name in NON_METEOROLOGICAL_CLASSES]
    candidate = np.isin(classes, reject_codes)

    protected = protected_mask(sweep, rotation_signatures)
    rejected = candidate & ~protected

    reasons = np.full(classes.shape, "", dtype=object)
    for code in reject_codes:
        reasons[rejected & (classes == code)] = CODE_TO_CLASS[code]

    filtered_reflectivity = np.array(sweep.reflectivity, dtype=float, copy=True)
    filtered_reflectivity[rejected] = np.nan

    total_gates = float(classes.size)
    class_fractions = {
        name: float(np.count_nonzero(classes == code)) / total_gates
        for code, name in CODE_TO_CLASS.items()
    }

    finite_confidence = classification.confidence[np.isfinite(classification.confidence)]
    mean_confidence = float(finite_confidence.mean()) if finite_confidence.size else 0.0

    degraded_modes: list[str] = []
    if sweep.rhohv is None or sweep.zdr is None:
        degraded_modes.append(DEGRADED_NO_DUAL_POL)
    if velocity is None:
        degraded_modes.append(DEGRADED_NO_VELOCITY)
    if mean_confidence < LOW_CONFIDENCE_THRESHOLD:
        degraded_modes.append(DEGRADED_LOW_CONFIDENCE)
    if float(np.max(sweep.ranges_m)) > MAX_VELOCITY_RANGE_M:
        degraded_modes.append(DEGRADED_BEYOND_VELOCITY_RANGE)

    report = QualityReport(
        class_fractions=class_fractions,
        rejected_fraction=float(np.count_nonzero(rejected)) / total_gates,
        mean_confidence=mean_confidence,
        degraded_modes=degraded_modes,
    )

    filtered_sweep = replace(
        sweep, reflectivity=filtered_reflectivity, gate_classification=classes
    )
    return filtered_sweep, RejectedEcho(mask=rejected, reasons=reasons), report
