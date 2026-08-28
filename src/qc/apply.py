"""Classify every gate in a sweep and report what quality control observed.

Amendment, 2026-08-26 ("quality control flags, it does not delete"): this
module used to derive a filtered reflectivity field with non-meteorological
gates set to NaN. It no longer does. Proof 4 (the Newcastle-Moore EF5 replay)
measured that 27.4% of real tornado-debris gates were condemned by the
classifier and survived only because protection rescued them -- a coupling
where a classifier defect and an over-sensitive shear detector cancelled each
other out, silently. Deleting nothing removes that failure mode outright:
the classifier's accuracy becomes a quality problem, not a life-safety one.

`apply_quality_control` still computes which gates WOULD have been flagged --
that information is valuable for the speech layer, GeoJSON consumers and the
clutter-persistence proof -- but it is advisory only, carried in an
`EchoAdvisory`. Nothing is ever removed from the returned reflectivity field.
"""

from dataclasses import replace

import numpy as np

from src.qc.classifier import CLASS_CODES, CODE_TO_CLASS, classify_gates
from src.qc.parameters import GateClass
from src.qc.protection import protected_mask
from src.qc.report import EchoAdvisory, QualityReport

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
# be flagged for removal here -- and, since the 2026-08-26 amendment, no gate
# of any class is ever actually removed by this module.
NON_METEOROLOGICAL_CLASSES = (GateClass.GROUND_CLUTTER, GateClass.BIOLOGICAL)


def apply_quality_control(sweep, rotation_signatures, velocity=None):
    """Classify a sweep and report what quality control observed.

    Returns (classified_sweep, advisory, quality_report). `classified_sweep`
    carries `gate_classification` and reflectivity UNCHANGED from the input --
    quality control does not remove or alter any echo. `advisory` records
    which gates WOULD have been flagged as non-meteorological and why; it is
    informational only, never applied to the returned reflectivity.
    """
    classification = classify_gates(sweep, velocity=velocity)
    classes = classification.classes

    flag_codes = [CLASS_CODES[name] for name in NON_METEOROLOGICAL_CLASSES]
    candidate = np.isin(classes, flag_codes)

    protected = protected_mask(sweep, rotation_signatures)
    # Advisory only: gates a filter WOULD exclude if one were applied. Never
    # used to modify the reflectivity field returned below.
    flagged = candidate & ~protected

    reasons = np.full(classes.shape, "", dtype=object)
    for code in flag_codes:
        reasons[flagged & (classes == code)] = CODE_TO_CLASS[code]

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
        advisory_fraction=float(np.count_nonzero(flagged)) / total_gates,
        mean_confidence=mean_confidence,
        degraded_modes=degraded_modes,
    )

    # Reflectivity is passed through unchanged -- only gate_classification is
    # attached. A copy is made only so downstream mutation of the returned
    # sweep's reflectivity array can never alias the caller's original.
    classified_sweep = replace(
        sweep,
        reflectivity=np.array(sweep.reflectivity, dtype=float, copy=True),
        gate_classification=classes,
    )
    return classified_sweep, EchoAdvisory(mask=flagged, reasons=reasons), report
