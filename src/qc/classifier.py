"""Fuzzy-logic gate classification.

Structured after the operational Hydrometeor Classification Algorithm: each
class scores every gate as a weighted mean of trapezoidal memberships across
the available variables, and the highest-scoring class wins.

Deliberately not a threshold cascade. Low correlation coefficient is ambiguous
between ground clutter, biological scatterers, hail and tornado debris, so no
single variable may decide the outcome.

Classes requiring melting-layer height (wet snow, dry snow, ice crystals,
graupel, big drops) are absent by design — see the design spec, section 8.
"""

from dataclasses import dataclass

import numpy as np

from src.geometry import beam_height_m
from src.qc.membership import trapezoid
from src.qc.parameters import CLASS_PARAMETERS, GateClass
from src.qc.texture import local_standard_deviation

CLASS_CODES: dict[str, int] = {
    GateClass.UNKNOWN: 0,
    GateClass.PRECIPITATION: 1,
    GateClass.GROUND_CLUTTER: 2,
    GateClass.BIOLOGICAL: 3,
    GateClass.HAIL: 4,
    GateClass.DEBRIS: 5,
}
CODE_TO_CLASS: dict[int, str] = {code: name for name, code in CLASS_CODES.items()}

MIN_CONFIDENCE_TO_CLASSIFY = 0.2


def _build_variables(sweep, velocity: np.ndarray | None) -> dict[str, np.ndarray]:
    """Assemble the discriminator fields available for this sweep.

    Variables absent from the volume are simply omitted. They are never
    substituted with zeros, which would make a pre-2013 scan look like clutter.
    """
    reflectivity = np.asarray(sweep.reflectivity, dtype=float)
    variables: dict[str, np.ndarray] = {
        "reflectivity": reflectivity,
        "texture_z": local_standard_deviation(reflectivity),
    }

    if sweep.rhohv is not None:
        variables["rhohv"] = np.asarray(sweep.rhohv, dtype=float)
    if sweep.zdr is not None:
        variables["zdr"] = np.asarray(sweep.zdr, dtype=float)
    if velocity is not None:
        variables["abs_velocity"] = np.abs(np.asarray(velocity, dtype=float))

    heights_km = beam_height_m(sweep.ranges_m, sweep.elevation_angle, sweep.radar_alt_m) / 1000.0
    variables["beam_height_km"] = np.broadcast_to(
        heights_km[None, :], reflectivity.shape
    ).copy()

    return variables


def _score_class(class_name: str, variables: dict[str, np.ndarray], shape) -> np.ndarray:
    """Weighted mean membership across the variables this volume provides."""
    total = np.zeros(shape, dtype=float)
    weight_sum = 0.0
    for variable_name, parameter in CLASS_PARAMETERS[class_name].items():
        if variable_name not in variables:
            continue
        membership = trapezoid(
            variables[variable_name], parameter.x1, parameter.x2, parameter.x3, parameter.x4
        )
        total += np.nan_to_num(membership, nan=0.0) * parameter.weight
        weight_sum += parameter.weight
    if weight_sum == 0.0:
        return np.zeros(shape, dtype=float)
    return total / weight_sum


@dataclass
class ClassificationResult:
    classes: np.ndarray
    confidence: np.ndarray
    class_names: list[str]


def classify_gates(sweep, velocity: np.ndarray | None = None) -> ClassificationResult:
    """Classify every gate in a sweep."""
    reflectivity = np.asarray(sweep.reflectivity, dtype=float)
    shape = reflectivity.shape
    variables = _build_variables(sweep, velocity)

    class_names = list(CLASS_PARAMETERS)
    scores = np.stack([_score_class(name, variables, shape) for name in class_names])

    best_index = np.argmax(scores, axis=0)
    confidence = np.max(scores, axis=0)

    classes = np.full(shape, CLASS_CODES[GateClass.UNKNOWN], dtype=np.int8)
    for index, name in enumerate(class_names):
        classes[(best_index == index) & (confidence >= MIN_CONFIDENCE_TO_CLASSIFY)] = CLASS_CODES[name]

    invalid = ~np.isfinite(reflectivity)
    classes[invalid] = CLASS_CODES[GateClass.UNKNOWN]
    confidence = np.where(invalid, np.nan, confidence)

    return ClassificationResult(
        classes=classes, confidence=confidence, class_names=class_names
    )
