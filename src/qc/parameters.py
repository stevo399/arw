"""Membership function parameters, with mandatory provenance.

Provenance values:
  "park2009"           value taken from Park, Ryzhkov, Zrnic and Kim,
                       Weather and Forecasting 2009
  "arw-tuned-initial"  derived from the class discriminator table in the design
                       spec, not yet validated against live cases
  "arw-tuned"          set by ARW against a named validation case, recorded in
                       the reason field
"""

from dataclasses import dataclass


class GateClass:
    PRECIPITATION = "precipitation"
    GROUND_CLUTTER = "ground_clutter"
    BIOLOGICAL = "biological"
    HAIL = "hail"
    DEBRIS = "debris"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class MembershipParameter:
    x1: float
    x2: float
    x3: float
    x4: float
    weight: float
    provenance: str
    reason: str = ""


_INITIAL = "arw-tuned-initial"
_SPEC = "derived from design spec section 8 class table"

CLASS_PARAMETERS: dict[str, dict[str, MembershipParameter]] = {
    GateClass.PRECIPITATION: {
        "rhohv": MembershipParameter(0.93, 0.96, 1.0, 1.0, 1.0, _INITIAL, _SPEC),
        "zdr": MembershipParameter(-1.0, 0.0, 4.0, 5.5, 0.8, _INITIAL, _SPEC),
        "texture_z": MembershipParameter(0.0, 0.0, 3.0, 6.0, 1.0, _INITIAL, _SPEC),
    },
    GateClass.GROUND_CLUTTER: {
        "rhohv": MembershipParameter(0.0, 0.0, 0.85, 0.92, 1.0, _INITIAL, _SPEC),
        "texture_z": MembershipParameter(4.0, 8.0, 100.0, 100.0, 1.0, _INITIAL, _SPEC),
        "abs_velocity": MembershipParameter(0.0, 0.0, 1.0, 3.0, 1.2, _INITIAL, _SPEC),
        "beam_height_km": MembershipParameter(0.0, 0.0, 1.0, 2.5, 0.8, _INITIAL, _SPEC),
    },
    GateClass.BIOLOGICAL: {
        "rhohv": MembershipParameter(0.0, 0.0, 0.85, 0.92, 1.0, _INITIAL, _SPEC),
        "zdr": MembershipParameter(3.0, 4.5, 8.0, 10.0, 1.2, _INITIAL, _SPEC),
        "reflectivity": MembershipParameter(0.0, 0.0, 25.0, 32.0, 1.0, _INITIAL, _SPEC),
        "texture_z": MembershipParameter(3.0, 6.0, 100.0, 100.0, 0.8, _INITIAL, _SPEC),
    },
    GateClass.HAIL: {
        "reflectivity": MembershipParameter(48.0, 55.0, 80.0, 80.0, 1.2, _INITIAL, _SPEC),
        "zdr": MembershipParameter(-1.5, -0.5, 1.0, 2.0, 1.0, _INITIAL, _SPEC),
        "rhohv": MembershipParameter(0.82, 0.87, 0.95, 0.97, 1.0, _INITIAL, _SPEC),
    },
    GateClass.DEBRIS: {
        "reflectivity": MembershipParameter(38.0, 45.0, 80.0, 80.0, 1.0, _INITIAL, _SPEC),
        "zdr": MembershipParameter(-4.0, -3.0, 0.0, 1.0, 1.2, _INITIAL, _SPEC),
        "rhohv": MembershipParameter(0.0, 0.0, 0.80, 0.87, 1.2, _INITIAL, _SPEC),
        "texture_z": MembershipParameter(3.0, 6.0, 100.0, 100.0, 0.6, _INITIAL, _SPEC),
    },
}
