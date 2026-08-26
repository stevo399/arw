"""Membership function parameters, with mandatory provenance.

Provenance values:
  "park2009"           value taken from Park, Ryzhkov, Zrnic and Kim,
                       Weather and Forecasting 2009
  "arw-tuned-initial"  derived from the class discriminator table in the design
                       spec, not yet validated against live cases
  "arw-tuned"          set by ARW against a named validation case, recorded in
                       the reason field

Task 19a (2026-08-25) attempted to source published values for all 18
active parameters below from:

  Park, H. S., A. V. Ryzhkov, D. S. Zrnic and K. Kim, 2009: The Hydrometeor
  Classification Algorithm for the Polarimetric WSR-88D: Description and
  Application to an MCS. Wea. Forecasting, 24, 730-748.

The full text (via
https://training.weather.gov/wdtd/courses/rac/principles/qpe/story_content/external_files/Park_et_al_2009.pdf)
was read directly, including its Table 1 (membership function x1-x4
parameters for 10 classes: GC/AP, BS, DS, WS, CR, GR, BD, RA, HR, RH) and
Table 2 (variable weights).

Task 19a fix round 1 (2026-08-26): the first pass ported 6 of the 18 active
parameters to their Park 2009 numbers. Measurement showed this made
classification WORSE on every available metric (velocity separation dropped
scan-by-scan, e.g. 2.70x->1.76x and 3.14x->2.09x; the velocity-omitted
polarimetric-only separation the whole task exists to fix moved from 1.05x
to 0.96x; the suite went from 315 passed/2 xfailed to 312 passed/3
failed/2 xfailed) -- because Park's breakpoints were fitted against Park's
variable definitions (a 1-D radial SD(Z) texture, velocity as a binary hard
threshold rather than a fuzzy term, Z-dependent ZDR breakpoints, and Park's
full 10-class set with melting-layer input), not ARW's. Splicing 6 of
Park's numbers into ARW's different structure adopted the numbers and left
the algorithm behind -- a hybrid that is not actually closer to Park 2009
conformance despite being better-cited. All 18 active parameters below have
therefore been reverted to their pre-19a "arw-tuned-initial" values.

The values that WERE read from Park 2009 are preserved, not deleted, in
`PARK2009_REFERENCE` below -- a non-active dict, not consumed by the
classifier -- along with what porting them properly would require. See
.superpowers/sdd/2026-08-23-radar-data-layer-correctness/task-19a-report.md
for the full sourcing rationale, the measurements above, and this reversion.
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


# ---------------------------------------------------------------------------
# NON-ACTIVE REFERENCE ONLY. Not imported or read by src/qc/classifier.py or
# anything else in the pipeline -- CLASS_PARAMETERS above is the only table
# the classifier consumes. This dict exists so the six published values Task
# 19a actually read from Park et al. 2009 are not lost, and so a future
# attempt at genuine Park-2009 conformance starts from here instead of
# re-doing the literature search.
#
# Source (read in full, Table 1 "Parameters of the membership functions for
# 10 classes" and Table 2 "Matrix of weights"):
#   Park, H. S., A. V. Ryzhkov, D. S. Zrnic and K. Kim, 2009: The
#   Hydrometeor Classification Algorithm for the Polarimetric WSR-88D:
#   Description and Application to an MCS. Wea. Forecasting, 24, 730-748.
#   https://training.weather.gov/wdtd/courses/rac/principles/qpe/story_content/external_files/Park_et_al_2009.pdf
#
# NOT IN USE. When 6 of these were substituted into CLASS_PARAMETERS above
# (commit 403962c), classification measurably got WORSE on every metric
# available, not better:
#   velocity separation, scan 170029:        2.70x -> 1.76x
#   velocity separation, scan 170914:        3.14x -> 2.09x
#   polarimetric-only separation (the target
#     metric test_polarimetric_texture_alone
#     _separate_biological_from_clutter measures,
#     velocity omitted):                     1.05x -> 0.96x
#   ground_clutter mean |velocity|:           ~2.5 m/s -> ~3.4 m/s
#   test suite:              315 passed/2 xfailed -> 312 passed/3 failed/2 xfailed
#
# Why: these breakpoints were fitted against Park's variable definitions,
# Park's aggregation, and Park's 10-class set -- not ARW's. Dropping isolated
# numbers into a different structure adopts the citation without adopting
# the algorithm they were fitted for. A proper port, not attempted here,
# would require at minimum:
#   1. A 1-D along-radial SD(Z) texture (Z run-averaged over a 1-km/4-gate
#      window, subtract, RMS the residual) to replace ARW's 2-D 3x3-gate
#      windowed local_standard_deviation (src/qc/texture.py) -- these are
#      different statistics with no established magnitude correspondence.
#   2. Velocity as a binary post-classification hard threshold ("V > 1 m/s
#      suppresses GC/AP", Park's Table 3), not a fuzzy trapezoid term inside
#      the weighted-mean aggregation -- Park's HCA does not include velocity
#      in its 6-variable fuzzy classifier at all.
#   3. Z-dependent ZDR (and LKdp) breakpoints for the rain/hail categories
#      (functions f1-f3 of reflectivity, Park's Eq. 4), not fixed constants.
#   4. Park's full 10-class set (GC/AP, BS, DS, WS, CR, GR, BD, RA, HR, RH)
#      plus melting-layer height as an input, which ARW's 5-class,
#      melting-layer-free design deliberately excludes (spec section 8) --
#      so a full port is a design decision beyond this task, not just a
#      parameter change.
#   5. A KDP/LKdp and SD(phi_DP) input, which ARW's classifier does not
#      currently accept at all (src/qc/classifier.py:_build_variables has
#      no such branch).
PARK2009_REFERENCE: dict[str, dict[str, MembershipParameter]] = {
    GateClass.GROUND_CLUTTER: {
        "rhohv": MembershipParameter(
            0.5, 0.6, 0.9, 0.95, 1.0, "park2009",
            "Park et al. 2009, Table 1, P(rhv), column GC/AP. NOT ACTIVE -- "
            "see PARK2009_REFERENCE docstring.",
        ),
    },
    GateClass.BIOLOGICAL: {
        "rhohv": MembershipParameter(
            0.3, 0.5, 0.8, 0.83, 1.0, "park2009",
            "Park et al. 2009, Table 1, P(rhv), column BS. NOT ACTIVE -- "
            "see PARK2009_REFERENCE docstring.",
        ),
        "zdr": MembershipParameter(
            0.0, 2.0, 10.0, 12.0, 1.2, "park2009",
            "Park et al. 2009, Table 1, P[ZDR(dB)], column BS. NOT ACTIVE -- "
            "see PARK2009_REFERENCE docstring.",
        ),
        "reflectivity": MembershipParameter(
            5.0, 10.0, 20.0, 30.0, 1.0, "park2009",
            "Park et al. 2009, Table 1, P[Z(dBZ)], column BS. NOT ACTIVE -- "
            "see PARK2009_REFERENCE docstring.",
        ),
    },
    GateClass.HAIL: {
        "reflectivity": MembershipParameter(
            45.0, 50.0, 75.0, 80.0, 1.2, "park2009",
            "Park et al. 2009, Table 1, P[Z(dBZ)], column RH (rain/hail "
            "mixture, closest Park class to 'hail' -- Park has no pure-hail "
            "class). NOT ACTIVE -- see PARK2009_REFERENCE docstring.",
        ),
        "rhohv": MembershipParameter(
            0.85, 0.90, 1.00, 1.01, 1.0, "park2009",
            "Park et al. 2009, Table 1, P(rhv), column RH. NOT ACTIVE -- "
            "see PARK2009_REFERENCE docstring.",
        ),
    },
}
