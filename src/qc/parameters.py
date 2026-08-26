"""Membership function parameters, with mandatory provenance.

Provenance values:
  "park2009"           value taken from Park, Ryzhkov, Zrnic and Kim,
                       Weather and Forecasting 2009
  "arw-tuned-initial"  derived from the class discriminator table in the design
                       spec, not yet validated against live cases
  "arw-tuned"          set by ARW against a named validation case, recorded in
                       the reason field

Task 19a (2026-08-25) attempted to source published values for all 18
parameters below from:

  Park, H. S., A. V. Ryzhkov, D. S. Zrnic and K. Kim, 2009: The Hydrometeor
  Classification Algorithm for the Polarimetric WSR-88D: Description and
  Application to an MCS. Wea. Forecasting, 24, 730-748.

The full text (via
https://training.weather.gov/wdtd/courses/rac/principles/qpe/story_content/external_files/Park_et_al_2009.pdf)
was read directly, including its Table 1 (membership function x1-x4
parameters for 10 classes: GC/AP, BS, DS, WS, CR, GR, BD, RA, HR, RH) and
Table 2 (variable weights). 6 of 18 parameters were updated to "park2009"
below; the other 12 were deliberately left "arw-tuned-initial" because no
value could be read that maps onto our variable without an unstated
assumption. Reasons, per parameter, are recorded inline and in the
provenance `reason` field. Full detail: see
.superpowers/sdd/2026-08-23-radar-data-layer-correctness/task-19a-report.md

Class-name mapping used (Park's 10 classes minus the 5 requiring
melting-layer height, which this project already excludes by design):
  GC/AP (ground clutter / anomalous propagation) -> ground_clutter
  BS    (biological scatterers)                  -> biological
  RA/HR (moderate/heavy rain)                     -> precipitation, PARTIALLY
        (Park has no single generic "rain" class; RA and HR differ in rhv
        and both have Z-dependent ZDR breakpoints, so only the one
        parameter that is IDENTICAL across every non-clutter/non-biological
        column of Table 1 -- SD(Z) -- would be a non-arbitrary pick for
        precipitation, and even that is blocked here: see the texture_z
        note below.)
  RH    (rain/hail mixture)                       -> hail, PARTIALLY
        (Park's HCA has no pure-hail class at S-band; RH is the closest
        correspondent and is used only for the two RH parameters that are
        constant, not Z-dependent.)
  (none)                                          -> debris
        (Park et al. 2009 defines no debris/TDS class at all -- tornado
        debris signature classification was added to the WSR-88D HCA in
        later work, not in this 2009 paper. Every debris parameter here is
        left arw-tuned-initial for that reason alone.)

Texture_z caveat (applies to every texture_z parameter, sourced or not):
Park's SD(Z) is a 1-D statistic computed ALONG THE RADIAL ONLY -- Z is
run-averaged over a 1-km / 4-gate window, subtracted from the raw profile,
and the RMS of that residual taken (paper, section 2, p. 731). ARW's
`local_standard_deviation` (src/qc/texture.py) is a 2-D 3x3-gate square
window standard deviation (range AND azimuth). These are different
statistics with no established magnitude correspondence, so no texture_z
value below is sourced from Park 2009 even where Park's own table would
otherwise give a clean, non-Z-dependent number (e.g., ground_clutter and
biological both have such numbers in Table 1) -- importing Park's
breakpoints against ARW's differently defined texture statistic would be
exactly the unjustified assumption the task instructions warn against.

Smoothing caveat (applies to the reflectivity/rhohv/zdr park2009 values
below): Park additionally smooths Z (1-km running average), and ZDR/rhv
(2-km running average) along the radial before scoring (section 2, p. 731).
ARW's classifier scores raw, unsmoothed gate values. This is disclosed as a
known limitation in the task report rather than treated as a blocker, on
the judgment that it is the same physical quantity in the same units and
smoothing mainly affects gate-to-gate noise variance rather than the
class-boundary values themselves -- unlike the texture case above, where
the window shape/dimensionality change makes it a fundamentally different
derived statistic.
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
_PARK = "park2009"
_CITE = (
    "Park et al. 2009, Wea. Forecasting 24, 730-748, Table 1, "
    "P(rhv)/P[Z(dBZ)]/P[ZDR(dB)] rows"
)

CLASS_PARAMETERS: dict[str, dict[str, MembershipParameter]] = {
    GateClass.PRECIPITATION: {
        # Park has no single generic "rain" class (RA and HR differ in rhv:
        # (0.95,0.97,1.00,1.01) vs (0.92,0.95,1.00,1.01) in Table 1) and
        # picking one over the other for a generic precipitation class would
        # be an unstated assumption. Left as-is.
        "rhohv": MembershipParameter(0.93, 0.96, 1.0, 1.0, 1.0, _INITIAL, _SPEC),
        # Park's ZDR breakpoints for every rain/snow subclass are partly
        # Z-dependent (functions f1-f3 of Table 1), not a constant trapezoid
        # ARW can copy without assuming a reflectivity value. Left as-is.
        "zdr": MembershipParameter(-1.0, 0.0, 4.0, 5.5, 0.8, _INITIAL, _SPEC),
        # Park's Table 1 SD(Z) row is (0, 0.5, 3, 6) uniformly across every
        # non-clutter, non-biological column (DS/WS/CR/GR/BD/RA/HR/RH), so no
        # rain-subclass choice is needed here -- but see the texture_z
        # caveat in the module docstring: Park's SD(Z) is a 1-D radial
        # statistic, ARW's texture_z is a 2-D windowed statistic. Different
        # derived quantities with no established magnitude correspondence,
        # so left as-is rather than imported.
        "texture_z": MembershipParameter(0.0, 0.0, 3.0, 6.0, 1.0, _INITIAL, _SPEC),
    },
    GateClass.GROUND_CLUTTER: {
        "rhohv": MembershipParameter(
            0.5, 0.6, 0.9, 0.95, 1.0, _PARK,
            f"{_CITE}, column GC/AP: x1..x4 = (0.5, 0.6, 0.9, 0.95)",
        ),
        # See texture_z caveat in module docstring (1-D radial RMS-of-
        # residual in Park vs 2-D 3x3-gate windowed std-dev in ARW) -- Park's
        # GC/AP SD(Z) value (2, 4, 10, 15) exists but is not imported.
        "texture_z": MembershipParameter(4.0, 8.0, 100.0, 100.0, 1.0, _INITIAL, _SPEC),
        # Park 2009 has no velocity-based trapezoid membership function at
        # all; V is used only as a post-classification hard threshold
        # (Table 3: "V > 1 m/s suppresses GC/AP"), a different mechanism
        # (binary override, not a 4-point trapezoid) that cannot be mapped
        # into x1..x4 without inventing a shape. Left as-is.
        "abs_velocity": MembershipParameter(0.0, 0.0, 1.0, 3.0, 1.2, _INITIAL, _SPEC),
        # beam_height_km is not one of Park's 6 input variables at all.
        "beam_height_km": MembershipParameter(0.0, 0.0, 1.0, 2.5, 0.8, _INITIAL, _SPEC),
    },
    GateClass.BIOLOGICAL: {
        "rhohv": MembershipParameter(
            0.3, 0.5, 0.8, 0.83, 1.0, _PARK,
            f"{_CITE}, column BS: x1..x4 = (0.3, 0.5, 0.8, 0.83)",
        ),
        "zdr": MembershipParameter(
            0.0, 2.0, 10.0, 12.0, 1.2, _PARK,
            f"{_CITE}, column BS: x1..x4 = (0.0, 2.0, 10.0, 12.0)",
        ),
        "reflectivity": MembershipParameter(
            5.0, 10.0, 20.0, 30.0, 1.0, _PARK,
            f"{_CITE}, column BS: x1..x4 = (5.0, 10.0, 20.0, 30.0)",
        ),
        # See texture_z caveat in module docstring -- Park's BS SD(Z) value
        # (1, 2, 4, 7) exists but is not imported (1-D vs 2-D statistic).
        "texture_z": MembershipParameter(3.0, 6.0, 100.0, 100.0, 0.8, _INITIAL, _SPEC),
    },
    GateClass.HAIL: {
        # Park 2009 has no pure-hail class at S-band; RH (rain/hail mixture)
        # is the closest correspondent and is used only where its
        # breakpoints are constant (not Z-dependent).
        "reflectivity": MembershipParameter(
            45.0, 50.0, 75.0, 80.0, 1.2, _PARK,
            f"{_CITE}, column RH (rain/hail mixture, closest Park class to "
            "'hail' -- see module docstring): x1..x4 = (45.0, 50.0, 75.0, 80.0)",
        ),
        # Park's RH ZDR breakpoints are only partly constant: x1=-0.3, x2=0.0
        # are fixed, but x3=f1(Z), x4=f1(Z)+0.5 depend on reflectivity
        # (Table 1, Eq. 4). Importing x1/x2 alone and keeping ARW's x3/x4
        # would misrepresent a mixed sourced/unsourced trapezoid as fully
        # sourced. Left as-is entirely.
        "zdr": MembershipParameter(-1.5, -0.5, 1.0, 2.0, 1.0, _INITIAL, _SPEC),
        "rhohv": MembershipParameter(
            0.85, 0.90, 1.00, 1.01, 1.0, _PARK,
            f"{_CITE}, column RH: x1..x4 = (0.85, 0.90, 1.00, 1.01)",
        ),
    },
    GateClass.DEBRIS: {
        # Park et al. 2009 defines no debris/TDS class -- see module
        # docstring. Nothing here can be sourced from this paper.
        "reflectivity": MembershipParameter(38.0, 45.0, 80.0, 80.0, 1.0, _INITIAL, _SPEC),
        "zdr": MembershipParameter(-4.0, -3.0, 0.0, 1.0, 1.2, _INITIAL, _SPEC),
        "rhohv": MembershipParameter(0.0, 0.0, 0.80, 0.87, 1.2, _INITIAL, _SPEC),
        "texture_z": MembershipParameter(3.0, 6.0, 100.0, 100.0, 0.6, _INITIAL, _SPEC),
    },
}
