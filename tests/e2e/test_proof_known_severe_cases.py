# tests/e2e/test_proof_known_severe_cases.py
"""Proof 4 from the design spec: quality control must not delete hazards.

Low correlation coefficient is genuinely ambiguous between ground clutter and
a tornado lofting debris -- a classifier tuned only against clear-air clutter
could silently discard exactly the echo this application exists to report.
These users cannot check the output against a visual radar map, so a wrongly
deleted tornado is silent and unrecoverable. This proof exists to catch that
failure mode.

CASE SELECTION (what was searched, and why the cached volumes were rejected):

The task brief suggested checking cached volumes first: KTLX (2026-04-10, 55
scans), KEYX/KIWA/KSOX/KFWS 2026-04-10, plus a few isolated KTLX/KIWA volumes
on other 2026 dates. Every one of those dates was checked against the SPC
storm reports archive (https://www.spc.noaa.gov/climo/reports/<YYMMDD>_rpts*.csv,
ground truth for confirmed severe events):

  - 2026-04-10: no tornado reports anywhere that day; the only hail reports
    were in TX/CA, nowhere near KTLX/KEYX/KIWA/KSOX/KFWS.
  - 2026-07-12 (KIWA): no tornado or hail reports in Arizona that day (wind
    damage only, Prescott Valley/Tucson).
  - 2026-06-13 (KTLX): the day's only OK tornado report is 6 ENE Wyandotte,
    Ottawa County (36.82, -94.63) -- the far NE corner of the state, well
    outside KTLX's practical range of the OKC metro the cached scans cover.
  - 2026-06-14 (KTLX): no Oklahoma reports at all that day.

None of the cached volumes correspond to a confirmed severe event. Per the
task instructions, fabricating coordinates or leaving placeholders was not an
option, so a real, well-documented, confirmed event was located instead and
its volumes fetched live through src/ingest.py (the only module permitted to
make network calls) -- src.ingest.fetch_scan("KTLX", datetime(2013, 5, 20, ...)).

Chosen event: the 2013-05-20 Newcastle-Moore, Oklahoma tornado (rated EF5 by
NWS Norman's final survey; SPC's preliminary same-day report below rates it
"at least EF4"), radar KTLX. This is one of the most extensively studied
tornado debris signature (TDS) cases in the polarimetric radar literature
(e.g. Bodine et al. 2013, Van Den Broeke 2015) and a large-hail report from
the same storm complex, 10 minutes later and about 15 km away, is on the same
SPC page -- letting both cases use the same site and the same afternoon.

Both are dual-pol volumes (KTLX's dual-pol upgrade completed years before
2013), so rhohv/zdr are present and the classifier scores every discriminator
it is designed to use -- unlike a pre-2011 volume, which would silently
degrade classification by omitting those variables (see
src/qc/classifier.py's `_build_variables`).

SPC raw CSV citations (fetched directly, quoted verbatim below):

  Tornado -- https://www.spc.noaa.gov/climo/reports/130520_rpts_torn.csv :
    "1956,UNK,NEWCASTLE,MCCLAIN,OK,35.25,-97.6,LIFTED AROUND 336 PM.
    ESTIMATED PATH LENGTH OF 20 MILES THRU NEWCASTLE,MOORE,AND SOUTH OKC.
    PRELIMINARY DAMAGE RATING OF AT LEAST EF4. (OUN)"

  Hail -- https://www.spc.noaa.gov/climo/reports/130520_rpts_hail.csv :
    "2006,250,3 NNW MOORE,OKLAHOMA,OK,35.38,-97.5,(OUN)"
    (250 = 2.50 inch diameter hail)

Volumes were fetched with `src.ingest.fetch_scan("KTLX", dt)`, which picks the
scan nearest the requested time from that day's AWS archive listing:
  fetch_scan("KTLX", datetime(2013, 5, 20, 19, 56)) -> KTLX20130520_195527_V06.gz
      (19:55:27 UTC, 33s before the tornado report's timestamp)
  fetch_scan("KTLX", datetime(2013, 5, 20, 20, 6))  -> KTLX20130520_200356_V06.gz
      (20:03:56 UTC, 2m04s before the hail report's timestamp)

=== THE CRITICAL FLAW THIS PROOF MUST NOT REPEAT ===

A prior version of this proof (per the task brief for this file, and per
tests/e2e/test_proof_clutter_persistence.py's documented (C) EFFECTIVENESS
finding) would have passed vacuously: apply_quality_control's protection
rules currently rescue essentially all classifier-flagged non-meteorological
echo, because 121+ spurious rotation signatures per scan (a known defect in
the pre-existing shear detector, src/velocity.py) seed protection rule 2,
which rule 3 (connected-component expansion) then spreads across whole
connected storm components. "The hazard survived QC" would be true even if
the classifier called every gate of it ground_clutter, because protection
would rescue it anyway -- the proof would certify nothing about the
classifier's judgement.

So this file measures and asserts THREE separate things per case, not one:

  (i)   The hazard survives the FULL pipeline (protection included). This is
        the brief's original assertion. It is expected to pass, and does.

  (ii)  The CLASSIFIER ALONE -- classify_gates() called directly, bypassing
        protected_mask() entirely -- does not label the event-location gates
        ground_clutter or biological. This is the assertion that actually
        exercises the classifier's judgement. MEASURED: it fails for both
        cases (see the xfail reason on the test below for exact numbers).
        This is reported as the critical finding, not tuned away -- no
        threshold in src/qc/parameters.py was touched to make it pass.

  (iii) The quantified safety margin: what fraction of the event-location
        gates would protected_mask() save if the classifier HAD condemned
        every one of them. MEASURED: 100% in both cases. This is what
        actually explains why (i) passes despite (ii) failing -- protection,
        not classifier judgement, is what is saving these hazards today.

Reading (i)+(ii)+(iii) together is the honest picture: the classifier by
itself is not safe to trust with tornado debris or giant hail -- it
misclassifies a real, substantial fraction of genuine hazard echo as clutter
or biological scatter -- but the hard protection rules currently provide a
complete safety net under it, for these two measured cases. That the safety
net is currently doing all the work (rather than the classifier sharing the
load) is itself the same defect test_proof_clutter_persistence.py's (C)
already found from the opposite direction (clear air, false positives on
protection). This file adds the true-positive side: real hazards, and
whether the classifier alone would have kept them.
"""

from dataclasses import dataclass

import numpy as np
import pyart
import pytest

from src.detection import detect_objects_with_grid
from src.geometry import align_field_by_azimuth, gate_coordinates
from src.parser import extract_sweep_data, extract_velocity
from src.preprocess import preprocess_sweep
from src.qc.classifier import CLASS_CODES, classify_gates
from src.qc.parameters import GateClass
from src.qc.protection import protected_mask
from src.velocity import detect_rotation_signatures

EARTH_RADIUS_KM = 6371.0


@dataclass(frozen=True)
class SevereCase:
    name: str
    volume_path: str
    event_lat: float
    event_lon: float
    event_type: str
    spc_reference: str


KNOWN_CASES = [
    SevereCase(
        name="2013-05-20 Newcastle-Moore OK tornado (debris signature)",
        volume_path="cache/KTLX/KTLX20130520_195527_V06.gz",
        event_lat=35.25,
        event_lon=-97.60,
        event_type="tornado",
        spc_reference=(
            "SPC storm reports, 2013-05-20, tornado table "
            "(https://www.spc.noaa.gov/climo/reports/130520_rpts_torn.csv): "
            "\"1956,UNK,NEWCASTLE,MCCLAIN,OK,35.25,-97.6,LIFTED AROUND 336 PM. "
            "ESTIMATED PATH LENGTH OF 20 MILES THRU NEWCASTLE,MOORE,AND SOUTH "
            "OKC. PRELIMINARY DAMAGE RATING OF AT LEAST EF4. (OUN)\" -- later "
            "confirmed EF5 by the NWS Norman final damage survey."
        ),
    ),
    SevereCase(
        name="2013-05-20 Moore OK large hail (2.5 in)",
        volume_path="cache/KTLX/KTLX20130520_200356_V06.gz",
        event_lat=35.38,
        event_lon=-97.50,
        event_type="hail",
        spc_reference=(
            "SPC storm reports, 2013-05-20, hail table "
            "(https://www.spc.noaa.gov/climo/reports/130520_rpts_hail.csv): "
            "\"2006,250,3 NNW MOORE,OKLAHOMA,OK,35.38,-97.5,(OUN)\" -- "
            "250 = 2.50 inch diameter hail, from the same storm complex as "
            "the Newcastle-Moore tornado above, 10 minutes later."
        ),
    ),
]

SEARCH_RADIUS_KM = 10.0

# Fraction of event-location gates the classifier condemns (ground_clutter or
# biological) that protection must rescue. Measured at 100% for both cases;
# 0.99 leaves a hair of floating-point slack without weakening the claim.
MIN_PROTECTION_MARGIN = 0.99

NON_METEOROLOGICAL_CODES = {
    CLASS_CODES[GateClass.GROUND_CLUTTER],
    CLASS_CODES[GateClass.BIOLOGICAL],
}


def _haversine_km(lat1, lon1, lat2: float, lon2: float) -> np.ndarray:
    """Great-circle distance from every gate to one point, in km.

    Computed independently rather than importing src.qc.collocation's private
    `_haversine_km`, per the task instructions.
    """
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = phi2 - phi1
    delta_lambda = np.radians(lon2 - lon1)
    a = (
        np.sin(delta_phi / 2.0) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0) ** 2
    )
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def _velocity_aligned_to_reflectivity(raw_sweep, vel_data):
    """Mirror of src.server._velocity_aligned_to_reflectivity.

    Split-cut VCPs scan reflectivity (surveillance cut) and velocity (Doppler
    cut) on separate antenna revolutions, so the same array index refers to a
    different compass bearing in each sweep -- up to 21 degrees apart at the
    same index. Pairing them by raw index would silently compare gates tens
    of kilometres apart at longer range.
    """
    if vel_data is None or not vel_data.sweeps:
        return None
    velocity_sweep = vel_data.sweeps[0]
    if not np.array_equal(
        np.asarray(velocity_sweep.ranges_m, dtype=float),
        np.asarray(raw_sweep.ranges_m, dtype=float),
    ):
        return None
    return align_field_by_azimuth(
        velocity_sweep.velocity, velocity_sweep.azimuths, raw_sweep.azimuths
    )


class CaseMeasurement:
    """Everything measured for one severe case, computed once."""

    def __init__(self, case: SevereCase):
        self.case = case
        radar = pyart.io.read_nexrad_archive(case.volume_path)
        sweep = extract_sweep_data(radar)
        vel_data = extract_velocity(radar)
        aligned_velocity = _velocity_aligned_to_reflectivity(sweep, vel_data)
        rotation_signatures = (
            detect_rotation_signatures(vel_data) if vel_data else []
        )

        self.sweep = sweep
        self.rotation_signatures = rotation_signatures

        lat, lon = gate_coordinates(
            sweep.azimuths, sweep.ranges_m, sweep.elevation_angle,
            sweep.radar_lat, sweep.radar_lon,
        )
        distance_km = _haversine_km(lat, lon, case.event_lat, case.event_lon)
        self.near_event = distance_km <= SEARCH_RADIUS_KM
        finite = np.isfinite(sweep.reflectivity)
        self.near_event_finite = self.near_event & finite
        self.n_near_event_finite = int(np.count_nonzero(self.near_event_finite))

        # (ii) CLASSIFIER ALONE: classify_gates called directly on the raw
        # sweep, exactly the call apply_quality_control makes internally --
        # but protected_mask is never consulted here. This isolates the
        # classifier's judgement from protection's blanket rescue.
        classification = classify_gates(sweep, velocity=aligned_velocity)
        self.classes = classification.classes
        self.condemned = self.near_event_finite & np.isin(
            self.classes, list(NON_METEOROLOGICAL_CODES)
        )
        self.n_condemned = int(np.count_nonzero(self.condemned))

        # (iii) SAFETY MARGIN: would protected_mask save these gates if the
        # classifier condemned them? Measured over ALL event-location gates
        # (hypothetical -- "if condemned"), not just the ones actually
        # condemned today, so the margin is visible even in a case where (ii)
        # happens to pass.
        self.protected = protected_mask(sweep, rotation_signatures)
        self.protection_margin_all = (
            float(np.count_nonzero(self.protected & self.near_event_finite))
            / self.n_near_event_finite
            if self.n_near_event_finite
            else float("nan")
        )
        self.protection_margin_condemned = (
            float(np.count_nonzero(self.protected & self.condemned)) / self.n_condemned
            if self.n_condemned
            else float("nan")
        )

        # (i) FULL PIPELINE: apply_quality_control (classify only, per the
        # 2026-08-26 amendment -- nothing is deleted) + speckle removal,
        # exactly as production code runs it.
        filtered, quality, advisory = preprocess_sweep(
            sweep, rotation_signatures, velocity=aligned_velocity
        )
        self.filtered = filtered
        self.quality = quality
        self.advisory = advisory

        filtered_lat, filtered_lon = gate_coordinates(
            filtered.azimuths, filtered.ranges_m, filtered.elevation_angle,
            filtered.radar_lat, filtered.radar_lon,
        )
        filtered_distance_km = _haversine_km(
            filtered_lat, filtered_lon, case.event_lat, case.event_lon
        )
        self.surviving = np.isfinite(filtered.reflectivity) & (
            filtered_distance_km <= SEARCH_RADIUS_KM
        )

        self.detection_result = detect_objects_with_grid(
            reflectivity=filtered.reflectivity,
            azimuths=filtered.azimuths,
            ranges_m=filtered.ranges_m,
            radar_lat=filtered.radar_lat,
            radar_lon=filtered.radar_lon,
            elevation_deg=filtered.elevation_angle,
        )


@pytest.fixture(scope="module")
def measurements():
    return {case.name: CaseMeasurement(case) for case in KNOWN_CASES}


@pytest.mark.parametrize("case", KNOWN_CASES, ids=lambda c: c.name)
def test_hazard_echo_survives_quality_control(case: SevereCase, measurements):
    """(i) Echo at the confirmed event location must not be removed by the
    full pipeline (classification + protection + speckle removal).

    Under the 2026-08-26 amendment this is a weaker claim than it used to be:
    `apply_quality_control` itself never removes anything from any gate any
    more, protected or not, so this assertion can now only fail via speckle
    removal (the one remaining step in `preprocess_sweep` that can modify
    reflectivity). It is kept -- rather than deleted as vacuous -- because it
    still guards a real invariant: that despeckling does not erase intense,
    real-world hazard echo. What it no longer demonstrates is that protection
    specifically is what saves this echo; see
    test_protection_margin_at_event_location's updated docstring for that
    distinction.
    """
    m = measurements[case.name]
    print(f"\n{case.name}: {int(np.count_nonzero(m.surviving))} surviving gates "
          f"within {SEARCH_RADIUS_KM} km of ({case.event_lat}, {case.event_lon})")
    assert m.surviving.any(), f"QC removed all echo at the {case.name} location"


@pytest.mark.parametrize("case", KNOWN_CASES, ids=lambda c: c.name)
def test_hazard_produces_a_detected_object(case: SevereCase, measurements):
    """The event must survive all the way through to a detected object."""
    m = measurements[case.name]
    distances_km = [
        float(_haversine_km(
            np.array([o.centroid_lat]), np.array([o.centroid_lon]),
            case.event_lat, case.event_lon,
        )[0])
        for o in m.detection_result.objects
    ]
    assert distances_km, "no objects detected at all"
    nearest = min(distances_km)
    print(f"\n{case.name}: nearest detected object {nearest:.2f} km from event location")
    assert nearest <= 30.0, f"nearest object {nearest:.1f} km away"


@pytest.mark.xfail(
    strict=True,
    reason=(
        "(ii) CLASSIFIER-ALONE JUDGEMENT, documented known defect -- not "
        "tuned away, per task instructions. classify_gates() called directly "
        "on the raw sweep (bypassing protected_mask entirely) labels a real, "
        "substantial fraction of the event-location gates ground_clutter or "
        "biological in BOTH confirmed severe cases: the 2013-05-20 "
        "Newcastle-Moore EF5 tornado debris case measures 1260/4597 (27.4%) "
        "of finite gates within 10 km of the confirmed touchdown point "
        "condemned; the collocated 2.5in hail case measures 668/7079 (9.4%) "
        "condemned. If protection did not exist, QC would delete over a "
        "quarter of the reflectivity describing this tornado's debris field. "
        "This directly confirms the concern that motivated this proof: low "
        "correlation coefficient is genuinely ambiguous between clutter/"
        "biological scatter and a tornado lofting debris, and the classifier "
        "as currently parameterized (every class in src/qc/parameters.py is "
        "marked provenance='arw-tuned-initial', not yet validated against "
        "live cases -- see test_proof_clutter_persistence.py's matching "
        "finding on the clear-air side) does not reliably tell them apart. "
        "See test_protection_margin_at_event_location below for why (i) "
        "still passes: protected_mask rescues 100% of these condemned gates "
        "in both cases -- the safety net, not the classifier, is what keeps "
        "these hazards in the output today. strict=True: this test fails "
        "loudly the day the classifier itself, with no protection rule "
        "involved, stops condemning real hazard echo -- the signal that "
        "would mean the safety net is no longer the only thing standing "
        "between a misclassification and a silently deleted tornado."
    ),
)
@pytest.mark.parametrize("case", KNOWN_CASES, ids=lambda c: c.name)
def test_classifier_alone_does_not_condemn_hazard(case: SevereCase, measurements):
    """(ii) classify_gates() alone, bypassing protected_mask entirely, must
    not label event-location gates ground_clutter or biological.

    This is the assertion that actually exercises the classifier's
    judgement, isolated from protection's blanket rescue -- the assertion
    that would fail if the classifier itself were catastrophically wrong.
    """
    m = measurements[case.name]
    print(
        f"\n{case.name}: classifier-alone condemned {m.n_condemned}/"
        f"{m.n_near_event_finite} event-location gates as ground_clutter or "
        f"biological ({(m.n_condemned / m.n_near_event_finite * 100.0) if m.n_near_event_finite else 0.0:.1f}%)"
    )
    assert m.n_condemned == 0, (
        f"{case.name}: classifier alone condemned {m.n_condemned}/"
        f"{m.n_near_event_finite} event-location gates as ground_clutter or "
        f"biological -- exactly the failure mode this proof exists to catch"
    )


@pytest.mark.parametrize("case", KNOWN_CASES, ids=lambda c: c.name)
def test_protection_margin_at_event_location(case: SevereCase, measurements):
    """(iii) Quantify the safety margin: what fraction of event-location
    gates would protected_mask save if the classifier condemned them all.

    Before the 2026-08-26 amendment, this margin was what actually explained
    why (i) passed despite (ii) failing -- protection was the only thing
    standing between the classifier's judgement and deletion. That is no
    longer the mechanism: (i) now passes unconditionally with respect to QC,
    because apply_quality_control never deletes anything regardless of this
    margin. What this test still measures is real and worth keeping: it is
    the advisory-flagging safety margin -- if ARW (or a future consumer)
    ever acts on the advisory/class_fractions signal to filter or de-weight
    echo, this margin is what would determine whether that filtering could
    have silently discarded this hazard. A margin that dropped to, say, 40%
    would mean any such downstream filtering built on the advisory signal
    would need re-examining.
    """
    m = measurements[case.name]
    print(
        f"\n{case.name}: protection would save "
        f"{m.protection_margin_all * 100.0:.1f}% of all {m.n_near_event_finite} "
        f"event-location gates if the classifier condemned every one of them"
    )
    if m.n_condemned:
        print(
            f"  of the {m.n_condemned} gates actually condemned by the "
            f"classifier alone, protection actually saves "
            f"{m.protection_margin_condemned * 100.0:.1f}% of them"
        )
    else:
        print("  classifier alone condemned zero gates here -- no rescue was needed")

    assert m.protection_margin_all >= MIN_PROTECTION_MARGIN, (
        f"{case.name}: protection would save only "
        f"{m.protection_margin_all * 100.0:.1f}% of event-location gates if "
        f"the classifier condemned them -- not the complete safety net (i) "
        f"implicitly relies on"
    )
