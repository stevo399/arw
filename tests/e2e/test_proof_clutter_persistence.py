# tests/e2e/test_proof_clutter_persistence.py
"""Proof 2 from the design spec: ground clutter sits still.

Buildings and terrain return echo from the same gates on every scan. A
classifier that works must flag substantially the same gate set each time, and
those gates must show near-zero radial velocity. This checks a property that is
true by physics, so it needs no human to eyeball an expected answer.

In addition to the brief's classification-stability proof, this module
measures (per Task 15's controller extension) how much of the classifier's
non-meteorological judgement actually reaches the rejected-echo output versus
being overridden by protection rule 2 (rotation collocation). These are
reported, not asserted, per the task instructions: the numbers are the
deliverable, not a threshold to hit.
"""

import numpy as np
import pyart
import pytest

from src.geometry import align_field_by_azimuth
from src.parser import extract_sweep_data, extract_velocity
from src.qc.apply import apply_quality_control
from src.qc.classifier import CLASS_CODES
from src.qc.parameters import GateClass
from src.qc.protection import protected_mask
from src.velocity import detect_rotation_signatures

# Consecutive KIWA scans on 2026-07-12. Measured at ~0.2% of valid gates above
# 20 dBZ, so what persists between them is terrain and structures rather than
# weather. Roughly 9 minutes apart.
CLEAR_AIR_SCANS = [
    "cache/KIWA/KIWA20260712_163410_V06",
    "cache/KIWA/KIWA20260712_164255_V06",
    "cache/KIWA/KIWA20260712_165142_V06",
    "cache/KIWA/KIWA20260712_170029_V06",
    "cache/KIWA/KIWA20260712_170914_V06",
    "cache/KIWA/KIWA20260712_171800_V06",
]

MIN_JACCARD_OVERLAP = 0.6
MAX_MEAN_ABS_VELOCITY_MS = 3.0

NON_METEOROLOGICAL_CODES = {
    CLASS_CODES[GateClass.GROUND_CLUTTER],
    CLASS_CODES[GateClass.BIOLOGICAL],
}


def _velocity_aligned_to_reflectivity(raw_sweep, vel_data):
    """Mirror of src.server._velocity_aligned_to_reflectivity.

    Split-cut VCPs scan reflectivity (surveillance cut) and velocity (Doppler
    cut) on separate antenna revolutions, so the same array index refers to a
    different compass bearing in each sweep. Both arrays happen to share a
    shape, so pairing them by raw index never raises -- it silently compares
    gates that can be tens of kilometres apart at longer range.
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


class ScanMeasurement:
    """All quantities measured for a single cached scan, computed once."""

    def __init__(self, path: str):
        self.path = path
        radar = pyart.io.read_nexrad_archive(path)
        sweep = extract_sweep_data(radar)
        vel_data = extract_velocity(radar)

        aligned_velocity = _velocity_aligned_to_reflectivity(sweep, vel_data)

        rotation_signatures = (
            detect_rotation_signatures(vel_data) if vel_data else []
        )

        filtered, rejected, report = apply_quality_control(
            sweep, rotation_signatures, velocity=aligned_velocity
        )

        self.sweep = sweep
        self.vel_data = vel_data
        self.aligned_velocity = aligned_velocity
        self.rotation_signatures = rotation_signatures
        self.filtered = filtered
        self.rejected = rejected
        self.report = report

        reflectivity = np.asarray(sweep.reflectivity, dtype=float)
        self.valid_mask = np.isfinite(reflectivity)
        self.n_valid = int(np.count_nonzero(self.valid_mask))

        classes = filtered.gate_classification
        self.nonmet_mask = np.isin(classes, list(NON_METEOROLOGICAL_CODES)) & self.valid_mask
        self.clutter_mask = classes == CLASS_CODES[GateClass.GROUND_CLUTTER]

        # (c): decompose protection into rule-2 (rotation collocation) vs
        # rule-1 (50 dBZ floor) contributions, per the task's instruction to
        # determine this by calling protected_mask with an empty rotation list.
        self.protected_with_rotation = protected_mask(sweep, rotation_signatures)
        self.protected_without_rotation = protected_mask(sweep, [])


_CACHE: dict[str, ScanMeasurement] = {}


@pytest.fixture(scope="module")
def measurements():
    for path in CLEAR_AIR_SCANS:
        if path not in _CACHE:
            _CACHE[path] = ScanMeasurement(path)
    return [_CACHE[path] for path in CLEAR_AIR_SCANS]


def test_clutter_gates_persist_across_consecutive_scans(measurements):
    masks = [m.clutter_mask for m in measurements]
    assert any(m.any() for m in masks), "no clutter identified in any scan"

    jaccards = []
    for earlier, later in zip(masks, masks[1:]):
        intersection = np.count_nonzero(earlier & later)
        union = np.count_nonzero(earlier | later)
        assert union > 0
        jaccard = intersection / union
        jaccards.append(jaccard)

    print("\nJaccard overlap between consecutive scans (ground_clutter mask):")
    for (earlier_path, later_path), jaccard in zip(
        zip(CLEAR_AIR_SCANS, CLEAR_AIR_SCANS[1:]), jaccards
    ):
        print(f"  {earlier_path.split('_')[1]} -> {later_path.split('_')[1]}: {jaccard:.4f}")

    for jaccard in jaccards:
        assert jaccard >= MIN_JACCARD_OVERLAP, f"clutter set unstable: {jaccard:.2f}"


def test_clutter_gates_are_nearly_stationary(measurements):
    print("\nMean |velocity| of ground_clutter-classified gates:")
    any_checked = False
    for m in measurements:
        clutter = m.clutter_mask
        velocity = m.aligned_velocity
        if velocity is None or not clutter.any():
            print(f"  {m.path}: skipped (no velocity or no clutter gates)")
            continue
        overlap = min(clutter.shape[0], velocity.shape[0])
        clutter_velocity = velocity[:overlap][clutter[:overlap]]
        finite = clutter_velocity[np.isfinite(clutter_velocity)]
        if finite.size == 0:
            print(f"  {m.path}: skipped (no finite velocity at clutter gates)")
            continue
        mean_abs = np.abs(finite).mean()
        print(f"  {m.path}: {mean_abs:.4f} m/s (n={finite.size})")
        any_checked = True
        assert mean_abs < MAX_MEAN_ABS_VELOCITY_MS
    assert any_checked, "no scan had both velocity data and clutter gates to check"


def test_measure_qc_effectiveness_gap(measurements):
    """Controller extension (not in the brief): does QC actually reject what
    the classifier calls non-meteorological, or is it overridden?

    Reports, per scan:
      (a) fraction of valid echo classified ground_clutter/biological vs.
          fraction apply_quality_control actually rejects
      (b) number of rotation signatures detected
      (c) how much of the gap in (a) comes from protection rule 2 (rotation
          collocation) vs rule 1 (50 dBZ floor), via protected_mask([])

    No assertions on the numbers themselves -- they are measurements, not a
    pass/fail gate. This test only fails if the measurement itself cannot be
    computed.
    """
    header = (
        f"{'scan':>8} | {'classified%':>11} | {'rejected%':>9} | "
        f"{'rot_sigs':>8} | {'prot_norot%':>11} | {'prot_norot_of_nonmet%':>21}"
    )
    print("\n" + header)
    print("-" * len(header))

    rows = []
    for m in measurements:
        n_valid = m.n_valid
        classified_frac = (
            float(np.count_nonzero(m.nonmet_mask)) / n_valid if n_valid else 0.0
        )
        rejected_frac = m.report.rejected_fraction
        n_rotations = len(m.rotation_signatures)

        # Fraction of valid gates protected under an EMPTY rotation list --
        # i.e. by rule 1 (50 dBZ floor) and rule 3 (component expansion from
        # rule-1 seeds) alone, with rule 2 removed entirely.
        protected_norot_frac = (
            float(np.count_nonzero(m.protected_without_rotation & m.valid_mask)) / n_valid
            if n_valid
            else 0.0
        )
        # Of the gates the classifier calls non-meteorological, what fraction
        # is protected even with rotation entirely disabled? What remains is
        # attributable to rule 2 (rotation collocation).
        nonmet_and_valid = m.nonmet_mask
        n_nonmet = int(np.count_nonzero(nonmet_and_valid))
        protected_norot_of_nonmet = (
            float(np.count_nonzero(m.protected_without_rotation & nonmet_and_valid)) / n_nonmet
            if n_nonmet
            else 0.0
        )
        protected_withrot_of_nonmet = (
            float(np.count_nonzero(m.protected_with_rotation & nonmet_and_valid)) / n_nonmet
            if n_nonmet
            else 0.0
        )

        scan_label = m.path.split("_")[1]
        rows.append(
            dict(
                scan=scan_label,
                classified_frac=classified_frac,
                rejected_frac=rejected_frac,
                n_rotations=n_rotations,
                protected_norot_frac=protected_norot_frac,
                protected_norot_of_nonmet=protected_norot_of_nonmet,
                protected_withrot_of_nonmet=protected_withrot_of_nonmet,
                n_nonmet=n_nonmet,
                n_valid=n_valid,
            )
        )
        print(
            f"{scan_label:>8} | {classified_frac*100:10.2f}% | {rejected_frac*100:8.2f}% | "
            f"{n_rotations:8d} | {protected_norot_frac*100:10.2f}% | "
            f"{protected_norot_of_nonmet*100:20.2f}%"
        )

    total_rotations = sum(r["n_rotations"] for r in rows)
    print(f"\nTotal rotation signatures across all six scans: {total_rotations}")
    print(
        "Interpretation: protected_norot_of_nonmet = fraction of classifier-flagged "
        "non-met gates still protected by rule 1/3 alone (empty rotation list). "
        "protected_withrot_of_nonmet = same, with real rotation signatures included. "
        "The difference between those two is attributable to rule 2 (rotation collocation)."
    )
    for r in rows:
        gap_rule2 = r["protected_withrot_of_nonmet"] - r["protected_norot_of_nonmet"]
        print(
            f"  {r['scan']}: rule2-attributable protection of non-met gates = "
            f"{gap_rule2*100:.2f} pct points "
            f"(rule1/3-only={r['protected_norot_of_nonmet']*100:.2f}%, "
            f"rule1+2+3={r['protected_withrot_of_nonmet']*100:.2f}%)"
        )

    # Sanity: the measurement must actually have run over real gates.
    assert all(r["n_valid"] > 0 for r in rows)
