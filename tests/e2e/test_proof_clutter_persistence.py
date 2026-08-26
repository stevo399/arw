# tests/e2e/test_proof_clutter_persistence.py
"""Proof 2 from the design spec: ground clutter sits still.

Fix round 1: the original proof (gate-level Jaccard overlap of the
classifier's `ground_clutter` label, over ALL gates, between consecutive
scans) tested a premise that turned out to be false on this data. The
WSR-88D's own signal processor already runs a clutter filter before ARW ever
sees the data: `clutter_filter_power_removed` shows power suppressed at
144,189 gates across these six scans, median 25 dB, max 73 dB. Reflectivity
>= 30 dBZ is ZERO gates in all six clear-air scans. There is no persistent
strong terrain return left in this data for a gate-level stability test to
find -- the hardware already removed it. Restricting the old Jaccard test to
stronger echo does not rescue it: it gets WORSE (all gates 0.120, >=20 dBZ
0.006, >=30 dBZ undefined/0.000), because the >=20 dBZ population in clear
air is transient biology (insects, birds drifting with the wind), not fixed
terrain, and lowering the bar to make gate-level overlap pass would be
exactly the kind of dishonesty this project exists to remove.

The original gate-level stability test is REMOVED (not xfailed) for that
reason: its premise -- that a persistent >=X-dBZ clutter footprint exists in
this hardware-filtered data for gate overlap to measure -- is false, not
merely "not yet true." An xfail implies we expect it to become true once a
defect is fixed; nothing about fixing QC would ever change the fact that the
WSR-88D's clutter filter already removed the persistent terrain signal before
ARW's classifier gets a look at it. Keeping this test only as documentation
of a bug does not apply here, so it is deleted rather than xfailed. See the
report for the full diagnosis.

What remains, re-scoped to gates >= detection.MIN_DBZ_THRESHOLD (20 dBZ) --
the only population that can ever become a detected object, since anything
below that threshold never reaches the user:

(A) STATIONARITY: ground_clutter-classified gates must show low mean
    absolute radial velocity. This is provable physics and passes.

(B) DISCRIMINATION: biological scatterers drift with the wind; ground
    clutter does not. Among gates >= 20 dBZ, mean |velocity| of
    BIOLOGICAL-classified gates must be meaningfully higher than that of
    GROUND_CLUTTER-classified gates. If the classifier assigned these two
    classes arbitrarily, the two means would be indistinguishable.

(C) EFFECTIVENESS: documented as a strict xfail, not a passing test and not
    a silent omission. In a clear-air scan, essentially all echo >= 20 dBZ is
    non-meteorological (insects, birds, residual clutter) -- roughly 60% of
    it is classified ground_clutter or biological in every one of these six
    scans. Quality control should reject most of that. It currently rejects
    well under 1%. The cause (measured, not fixed here): 121-132 spurious
    rotation signatures per clear-air scan seed protection rule 2, which rule
    3 then expands across whole connected components -- a known defect in
    the pre-existing shear detector. strict=True means this test will fail
    loudly the day that defect is fixed and QC starts actually rejecting
    this echo, which is the signal we want.
"""

import numpy as np
import pyart
import pytest

from src.detection import MIN_DBZ_THRESHOLD
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

MAX_MEAN_ABS_VELOCITY_MS = 3.0

# (B) Justification: pooled across all six scans, biological mean |velocity|
# is ~7.2 m/s (std ~6.0) versus ~2.5 m/s (std ~4.5) for ground_clutter -- a
# ratio between 2.6x and 3.1x in every individual scan. Requiring at least a
# 2x ratio is comfortably below every observed scan (margin of >0.6x against
# the weakest case) while still being a real physical separation, not noise.
MIN_BIOLOGICAL_TO_CLUTTER_VELOCITY_RATIO = 2.0

# (C) Justification: the classifier labels roughly 59-62% of >= 20 dBZ echo
# non-meteorological (ground_clutter or biological) in every one of these six
# clear-air scans -- see the extension table. Effective QC should reject a
# majority of that population; requiring rejection of over half of >= 20 dBZ
# echo is a conservative reading of "essentially all of it is
# non-meteorological" that still leaves room for legitimately protected
# echo (e.g. real weather that happens to co-occur). Measured rejection is
# under 1%, so this is expected to fail today.
MIN_GE20_REJECTED_FRACTION = 0.5

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

        self.ge20_mask = self.valid_mask & (reflectivity >= MIN_DBZ_THRESHOLD)
        self.n_ge20 = int(np.count_nonzero(self.ge20_mask))

        classes = filtered.gate_classification
        self.nonmet_mask = np.isin(classes, list(NON_METEOROLOGICAL_CODES)) & self.valid_mask
        self.clutter_mask = classes == CLASS_CODES[GateClass.GROUND_CLUTTER]
        self.biological_mask = classes == CLASS_CODES[GateClass.BIOLOGICAL]

        self.clutter_ge20_mask = self.clutter_mask & self.ge20_mask
        self.biological_ge20_mask = self.biological_mask & self.ge20_mask
        self.nonmet_ge20_mask = self.nonmet_mask & self.ge20_mask
        self.n_nonmet_ge20 = int(np.count_nonzero(self.nonmet_ge20_mask))

        self.rejected_ge20 = int(np.count_nonzero(rejected.mask & self.ge20_mask))
        self.rejected_frac_ge20 = (
            self.rejected_ge20 / self.n_ge20 if self.n_ge20 else float("nan")
        )

        # (c): decompose protection into rule-2 (rotation collocation) vs
        # rule-1 (50 dBZ floor) contributions, per the task's instruction to
        # determine this by calling protected_mask with an empty rotation list.
        self.protected_with_rotation = protected_mask(sweep, rotation_signatures)
        self.protected_without_rotation = protected_mask(sweep, [])

    def _abs_velocity(self, mask: np.ndarray) -> np.ndarray:
        velocity = self.aligned_velocity
        if velocity is None:
            return np.array([])
        overlap = min(mask.shape[0], velocity.shape[0])
        values = velocity[:overlap][mask[:overlap]]
        return np.abs(values[np.isfinite(values)])


_CACHE: dict[str, ScanMeasurement] = {}


@pytest.fixture(scope="module")
def measurements():
    for path in CLEAR_AIR_SCANS:
        if path not in _CACHE:
            _CACHE[path] = ScanMeasurement(path)
    return [_CACHE[path] for path in CLEAR_AIR_SCANS]


def test_clutter_gates_are_nearly_stationary(measurements):
    """(A) STATIONARITY -- provable physics, unchanged from the original proof."""
    print("\nMean |velocity| of ground_clutter-classified gates:")
    any_checked = False
    for m in measurements:
        clutter_velocity = m._abs_velocity(m.clutter_mask)
        if m.aligned_velocity is None or not m.clutter_mask.any():
            print(f"  {m.path}: skipped (no velocity or no clutter gates)")
            continue
        if clutter_velocity.size == 0:
            print(f"  {m.path}: skipped (no finite velocity at clutter gates)")
            continue
        mean_abs = clutter_velocity.mean()
        print(f"  {m.path}: {mean_abs:.4f} m/s (n={clutter_velocity.size})")
        any_checked = True
        assert mean_abs < MAX_MEAN_ABS_VELOCITY_MS
    assert any_checked, "no scan had both velocity data and clutter gates to check"


def test_biological_gates_drift_faster_than_clutter_gates_at_ge20dbz(measurements):
    """(B) DISCRIMINATION -- the sharpest check available on this data.

    Biological scatterers (insects, birds) drift with the wind; ground
    clutter does not move. Restricted to gates >= 20 dBZ (the only population
    that can ever reach the user), biological-classified gates must show
    meaningfully higher mean |velocity| than ground_clutter-classified gates.
    If the classifier assigned these labels arbitrarily on this data, the two
    means would be statistically indistinguishable.
    """
    print(f"\nGates >= {MIN_DBZ_THRESHOLD} dBZ: biological vs ground_clutter mean |velocity|:")
    all_bio = []
    all_clu = []
    any_checked = False
    for m in measurements:
        bio_velocity = m._abs_velocity(m.biological_ge20_mask)
        clu_velocity = m._abs_velocity(m.clutter_ge20_mask)
        if bio_velocity.size == 0 or clu_velocity.size == 0:
            print(f"  {m.path}: skipped (n_bio={bio_velocity.size}, n_clu={clu_velocity.size})")
            continue
        bio_mean = bio_velocity.mean()
        clu_mean = clu_velocity.mean()
        ratio = bio_mean / clu_mean if clu_mean else float("inf")
        print(
            f"  {m.path}: biological={bio_mean:.4f} m/s (n={bio_velocity.size}), "
            f"clutter={clu_mean:.4f} m/s (n={clu_velocity.size}), ratio={ratio:.2f}x"
        )
        all_bio.append(bio_velocity)
        all_clu.append(clu_velocity)
        any_checked = True
        assert bio_mean >= MIN_BIOLOGICAL_TO_CLUTTER_VELOCITY_RATIO * clu_mean, (
            f"{m.path}: biological mean {bio_mean:.3f} m/s is not "
            f"{MIN_BIOLOGICAL_TO_CLUTTER_VELOCITY_RATIO}x clutter mean {clu_mean:.3f} m/s"
        )

    assert any_checked, "no scan had both biological and clutter gates >= 20 dBZ to compare"

    pooled_bio = np.concatenate(all_bio)
    pooled_clu = np.concatenate(all_clu)
    pooled_bio_mean = pooled_bio.mean()
    pooled_clu_mean = pooled_clu.mean()
    print(
        f"  POOLED: biological={pooled_bio_mean:.4f} m/s (n={pooled_bio.size}, "
        f"std={pooled_bio.std():.4f}), clutter={pooled_clu_mean:.4f} m/s "
        f"(n={pooled_clu.size}, std={pooled_clu.std():.4f}), "
        f"ratio={pooled_bio_mean / pooled_clu_mean:.2f}x"
    )
    assert pooled_bio_mean >= MIN_BIOLOGICAL_TO_CLUTTER_VELOCITY_RATIO * pooled_clu_mean


@pytest.mark.xfail(
    strict=True,
    reason=(
        "(C) EFFECTIVENESS, documented known defect. Measured: pooled across "
        "the six clear-air KIWA scans of 2026-07-12, apply_quality_control "
        "rejects only ~0.39% of gates >= 20 dBZ (per-scan range 0.11%-0.80%), "
        "even though the classifier labels roughly 59-62% of that same "
        "population ground_clutter or biological in every scan. Cause: "
        "121-132 spurious rotation signatures per clear-air scan (755 total "
        "across the six scans) seed protection rule 2 (rotation collocation "
        "in src/qc/protection.py), which rule 3 (connected-component "
        "expansion) then spreads across whole connected regions of the "
        "clutter/biological field -- overriding nearly all of the "
        "classifier's judgement before it reaches the rejected-echo output. "
        "This is a known defect in the pre-existing shear detector "
        "(src/velocity.py detect_rotation_signatures / "
        "_detect_shear_single_sweep), not in the classifier or in QC's "
        "reject logic. strict=True: this test will fail loudly -- signaling "
        "the defect is fixed -- the day rejected_frac_ge20 actually exceeds "
        f"{MIN_GE20_REJECTED_FRACTION:.0%}."
    ),
)
def test_qc_rejects_most_ge20dbz_nonmeteorological_echo(measurements):
    total_ge20 = sum(m.n_ge20 for m in measurements)
    total_rejected_ge20 = sum(m.rejected_ge20 for m in measurements)
    pooled_frac = total_rejected_ge20 / total_ge20 if total_ge20 else 0.0
    print(f"\nPooled rejected fraction of >= {MIN_DBZ_THRESHOLD} dBZ echo: {pooled_frac:.4f}")
    for m in measurements:
        print(f"  {m.path}: rejected_frac_ge20={m.rejected_frac_ge20:.4f} (n_ge20={m.n_ge20})")
    assert pooled_frac > MIN_GE20_REJECTED_FRACTION


def test_measure_qc_effectiveness_gap(measurements):
    """Controller extension (not in the brief): does QC actually reject what
    the classifier calls non-meteorological, or is it overridden?

    Reports, per scan:
      (a) fraction of valid echo classified ground_clutter/biological vs.
          fraction apply_quality_control actually rejects, both over ALL
          valid gates and restricted to the >= 20 dBZ population that can
          ever reach the user
      (b) number of rotation signatures detected
      (c) how much of the gap in (a) comes from protection rule 2 (rotation
          collocation) vs rule 1 (50 dBZ floor), via protected_mask([])

    No assertions on the numbers themselves -- they are measurements, not a
    pass/fail gate. This test only fails if the measurement itself cannot be
    computed.
    """
    header = (
        f"{'scan':>8} | {'class%(all)':>11} | {'rej%(all)':>9} | "
        f"{'class%(>=20)':>13} | {'rej%(>=20)':>10} | {'rot_sigs':>8} | "
        f"{'prot_norot%':>11} | {'prot_norot_of_nonmet%':>21}"
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
        classified_frac_ge20 = (
            float(m.n_nonmet_ge20) / m.n_ge20 if m.n_ge20 else 0.0
        )
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
                classified_frac_ge20=classified_frac_ge20,
                rejected_frac_ge20=m.rejected_frac_ge20,
                n_rotations=n_rotations,
                protected_norot_frac=protected_norot_frac,
                protected_norot_of_nonmet=protected_norot_of_nonmet,
                protected_withrot_of_nonmet=protected_withrot_of_nonmet,
                n_nonmet=n_nonmet,
                n_valid=n_valid,
            )
        )
        print(
            f"{scan_label:>8} | {classified_frac*100:9.2f}% | {rejected_frac*100:8.2f}% | "
            f"{classified_frac_ge20*100:11.2f}% | {m.rejected_frac_ge20*100:9.2f}% | "
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
