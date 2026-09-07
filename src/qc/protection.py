"""Hard protection rules applied after classification.

These exist because low correlation coefficient cannot distinguish a flock of
birds from a tornado lofting debris. A misclassification that discards weather
is silent and unrecoverable downstream, so three overrides guarantee that
anything plausibly severe survives quality control.

Erring toward over-reporting is deliberate: a false storm is recoverable, a
deleted tornado is not.

Amendment, 2026-08-26: `apply_quality_control` no longer deletes any echo at
all, protected or not, so "discard"/"never discard" below now describes
exemption from the ADVISORY flag `apply_quality_control` computes, not actual
removal. `protected_mask` itself is unchanged -- it remains the input that
decides which classifier-flagged gates are excluded from that advisory
signal.
"""

import numpy as np

from src.geometry import label_periodic_azimuth
from src.qc.collocation import rotation_proximity_mask
from src.velocity import is_quality_protection_eligible

# ARW-tuned. Above this, echo is too intense to be biological or ground clutter
# in any operationally meaningful case.
PROTECTED_MIN_DBZ = 50.0

_CONNECTIVITY = np.ones((3, 3), dtype=int)


def protected_mask(
    sweep,
    rotation_signatures,
    keep_dbz: float = PROTECTED_MIN_DBZ,
) -> np.ndarray:
    """Gates that quality control may never discard.

    Rule 1: reflectivity at or above keep_dbz.
    Rule 2: within an assessed, cell-associated and independently supported
    rotation signature's radius plus margin. Raw/unconfirmed couplets are
    diagnostic data, never QC-advisory exemptions.
    Rule 3: any connected component containing a gate protected by 1 or 2.
    """
    reflectivity = np.asarray(sweep.reflectivity, dtype=float)
    finite = np.isfinite(reflectivity)

    seeds = (finite & (reflectivity >= keep_dbz))
    eligible_rotations = [
        signature for signature in rotation_signatures
        if is_quality_protection_eligible(signature)
    ]
    seeds |= rotation_proximity_mask(sweep, eligible_rotations) & finite

    if not seeds.any():
        return np.zeros_like(seeds, dtype=bool)

    labeled, component_count = label_periodic_azimuth(finite, structure=_CONNECTIVITY)
    if component_count == 0:
        return seeds

    protected_component_ids = np.unique(labeled[seeds])
    protected_component_ids = protected_component_ids[protected_component_ids > 0]

    return np.isin(labeled, protected_component_ids)
