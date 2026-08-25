"""Hard protection rules applied after classification.

These exist because low correlation coefficient cannot distinguish a flock of
birds from a tornado lofting debris. A misclassification that discards weather
is silent and unrecoverable downstream, so three overrides guarantee that
anything plausibly severe survives quality control.

Erring toward over-reporting is deliberate: a false storm is recoverable, a
deleted tornado is not.
"""

import numpy as np
from scipy.ndimage import label

from src.qc.collocation import rotation_proximity_mask

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
    Rule 2: within a rotation signature's radius plus margin.
    Rule 3: any connected component containing a gate protected by 1 or 2.
    """
    reflectivity = np.asarray(sweep.reflectivity, dtype=float)
    finite = np.isfinite(reflectivity)

    seeds = (finite & (reflectivity >= keep_dbz))
    seeds |= rotation_proximity_mask(sweep, rotation_signatures) & finite

    if not seeds.any():
        return np.zeros_like(seeds, dtype=bool)

    labeled, component_count = label(finite, structure=_CONNECTIVITY)
    if component_count == 0:
        return seeds

    protected_component_ids = np.unique(labeled[seeds])
    protected_component_ids = protected_component_ids[protected_component_ids > 0]

    return np.isin(labeled, protected_component_ids)
