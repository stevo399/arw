"""Quality control reporting.

Every degradation is recorded, and every gate's classification is reported.
Nothing about quality control is allowed to be silent -- a scan where QC
quietly ate a storm must be distinguishable from a scan that was genuinely
clear.

Amendment, 2026-08-26 ("quality control flags, it does not delete"): quality
control no longer removes any echo from the reflectivity field. Everything
below is advisory only -- it tells a consumer which gates a classifier-driven
filter WOULD exclude if one were applied, not which gates were excluded.
Nothing here is a record of deletion.
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class EchoAdvisory:
    """Gates quality control WOULD flag as non-meteorological, and why.

    Advisory only. `apply_quality_control` never removes these gates from the
    reflectivity field it returns -- `mask` is a would-flag signal for
    consumers (speech, GeoJSON, further analysis) to use as they see fit, not
    a record of anything actually discarded. Formerly named `RejectedEcho`,
    which implied removal that no longer happens; renamed under the
    2026-08-26 amendment to make that unambiguous.

    Still required by the clutter-persistence proof, which checks that the
    same fixed gates are flagged across consecutive scans.
    """

    mask: np.ndarray
    reasons: np.ndarray


@dataclass
class QualityReport:
    class_fractions: dict[str, float] = field(default_factory=dict)
    # Fraction of gates that WOULD be flagged as non-meteorological and are
    # not exempted by protection -- advisory only, not a fraction actually
    # removed. Formerly `rejected_fraction`; renamed under the 2026-08-26
    # amendment because nothing is rejected from the output any more.
    advisory_fraction: float = 0.0
    mean_confidence: float = 0.0
    degraded_modes: list[str] = field(default_factory=list)
