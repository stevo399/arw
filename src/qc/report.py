"""Quality control reporting.

Every removal and every degradation is recorded. Nothing about quality control
is allowed to be silent — a scan where QC quietly ate a storm must be
distinguishable from a scan that was genuinely clear.
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class RejectedEcho:
    """Everything quality control removed, retained for inspection.

    Required by the clutter-persistence proof, which checks that the same fixed
    gates are flagged across consecutive scans.
    """

    mask: np.ndarray
    reasons: np.ndarray


@dataclass
class QualityReport:
    class_fractions: dict[str, float] = field(default_factory=dict)
    rejected_fraction: float = 0.0
    mean_confidence: float = 0.0
    degraded_modes: list[str] = field(default_factory=list)
