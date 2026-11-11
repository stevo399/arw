"""A storm is split only around cores big enough to persist between scans.

KIWA 2026-10-30: one 20 dBZ outline (6,200 to 7,700 gates every scan) was cut
into 1, 1, 1, 4, 2, 2 and 1 pieces in consecutive scans, because a 50 dBZ core
of 6 gates could seed its own storm.  Each new cut handed the tracker a new
set of storms and broke the complex's identity.
"""
import numpy as np
from hypothesis import given, settings, strategies as st

from src.detection import detect_objects

AZIMUTHS = np.linspace(0, 359, 360)
RANGES_M = np.linspace(2000, 250000, 500)


def _two_branch_blob(small_core_rows: int, small_core_cols: int) -> np.ndarray:
    """One 25 dBZ blob with two separate 40 dBZ branches.

    The left branch holds a 10-by-6 (60 gate) 55 dBZ core; the right branch
    holds a 55 dBZ core of the given size.
    """
    reflectivity = np.full((360, 500), np.nan)
    reflectivity[80:110, 190:230] = 25.0
    reflectivity[85:100, 195:208] = 40.0
    reflectivity[85:100, 212:225] = 40.0
    reflectivity[87:97, 198:204] = 55.0
    reflectivity[88:88 + small_core_rows, 216:216 + small_core_cols] = 55.0
    return reflectivity


def _count(reflectivity: np.ndarray) -> int:
    return len(detect_objects(reflectivity, AZIMUTHS, RANGES_M, radar_lat=33.29, radar_lon=-111.67))


def test_a_small_second_core_does_not_split_the_storm():
    assert _count(_two_branch_blob(small_core_rows=4, small_core_cols=5)) == 1


def test_two_large_cores_still_split_the_storm():
    assert _count(_two_branch_blob(small_core_rows=8, small_core_cols=6)) == 2


@given(seed=st.integers(0, 2**32 - 1))
@settings(max_examples=25, deadline=None)
def test_one_db_of_noise_does_not_change_the_split(seed):
    """Scan-to-scan noise of +-1 dB must not add or remove a storm."""
    rng = np.random.default_rng(seed)
    for small_rows, small_cols, expected in ((4, 5, 1), (8, 6, 2)):
        reflectivity = _two_branch_blob(small_rows, small_cols)
        reflectivity += rng.uniform(-1.0, 1.0, reflectivity.shape)
        assert _count(reflectivity) == expected
