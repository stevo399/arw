import numpy as np
import pytest

from src.sweeps import select_reflectivity_sweep, select_velocity_sweeps


class FakeRadar:
    """Minimal stand-in exposing only what sweep selection reads.

    Real NEXRAD split-cut volumes interleave surveillance cuts (reflectivity,
    low Nyquist, no usable velocity) with Doppler cuts (velocity, high Nyquist)
    at the same elevation. Tests must reproduce that structure — the previous
    synthetic fixtures gave every sweep velocity, which is why the defect
    survived.
    """

    def __init__(self, fixed_angles, reflectivity_valid, velocity_valid, rays_per_sweep=8, gates=10):
        self.nsweeps = len(fixed_angles)
        self.fixed_angle = {"data": np.array(fixed_angles, dtype=float)}
        self._rays = rays_per_sweep
        self._gates = gates
        self.fields = {
            "reflectivity": {"data": self._build(reflectivity_valid)},
            "velocity": {"data": self._build(velocity_valid)},
        }

    def _build(self, valid_flags):
        rows = []
        for is_valid in valid_flags:
            block = np.full((self._rays, self._gates), 1.0 if is_valid else np.nan)
            rows.append(block)
        return np.ma.masked_invalid(np.vstack(rows))

    def get_start_end(self, sweep_index):
        start = sweep_index * self._rays
        return start, start + self._rays - 1


def make_split_cut_radar():
    """Mirrors KEMX20260712_022646_V06: paired cuts, velocity only on odd sweeps."""
    return FakeRadar(
        fixed_angles=[0.48, 0.48, 0.88, 0.88, 1.27, 1.27, 1.80],
        reflectivity_valid=[True, True, True, True, True, True, True],
        velocity_valid=[False, True, False, True, False, True, True],
    )


def test_selects_surveillance_cut_for_reflectivity():
    radar = make_split_cut_radar()
    assert select_reflectivity_sweep(radar) == 0


def test_skips_empty_velocity_sweeps():
    """The defect: index-based selection returned [0, 1, 2], two of them empty."""
    radar = make_split_cut_radar()
    assert select_velocity_sweeps(radar, max_sweeps=3) == [1, 3, 5]


def test_legacy_volume_without_split_cuts():
    radar = FakeRadar(
        fixed_angles=[0.5, 1.5, 2.4],
        reflectivity_valid=[True, True, True],
        velocity_valid=[True, True, True],
    )
    assert select_reflectivity_sweep(radar) == 0
    assert select_velocity_sweeps(radar, max_sweeps=3) == [0, 1, 2]


def test_volume_with_no_velocity_at_all():
    radar = FakeRadar(
        fixed_angles=[0.5, 1.5],
        reflectivity_valid=[True, True],
        velocity_valid=[False, False],
    )
    assert select_velocity_sweeps(radar) == []


def test_respects_max_sweeps():
    radar = make_split_cut_radar()
    assert select_velocity_sweeps(radar, max_sweeps=2) == [1, 3]


def test_one_sweep_per_elevation_group():
    """Both cuts at 0.88 carry velocity — only the better one may be returned."""
    radar = FakeRadar(
        fixed_angles=[0.48, 0.48, 0.88, 0.88],
        reflectivity_valid=[True, True, True, True],
        velocity_valid=[False, True, True, True],
    )
    selected = select_velocity_sweeps(radar, max_sweeps=3)
    assert len(selected) == 2
    assert selected[0] == 1
    assert selected[1] in (2, 3)


import pyart

REFERENCE_VOLUME = "cache/KEMX/KEMX20260712_022646_V06"


@pytest.fixture(scope="module")
def real_radar():
    return pyart.io.read_nexrad_archive(REFERENCE_VOLUME)


def test_real_volume_selects_doppler_cuts(real_radar):
    """On the reference volume, velocity lives on sweeps 1, 3, 5."""
    assert select_velocity_sweeps(real_radar, max_sweeps=3) == [1, 3, 5]


def test_real_volume_selected_sweeps_all_carry_velocity(real_radar):
    from src.sweeps import sweep_field_coverage

    for sweep_index in select_velocity_sweeps(real_radar, max_sweeps=3):
        assert sweep_field_coverage(real_radar, sweep_index, "velocity") > 0.0


def test_real_volume_reflectivity_sweep_unchanged(real_radar):
    """Sweep 0 was already correct. This must not change, so QC's effect on
    detection can be measured in isolation."""
    assert select_reflectivity_sweep(real_radar) == 0
