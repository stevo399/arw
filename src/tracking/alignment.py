"""Put an earlier radar scan on a later scan's beam and gate order.

Each Level II volume starts at whatever azimuth the antenna happens to be
pointing, so beam index 0 points a different way in every scan (336, 343,
341 and 8 degrees on consecutive KEYX volumes; 177 degrees apart on a KTLX
pair).  Comparing masks or reflectivity by array index therefore compares
different places.  Tracking aligns the earlier scan to the later scan's
beam order, by nearest true azimuth and nearest range, before comparing.
"""

from dataclasses import replace

import numpy as np

from src.buffer import BufferedScan
from src.label_masks import LabelMaskView


def _nearest_azimuth_indices(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """For each target azimuth, the index of the circularly nearest source azimuth."""
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    if source.shape == target.shape and np.array_equal(source, target):
        return np.arange(len(target))
    order = np.argsort(source % 360.0)
    sorted_source = source[order] % 360.0
    wrapped = np.concatenate([sorted_source[-1:] - 360.0, sorted_source, sorted_source[:1] + 360.0])
    wrapped_index = np.concatenate([order[-1:], order, order[:1]])
    positions = np.searchsorted(wrapped, target % 360.0)
    positions = np.clip(positions, 1, len(wrapped) - 1)
    below, above = wrapped[positions - 1], wrapped[positions]
    choose_above = (above - (target % 360.0)) < ((target % 360.0) - below)
    return np.where(choose_above, wrapped_index[positions], wrapped_index[positions - 1])


def _nearest_range_indices(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    if source.shape == target.shape and np.array_equal(source, target):
        return np.arange(len(target))
    positions = np.clip(np.searchsorted(source, target), 1, len(source) - 1)
    choose_above = (source[positions] - target) < (target - source[positions - 1])
    return np.where(choose_above, positions, positions - 1)


def align_scan_to(earlier: BufferedScan, later: BufferedScan) -> BufferedScan:
    """`earlier` re-indexed onto `later`'s beam and gate order.

    Reflectivity, labels, object masks, azimuths and elevations are
    re-indexed; each row keeps its own true azimuth, so georeferencing stays
    correct.  With identical beam and gate order this is exactly the identity.
    """
    sweep = earlier.reflectivity_data
    target = later.reflectivity_data
    rows = _nearest_azimuth_indices(sweep.azimuths, target.azimuths)
    cols = _nearest_range_indices(sweep.ranges_m, target.ranges_m)
    identity = (
        len(rows) == len(sweep.azimuths)
        and np.array_equal(rows, np.arange(len(rows)))
        and len(cols) == len(sweep.ranges_m)
        and np.array_equal(cols, np.arange(len(cols)))
    )
    if identity:
        return earlier
    grid = np.ix_(rows, cols)
    labels = np.asarray(earlier.labeled_grid)[grid]
    aligned_sweep = replace(
        sweep,
        reflectivity=np.asarray(sweep.reflectivity)[grid],
        azimuths=np.asarray(sweep.azimuths)[rows],
        elevations=np.asarray(sweep.elevations)[rows],
        ranges_m=np.asarray(sweep.ranges_m)[cols],
        velocity=None,
        rhohv=None,
        zdr=None,
        phidp=None,
        spectrum_width=None,
        clutter_power_removed=None,
        gate_classification=None,
    )
    return replace(
        earlier,
        reflectivity_data=aligned_sweep,
        labeled_grid=labels,
        object_masks=LabelMaskView(labels, [obj.object_id for obj in earlier.detected_objects]),
    )
