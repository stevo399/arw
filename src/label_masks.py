"""Object masks derived on demand from a label grid."""

from collections.abc import Mapping

import numpy as np


class LabelMaskView(Mapping):
    """Object masks derived on demand from one label grid.

    Replaces a dict of full-grid boolean masks (one per object) without
    holding any of them: each lookup computes `labels == object_id`.
    """

    def __init__(self, labels: np.ndarray, object_ids):
        self._labels = labels
        self._object_ids = tuple(int(object_id) for object_id in object_ids)
        self._id_set = frozenset(self._object_ids)

    def __getitem__(self, object_id) -> np.ndarray:
        if object_id not in self._id_set:
            raise KeyError(object_id)
        return self._labels == object_id

    def __iter__(self):
        return iter(self._object_ids)

    def __len__(self) -> int:
        return len(self._object_ids)
