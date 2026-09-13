"""Bounded cache of rendered GeoJSON map layers.

Each radar's newest retained scan is pinned and never evicted, since that is
what live maps request.  Other rendered scans (stepped-back or historical)
are kept least recently used up to a limit; an evicted rendering is rebuilt
identically from its compact scan on demand.
"""

from collections import OrderedDict
from threading import RLock

LayerKey = tuple[str, str]  # (site_id, source volume path)


class RenderedLayerCache:
    def __init__(self, max_unpinned_scans: int = 10):
        self._max_unpinned = max(0, int(max_unpinned_scans))
        self._entries: OrderedDict[LayerKey, dict[str, dict]] = OrderedDict()
        self._pinned: dict[str, LayerKey] = {}
        self._lock = RLock()

    def get(self, key: LayerKey) -> dict[str, dict] | None:
        with self._lock:
            layers = self._entries.get(key)
            if layers is not None:
                self._entries.move_to_end(key)
            return layers

    def put(self, key: LayerKey, layers: dict[str, dict], *, pinned: bool) -> None:
        with self._lock:
            self._entries[key] = layers
            self._entries.move_to_end(key)
            if pinned:
                self._pinned[key[0]] = key
            pinned_keys = set(self._pinned.values())
            unpinned = [entry for entry in self._entries if entry not in pinned_keys]
            for entry in unpinned[: max(0, len(unpinned) - self._max_unpinned)]:
                del self._entries[entry]

    def __contains__(self, key: LayerKey) -> bool:
        with self._lock:
            return key in self._entries

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._pinned.clear()
