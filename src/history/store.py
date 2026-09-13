"""Per-radar compact scan history: ring, tracker, persistence and rebuild.

A RadarHistory owns one radar's newest tracked scans and the tracker that
produced them.  The tracker never retains a full previous scan; it reloads
it from the ring.  Every retained scan carries the tracking snapshot from its
own time.  On disk each scan is one `.arwscan` file next to the tracker
state; after a restart the state is used only if it matches the newest
retained scan, otherwise continuity is rebuilt from the ring and flagged.
"""

from collections import OrderedDict
from contextlib import contextmanager
from datetime import datetime
import logging
from pathlib import Path
from threading import RLock

from src.buffer import BufferedScan
from src.history import records as record_codec
from src.history.compact_scan import CompactScan, timestamp_key, verify_label_masks
from src.history.disk import (
    HISTORY_SUFFIX,
    CompactScanLoadError,
    atomic_write_bytes,
    history_filename,
    load_compact_scan,
    save_compact_scan,
)
from src.tracker import StormTracker

_logger = logging.getLogger(__name__)

RING_SIZE = 5
TRACKER_STATE_FILENAME = "tracker_state.json"


def _unlink_quietly(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        _logger.warning("Unable to delete radar history file %s", path, exc_info=True)


class RadarHistory:
    def __init__(self, site_id: str, directory: Path | None, *, ring_size: int = RING_SIZE):
        self.site_id = site_id.upper()
        self.directory = Path(directory) if directory is not None else None
        self.ring_size = max(1, int(ring_size))
        self.lock = RLock()
        self.last_write_error: str | None = None
        self._ring: list[CompactScan] = []
        self.tracker = self._new_tracker()

    @classmethod
    def open(cls, site_id: str, directory: Path | None, *, ring_size: int = RING_SIZE) -> "RadarHistory":
        history = cls(site_id, directory, ring_size=ring_size)
        if history.directory is not None and history.directory.is_dir():
            with history.lock:
                history._restore_from_disk()
        return history

    def _new_tracker(self) -> StormTracker:
        return StormTracker(scan_loader=self._load_for_tracker, reacquire=True)

    def _restore_tracker(self, state: dict) -> StormTracker:
        return StormTracker.from_state(state, self._load_for_tracker, reacquire=True)

    # -- reads -----------------------------------------------------------

    def scans(self) -> list[CompactScan]:
        with self.lock:
            return list(self._ring)

    def newest(self) -> CompactScan | None:
        with self.lock:
            return self._ring[-1] if self._ring else None

    def find_by_timestamp(self, when: datetime) -> CompactScan | None:
        key = timestamp_key(when)
        with self.lock:
            return next((scan for scan in self._ring if timestamp_key(scan.timestamp) == key), None)

    def find_by_source(self, source_path: str) -> CompactScan | None:
        with self.lock:
            return next((scan for scan in self._ring if scan.source_path == source_path), None)

    def _load_for_tracker(self, site_id: str, timestamp: datetime) -> BufferedScan | None:
        if site_id.upper() != self.site_id:
            return None
        compact = self.find_by_timestamp(timestamp)
        return compact.to_buffered_scan() if compact is not None else None

    # -- live path -------------------------------------------------------

    def add_live_scan(self, buffered: BufferedScan) -> CompactScan | None:
        """Track and retain this radar's newest volume.

        Returns None, without tracking, for a volume that is not newer than
        the newest retained scan.
        """
        with self.lock:
            newest = self.newest()
            if newest is not None and timestamp_key(buffered.timestamp) <= timestamp_key(newest.timestamp):
                return None
            verify_label_masks(buffered)  # refuse before the tracker advances
            self.tracker.update(buffered)
            for obj in buffered.detected_objects:
                obj.temporal_status = self.tracker.temporal_status_for_current_object(obj.object_id)
            buffered.tracking = self.tracker.snapshot()
            compact = CompactScan.from_buffered_scan(buffered)
            self._ring.append(compact)
            evicted = self._ring[: -self.ring_size]
            self._ring = self._ring[-self.ring_size:]
            for scan in evicted:
                # Missing storms last seen in an evicted scan can never be reacquired.
                self.tracker.expire_scan(self.site_id, scan.timestamp)
            self.tracker.release_previous_scan()
            self._persist([compact], evicted)
            return compact

    # -- persistence -----------------------------------------------------

    def _persist(self, written: list[CompactScan], evicted: list[CompactScan]) -> None:
        if self.directory is None:
            return
        try:
            for scan in written:
                save_compact_scan(scan, self.directory)
            self._write_tracker_state()
        except OSError as exc:
            _logger.exception("Unable to write radar history for %s", self.site_id)
            self.last_write_error = f"{type(exc).__name__}: {exc}"
            return
        self.last_write_error = None
        for scan in evicted:
            _unlink_quietly(self.directory / history_filename(scan.timestamp))

    def _write_tracker_state(self) -> None:
        document = {
            "site_id": self.site_id,
            "last_scan_timestamp": self.tracker.last_scan_timestamp,
            "state": self.tracker.export_state(),
        }
        atomic_write_bytes(
            self.directory / TRACKER_STATE_FILENAME,
            record_codec.dumps(document).encode("utf-8"),
        )

    def _restore_from_disk(self) -> None:
        for leftover in self.directory.glob("*.tmp"):
            _unlink_quietly(leftover)
        loaded: list[CompactScan] = []
        for path in sorted(self.directory.glob(f"*{HISTORY_SUFFIX}")):
            try:
                scan = load_compact_scan(path)
            except CompactScanLoadError as exc:
                _logger.warning("Discarding unreadable radar history file %s: %s", path, exc)
                _unlink_quietly(path)
                continue
            if scan.site_id != self.site_id:
                _logger.warning("Ignoring %s history file %s in %s's directory", scan.site_id, path, self.site_id)
                continue
            loaded.append(scan)
        loaded.sort(key=lambda scan: timestamp_key(scan.timestamp))
        for stale in loaded[: -self.ring_size]:
            _unlink_quietly(self.directory / history_filename(stale.timestamp))
        self._ring = loaded[-self.ring_size:]
        if not self._ring:
            return
        restored = self._load_saved_tracker()
        if restored is not None:
            self.tracker = restored
        else:
            self._rebuild_tracking()

    def _load_saved_tracker(self) -> StormTracker | None:
        path = self.directory / TRACKER_STATE_FILENAME
        try:
            document = record_codec.loads(path.read_text(encoding="utf-8"))
            if document["site_id"] != self.site_id:
                raise ValueError(f"state belongs to {document['site_id']}")
            saved_at = document["last_scan_timestamp"]
            newest_at = self._ring[-1].timestamp
            if saved_at is None or timestamp_key(saved_at) != timestamp_key(newest_at):
                raise ValueError(f"state was saved at {saved_at}, newest retained scan is {newest_at}")
            return self._restore_tracker(document["state"])
        except Exception as exc:  # missing, unreadable, stale or incompatible
            _logger.warning(
                "Saved tracker state for %s is unusable (%s); rebuilding continuity from retained scans",
                self.site_id, exc,
            )
            return None

    def _rebuild_tracking(self) -> None:
        self.tracker = self._new_tracker()
        for index, compact in enumerate(list(self._ring)):
            buffered = compact.to_buffered_scan()
            self.tracker.update(buffered)
            for obj in buffered.detected_objects:
                obj.temporal_status = self.tracker.temporal_status_for_current_object(obj.object_id)
            snapshot = self.tracker.snapshot()
            snapshot.continuity_rebuilt = True
            buffered.tracking = snapshot
            self._ring[index] = CompactScan.from_buffered_scan(buffered)
            self.tracker.release_previous_scan()
        self._persist(list(self._ring), [])


class HistoryRegistry:
    """Radar histories in memory, least recently used evicted first.

    An evicted radar's ring and tracker state reload from disk on next use.
    A radar whose lock is held (in use) is never evicted.
    """

    def __init__(self, root: Path | None, *, max_radars: int = 20, ring_size: int = RING_SIZE):
        self._root = Path(root) if root is not None else None
        self._max_radars = max(1, int(max_radars))
        self._ring_size = ring_size
        self._histories: OrderedDict[str, RadarHistory] = OrderedDict()
        self._lock = RLock()

    @contextmanager
    def use(self, site_id: str):
        key = site_id.upper()
        with self._lock:
            history = self._histories.get(key)
            if history is None:
                directory = None if self._root is None else self._root / key / "history"
                history = RadarHistory.open(key, directory, ring_size=self._ring_size)
                self._histories[key] = history
            self._histories.move_to_end(key)
            history.lock.acquire()
            self._evict_idle(in_use=key)
        try:
            yield history
        finally:
            history.lock.release()

    def _evict_idle(self, in_use: str) -> None:
        for key in list(self._histories):
            if len(self._histories) <= self._max_radars:
                return
            if key == in_use:
                # This thread holds its lock, and RLock is reentrant, so a
                # non-blocking acquire would wrongly succeed.
                continue
            candidate = self._histories[key]
            if candidate.lock.acquire(blocking=False):
                try:
                    del self._histories[key]
                finally:
                    candidate.lock.release()

    def loaded_sites(self) -> list[str]:
        with self._lock:
            return list(self._histories)

    def clear(self) -> None:
        with self._lock:
            self._histories.clear()
