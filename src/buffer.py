# src/buffer.py
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import deque
import numpy as np
from src.parser import ReflectivityData
from src.detection import DetectedObject


@dataclass
class BufferedScan:
    """A single scan stored in the replay buffer."""
    timestamp: datetime
    site_id: str
    reflectivity_data: ReflectivityData
    detected_objects: list[DetectedObject]
    labeled_grid: np.ndarray
    object_masks: dict[int, np.ndarray]


class ReplayBuffer:
    """Stores up to max_age_minutes of parsed scan data in memory."""

    def __init__(self, max_age_minutes: int = 120):
        self._scans: deque[BufferedScan] = deque()
        self._max_age = timedelta(minutes=max_age_minutes)
        self._current_site: str | None = None

    def add_scan(self, scan: BufferedScan) -> None:
        """Add a scan to the buffer. Resets if site changes. Evicts old scans."""
        if self._current_site is not None and scan.site_id != self._current_site:
            self._scans.clear()
        self._current_site = scan.site_id
        self._scans.append(scan)
        self._evict_old()

    def _evict_old(self) -> None:
        """Remove scans older than max_age from the latest scan."""
        if not self._scans:
            return
        cutoff = self._scans[-1].timestamp - self._max_age
        while self._scans and self._scans[0].timestamp < cutoff:
            self._scans.popleft()

    @property
    def scan_count(self) -> int:
        return len(self._scans)

    @property
    def current_scan(self) -> BufferedScan | None:
        return self._scans[-1] if self._scans else None

    @property
    def previous_scan(self) -> BufferedScan | None:
        return self._scans[-2] if len(self._scans) >= 2 else None

    @property
    def all_scans(self) -> list[BufferedScan]:
        return list(self._scans)

    @property
    def time_range(self) -> tuple[datetime, datetime] | None:
        if not self._scans:
            return None
        return (self._scans[0].timestamp, self._scans[-1].timestamp)
