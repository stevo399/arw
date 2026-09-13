# src/buffer.py
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
from typing import Any
import numpy as np
from src.parser import SweepData, VelocityData
from src.detection import DetectedObject
from src.preprocess import ScanQuality
from src.qc.report import EchoAdvisory
from src.velocity import VelocityRegion, RotationSignature


@dataclass
class TrackingSnapshot:
    """Tracker state as it stood immediately after one scan was tracked.

    Served with that scan, so a scan is never described with tracks from a
    later scan.  `active_tracks` holds deep copies of `Track` objects (typed
    loosely to avoid importing the tracker here).
    """
    active_tracks: list[Any]
    recent_events: list[dict]
    continuity_rebuilt: bool = False


@dataclass
class BufferedScan:
    """A single scan stored in the replay buffer."""
    timestamp: datetime
    site_id: str
    reflectivity_data: SweepData
    detected_objects: list[DetectedObject]
    labeled_grid: np.ndarray
    object_masks: dict[int, np.ndarray]
    scan_quality: ScanQuality | None = None
    # Local Level II volume this interpretation came from.  It lets the API
    # cache rendered map layers by immutable scan identity rather than merely
    # by timestamp.
    source_path: str | None = None
    velocity_data: VelocityData | None = None
    velocity_regions: list[VelocityRegion] = field(default_factory=list)
    rotation_signatures: list[RotationSignature] = field(default_factory=list)
    # Advisory only (2026-08-26 amendment): gates quality control would have
    # flagged, retained for inspection -- never actually removed from
    # reflectivity_data.
    echo_advisory: EchoAdvisory | None = None
    # Evidence for each precipitation band (see
    # src.map_layer.compute_precipitation_band_evidence), computed while the
    # dual-pol grids still exist.  The live pipeline releases those grids
    # before the precipitation layer is rendered.
    precipitation_band_evidence: dict[tuple[float, float], dict[str, Any]] | None = None
    # Set only when the live tracker tracked this scan.  None means the scan
    # came from the historical path and has no tracking context.
    tracking: TrackingSnapshot | None = None


class ReplayBuffer:
    """Stores the short scan history required for continuity tracking.

    A full-resolution Level II object mask is one boolean grid *per detected
    object*.  Retaining a two-hour sequence can therefore consume gigabytes
    on a convective day.  The tracker only compares the current scan with its
    immediate predecessor, so keep that pair and no more by default.
    """

    def __init__(self, max_age_minutes: int = 120, max_scans: int = 2):
        self._scans: deque[BufferedScan] = deque()
        self._max_age = timedelta(minutes=max_age_minutes)
        # Tracking needs a previous volume.  Do not allow a configuration
        # typo to silently disable that continuity check.
        self._max_scans = max(2, int(max_scans))
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
        while len(self._scans) > self._max_scans:
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
