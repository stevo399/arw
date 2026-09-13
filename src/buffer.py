# src/buffer.py
from dataclasses import dataclass, field
from datetime import datetime
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
