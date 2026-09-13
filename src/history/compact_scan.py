"""One processed radar volume, stored compactly and losslessly.

Holds exactly what map layers, detection consumers, speech and the tracker
read after processing: reflectivity at every gate (contour interpolation
needs sub-threshold gates too), one label grid in place of per-object masks,
sweep geometry, and records (objects, scan quality, precipitation-band
evidence, tracking snapshot).  Dual-pol grids are never kept; precipitation
evidence is precomputed during processing (spec Amendment 1, A1).
"""

from dataclasses import dataclass, replace
from datetime import datetime, timezone
import zlib

import numpy as np

from src.buffer import BufferedScan, TrackingSnapshot
from src.history import records as record_codec
from src.history.codec import (
    REFLECTIVITY_OFFSET_DBZ,
    REFLECTIVITY_STEP_DBZ,
    ZLIB_LEVEL,
    EncodedGrid,
    decode,
    encode_integer,
    encode_raw,
    encode_stepped,
)
from src.label_masks import LabelMaskView  # re-exported for history callers
from src.parser import SweepData

SCHEMA_VERSION = 1


def timestamp_key(value: datetime) -> datetime:
    """Naive-UTC form used to compare scan times from any source."""
    if value.tzinfo is not None:
        return value.astimezone(timezone.utc).replace(tzinfo=None)
    return value


def verify_label_masks(scan: BufferedScan) -> None:
    """Refuse a scan whose masks are not exactly their label-grid regions.

    Detection builds the label grid by writing each object's mask in turn,
    so overlapping masks would silently lose gates.  Checked on real data
    (no overlaps on KTLX, KEMX, KIWA), but verified on every scan anyway.
    """
    labels = np.asarray(scan.labeled_grid)
    for obj in scan.detected_objects:
        mask = scan.object_masks.get(obj.object_id)
        if mask is None or not np.array_equal(labels == obj.object_id, mask):
            raise ValueError(
                f"{scan.site_id} {scan.timestamp.isoformat()}: object {obj.object_id} "
                "mask is not exactly its label-grid region"
            )


@dataclass(frozen=True)
class CompactScan:
    site_id: str
    timestamp: datetime
    scan_timestamp_text: str
    source_path: str | None
    object_count: int
    tracked: bool
    reflectivity: EncodedGrid
    labels: EncodedGrid
    azimuths: EncodedGrid
    ranges_m: EncodedGrid
    elevations: EncodedGrid
    records: bytes
    schema_version: int = SCHEMA_VERSION

    @classmethod
    def from_buffered_scan(cls, scan: BufferedScan) -> "CompactScan":
        verify_label_masks(scan)
        sweep = scan.reflectivity_data
        label = f"{scan.site_id} {scan.timestamp.isoformat()}"
        records = {
            "elevation_angle": sweep.elevation_angle,
            "elevation_angles": list(sweep.elevation_angles),
            "radar_lat": sweep.radar_lat,
            "radar_lon": sweep.radar_lon,
            "radar_alt_m": sweep.radar_alt_m,
            "sweep_timestamp": sweep.timestamp,
            "detected_objects": list(scan.detected_objects),
            "scan_quality": scan.scan_quality,
            "rotation_signatures": list(scan.rotation_signatures),
            "precipitation_band_evidence": scan.precipitation_band_evidence,
            "tracking": scan.tracking,
        }
        timestamp_text = (
            sweep.timestamp if isinstance(sweep.timestamp, str) else sweep.timestamp.isoformat()
        )
        return cls(
            site_id=scan.site_id.upper(),
            timestamp=scan.timestamp,
            scan_timestamp_text=timestamp_text,
            source_path=scan.source_path,
            object_count=len(scan.detected_objects),
            tracked=scan.tracking is not None,
            reflectivity=encode_stepped(
                sweep.reflectivity,
                step=REFLECTIVITY_STEP_DBZ,
                offset=REFLECTIVITY_OFFSET_DBZ,
                label=f"{label} reflectivity",
            ),
            labels=encode_integer(np.asarray(scan.labeled_grid), label=f"{label} labels"),
            azimuths=encode_raw(sweep.azimuths),
            ranges_m=encode_raw(sweep.ranges_m),
            elevations=encode_raw(sweep.elevations),
            records=_encode_records(records),
        )

    def read_records(self) -> dict:
        """Fresh copies of every record; safe for callers to mutate."""
        return record_codec.loads(zlib.decompress(self.records).decode("utf-8"))

    def to_buffered_scan(self) -> BufferedScan:
        records = self.read_records()
        labels = decode(self.labels)
        objects = records["detected_objects"]
        sweep = SweepData(
            reflectivity=decode(self.reflectivity),
            azimuths=decode(self.azimuths),
            ranges_m=decode(self.ranges_m),
            elevation_angle=records["elevation_angle"],
            elevations=decode(self.elevations),
            elevation_angles=records["elevation_angles"],
            radar_lat=records["radar_lat"],
            radar_lon=records["radar_lon"],
            radar_alt_m=records["radar_alt_m"],
            timestamp=records["sweep_timestamp"],
        )
        return BufferedScan(
            timestamp=self.timestamp,
            site_id=self.site_id,
            reflectivity_data=sweep,
            detected_objects=objects,
            labeled_grid=labels,
            object_masks=LabelMaskView(labels, [obj.object_id for obj in objects]),
            scan_quality=records["scan_quality"],
            source_path=self.source_path,
            rotation_signatures=records["rotation_signatures"],
            precipitation_band_evidence=records["precipitation_band_evidence"],
            tracking=records["tracking"],
        )

    def with_tracking(self, tracking: TrackingSnapshot | None) -> "CompactScan":
        records = self.read_records()
        records["tracking"] = tracking
        return replace(self, records=_encode_records(records), tracked=tracking is not None)

    def validate(self) -> None:
        """Decode everything once; raises if any part is corrupt or inconsistent."""
        reflectivity = decode(self.reflectivity)
        labels = decode(self.labels)
        records = self.read_records()
        if reflectivity.shape != labels.shape:
            raise ValueError("reflectivity and label grids differ in shape")
        if decode(self.azimuths).shape[0] != reflectivity.shape[0]:
            raise ValueError("azimuth count does not match the grid")
        if decode(self.ranges_m).shape[0] != reflectivity.shape[1]:
            raise ValueError("range count does not match the grid")
        if len(records["detected_objects"]) != self.object_count:
            raise ValueError("object count does not match the records")

    @property
    def nbytes(self) -> int:
        grids = (self.reflectivity, self.labels, self.azimuths, self.ranges_m, self.elevations)
        return sum(grid.nbytes for grid in grids) + len(self.records)


def _encode_records(records: dict) -> bytes:
    return zlib.compress(record_codec.dumps(records).encode("utf-8"), ZLIB_LEVEL)
