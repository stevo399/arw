"""The `.arwscan` file: one CompactScan as an uncompressed NumPy .npz archive.

Grid payloads and records are already zlib-compressed; the archive only
frames them next to a JSON header.  Loading never unpickles.  Every write
goes to a temporary file in the same directory and is then atomically
renamed, so a crash can never leave a half-written scan in place.
"""

from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path

import numpy as np

from src.history.codec import EncodedGrid
from src.history.compact_scan import SCHEMA_VERSION, CompactScan

HISTORY_SUFFIX = ".arwscan"
_GRID_NAMES = ("reflectivity", "labels", "azimuths", "ranges_m", "elevations")


class CompactScanLoadError(Exception):
    """A history file is unreadable, corrupt, or from another schema."""


def history_filename(timestamp: datetime) -> str:
    if timestamp.tzinfo is not None:
        timestamp = timestamp.astimezone(timezone.utc)
    return timestamp.strftime("%Y%m%d_%H%M%S") + HISTORY_SUFFIX


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with open(temporary, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def serialize_compact_scan(scan: CompactScan) -> bytes:
    header = {
        "schema_version": scan.schema_version,
        "site_id": scan.site_id,
        "timestamp": scan.timestamp.isoformat(),
        "scan_timestamp_text": scan.scan_timestamp_text,
        "source_path": scan.source_path,
        "object_count": scan.object_count,
        "tracked": scan.tracked,
        "grids": {
            name: {
                "kind": grid.kind,
                "stored_dtype": grid.stored_dtype,
                "original_dtype": grid.original_dtype,
                "shape": list(grid.shape),
                "step": grid.step,
                "offset": grid.offset,
            }
            for name, grid in ((name, getattr(scan, name)) for name in _GRID_NAMES)
        },
    }
    arrays = {
        f"grid_{name}": np.frombuffer(getattr(scan, name).payload, dtype=np.uint8)
        for name in _GRID_NAMES
    }
    arrays["records"] = np.frombuffer(scan.records, dtype=np.uint8)
    arrays["header"] = np.frombuffer(json.dumps(header).encode("utf-8"), dtype=np.uint8)
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    return buffer.getvalue()


def deserialize_compact_scan(data: bytes) -> CompactScan:
    try:
        with np.load(io.BytesIO(data), allow_pickle=False) as archive:
            header = json.loads(archive["header"].tobytes().decode("utf-8"))
            if header.get("schema_version") != SCHEMA_VERSION:
                raise CompactScanLoadError(
                    f"schema version {header.get('schema_version')!r} is not {SCHEMA_VERSION}"
                )
            grids = {
                name: EncodedGrid(
                    kind=spec["kind"],
                    stored_dtype=spec["stored_dtype"],
                    original_dtype=spec["original_dtype"],
                    shape=tuple(spec["shape"]),
                    payload=archive[f"grid_{name}"].tobytes(),
                    step=spec["step"],
                    offset=spec["offset"],
                )
                for name, spec in header["grids"].items()
            }
            scan = CompactScan(
                site_id=header["site_id"],
                timestamp=datetime.fromisoformat(header["timestamp"]),
                scan_timestamp_text=header["scan_timestamp_text"],
                source_path=header["source_path"],
                object_count=header["object_count"],
                tracked=header["tracked"],
                records=archive["records"].tobytes(),
                **grids,
            )
        scan.validate()
        return scan
    except CompactScanLoadError:
        raise
    except Exception as exc:  # zip, zlib, JSON and shape errors all mean corrupt
        raise CompactScanLoadError(f"{type(exc).__name__}: {exc}") from exc


def save_compact_scan(scan: CompactScan, directory: Path) -> Path:
    path = Path(directory) / history_filename(scan.timestamp)
    atomic_write_bytes(path, serialize_compact_scan(scan))
    return path


def load_compact_scan(path: Path) -> CompactScan:
    try:
        data = Path(path).read_bytes()
    except OSError as exc:
        raise CompactScanLoadError(f"{type(exc).__name__}: {exc}") from exc
    return deserialize_compact_scan(data)
