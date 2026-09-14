"""JSON records for the analysis objects a compact scan keeps.

Every value is tagged, so decoding is unambiguous: dataclasses, datetimes,
tuples and dicts each carry a marker, and plain JSON objects are never
produced for data.  Only registered dataclasses can be decoded, so a record
file can never construct an arbitrary type.  NaN and infinity use Python's
JSON extensions; these files are internal and never served to clients.
"""

from dataclasses import fields, is_dataclass
from datetime import datetime
import json
from typing import Any

import numpy as np

from src.buffer import TrackingSnapshot
from src.detection import DetectedObject, IntensityLayerData
from src.preprocess import ScanQuality
from src.tracking.motion import MotionVector
from src.tracking.types import (
    FocusContinuity,
    IdentityConfidence,
    MotionConfidence,
    MotionSample,
    PeakEntry,
    RotationHistoryEntry,
    Track,
    TrackPosition,
    TrendSample,
    VelocitySample,
)
from src.velocity import RotationSignature, VelocityRegion

_RECORD_TYPES: dict[str, type] = {}


def register_record_type(cls: type) -> type:
    if not is_dataclass(cls):
        raise TypeError(f"{cls.__name__} is not a dataclass")
    _RECORD_TYPES[cls.__name__] = cls
    return cls


for _cls in (
    DetectedObject, IntensityLayerData, RotationSignature, ScanQuality,
    TrackingSnapshot, Track, TrackPosition, PeakEntry, IdentityConfidence,
    FocusContinuity, MotionConfidence, MotionSample, RotationHistoryEntry,
    MotionVector, TrendSample, VelocitySample, VelocityRegion,
):
    register_record_type(_cls)


def to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value)
    if isinstance(value, datetime):
        return {"__datetime__": value.isoformat()}
    if isinstance(value, tuple):
        return {"__tuple__": [to_jsonable(item) for item in value]}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {"__dict__": [[to_jsonable(k), to_jsonable(v)] for k, v in value.items()]}
    if is_dataclass(value) and not isinstance(value, type):
        name = type(value).__name__
        if _RECORD_TYPES.get(name) is not type(value):
            raise TypeError(f"{name} is not a registered record type")
        return {
            "__type__": name,
            "fields": {f.name: to_jsonable(getattr(value, f.name)) for f in fields(value)},
        }
    raise TypeError(f"cannot encode {type(value).__name__} as a record")


def from_jsonable(data: Any) -> Any:
    if isinstance(data, list):
        return [from_jsonable(item) for item in data]
    if not isinstance(data, dict):
        return data
    if set(data) == {"__datetime__"}:
        return datetime.fromisoformat(data["__datetime__"])
    if set(data) == {"__tuple__"}:
        return tuple(from_jsonable(item) for item in data["__tuple__"])
    if set(data) == {"__dict__"}:
        return {from_jsonable(k): from_jsonable(v) for k, v in data["__dict__"]}
    if set(data) == {"__type__", "fields"}:
        cls = _RECORD_TYPES.get(data["__type__"])
        if cls is None:
            raise ValueError(f"unregistered record type {data['__type__']!r}")
        return cls(**{name: from_jsonable(item) for name, item in data["fields"].items()})
    raise ValueError(f"unrecognized record structure with keys {sorted(data)}")


def dumps(value: Any) -> str:
    return json.dumps(to_jsonable(value), allow_nan=True, separators=(",", ":"))


def loads(text: str) -> Any:
    return from_jsonable(json.loads(text))
