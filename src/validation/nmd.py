"""NWS Level III mesocyclone detection (NMD, product 141) from its tabular text.

The product's alphanumeric table lists each circulation as
  P CIRC  AZRAN   SR STM |-LOW LEVEL-| ... TVS  MOTION   MSI
  P 916  147/110  6  T1  27   34  <15  >21  89  15  27  N   ...   2676
with azimuth in degrees and range in nautical miles from the radar.
"""

from dataclasses import dataclass
from datetime import datetime
import math
import re

import numpy as np
from pyart.core.transforms import cartesian_to_geographic_aeqd

from src.sites import NEXRAD_SITES

KM_PER_NM = 1.852
_TIME = re.compile(rb"DATE:\s*(\d\d)/(\d\d)/(\d{4})\s+TIME:\s*(\d\d):(\d\d):(\d\d)")
_ROW = re.compile(rb"^P\s+(\w+)\s+(\d{1,3})/\s*(\d{1,3})\s+(\d+)[A-Z]?\s+(\w+)\s+(.*)$")


@dataclass(frozen=True)
class MesocycloneDetection:
    circulation_id: str
    azimuth_deg: float
    range_km: float
    strength_rank: int
    storm_id: str
    tvs: bool
    latitude: float
    longitude: float


def parse_nmd(data: bytes, site_id: str) -> tuple[datetime | None, list[MesocycloneDetection]]:
    site = next(s for s in NEXRAD_SITES if s["site_id"] == site_id)
    stamp = _TIME.search(data)
    when = None
    if stamp is not None:
        month, day, year, hour, minute, second = (int(g) for g in stamp.groups())
        when = datetime(year, month, day, hour, minute, second)
    detections: list[MesocycloneDetection] = []
    for line in re.split(rb"[\r\n]+|(?=P {1,2}\d)", data):
        line = line.strip(b"\x00 ")
        match = _ROW.match(line)
        if match is None:
            continue
        circulation, azimuth, range_nm, rank, storm, rest = match.groups()
        tokens = rest.split()
        # rest: RV DV BASE DEPTH STMREL% MAXRV-kft MAXRV-kts TVS ...
        if len(tokens) < 8 or tokens[7] not in (b"Y", b"N"):
            continue
        range_km = int(range_nm) * KM_PER_NM
        bearing = math.radians(int(azimuth))
        lon, lat = cartesian_to_geographic_aeqd(
            np.array([range_km * 1000.0 * math.sin(bearing)]),
            np.array([range_km * 1000.0 * math.cos(bearing)]),
            site["longitude"], site["latitude"],
        )
        detections.append(MesocycloneDetection(
            circulation_id=circulation.decode(),
            azimuth_deg=float(int(azimuth)),
            range_km=range_km,
            strength_rank=int(rank),
            storm_id=storm.decode(),
            tvs=tokens[7] == b"Y",
            latitude=float(np.ravel(lat)[0]),
            longitude=float(np.ravel(lon)[0]),
        ))
    return when, detections
