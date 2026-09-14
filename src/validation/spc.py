"""SPC storm reports (tornado, hail, wind) parsed from the daily CSV files.

The SPC report day runs 12Z to 12Z: a report time before 1200 belongs to the
next UTC calendar date.
"""

from dataclasses import dataclass
from datetime import date, datetime, timedelta
import csv
import io
import re

MAGNITUDE_COLUMN = {"tornado": "F_Scale", "hail": "Size", "wind": "Speed"}


@dataclass(frozen=True)
class StormReport:
    kind: str
    time_utc: datetime
    latitude: float
    longitude: float
    magnitude: float | None  # tornado EF rating, hail inches, wind knots
    location: str
    state: str


def _magnitude(kind: str, raw: str) -> float | None:
    digits = re.search(r"\d+", raw or "")
    if digits is None:
        return None
    value = float(digits.group())
    return value / 100.0 if kind == "hail" else value


def parse_spc_csv(text: str, spc_day: date, kind: str) -> list[StormReport]:
    reports: list[StormReport] = []
    for row in csv.DictReader(io.StringIO(text)):
        hhmm = (row.get("Time") or "").strip()
        if not hhmm.isdigit() or len(hhmm) > 4:
            continue
        hhmm = hhmm.zfill(4)
        when = datetime(spc_day.year, spc_day.month, spc_day.day, int(hhmm[:2]), int(hhmm[2:]))
        if when.hour < 12:
            when += timedelta(days=1)
        reports.append(StormReport(
            kind=kind,
            time_utc=when,
            latitude=float(row["Lat"]),
            longitude=float(row["Lon"]),
            magnitude=_magnitude(kind, row.get(MAGNITUDE_COLUMN[kind], "")),
            location=row.get("Location", ""),
            state=row.get("State", ""),
        ))
    return reports
