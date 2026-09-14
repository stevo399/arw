"""Build the rotation validation corpus (spec 2026-09-14, A1).

Cases:
- every SPC tornado report on SPC_DAYS, nearest WSR-88D within 150 km,
  volumes from 20 min before to 10 min after the report;
- the 6 largest hail (>= 2.00 in) and 6 strongest wind (>= 65 kt) reports on
  the same days, volumes from 10 min before to 5 min after;
- clear air: the existing six KIWA 2026-07-12 volumes, plus for each tornado
  radar the volume nearest 12:00Z on its SPC day when no report of any kind
  lies within 200 km within 3 h.  Such a volume is clear air only when under
  1% of its valid gates reach 35 dBZ (spec B1); otherwise it is a
  report-free storm case ("null_storm"), scored only by null-storm probes.
Each case gets the NWS NMD products within 5 minutes of its window.
Network access goes through src.ingest only.
"""
import json
import sys
from dataclasses import asdict
from datetime import date, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ingest import download_level3, fetch_scans_between, fetch_spc_reports, list_level3_keys
from src.sites import NEXRAD_SITES, haversine_distance_km
from src.validation.spc import parse_spc_csv

SPC_DAYS = [date(2026, 7, 12), date(2026, 7, 14), date(2026, 9, 7), date(2026, 9, 11), date(2026, 9, 12)]
MAX_RADAR_RANGE_KM = 150.0
SEVERE_PER_KIND = 6
SEVERE_FLOORS = {"hail": 2.0, "wind": 65.0}
CLEAR_AIR_MAX_STRONG_SHARE = 0.01
CLEAR_AIR_STRONG_DBZ = 35.0
KIWA_CLEAR_AIR = [f"cache/KIWA/KIWA20260712_{t}_V06" for t in ("163410", "164255", "165142", "170029", "170914", "171800")]
OUT = ROOT / "docs/validation/rotation-signal-corpus-v2.json"


def nearest_site(lat, lon):
    distance, site = min((haversine_distance_km(s["latitude"], s["longitude"], lat, lon), s["site_id"]) for s in NEXRAD_SITES)
    return (site, distance) if distance <= MAX_RADAR_RANGE_KM else (None, distance)


def relative(path):
    return str(Path(path).resolve().relative_to(ROOT)).replace("\\", "/")


def report_dict(report):
    row = asdict(report)
    row["time_utc"] = report.time_utc.isoformat()
    return row


def case_for(report, before, after, label, index):
    site, distance = nearest_site(report.latitude, report.longitude)
    if site is None:
        print(f"skip {label} {report.time_utc} {report.location} {report.state}: nearest radar {distance:.0f} km", flush=True)
        return None
    start, end = report.time_utc - before, report.time_utc + after
    volumes = [relative(p) for p in fetch_scans_between(site, start, end)]
    nmd = [relative(download_level3(k)) for k in list_level3_keys(site, "NMD", start - timedelta(minutes=5), end + timedelta(minutes=5))]
    print(f"{label} {report.time_utc} {report.location} {report.state}: {site} {distance:.0f} km, "
          f"{len(volumes)} volumes, {len(nmd)} NMD", flush=True)
    return {"id": f"{label}-{report.time_utc:%Y%m%d-%H%M}-{index}", "label": label, "site_id": site,
            "radar_distance_km": round(distance, 1), "report": report_dict(report), "volumes": volumes, "nmd": nmd}


def strong_echo_share(path):
    import numpy as np
    from src.parser import extract_sweep_data, parse_radar_file

    reflectivity = np.asarray(extract_sweep_data(parse_radar_file(str(path))).reflectivity, dtype=float)
    valid = np.isfinite(reflectivity)
    return float(np.count_nonzero(reflectivity >= CLEAR_AIR_STRONG_DBZ)) / max(int(np.count_nonzero(valid)), 1)


def spc_noon(when):
    """12:00Z at the start of the SPC day containing `when`."""
    noon = datetime(when.year, when.month, when.day, 12, 0)
    return noon if when >= noon else noon - timedelta(days=1)


def main():
    reports = []
    for day in SPC_DAYS:
        for kind in ("tornado", "hail", "wind"):
            text = Path(fetch_spc_reports(day, kind)).read_text(encoding="utf-8", errors="replace")
            reports.extend(parse_spc_csv(text, day, kind))
    print("reports:", {kind: sum(r.kind == kind for r in reports) for kind in ("tornado", "hail", "wind")}, flush=True)

    cases = []
    for index, report in enumerate(r for r in reports if r.kind == "tornado"):
        case = case_for(report, timedelta(minutes=20), timedelta(minutes=10), "tornado", index)
        if case:
            cases.append(case)
    for kind, floor in SEVERE_FLOORS.items():
        strongest = sorted(
            (r for r in reports if r.kind == kind and r.magnitude is not None and r.magnitude >= floor),
            key=lambda r: (-r.magnitude, r.time_utc),
        )[:SEVERE_PER_KIND]
        for index, report in enumerate(strongest):
            case = case_for(report, timedelta(minutes=10), timedelta(minutes=5), kind, index)
            if case:
                cases.append(case)

    cases.append({"id": "clear-air-KIWA-20260712", "label": "clear_air", "site_id": "KIWA", "radar_distance_km": None,
                  "report": None, "volumes": KIWA_CLEAR_AIR, "nmd": []})
    for case in [c for c in cases if c["label"] == "tornado"]:
        noon = spc_noon(datetime.fromisoformat(case["report"]["time_utc"]))
        case_id = f"quiet-{case['site_id']}-{noon:%Y%m%d}"
        if any(c["id"] == case_id for c in cases):
            continue
        site = next(s for s in NEXRAD_SITES if s["site_id"] == case["site_id"])
        busy = any(
            abs((r.time_utc - noon).total_seconds()) <= 3 * 3600
            and haversine_distance_km(site["latitude"], site["longitude"], r.latitude, r.longitude) <= 200.0
            for r in reports
        )
        if busy:
            print(f"no clear-air volume for {case['site_id']} at {noon}: reports within 200 km and 3 h", flush=True)
            continue
        paths = fetch_scans_between(case["site_id"], noon - timedelta(minutes=5), noon + timedelta(minutes=5))
        if paths:
            share = strong_echo_share(paths[0])
            label = "clear_air" if share < CLEAR_AIR_MAX_STRONG_SHARE else "null_storm"
            nmd = [relative(download_level3(k)) for k in list_level3_keys(case["site_id"], "NMD", noon - timedelta(minutes=10), noon + timedelta(minutes=10))]
            cases.append({"id": case_id, "label": label, "site_id": case["site_id"], "radar_distance_km": None,
                          "strong_echo_share": round(share, 4), "report": None, "volumes": [relative(paths[0])], "nmd": nmd})
            print(f"{label} {case['site_id']} {noon}: {Path(paths[0]).name}, {100 * share:.2f}% of gates >= 35 dBZ", flush=True)

    OUT.write_text(json.dumps({
        "schema_version": 2,
        "built_utc": datetime.utcnow().isoformat(timespec="seconds"),
        "purpose": "Rotation detection validation (docs/superpowers/specs/2026-09-14-detection-truth-design.md, A1). "
                   "Report locations and NWS detections describe events, not per-gate truth.",
        "spc_days": [d.isoformat() for d in SPC_DAYS],
        "cases": cases,
        "reports": [report_dict(r) for r in reports],
    }, indent=2), encoding="utf-8")
    print("wrote", OUT, "cases", len(cases), flush=True)


if __name__ == "__main__":
    main()
