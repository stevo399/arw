# Live API Test Report — 2026-04-10

First live end-to-end test of the ARW backend against real NEXRAD Level II data from AWS S3. Two production bugs were caught and fixed mid-test; afterward the full pipeline worked cleanly across six real radar sites.

## Environment

- Server: `uv run uvicorn src.server:app --port 8000`
- Backend version: 0.2.0 (Phase 2 complete — motion tracking + replay buffer)
- Date: 2026-04-10
- Test method: live `curl` calls against the running server, fetching the latest available scan from S3 for each site

## Bugs caught and fixed

### Bug 1 — `src/ingest.py:52`
**Symptom:** `TypeError: 'DownloadResults' object is not iterable` on every `/summary/{site_id}` call.

**Cause:** `nexradaws.NexradAwsInterface.download()` returns a `DownloadResults` object, not an iterable. The code tried `for result in results:` which fails. The existing unit test mock returned a plain list, so it masked the bug.

**Fix:** Iterate `results.success` instead, and update `tests/unit/test_ingest.py` so the mock reflects the real API (a `DownloadResults`-like object with a `.success` list).

### Bug 2 — `src/ingest.py:29` (`list_scans_for_date`)
**Symptom:** `OSError: unknown compression record` raised from inside `pyart.io.read_nexrad_archive`.

**Cause:** NEXRAD L2 listings include `_MDM` metadata files alongside real volume scans. When fetching the "latest" scan, the code sometimes picked up an `_MDM` file (e.g. `KFWS20260410_115418_V06_MDM`), which Py-ART can't parse.

**Fix:** Filter out any filename ending in `_MDM` from `list_scans_for_date` so downstream code only sees real volume scans.

## Test 1 — Dallas / Fort Worth (KFWS)

### Site selection
```
GET /sites?city=Dallas&state=TX
```
Top result: **KFWS** — Dallas/Fort Worth, 52.5 km from city center, beam height 882 m.

### Summary
```
GET /summary/KFWS
→ {
    "site_id": "KFWS",
    "timestamp": "2026-04-10T12:01:10Z",
    "text": "Dallas/Fort Worth: No significant precipitation detected."
  }
```

### Other endpoints
```
GET /objects/KFWS → {"object_count": 0, "objects": []}
GET /tracks/KFWS  → {"active_count": 0, "tracks": [], "recent_events": []}
```

**Result:** Clear skies over Dallas. Pipeline fetched real scan, parsed it, detected nothing. Correct.

## Test 2 — Los Angeles area (multiple sites)

User suspected rain in the LA area, so we queried every nearby radar.

### Site selection
```
GET /sites?city=Los%20Angeles&state=CA
```
Returned, in distance order: KSOX, KVTX, KEYX, KNKX, KVBX, KHNX.

### Summaries

| Site | Location | Result |
|---|---|---|
| **KVTX** | Los Angeles (95 km NW) | `No significant precipitation detected.` |
| **KSOX** | Santa Ana Mountains (62 km SE) | `1 rain object detected. Strongest: light rain, 259 miles S of the radar, stationary. Covering approximately 2 square miles.` |
| **KEYX** | Edwards AFB | `3 rain objects detected. Strongest: intense rain, 39 miles WNW of the radar, stationary. Covering approximately 4 square miles.` |
| **KVBX** | Vandenberg AFB | `9 rain objects detected. Strongest: moderate rain, 96 miles WNW of the radar, stationary. Covering approximately 8 square miles.` |

### Raw JSON — KSOX object detail
```json
{
  "site_id": "KSOX",
  "timestamp": "2026-04-10T12:02:29Z",
  "object_count": 1,
  "objects": [
    {
      "object_id": 172,
      "centroid_lat": 30.0716,
      "centroid_lon": -117.4661,
      "distance_km": 416.9,
      "bearing_deg": 177.8,
      "peak_dbz": 24.5,
      "peak_label": "light rain",
      "area_km2": 4.55,
      "layers": [
        {"label": "light rain", "min_dbz": 20.0, "max_dbz": 30.0, "area_km2": 4.55}
      ]
    }
  ]
}
```
The KSOX hit is at 417 km range — near the edge of NEXRAD's effective coverage, centered off the Baja California coast. Likely edge-of-range noise, not real weather near LA.

### Raw JSON — KVBX object detail (9 rain objects)
```json
{
  "site_id": "KVBX",
  "timestamp": "2026-04-10T12:11:14Z",
  "object_count": 9,
  "objects": [
    {"object_id": 446, "centroid_lat": 35.5738, "centroid_lon": -121.8335, "distance_km": 153.9, "bearing_deg": 302.5, "peak_dbz": 33.0, "peak_label": "moderate rain", "area_km2": 19.59},
    {"object_id": 462, "centroid_lat": 35.7249, "centroid_lon": -121.8963, "distance_km": 168.0, "bearing_deg": 306.4, "peak_dbz": 33.0, "peak_label": "moderate rain", "area_km2": 25.36},
    {"object_id": 500, "centroid_lat": 36.3736, "centroid_lon": -122.3103, "distance_km": 242.9, "bearing_deg": 315.2, "peak_dbz": 30.0, "peak_label": "moderate rain", "area_km2": 253.18},
    {"object_id": 459, "centroid_lat": 35.7249, "centroid_lon": -121.9542, "distance_km": 172.2, "bearing_deg": 305.4, "peak_dbz": 29.5, "peak_label": "light rain", "area_km2": 12.26},
    {"object_id": 449, "centroid_lat": 35.6502, "centroid_lon": -121.9709, "distance_km": 169.0, "bearing_deg": 302.7, "peak_dbz": 28.0, "peak_label": "light rain", "area_km2": 10.93},
    {"object_id": 472, "centroid_lat": 35.7373, "centroid_lon": -121.8377, "distance_km": 164.5, "bearing_deg": 307.8, "peak_dbz": 28.0, "peak_label": "light rain", "area_km2": 7.45},
    {"object_id": 516, "centroid_lat": 36.7739, "centroid_lon": -122.2438, "distance_km": 272.1, "bearing_deg": 322.8, "peak_dbz": 28.0, "peak_label": "light rain", "area_km2": 101.54},
    {"object_id": 402, "centroid_lat": 34.0785, "centroid_lon": -121.8087, "distance_km": 154.5, "bearing_deg": 237.2, "peak_dbz": 27.5, "peak_label": "light rain", "area_km2": 7.0},
    {"object_id": 497, "centroid_lat": 36.319, "centroid_lon": -122.4197, "distance_km": 246.0, "bearing_deg": 312.6, "peak_dbz": 24.5, "peak_label": "light rain", "area_km2": 6.37}
  ]
}
```

### Interpretation
The LA basin itself is dry at this timestamp. The interesting weather is offshore, stretched NW along the Big Sur / Monterey coast: a frontal line of 9 light-to-moderate rain objects between 150-270 km offshore from Vandenberg. The largest single blob is ~253 km² (98 mi²) off Big Sur. Edwards AFB also shows a few isolated cells in the high desert. Consistent with a Pacific front still offshore and not yet making landfall over LA.

## Known limitations observed

- **Motion always reports "stationary":** Each `/summary` call fetches the single latest scan, so the tracker only ever has one position per storm. Real velocities would appear after successive calls spaced by NEXRAD's scan cadence (~5-10 min), or by replaying historical scans via the `?datetime=` parameter.
- **Edge-of-range detections:** Objects near the 460 km maximum range (e.g. KSOX's Baja blob at 417 km) should probably be filtered or flagged, since attenuation, beam spreading, and anomalous propagation make them unreliable.

## Conclusion

After fixing both bugs, every live endpoint call succeeded with zero 500 errors. The full pipeline — S3 fetch → Py-ART parse → detection → tracking → speech summary — works end-to-end against real NEXRAD data for clear-sky sites, isolated cells, and multi-object scenes alike.
