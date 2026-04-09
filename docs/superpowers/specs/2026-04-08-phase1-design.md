# ARW Phase 1 Design Spec

## Overview

Phase 1 delivers the foundational pipeline: a user enters a city/state, picks a nearby NEXRAD radar, and receives a spoken summary of detected rain objects from the latest (or a historical) scan.

**Architecture:** Python backend (FastAPI) exposes a local REST API. NVGT frontend makes HTTP requests and handles speech output.

## API Endpoints

| Method | Path | Input | Output |
|--------|------|-------|--------|
| `GET` | `/sites` | `?city=...&state=...` | Ranked list of nearby radars (id, name, distance, beam height) |
| `GET` | `/scan/{site_id}` | `?datetime=...` (optional) | Scan metadata: timestamp, elevation angles available |
| `GET` | `/objects/{site_id}` | `?datetime=...` (optional) | Detected rain objects with nested intensity layers |
| `GET` | `/summary/{site_id}` | `?datetime=...` (optional) | Speech-ready text summary |

All responses are JSON. When `datetime` is omitted, the latest scan is used.

## Site Database & Geocoding

**Site database:** A static Python dict of all ~160 NEXRAD WSR-88D sites, each with:
- Site ID (e.g., `KTLX`)
- Name (e.g., `Oklahoma City`)
- Latitude, longitude
- Elevation (meters above sea level)

Data is publicly available and rarely changes — no dynamic fetching needed.

**Geocoding:** geopy with Nominatim converts city/state to lat/lon. No API key required. City-level accuracy is sufficient.

**Ranking radars:** Given the user's lat/lon:
1. Compute great-circle distance to each NEXRAD site
2. Compute beam height at user's location using the beam height formula:
   `beam_height = distance * tan(elevation_angle) + (distance² / (2 * earth_radius)) + radar_elevation`
   - Lowest elevation angle: 0.5°
   - Earth radius: 6371 km
3. Filter out sites where beam height > 10 km
4. Sort by beam height ascending
5. Return the list

## Reflectivity Ingest

**Data source:** NEXRAD Level II files from `s3://noaa-nexrad-level2/` (public, no credentials needed).

**Bucket path structure:** `{YYYY}/{MM}/{DD}/{SITE_ID}/{SITE_ID}{YYYYMMDD}_{HHMMSS}_V06`

**Libraries:** `nexradaws` for bucket listing/download, `pyart` (Py-ART) for Level II parsing.

**Flow:**
1. Ingest Manager receives a request for a site + optional datetime
2. If datetime provided → list files in that day's bucket folder, pick the closest timestamp
3. If no datetime → list the most recent files for that site, pick the latest
4. Download the file to a local cache directory
5. Parse with Py-ART → extract the reflectivity field from the lowest elevation sweep
6. Return a structured object: 2D polar grid of reflectivity values, plus metadata (timestamp, site, elevation angle)

**Caching:** Downloaded files are kept in `cache/` keyed by `{site_id}/{filename}`. Skip download if file exists locally.

**Key constraint:** Only the Ingest Manager makes network calls.

## Rain Object Detection

**Input:** 2D polar reflectivity grid (azimuth × range bins, values in dBZ).

**Step 1 — Threshold & mask:** Filter out everything below 20 dBZ.

**Step 2 — Connected component labeling:** `scipy.ndimage.label` on the masked grid to identify contiguous regions ≥ 20 dBZ. Each connected region becomes a rain object.

**Step 3 — Nested intensity layers:** For each object, identify sub-regions at each threshold:
- 20–30 dBZ: light rain shell
- 30–40 dBZ: moderate rain core
- 40–50 dBZ: heavy rain core
- 50–60 dBZ: intense core
- 60+ dBZ: severe core

Each object gets a layered structure — like contour rings on a topographic map.

**Step 4 — Object properties:** For each object, compute:
- Centroid (polar coords converted to lat/lon)
- Distance and bearing from the radar
- Peak intensity (dBZ) and classification label
- Areal extent (approximate area in km²)
- List of intensity layers present

**Step 5 — Filtering:** Discard objects smaller than `MIN_OBJECT_AREA_KM2` (default: 4 km²) to reduce noise. Defined as a constant in `detection.py`.

## Speech Summaries

**Input:** List of detected rain objects with their properties.

**Output:** Plain-text string for NVGT to pass to TTS/screen reader.

**Format:**
> "{site_name}: {count} rain objects detected. Strongest: {intensity_label}, {distance} miles {bearing} of the radar. Covering approximately {area} square miles."

No objects:
> "{site_name}: No significant precipitation detected."

**Bearing:** Compass degrees converted to 16-point cardinal directions (N, NNE, NE, ENE, E, etc.).

**Distance:** Km converted to miles, rounded to nearest whole number.

The backend returns the text. NVGT handles speaking it — no TTS dependency in Python.

## Project Layout & Dependencies

**Python dependencies:**
- `fastapi` + `uvicorn` — API server
- `pyart` — NEXRAD Level II parsing
- `nexradaws` — NEXRAD AWS bucket listing/download
- `boto3` — S3 access (anonymous)
- `scipy` — connected component labeling
- `numpy` — array operations
- `geopy` — geocoding
- `pytest` — testing

**Source layout:**
```
src/
├── server.py          # FastAPI app, endpoint definitions
├── sites.py           # Site database, geocoding, ranking
├── ingest.py          # NEXRAD download and caching
├── parser.py          # Py-ART parsing, reflectivity extraction
├── detection.py       # Object detection, layering, properties
├── summary.py         # Speech text generation
└── models.py          # Pydantic models for API responses
tests/
├── smoke/             # Server starts, endpoints respond
├── unit/              # Each module in isolation
└── e2e/               # Full flow: city → sites → scan → summary
```

One module per pipeline stage. Pure functions except `ingest.py` (I/O), making each module easy to test in isolation.
