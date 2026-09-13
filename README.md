# ARW — Accessible Radar Workstation

A standalone application enabling blind users to analyze NEXRAD weather radar data through keyboard navigation, speech output, and spatialized audio.

## What It Does

ARW ingests live NEXRAD Level II radar data from AWS Open Data, detects rain objects and storm cells, tracks their motion across scans, and produces spoken summaries describing the weather scene — all designed for non-visual use.

**Example output:**
> Oklahoma City: 57 rain objects detected. Strongest: severe core, 31 miles E of the radar, moving SE at 8 mph. Note: 11 storms merged in the last scan. Covering approximately 3600 square miles.

## Architecture

```
NEXRAD (AWS S3)
  → Ingest Manager
  → Cache Store
  → Product Parsers (reflectivity)
  → Object Detection (multilevel segmentation)
  → Tracking Engine (association, motion field, confidence)
  → Hazard Analysis (planned)
  → Scene Model
  → Speech Manager + Audio Renderer
  → Web App
```

Key rule: **only the Ingest Manager makes network calls.**

## Current Status

- **Phase 1** — Complete: site database, reflectivity ingest, rain object detection, speech summaries
- **Phase 2** — Complete: motion tracking with segmentation, association, confidence-aware motion, focus stability, and live replay validation
- **Phase 3** — Complete: velocity ingestion, multi-sweep extraction, inbound/outbound region detection, rotation signature detection, rotation persistence tracking (195 tests passing)
- **Phase 4** — Planned: hail detection, debris scoring
- **Phase 5** — Planned: spatial audio scene
- **Phase 6** — Planned: native web app frontend

## Tech Stack

- **Language:** Python
- **Data Source:** NEXRAD Level II via `s3://noaa-nexrad-level2/`
- **Radar Processing:** Py-ART
- **API:** FastAPI

## Project Structure

```
arw/
├── src/
│   ├── ingest/       # Data ingestion from NEXRAD
│   ├── cache/        # Local data caching
│   ├── parsers/      # Radar product parsers
│   ├── detection/    # Object detection and extraction
│   ├── tracking/     # Storm motion tracking
│   ├── hazards/      # Hail, debris, rotation analysis
│   ├── scene/        # Scene model for audio/speech
│   ├── audio/        # Spatialized audio renderer
│   └── speech/       # Speech output manager
├── tests/            # Smoke, unit, and e2e tests
├── scripts/          # Replay and evaluation harnesses
├── docs/             # Specs and test reports
└── cache/            # Local NEXRAD scan cache
```

## Running

```bash
# Install dependencies
uv sync

# Run the API server
uv run uvicorn src.server:app

# Run tests
uv run pytest tests/ -q

# Run a local replay (requires cached data)
uv run python scripts/live_replay.py KTLX --date 2026-04-10 --scans 8 --local-only
```

### Live map prewarming

Map visitors receive the most recent completed interpretation immediately while
ARW checks for a newer volume in the background. Configure the radar sites to
keep warm when starting the server; use a comma-separated list of NEXRAD IDs.

```powershell
$env:ARW_PREWARM_SITES = "KIWA"
$env:ARW_REFRESH_INTERVAL_SECONDS = "120" # optional; default is 120 seconds
$env:ARW_REFRESH_WORKERS = "2" # independent sites may prepare in parallel
uv run uvicorn src.server:app --host 127.0.0.1 --port 8000
```

`GET /live/KIWA/status` reports whether a completed interpretation is
available, its scan timestamp, and whether a refresh is running. Map GeoJSON
also includes `metadata.scanTimestamp` and `metadata.refreshState`.

Prewarming is optional. Without it, any requested location is prepared on
demand; duplicate requests for one radar site share work, while unrelated
sites can prepare concurrently.

### Radar scan history

Each radar keeps its most recent tracked scans as compact, lossless records
(reflectivity in its Level II 0.5 dBZ encoding, one object label grid, and the
objects, evidence and tracking state from that scan's own time). They are held
in memory and written to `cache/<SITE>/history/`, so storm identities, ages and
reacquisition survive switching radars and restarting the server.

```powershell
$env:ARW_HISTORY_ROOT = "cache"          # where per-radar history is written
$env:ARW_SCANS_PER_RADAR = "5"           # retained tracked scans per radar
$env:ARW_MAX_RADARS_IN_MEMORY = "20"     # idle radars beyond this reload from disk
$env:ARW_MAX_HISTORICAL_SCANS = "12"     # specific-time scans kept compactly, never tracked
$env:ARW_MAX_RENDERED_SCANS = "10"       # rendered map layers kept besides each radar's newest
```

`GET /map/history?latitude=..&longitude=..` (or `city`/`state`, or `zipcode`)
lists the retained scans for the radar ARW selects, newest first. Each
timestamp can be passed back unchanged as `datetime=` to the map, status,
objects, tracks and summary endpoints to select exactly that scan, served with
its own tracking context. `/map/status?datetime=` is ready immediately for a
retained scan and prepares any other time in the background.

A storm that is not detected in a scan becomes `missing`: it is not described
or listed, but can be reacquired for up to 20 minutes while its last-seen scan
is retained, and is marked `lost` after that.

## License

This project is not yet licensed. All rights reserved.
