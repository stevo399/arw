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

## License

This project is not yet licensed. All rights reserved.
