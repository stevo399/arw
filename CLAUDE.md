# Accessible Radar Workstation (ARW)

A standalone Windows application enabling blind users to analyze NEXRAD weather radar data through keyboard navigation, speech output, and spatialized audio.

## Tech Stack

- **Language:** Python
- **UI Framework:** NVGT (keyboard-driven, accessible)
- **Data Source:** NEXRAD Level II via AWS Open Data (`s3://noaa-nexrad-level2/`)
- **Audio:** Spatialized audio for spatial scene representation

## Architecture

Pipeline: Ingest Manager → Cache Store → Product Parsers → Object Extraction → Tracking Engine → Hazard Analysis → Scene Model → Speech Manager + Audio Renderer → NVGT UI

Key rule: **only the Ingest Manager makes network calls.**

## Project Structure

```
arw/
├── devspec/          # Product requirements and technical specifications
├── src/              # Application source code
│   ├── ingest/       # Data ingestion from NEXRAD
│   ├── cache/        # Local data caching
│   ├── parsers/      # Radar product parsers
│   ├── detection/    # Object detection and extraction
│   ├── tracking/     # Storm motion tracking
│   ├── hazards/      # Hail, debris, rotation analysis
│   ├── scene/        # Scene model for audio/speech
│   ├── audio/        # Spatialized audio renderer
│   ├── speech/       # Speech output manager
│   └── ui/           # NVGT keyboard UI
├── tests/            # All tests
│   ├── smoke/        # Smoke tests
│   ├── unit/         # Unit tests
│   └── e2e/          # End-to-end tests
└── CLAUDE.md
```

## Development Rules

- **Commit frequently and sensibly.** After each change or feature, run tests and commit.
- **Test everything.** Write smoke, unit, and end-to-end tests for all functionality.
- **Run tests before committing.** No commit without green tests for the affected code.
- **Accessibility first.** Every feature must be fully operable via keyboard and speech feedback.
- **Track progress at end of session.** When development work is done for a session, update `PROGRESS.md` at the project root with: what was completed, what is in progress, what comes next, and any blockers or decisions pending. This ensures any developer or agent picking up the project knows exactly where the last session stopped.

## NVGT Reference Sources

When building the NVGT frontend, reference these local resources:

- **NVGT includes** — `C:\nvgt\include` — standard includes like `form.nvgt` for form-based UI
- **NVGT repository** — `C:\Users\steve\Documents\nvgt` — C++ source, documentation, engine dumps of all wrapped functions and classes
- **tiles2 (working game example)** — `C:\Users\steve\Documents\tiles2` — form examples, non-form UI, sound management, and HTTP client patterns (in `utils/`) for calling JSON APIs

## Development Phases

1. Site database, reflectivity ingest, rain object detection, speech summaries
2. Motion tracking, replay buffer
3. Velocity ingestion, velocity region detection
4. Hail detection, debris scoring
5. Spatial audio scene
6. Native web app frontend
