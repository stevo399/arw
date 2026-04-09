
# Technical Architecture

High level pipeline:

Weather Sources
↓
Ingest Manager
↓
Cache Store
↓
Product Parsers
↓
Object Extraction Engine
↓
Tracking Engine
↓
Hazard Analysis Engine
↓
Scene Model
↓
Speech Manager + Audio Renderer
↓
NVGT UI

Key principle: only the ingest manager makes network calls.
