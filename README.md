# Label Extractor

FastAPI backend, web UI, and Expo mobile app for extracting structured label data from product/package images using Gemini (Vertex AI). Includes quality gating, OCR (EasyOCR default), LLM extraction, verification against OCR, result storage, and metrics.

## Table of Contents
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Configuration](#configuration)
- [Running the Backend](#running-the-backend)
- [Web UI](#web-ui)
- [Mobile App (Expo)](#mobile-app-expo)
- [API Endpoints](#api-endpoints)
- [Data & Metrics](#data--metrics)
- [Quality, OCR, LLM](#quality-ocr-llm)
- [File Map](#file-map)
- [Troubleshooting](#troubleshooting)

## Architecture
- Backend: FastAPI (`main.py`) orchestrates quality check → enhancement (OCR-only) → OCR → LLM extraction → verification → persistence.
- Vision/OCR: `agent.py` with EasyOCR (default) or Tesseract (optional); OpenCV preprocessing and quality metrics.
- Frontend: Static web app served from `public/`.
- Mobile: Expo React Native client (`mobile/hdsupply-label-extractor`) calling the same API.
- Storage: `storage/images/` (original uploads), `storage/results/*.json` (per-run output), `metrics/metrics_log.txt` (per extract run).

## Prerequisites
- Python 3.8+ (Vertex AI SDK warns <3.10; prefer 3.10+).
- Node.js + npm (for mobile/web dev tooling; not required for backend-only).
- Google Cloud project with Vertex AI access; ADC credentials set up (`gcloud auth application-default login` or `GOOGLE_APPLICATION_CREDENTIALS`).

## Configuration
Create `.env` (see sample values):
```
GOOGLE_CLOUD_PROJECT=your-project
GOOGLE_CLOUD_LOCATION=your-region
GEMINI_MODEL=gemini-2.0-flash
LLM_PROVIDER=vertex
OCR_ENGINE=easyocr          # or tesseract
EASYOCR_LANGS=en            # comma-separated langs
# Quality thresholds (defaults shown)
QUALITY_MIN_BLUR_SCORE=70
QUALITY_MIN_BRIGHTNESS=20
QUALITY_MAX_BRIGHTNESS=245
QUALITY_MIN_W=600
QUALITY_MIN_H=600
```

## Running the Backend
```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1  # or source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```
Backend serves the web UI at `/` and APIs under `/api/*`.

## Web UI
- Open `http://localhost:8000`.
- Flow: upload image → quality status → extract → view/edit results, history, metrics.

## Mobile App (Expo)
- Path: `mobile/hdsupply-label-extractor`.
- Run:
  ```bash
  cd mobile/hdsupply-label-extractor
  npm install
  npx expo start
  ```
- Default backend URL: `http://10.0.2.2:8000` (Android emulator). Set to your host/IP in-app for device testing.

## API Endpoints
- `GET /` — Web UI.
- `POST /api/quality` — multipart `image`; returns quality metrics (no metrics log).
- `POST /api/extract` — multipart `image` (and optional `quality_*`); runs full pipeline, saves run, logs metrics.
- `POST /api/save_edits` — body `{source_id, edited_fields}`; saves a new run copy with edits.
- `GET /api/history` — list runs.
- `GET /api/history/{id}` — run details.
- `GET /api/metrics` — metrics log (header + rows) from `metrics/metrics_log.txt`.
- Static assets at `/public/*`.

## Data & Metrics
- Results: `storage/results/<run_id>.json` (fields, raw_json, quality, ocr, enhancement_meta, verification info, llm_usage).
- Images: `storage/images/<run_id>_*.jpg`.
- Metrics: `metrics/metrics_log.txt` (tab-delimited, one row per extract).

## Quality, OCR, LLM
- Quality: adaptive blur on label ROI (Laplacian), brightness/contrast/resolution checks; no hard stop (continues even if “hard_reject” flagged).
- Enhancements (OCR-only): conditional gamma/brightness scaling, CLAHE, mild unsharp; safety fallback to original.
- OCR: EasyOCR by default (set `OCR_ENGINE=tesseract` to switch).
- LLM: Gemini via Vertex AI, prompt from `prompt.txt` plus guidance to return numeric/unit/currency for price/quantity.
- Verification: caps confidence and flags when OCR text doesn’t support extracted values (keyword proximity for key fields).

## File Map
- `main.py` — FastAPI app, endpoints, pipeline, metrics logging.
- `agent.py` — OCR, quality, enhancement, verification, LLM helpers.
- `prompt.txt` — LLM prompt body (augmented in code with guidance).
- `public/` — Web UI assets.
- `docs/app_overview.txt` — High-level behavior.
- `docs/code_walkthrough.txt` — Detailed file-by-file explanation.
- `storage/` — Runtime images/results (generated).
- `metrics/metrics_log.txt` — Runtime metrics log (generated).
- `mobile/hdsupply-label-extractor/` — Expo app (App.tsx, src/*).
- `requirements.txt` — Backend dependencies.

## Troubleshooting
- Metrics empty: metrics_log.txt is created only on `/api/extract` success; if missing, you’ll see header + empty rows. Run an extract to log.
- Quality “blurry” on clean labels: adaptive ROI blur is in place; if still an issue, adjust `QUALITY_MIN_BLUR_SCORE`.
- OCR engine issues: set `OCR_ENGINE=tesseract` or EasyOCR; ensure the binary/backends are installed.
- Vertex errors: verify ADC credentials and env (`GOOGLE_CLOUD_PROJECT`/`LOCATION`/`GEMINI_MODEL`). Upgrade Python to 3.10+ to avoid api_core warnings.
