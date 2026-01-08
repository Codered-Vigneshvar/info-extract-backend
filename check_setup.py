import os
import dotenv
import sys
import logging
import traceback

# Force unbuffered stdout/stderr
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

print("Starting checks...", flush=True)

try:
    dotenv.load_dotenv()
    print("Dotenv loaded.", flush=True)

    PROJ = os.getenv("GOOGLE_CLOUD_PROJECT")
    LOC = os.getenv("GOOGLE_CLOUD_LOCATION")
    print(f"PROJECT: {PROJ}", flush=True)
    print(f"LOCATION: {LOC}", flush=True)

    print("Importing agent...", flush=True)
    from agent import _ensure_model, _ensure_easyocr_reader
    print("Imports OK.", flush=True)

    print("Checking EasyOCR...", flush=True)
    _ensure_easyocr_reader()
    print("EasyOCR OK.", flush=True)

    print("Checking Vertex AI...", flush=True)
    model = _ensure_model()
    print("Vertex AI Model initialized.", flush=True)

except Exception:
    traceback.print_exc()
