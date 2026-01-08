import dotenv
dotenv.load_dotenv()

import json
import os
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import time

# Shim importlib.metadata on Python 3.8 to provide packages_distributions
try:
    import importlib.metadata as _ilm  # type: ignore

    if not hasattr(_ilm, "packages_distributions"):
        import importlib_metadata as _ilm_backport  # type: ignore
        import sys

        sys.modules["importlib.metadata"] = _ilm_backport
except Exception:
    pass

import cv2
import numpy as np

from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import logging
import csv

from agent import (
    AgentConfigError,
    build_ocr_text_block,
    compute_image_quality,
    extract_label_json,
    analyze_quality,
    enhance_for_ocr_if_needed,
    run_ocr,
    verify_against_ocr,
)






# Configure logging with ISO8601-like timestamp and stdout
import sys
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout
)

logger = logging.getLogger("label-extractor")
# Enable verbose logging for google ADK
adk_logger = logging.getLogger("google.adk")
adk_logger.setLevel(logging.DEBUG)

ROOT = Path(__file__).parent.resolve()

PROMPT_FILE = ROOT / "prompt.txt"

# Pre-load prompt template
try:
    CACHED_PROMPT_TEXT = PROMPT_FILE.read_text(encoding="utf-8")
except FileNotFoundError:
    CACHED_PROMPT_TEXT = ""  # Will be handled in request
STORAGE_ROOT = ROOT / "storage"
IMAGES_DIR = STORAGE_ROOT / "images"
RESULTS_DIR = STORAGE_ROOT / "results"
METRICS_DIR = ROOT / "metrics"
METRICS_FILE = METRICS_DIR / "metrics_log.txt"
METRICS_COLUMNS = [
    "run_id",
    "timestamp",
    "quality_pass",
    "blur_score",
    "brightness",
    "ocr_avg_conf",
    "ocr_num_lines",
    "sure_count",
    "unsure_count",
    "verification_failures",
    "validation_failures",
    "prompt_tokens",
    "candidates_tokens",
    "total_tokens",
    "total_latency_ms",
    "package_name_conf", "package_name_status",
    "brand_name_conf", "brand_name_status",
    "mfg_date_conf", "mfg_date_status",
    "exp_date_conf", "exp_date_status",
    "net_quantity_conf", "net_quantity_status",
    "mrp_price_conf", "mrp_price_status",
    "batch_lot_no_conf", "batch_lot_no_status",
]


# Extra guidance injected for numeric/unit handling without editing prompt.txt
LLM_ADDITIONAL_GUIDANCE = """
When returning JSON:
- For "mrp_price": include both the raw string value as seen and a numeric field `value_num` (float) and a `currency` code/symbol (e.g., "INR" for ₹/Rs/INR, "USD" for $). Example: {"value": "₹ 5700", "value_num": 5700, "currency": "INR"}.
- For "net_quantity": include both the raw string value as seen and a numeric field `value_num` and a `unit` string (e.g., "N", "ml", "g"). Example: {"value": "1 N", "value_num": 1, "unit": "N"}.
Keep the existing fields and schema for all other fields.
"""

EXPECTED_FIELDS = [
    "package_name",
    "brand_name",
    "mfg_date",
    "exp_date",
    "net_quantity",
    "mrp_price",
    "batch_lot_no",
]

for path in (STORAGE_ROOT, IMAGES_DIR, RESULTS_DIR):
    path.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Label Extractor")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)








@app.post("/api/quality")
async def quality(image: UploadFile = File(...)):
    if not image:
        raise HTTPException(status_code=400, detail="Image is required")
    try:
        contents = await image.read()
        result = compute_image_quality(contents)
        logger.info(
            "Quality check done for filename=%s pass=%s blur=%.2f bright=%.2f contrast=%.2f size=%sx%s",
            getattr(image, "filename", "unknown"),
            result.get("quality_pass"),
            result["metrics"].get("blur_score", 0.0),
            result["metrics"].get("brightness_mean", 0.0),
            result["metrics"].get("contrast_std", 0.0),
            result["metrics"].get("width"),
            result["metrics"].get("height"),
        )
        return JSONResponse(result)
    except AgentConfigError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Quality check failed: {exc}")


@app.post("/api/extract")
async def extract(
    image: UploadFile = File(...),
    quality_pass: Optional[bool] = Form(None),
    quality_metrics_json: Optional[str] = Form(None),
):
    print("\n" + "="*60)
    logger.info("New Extraction Request Started")
    start_time = time.perf_counter()

    if not CACHED_PROMPT_TEXT:
        raise HTTPException(status_code=500, detail="prompt.txt not found or empty")
    
    prompt_text = CACHED_PROMPT_TEXT + "\n\n" + LLM_ADDITIONAL_GUIDANCE

    if not image:
        raise HTTPException(status_code=400, detail="Image is required")

    run_id = str(uuid.uuid4())
    created_at = datetime.utcnow().isoformat() + "Z"
    image_filename = f"{run_id}_{image.filename}"
    image_path = IMAGES_DIR / image_filename

    try:
        contents = await image.read()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read image: {exc}")

    np_arr = np.frombuffer(contents, np.uint8)
    image_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise HTTPException(status_code=400, detail="Unable to decode image.")

    quality = analyze_quality(image_bgr)
    if quality.get("hard_reject"):
        logger.info(
            "Quality reject (continuing extraction): blur=%.2f bright=%.2f contrast=%.2f size=%sx%s reason=%s",
            quality.get("blur_score", 0.0),
            quality.get("brightness_mean", 0.0),
            quality.get("contrast_std", 0.0),
            quality.get("width"),
            quality.get("height"),
            quality.get("message"),
        )

    ocr_image_bgr, enhancement_meta = enhance_for_ocr_if_needed(image_bgr, quality)
    logger.info(
        "Quality accept: blur=%.2f bright=%.2f contrast=%.2f size=%sx%s enhancement_applied=%s steps=%s fallback=%s used_image=%s",
        quality.get("blur_score", 0.0),
        quality.get("brightness_mean", 0.0),
        quality.get("contrast_std", 0.0),
        quality.get("width"),
        quality.get("height"),
        enhancement_meta.get("applied"),
        ",".join(enhancement_meta.get("steps", [])),
        enhancement_meta.get("fallback_to_original"),
        enhancement_meta.get("used_image"),
    )

    # OCR phase
    t_ocr_start = time.perf_counter()
    ocr_failed = False
    try:
        ocr = run_ocr(image_bgr=ocr_image_bgr)
        t_ocr_end = time.perf_counter()
        logger.info(f"OCR completed in {int((t_ocr_end - t_ocr_start)*1000)}ms")
    except Exception as exc:
        logger.warning("OCR failed: %s", exc)
        ocr_failed = True
        ocr = {
            "raw_text": "",
            "lines": [],
            "metrics": {"avg_conf": 0.0, "num_lines": 0, "raw_text_length": 0},
        }
    ocr_text_block = build_ocr_text_block(ocr)

    # Save image
    try:
        image_path.write_bytes(contents)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to save image: {exc}")

    llm_usage = None
    # LLM extraction
    t_llm_start = time.perf_counter()
    try:
        raw_json, llm_usage = await extract_label_json(str(image_path), prompt_text, ocr_text_block)
        t_llm_end = time.perf_counter()
        logger.info(f"LLM completed in {int((t_llm_end - t_llm_start)*1000)}ms")
    except AgentConfigError as exc:
        logger.error(f"Agent Config Error: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))
    except Exception as exc:
        logger.error(f"LLM Extraction failed: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"LLM error: {exc}")


    verification_failures = 0
    if not ocr_failed:
        raw_text = ocr.get("raw_text", "") if isinstance(ocr, dict) else ""
        ocr_lines = ocr.get("lines", []) if isinstance(ocr, dict) else []
        raw_json, verification_failures = verify_against_ocr(raw_json, raw_text, ocr_lines)

    raw_json = coerce_field_types(raw_json)

    fields_table = normalize_fields(raw_json)
    sure_count = sum(1 for row in fields_table if row["status"] == "sure")
    unsure_count = sum(1 for row in fields_table if row["status"] == "unsure")

    computed_quality_pass = bool(quality.get("quality_pass"))
    if quality_pass is None:
        quality_pass = computed_quality_pass

    quality_metrics = {"metrics": quality, "quality_pass": quality_pass}
    if quality_metrics_json:
        try:
            client_quality_metrics = json.loads(quality_metrics_json)
            if isinstance(client_quality_metrics, dict):
                quality_metrics.update(client_quality_metrics)
        except Exception:
            client_quality_metrics = None

    result = {
        "run_id": run_id,
        "created_at": created_at,
        "fields_table": fields_table,
        "raw_json": raw_json,
        "sure_count": sure_count,
        "unsure_count": unsure_count,
        "verification_failures": verification_failures,
        "quality": quality,
        "quality_metrics": quality_metrics,
        "enhancement_meta": enhancement_meta,
        "ocr_image_used": enhancement_meta.get("used_image", "original"),
        "llm_usage": _usage_to_dict(llm_usage),
        "ocr": {
            "raw_text": ocr.get("raw_text", ""),
            "lines": ocr.get("lines", []),
            "metrics": ocr.get("metrics", {}),
        },
    }
    if ocr_failed:
        result["ocr_failed"] = True
    if quality_pass is not None:
        result["quality_pass"] = bool(quality_pass)
    if quality_metrics:
        result["quality_metrics"] = quality_metrics

    try:
        (RESULTS_DIR / f"{run_id}_extracted.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to save result: {exc}")

    response_payload = {
        "run_id": run_id,
        "created_at": created_at,
        "fields_table": fields_table,
        "raw_json": raw_json,
        "sure_count": sure_count,
        "unsure_count": unsure_count,
        "verification_failures": verification_failures,
        "quality": quality,
        "enhancement_meta": enhancement_meta,
        "ocr_image_used": enhancement_meta.get("used_image", "original"),
        "ocr": {
            "raw_text": result["ocr"]["raw_text"],
            "metrics": result["ocr"]["metrics"],
        },
        "llm_usage": result.get("llm_usage"),
    }
    if quality_pass is not None:
        response_payload["quality_pass"] = bool(quality_pass)
    if quality_metrics:
        response_payload["quality_metrics"] = quality_metrics
    response_payload["quality"] = quality
    response_payload["enhancement_meta"] = enhancement_meta
    response_payload["ocr_image_used"] = enhancement_meta.get("used_image", "original")
    if ocr_failed:
        response_payload["ocr_failed"] = True

    end_time = time.perf_counter()
    total_latency_ms = int((end_time - start_time) * 1000)

    # Metrics logging (non-blocking)
    # Metrics logging (non-blocking)
    try:
        metrics_payload = {
            "run_id": run_id,
            "timestamp": created_at,
            "quality_pass": response_payload.get("quality_pass"),
            "blur_score": (quality_metrics or {}).get("metrics", {}).get("blur_score") if quality_metrics else None,
            "brightness": (quality_metrics or {}).get("metrics", {}).get("brightness_mean") if quality_metrics else None,
            "ocr_avg_conf": result["ocr"].get("metrics", {}).get("avg_conf"),
            "ocr_num_lines": result["ocr"].get("metrics", {}).get("num_lines"),
            "sure_count": sure_count,
            "unsure_count": unsure_count,
            "verification_failures": verification_failures,
            "validation_failures": 0,
            "prompt_tokens": result.get("llm_usage", {}).get("prompt_tokens") if isinstance(result.get("llm_usage"), dict) else None,
            "candidates_tokens": result.get("llm_usage", {}).get("candidates_tokens") if isinstance(result.get("llm_usage"), dict) else None,
            "total_tokens": result.get("llm_usage", {}).get("total_tokens") if isinstance(result.get("llm_usage"), dict) else None,
            "total_latency_ms": total_latency_ms,
        }

        # Add per-field metrics
        for row in fields_table:
            f_name = row["field"]
            if f_name in EXPECTED_FIELDS:
                safe_name = f_name.replace("/", "_").replace(" ", "_")
                metrics_payload[f"{safe_name}_conf"] = row.get("confidence")
                metrics_payload[f"{safe_name}_status"] = row.get("status")

        append_metrics_row(metrics_payload)
    except Exception as exc:
        logger.error("Failed to append metrics: %s", exc)

    return JSONResponse(response_payload)








@app.get("/api/metrics")
async def metrics():
    if not METRICS_FILE.exists():
        return JSONResponse({"header": METRICS_COLUMNS, "rows": []})
    try:
        with METRICS_FILE.open("r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            rows = list(reader)
        if not rows:
            return JSONResponse({"header": METRICS_COLUMNS, "rows": []})
        header = rows[0]
        data_rows = rows[1:]
        return JSONResponse({"header": header, "rows": data_rows})
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read metrics log: {exc}")


def normalize_fields(raw: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for field in EXPECTED_FIELDS:
        entry = raw.get(field) if isinstance(raw, dict) else None
        value = entry.get("value") if isinstance(entry, dict) else None
        confidence = entry.get("confidence") if isinstance(entry, dict) else None
        evidence = entry.get("evidence") if isinstance(entry, dict) else None
        verification_flags = entry.get("verification_flags") if isinstance(entry, dict) else None
        value_num = entry.get("value_num") if isinstance(entry, dict) else None
        currency = entry.get("currency") if isinstance(entry, dict) else None
        unit = entry.get("unit") if isinstance(entry, dict) else None

        if not isinstance(confidence, (int, float)):
            confidence = 0.0
        confidence = max(0.0, min(1.0, float(confidence)))
        status = "sure" if confidence >= 0.9 else "unsure"

        if not isinstance(evidence, list):
            evidence = []
        if not isinstance(verification_flags, list):
            verification_flags = []

        rows.append(
            {
                "field": field,
                "value": value,
                "confidence": confidence,
                "status": status,
                "evidence": evidence,
                "verification_flags": verification_flags,
                "value_num": value_num,
                "currency": currency,
                "unit": unit,
            }
        )
    return rows


def _parse_price(val: Any) -> Optional[float]:
    if val is None:
        return None
    s = str(val)
    # Remove currency symbols/spaces/commas.
    s = re.sub(r"[^\d.]", "", s)
    if s.count(".") > 1:
        # Keep first dot, drop the rest.
        first = s.find(".")
        s = s[: first + 1] + s[first + 1 :].replace(".", "")
    try:
        return float(s) if s else None
    except ValueError:
        return None


def _parse_quantity(val: Any) -> Dict[str, Any]:
    if val is None:
        return {"value_num": None, "unit": ""}
    s = str(val).strip()
    m = re.match(r"^\s*([0-9]+(?:\.[0-9]+)?)\s*([a-zA-Z]+)?\s*$", s)
    if not m:
        return {"value_num": None, "unit": ""}
    num = m.group(1)
    unit = m.group(2) or ""
    try:
        num_val = float(num) if "." in num else int(num)
    except ValueError:
        num_val = None
    return {"value_num": num_val, "unit": unit}


def coerce_field_types(raw: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(raw, dict):
        return raw
    coerced = dict(raw)
    for field_name, entry in raw.items():
        if not isinstance(entry, dict):
            continue
        lower = field_name.lower()
        if lower == "mrp_price":
            price_num = _parse_price(entry.get("value"))
            if price_num is not None:
                entry["value_num"] = price_num
                # Clean string value to remove embedded spaces between digits.
                entry["value"] = re.sub(r"\s+(?=[0-9])", "", str(entry.get("value", "")))
        elif lower == "net_quantity":
            qty_info = _parse_quantity(entry.get("value"))
            entry["value_num"] = qty_info.get("value_num")
            entry["unit"] = qty_info.get("unit", "")
    return coerced


def _usage_to_dict(usage: Any) -> Dict[str, Any]:
    if usage is None:
        return {}
    return {
        "prompt_tokens": getattr(usage, "prompt_token_count", None),
        "candidates_tokens": getattr(usage, "candidates_token_count", None),
        "total_tokens": getattr(usage, "total_token_count", None),
    }


def append_metrics_row(metrics: Dict[str, Any]) -> None:
    header = "\t".join(METRICS_COLUMNS) + "\n"
    # Logic to migrate legacy file effectively relies on column count or header check
    
    def fmt_bool(val):
        if val is True:
            return "true"
        if val is False:
            return "false"
        return ""

    def fmt(val):
        if val is None:
            return ""
        if isinstance(val, bool):
            return fmt_bool(val)
        return str(val)

    # Build the list of values matching METRICS_COLUMNS order
    values = []
    for col in METRICS_COLUMNS:
        values.append(fmt(metrics.get(col)))

    row = "\t".join(values) + "\n"

    try:
        if not METRICS_FILE.exists():
            with METRICS_FILE.open("w", encoding="utf-8") as f:
                f.write(header)
        else:
            # Check if header matches. If not, we might need migration or just append if we don't care about old columns mismatch (but we do).
            # Simple migration: read all, rewrite with new header. Old rows will have fewer columns (pandas handles this, but raw CSV might break).
            # We will just pad existing rows if header changed.
            with METRICS_FILE.open("r", encoding="utf-8") as f:
                lines = f.readlines()
            
            if not lines or lines[0].strip() != header.strip():
                # Header mismatch - rewrite file with new header
                # We won't attempt complex migration of data columns for now, just preserve old lines (they will be short)
                # Or better: Pad them.
                new_col_count = len(METRICS_COLUMNS)
                migrated_lines = []
                # Write new header
                if lines:
                     old_header_cols = len(lines[0].split("\t"))
                else:
                     old_header_cols = 0
                
                # We just write new header. Old rows effectively become truncated or messy. 
                # Given user request, it's acceptable to just switch schema.
                with METRICS_FILE.open("w", encoding="utf-8") as f:
                    f.write(header)
                    # Write old lines padded
                    if lines:
                        for line in lines[1:]: # Skip old header
                             parts = line.strip().split("\t")
                             # pad with empty strings
                             while len(parts) < new_col_count:
                                 parts.append("")
                             f.write("\t".join(parts) + "\n")
            
        with METRICS_FILE.open("a", encoding="utf-8") as f:
            f.write(row)
    except Exception as exc:
        logger.error("Failed to write metrics row: %s", exc)

@app.get("/api/history")
async def history():
    """List all past extraction runs."""
    runs = []
    # files like {run_id}_extracted.json
    try:
        for f in RESULTS_DIR.glob("*_extracted.json"):
            # Parse filename to get run_id
            run_id = f.name.replace("_extracted.json", "")
            
            try:
                # Validation / Cleanup Logic
                content = f.read_text(encoding="utf-8")
                data = json.loads(content)
                
                # Check for modern schema
                # Must have 'fields_table' and 'run_id'
                if "fields_table" not in data or "run_id" not in data:
                    logger.warning(f"Skipping legacy/invalid history file (not deleting): {f.name}")
                    # f.unlink() # SAFETY: Do not auto-delete
                    continue
                    
                stat = f.stat()
                runs.append({
                    "run_id": run_id,
                    "timestamp": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    "filename": f.name
                })
            except Exception as e:
                logger.warning(f"Deleting corrupted history file {f.name}: {e}")
                try:
                    f.unlink()
                except:
                    pass
                continue

        # Sort by timestamp desc
        runs.sort(key=lambda x: x["timestamp"], reverse=True)
        return JSONResponse(runs)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to list history: {exc}")


@app.get("/api/history/{run_id}")
async def history_detail(run_id: str):
    """Get details for a specific run."""
    result_file = RESULTS_DIR / f"{run_id}_extracted.json"
    if not result_file.exists():
        raise HTTPException(status_code=404, detail="Run not found")
    
    try:
        data = json.loads(result_file.read_text(encoding="utf-8"))
        return JSONResponse(data)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read result: {exc}")



class EditRequest(BaseModel):
    run_id: str
    updates: Dict[str, Any]


@app.post("/api/save_edits")
async def save_edits(req: EditRequest):
    """Save user edits to a result file."""
    result_file = RESULTS_DIR / f"{req.run_id}_extracted.json"
    if not result_file.exists():
        raise HTTPException(status_code=404, detail="Run not found")
    
    try:
        # Load existing data
        data = json.loads(result_file.read_text(encoding="utf-8"))
        
        # Apply updates
        dirty = False
        if "fields_table" not in data:
            data["fields_table"] = []
            
        # Update raw_json as well if possible, but mainly fields_table for now
        # We need to map field names to the structure
        
        for field_name, new_value in req.updates.items():
            # Update fields_table
            found = False
            for row in data["fields_table"]:
                if row["field"] == field_name:
                    row["value"] = new_value
                    row["status"] = "sure" # User confirmed
                    row["confidence"] = 1.0 # User confirmed
                    found = True
                    dirty = True
                    break
            
            # If not found in table, maybe we should add it? 
            # For now, let's just stick to updating existing fields.
            
            # Also update raw_json structure to keep consistency
            if "raw_json" in data and field_name in data["raw_json"]:
                 data["raw_json"][field_name]["value"] = new_value
                 data["raw_json"][field_name]["status"] = "sure"
                 data["raw_json"][field_name]["confidence"] = 1.0

        if dirty:
            result_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
        
        return JSONResponse({"status": "success"})
        
    except Exception as exc:
        logger.error(f"Failed to save edits: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to save: {exc}")


WEB_DIR = ROOT / "web"
app.mount("/storage", StaticFiles(directory=STORAGE_ROOT), name="storage")

if WEB_DIR.exists():
    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
    logger.info("Web UI mounted at /")
else:
    logger.warning("Web directory not found, UI not served.")

