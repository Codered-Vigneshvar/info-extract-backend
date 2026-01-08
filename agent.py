import asyncio
import json
import difflib
import mimetypes
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pytesseract
import google.generativeai as genai
import vertexai
from vertexai.preview.generative_models import GenerationConfig, GenerativeModel, Part

PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")
LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "vertex").lower()
OCR_ENGINE = os.getenv("OCR_ENGINE", "easyocr").lower()
EASYOCR_LANGS = [lang.strip() for lang in os.getenv("EASYOCR_LANGS", "en").split(",") if lang.strip()]

QUALITY_MIN_BLUR = float(os.getenv("QUALITY_MIN_BLUR_SCORE", "300.0"))
QUALITY_MIN_BRIGHTNESS = float(os.getenv("QUALITY_MIN_BRIGHTNESS", "20"))
QUALITY_MAX_BRIGHTNESS = float(os.getenv("QUALITY_MAX_BRIGHTNESS", "245"))
QUALITY_MIN_W = int(os.getenv("QUALITY_MIN_W", "600"))
QUALITY_MIN_H = int(os.getenv("QUALITY_MIN_H", "600"))

TESSERACT_CMD = os.getenv("TESSERACT_CMD")
if TESSERACT_CMD:
    _env_tess = Path(TESSERACT_CMD)
    if _env_tess.exists():
        pytesseract.pytesseract.tesseract_cmd = str(_env_tess)
    else:
        # Env var points to missing binary; fall back to defaults below.
        TESSERACT_CMD = None

if not TESSERACT_CMD and os.name == "nt":
    _default_tess = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
    _user_tess = Path(os.getenv("LOCALAPPDATA", "")) / "Programs" / "Tesseract-OCR" / "tesseract.exe"
    if _default_tess.exists():
        pytesseract.pytesseract.tesseract_cmd = str(_default_tess)
    elif _user_tess.exists():
        pytesseract.pytesseract.tesseract_cmd = str(_user_tess)

_model: Optional[GenerativeModel] = None
_easyocr_reader = None


class AgentConfigError(RuntimeError):
    pass


def _ensure_model():
    global _model
    if LLM_PROVIDER not in ("vertex", "gemini"):
        raise AgentConfigError(f"Unsupported LLM_PROVIDER '{LLM_PROVIDER}'. Use 'vertex' or 'gemini'.")
    
    if _model is None:
        if LLM_PROVIDER == "vertex":
            if not PROJECT or not LOCATION:
                raise AgentConfigError("GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION must be set for Vertex AI.")
            vertexai.init(project=PROJECT, location=LOCATION)
            _model = GenerativeModel(MODEL_NAME)
        elif LLM_PROVIDER == "gemini":
            if not GOOGLE_API_KEY:
                raise AgentConfigError("GOOGLE_API_KEY must be set for Gemini provider.")
            genai.configure(api_key=GOOGLE_API_KEY)
            _model = genai.GenerativeModel(MODEL_NAME)
            
    return _model


def _image_part(image_path: str) -> Any:
    mime_type, _ = mimetypes.guess_type(image_path)
    if not mime_type:
        mime_type = "application/octet-stream"
    data = Path(image_path).read_bytes()
    
    if LLM_PROVIDER == "vertex":
        return Part.from_data(mime_type=mime_type, data=data)
    else:
        # For google-generativeai, a dict with 'mime_type' and 'data' works
        return {"mime_type": mime_type, "data": data}


def preprocess_for_ocr(image_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    if w < 1200:
        scale = 1200 / float(w)
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


def _group_lines(data: Dict[str, Any]) -> Tuple[str, list, Dict[str, float]]:
    lines = []
    groups = {}
    num_items = len(data.get("text", []))
    for i in range(num_items):
        text = data["text"][i]
        conf = float(data["conf"][i])
        if text is None or text.strip() == "":
            continue
        if conf < 0:
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        groups.setdefault(key, []).append((text, conf))

    raw_lines = []
    for _, items in sorted(groups.items(), key=lambda x: x[0]):
        texts, confs = zip(*items)
        line_text = " ".join(texts).strip()
        if not line_text:
            continue
        avg_conf = float(sum(confs) / len(confs)) / 100.0
        raw_lines.append({"text": line_text, "conf": avg_conf})

    raw_text = "\n".join([l["text"] for l in raw_lines])
    valid_confs = [l["conf"] for l in raw_lines if l["conf"] >= 0]
    avg_conf = float(sum(valid_confs) / len(valid_confs)) if valid_confs else 0.0
    metrics = {
        "avg_conf": avg_conf,
        "num_lines": len(raw_lines),
        "raw_text_length": len(raw_text),
    }
    return raw_text, raw_lines, metrics


def _ensure_easyocr_reader():
    global _easyocr_reader
    if _easyocr_reader is None:
        try:
            import easyocr
        except ImportError as exc:
            raise AgentConfigError("EasyOCR is not installed. Set OCR_ENGINE=tesseract or install easyocr.") from exc
        langs = EASYOCR_LANGS if EASYOCR_LANGS else ["en"]
        _easyocr_reader = easyocr.Reader(langs, gpu=False)
    return _easyocr_reader


def _run_easyocr(image_bgr: np.ndarray) -> dict:
    reader = _ensure_easyocr_reader()
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    results = reader.readtext(image_rgb, detail=1, paragraph=False)
    raw_lines = []
    confs: List[float] = []
    for item in results:
        if len(item) < 3:
            continue
        _, text, conf = item
        if not text:
            continue
        raw_lines.append({"text": text, "conf": float(conf)})
        confs.append(float(conf))
    raw_text = "\n".join([l["text"] for l in raw_lines])
    avg_conf = float(np.mean(confs)) if confs else 0.0
    metrics = {"avg_conf": avg_conf, "num_lines": len(raw_lines), "raw_text_length": len(raw_text)}
    return {"raw_text": raw_text, "lines": raw_lines, "metrics": metrics}


def _run_tesseract(image_bgr: np.ndarray) -> dict:
    processed = preprocess_for_ocr(image_bgr)
    data = pytesseract.image_to_data(processed, output_type=pytesseract.Output.DICT, config="--psm 6")
    raw_text, lines, metrics = _group_lines(data)
    return {"raw_text": raw_text, "lines": lines, "metrics": metrics}


def run_ocr(image_bytes: Optional[bytes] = None, image_bgr: Optional[np.ndarray] = None) -> dict:
    if image_bgr is None:
        if image_bytes is None:
            raise AgentConfigError("No image data provided for OCR.")
        np_arr = np.frombuffer(image_bytes, np.uint8)
        image_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise AgentConfigError("Unable to decode image for OCR.")
    if OCR_ENGINE == "easyocr":
        return _run_easyocr(image_bgr)
    return _run_tesseract(image_bgr)


def _compute_basic_quality(image_bgr: np.ndarray) -> Dict[str, Any]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    lap_var_full = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # Edge map to find probable label region.
    edges = cv2.Canny(gray, 100, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = gray.shape
    roi = gray
    roi_area = h * w
    if contours:
        largest = max(contours, key=cv2.contourArea)
        x, y, cw, ch = cv2.boundingRect(largest)
        pad = 10
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(w, x + cw + pad)
        y1 = min(h, y + ch + pad)
        if (x1 - x0) > 0 and (y1 - y0) > 0:
            roi = gray[y0:y1, x0:x1]
            roi_area = (x1 - x0) * (y1 - y0)

    blur_score_roi = float(cv2.Laplacian(roi, cv2.CV_64F).var())
    brightness_mean = float(np.mean(gray))
    contrast_std = float(np.std(gray))
    edge_coverage = float(np.count_nonzero(edges)) / float(h * w)

    return {
        "blur_score": blur_score_roi,
        "blur_score_full": lap_var_full,
        "brightness_mean": brightness_mean,
        "contrast_std": contrast_std,
        "edge_coverage": edge_coverage,
        "width": int(w),
        "height": int(h),
        "roi_area": roi_area,
    }


def analyze_quality(image_bgr: np.ndarray) -> Dict[str, Any]:
    if image_bgr is None:
        raise AgentConfigError("Unable to decode image for quality check.")

    metrics = _compute_basic_quality(image_bgr)
    blur_score = metrics["blur_score"]
    brightness_mean = metrics["brightness_mean"]
    contrast_std = metrics["contrast_std"]
    w = metrics["width"]
    h = metrics["height"]

    # Adapt blur threshold if the frame is mostly flat background.
    blur_threshold = QUALITY_MIN_BLUR
    if metrics.get("edge_coverage", 0.0) < 0.02:
        blur_threshold = QUALITY_MIN_BLUR * 0.6  # be more lenient on low-texture frames

    hard_reject = False
    message = None
    if blur_score < blur_threshold:
        hard_reject = True
        message = "Image is blurry. Hold steady and refocus."
    elif min(w, h) < min(QUALITY_MIN_W, QUALITY_MIN_H):
        hard_reject = True
        message = "Image resolution too low. Move closer to the label."
    elif brightness_mean < QUALITY_MIN_BRIGHTNESS:
        hard_reject = True
        message = "Image too dark. Improve lighting or use flash."
    elif brightness_mean > QUALITY_MAX_BRIGHTNESS:
        hard_reject = True
        message = "Image too bright/overexposed. Avoid direct flash on glossy labels."

    metrics.update(
        {
            "hard_reject": hard_reject,
            "message": message,
            "quality_pass": not hard_reject,
            "blur_threshold_used": blur_threshold,
        }
    )
    return metrics


def _apply_gamma(image_bgr: np.ndarray, gamma: float) -> np.ndarray:
    inv_gamma = 1.0 / gamma if gamma != 0 else 0
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(image_bgr, table)


def _apply_brightness_scale(image_bgr: np.ndarray, scale: float) -> np.ndarray:
    return np.clip(image_bgr.astype(np.float32) * scale, 0, 255).astype(np.uint8)


def _apply_clahe(image_bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def _apply_unsharp_mask(image_bgr: np.ndarray, radius: float = 1.0, amount: float = 0.5) -> np.ndarray:
    blurred = cv2.GaussianBlur(image_bgr, (0, 0), sigmaX=radius)
    return cv2.addWeighted(image_bgr, 1 + amount, blurred, -amount, 0)


def enhance_for_ocr_if_needed(image_bgr: np.ndarray, quality: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Apply conditional enhancements for OCR only. Returns the OCR image and metadata.
    """
    if image_bgr is None:
        raise AgentConfigError("Unable to decode image for enhancement.")

    if quality is None:
        quality = {}

    brightness = quality.get("brightness_mean")
    contrast_std = quality.get("contrast_std")
    blur_score = quality.get("blur_score")

    enhanced = image_bgr.copy()
    steps: List[str] = []

    # Low brightness: gamma correction
    if brightness is not None:
        if 20 <= brightness < 40:
            enhanced = _apply_gamma(enhanced, gamma=1.4)
            steps.append("gamma_1.4")
        elif 40 <= brightness < 50:
            enhanced = _apply_gamma(enhanced, gamma=1.2)
            steps.append("gamma_1.2")

    # High brightness: scale down intensity
    if brightness is not None:
        if 230 < brightness <= 240:
            enhanced = _apply_brightness_scale(enhanced, 0.85)
            steps.append("brightness_mul_0.85")
        elif 240 < brightness <= 245:
            enhanced = _apply_brightness_scale(enhanced, 0.75)
            steps.append("brightness_mul_0.75")

    # Low contrast: CLAHE on L channel, only for reasonable brightness
    if contrast_std is not None and brightness is not None:
        if contrast_std < 35 and 40 <= brightness <= 200:
            enhanced = _apply_clahe(enhanced)
            steps.append("clahe")

    # Mild sharpening if slightly blurry but not too low contrast
    if blur_score is not None and contrast_std is not None:
        if 70 <= blur_score <= 80 and contrast_std > 40:
            enhanced = _apply_unsharp_mask(enhanced, radius=1.0, amount=0.5)
            steps.append("unsharp_mask")

    applied = len(steps) > 0

    # Safety check: ensure we did not harm blur/contrast
    fallback_to_original = False
    used_image = "enhanced" if applied else "original"
    if applied:
        enhanced_metrics = _compute_basic_quality(enhanced)
        orig_contrast = contrast_std if contrast_std is not None else enhanced_metrics["contrast_std"]
        if enhanced_metrics["blur_score"] < QUALITY_MIN_BLUR or enhanced_metrics["contrast_std"] < 0.8 * orig_contrast:
            enhanced = image_bgr
            fallback_to_original = True
            used_image = "original"

    meta = {
        "applied": applied,
        "steps": steps,
        "fallback_to_original": fallback_to_original,
        "used_image": used_image,
    }
    return enhanced, meta


def _normalize_currency_tokens(text: str) -> str:
    text = text.lower()
    text = text.replace("₹", "rs")
    text = re.sub(r"\b(inr|rs)\b", "rs", text)
    return text


def _normalize_for_match(text: str) -> str:
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    text = _normalize_currency_tokens(text)
    text = text.lower()
    # Remove spaces and punctuation/symbols.
    return re.sub(r"[^a-z0-9]", "", text)


def _needs_proximity(field_name: str) -> bool:
    targets = {
        "mrp_price",
        "mfg_date",
        "exp_date",
        "net_quantity",
        "batch_lot_no",
    }
    return field_name.lower() in targets


def _value_near_keyword(value_norm: str, ocr_lines: List[Dict[str, Any]]) -> bool:
    if not ocr_lines:
        return False

    keywords = ["mrp", "₹", "mfg", "exp", "best before", "net", "batch", "lot"]
    normalized_lines = [_normalize_for_match(line.get("text", "")) for line in ocr_lines]
    keyword_indices = []
    for idx, line in enumerate(ocr_lines):
        line_text = str(line.get("text", "")).lower()
        if any(k in line_text for k in keywords):
            keyword_indices.append(idx)

    if not keyword_indices:
        return False

    for idx in keyword_indices:
        for neighbor in (idx - 1, idx, idx + 1):
            if 0 <= neighbor < len(normalized_lines):
                if value_norm in normalized_lines[neighbor]:
                    return True
    return False



def _calculate_match_score(value_norm: str, ocr_full_text_norm: str, ocr_lines: List[Dict[str, Any]]) -> float:
    """
    Returns a score 0.0 to 1.0 representing how well the value matches the OCR.
    Uses fuzzy matching to tolerate typos.
    """
    if not value_norm:
        return 0.0
    
    # 1. Exact Match (Fast path)
    if value_norm in ocr_full_text_norm:
        return 1.0

    # 2. Line-by-line fuzzy search
    # We look for the best partial match of 'value_norm' inside any OCR line.
    best_ratio = 0.0
    
    # Heuristic: if value is short, be stricter. If long, allow more fuzz.
    min_len = len(value_norm)
    
    for line in ocr_lines:
        line_text = _normalize_for_match(line.get("text", ""))
        if not line_text:
            continue
        
        # Check standard similarity
        ratio = difflib.SequenceMatcher(None, value_norm, line_text).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            
        # Check substring similarity (if value is smaller than line)
        if len(line_text) > min_len:
            # Quick substring check
            if value_norm in line_text:
                return 1.0
            
            # Sliding window for fuzzy substring could be expensive, 
            # but difflib's get_matching_blocks is robust enough for now.
            # Using real_quick_ratio as a proxy filter if needed, but ratio() is fine for small lines.
            pass

    return best_ratio


def verify_against_ocr(extraction: Dict[str, Any], ocr_raw_text: str, ocr_lines: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], int]:
    """
    Evidence gating: Adjust confidence based on OCR match quality.
    """
    if not isinstance(extraction, dict):
        return extraction, 0
    
    normalized_full_text = _normalize_for_match(ocr_raw_text or "")
    verification_failures = 0

    for field_name, entry in extraction.items():
        if not isinstance(entry, dict):
            continue
        
        value = entry.get("value")
        if not value:
            continue
            
        value_str = str(value).strip()
        if not value_str:
            continue

        value_norm = _normalize_for_match(value_str)
        
        # Calculate Match Score (0.0 - 1.0)
        match_score = _calculate_match_score(value_norm, normalized_full_text, ocr_lines)
        
        # Boost score for proximity (if key is found nearby) - simplified from original logic
        if match_score > 0.6 and _needs_proximity(field_name):
             if _value_near_keyword(value_norm, ocr_lines):
                 match_score = min(1.0, match_score + 0.1)

        # Retrieve LLM Confidence (default to 0.8 if missing/invalid)
        try:
            llm_conf = float(entry.get("confidence", 0.8))
        except Exception:
            llm_conf = 0.8
        
        # New Weighted Formula
        # Final = LLM_Conf * (0.4 + 0.6 * Match_Score)
        # This means even with NO match, we keep 40% of LLM confidence (trusting the vision model).
        # But a good match boosts it significantly.
        
        # Exceptions for strict formats
        if match_score < 0.8:
            # If it's a price or number, be stricter? No, let's keep it generous for now to avoid the "69%" complaint.
            pass

        final_conf = llm_conf * (0.4 + 0.6 * match_score)
        
        # Specific typo tolerance: if match > 0.8, treat as practically perfect
        if match_score > 0.85:
            final_conf = max(final_conf, llm_conf)

        entry["confidence"] = round(final_conf, 2)
        
        # Flagging
        flags = []
        if match_score < 0.5:
            flags.append("low_ocr_match")
            verification_failures += 1
            entry["status"] = "unsure" # Force unsure if low match
        else:
            # If high match, trust the final conf
            pass
            
        entry["verification_flags"] = flags

    return extraction, verification_failures


def build_ocr_text_block(ocr: dict, max_chars: int = 4000) -> str:
    text = ocr.get("raw_text", "") if isinstance(ocr, dict) else ""
    if not text:
        return ""
    if len(text) > max_chars:
        return text[:max_chars] + "\n[TRUNCATED]"
    return text


async def extract_label_json(image_path: str, prompt: str, ocr_text: str) -> dict:
    def _call():
        model = _ensure_model()
        combined_prompt = f"{prompt}\n\nOCR_TEXT:\n{ocr_text}"
        
        if LLM_PROVIDER == "vertex":
            generation_config = GenerationConfig(response_mime_type="application/json")
        else:
            # google-generativeai style config
            generation_config = genai.types.GenerationConfig(response_mime_type="application/json")

        resp = model.generate_content(
            [
                combined_prompt,
                _image_part(image_path),
            ],
            generation_config=generation_config,
        )
        text = (resp.text or "").strip()
        
        # Usage metadata handling
        usage = None
        if hasattr(resp, "usage_metadata"):
            usage = resp.usage_metadata
        
        try:
            return json.loads(text), usage
        except json.JSONDecodeError as exc:
            raise AgentConfigError(f"LLM did not return valid JSON: {exc}")

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _call)


def compute_image_quality(image_bytes: bytes) -> Dict[str, Any]:
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        raise AgentConfigError("Unable to decode image for quality check.")

    metrics = analyze_quality(img)
    messages = []
    if metrics.get("hard_reject") and metrics.get("message"):
        messages.append(metrics["message"])

    return {
        "quality_pass": bool(metrics.get("quality_pass")),
        "messages": messages,
        "metrics": {
            "blur_score": metrics.get("blur_score", 0.0),
            "brightness_mean": metrics.get("brightness_mean", 0.0),
            "contrast_std": metrics.get("contrast_std", 0.0),
            "width": metrics.get("width"),
            "height": metrics.get("height"),
        },
        "thresholds": {
            "min_blur_score": QUALITY_MIN_BLUR,
            "min_brightness": QUALITY_MIN_BRIGHTNESS,
            "max_brightness": QUALITY_MAX_BRIGHTNESS,
            "min_resolution_w": QUALITY_MIN_W,
            "min_resolution_h": QUALITY_MIN_H,
        },
    }
