#!/usr/bin/env python3
"""
AI-POWERED ART CROP SYSTEM
Multi-tier artwork detection with AI vision analysis

Tiers:
1. rembg (fast baseline)
2. AI Vision Analysis (GPT-4V, Claude, Gemini) - consensus system
3. CV Edge Detection (fallback)
4. Manual review queue

Features:
- Multi-AI consensus for accuracy
- Signature/edition number detection
- Quality assessment (lighting, focus, angle)
- Confidence scoring
"""

import os
import io
import cv2
import json
import base64
import numpy as np
from tqdm import tqdm
from datetime import datetime
from PIL import Image
import rawpy

# Google Cloud Vision for rotation detection
try:
    from google.cloud import vision
    GOOGLE_VISION_AVAILABLE = True
except ImportError:
    GOOGLE_VISION_AVAILABLE = False

# Path to Google Vision credentials
GOOGLE_VISION_CREDENTIALS = "/Users/johnshay/3DSELLERS/google_vision_credentials.json"
from pathlib import Path
from typing import Optional, Tuple, Dict, List
import requests

# Background removal
try:
    from rembg import remove
    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False
    print("Warning: rembg not installed. Install with: pip install rembg")

# ==============================================================
# CONFIGURATION
# ==============================================================

BASE_DIR = Path("/Users/johnshay/3DSELLERS")
INPUT_DIR = BASE_DIR / "input"
OUTPUT_MASTER = BASE_DIR / "output_master"
OUTPUT_EBAY = BASE_DIR / "output_ebay"
OUTPUT_PNG = BASE_DIR / "output_png"
QUARANTINE_DIR = BASE_DIR / "quarantine"
DEBUG_DIR = BASE_DIR / "debug"
LOG_DIR = BASE_DIR / "logs"
LOG_FILE = LOG_DIR / "ai_crop_log.txt"
ANALYSIS_DIR = BASE_DIR / "analysis"  # Store AI analysis results

# API Keys - load from config or environment
CONFIG_FILE = BASE_DIR / "config.js"

# Crop settings
MIN_AREA_RATIO = 0.05
MAX_AREA_RATIO = 0.95
AI_CROP_MARGIN_PERCENT = 10.0  # Extra margin around AI-detected bounds (10% = safer crop)
PADDING_PERCENT = 0.03  # White padding added after crop (3%)
EBAY_LONGEST_PX = 2000
EBAY_JPEG_QUALITY = 92

# AI Consensus settings
MIN_CONFIDENCE = 0.7  # Minimum confidence to accept a crop
CONSENSUS_THRESHOLD = 2  # Number of AIs that must agree

# ==============================================================
# LOGGING & UTILITIES
# ==============================================================

def ensure_dirs():
    for d in [INPUT_DIR, OUTPUT_MASTER, OUTPUT_EBAY, OUTPUT_PNG,
              QUARANTINE_DIR, DEBUG_DIR, LOG_DIR, ANALYSIS_DIR]:
        d.mkdir(parents=True, exist_ok=True)

def log(msg: str):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"{timestamp} - {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")

def load_api_keys():
    """Load API keys from config file or hardcoded."""
    keys = {
        'openai': os.environ.get('OPENAI_API_KEY'),
        'anthropic': os.environ.get('ANTHROPIC_API_KEY'),
        'google': os.environ.get('GOOGLE_API_KEY'),
        'xai': os.environ.get('XAI_API_KEY')  # Grok
    }

    # Keys should be loaded from environment variables
    # Do not hardcode API keys here

    # Remove.bg API key
    keys['removebg'] = os.environ.get('REMOVEBG_API_KEY', '')

    # Try to load from config.js if exists
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, 'r') as f:
                content = f.read()
                # Parse JavaScript config
                import re
                for key_name, env_name in [('openai', 'OPENAI_API_KEY'), ('anthropic', 'ANTHROPIC_API_KEY'),
                                            ('google', 'GOOGLE_API_KEY'), ('xai', 'XAI_API_KEY')]:
                    match = re.search(rf"{env_name}['\"]?\s*[:=]\s*['\"]([^'\"]+)['\"]", content)
                    if match:
                        keys[key_name] = match.group(1)
        except Exception as e:
            log(f"Error loading config: {e}")

    return keys

def image_to_base64(image_path: str) -> str:
    """Convert image to base64 for API calls."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

def list_images(folder: Path):
    exts = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".heic", ".dng")
    return [f for f in folder.iterdir() if f.suffix.lower() in exts]

def read_image(filepath: Path) -> Optional[np.ndarray]:
    """
    Read image file, handling DNG/RAW files with rawpy for full resolution.
    Returns image in BGR format (OpenCV standard).
    """
    filepath_str = str(filepath)
    ext = filepath.suffix.lower()

    if ext == '.dng':
        # Use rawpy for DNG files to get full resolution
        try:
            with rawpy.imread(filepath_str) as raw:
                # Process RAW to RGB
                rgb = raw.postprocess(
                    use_camera_wb=True,       # Use camera white balance
                    half_size=False,          # Full resolution
                    no_auto_bright=False,     # Auto brightness
                    output_bps=8              # 8-bit output
                )
                # Convert RGB to BGR for OpenCV
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                log(f"  DNG file read at full resolution: {bgr.shape[1]}x{bgr.shape[0]} pixels")
                return bgr
        except Exception as e:
            log(f"  Failed to read DNG with rawpy: {e}, trying cv2.imread fallback")
            return cv2.imread(filepath_str)
    else:
        # Use standard cv2.imread for other formats
        return cv2.imread(filepath_str)

# ==============================================================
# AI VISION ANALYSIS PROMPT
# ==============================================================

ARTWORK_ANALYSIS_PROMPT = """Analyze this photograph of artwork and provide a detailed JSON response.

The image shows a piece of art (print, painting, photograph, etc.) that needs to be cropped for an eBay listing.

Please analyze and return a JSON object with:

{
  "artwork_bounds": {
    "description": "Describe where the artwork is in the image",
    "top_percent": 0-100,
    "left_percent": 0-100,
    "width_percent": 0-100,
    "height_percent": 0-100,
    "confidence": 0.0-1.0
  },
  "signature": {
    "present": true/false,
    "location": "bottom_left/bottom_right/bottom_center/other/none",
    "description": "What the signature looks like"
  },
  "edition_number": {
    "present": true/false,
    "location": "bottom_left/bottom_right/bottom_center/other/none",
    "value": "e.g., 45/100, AP, HC, or null"
  },
  "artwork_details": {
    "title": "descriptive title of the artwork subject (e.g., 'Son of Man with Coca-Cola', 'Mickey Mouse Pop Art', 'Banksy Girl with Balloon')",
    "type": "print/painting/photograph/mixed_media/other",
    "orientation": "portrait/landscape/square",
    "has_mat": true/false,
    "has_frame": true/false,
    "background_surface": "table/floor/wall/mat/other"
  },
  "quality_assessment": {
    "overall_score": 0-10,
    "lighting": "good/acceptable/poor",
    "focus": "sharp/acceptable/blurry",
    "angle": "straight/slightly_tilted/very_tilted",
    "glare_reflections": true/false,
    "shadows": "none/minimal/significant",
    "issues": ["list of any issues detected"]
  },
  "recommendations": {
    "crop_adjustments": "any specific crop recommendations",
    "retake_photo": true/false,
    "retake_reasons": ["reasons if retake recommended"]
  }
}

Be precise with the percentage bounds - they should tightly crop to just the artwork itself, excluding any mat, frame, or background surface. Include any signature or edition numbers in the crop.

NOTE: All photos are already correctly oriented - no rotation is needed."""

# ==============================================================
# AI VISION API CALLS
# ==============================================================

def analyze_with_openai(image_path: str, api_key: str) -> Optional[Dict]:
    """Analyze image using OpenAI GPT-4 Vision."""
    if not api_key:
        return None

    try:
        base64_image = image_to_base64(image_path)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": ARTWORK_ANALYSIS_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 2000
        }

        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=60
        )

        if response.status_code == 200:
            result = response.json()
            content = result['choices'][0]['message']['content']
            # Extract JSON from response
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                return json.loads(content[json_start:json_end])
        else:
            log(f"OpenAI API error: {response.status_code} - {response.text}")

    except Exception as e:
        log(f"OpenAI analysis error: {e}")

    return None

def analyze_with_anthropic(image_path: str, api_key: str) -> Optional[Dict]:
    """Analyze image using Anthropic Claude Vision."""
    if not api_key:
        return None

    try:
        # Check file size and resize if > 4MB (Claude limit is 5MB)
        file_size = Path(image_path).stat().st_size
        if file_size > 4 * 1024 * 1024:  # 4MB
            # Resize image to fit under limit
            img = cv2.imread(image_path)
            h, w = img.shape[:2]
            # Reduce to 50% or until under 4MB
            scale = 0.5
            while scale > 0.1:
                new_w, new_h = int(w * scale), int(h * scale)
                resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                # Encode to check size
                _, buffer = cv2.imencode('.jpg', resized, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if len(buffer) < 4 * 1024 * 1024:
                    base64_image = base64.b64encode(buffer).decode('utf-8')
                    break
                scale -= 0.1
            else:
                # Still too big, skip
                log(f"  Image too large for Claude even after resize")
                return None
        else:
            base64_image = image_to_base64(image_path)

        # Determine media type
        ext = Path(image_path).suffix.lower()
        media_type = "image/jpeg"  # Always JPEG after resize

        headers = {
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01"
        }

        payload = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 2000,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": base64_image
                            }
                        },
                        {
                            "type": "text",
                            "text": ARTWORK_ANALYSIS_PROMPT
                        }
                    ]
                }
            ]
        }

        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            json=payload,
            timeout=60
        )

        if response.status_code == 200:
            result = response.json()
            content = result['content'][0]['text']
            # Extract JSON from response
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                return json.loads(content[json_start:json_end])
        else:
            log(f"Anthropic API error: {response.status_code} - {response.text}")

    except Exception as e:
        log(f"Anthropic analysis error: {e}")

    return None

def analyze_with_gemini(image_path: str, api_key: str) -> Optional[Dict]:
    """Analyze image using Google Gemini Vision."""
    if not api_key:
        return None

    try:
        base64_image = image_to_base64(image_path)

        # Determine media type
        ext = Path(image_path).suffix.lower()
        media_type = "image/jpeg" if ext in ['.jpg', '.jpeg'] else "image/png"

        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent?key={api_key}"

        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": ARTWORK_ANALYSIS_PROMPT},
                        {
                            "inline_data": {
                                "mime_type": media_type,
                                "data": base64_image
                            }
                        }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.1,
                "maxOutputTokens": 2000
            }
        }

        response = requests.post(url, json=payload, timeout=60)

        if response.status_code == 200:
            result = response.json()
            content = result['candidates'][0]['content']['parts'][0]['text']
            # Extract JSON from response
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                return json.loads(content[json_start:json_end])
        else:
            log(f"Gemini API error: {response.status_code} - {response.text}")

    except Exception as e:
        log(f"Gemini analysis error: {e}")

    return None

def analyze_with_grok(image_path: str, api_key: str) -> Optional[Dict]:
    """Analyze image using xAI Grok Vision."""
    if not api_key:
        return None

    try:
        base64_image = image_to_base64(image_path)

        # Determine media type
        ext = Path(image_path).suffix.lower()
        media_type = "image/jpeg" if ext in ['.jpg', '.jpeg'] else "image/png"

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

        payload = {
            "model": "grok-2-vision-1212",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": ARTWORK_ANALYSIS_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{media_type};base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 2000
        }

        response = requests.post(
            "https://api.x.ai/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=60
        )

        if response.status_code == 200:
            result = response.json()
            content = result['choices'][0]['message']['content']
            # Extract JSON from response
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                return json.loads(content[json_start:json_end])
        else:
            log(f"Grok API error: {response.status_code} - {response.text}")

    except Exception as e:
        log(f"Grok analysis error: {e}")

    return None

# ==============================================================
# AI CONSENSUS SYSTEM
# ==============================================================

def get_ai_consensus(image_path: str, api_keys: Dict) -> Tuple[Optional[Dict], float]:
    """
    Get consensus from multiple AI vision models.
    Returns (best_analysis, confidence_score)
    """
    analyses = []

    # Run all available AIs
    log("Running AI vision analysis...")

    if api_keys.get('openai'):
        log("  - Analyzing with GPT-4o...")
        result = analyze_with_openai(image_path, api_keys['openai'])
        if result:
            result['_source'] = 'openai'
            analyses.append(result)
            log("    GPT-4o: SUCCESS")
        else:
            log("    GPT-4o: FAILED")

    if api_keys.get('anthropic'):
        log("  - Analyzing with Claude...")
        result = analyze_with_anthropic(image_path, api_keys['anthropic'])
        if result:
            result['_source'] = 'anthropic'
            analyses.append(result)
            log("    Claude: SUCCESS")
        else:
            log("    Claude: FAILED")

    if api_keys.get('xai'):
        log("  - Analyzing with Grok...")
        result = analyze_with_grok(image_path, api_keys['xai'])
        if result:
            result['_source'] = 'grok'
            analyses.append(result)
            log("    Grok: SUCCESS")
        else:
            log("    Grok: FAILED")

    if api_keys.get('google'):
        log("  - Analyzing with Gemini...")
        result = analyze_with_gemini(image_path, api_keys['google'])
        if result:
            result['_source'] = 'gemini'
            analyses.append(result)
            log("    Gemini: SUCCESS")
        else:
            log("    Gemini: FAILED")

    if not analyses:
        return None, 0.0

    # Calculate consensus on bounds
    bounds_list = []
    for a in analyses:
        if 'artwork_bounds' in a:
            bounds = a['artwork_bounds']
            bounds_list.append({
                'top': bounds.get('top_percent', 0),
                'left': bounds.get('left_percent', 0),
                'width': bounds.get('width_percent', 100),
                'height': bounds.get('height_percent', 100),
                'confidence': bounds.get('confidence', 0.5)
            })

    if not bounds_list:
        return analyses[0], 0.5

    # Average the bounds
    avg_bounds = {
        'top': np.mean([b['top'] for b in bounds_list]),
        'left': np.mean([b['left'] for b in bounds_list]),
        'width': np.mean([b['width'] for b in bounds_list]),
        'height': np.mean([b['height'] for b in bounds_list])
    }

    # Calculate variance to determine agreement
    variance = np.mean([
        np.var([b['top'] for b in bounds_list]),
        np.var([b['left'] for b in bounds_list]),
        np.var([b['width'] for b in bounds_list]),
        np.var([b['height'] for b in bounds_list])
    ])

    # Higher variance = less agreement = lower confidence
    agreement_score = max(0, 1 - (variance / 100))
    avg_confidence = np.mean([b['confidence'] for b in bounds_list])
    final_confidence = (agreement_score + avg_confidence) / 2

    # Use the analysis with highest individual confidence, but update bounds to consensus
    best_analysis = max(analyses, key=lambda a: a.get('artwork_bounds', {}).get('confidence', 0))
    best_analysis['artwork_bounds']['top_percent'] = avg_bounds['top']
    best_analysis['artwork_bounds']['left_percent'] = avg_bounds['left']
    best_analysis['artwork_bounds']['width_percent'] = avg_bounds['width']
    best_analysis['artwork_bounds']['height_percent'] = avg_bounds['height']
    best_analysis['_consensus'] = {
        'num_models': len(analyses),
        'models': [a['_source'] for a in analyses],
        'agreement_score': agreement_score,
        'final_confidence': final_confidence
    }

    log(f"AI Consensus: {len(analyses)} models, agreement={agreement_score:.2f}, confidence={final_confidence:.2f}")

    return best_analysis, final_confidence

# ==============================================================
# REMOVE.BG API CROPPING
# ==============================================================

def crop_with_removebg(image: np.ndarray, api_key: str, crop_margin: int = 5, max_size: int = 4000) -> Optional[np.ndarray]:
    """
    Use remove.bg API to remove background and tight crop the artwork.

    Args:
        image: Input image as numpy array (BGR)
        api_key: remove.bg API key
        crop_margin: Margin around cropped subject (0-100%)
        max_size: Maximum dimension for API (resize if larger)

    Returns:
        Cropped image as numpy array (BGR) or None on failure
    """
    if not api_key:
        return None

    try:
        # Resize if image is too large for API
        h, w = image.shape[:2]
        scale = 1.0
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            log(f"  Resizing for remove.bg API: {w}x{h} → {new_w}x{new_h}")
        else:
            resized = image

        # Encode to JPEG for API
        _, buffer = cv2.imencode('.jpg', resized, [cv2.IMWRITE_JPEG_QUALITY, 95])

        response = requests.post(
            "https://api.remove.bg/v1.0/removebg",
            files={"image_file": ("image.jpg", buffer.tobytes(), "image/jpeg")},
            data={
                "size": "full",           # Keep full resolution
                "crop": "true",           # Tight crop to subject
                "crop_margin": f"{crop_margin}%",  # Add margin
                "bg_color": "ffffff",     # White background
                "format": "jpg",          # JPG output
                "quality": "100"          # Max quality
            },
            headers={"X-Api-Key": api_key},
            timeout=120
        )

        if response.status_code == 200:
            # Convert response to numpy array
            img_array = np.frombuffer(response.content, np.uint8)
            crop = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            # Scale back up to original size if we resized
            if scale < 1.0:
                original_h = int(crop.shape[0] / scale)
                original_w = int(crop.shape[1] / scale)
                crop = cv2.resize(crop, (original_w, original_h), interpolation=cv2.INTER_LANCZOS4)
                log(f"  Scaled result back to original size: {original_w}x{original_h}")

            return crop
        else:
            error_data = response.json() if response.headers.get('content-type') == 'application/json' else {}
            error_msg = error_data.get('errors', [{}])[0].get('title', 'Unknown error')
            log(f"remove.bg API error: {response.status_code} - {error_msg}")
            return None

    except Exception as e:
        log(f"remove.bg error: {e}")
        return None

# ==============================================================
# CROPPING FUNCTIONS
# ==============================================================

def crop_with_ai_bounds(image: np.ndarray, analysis: Dict, margin_percent: float = 10.0) -> Optional[np.ndarray]:
    """
    Crop image using AI-detected bounds with added margin.

    Args:
        image: Input image
        analysis: AI analysis with artwork_bounds
        margin_percent: Extra margin to add around detected bounds (default 10%)
    """
    if not analysis or 'artwork_bounds' not in analysis:
        return None

    bounds = analysis['artwork_bounds']
    h, w = image.shape[:2]

    # Convert percentages to pixels
    top = int(h * bounds.get('top_percent', 0) / 100)
    left = int(w * bounds.get('left_percent', 0) / 100)
    crop_h = int(h * bounds.get('height_percent', 100) / 100)
    crop_w = int(w * bounds.get('width_percent', 100) / 100)

    # Add margin around the detected bounds
    margin_h = int(crop_h * margin_percent / 100)
    margin_w = int(crop_w * margin_percent / 100)

    top = max(0, top - margin_h)
    left = max(0, left - margin_w)
    crop_h = min(crop_h + 2 * margin_h, h - top)
    crop_w = min(crop_w + 2 * margin_w, w - left)

    # Ensure valid bounds
    bottom = min(top + crop_h, h)
    right = min(left + crop_w, w)

    if bottom <= top or right <= left:
        return None

    return image[top:bottom, left:right]

def crop_with_rembg(image: np.ndarray) -> Optional[np.ndarray]:
    """Use rembg to isolate artwork."""
    if not REMBG_AVAILABLE:
        return None

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb).convert('RGBA')

    try:
        no_bg = remove(pil_img)
    except Exception as e:
        log(f"rembg error: {e}")
        return None

    bbox = no_bg.getbbox()
    if bbox is None:
        return None

    x1, y1, x2, y2 = bbox
    return image[y1:y2, x1:x2]

def add_padding(img: np.ndarray, pct: float = PADDING_PERCENT) -> np.ndarray:
    """Add white padding around image."""
    h, w = img.shape[:2]
    pad_h = int(h * pct)
    pad_w = int(w * pct)
    return cv2.copyMakeBorder(
        img, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, value=(255, 255, 255)
    )

def detect_rotation_with_vision(image_path: str) -> int:
    """
    Use Google Cloud Vision OCR to detect correct rotation.
    Returns rotation angle (0, 90, 180, 270) that puts text in bottom corners.
    """
    if not GOOGLE_VISION_AVAILABLE:
        return 0

    try:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GOOGLE_VISION_CREDENTIALS
        client = vision.ImageAnnotatorClient()

        # Load image
        pil_image = Image.open(image_path).convert("RGB")

        best_angle = 0
        best_score = float("-inf")

        for angle in [0, 90, 180, 270]:
            rotated = pil_image.rotate(angle, expand=True)

            # Convert to bytes
            with io.BytesIO() as output:
                rotated.save(output, format="PNG")
                image_bytes = output.getvalue()

            width, height = rotated.size

            # Get text positions
            gcv_image = vision.Image(content=image_bytes)
            response = client.text_detection(image=gcv_image)

            if response.error.message:
                continue

            # Score based on text in bottom corners
            bottom_count = 0
            top_count = 0

            if response.text_annotations:
                for entity in response.text_annotations[1:]:
                    vertices = entity.bounding_poly.vertices
                    if not vertices:
                        continue

                    # Get center
                    xs = [v.x for v in vertices if v.x is not None]
                    ys = [v.y for v in vertices if v.y is not None]
                    if not xs or not ys:
                        continue

                    cx = sum(xs) / len(xs)
                    cy = sum(ys) / len(ys)

                    x_norm = cx / float(width)
                    y_norm = cy / float(height)

                    # Bottom corners (y > 0.65 and x < 0.3 or x > 0.7)
                    if y_norm >= 0.65 and (x_norm <= 0.3 or x_norm >= 0.7):
                        bottom_count += 1

                    # Top region (penalty)
                    if y_norm <= 0.35:
                        top_count += 1

            score = bottom_count * 2 - top_count

            if score > best_score:
                best_score = score
                best_angle = angle

        log(f"  Google Vision detected rotation: {best_angle}° (score: {best_score})")
        return best_angle

    except Exception as e:
        log(f"  Google Vision error: {e}")
        return 0


def correct_orientation(img: np.ndarray, analysis: Optional[Dict]) -> np.ndarray:
    """
    Correct image orientation based on AI analysis.

    Uses rotation_needed from AI analysis to rotate image to upright position.
    """
    if not analysis:
        return img

    # Get rotation needed from artwork_details
    artwork_details = analysis.get('artwork_details', {})
    rotation_needed = artwork_details.get('rotation_needed', 0)

    # Convert to int if it's a string
    try:
        rotation_needed = int(rotation_needed)
    except (ValueError, TypeError):
        rotation_needed = 0

    if rotation_needed == 0:
        return img

    # Apply rotation - AI tells us degrees to rotate clockwise to fix
    if rotation_needed == 90:
        # Rotate 90 degrees clockwise
        rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        log(f"  Rotated image 90° clockwise")
    elif rotation_needed == 180:
        # Rotate 180 degrees
        rotated = cv2.rotate(img, cv2.ROTATE_180)
        log(f"  Rotated image 180°")
    elif rotation_needed == 270:
        # Rotate 270 degrees clockwise = 90 degrees clockwise (same visual result)
        rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        log(f"  Rotated image 90° clockwise (from 270° suggestion)")
    else:
        log(f"  Unknown rotation value: {rotation_needed}, skipping")
        return img

    return rotated

def resize_for_ebay(img: np.ndarray, longest: int = EBAY_LONGEST_PX) -> np.ndarray:
    """Resize image for eBay (max dimension)."""
    h, w = img.shape[:2]
    scale = longest / max(h, w)
    if scale >= 1.0:
        return img
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

# ==============================================================
# MAIN PROCESSING PIPELINE
# ==============================================================

def process_single_image(filepath: Path, api_keys: Dict) -> Optional[Dict]:
    """
    Process a single image with full AI analysis.
    Returns analysis results with output paths.
    """
    filename = filepath.name
    stem = filepath.stem
    original = read_image(filepath)

    if original is None:
        log(f"{filename}: could not read image")
        filepath.replace(QUARANTINE_DIR / filename)
        return None

    log(f"\n{'='*60}")
    log(f"Processing: {filename}")
    log(f"{'='*60}")

    # STEP 1: Get AI analysis for crop bounds
    analysis, ai_confidence = get_ai_consensus(str(filepath), api_keys)

    crop = None
    method_used = None

    # Tier 1: remove.bg API (best quality, tight crop with margin)
    if api_keys.get('removebg'):
        log("Trying remove.bg API for background removal + tight crop...")
        crop = crop_with_removebg(original, api_keys['removebg'], crop_margin=5)
        if crop is not None:
            method_used = "removebg"
            log("✅ remove.bg: SUCCESS - Background removed, tight crop applied")

    # Tier 2: AI-guided crop (if confident)
    if crop is None and analysis and ai_confidence >= MIN_CONFIDENCE:
        log(f"remove.bg failed, using AI-guided crop (confidence: {ai_confidence:.2f}) with {AI_CROP_MARGIN_PERCENT}% margin")
        crop = crop_with_ai_bounds(original, analysis, margin_percent=AI_CROP_MARGIN_PERCENT)
        if crop is not None:
            method_used = "ai_consensus"

    # Tier 3: rembg fallback
    if crop is None and REMBG_AVAILABLE:
        log("AI crop failed or low confidence, trying rembg...")
        crop = crop_with_rembg(original)
        if crop is not None:
            method_used = "rembg"
            log("rembg: SUCCESS")

    # Tier 4: Use full image with warning
    if crop is None:
        log("All detection methods failed, using full image")
        crop = original
        method_used = "full_image"

    # No rotation needed - images are already correctly oriented
    log("  ✓ Image orientation: Already correct (no rotation needed)")

    # Check quality assessment for issues
    if analysis and 'quality_assessment' in analysis:
        qa = analysis['quality_assessment']
        if qa.get('overall_score', 10) < 5:
            log(f"WARNING: Low quality score ({qa.get('overall_score')})")
            for issue in qa.get('issues', []):
                log(f"  - {issue}")

        if analysis.get('recommendations', {}).get('retake_photo'):
            log("RECOMMENDATION: Retake photo")
            for reason in analysis.get('recommendations', {}).get('retake_reasons', []):
                log(f"  - {reason}")

    # Save outputs
    master_path = OUTPUT_MASTER / f"{stem}_master.jpg"
    cv2.imwrite(str(master_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 100])

    ebay_img = add_padding(crop)
    ebay_img = resize_for_ebay(ebay_img)
    ebay_path = OUTPUT_EBAY / f"{stem}_ebay.jpg"
    cv2.imwrite(str(ebay_path), ebay_img, [cv2.IMWRITE_JPEG_QUALITY, EBAY_JPEG_QUALITY])

    png_path = OUTPUT_PNG / f"{stem}.png"
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    Image.fromarray(rgb).save(str(png_path))

    # Save analysis results
    if analysis:
        analysis['_processing'] = {
            'method_used': method_used,
            'ai_confidence': ai_confidence,
            'master_path': str(master_path),
            'ebay_path': str(ebay_path),
            'png_path': str(png_path)
        }
        analysis_path = ANALYSIS_DIR / f"{stem}_analysis.json"
        with open(analysis_path, 'w') as f:
            json.dump(analysis, f, indent=2)

    log(f"SUCCESS: {filename}")
    log(f"  Method: {method_used}")
    log(f"  Master: {master_path}")
    log(f"  eBay: {ebay_path}")

    return {
        'filename': filename,
        'method': method_used,
        'confidence': ai_confidence,
        'analysis': analysis,
        'paths': {
            'master': str(master_path),
            'ebay': str(ebay_path),
            'png': str(png_path)
        }
    }

# ==============================================================
# ENTRY POINT
# ==============================================================

def main():
    ensure_dirs()
    api_keys = load_api_keys()

    # Check API keys
    available_apis = [k for k, v in api_keys.items() if v]
    if not available_apis:
        log("WARNING: No API keys found. AI analysis will be skipped.")
        log("Set environment variables: OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY")
    else:
        log(f"Available AI APIs: {', '.join(available_apis)}")

    # Process both landscape and portrait folders
    landscape_dir = INPUT_DIR / "landscape" / "LANDSCAPE"
    portrait_dir = INPUT_DIR / "portrait" / "PORTRAIT"

    # Collect all images from both folders
    all_images = []

    if landscape_dir.exists():
        landscape_images = list_images(landscape_dir)
        log(f"Found {len(landscape_images)} landscape images")
        all_images.extend(landscape_images)
    else:
        log(f"Landscape directory not found: {landscape_dir}")

    if portrait_dir.exists():
        portrait_images = list_images(portrait_dir)
        log(f"Found {len(portrait_images)} portrait images")
        all_images.extend(portrait_images)
    else:
        log(f"Portrait directory not found: {portrait_dir}")

    if not all_images:
        log(f"No images found in landscape or portrait folders")
        return

    log(f"\nStarting AI-powered processing of {len(all_images)} image(s)")

    results = []
    for filepath in tqdm(all_images, desc="Processing"):
        try:
            result = process_single_image(filepath, api_keys)
            if result:
                results.append(result)
        except Exception as e:
            log(f"{filepath.name}: ERROR - {e}")
            import traceback
            traceback.print_exc()

    # Summary
    log(f"\n{'='*60}")
    log("PROCESSING COMPLETE")
    log(f"{'='*60}")
    log(f"Total: {len(all_images)}, Success: {len(results)}, Failed: {len(all_images) - len(results)}")

    # Method breakdown
    methods = {}
    for r in results:
        m = r.get('method', 'unknown')
        methods[m] = methods.get(m, 0) + 1
    log(f"Methods used: {methods}")

if __name__ == "__main__":
    main()
