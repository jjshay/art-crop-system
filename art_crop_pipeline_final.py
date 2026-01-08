#!/usr/bin/env python3
"""
ART CROP PIPELINE - Tight Crop & Padded Full Image
Detects rectangular artwork in photos using:
- Tier 1: rembg background removal (isolates artwork subject)
- Tier 2: Classical CV fallback (edges + contours)

Outputs:
- _master: Full-res tight crop
- _ebay: eBay-optimized JPEG with white border
- _png: PNG version for further processing
"""

import os
import cv2
import numpy as np
from tqdm import tqdm
from datetime import datetime
from PIL import Image
from pathlib import Path

# Background removal
try:
    from rembg import remove
    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False
    print("Warning: rembg not installed. Install with: pip install rembg")

# ==============================================================
# CONFIGURATION (EDITABLE)
# ==============================================================

# Using existing 3DSELLERS folder structure
BASE_DIR = Path("/Users/johnshay/3DSELLERS")
INPUT_DIR = BASE_DIR / "input"
OUTPUT_MASTER = BASE_DIR / "output_master"
OUTPUT_EBAY = BASE_DIR / "output_ebay"
OUTPUT_PNG = BASE_DIR / "output_png"
QUARANTINE_DIR = BASE_DIR / "quarantine"
DEBUG_DIR = BASE_DIR / "debug"
LOG_DIR = BASE_DIR / "logs"
LOG_FILE = LOG_DIR / "art_crop_log.txt"

# How much of original image area the crop should occupy (for validation)
MIN_AREA_RATIO = 0.05   # at least 5% of image
MAX_AREA_RATIO = 0.95   # at most 95% of image

# eBay output settings
PADDING_PERCENT = 0.07     # 7% white border around crop
EBAY_LONGEST_PX = 2000     # max side length
EBAY_JPEG_QUALITY = 92

# Overwrite behavior
OVERWRITE_OUTPUT = True    # if False, will version filenames (_v2, _v3, ...)

# Use CV fallback if rembg fails
ENABLE_CV_FALLBACK = True

# ==============================================================
# LOGGING & UTILITIES
# ==============================================================

def ensure_dirs():
    for d in [INPUT_DIR, OUTPUT_MASTER, OUTPUT_EBAY, OUTPUT_PNG,
              QUARANTINE_DIR, DEBUG_DIR, LOG_DIR]:
        d.mkdir(parents=True, exist_ok=True)

def log(msg: str):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"{timestamp} - {msg}"
    print(line)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")

def versioned_path(base_path: Path) -> Path:
    if OVERWRITE_OUTPUT or not base_path.exists():
        return base_path
    stem = base_path.stem
    suffix = base_path.suffix
    parent = base_path.parent
    v = 2
    candidate = parent / f"{stem}_v{v}{suffix}"
    while candidate.exists():
        v += 1
        candidate = parent / f"{stem}_v{v}{suffix}"
    return candidate

def list_images(folder: Path):
    exts = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".heic")
    return [f for f in folder.iterdir() if f.suffix.lower() in exts]

# ==============================================================
# GEOMETRY HELPERS
# ==============================================================

def order_points(pts: np.ndarray) -> np.ndarray:
    """Order quadrilateral points as TL, TR, BR, BL."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def warp_perspective(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxWidth = int(max(widthA, widthB))
    maxHeight = int(max(heightA, heightB))
    dst = np.array([[0, 0],
                    [maxWidth - 1, 0],
                    [maxWidth - 1, maxHeight - 1],
                    [0, maxHeight - 1]],
                   dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def correct_orientation(img: np.ndarray) -> np.ndarray:
    """Ensure portrait orientation (height > width)."""
    h, w = img.shape[:2]
    if w > h:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    return img

# ==============================================================
# TIER 1: REMBG BACKGROUND REMOVAL (PRIMARY METHOD)
# ==============================================================

def find_artwork_with_rembg(image: np.ndarray):
    """
    Use rembg to remove background and find the artwork bounding box.
    Returns (x1, y1, x2, y2) bbox or None.
    """
    if not REMBG_AVAILABLE:
        return None

    # Convert BGR to RGB for PIL
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb).convert('RGBA')

    # Remove background
    try:
        no_bg = remove(pil_img)
    except Exception as e:
        log(f"rembg error: {e}")
        return None

    # Find bounding box of non-transparent pixels
    bbox = no_bg.getbbox()
    if bbox is None:
        return None

    # Validate the bbox makes sense
    x1, y1, x2, y2 = bbox
    crop_w = x2 - x1
    crop_h = y2 - y1

    if crop_w <= 0 or crop_h <= 0:
        return None

    # Check area ratio
    orig_area = image.shape[0] * image.shape[1]
    crop_area = crop_w * crop_h
    area_ratio = crop_area / orig_area

    if area_ratio < MIN_AREA_RATIO:
        log(f"rembg crop too small: {area_ratio:.2%} of original")
        return None

    if area_ratio > MAX_AREA_RATIO:
        # Almost entire image - might not have found the artwork properly
        # Still return it, but log a warning
        log(f"rembg crop is {area_ratio:.2%} of original (nearly full image)")

    return bbox


def crop_with_rembg(image: np.ndarray):
    """
    Use rembg to isolate artwork and return the cropped image.
    Returns cropped BGR image or None.
    """
    bbox = find_artwork_with_rembg(image)
    if bbox is None:
        return None

    x1, y1, x2, y2 = bbox
    crop = image[y1:y2, x1:x2]

    return crop

# ==============================================================
# TIER 2: CLASSICAL CV RECTANGLE DETECTION (FALLBACK)
# ==============================================================

def find_best_rectangle_cv(image: np.ndarray):
    """
    Use edges + contours to find the largest plausible rectangular artwork.
    Returns cropped image or None.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Try multiple edge detection parameters
    edged = cv2.Canny(gray, 50, 150)
    edged = cv2.dilate(edged, None, iterations=2)
    edged = cv2.erode(edged, None, iterations=1)

    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    h, w = image.shape[:2]
    img_area = h * w

    best_pts = None
    best_score = -1

    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        if peri < 100:   # ignore tiny shapes
            continue

        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        # If not 4 points, squeeze with minAreaRect
        if len(approx) < 4:
            continue
        if len(approx) > 4:
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            pts = box
        else:
            pts = approx.reshape(4, 2)

        # area
        area = cv2.contourArea(pts.astype(np.float32))
        if area <= 0:
            continue

        area_ratio = area / img_area
        if area_ratio < MIN_AREA_RATIO or area_ratio > MAX_AREA_RATIO:
            continue

        # score: bigger area is better
        score = area_ratio * 10.0
        if score > best_score:
            best_score = score
            best_pts = pts

    if best_pts is None:
        return None

    # Warp perspective to get rectangular crop
    crop = warp_perspective(image, best_pts.astype(np.float32))
    return crop

# ==============================================================
# VALIDATION & OUTPUT
# ==============================================================

def validate_crop(original: np.ndarray, crop: np.ndarray):
    """Validate the crop is reasonable."""
    oh, ow = original.shape[:2]
    ch, cw = crop.shape[:2]
    if ch == 0 or cw == 0:
        return False, "Zero-dimension crop"

    area_ratio = (ch * cw) / (oh * ow)
    if area_ratio < MIN_AREA_RATIO:
        return False, f"Area ratio {area_ratio:.2%} too small (min {MIN_AREA_RATIO:.0%})"

    # No aspect ratio restriction - artwork can be any shape

    return True, "OK"

def add_padding(img: np.ndarray, pct: float = PADDING_PERCENT) -> np.ndarray:
    h, w = img.shape[:2]
    pad_h = int(h * pct)
    pad_w = int(w * pct)
    return cv2.copyMakeBorder(
        img, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, value=(255, 255, 255)
    )

def resize_for_ebay(img: np.ndarray, longest: int = EBAY_LONGEST_PX) -> np.ndarray:
    h, w = img.shape[:2]
    scale = longest / max(h, w)
    if scale >= 1.0:
        return img  # don't upscale
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

def save_outputs(base_name: str, crop_bgr: np.ndarray):
    """Save the three output versions of the crop."""
    stem = Path(base_name).stem

    # master JPEG - full resolution tight crop
    master_path = versioned_path(OUTPUT_MASTER / f"{stem}_master.jpg")
    cv2.imwrite(str(master_path), crop_bgr, [cv2.IMWRITE_JPEG_QUALITY, 100])

    # eBay JPEG (padded + resized)
    ebay_img = add_padding(crop_bgr)
    ebay_img = resize_for_ebay(ebay_img)
    ebay_path = versioned_path(OUTPUT_EBAY / f"{stem}_ebay.jpg")
    cv2.imwrite(str(ebay_path), ebay_img, [cv2.IMWRITE_JPEG_QUALITY, EBAY_JPEG_QUALITY])

    # PNG (for 3D framing / further work)
    png_path = versioned_path(OUTPUT_PNG / f"{stem}.png")
    # convert BGR to RGB
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    pil_img.save(str(png_path))

    return master_path, ebay_path, png_path

# ==============================================================
# MAIN PER-IMAGE PIPELINE
# ==============================================================

def process_single_image(filepath: Path):
    """
    Process a single image file.
    Returns (master_path, ebay_path, png_path) on success, or None on failure.
    """
    filename = filepath.name
    original = cv2.imread(str(filepath))
    if original is None:
        log(f"{filename}: could not read image, sending to quarantine.")
        filepath.replace(QUARANTINE_DIR / filename)
        return None

    h, w = original.shape[:2]
    log_prefix = f"{filename}:"

    crop = None

    # --- Tier 1: rembg background removal (primary) ---
    if REMBG_AVAILABLE:
        log(f"{log_prefix} trying rembg detection...")
        crop = crop_with_rembg(original)
        if crop is not None:
            log(f"{log_prefix} rembg SUCCESS - found artwork")

    # --- Tier 2: Classical CV fallback ---
    if crop is None and ENABLE_CV_FALLBACK:
        log(f"{log_prefix} rembg failed, trying CV fallback...")
        crop = find_best_rectangle_cv(original)
        if crop is not None:
            log(f"{log_prefix} CV fallback SUCCESS")

    if crop is None:
        # total failure - save debug and quarantine
        debug_path = DEBUG_DIR / filename
        cv2.imwrite(str(debug_path), original)
        log(f"{log_prefix} no artwork detected; moved to quarantine.")
        filepath.replace(QUARANTINE_DIR / filename)
        return None

    # validate
    valid, reason = validate_crop(original, crop)
    if not valid:
        debug_path = DEBUG_DIR / filename
        cv2.imwrite(str(debug_path), crop)
        log(f"{log_prefix} invalid crop -> {reason}; moved to quarantine.")
        filepath.replace(QUARANTINE_DIR / filename)
        return None

    # success
    master_path, ebay_path, png_path = save_outputs(filename, crop)
    log(f"{log_prefix} SUCCESS")
    log(f"    master: {master_path}")
    log(f"    ebay: {ebay_path}")
    log(f"    png: {png_path}")

    return master_path, ebay_path, png_path

# ==============================================================
# STANDALONE FUNCTION FOR INTEGRATION
# ==============================================================

def detect_and_crop_artwork(image_path: str):
    """
    Standalone function to detect and crop artwork from a single image.
    Can be called from other scripts (like MASTER_CORNER_DETAIL_SYSTEM.py).

    Args:
        image_path: Path to input image

    Returns:
        tuple: (crop_bgr, success) where crop_bgr is the cropped image as numpy array
               or (None, False) on failure
    """
    original = cv2.imread(image_path)
    if original is None:
        return None, False

    crop = None

    # Try Tier 1: rembg
    if REMBG_AVAILABLE:
        crop = crop_with_rembg(original)

    # Try Tier 2: CV fallback
    if crop is None and ENABLE_CV_FALLBACK:
        crop = find_best_rectangle_cv(original)

    if crop is None:
        return None, False

    # Validate
    valid, reason = validate_crop(original, crop)
    if not valid:
        return None, False

    return crop, True


def get_tight_crop_and_padded(image_path: str, padding_pct: float = PADDING_PERCENT):
    """
    Get both tight crop and padded version for integration with other systems.

    Args:
        image_path: Path to input image
        padding_pct: Padding percentage (default 7%)

    Returns:
        tuple: (tight_crop_pil, padded_pil, success)
               Both are PIL Images in RGB format, or (None, None, False) on failure
    """
    crop_bgr, success = detect_and_crop_artwork(image_path)
    if not success:
        return None, None, False

    # Convert to PIL RGB
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    tight_crop_pil = Image.fromarray(rgb)

    # Create padded version
    padded_bgr = add_padding(crop_bgr, padding_pct)
    padded_rgb = cv2.cvtColor(padded_bgr, cv2.COLOR_BGR2RGB)
    padded_pil = Image.fromarray(padded_rgb)

    return tight_crop_pil, padded_pil, True

# ==============================================================
# ENTRY POINT
# ==============================================================

def main():
    ensure_dirs()
    images = list_images(INPUT_DIR)
    if not images:
        log(f"No images found in {INPUT_DIR}. Put your photos there and rerun.")
        return

    log(f"Starting processing of {len(images)} image(s).")
    log(f"Input: {INPUT_DIR}")
    log(f"Output master: {OUTPUT_MASTER}")
    log(f"Output eBay: {OUTPUT_EBAY}")
    log(f"Output PNG: {OUTPUT_PNG}")

    success_count = 0
    fail_count = 0

    for filepath in tqdm(images, desc="Processing"):
        try:
            result = process_single_image(filepath)
            if result:
                success_count += 1
            else:
                fail_count += 1
        except Exception as e:
            log(f"{filepath.name}: UNHANDLED ERROR -> {e}")
            fail_count += 1
            # Try to move to quarantine
            if filepath.exists():
                filepath.replace(QUARANTINE_DIR / filepath.name)

    log("=== RUN COMPLETE ===")
    log(f"Success: {success_count}, Failed: {fail_count}")

if __name__ == "__main__":
    main()
