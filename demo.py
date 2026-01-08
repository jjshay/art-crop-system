#!/usr/bin/env python3
"""
Art Crop System Demo
Demonstrates artwork detection and cropping without needing API keys.

Run: python demo.py
"""

import os
from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageFilter
except ImportError:
    print("Installing Pillow...")
    os.system("pip install Pillow")
    from PIL import Image, ImageDraw, ImageFilter


def print_header(text):
    print(f"\n{'='*60}")
    print(f" {text}")
    print(f"{'='*60}\n")


def create_sample_photo():
    """Create a simulated photo of artwork on a wall"""
    print("Creating simulated artwork photo...")

    # Create a "wall" background
    wall = Image.new('RGB', (1200, 900), '#E8E4E0')
    draw = ImageDraw.Draw(wall)

    # Add wall texture
    for i in range(0, 900, 50):
        draw.line([(0, i), (1200, i)], fill='#E0DCD8', width=1)

    # Create the "artwork" (abstract art)
    artwork_size = (500, 650)
    artwork = Image.new('RGB', artwork_size, '#FAFAFA')
    art_draw = ImageDraw.Draw(artwork)

    # Add abstract shapes to artwork
    art_draw.ellipse([50, 100, 250, 300], fill='#E74C3C')
    art_draw.rectangle([200, 250, 450, 550], fill='#3498DB')
    art_draw.polygon([(250, 50), (400, 200), (150, 250)], fill='#F1C40F')

    # Add signature area
    art_draw.text((380, 600), "Artist '24", fill='#2C3E50')

    # Add frame around artwork
    frame_width = 30
    framed_size = (artwork_size[0] + frame_width*2, artwork_size[1] + frame_width*2)
    framed = Image.new('RGB', framed_size, '#1a1a1a')
    framed.paste(artwork, (frame_width, frame_width))

    # Add mat
    mat_width = 40
    with_mat = Image.new('RGB', (framed_size[0] + mat_width*2, framed_size[1] + mat_width*2), '#FFFFFF')
    with_mat.paste(framed, (mat_width, mat_width))

    # Outer frame
    outer_frame = 20
    final_frame = Image.new('RGB', (with_mat.size[0] + outer_frame*2, with_mat.size[1] + outer_frame*2), '#2C2C2C')
    final_frame.paste(with_mat, (outer_frame, outer_frame))

    # Place on wall (centered, slightly up)
    x = (1200 - final_frame.size[0]) // 2
    y = (900 - final_frame.size[1]) // 2 - 50
    wall.paste(final_frame, (x, y))

    # Add shadow
    shadow = Image.new('RGBA', final_frame.size, (0, 0, 0, 0))
    shadow_draw = ImageDraw.Draw(shadow)
    shadow_draw.rectangle([10, 10, final_frame.size[0]+10, final_frame.size[1]+10], fill=(0, 0, 0, 50))
    shadow = shadow.filter(ImageFilter.GaussianBlur(15))

    print(f"   Wall photo size: {wall.size}")
    print(f"   Artwork location: ({x}, {y})")

    return wall, artwork, (x, y, final_frame.size)


def simulate_detection(wall_photo, actual_bounds):
    """Simulate AI detection of artwork boundaries"""
    print_header("TIER 1: BACKGROUND REMOVAL (rembg)")

    print("Simulating rembg background removal...")
    print("   Detecting foreground objects...")
    print("   Found: 1 rectangular object (likely framed artwork)")

    # Simulate detected bounds (slightly off to show correction)
    x, y, size = actual_bounds
    detected = {
        'x': x + 5,
        'y': y - 3,
        'width': size[0] - 10,
        'height': size[1] + 5,
        'confidence': 0.87
    }

    print(f"   Initial bounds: ({detected['x']}, {detected['y']}) {detected['width']}x{detected['height']}")
    print(f"   Confidence: {detected['confidence']:.0%}")

    return detected


def simulate_ai_analysis(detected_bounds, actual_bounds):
    """Simulate multi-AI consensus analysis"""
    print_header("TIER 2: MULTI-AI VISION ANALYSIS")

    x, y, size = actual_bounds

    # Simulate each AI's analysis
    ai_results = {
        'GPT-4V': {
            'bounds': (x + 2, y + 1, size[0] - 4, size[1] - 2),
            'confidence': 0.94,
            'signature_found': True,
            'signature_location': 'bottom-right',
            'notes': 'Abstract art, black frame, white mat'
        },
        'Claude': {
            'bounds': (x + 1, y + 2, size[0] - 2, size[1] - 3),
            'confidence': 0.92,
            'signature_found': True,
            'signature_location': 'bottom-right',
            'notes': 'Contemporary abstract, well-lit'
        },
        'Gemini': {
            'bounds': (x + 3, y + 1, size[0] - 5, size[1] - 2),
            'confidence': 0.91,
            'signature_found': True,
            'signature_location': 'bottom-right',
            'notes': 'Geometric shapes, primary colors'
        },
        'Grok': {
            'bounds': (x + 2, y + 2, size[0] - 3, size[1] - 3),
            'confidence': 0.89,
            'signature_found': True,
            'signature_location': 'bottom-right',
            'notes': 'Looks like museum-quality framing'
        }
    }

    for ai_name, result in ai_results.items():
        print(f"\n{ai_name}:")
        print(f"   Bounds: {result['bounds']}")
        print(f"   Confidence: {result['confidence']:.0%}")
        print(f"   Signature: {result['signature_location'] if result['signature_found'] else 'Not found'}")
        print(f"   Notes: {result['notes']}")

    # Calculate consensus
    print("\n" + "-"*50)
    print("CONSENSUS CALCULATION:")
    avg_confidence = sum(r['confidence'] for r in ai_results.values()) / len(ai_results)
    print(f"   Average confidence: {avg_confidence:.0%}")
    print(f"   All AIs agree on signature location: YES")
    print(f"   Consensus reached: YES")

    return {
        'x': x,
        'y': y,
        'width': size[0],
        'height': size[1],
        'confidence': avg_confidence,
        'signature_area': (380, 600, 120, 30)  # Relative to artwork
    }


def simulate_quality_check(bounds):
    """Simulate quality validation"""
    print_header("TIER 4: QUALITY VALIDATION")

    checks = {
        'Blur Detection': ('PASS', 'Image is sharp (score: 0.92)'),
        'Lighting': ('PASS', 'Even lighting, no harsh shadows'),
        'Angle': ('PASS', 'Perpendicular to artwork (< 2° deviation)'),
        'Color Balance': ('PASS', 'White balance appears correct'),
        'Resolution': ('PASS', 'Sufficient for detail crops (> 500px)')
    }

    for check, (status, detail) in checks.items():
        status_icon = "OK" if status == "PASS" else "WARN"
        print(f"   [{status_icon}] {check}: {detail}")

    return True


def generate_detail_crops(artwork):
    """Generate 8 detail crop shots"""
    print_header("GENERATING DETAIL SHOTS")

    w, h = artwork.size

    crops = {
        'full': (0, 0, w, h),
        'top_left': (0, 0, w//2, h//2),
        'top_right': (w//2, 0, w, h//2),
        'bottom_left': (0, h//2, w//2, h),
        'bottom_right': (w//2, h//2, w, h),
        'center': (w//4, h//4, 3*w//4, 3*h//4),
        'signature': (int(w*0.6), int(h*0.85), w, h),
        'texture': (int(w*0.3), int(h*0.3), int(w*0.7), int(h*0.6))
    }

    results = {}
    for name, bounds in crops.items():
        crop = artwork.crop(bounds)
        results[name] = crop
        print(f"   {name:15s} : {crop.size[0]:4d} x {crop.size[1]:4d} px")

    return results


def save_outputs(wall, artwork, crops):
    """Save all outputs"""
    print_header("SAVING OUTPUT FILES")

    output_dir = Path("demo_output")
    output_dir.mkdir(exist_ok=True)

    # Save original photo
    wall.save(output_dir / "original_photo.jpg", quality=95)
    print(f"   Saved: original_photo.jpg")

    # Save extracted artwork
    artwork.save(output_dir / "extracted_artwork.jpg", quality=95)
    print(f"   Saved: extracted_artwork.jpg")

    # Save detail crops
    for name, crop in crops.items():
        filename = f"detail_{name}.jpg"
        crop.save(output_dir / filename, quality=95)
        print(f"   Saved: {filename}")

    return output_dir


def main():
    print_header("ART CROP SYSTEM - DEMO")

    print("This demo shows how the multi-tier crop system works")
    print("without requiring any API keys.\n")

    # Create sample photo
    print_header("CREATING SAMPLE ARTWORK PHOTO")
    wall_photo, artwork, bounds = create_sample_photo()

    # Tier 1: Background removal
    detected = simulate_detection(wall_photo, bounds)

    # Tier 2: AI Analysis
    consensus = simulate_ai_analysis(detected, bounds)

    # Skip Tier 3 (CV fallback) - not needed with good consensus
    print_header("TIER 3: CV EDGE DETECTION")
    print("   Skipped - AI consensus confidence > 90%")

    # Tier 4: Quality check
    simulate_quality_check(consensus)

    # Generate crops
    crops = generate_detail_crops(artwork)

    # Save outputs
    output_dir = save_outputs(wall_photo, artwork, crops)

    print_header("SUMMARY")
    print(f"Input: 1 photo of artwork on wall")
    print(f"Output: 1 extracted artwork + 8 detail crops")
    print(f"Location: {output_dir}/")
    print(f"\nFiles generated:")
    print(f"   - original_photo.jpg (simulated input)")
    print(f"   - extracted_artwork.jpg (clean crop)")
    print(f"   - detail_*.jpg (8 detail shots)")

    print_header("NEXT STEPS")
    print("To use with real photos:")
    print("   1. Install rembg: pip install rembg")
    print("   2. Add API keys to .env file")
    print("   3. Run: python ai_art_crop_system.py")

    print_header("DEMO COMPLETE")


if __name__ == "__main__":
    main()
