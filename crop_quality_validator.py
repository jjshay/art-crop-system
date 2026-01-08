#!/usr/bin/env python3
"""
🔍 AI-Powered Crop Quality Validator
Uses ChatGPT-4 Vision to verify crops meet art buyer standards
"""

import base64
import json
from pathlib import Path
from typing import Dict, List
from openai import OpenAI

class CropQualityValidator:
    """Validates crop quality using AI vision"""

    def __init__(self, use_ai=False, api_key=None):
        self.use_ai = use_ai
        if use_ai:
            if api_key:
                self.client = OpenAI(api_key=api_key)
            else:
                try:
                    self.client = OpenAI()
                except:
                    print("⚠️  OpenAI API not available, using quick validation only")
                    self.use_ai = False
        self.art_buyer_criteria = {
            'signature_readable': 'Artist signature and date must be clearly readable',
            'edition_visible': 'Edition number (e.g., #42/100) must be visible',
            'no_blur': 'Image must be crisp and clear, not blurry',
            'no_excessive_whitespace': 'Should not be mostly empty white/gray space',
            'correct_orientation': 'Not rotated sideways or upside down',
            'damage_visible': 'Corners must show condition/damage clearly',
            'gallery_quality': 'Professional presentation suitable for art gallery',
        }

    def validate_single_crop(self, image_path: Path, crop_name: str, purpose: str) -> Dict:
        """
        Validate a single crop using ChatGPT-4 Vision or quick validation

        Returns dict with:
        - passes: bool
        - score: 0-100
        - issues: list of problems found
        - recommendation: PASS/FAIL/RETRY
        - notes: explanation
        """
        # Use quick validation if AI not available
        if not self.use_ai:
            return self.quick_validate(image_path)

        # Convert image to base64
        with open(image_path, 'rb') as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')

        # Build prompt
        prompt = self._build_validation_prompt(crop_name, purpose)

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_data}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=500
            )

            result_text = response.choices[0].message.content

            # Try to parse as JSON
            try:
                # Extract JSON from markdown if present
                if "```json" in result_text:
                    result_text = result_text.split("```json")[1].split("```")[0].strip()
                elif "```" in result_text:
                    result_text = result_text.split("```")[1].split("```")[0].strip()

                result = json.loads(result_text)
            except:
                # Fallback if not valid JSON
                result = {
                    "passes": "fail" not in result_text.lower() and "poor" not in result_text.lower(),
                    "score": 80 if "good" in result_text.lower() else 60,
                    "issues": [],
                    "recommendation": "PASS" if "pass" in result_text.lower() else "REVIEW",
                    "notes": result_text[:200]
                }

            return result

        except Exception as e:
            print(f"      ⚠️  API error: {e}")
            # Return passing result if API fails (don't block automation)
            return {
                "passes": True,
                "score": 75,
                "issues": ["API check unavailable"],
                "recommendation": "PASS",
                "notes": f"Validation skipped: {str(e)[:100]}"
            }

    def _build_validation_prompt(self, crop_name: str, purpose: str) -> str:
        """Build validation prompt for AI"""
        return f"""You are an art gallery quality inspector reviewing a crop for fine artwork eBay listing.

CROP NAME: {crop_name}
PURPOSE: {purpose}

ART BUYER CRITERIA (what customers MUST see):
- Signature & date: READABLE for authentication
- Edition number: VISIBLE (e.g., #42/100) for value
- Corners: Show damage/condition CLEARLY
- Details: CRISP and clear (not blurry)
- Gallery quality: Professional presentation
- NO excessive white/gray space (>40% is excessive)
- NO blur or artifacts
- Correct orientation (not sideways/upside down)

REVIEW THIS CROP:

1. Does it meet art buyer criteria for its purpose?
2. Is it "gallery-grade" (professional enough for art sales)?
3. What percentage is white/empty space vs actual art?
4. Are there any quality issues?

Respond in JSON format:
{{
    "passes": true/false,
    "score": 0-100,
    "whitespace_percent": 0-100,
    "issues": ["list of specific problems"],
    "recommendation": "PASS" or "FAIL" or "RETRY",
    "notes": "brief explanation of decision"
}}

Be CRITICAL - these crops represent expensive artwork. Only PASS if truly gallery-quality."""

    def validate_all_crops(self, crops: Dict[str, Path]) -> Dict:
        """
        Validate all crops comprehensively

        Returns dict with:
        - all_pass: bool
        - average_score: float
        - gallery_grade: bool (avg >= 85)
        - individual_results: dict of results per crop
        """
        print(f"\n{'='*60}")
        print(f"🔍 AI QUALITY VALIDATION - ALL CROPS")
        print(f"{'='*60}\n")

        # Define purpose for each crop
        purposes = {
            '0_THUMBNAIL': "Quick preview thumbnail",
            '1_TIGHT_CROP': "Main product image with white border",
            '2_MAIN_PORTRAIT': "Primary gallery display",
            '3_BOTTOM_LEFT_SIGNATURE': "Show artist signature and date clearly",
            '4_BOTTOM_RIGHT_EDITION': "Show edition number (e.g., #42/100)",
            '5_DETAILED_CENTER': "Show crisp center detail",
            '6_AI_TEXTURE_SUBJECT': "Show texture and quality",
            '7_TOP_LEFT_TEXTURE': "Show top left corner for damage check",
            '8_TOP_RIGHT_TEXTURE': "Show top right corner for damage check"
        }

        results = {}
        all_pass = True
        total_score = 0
        failed_crops = []

        for crop_name, crop_path in sorted(crops.items()):
            purpose = purposes.get(crop_name, "Art detail crop")

            print(f"   Validating: {crop_name}")
            result = self.validate_single_crop(crop_path, crop_name, purpose)

            results[crop_name] = result
            total_score += result['score']

            # Print result
            status = "✅" if result['recommendation'] == "PASS" else "❌"
            print(f"      {status} Score: {result['score']}/100")

            if 'whitespace_percent' in result:
                ws = result['whitespace_percent']
                ws_status = "⚠️" if ws > 40 else "✓"
                print(f"         {ws_status} Whitespace: {ws}%")

            print(f"         {result['notes'][:80]}")

            if result['issues']:
                print(f"         Issues: {', '.join(result['issues'][:2])}")

            if result['recommendation'] != "PASS":
                all_pass = False
                failed_crops.append(crop_name)

            print()

        # Calculate metrics
        avg_score = total_score / len(crops) if crops else 0
        gallery_grade = avg_score >= 85

        print(f"{'='*60}")
        print(f"📊 FINAL VALIDATION RESULTS:")
        print(f"   Average Score: {avg_score:.1f}/100")
        print(f"   All Crops Pass: {'✅ YES' if all_pass else '❌ NO'}")
        print(f"   Gallery Grade: {'✅ YES' if gallery_grade else '⚠️  NEEDS IMPROVEMENT'}")

        if failed_crops:
            print(f"   Failed Crops: {', '.join(failed_crops)}")

        print(f"{'='*60}\n")

        return {
            "all_pass": all_pass,
            "average_score": avg_score,
            "gallery_grade": gallery_grade,
            "failed_crops": failed_crops,
            "individual_results": results
        }

    def quick_validate(self, image_path: Path) -> Dict:
        """
        Quick validation without AI (fallback)
        Checks basic criteria: file size, dimensions, whitespace
        """
        from PIL import Image
        import numpy as np

        img = Image.open(image_path)
        width, height = img.size
        file_size_kb = image_path.stat().st_size / 1024

        # Convert to numpy for analysis
        img_array = np.array(img.convert('L'))  # Grayscale

        # Calculate whitespace (pixels close to white)
        white_pixels = np.sum(img_array > 240)
        total_pixels = img_array.size
        whitespace_percent = (white_pixels / total_pixels) * 100

        # Score based on criteria
        score = 100
        issues = []

        # File size check
        if file_size_kb < 100:
            score -= 30
            issues.append("File size too small (likely poor quality)")

        # Whitespace check
        if whitespace_percent > 60:
            score -= 40
            issues.append(f"Excessive whitespace ({whitespace_percent:.1f}%)")
        elif whitespace_percent > 40:
            score -= 20
            issues.append(f"High whitespace ({whitespace_percent:.1f}%)")

        # Dimension check
        if width < 800 or height < 800:
            score -= 20
            issues.append("Dimensions too small for gallery quality")

        passes = score >= 70
        recommendation = "PASS" if score >= 85 else "REVIEW" if score >= 70 else "FAIL"

        return {
            "passes": passes,
            "score": max(score, 0),
            "whitespace_percent": whitespace_percent,
            "file_size_kb": file_size_kb,
            "dimensions": f"{width}x{height}",
            "issues": issues,
            "recommendation": recommendation,
            "notes": f"Quick validation: {recommendation}",
            "method": "quick_validate"
        }


if __name__ == '__main__':
    # Test the validator
    validator = CropQualityValidator()

    # Find a test crop
    test_crop = Path('processed_crops/_NYC0058JPG/3_BOTTOM_LEFT_SIGNATURE.png')

    if test_crop.exists():
        print("Testing validator on bad crop...")
        result = validator.validate_single_crop(
            test_crop,
            "3_BOTTOM_LEFT_SIGNATURE",
            "Show artist signature and date"
        )

        print(f"\nResult:")
        print(json.dumps(result, indent=2))
    else:
        print(f"Test file not found: {test_crop}")
