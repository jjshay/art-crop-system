"""
Tests for Art Crop System
"""
import pytest
import json
from pathlib import Path


class TestCropConfig:
    """Test crop configuration handling"""

    def test_example_config_exists(self):
        """Verify example config file exists"""
        config_path = Path(__file__).parent.parent / "examples" / "crop_config.json"
        assert config_path.exists(), "Example crop config should exist"

    def test_example_config_valid_json(self):
        """Verify example config is valid JSON"""
        config_path = Path(__file__).parent.parent / "examples" / "crop_config.json"
        with open(config_path) as f:
            config = json.load(f)
        assert "detection_settings" in config, "Config should have detection_settings section"
        assert "output_settings" in config, "Config should have output_settings section"

    def test_config_has_required_fields(self):
        """Verify config has all required fields"""
        config_path = Path(__file__).parent.parent / "examples" / "crop_config.json"
        with open(config_path) as f:
            config = json.load(f)

        required_fields = ["input", "detection_settings", "output_settings"]
        for field in required_fields:
            assert field in config, f"Config missing required field: {field}"

    def test_detection_tiers_defined(self):
        """Verify all 4 detection tiers are configured"""
        config_path = Path(__file__).parent.parent / "examples" / "crop_config.json"
        with open(config_path) as f:
            config = json.load(f)

        detection = config["detection_settings"]
        assert "tier_1_rembg" in detection, "Should have tier_1_rembg"
        assert "tier_2_ai_consensus" in detection, "Should have tier_2_ai_consensus"
        assert "tier_3_cv_fallback" in detection, "Should have tier_3_cv_fallback"
        assert "tier_4_quality" in detection, "Should have tier_4_quality"


class TestSampleData:
    """Test sample data files"""

    def test_wall_photo_exists(self):
        """Verify sample wall photo exists"""
        photo_path = Path(__file__).parent.parent / "examples" / "wall_photo.jpg"
        assert photo_path.exists(), "Sample wall photo should exist"

    def test_wall_photo_is_image(self):
        """Verify wall photo is a valid image"""
        photo_path = Path(__file__).parent.parent / "examples" / "wall_photo.jpg"
        with open(photo_path, 'rb') as f:
            header = f.read(3)
        assert header[:2] == b'\xff\xd8', "File should be a valid JPEG"


class TestOutputFormat:
    """Test output format specifications"""

    def test_sample_output_exists(self):
        """Verify sample output directory exists"""
        output_path = Path(__file__).parent.parent / "sample_output"
        assert output_path.exists(), "Sample output directory should exist"

    def test_crop_report_format(self):
        """Verify crop report has correct format"""
        report_path = Path(__file__).parent.parent / "sample_output" / "crop_report.json"
        with open(report_path) as f:
            report = json.load(f)

        assert "input" in report, "Report should have input section"
        assert "detection" in report, "Report should have detection section"
        assert "outputs" in report, "Report should have outputs section"

    def test_report_has_all_tiers(self):
        """Verify report contains all detection tier results"""
        report_path = Path(__file__).parent.parent / "sample_output" / "crop_report.json"
        with open(report_path) as f:
            report = json.load(f)

        detection = report["detection"]
        assert "tier_1_rembg" in detection
        assert "tier_2_ai_consensus" in detection
        assert "tier_3_cv_fallback" in detection
        assert "tier_4_quality" in detection


class TestDetailShots:
    """Test detail shot generation specs"""

    def test_output_crops_defined(self):
        """Verify output crops are defined in config"""
        config_path = Path(__file__).parent.parent / "examples" / "crop_config.json"
        with open(config_path) as f:
            config = json.load(f)

        crops = config["output_settings"]["generate_crops"]
        crop_names = [c["name"] for c in crops]

        expected = ["full", "top_left", "top_right", "bottom_left",
                   "bottom_right", "center", "signature"]

        for shot in expected:
            assert shot in crop_names, f"Missing crop type: {shot}"

    def test_report_outputs_generated(self):
        """Verify report shows generated outputs"""
        report_path = Path(__file__).parent.parent / "sample_output" / "crop_report.json"
        with open(report_path) as f:
            report = json.load(f)

        outputs = report["outputs"]
        assert len(outputs) >= 7, "Should generate at least 7 output crops"
