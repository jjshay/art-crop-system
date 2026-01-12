"""
Tests for Art Crop System
"""
import pytest
import os
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
        assert "ai_models" in config, "Config should have ai_models section"
        assert "output_settings" in config, "Config should have output_settings section"

    def test_config_has_required_fields(self):
        """Verify config has all required fields"""
        config_path = Path(__file__).parent.parent / "examples" / "crop_config.json"
        with open(config_path) as f:
            config = json.load(f)

        required_fields = ["ai_models", "output_settings", "detail_shots"]
        for field in required_fields:
            assert field in config, f"Config missing required field: {field}"


class TestSampleData:
    """Test sample data files"""

    def test_wall_photo_exists(self):
        """Verify sample wall photo exists"""
        photo_path = Path(__file__).parent.parent / "examples" / "wall_photo.jpg"
        assert photo_path.exists(), "Sample wall photo should exist"

    def test_wall_photo_is_image(self):
        """Verify wall photo is a valid image"""
        photo_path = Path(__file__).parent.parent / "examples" / "wall_photo.jpg"
        # Check file has content and starts with JPEG magic bytes
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

        assert "input_image" in report, "Report should have input_image"
        assert "detected_artwork" in report, "Report should have detected_artwork"
        assert "crops_generated" in report, "Report should have crops_generated"


class TestDetailShots:
    """Test detail shot generation specs"""

    def test_detail_shot_types(self):
        """Verify all detail shot types are defined"""
        config_path = Path(__file__).parent.parent / "examples" / "crop_config.json"
        with open(config_path) as f:
            config = json.load(f)

        expected_shots = ["full", "top_left", "top_right", "bottom_left",
                         "bottom_right", "center", "signature", "edition"]

        for shot in expected_shots:
            assert shot in config["detail_shots"], f"Missing detail shot type: {shot}"
