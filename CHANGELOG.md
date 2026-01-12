# Changelog

All notable changes to the Art Crop System project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.0] - 2025-01-11

### Added

- Multi-tier AI artwork detection using rembg, GPT-4V, Claude, Gemini, and Grok
- Automatic artwork boundary detection with multi-AI consensus
- Perfect cropping to remove frames, walls, and shadows from artwork photos
- Generation of 8 detail shots per artwork (full, corners, center, signature, edition)
- Signature detection and closeup extraction
- Edition number detection for limited prints
- Quality validation with blur, lighting, and angle assessment
- OpenCV edge detection as fallback for challenging images
- Batch processing for entire folders
- Google Drive integration for direct upload of results
- Configurable crop settings via JSON configuration
- Demo mode that works without API keys using rembg only
- Visual showcase script for quick demonstration
- Comprehensive test suite
- CI workflow for automated testing
