"""Unit tests for configuration validation."""

import pytest

from src.explainability.xai.config import XAIConfig


class TestConfigValidation:
    """Test configuration validation for target layers."""

    def test_explicit_target_layer_returned_directly(self):
        """Explicit target_layer is returned as-is regardless of architecture."""
        config = XAIConfig(enabled=True, target_layer="model.9")
        assert config.get_target_layer("yolov11n") == "model.9"
        assert config.get_target_layer("yolov8n") == "model.9"
        assert config.get_target_layer(None) == "model.9"

    def test_fallback_to_architecture_map(self):
        """When target_layer is None the correct default per-arch layer is returned."""
        config = XAIConfig(enabled=True)  # target_layer=None
        assert config.get_target_layer("yolov11n") == "model.22"
        assert config.get_target_layer("yolov8n") == "model.18"
        assert config.get_target_layer("yolov12s") == "model.20"
        assert config.get_target_layer("yolo26m") == "model.22"

    def test_unknown_architecture_returns_default(self):
        """Unknown architecture falls back to yolov11 default (non-strict)."""
        config = XAIConfig(enabled=True)
        assert config.get_target_layer("unknown_model") == "model.22"

    def test_strict_mode_raises_for_unknown_arch(self):
        """Strict mode raises KeyError for unknown architecture when target_layer not set."""
        config = XAIConfig(enabled=True)
        with pytest.raises(KeyError):
            config.get_target_layer("completely_unknown", strict=True)

    def test_strict_mode_succeeds_when_target_layer_set(self):
        """Strict mode does not raise when explicit target_layer is configured."""
        config = XAIConfig(enabled=True, target_layer="model.15")
        assert config.get_target_layer("completely_unknown", strict=True) == "model.15"

    def test_partial_match_detection(self):
        """Partial architecture name matches work when using arch fallback."""
        config = XAIConfig(enabled=True)
        assert config.get_target_layer("yolov11n") == "model.22"
        assert config.get_target_layer("yolov11s") == "model.22"
        assert config.get_target_layer("yolov12x") == "model.20"

    def test_case_insensitive_matching(self):
        """Architecture matching is case-insensitive."""
        config = XAIConfig(enabled=True)
        assert config.get_target_layer("YOLOv11n") == "model.22"
        assert config.get_target_layer("YOLOV11N") == "model.22"
