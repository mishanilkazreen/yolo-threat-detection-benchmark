"""Unit tests for configuration validation."""

import pytest

from src.explainability.xai.config import XAIConfig


class TestConfigValidation:
    """Test configuration validation for target layers."""

    def test_missing_variant_raises_keyerror(self):
        """Test that missing model variant in target_layers raises KeyError."""
        # Create config with only yolov8 target layer
        config = XAIConfig(enabled=True, target_layers={"yolov8": "model.9"})

        # Try to get target layer for yolov11 (not in config)
        with pytest.raises(KeyError) as exc_info:
            config.get_target_layer("yolov11n", strict=True)

        # Verify error message contains the missing variant name
        assert "yolov11" in str(exc_info.value)

    def test_existing_variant_returns_layer(self):
        """Test that existing model variant returns correct target layer."""
        config = XAIConfig(enabled=True, target_layers={"yolov8": "model.9", "yolov11": "model.10"})

        # Should return the configured layer
        layer = config.get_target_layer("yolov11n", strict=True)
        assert layer == "model.10"

    def test_non_strict_mode_uses_fallback(self):
        """Test that non-strict mode uses fallback for missing variants."""
        config = XAIConfig(enabled=True, target_layers={"yolov8": "model.9"})

        # Should use fallback without raising error
        layer = config.get_target_layer("yolov11n", strict=False)
        assert layer is not None  # Should return some fallback value

    def test_partial_match_detection(self):
        """Test that partial architecture name matches work."""
        config = XAIConfig(
            enabled=True, target_layers={"yolov11": "model.9", "yolov12": "model.10"}
        )

        # Test various model name formats
        assert config.get_target_layer("yolov11n", strict=True) == "model.9"
        assert config.get_target_layer("yolov11s", strict=True) == "model.9"
        assert config.get_target_layer("yolov12x", strict=True) == "model.10"

    def test_case_insensitive_matching(self):
        """Test that architecture matching is case-insensitive."""
        config = XAIConfig(enabled=True, target_layers={"yolov11": "model.9"})

        # Should work with different cases
        assert config.get_target_layer("YOLOv11n", strict=True) == "model.9"
        assert config.get_target_layer("YOLOV11N", strict=True) == "model.9"
