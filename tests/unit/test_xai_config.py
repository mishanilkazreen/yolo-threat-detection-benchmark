"""Unit tests for XAI configuration classes."""

import pytest

from src.explainability.xai.config import XAIConfig, get_architecture_config, validate_target_layer


class TestXAIConfig:
    """Test XAI configuration functionality."""

    def test_default_config(self):
        """Test default XAI configuration."""
        config = XAIConfig()

        assert config.enabled is False
        assert config.methods["gradcam"] is True
        assert config.methods["lrp"] is False
        assert config.methods["shap"] is False
        assert config.sample_limit is None
        assert config.background_set_size == 75
        assert config.output_overlays is True

    def test_is_method_enabled(self):
        """Test method enabled checking."""
        # Disabled XAI
        config = XAIConfig(enabled=False)
        assert config.is_method_enabled("gradcam") is False

        # Enabled XAI with specific methods
        config = XAIConfig(enabled=True, methods={"gradcam": True, "lrp": False})
        assert config.is_method_enabled("gradcam") is True
        assert config.is_method_enabled("lrp") is False
        assert config.is_method_enabled("shap") is False  # Not in methods dict

    def test_get_target_layer(self):
        """Test target layer resolution for different architectures."""
        config = XAIConfig()

        assert config.get_target_layer("yolov11n") == "model.9"
        assert config.get_target_layer("yolov12s") == "model.9"
        assert config.get_target_layer("yolo26m") == "model.9"
        assert config.get_target_layer("yolov8l") == "model.9"

        # Unknown architecture should default to yolov11
        assert config.get_target_layer("unknown_model") == "model.9"

    def test_get_enabled_methods(self):
        """Test getting list of enabled methods."""
        # Disabled XAI
        config = XAIConfig(enabled=False)
        assert config.get_enabled_methods() == []

        # Enabled XAI with mixed methods
        config = XAIConfig(enabled=True, methods={"gradcam": True, "lrp": True, "shap": False})
        enabled = config.get_enabled_methods()
        assert "gradcam" in enabled
        assert "lrp" in enabled
        assert "shap" not in enabled


class TestArchitectureConfig:
    """Test architecture-specific configuration functions."""

    def test_get_architecture_config(self):
        """Test getting architecture-specific configuration."""
        # Test supported architectures
        yolov11_config = get_architecture_config("yolov11n")
        assert yolov11_config["gradcam_target"] == "model.9"
        assert "backbone_layers" in yolov11_config

        yolo26_config = get_architecture_config("yolo26s")
        assert yolo26_config["gradcam_target"] == "model.9"

        # Test case insensitive
        config = get_architecture_config("YOLOv12M")
        assert config["gradcam_target"] == "model.9"

    def test_get_architecture_config_unsupported(self):
        """Test error handling for unsupported architectures."""
        with pytest.raises(ValueError, match="Unsupported YOLO architecture"):
            get_architecture_config("unsupported_model")

    def test_validate_target_layer(self):
        """Test target layer validation."""
        # Valid layers
        assert validate_target_layer("yolov11n", "model.22") is True
        assert validate_target_layer("yolo26s", "model.21") is True
        assert validate_target_layer("yolov12m", "model.0") is True

        # Invalid layers
        assert validate_target_layer("yolov11n", "model.99") is False
        assert validate_target_layer("yolo26s", "invalid_layer") is False

        # Unsupported architecture
        assert validate_target_layer("unsupported", "model.22") is False
