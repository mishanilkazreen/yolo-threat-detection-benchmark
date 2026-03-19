"""Unit tests for XAI configuration parsing."""

from pathlib import Path
import tempfile

import pytest
import yaml

from src.config.manager import ConfigurationManager
from src.config.parser import ConfigurationParseError, ConfigurationParser, XAIConfig
from src.config.serializer import ConfigurationSerializer


class TestXAIConfigurationParsing:
    """Test XAI configuration parsing in the main configuration system."""

    def test_parse_config_without_xai_section(self):
        """Test parsing configuration without XAI section (should use defaults)."""
        config_data = {
            "training": {
                "epochs": 50,
                "patience": 10,
                "image_size": 640,
                "device": "cuda",
            },
            "model": {
                "name": "yolov11n",
                "weights": "yolov11n.pt",
            },
            "data": {
                "yaml_path": "data.yaml",
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name

        try:
            config = ConfigurationParser.parse(config_path)

            # Should have default XAI config
            assert config.xai.enabled is False
            assert config.xai.methods["gradcam"] is True
            assert config.xai.methods["lrp"] is False
            assert config.xai.background_set_size == 75
        finally:
            Path(config_path).unlink()

    def test_parse_config_with_xai_section(self):
        """Test parsing configuration with XAI section."""
        config_data = {
            "training": {
                "epochs": 50,
                "patience": 10,
                "image_size": 640,
                "device": "cuda",
            },
            "model": {
                "name": "yolov11n",
                "weights": "yolov11n.pt",
            },
            "data": {
                "yaml_path": "data.yaml",
            },
            "xai": {
                "enabled": True,
                "methods": {
                    "gradcam": True,
                    "lrp": True,
                    "shap": False,
                },
                "sample_limit": 10,
                "background_set_size": 50,
                "output_overlays": False,
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name

        try:
            config = ConfigurationParser.parse(config_path)

            assert config.xai.enabled is True
            assert config.xai.methods["gradcam"] is True
            assert config.xai.methods["lrp"] is True
            assert config.xai.methods["shap"] is False
            assert config.xai.sample_limit == 10
            assert config.xai.background_set_size == 50
            assert config.xai.output_overlays is False
        finally:
            Path(config_path).unlink()

    def test_parse_config_invalid_xai_method(self):
        """Test parsing configuration with invalid XAI method."""
        config_data = {
            "training": {"epochs": 50, "patience": 10, "image_size": 640},
            "model": {"name": "yolov11n", "weights": "yolov11n.pt"},
            "data": {"yaml_path": "data.yaml"},
            "xai": {
                "enabled": True,
                "methods": {
                    "gradcam": True,
                    "invalid_method": True,  # Invalid method
                },
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name

        try:
            with pytest.raises(ConfigurationParseError, match="Invalid XAI methods"):
                ConfigurationParser.parse(config_path)
        finally:
            Path(config_path).unlink()

    def test_parse_config_invalid_sample_limit(self):
        """Test parsing configuration with invalid sample limit."""
        config_data = {
            "training": {"epochs": 50, "patience": 10, "image_size": 640},
            "model": {"name": "yolov11n", "weights": "yolov11n.pt"},
            "data": {"yaml_path": "data.yaml"},
            "xai": {
                "enabled": True,
                "sample_limit": -5,  # Invalid negative value
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name

        try:
            with pytest.raises(ConfigurationParseError, match="sample_limit must be positive"):
                ConfigurationParser.parse(config_path)
        finally:
            Path(config_path).unlink()

    def test_parse_config_invalid_enabled_type(self):
        """Test parsing configuration with invalid enabled type."""
        config_data = {
            "training": {"epochs": 50, "patience": 10, "image_size": 640},
            "model": {"name": "yolov11n", "weights": "yolov11n.pt"},
            "data": {"yaml_path": "data.yaml"},
            "xai": {
                "enabled": "yes",  # Should be boolean
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name

        try:
            with pytest.raises(ConfigurationParseError, match="enabled must be a boolean"):
                ConfigurationParser.parse(config_path)
        finally:
            Path(config_path).unlink()

    def test_parse_config_invalid_method_value_type(self):
        """Test parsing configuration with invalid method value type."""
        config_data = {
            "training": {"epochs": 50, "patience": 10, "image_size": 640},
            "model": {"name": "yolov11n", "weights": "yolov11n.pt"},
            "data": {"yaml_path": "data.yaml"},
            "xai": {
                "methods": {
                    "gradcam": "yes",  # Should be boolean
                },
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name

        try:
            with pytest.raises(ConfigurationParseError, match="gradcam must be a boolean"):
                ConfigurationParser.parse(config_path)
        finally:
            Path(config_path).unlink()

    def test_parse_config_invalid_background_set_size(self):
        """Test parsing configuration with invalid background set size."""
        config_data = {
            "training": {"epochs": 50, "patience": 10, "image_size": 640},
            "model": {"name": "yolov11n", "weights": "yolov11n.pt"},
            "data": {"yaml_path": "data.yaml"},
            "xai": {
                "background_set_size": 0,  # Should be positive
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name

        try:
            with pytest.raises(
                ConfigurationParseError, match="background_set_size must be positive"
            ):
                ConfigurationParser.parse(config_path)
        finally:
            Path(config_path).unlink()

    def test_parse_config_architecture_specific_target_layers(self):
        """Test parsing configuration with architecture-specific target layers."""
        config_data = {
            "training": {"epochs": 50, "patience": 10, "image_size": 640},
            "model": {"name": "yolov11n", "weights": "yolov11n.pt"},
            "data": {"yaml_path": "data.yaml"},
            "xai": {
                "enabled": True,
                "target_layers": {
                    "yolov11": "model.22",
                    "yolov12": "model.22",
                    "yolo26": "model.21",
                    "yolov8": "model.22",
                },
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name

        try:
            config = ConfigurationParser.parse(config_path)
            assert config.xai.target_layers["yolov11"] == "model.22"
            assert config.xai.target_layers["yolo26"] == "model.21"
        finally:
            Path(config_path).unlink()


class TestXAIConfigurationSerialization:
    """Test XAI configuration serialization."""

    def test_serialize_config_with_default_xai(self):
        """Test serializing configuration with default XAI settings (should not include XAI section)."""
        config_data = {
            "training": {"epochs": 50, "patience": 10, "image_size": 640},
            "model": {"name": "yolov11n", "weights": "yolov11n.pt"},
            "data": {"yaml_path": "data.yaml"},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name

        try:
            config = ConfigurationParser.parse(config_path)

            # Serialize back to dict
            serialized = ConfigurationSerializer.to_dict(config)

            # Should not include XAI section since it's all defaults
            assert "xai" not in serialized
        finally:
            Path(config_path).unlink()

    def test_serialize_config_with_custom_xai(self):
        """Test serializing configuration with custom XAI settings."""
        config_data = {
            "training": {"epochs": 50, "patience": 10, "image_size": 640},
            "model": {"name": "yolov11n", "weights": "yolov11n.pt"},
            "data": {"yaml_path": "data.yaml"},
            "xai": {
                "enabled": True,
                "methods": {"gradcam": True, "lrp": True, "shap": False},
                "sample_limit": 10,
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name

        try:
            config = ConfigurationParser.parse(config_path)

            # Serialize back to dict
            serialized = ConfigurationSerializer.to_dict(config)

            # Should include XAI section with non-default values
            assert "xai" in serialized
            assert serialized["xai"]["enabled"] is True
            assert serialized["xai"]["sample_limit"] == 10
            # Should include methods since they differ from defaults
            assert "methods" in serialized["xai"]
        finally:
            Path(config_path).unlink()

    def test_serialize_roundtrip_consistency(self):
        """Test that parsing and serializing maintains consistency."""
        config_data = {
            "training": {"epochs": 50, "patience": 10, "image_size": 640},
            "model": {"name": "yolov11n", "weights": "yolov11n.pt"},
            "data": {"yaml_path": "data.yaml"},
            "xai": {
                "enabled": True,
                "methods": {"gradcam": False, "lrp": True, "shap": True},
                "sample_limit": 5,
                "background_set_size": 100,
                "output_overlays": False,
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name

        try:
            # Parse original
            config1 = ConfigurationParser.parse(config_path)

            # Serialize and write to new file
            with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f2:
                ConfigurationSerializer.serialize(config1, f2.name)
                config_path2 = f2.name

            try:
                # Parse serialized version
                config2 = ConfigurationParser.parse(config_path2)

                # Should be identical
                assert config1.xai.enabled == config2.xai.enabled
                assert config1.xai.methods == config2.xai.methods
                assert config1.xai.sample_limit == config2.xai.sample_limit
                assert config1.xai.background_set_size == config2.xai.background_set_size
                assert config1.xai.output_overlays == config2.xai.output_overlays
            finally:
                Path(config_path2).unlink()
        finally:
            Path(config_path).unlink()


class TestXAIConfigurationValidation:
    """Test XAI configuration validation in ConfigurationManager."""

    def test_validate_xai_enabled_without_methods(self):
        """Test validation fails when XAI is enabled but no methods are enabled."""
        config_data = {
            "training": {"epochs": 50, "patience": 10, "image_size": 640},
            "model": {"name": "yolov11n", "weights": "yolov11n.pt"},
            "data": {"yaml_path": "data.yaml"},
            "xai": {
                "enabled": True,
                "methods": {"gradcam": False, "lrp": False, "shap": False},
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name

        try:
            with pytest.raises(
                ConfigurationParseError, match="XAI is enabled but no methods are enabled"
            ):
                ConfigurationManager.load_training_config(config_path)
        finally:
            Path(config_path).unlink()

    def test_validate_missing_target_layer_for_architecture(self):
        """Test validation fails when target layer is missing for the model architecture."""
        config_data = {
            "training": {"epochs": 50, "patience": 10, "image_size": 640},
            "model": {"name": "yolov11n", "weights": "yolov11n.pt"},
            "data": {"yaml_path": "data.yaml"},
            "xai": {
                "enabled": True,
                "methods": {"gradcam": True},
                "target_layers": {"yolov8": "model.22"},  # Missing yolov11
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name

        try:
            with pytest.raises(
                ConfigurationParseError,
                match="Missing target layer configuration for architecture 'yolov11'",
            ):
                ConfigurationManager.load_training_config(config_path)
        finally:
            Path(config_path).unlink()

    def test_validate_valid_xai_configuration(self):
        """Test validation passes for valid XAI configuration."""
        config_data = {
            "training": {"epochs": 50, "patience": 10, "image_size": 640},
            "model": {"name": "yolov11n", "weights": "yolov11n.pt"},
            "data": {"yaml_path": "data.yaml"},
            "xai": {
                "enabled": True,
                "methods": {"gradcam": True, "lrp": False, "shap": False},
                "sample_limit": 10,
                "target_layers": {
                    "yolov11": "model.22",
                    "yolov12": "model.22",
                    "yolo26": "model.21",
                    "yolov8": "model.22",
                },
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name

        try:
            # Should not raise any exceptions
            config = ConfigurationManager.load_training_config(config_path)
            assert config.xai.enabled is True
            assert config.xai.methods["gradcam"] is True
        finally:
            Path(config_path).unlink()


class TestXAIConfigDefaults:
    """Test XAI configuration default values."""

    def test_xai_config_defaults(self):
        """Test that XAIConfig has correct default values."""
        xai_config = XAIConfig()

        assert xai_config.enabled is False
        assert xai_config.methods["gradcam"] is True
        assert xai_config.methods["lrp"] is False
        assert xai_config.methods["shap"] is False
        assert xai_config.sample_limit is None
        assert xai_config.background_set_size == 75
        assert xai_config.output_overlays is True

        # Check default target layers
        assert "yolov11" in xai_config.target_layers
        assert "yolov12" in xai_config.target_layers
        assert "yolo26" in xai_config.target_layers
        assert "yolov8" in xai_config.target_layers
        assert xai_config.target_layers["yolov11"] == "model.9"
        assert xai_config.target_layers["yolo26"] == "model.9"
