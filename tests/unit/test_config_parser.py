"""Unit tests for ConfigurationParser."""

from pathlib import Path
import tempfile

import pytest
import yaml

from src.config.parser import (
    ConfigurationParseError,
    ConfigurationParser,
)


class TestConfigurationParser:
    """Tests for ConfigurationParser."""

    def create_temp_config(self, config_dict):
        """Helper to create temporary config file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_dict, f)
            return f.name

    def test_parse_valid_minimal_config(self):
        """Test parsing valid minimal configuration."""
        config_dict = {
            "training": {"epochs": 100, "patience": 10, "image_size": 640, "device": "cuda"},
            "model": {"name": "yolov8n", "weights": "yolov8n.pt"},
            "data": {"yaml_path": "config/data/dataset.yaml"},
        }

        config_path = self.create_temp_config(config_dict)
        try:
            config = ConfigurationParser.parse(config_path)

            assert config.training.epochs == 100
            assert config.training.patience == 10
            assert config.training.image_size == 640
            assert config.training.device == "cuda"
            assert config.training.runs == 1  # Default
            assert config.training.seeds is None  # Default

            assert config.model.name == "yolov8n"
            assert config.model.weights == "yolov8n.pt"

            assert config.data.yaml_path == "config/data/dataset.yaml"
        finally:
            Path(config_path).unlink()

    def test_parse_valid_config_with_optional_fields(self):
        """Test parsing configuration with optional fields."""
        config_dict = {
            "training": {
                "epochs": 100,
                "patience": 10,
                "image_size": 640,
                "device": "cuda",
                "runs": 3,
                "seeds": [42, 123, 456],
            },
            "model": {"name": "yolov8n", "weights": "yolov8n.pt"},
            "data": {"yaml_path": "config/data/dataset.yaml"},
        }

        config_path = self.create_temp_config(config_dict)
        try:
            config = ConfigurationParser.parse(config_path)

            assert config.training.runs == 3
            assert config.training.seeds == [42, 123, 456]
        finally:
            Path(config_path).unlink()

    def test_parse_file_not_found(self):
        """Test parsing non-existent file raises error."""
        with pytest.raises(ConfigurationParseError, match="not found"):
            ConfigurationParser.parse("nonexistent.yaml")

    def test_parse_empty_file(self):
        """Test parsing empty file raises error."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            config_path = f.name

        try:
            with pytest.raises(ConfigurationParseError, match="empty"):
                ConfigurationParser.parse(config_path)
        finally:
            Path(config_path).unlink()

    def test_parse_invalid_yaml_syntax(self):
        """Test parsing file with invalid YAML syntax."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("invalid: yaml: syntax: here:")
            config_path = f.name

        try:
            with pytest.raises(ConfigurationParseError, match="Invalid YAML"):
                ConfigurationParser.parse(config_path)
        finally:
            Path(config_path).unlink()

    def test_parse_missing_training_section(self):
        """Test parsing config missing training section."""
        config_dict = {
            "model": {"name": "yolov8n", "weights": "yolov8n.pt"},
            "data": {"yaml_path": "config/data/dataset.yaml"},
        }

        config_path = self.create_temp_config(config_dict)
        try:
            with pytest.raises(
                ConfigurationParseError, match="Missing required sections.*training"
            ):
                ConfigurationParser.parse(config_path)
        finally:
            Path(config_path).unlink()

    def test_parse_missing_model_section(self):
        """Test parsing config missing model section."""
        config_dict = {
            "training": {"epochs": 100, "patience": 10, "image_size": 640, "device": "cuda"},
            "data": {"yaml_path": "config/data/dataset.yaml"},
        }

        config_path = self.create_temp_config(config_dict)
        try:
            with pytest.raises(ConfigurationParseError, match="Missing required sections.*model"):
                ConfigurationParser.parse(config_path)
        finally:
            Path(config_path).unlink()

    def test_parse_missing_data_section(self):
        """Test parsing config missing data section."""
        config_dict = {
            "training": {"epochs": 100, "patience": 10, "image_size": 640, "device": "cuda"},
            "model": {"name": "yolov8n", "weights": "yolov8n.pt"},
        }

        config_path = self.create_temp_config(config_dict)
        try:
            with pytest.raises(ConfigurationParseError, match="Missing required sections.*data"):
                ConfigurationParser.parse(config_path)
        finally:
            Path(config_path).unlink()

    def test_parse_missing_training_field_epochs(self):
        """Test parsing config missing epochs field."""
        config_dict = {
            "training": {"patience": 10, "image_size": 640, "device": "cuda"},
            "model": {"name": "yolov8n", "weights": "yolov8n.pt"},
            "data": {"yaml_path": "config/data/dataset.yaml"},
        }

        config_path = self.create_temp_config(config_dict)
        try:
            with pytest.raises(
                ConfigurationParseError,
                match="Either 'epochs' or 'epochs_per_round' must be specified",
            ):
                ConfigurationParser.parse(config_path)
        finally:
            Path(config_path).unlink()

    def test_parse_missing_model_field_name(self):
        """Test parsing config missing model name field."""
        config_dict = {
            "training": {"epochs": 100, "patience": 10, "image_size": 640, "device": "cuda"},
            "model": {"weights": "yolov8n.pt"},
            "data": {"yaml_path": "config/data/dataset.yaml"},
        }

        config_path = self.create_temp_config(config_dict)
        try:
            with pytest.raises(
                ConfigurationParseError, match="Missing required model fields.*name"
            ):
                ConfigurationParser.parse(config_path)
        finally:
            Path(config_path).unlink()

    def test_parse_missing_data_field_yaml_path(self):
        """Test parsing config missing yaml_path field."""
        config_dict = {
            "training": {"epochs": 100, "patience": 10, "image_size": 640, "device": "cuda"},
            "model": {"name": "yolov8n", "weights": "yolov8n.pt"},
            "data": {},
        }

        config_path = self.create_temp_config(config_dict)
        try:
            with pytest.raises(
                ConfigurationParseError, match="Missing required data fields.*yaml_path"
            ):
                ConfigurationParser.parse(config_path)
        finally:
            Path(config_path).unlink()

    def test_parse_invalid_epochs_negative(self):
        """Test parsing config with negative epochs."""
        config_dict = {
            "training": {"epochs": -10, "patience": 10, "image_size": 640, "device": "cuda"},
            "model": {"name": "yolov8n", "weights": "yolov8n.pt"},
            "data": {"yaml_path": "config/data/dataset.yaml"},
        }

        config_path = self.create_temp_config(config_dict)
        try:
            with pytest.raises(ConfigurationParseError, match="epochs must be positive"):
                ConfigurationParser.parse(config_path)
        finally:
            Path(config_path).unlink()

    def test_parse_invalid_epochs_non_numeric(self):
        """Test parsing config with non-numeric epochs."""
        config_dict = {
            "training": {"epochs": "abc", "patience": 10, "image_size": 640, "device": "cuda"},
            "model": {"name": "yolov8n", "weights": "yolov8n.pt"},
            "data": {"yaml_path": "config/data/dataset.yaml"},
        }

        config_path = self.create_temp_config(config_dict)
        try:
            with pytest.raises(ConfigurationParseError, match="epochs must be an integer"):
                ConfigurationParser.parse(config_path)
        finally:
            Path(config_path).unlink()

    def test_parse_invalid_patience_negative(self):
        """Test parsing config with negative patience."""
        config_dict = {
            "training": {"epochs": 100, "patience": -5, "image_size": 640, "device": "cuda"},
            "model": {"name": "yolov8n", "weights": "yolov8n.pt"},
            "data": {"yaml_path": "config/data/dataset.yaml"},
        }

        config_path = self.create_temp_config(config_dict)
        try:
            with pytest.raises(ConfigurationParseError, match="patience must be non-negative"):
                ConfigurationParser.parse(config_path)
        finally:
            Path(config_path).unlink()

    def test_parse_invalid_image_size_zero(self):
        """Test parsing config with zero image_size."""
        config_dict = {
            "training": {"epochs": 100, "patience": 10, "image_size": 0, "device": "cuda"},
            "model": {"name": "yolov8n", "weights": "yolov8n.pt"},
            "data": {"yaml_path": "config/data/dataset.yaml"},
        }

        config_path = self.create_temp_config(config_dict)
        try:
            with pytest.raises(ConfigurationParseError, match="image_size must be positive"):
                ConfigurationParser.parse(config_path)
        finally:
            Path(config_path).unlink()

    def test_parse_empty_device(self):
        """Test parsing config with empty device string."""
        config_dict = {
            "training": {"epochs": 100, "patience": 10, "image_size": 640, "device": ""},
            "model": {"name": "yolov8n", "weights": "yolov8n.pt"},
            "data": {"yaml_path": "config/data/dataset.yaml"},
        }

        config_path = self.create_temp_config(config_dict)
        try:
            with pytest.raises(ConfigurationParseError, match="device cannot be empty"):
                ConfigurationParser.parse(config_path)
        finally:
            Path(config_path).unlink()

    def test_parse_empty_model_name(self):
        """Test parsing config with empty model name."""
        config_dict = {
            "training": {"epochs": 100, "patience": 10, "image_size": 640, "device": "cuda"},
            "model": {"name": "", "weights": "yolov8n.pt"},
            "data": {"yaml_path": "config/data/dataset.yaml"},
        }

        config_path = self.create_temp_config(config_dict)
        try:
            with pytest.raises(ConfigurationParseError, match="model name cannot be empty"):
                ConfigurationParser.parse(config_path)
        finally:
            Path(config_path).unlink()


class TestYolov8nConfig:
    """Tests for yolov8n.yaml config correctness (Bug 1.3 fix)."""

    YOLOV8N_PATH = "config/models/yolov8n.yaml"

    def test_run_baseline_is_true(self):
        """yolov8n.yaml must have run_baseline == True."""
        config = ConfigurationParser.parse(self.YOLOV8N_PATH)
        assert config.training.run_baseline is True

    def test_baseline_epochs_is_50(self):
        """yolov8n.yaml must have baseline_epochs == 50."""
        config = ConfigurationParser.parse(self.YOLOV8N_PATH)
        assert config.training.baseline_epochs == 50

    def test_pre_existing_fields_preserved(self):
        """All pre-existing fields in yolov8n.yaml retain their original values."""
        config = ConfigurationParser.parse(self.YOLOV8N_PATH)

        # Training section pre-existing fields
        assert config.training.epochs_per_round == 10
        assert config.training.rounds == 5
        assert config.training.patience == 10
        assert config.training.image_size == 640
        assert config.training.batch_size == 16
        assert config.training.optimizer == "AdamW"
        assert config.training.lr0 == 0.001
        assert config.training.lrf == 0.1
        assert config.training.runs == 1
        assert config.training.seeds == [42]
        assert config.training.device == "0"

        # Model section
        assert config.model.name == "yolov8n"
        assert config.model.weights == "yolov8n.pt"

        # Data section
        assert config.data.yaml_path == "config/data/weapon_detection_data.yaml"
        assert config.data.train_init_percentage == 0.2
        assert config.data.iou_threshold == 0.5
