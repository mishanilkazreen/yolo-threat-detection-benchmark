"""Configuration parser for YAML configuration files."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class TrainingConfig:
    """Training configuration parameters."""

    epochs: int
    patience: int
    image_size: int
    device: str = "auto"  # Default to auto-detection
    runs: int = 1
    seeds: list[int] | None = None


@dataclass
class ModelConfig:
    """Model configuration parameters."""

    name: str
    weights: str


@dataclass
class DataConfig:
    """Data configuration parameters."""

    yaml_path: str


@dataclass
class Configuration:
    """Complete configuration object."""

    training: TrainingConfig
    model: ModelConfig
    data: DataConfig


class ConfigurationParseError(Exception):
    """Exception raised when configuration parsing fails."""

    pass


class ConfigurationParser:
    """Parser for YAML configuration files to Configuration objects."""

    @staticmethod
    def parse(config_path: str | Path) -> Configuration:
        """
        Parse YAML configuration file into Configuration object.

        Args:
            config_path: Path to YAML configuration file

        Returns:
            Configuration object

        Raises:
            ConfigurationParseError: If parsing fails or validation fails
        """
        try:
            with open(config_path) as f:
                data = yaml.safe_load(f)
        except FileNotFoundError:
            raise ConfigurationParseError(f"Configuration file not found: {config_path}")
        except yaml.YAMLError as e:
            raise ConfigurationParseError(f"Invalid YAML syntax in {config_path}: {e}")
        except Exception as e:
            raise ConfigurationParseError(f"Failed to read configuration file {config_path}: {e}")

        if data is None:
            raise ConfigurationParseError(f"Configuration file is empty: {config_path}")

        return ConfigurationParser._parse_dict(data, config_path)

    @staticmethod
    def _parse_dict(data: dict[str, Any], source: str | Path) -> Configuration:
        """
        Parse dictionary into Configuration object with validation.

        Args:
            data: Dictionary from YAML
            source: Source file path for error messages

        Returns:
            Configuration object

        Raises:
            ConfigurationParseError: If required fields are missing or invalid
        """
        # Validate top-level sections
        required_sections = ["training", "model", "data"]
        missing_sections = [s for s in required_sections if s not in data]
        if missing_sections:
            raise ConfigurationParseError(
                f"Missing required sections in {source}: {', '.join(missing_sections)}"
            )

        # Parse training configuration
        try:
            training_config = ConfigurationParser._parse_training(data["training"], source)
        except Exception as e:
            raise ConfigurationParseError(f"Error parsing training section in {source}: {e}")

        # Parse model configuration
        try:
            model_config = ConfigurationParser._parse_model(data["model"], source)
        except Exception as e:
            raise ConfigurationParseError(f"Error parsing model section in {source}: {e}")

        # Parse data configuration
        try:
            data_config = ConfigurationParser._parse_data(data["data"], source)
        except Exception as e:
            raise ConfigurationParseError(f"Error parsing data section in {source}: {e}")

        return Configuration(training=training_config, model=model_config, data=data_config)

    @staticmethod
    def _parse_training(data: dict[str, Any], source: str | Path) -> TrainingConfig:
        """Parse training configuration section."""
        required_fields = ["epochs", "patience", "image_size"]
        missing_fields = [f for f in required_fields if f not in data]
        if missing_fields:
            raise ConfigurationParseError(
                f"Missing required training fields: {', '.join(missing_fields)}"
            )

        # Validate types and values
        try:
            epochs = int(data["epochs"])
            if epochs <= 0:
                raise ConfigurationParseError("epochs must be positive")
        except (ValueError, TypeError):
            raise ConfigurationParseError(f"epochs must be an integer, got: {data['epochs']}")

        try:
            patience = int(data["patience"])
            if patience < 0:
                raise ConfigurationParseError("patience must be non-negative")
        except (ValueError, TypeError):
            raise ConfigurationParseError(f"patience must be an integer, got: {data['patience']}")

        try:
            image_size = int(data["image_size"])
            if image_size <= 0:
                raise ConfigurationParseError("image_size must be positive")
        except (ValueError, TypeError):
            raise ConfigurationParseError(
                f"image_size must be an integer, got: {data['image_size']}"
            )

        # Optional device field (defaults to "auto")
        device = str(data.get("device", "auto"))
        if not device:
            raise ConfigurationParseError("device cannot be empty")

        # Optional fields
        runs = int(data.get("runs", 1))
        if runs <= 0:
            raise ConfigurationParseError("runs must be positive")

        seeds = data.get("seeds")
        if seeds is not None:
            if not isinstance(seeds, list):
                raise ConfigurationParseError("seeds must be a list")
            seeds = [int(s) for s in seeds]

        return TrainingConfig(
            epochs=epochs,
            patience=patience,
            image_size=image_size,
            device=device,
            runs=runs,
            seeds=seeds,
        )

    @staticmethod
    def _parse_model(data: dict[str, Any], source: str | Path) -> ModelConfig:
        """Parse model configuration section."""
        required_fields = ["name", "weights"]
        missing_fields = [f for f in required_fields if f not in data]
        if missing_fields:
            raise ConfigurationParseError(
                f"Missing required model fields: {', '.join(missing_fields)}"
            )

        name = str(data["name"])
        if not name:
            raise ConfigurationParseError("model name cannot be empty")

        weights = str(data["weights"])
        if not weights:
            raise ConfigurationParseError("model weights cannot be empty")

        return ModelConfig(name=name, weights=weights)

    @staticmethod
    def _parse_data(data: dict[str, Any], source: str | Path) -> DataConfig:
        """Parse data configuration section."""
        required_fields = ["yaml_path"]
        missing_fields = [f for f in required_fields if f not in data]
        if missing_fields:
            raise ConfigurationParseError(
                f"Missing required data fields: {', '.join(missing_fields)}"
            )

        yaml_path = str(data["yaml_path"])
        if not yaml_path:
            raise ConfigurationParseError("data yaml_path cannot be empty")

        return DataConfig(yaml_path=yaml_path)
