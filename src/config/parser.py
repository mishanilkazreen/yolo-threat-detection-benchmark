"""Configuration parser for YAML configuration files."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class TrainingConfig:
    """Training configuration parameters."""

    epochs: int | None = None  # Optional if epochs_per_round is provided
    patience: int = 10
    image_size: int = 640
    device: str = "auto"  # Default to auto-detection
    runs: int = 1
    seeds: list[int] | None = None
    # Incremental training fields
    rounds: int | None = None
    epochs_per_round: int | None = None
    # Hyperparameters
    batch_size: int | None = None
    optimizer: str | None = None
    lr0: float | None = None
    lrf: float | None = None
    # Baseline fields
    run_baseline: bool = False
    baseline_epochs: int = 50


@dataclass
class ModelConfig:
    """Model configuration parameters."""

    name: str
    weights: str


@dataclass
class DataConfig:
    """Data configuration parameters."""

    yaml_path: str
    # Incremental training fields
    train_init_percentage: float | None = None
    iou_threshold: float | None = None
    # Note: Dataset uses pre-existing train/val/test splits from Roboflow download.
    # train_init_percentage controls the initial training set size within the training split.



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
        # epochs is optional if epochs_per_round is provided
        epochs_per_round = data.get("epochs_per_round")
        epochs = data.get("epochs")

        if epochs is None and epochs_per_round is None:
            raise ConfigurationParseError(
                "Either 'epochs' or 'epochs_per_round' must be specified"
            )

        # Required fields (epochs is NOT required if epochs_per_round is provided)
        required_fields = ["patience", "image_size"]
        missing_fields = [f for f in required_fields if f not in data]
        if missing_fields:
            raise ConfigurationParseError(
                f"Missing required training fields: {', '.join(missing_fields)}"
            )

        # Validate epochs if provided
        if epochs is not None:
            try:
                epochs = int(epochs)
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

        # Incremental training optional fields
        rounds = data.get("rounds")
        if rounds is not None:
            rounds = int(rounds)
            if rounds <= 0:
                raise ConfigurationParseError("rounds must be positive")

        if epochs_per_round is not None:
            epochs_per_round = int(epochs_per_round)
            if epochs_per_round <= 0:
                raise ConfigurationParseError("epochs_per_round must be positive")

        # Hyperparameters
        batch_size = data.get("batch_size")
        if batch_size is not None:
            batch_size = int(batch_size)
            if batch_size <= 0:
                raise ConfigurationParseError("batch_size must be positive")

        optimizer = data.get("optimizer")
        if optimizer is not None:
            optimizer = str(optimizer)

        lr0 = data.get("lr0")
        if lr0 is not None:
            lr0 = float(lr0)
            if lr0 <= 0:
                raise ConfigurationParseError("lr0 must be positive")

        lrf = data.get("lrf")
        if lrf is not None:
            lrf = float(lrf)
            if lrf <= 0:
                raise ConfigurationParseError("lrf must be positive")

        # Baseline fields
        run_baseline = data.get("run_baseline", False)
        if not isinstance(run_baseline, bool):
            raise ConfigurationParseError(
                f"run_baseline must be a boolean, got: {type(run_baseline).__name__}"
            )

        baseline_epochs = data.get("baseline_epochs", 50)
        try:
            baseline_epochs = int(baseline_epochs)
            if baseline_epochs <= 0:
                raise ConfigurationParseError("baseline_epochs must be a positive integer")
        except (ValueError, TypeError):
            raise ConfigurationParseError(
                f"baseline_epochs must be a positive integer, got: {data['baseline_epochs']}"
            )

        return TrainingConfig(
            epochs=epochs,
            patience=patience,
            image_size=image_size,
            device=device,
            runs=runs,
            seeds=seeds,
            rounds=rounds,
            epochs_per_round=epochs_per_round,
            batch_size=batch_size,
            optimizer=optimizer,
            lr0=lr0,
            lrf=lrf,
            run_baseline=run_baseline,
            baseline_epochs=baseline_epochs,
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

        # Incremental training optional fields
        train_init_percentage = data.get("train_init_percentage")
        if train_init_percentage is not None:
            train_init_percentage = float(train_init_percentage)
            if not (0.0 < train_init_percentage <= 1.0):
                raise ConfigurationParseError("train_init_percentage must be in (0, 1]")

        iou_threshold = data.get("iou_threshold")
        if iou_threshold is not None:
            iou_threshold = float(iou_threshold)
            if not (0.0 <= iou_threshold <= 1.0):
                raise ConfigurationParseError("iou_threshold must be in [0, 1]")

        return DataConfig(
            yaml_path=yaml_path,
            train_init_percentage=train_init_percentage,
            iou_threshold=iou_threshold,
        )
