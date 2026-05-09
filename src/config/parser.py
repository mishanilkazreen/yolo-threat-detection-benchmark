"""Configuration parser for YAML configuration files."""

from dataclasses import dataclass, field
import logging
from pathlib import Path
from typing import Any

import yaml

from src.explainability.xai.config import XAIConfig

logger = logging.getLogger(__name__)


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
    # Augmentation fields (paper defaults)
    mosaic: float = 1.0
    scale: float = 0.5
    fliplr: float = 0.5
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4


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
    xai: XAIConfig = field(default_factory=XAIConfig)


class ConfigurationParseError(Exception):
    """Exception raised when configuration parsing fails."""


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
            with open(config_path, encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except FileNotFoundError as e:
            raise ConfigurationParseError(f"Configuration file not found: {config_path}") from e
        except yaml.YAMLError as e:
            raise ConfigurationParseError(f"Invalid YAML syntax in {config_path}: {e}") from e
        except Exception as e:
            raise ConfigurationParseError(
                f"Failed to read configuration file {config_path}: {e}"
            ) from e

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
            raise ConfigurationParseError(f"Error parsing training section in {source}: {e}") from e

        # Parse model configuration
        try:
            model_config = ConfigurationParser._parse_model(data["model"], source)
        except Exception as e:
            raise ConfigurationParseError(f"Error parsing model section in {source}: {e}") from e

        # Parse data configuration
        try:
            data_config = ConfigurationParser._parse_data(data["data"], source)
        except Exception as e:
            raise ConfigurationParseError(f"Error parsing data section in {source}: {e}") from e

        # Parse XAI configuration (optional)
        xai_config = XAIConfig()  # Default configuration
        if "xai" in data:
            try:
                xai_config = ConfigurationParser._parse_xai(data["xai"], source)
            except Exception as e:
                raise ConfigurationParseError(f"Error parsing xai section in {source}: {e}") from e

        return Configuration(
            training=training_config, model=model_config, data=data_config, xai=xai_config
        )

    @staticmethod
    def _parse_training(data: dict[str, Any], source: str | Path) -> TrainingConfig:
        """Parse training configuration section."""
        # epochs is optional if epochs_per_round is provided
        epochs_per_round = data.get("epochs_per_round")
        epochs = data.get("epochs")

        if epochs is None and epochs_per_round is None:
            raise ConfigurationParseError("Either 'epochs' or 'epochs_per_round' must be specified")

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
            except (ValueError, TypeError) as e:
                raise ConfigurationParseError(
                    f"epochs must be an integer, got: {data['epochs']}"
                ) from e

        try:
            patience = int(data["patience"])
            if patience < 0:
                raise ConfigurationParseError("patience must be non-negative")
        except (ValueError, TypeError) as e:
            raise ConfigurationParseError(
                f"patience must be an integer, got: {data['patience']}"
            ) from e

        try:
            image_size = int(data["image_size"])
            if image_size <= 0:
                raise ConfigurationParseError("image_size must be positive")
        except (ValueError, TypeError) as e:
            raise ConfigurationParseError(
                f"image_size must be an integer, got: {data['image_size']}"
            ) from e

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
        except (ValueError, TypeError) as e:
            raise ConfigurationParseError(
                f"baseline_epochs must be a positive integer, got: {data['baseline_epochs']}"
            ) from e

        # Augmentation fields — read from YAML augmentation subsection or top-level, with paper defaults
        aug_data = data.get("augmentation", {})
        mosaic = float(aug_data.get("mosaic", data.get("mosaic", 1.0)))
        scale = float(aug_data.get("scale", data.get("scale", 0.5)))
        fliplr = float(aug_data.get("fliplr", data.get("fliplr", 0.5)))
        hsv_h = float(aug_data.get("hsv_h", data.get("hsv_h", 0.015)))
        hsv_s = float(aug_data.get("hsv_s", data.get("hsv_s", 0.7)))
        hsv_v = float(aug_data.get("hsv_v", data.get("hsv_v", 0.4)))

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
            mosaic=mosaic,
            scale=scale,
            fliplr=fliplr,
            hsv_h=hsv_h,
            hsv_s=hsv_s,
            hsv_v=hsv_v,
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
            if not 0.0 < train_init_percentage <= 1.0:
                raise ConfigurationParseError("train_init_percentage must be in (0, 1]")

        iou_threshold = data.get("iou_threshold")
        if iou_threshold is not None:
            iou_threshold = float(iou_threshold)
            if not 0.0 <= iou_threshold <= 1.0:
                raise ConfigurationParseError("iou_threshold must be in [0, 1]")

        return DataConfig(
            yaml_path=yaml_path,
            train_init_percentage=train_init_percentage,
            iou_threshold=iou_threshold,
        )

    @staticmethod
    def _parse_xai(data: dict[str, Any], source: str | Path) -> XAIConfig:
        """Parse XAI configuration section."""
        # All XAI fields are optional with sensible defaults
        enabled = data.get("enabled", False)
        if not isinstance(enabled, bool):
            raise ConfigurationParseError(
                f"xai.enabled must be a boolean, got: {type(enabled).__name__}"
            )

        # Parse methods configuration
        methods = data.get("methods", {"gradcam": True, "lrp": False, "shap": False})
        if not isinstance(methods, dict):
            raise ConfigurationParseError("xai.methods must be a dictionary")

        # Validate method names
        valid_methods = {"gradcam", "lrp", "shap"}
        invalid_methods = set(methods.keys()) - valid_methods
        if invalid_methods:
            raise ConfigurationParseError(
                f"Invalid XAI methods: {', '.join(invalid_methods)}. "
                f"Valid methods: {', '.join(valid_methods)}"
            )

        # Ensure all method values are boolean
        for method, method_enabled in methods.items():
            if not isinstance(method_enabled, bool):
                raise ConfigurationParseError(
                    f"xai.methods.{method} must be a boolean, got: {type(method_enabled).__name__}"
                )

        # Parse sample limit
        sample_limit = data.get("sample_limit")
        if sample_limit is not None:
            try:
                sample_limit = int(sample_limit)
                if sample_limit <= 0:
                    raise ConfigurationParseError("xai.sample_limit must be positive")
            except (ValueError, TypeError) as e:
                raise ConfigurationParseError(
                    f"xai.sample_limit must be an integer, got: {sample_limit}"
                ) from e

        # Parse target layer
        _ARCH_KEYS = ("yolov11", "yolov12", "yolo26", "yolov8")
        target_layers: dict[str, str] = {}

        if "target_layers" in data:
            raw_layers = data["target_layers"]
            if not isinstance(raw_layers, dict):
                raise ConfigurationParseError(
                    f"xai.target_layers must be a dict, got: {type(raw_layers).__name__}"
                )
            for k, v in raw_layers.items():
                if not isinstance(v, str):
                    raise ConfigurationParseError(
                        f"xai.target_layers values must be strings, got {type(v).__name__} for key '{k}'"
                    )
            target_layers = dict(raw_layers)

        target_layer = data.get("target_layer")
        if target_layer is not None and not isinstance(target_layer, str):
            raise ConfigurationParseError(
                f"xai.target_layer must be a string, got: {type(target_layer).__name__}"
            )
        if isinstance(target_layer, str) and not target_layers:
            target_layers = dict.fromkeys(_ARCH_KEYS, target_layer)

        # Parse background set size
        background_set_size = data.get("background_set_size", 75)
        try:
            background_set_size = int(background_set_size)
            if background_set_size <= 0:
                raise ConfigurationParseError("xai.background_set_size must be positive")
        except (ValueError, TypeError) as e:
            raise ConfigurationParseError(
                f"xai.background_set_size must be an integer, got: {background_set_size}"
            ) from e

        # Parse background set seed
        background_set_seed = data.get("background_set_seed", 42)
        try:
            background_set_seed = int(background_set_seed)
        except (ValueError, TypeError) as e:
            raise ConfigurationParseError(
                f"xai.background_set_seed must be an integer, got: {background_set_seed}"
            ) from e

        # Parse output overlays
        output_overlays = data.get("output_overlays", True)
        if not isinstance(output_overlays, bool):
            raise ConfigurationParseError(
                f"xai.output_overlays must be a boolean, got: {type(output_overlays).__name__}"
            )

        # Parse save raw attributions
        save_raw_attributions = data.get("save_raw_attributions", False)
        if not isinstance(save_raw_attributions, bool):
            raise ConfigurationParseError(
                f"xai.save_raw_attributions must be a boolean, got: {type(save_raw_attributions).__name__}"
            )

        # Parse HFS computation
        hfs_computation = data.get("hfs_computation", True)
        if not isinstance(hfs_computation, bool):
            raise ConfigurationParseError(
                f"xai.hfs_computation must be a boolean, got: {type(hfs_computation).__name__}"
            )

        # Parse LRP rule
        lrp_rule = data.get("lrp_rule", "epsilon")
        valid_lrp_rules = {"epsilon", "gamma", "alpha-beta"}
        if lrp_rule not in valid_lrp_rules:
            raise ConfigurationParseError(
                f"xai.lrp_rule must be one of {valid_lrp_rules}, got: '{lrp_rule}'"
            )

        # Parse gradcam target classes
        gradcam_target_classes = data.get("gradcam_target_classes", ["knife", "pistol"])
        if not isinstance(gradcam_target_classes, list):
            raise ConfigurationParseError(
                f"xai.gradcam_target_classes must be a list, got: {type(gradcam_target_classes).__name__}"
            )
        for item in gradcam_target_classes:
            if not isinstance(item, str):
                raise ConfigurationParseError(
                    f"xai.gradcam_target_classes must be a list of strings, got item: {type(item).__name__}"
                )

        return XAIConfig(
            enabled=enabled,
            methods=methods,
            sample_limit=sample_limit,
            target_layer=target_layer,
            target_layers=target_layers,
            background_set_size=background_set_size,
            background_set_seed=background_set_seed,
            output_overlays=output_overlays,
            save_raw_attributions=save_raw_attributions,
            hfs_computation=hfs_computation,
            lrp_rule=lrp_rule,
            gradcam_target_classes=gradcam_target_classes,
        )
