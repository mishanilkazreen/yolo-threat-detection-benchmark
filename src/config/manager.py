"""Configuration manager for loading and validating configurations."""

from pathlib import Path
from typing import Any

import yaml

from .parser import Configuration, ConfigurationParseError, ConfigurationParser


class ExplainerConfig:
    """Explainer configuration container."""

    def __init__(self, data: dict[str, Any]):
        """Initialize explainer configuration from dictionary."""
        self.data = data
        self.model_weights_path = data.get("model", {}).get("weights_path")
        self.image_directory = data.get("images", {}).get("directory")
        self.num_images = data.get("images", {}).get("num_images", 10)
        self.target_class = data.get("images", {}).get("target_class")
        self.confidence_threshold = data.get("explainability", {}).get("confidence_threshold", 0.5)
        self.methods = data.get("explainability", {}).get("methods", [])
        self.use_predicted_boxes = data.get("explainability", {}).get("use_predicted_boxes", True)

        # Method-specific configurations
        self.occlusion_config = data.get("occlusion", {})
        self.gradcam_config = data.get("gradcam", {})
        self.lrp_config = data.get("lrp", {})
        self.shap_config = data.get("shap", {})


class ConfigurationManager:
    """Manager for loading and validating training and explainer configurations."""

    SUPPORTED_ARCHITECTURES = [
        "yolov8n",
        "yolov8s",
        "yolov8m",
        "yolov8l",
        "yolov8x",
        "yolov11n",
        "yolov11s",
        "yolov11m",
        "yolov11l",
        "yolov11x",
        "yolov12n",
        "yolov12s",
        "yolov12m",
        "yolov12l",
        "yolov12x",
        "yolo26n",
        "yolo26s",
        "yolo26m",
        "yolo26l",
        "yolo26x",
    ]

    @staticmethod
    def load_training_config(config_path: str | Path) -> Configuration:
        """
        Load and validate training configuration.

        Args:
            config_path: Path to training configuration YAML file

        Returns:
            Validated Configuration object

        Raises:
            ConfigurationParseError: If configuration is invalid
        """
        config = ConfigurationParser.parse(config_path)
        ConfigurationManager._validate_training_config(config, config_path)
        return config

    @staticmethod
    def _validate_training_config(config: Configuration, config_path: str | Path) -> None:
        """
        Validate training configuration fields.

        Args:
            config: Configuration object to validate
            config_path: Path to configuration file (for error messages)

        Raises:
            ConfigurationParseError: If validation fails
        """
        # Validate required fields exist (already done by parser, but double-check)
        required_fields = {
            "model.name": config.model.name,
            "model.weights": config.model.weights,
            "training.epochs": config.training.epochs,
            "training.image_size": config.training.image_size,
            "training.device": config.training.device,
            "data.yaml_path": config.data.yaml_path,
        }

        missing = [field for field, value in required_fields.items() if not value]
        if missing:
            raise ConfigurationParseError(
                f"Configuration {config_path} is missing or has empty required fields: {', '.join(missing)}"
            )

        # Validate model architecture
        model_name_lower = config.model.name.lower()
        if not any(arch in model_name_lower for arch in ["yolov8", "yolov11", "yolov12", "yolo26"]):
            raise ConfigurationParseError(
                f"Unsupported model architecture '{config.model.name}'. "
                f"Must be one of: YOLOv8, YOLOv11, YOLOv12, YOLO26 (any size variant)"
            )

        # Validate device
        valid_devices = ["auto", "cpu", "cuda", "mps"]
        device_lower = config.training.device.lower()
        if (
            device_lower not in valid_devices
            and not device_lower.startswith("cuda:")
            and not device_lower.isdigit()
        ):  # Allow single digit for GPU index
            raise ConfigurationParseError(
                f"Invalid device '{config.training.device}'. "
                f"Must be one of: {', '.join(valid_devices)}, 'cuda:N', or 'N' for specific GPU"
            )

        # Validate multi-run configuration
        if (
            config.training.runs > 1
            and config.training.seeds is not None
            and len(config.training.seeds) != config.training.runs
        ):
            raise ConfigurationParseError(
                f"Number of seeds ({len(config.training.seeds)}) must match "
                f"number of runs ({config.training.runs})"
            )

        # Validate incremental training fields
        ConfigurationManager._validate_incremental_training_fields(config, config_path)

        # Validate XAI configuration
        ConfigurationManager._validate_xai_config(config, config_path)

    @staticmethod
    def _validate_incremental_training_fields(
        config: Configuration, config_path: str | Path
    ) -> None:
        """
        Validate incremental training configuration fields.

        Args:
            config: Configuration object to validate
            config_path: Path to configuration file (for error messages)

        Raises:
            ConfigurationParseError: If validation fails
        """
        # Validate rounds (default: 5)
        if (
            hasattr(config.training, "rounds")
            and config.training.rounds is not None
            and (not isinstance(config.training.rounds, int) or config.training.rounds < 1)
        ):
            raise ConfigurationParseError(
                f"Configuration {config_path}: training.rounds must be a positive integer, "
                f"got {config.training.rounds}"
            )

        # Validate epochs_per_round
        if (
            hasattr(config.training, "epochs_per_round")
            and config.training.epochs_per_round is not None
            and (
                not isinstance(config.training.epochs_per_round, int)
                or config.training.epochs_per_round < 1
            )
        ):
            raise ConfigurationParseError(
                f"Configuration {config_path}: training.epochs_per_round must be a positive integer, "
                f"got {config.training.epochs_per_round}"
            )

        # Validate train_init_percentage (default: 0.2)
        if (
            hasattr(config.data, "train_init_percentage")
            and config.data.train_init_percentage is not None
        ):
            if not isinstance(config.data.train_init_percentage, (int, float)):
                raise ConfigurationParseError(
                    f"Configuration {config_path}: data.train_init_percentage must be a number, "
                    f"got {type(config.data.train_init_percentage).__name__}"
                )
            if not (0.0 < config.data.train_init_percentage <= 1.0):
                raise ConfigurationParseError(
                    f"Configuration {config_path}: data.train_init_percentage must be in (0, 1], "
                    f"got {config.data.train_init_percentage}"
                )

        # Validate iou_threshold (default: 0.5)
        if hasattr(config.data, "iou_threshold") and config.data.iou_threshold is not None:
            if not isinstance(config.data.iou_threshold, (int, float)):
                raise ConfigurationParseError(
                    f"Configuration {config_path}: data.iou_threshold must be a number, "
                    f"got {type(config.data.iou_threshold).__name__}"
                )
            if not (0.0 <= config.data.iou_threshold <= 1.0):
                raise ConfigurationParseError(
                    f"Configuration {config_path}: data.iou_threshold must be in [0, 1], "
                    f"got {config.data.iou_threshold}"
                )

    @staticmethod
    def _validate_xai_config(config: Configuration, config_path: str | Path) -> None:
        """
        Validate XAI configuration fields.

        Args:
            config: Configuration object to validate
            config_path: Path to configuration file (for error messages)

        Raises:
            ConfigurationParseError: If validation fails
        """
        # XAI configuration is optional, but if present, validate it
        if not hasattr(config, "xai") or config.xai is None:
            return

        xai = config.xai

        # Validate enabled methods when XAI is enabled
        if xai.enabled:
            enabled_methods = [method for method, enabled in xai.methods.items() if enabled]
            if not enabled_methods:
                raise ConfigurationParseError(
                    f"Configuration {config_path}: XAI is enabled but no methods are enabled. "
                    f"Enable at least one method: gradcam, lrp, shap"
                )

        # Validate target layers for supported architectures
        model_name_lower = config.model.name.lower()
        architecture = None
        for arch in ["yolov8", "yolov11", "yolov12", "yolo26"]:
            if arch in model_name_lower:
                architecture = arch
                break

        if architecture and architecture not in xai.target_layers:
            raise ConfigurationParseError(
                f"Configuration {config_path}: Missing target layer configuration for "
                f"architecture '{architecture}' in xai.target_layers"
            )

        # Validate sample limit consistency with expensive methods
        expensive_methods = ["shap", "lrp"]
        enabled_expensive = [m for m in expensive_methods if xai.methods.get(m, False)]

        if enabled_expensive and xai.sample_limit is None:
            # This is a warning case - expensive methods without sample limit
            # We don't raise an error but could log a warning in the future
            pass

    @staticmethod
    def load_explainer_config(config_path: str | Path) -> ExplainerConfig:
        """
        Load explainer configuration from YAML file.

        Args:
            config_path: Path to explainer configuration YAML file

        Returns:
            ExplainerConfig object

        Raises:
            ConfigurationParseError: If configuration is invalid
        """
        try:
            with open(config_path) as f:
                data = yaml.safe_load(f)
        except FileNotFoundError:
            raise ConfigurationParseError(f"Explainer configuration file not found: {config_path}")
        except yaml.YAMLError as e:
            raise ConfigurationParseError(f"Invalid YAML syntax in {config_path}: {e}")
        except Exception as e:
            raise ConfigurationParseError(
                f"Failed to read explainer configuration {config_path}: {e}"
            )

        if data is None:
            raise ConfigurationParseError(f"Explainer configuration file is empty: {config_path}")

        explainer_config = ExplainerConfig(data)
        ConfigurationManager._validate_explainer_config(explainer_config, config_path)
        return explainer_config

    @staticmethod
    def _validate_explainer_config(config: ExplainerConfig, config_path: str | Path) -> None:
        """
        Validate explainer configuration fields.

        Args:
            config: ExplainerConfig object to validate
            config_path: Path to configuration file (for error messages)

        Raises:
            ConfigurationParseError: If validation fails
        """
        # Validate required fields
        if not config.model_weights_path:
            raise ConfigurationParseError(
                f"Explainer configuration {config_path} missing required field: model.weights_path"
            )

        if not config.image_directory:
            raise ConfigurationParseError(
                f"Explainer configuration {config_path} missing required field: images.directory"
            )

        if config.num_images <= 0:
            raise ConfigurationParseError(
                f"Explainer configuration {config_path}: num_images must be positive"
            )

        if not config.methods:
            raise ConfigurationParseError(
                f"Explainer configuration {config_path} missing required field: explainability.methods"
            )

        # Validate methods
        valid_methods = ["occlusion", "gradcam", "lrp", "shap"]
        invalid_methods = [m for m in config.methods if m not in valid_methods]
        if invalid_methods:
            raise ConfigurationParseError(
                f"Explainer configuration {config_path} contains invalid methods: {', '.join(invalid_methods)}. "
                f"Valid methods: {', '.join(valid_methods)}"
            )

        # Validate confidence threshold
        if not (0.0 <= config.confidence_threshold <= 1.0):
            raise ConfigurationParseError(
                f"Explainer configuration {config_path}: confidence_threshold must be between 0.0 and 1.0"
            )

    @staticmethod
    def discover_training_configs(config_dir: str | Path = "config/models") -> list[Path]:
        """
        Discover all YAML configuration files in the specified directory.

        Args:
            config_dir: Directory to search for configuration files

        Returns:
            List of paths to configuration files
        """
        config_dir = Path(config_dir)
        if not config_dir.exists():
            return []

        return sorted(config_dir.glob("*.yaml")) + sorted(config_dir.glob("*.yml"))

    @staticmethod
    def discover_explainer_configs(
        config_dir: str | Path = "config/explainers",
    ) -> list[Path]:
        """
        Discover all explainer configuration files in the specified directory.

        Args:
            config_dir: Directory to search for explainer configuration files

        Returns:
            List of paths to explainer configuration files
        """
        config_dir = Path(config_dir)
        if not config_dir.exists():
            return []

        return sorted(config_dir.glob("*.yaml")) + sorted(config_dir.glob("*.yml"))

    @staticmethod
    def get_config_name(config_path: str | Path) -> str:
        """
        Extract configuration name from file path.

        Args:
            config_path: Path to configuration file

        Returns:
            Configuration name (filename without extension)
        """
        return Path(config_path).stem
