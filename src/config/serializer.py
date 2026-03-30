"""Configuration serializer for Configuration objects to YAML format."""

from pathlib import Path
from typing import Any

import yaml

from .parser import Configuration, DataConfig, ModelConfig, TrainingConfig, XAIConfig


class ConfigurationSerializer:
    """Serializer for Configuration objects to YAML format."""

    @staticmethod
    def serialize(config: Configuration, output_path: str | Path) -> None:
        """
        Serialize Configuration object to YAML file.

        Args:
            config: Configuration object to serialize
            output_path: Path where YAML file will be written

        Raises:
            IOError: If file cannot be written
        """
        data = ConfigurationSerializer.to_dict(config)

        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "w") as f:
                yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)
        except Exception as e:
            raise OSError(f"Failed to write configuration to {output_path}: {e}")

    @staticmethod
    def to_dict(config: Configuration) -> dict[str, Any]:
        """
        Convert Configuration object to dictionary.

        Args:
            config: Configuration object

        Returns:
            Dictionary representation suitable for YAML serialization
        """
        result = {
            "training": ConfigurationSerializer._training_to_dict(config.training),
            "model": ConfigurationSerializer._model_to_dict(config.model),
            "data": ConfigurationSerializer._data_to_dict(config.data),
        }

        # Only include XAI section if it's not using all default values
        if ConfigurationSerializer._should_include_xai(config.xai):
            result["xai"] = ConfigurationSerializer._xai_to_dict(config.xai)

        return result

    @staticmethod
    def _training_to_dict(training: TrainingConfig) -> dict[str, Any]:
        """Convert TrainingConfig to dictionary."""
        result: dict[str, Any] = {
            "epochs": training.epochs,
            "patience": training.patience,
            "image_size": training.image_size,
            "device": training.device,
        }

        # Only include optional fields if they differ from defaults
        if training.runs != 1:
            result["runs"] = training.runs

        if training.seeds is not None:
            result["seeds"] = training.seeds

        if training.run_baseline:
            result["run_baseline"] = training.run_baseline

        if training.baseline_epochs != 50:
            result["baseline_epochs"] = training.baseline_epochs

        return result

    @staticmethod
    def _model_to_dict(model: ModelConfig) -> dict[str, Any]:
        """Convert ModelConfig to dictionary."""
        return {"name": model.name, "weights": model.weights}

    @staticmethod
    def _data_to_dict(data: DataConfig) -> dict[str, Any]:
        """Convert DataConfig to dictionary."""
        result = {"yaml_path": data.yaml_path}

        # Only include optional fields if they are set
        if data.train_init_percentage is not None:
            result["train_init_percentage"] = data.train_init_percentage  # type: ignore[assignment]

        if data.iou_threshold is not None:
            result["iou_threshold"] = data.iou_threshold  # type: ignore[assignment]

        return result

    @staticmethod
    def _should_include_xai(xai: XAIConfig) -> bool:
        """Check if XAI config differs from defaults and should be included."""
        default_xai = XAIConfig()
        return (
            xai.enabled != default_xai.enabled
            or xai.methods != default_xai.methods
            or xai.sample_limit != default_xai.sample_limit
            or xai.target_layer != default_xai.target_layer
            or xai.background_set_size != default_xai.background_set_size
            or xai.background_set_seed != default_xai.background_set_seed
            or xai.output_overlays != default_xai.output_overlays
            or xai.save_raw_attributions != default_xai.save_raw_attributions
            or xai.hfs_computation != default_xai.hfs_computation
            or xai.lrp_rule != default_xai.lrp_rule
            or xai.gradcam_target_classes != default_xai.gradcam_target_classes
        )

    @staticmethod
    def _xai_to_dict(xai: XAIConfig) -> dict[str, Any]:
        """Convert XAIConfig to dictionary."""
        default_xai = XAIConfig()
        result: dict[str, Any] = {}

        # Only include fields that differ from defaults
        if xai.enabled != default_xai.enabled:
            result["enabled"] = xai.enabled

        if xai.methods != default_xai.methods:
            result["methods"] = xai.methods

        if xai.sample_limit != default_xai.sample_limit:
            result["sample_limit"] = xai.sample_limit

        if xai.target_layer != default_xai.target_layer:
            result["target_layer"] = xai.target_layer

        if xai.background_set_size != default_xai.background_set_size:
            result["background_set_size"] = xai.background_set_size

        if xai.background_set_seed != default_xai.background_set_seed:
            result["background_set_seed"] = xai.background_set_seed

        if xai.output_overlays != default_xai.output_overlays:
            result["output_overlays"] = xai.output_overlays

        if xai.save_raw_attributions != default_xai.save_raw_attributions:
            result["save_raw_attributions"] = xai.save_raw_attributions

        if xai.hfs_computation != default_xai.hfs_computation:
            result["hfs_computation"] = xai.hfs_computation

        if xai.lrp_rule != default_xai.lrp_rule:
            result["lrp_rule"] = xai.lrp_rule

        if xai.gradcam_target_classes != default_xai.gradcam_target_classes:
            result["gradcam_target_classes"] = xai.gradcam_target_classes

        return result
