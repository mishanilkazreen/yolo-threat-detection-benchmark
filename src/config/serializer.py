"""Configuration serializer for Configuration objects to YAML format."""

from pathlib import Path
from typing import Any

import yaml

from .parser import Configuration, DataConfig, ModelConfig, TrainingConfig


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
        return {
            "training": ConfigurationSerializer._training_to_dict(config.training),
            "model": ConfigurationSerializer._model_to_dict(config.model),
            "data": ConfigurationSerializer._data_to_dict(config.data),
        }

    @staticmethod
    def _training_to_dict(training: TrainingConfig) -> dict[str, Any]:
        """Convert TrainingConfig to dictionary."""
        result = {
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

        return result

    @staticmethod
    def _model_to_dict(model: ModelConfig) -> dict[str, Any]:
        """Convert ModelConfig to dictionary."""
        return {"name": model.name, "weights": model.weights}

    @staticmethod
    def _data_to_dict(data: DataConfig) -> dict[str, Any]:
        """Convert DataConfig to dictionary."""
        return {"yaml_path": data.yaml_path}
