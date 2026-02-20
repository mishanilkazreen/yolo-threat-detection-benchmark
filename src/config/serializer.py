"""Configuration serializer for Configuration objects to YAML format."""

from typing import Union, Dict, Any
from pathlib import Path
import yaml

from .parser import Configuration, TrainingConfig, ModelConfig, DataConfig


class ConfigurationSerializer:
    """Serializer for Configuration objects to YAML format."""
    
    @staticmethod
    def serialize(config: Configuration, output_path: Union[str, Path]) -> None:
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
            
            with open(output_path, 'w') as f:
                yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)
        except Exception as e:
            raise IOError(f"Failed to write configuration to {output_path}: {e}")
    
    @staticmethod
    def to_dict(config: Configuration) -> Dict[str, Any]:
        """
        Convert Configuration object to dictionary.
        
        Args:
            config: Configuration object
            
        Returns:
            Dictionary representation suitable for YAML serialization
        """
        return {
            'training': ConfigurationSerializer._training_to_dict(config.training),
            'model': ConfigurationSerializer._model_to_dict(config.model),
            'data': ConfigurationSerializer._data_to_dict(config.data)
        }
    
    @staticmethod
    def _training_to_dict(training: TrainingConfig) -> Dict[str, Any]:
        """Convert TrainingConfig to dictionary."""
        result = {
            'epochs': training.epochs,
            'patience': training.patience,
            'image_size': training.image_size,
            'device': training.device
        }
        
        # Only include optional fields if they differ from defaults
        if training.runs != 1:
            result['runs'] = training.runs
        
        if training.seeds is not None:
            result['seeds'] = training.seeds
        
        return result
    
    @staticmethod
    def _model_to_dict(model: ModelConfig) -> Dict[str, Any]:
        """Convert ModelConfig to dictionary."""
        return {
            'name': model.name,
            'weights': model.weights
        }
    
    @staticmethod
    def _data_to_dict(data: DataConfig) -> Dict[str, Any]:
        """Convert DataConfig to dictionary."""
        return {
            'yaml_path': data.yaml_path
        }
