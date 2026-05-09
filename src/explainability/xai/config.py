"""Configuration classes for XAI functionality."""

from typing import Any

from src.config.xai_config import ARCHITECTURE_LAYER_MAPPING, XAIConfig

__all__ = ["ARCHITECTURE_LAYER_MAPPING", "XAIConfig"]


def get_architecture_config(model_name: str) -> dict[str, Any]:
    """Get architecture-specific configuration for a YOLO model.

    Args:
        model_name: Name of the YOLO model (e.g., "yolov12s", "yolov11n")

    Returns:
        Dictionary containing architecture-specific layer mappings

    Raises:
        ValueError: If architecture is not supported
    """
    model_lower = model_name.lower()

    # Check architectures longest-first to avoid substring matches
    for arch in ["yolov12", "yolov11", "yolo26", "yolov8"]:
        if arch in model_lower:
            return ARCHITECTURE_LAYER_MAPPING[arch]

    raise ValueError(f"Unsupported YOLO architecture: {model_name}")


def validate_target_layer(model_name: str, target_layer: str) -> bool:
    """Validate that a target layer exists for the given architecture.

    Args:
        model_name: Name of the YOLO model
        target_layer: Target layer name to validate

    Returns:
        True if layer is valid for the architecture
    """
    try:
        arch_config = get_architecture_config(model_name)
        return target_layer in arch_config["backbone_layers"]
    except ValueError:
        return False
