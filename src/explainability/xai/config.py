"""Configuration classes for XAI functionality."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class XAIConfig:
    """Configuration for XAI explainability methods.

    Attributes:
        enabled: Whether XAI processing is enabled
        methods: Dictionary of method name to enabled status
        sample_limit: Maximum number of images to process per round (None = no limit)
        target_layers: Architecture-specific target layer mappings
        background_set_size: Number of images for SHAP background set
        background_set_seed: Random seed for SHAP background set sampling
        output_overlays: Whether to save attribution overlays on original images
        save_raw_attributions: Whether to save raw attribution arrays
        hfs_computation: Whether to compute HFS scores
        lrp_rule: LRP propagation rule to use ("epsilon", "gamma", "alpha-beta")
    """

    enabled: bool = False
    methods: dict[str, bool] = field(
        default_factory=lambda: {
            "gradcam": True,
            "lrp": False,  # Computationally expensive, disabled by default
            "shap": False,  # Very expensive, disabled by default
        }
    )
    sample_limit: int | None = None
    target_layers: dict[str, str] = field(
        default_factory=lambda: {
            # Hook model.9 (SPPF layer) for complete Grad-CAM across all scale predictions
            "yolov11": "model.9",
            "yolov12": "model.9",
            "yolo26": "model.9",
            "yolov8": "model.9",
        }
    )
    background_set_size: int = 75
    background_set_seed: int = 42
    output_overlays: bool = True
    save_raw_attributions: bool = False
    hfs_computation: bool = True
    lrp_rule: str = "epsilon"
    gradcam_target_classes: list[str] = field(
        default_factory=lambda: ["knife", "pistol"]
    )  # mask CAM to only these class names — must match model.names exactly (case-insensitive)

    def is_method_enabled(self, method: str) -> bool:
        """Check if a specific XAI method is enabled."""
        return self.enabled and self.methods.get(method, False)

    def get_target_layer(self, architecture: str, strict: bool = False) -> str:
        """Get target layer for a specific YOLO architecture.

        Args:
            architecture: Model architecture name (e.g., "yolov11n", "yolov12s")
            strict: If True, raise KeyError when variant not found in config.
                   If False, use fallback behavior (default).

        Returns:
            Target layer path string

        Raises:
            KeyError: If strict=True and detected variant not in target_layers
        """
        if architecture is None:
            architecture = ""

        # Check architectures longest-first to avoid substring matches
        arch_base = None
        for arch in ["yolov12", "yolov11", "yolo26", "yolov8"]:
            if arch in architecture.lower():
                arch_base = arch
                break

        if arch_base is None:
            arch_base = "yolov11"

        if strict and arch_base not in self.target_layers:
            raise KeyError(
                f"Model variant '{arch_base}' not found in configuration target_layers. "
                f"Available variants: {list(self.target_layers.keys())}"
            )

        return self.target_layers.get(arch_base, self.target_layers.get("yolov11", "model.9"))

    def get_enabled_methods(self) -> list[str]:
        """Get list of enabled XAI methods."""
        if not self.enabled:
            return []
        return [method for method, enabled in self.methods.items() if enabled]


# Architecture-specific layer mapping for different YOLO versions
ARCHITECTURE_LAYER_MAPPING = {
    "yolov11": {
        "gradcam_target": "model.9",  # SPPF — shared backbone bottleneck
        "lrp_target": "model.22",
        "backbone_layers": [
            "model.0",
            "model.1",
            "model.2",
            "model.3",
            "model.4",
            "model.5",
            "model.6",
            "model.7",
            "model.8",
            "model.9",
            "model.10",
            "model.11",
            "model.12",
            "model.13",
            "model.14",
            "model.15",
            "model.16",
            "model.17",
            "model.18",
            "model.19",
            "model.20",
            "model.21",
            "model.22",
        ],
    },
    "yolov12": {
        "gradcam_target": "model.9",
        "lrp_target": "model.22",
        "backbone_layers": [
            "model.0",
            "model.1",
            "model.2",
            "model.3",
            "model.4",
            "model.5",
            "model.6",
            "model.7",
            "model.8",
            "model.9",
            "model.10",
            "model.11",
            "model.12",
            "model.13",
            "model.14",
            "model.15",
            "model.16",
            "model.17",
            "model.18",
            "model.19",
            "model.20",
            "model.21",
            "model.22",
        ],
    },
    "yolo26": {
        "gradcam_target": "model.9",  # SPPF — shared backbone bottleneck
        "lrp_target": "model.21",
        "backbone_layers": [
            "model.0",
            "model.1",
            "model.2",
            "model.3",
            "model.4",
            "model.5",
            "model.6",
            "model.7",
            "model.8",
            "model.9",
            "model.10",
            "model.11",
            "model.12",
            "model.13",
            "model.14",
            "model.15",
            "model.16",
            "model.17",
            "model.18",
            "model.19",
            "model.20",
            "model.21",
        ],
    },
    "yolov8": {
        "gradcam_target": "model.9",
        "lrp_target": "model.22",
        "backbone_layers": [
            "model.0",
            "model.1",
            "model.2",
            "model.3",
            "model.4",
            "model.5",
            "model.6",
            "model.7",
            "model.8",
            "model.9",
            "model.10",
            "model.11",
            "model.12",
            "model.13",
            "model.14",
            "model.15",
            "model.16",
            "model.17",
            "model.18",
            "model.19",
            "model.20",
            "model.21",
            "model.22",
        ],
    },
}


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
