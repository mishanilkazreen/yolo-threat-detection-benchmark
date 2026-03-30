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
        target_layer: Target layer path for GradCAM/LRP (e.g. "model.22").
            None means auto-resolve from ARCHITECTURE_LAYER_MAPPING.
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
    target_layer: str | None = None
    background_set_size: int = 75
    background_set_seed: int = 42
    output_overlays: bool = True
    save_raw_attributions: bool = False
    hfs_computation: bool = True
    lrp_rule: str = "epsilon"
    gradcam_target_classes: list[str] = field(
        default_factory=lambda: ["knife", "pistol"]
    )  # mask CAM to only these class names — must match model.names exactly (case-insensitive)
    target_layers: dict[str, str] = field(default_factory=dict)

    def is_method_enabled(self, method: str) -> bool:
        """Check if a specific XAI method is enabled."""
        return self.enabled and self.methods.get(method, False)

    def get_target_layer(self, architecture: str | None = None, strict: bool = False) -> str:
        """Get target layer for a specific YOLO architecture.

        If ``target_layer`` is explicitly set on this config, it is returned
        directly regardless of architecture.  Otherwise the architecture name
        is matched against ARCHITECTURE_LAYER_MAPPING and its ``gradcam_target``
        is used as the fallback.

        Args:
            architecture: Model architecture name (e.g., "yolov11n", "yolov12s").
                Ignored when ``self.target_layer`` is set.
            strict: If True, raise KeyError when the architecture cannot be
                resolved and no explicit ``target_layer`` is configured.

        Returns:
            Target layer path string

        Raises:
            KeyError: If strict=True and the architecture is not recognised and
                no explicit target_layer is configured.
        """
        arch_str: str = architecture or ""

        # Resolve architecture base key (longest-first to avoid substring matches)
        arch_base = None
        for arch in ["yolov12", "yolo12", "yolov11", "yolo26", "yolov8"]:
            if arch in arch_str.lower():
                arch_base = arch
                break

        # yolo12 (no 'v') is an alias for yolov12
        lookup = "yolov12" if arch_base == "yolo12" else arch_base

        # 1. Check per-arch target_layers dict
        if self.target_layers and lookup and lookup in self.target_layers:
            return self.target_layers[lookup]

        # 2. Check target_layer string (applies to all architectures)
        if self.target_layer is not None:
            return self.target_layer

        # 3. Fall through to architecture mapping
        if arch_base is None:
            if strict:
                raise KeyError(
                    f"Cannot resolve target layer for unknown architecture: '{arch_str}'. "
                    f"Set xai.target_layer explicitly in the model config."
                )
            arch_base = "yolov11"  # safe default
            lookup = arch_base

        # At this point lookup is guaranteed to be a string
        assert lookup is not None
        return str(ARCHITECTURE_LAYER_MAPPING[lookup]["gradcam_target"])

    def get_enabled_methods(self) -> list[str]:
        """Get list of enabled XAI methods."""
        if not self.enabled:
            return []
        return [method for method, enabled in self.methods.items() if enabled]


# Architecture-specific layer mapping for different YOLO versions
ARCHITECTURE_LAYER_MAPPING = {
    "yolov11": {
        "gradcam_target": "model.22",  # C3k2 [1024] — neck P5, 20x20 (last neck block before Detect)
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
        # Backbone: 0-8 (Conv/C3k2/A2C2f). Neck: 9-20. Detect: 21.
        # A2C2f blocks are unreliable for standard GradCAM; EigenCAM is used instead.
        # model.20 = C3k2, neck P5, 20x20 — last block before Detect.
        "gradcam_target": "model.20",
        "lrp_target": "model.20",
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
        ],
    },
    "yolo26": {
        "gradcam_target": "model.22",  # C3k2 [1024] — neck P5, 20x20 (last neck block before Detect)
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
            "model.22",
        ],
    },
    "yolov8": {
        "gradcam_target": "model.18",  # C2f [512] — neck P5, 20x20 (last neck block before Detect)
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
