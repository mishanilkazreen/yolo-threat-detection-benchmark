"""XAI (Explainable AI) module for YOLO model explanations.

This module provides three complementary XAI methods:
- Grad-CAM: Gradient-weighted Class Activation Mapping
- LRP: Layer-wise Relevance Propagation
- SHAP: SHapley Additive exPlanations

Each method is implemented as an independent function that can be imported
and used without dependencies on other XAI methods.
"""

from .config import XAIConfig
from .gradcam import generate_gradcam_attribution
from .interfaces import AttributionMethod, XAIResult
from .lrp import generate_lrp_attribution
from .manager import XAIManager
from .shap import generate_shap_attribution

__all__ = [
    "AttributionMethod",
    "XAIConfig",
    "XAIManager",
    "XAIResult",
    "generate_gradcam_attribution",
    "generate_lrp_attribution",
    "generate_shap_attribution",
]
