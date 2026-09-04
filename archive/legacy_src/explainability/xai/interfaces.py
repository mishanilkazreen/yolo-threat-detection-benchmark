"""Core interfaces and data models for XAI methods."""

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np


class AttributionMethod(Enum):
    """Enumeration of supported XAI attribution methods."""

    GRADCAM = "gradcam"
    LRP = "lrp"
    SHAP = "shap"


@dataclass
class XAIResult:
    """Result from XAI attribution method processing.

    Attributes:
        method: The XAI method used to generate the attribution
        attribution_map: 2D numpy array containing attribution values
        hfs_score: Heatmap Focus Score (None for methods that don't support HFS)
        output_path: Path where the attribution visualization was saved
        processing_time: Time taken to generate the attribution in seconds
        image_id: Identifier for the processed image
        metadata: Additional method-specific metadata
    """

    method: AttributionMethod
    attribution_map: np.ndarray
    hfs_score: float | None
    output_path: str
    processing_time: float
    image_id: str
    metadata: dict[str, Any] | None = None


@dataclass
class Detection:
    """A single model detection in YOLO format (normalized coordinates).

    Distinct from annotation BoundingBox (ground truth): carries class_id
    and confidence from the model output.
    """

    x_center: float
    y_center: float
    width: float
    height: float
    class_id: int
    confidence: float = 1.0

    def to_tuple(self) -> tuple[float, float, float, float]:
        """Return (x_center, y_center, width, height)."""
        return (self.x_center, self.y_center, self.width, self.height)

    def to_pixel_coords(self, image_width: int, image_height: int) -> tuple[int, int, int, int]:
        """Convert to pixel bounding box (x1, y1, x2, y2)."""
        x_center_abs = self.x_center * image_width
        y_center_abs = self.y_center * image_height
        w_abs = self.width * image_width
        h_abs = self.height * image_height

        x1 = int(max(0, x_center_abs - w_abs / 2))
        y1 = int(max(0, y_center_abs - h_abs / 2))
        x2 = int(min(image_width, x_center_abs + w_abs / 2))
        y2 = int(min(image_height, y_center_abs + h_abs / 2))

        return (x1, y1, x2, y2)


@dataclass
class XAIBatchResult:
    """Results from processing a batch of images through XAI methods.

    Attributes:
        round_num: Training round number
        results: List of individual XAI results
        aggregate_hfs: Mean HFS scores by method
        total_processing_time: Total time for batch processing
        num_images_processed: Number of images successfully processed
        failed_images: List of image IDs that failed processing
    """

    round_num: int
    results: list[XAIResult]
    aggregate_hfs: dict[str, float]
    total_processing_time: float
    num_images_processed: int
    failed_images: list[str]
