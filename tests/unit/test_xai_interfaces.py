"""Unit tests for XAI interfaces and data models."""

import numpy as np

from src.explainability.xai.interfaces import AttributionMethod, BoundingBox, XAIResult


class TestBoundingBox:
    """Test BoundingBox data model."""

    def test_to_tuple(self):
        """Test conversion to tuple format."""
        bbox = BoundingBox(0.5, 0.3, 0.4, 0.6, class_id=1, confidence=0.9)
        assert bbox.to_tuple() == (0.5, 0.3, 0.4, 0.6)

    def test_to_pixel_coords(self):
        """Test conversion to pixel coordinates."""
        bbox = BoundingBox(0.5, 0.5, 0.4, 0.6, class_id=1)  # Center box
        x1, y1, x2, y2 = bbox.to_pixel_coords(100, 200)

        # Expected: center at (50, 100), size 40x120
        assert x1 == 30  # 50 - 20
        assert y1 == 40  # 100 - 60
        assert x2 == 70  # 50 + 20
        assert y2 == 160  # 100 + 60

    def test_to_pixel_coords_edge_cases(self):
        """Test pixel coordinate conversion with edge cases."""
        # Box at image boundary
        bbox = BoundingBox(0.0, 0.0, 0.2, 0.2, class_id=1)
        x1, y1, x2, y2 = bbox.to_pixel_coords(100, 100)

        assert x1 >= 0
        assert y1 >= 0
        assert x2 <= 100
        assert y2 <= 100


class TestXAIResult:
    """Test XAIResult data model."""

    def test_xai_result_creation(self):
        """Test XAI result creation."""
        attribution_map = np.random.rand(64, 64)

        result = XAIResult(
            method=AttributionMethod.GRADCAM,
            attribution_map=attribution_map,
            hfs_score=0.75,
            output_path="/path/to/output.png",
            processing_time=1.5,
            image_id="test_image",
        )

        assert result.method == AttributionMethod.GRADCAM
        assert result.hfs_score == 0.75
        assert result.processing_time == 1.5
        assert result.image_id == "test_image"
        assert np.array_equal(result.attribution_map, attribution_map)


class TestAttributionMethod:
    """Test AttributionMethod enumeration."""

    def test_attribution_method_values(self):
        """Test attribution method enum values."""
        assert AttributionMethod.GRADCAM.value == "gradcam"
        assert AttributionMethod.LRP.value == "lrp"
        assert AttributionMethod.SHAP.value == "shap"
