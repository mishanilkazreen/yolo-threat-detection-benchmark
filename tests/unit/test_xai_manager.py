"""Unit tests for XAI Manager."""

import tempfile
from unittest.mock import MagicMock, patch

import numpy as np

from src.explainability.xai.config import XAIConfig
from src.explainability.xai.interfaces import AttributionMethod
from src.explainability.xai.manager import XAIManager


class TestXAIManager:
    """Test XAI Manager functionality."""

    def test_manager_disabled(self):
        """Test XAI manager when disabled."""
        config = XAIConfig(enabled=False)
        manager = XAIManager(config)

        assert manager.is_enabled() is False

        # Should return None when processing batch
        result = manager.process_batch(
            model=MagicMock(),
            image_paths=["test.jpg"],
            detections={},
            gt_boxes={},
            round_num=1,
            output_dir="/tmp",
        )

        assert result is None

    def test_manager_enabled_no_methods(self):
        """Test XAI manager when enabled but no methods are enabled."""
        config = XAIConfig(enabled=True, methods={"gradcam": False, "lrp": False, "shap": False})
        manager = XAIManager(config)

        assert manager.is_enabled() is True

        # Should return None when no methods are enabled
        result = manager.process_batch(
            model=MagicMock(),
            image_paths=["test.jpg"],
            detections={},
            gt_boxes={},
            round_num=1,
            output_dir="/tmp",
        )

        assert result is None

    def test_get_target_layer(self):
        """Test target layer resolution."""
        config = XAIConfig()
        manager = XAIManager(config)

        # Mock model with different architectures
        model = MagicMock()

        # Test with model_name attribute
        model.model_name = "yolov11n"
        assert manager._get_target_layer(model) == "model.22"

        # Test with yaml attribute
        model.model_name = None
        model.yaml = {"name": "yolo26s"}
        assert manager._get_target_layer(model) == "model.22"

        # Test fallback
        model.yaml = {}
        assert manager._get_target_layer(model) == "model.22"  # Default (yolov11)

    def test_sample_limit_application(self):
        """Test that sample limit is correctly applied."""
        config = XAIConfig(enabled=True, methods={"gradcam": True}, sample_limit=2)
        manager = XAIManager(config)

        # Create more images than the limit
        image_paths = ["img1.jpg", "img2.jpg", "img3.jpg", "img4.jpg"]

        with patch.object(manager, "_process_single_image") as mock_process:
            mock_process.return_value = None  # Simulate processing

            manager.process_batch(
                model=MagicMock(),
                image_paths=image_paths,
                detections={},
                gt_boxes={},
                round_num=1,
                output_dir="/tmp",
            )

            # Should only process first 2 images due to sample limit
            assert mock_process.call_count == 2

    @patch("src.explainability.xai.manager.generate_gradcam_attribution")
    def test_process_single_image_gradcam(self, mock_gradcam):
        """Test processing single image with Grad-CAM."""
        config = XAIConfig(enabled=True, methods={"gradcam": True})
        manager = XAIManager(config)

        # Mock Grad-CAM output
        mock_attribution = np.random.rand(64, 64)
        mock_gradcam.return_value = (mock_attribution, 0.75, "/path/output.png")

        # Create temporary directory for output
        with tempfile.TemporaryDirectory() as temp_dir:
            result = manager._process_single_image(
                model=MagicMock(),
                image_path="test.jpg",
                detections={},
                gt_boxes=[(0.5, 0.5, 0.4, 0.6)],
                method="gradcam",
                output_dir=temp_dir,
                image_id="test",
            )

            assert result is not None
            assert result.method == AttributionMethod.GRADCAM
            assert result.hfs_score == 0.75
            assert result.image_id == "test"
            assert np.array_equal(result.attribution_map, mock_attribution)

    def test_compute_aggregate_hfs(self):
        """Test HFS aggregation computation."""
        config = XAIConfig()
        manager = XAIManager(config)

        # Create mock results
        from src.explainability.xai.interfaces import XAIResult

        results = [
            XAIResult(
                method=AttributionMethod.GRADCAM,
                attribution_map=np.random.rand(64, 64),
                hfs_score=0.8,
                output_path="",
                processing_time=1.0,
                image_id="img1",
            ),
            XAIResult(
                method=AttributionMethod.GRADCAM,
                attribution_map=np.random.rand(64, 64),
                hfs_score=0.6,
                output_path="",
                processing_time=1.0,
                image_id="img2",
            ),
            XAIResult(
                method=AttributionMethod.LRP,
                attribution_map=np.random.rand(64, 64),
                hfs_score=0.7,
                output_path="",
                processing_time=2.0,
                image_id="img1",
            ),
        ]

        aggregate_hfs = manager._compute_aggregate_hfs(results)

        # Should have mean HFS for each method
        assert "gradcam" in aggregate_hfs
        assert "lrp" in aggregate_hfs
        assert aggregate_hfs["gradcam"] == 0.7  # (0.8 + 0.6) / 2
        assert aggregate_hfs["lrp"] == 0.7
