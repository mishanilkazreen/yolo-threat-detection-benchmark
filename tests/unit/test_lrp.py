"""Unit tests for LRP (Layer-wise Relevance Propagation) implementation."""

import matplotlib

matplotlib.use("Agg")  # non-interactive backend for headless test environments

from pathlib import Path
import tempfile

import numpy as np
from PIL import Image
import pytest
from torch import nn

from src.explainability.xai.lrp import _save_lrp_visualization, generate_lrp_attribution


class TestLRPAttribution:
    """Test LRP attribution generation."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock YOLO model."""

        # Create a simple mock PyTorch model
        class MockYOLOModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 64, 3, padding=1)
                self.pool = nn.AdaptiveAvgPool2d((1, 1))
                self.fc = nn.Linear(64, 85)  # YOLO output features

            def forward(self, x):
                x = self.conv(x)
                x = self.pool(x)
                x = x.view(x.size(0), -1)
                x = self.fc(x)
                # Return in YOLO format: [batch, detections, features]
                return x.unsqueeze(1).expand(-1, 25200, -1)

        model = MockYOLOModel()
        model.imgsz = 640
        model.eval()
        return model

    @pytest.fixture
    def sample_image(self):
        """Create a sample test image."""
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            # Create a simple RGB image
            image = Image.new("RGB", (640, 480), color="red")
            image.save(f.name)
            yield f.name
        Path(f.name).unlink()

    @pytest.fixture
    def sample_detections(self):
        """Sample detection results."""
        return {"boxes": [[100, 100, 200, 200]], "scores": [0.9], "classes": [0]}

    @pytest.fixture
    def sample_gt_boxes(self):
        """Sample ground truth boxes in YOLO format (x_center, y_center, width, height)."""
        return [(0.25, 0.25, 0.2, 0.2)]  # Normalized coordinates

    def test_lrp_basic_functionality(
        self, mock_model, sample_image, sample_detections, sample_gt_boxes
    ):
        """Test basic LRP functionality with real libraries."""
        try:
            relevance_map, bbox_score, output_path = generate_lrp_attribution(
                model=mock_model,
                image_path=sample_image,
                detections=sample_detections,
                gt_boxes=sample_gt_boxes,
                rule="epsilon",
                device="cpu",
            )

            # Verify outputs
            assert isinstance(relevance_map, np.ndarray)
            assert relevance_map.shape == (480, 640)  # Should match original image size
            assert isinstance(bbox_score, (float, type(None)))
            assert isinstance(output_path, str)

        except ImportError as e:
            pytest.skip(f"Required libraries not available: {e}")

    def test_lrp_different_rules(
        self, mock_model, sample_image, sample_detections, sample_gt_boxes
    ):
        """Test LRP with different rules."""
        rules = ["epsilon", "gamma", "alpha-beta", "unknown"]

        try:
            for rule in rules:
                relevance_map, *_ = generate_lrp_attribution(
                    model=mock_model,
                    image_path=sample_image,
                    detections=sample_detections,
                    gt_boxes=sample_gt_boxes,
                    rule=rule,
                    device="cpu",
                )

                assert isinstance(relevance_map, np.ndarray)
                assert relevance_map.shape == (480, 640)

        except ImportError as e:
            pytest.skip(f"Required libraries not available: {e}")

    def test_lrp_with_output_dir(
        self, mock_model, sample_image, sample_detections, sample_gt_boxes
    ):
        """Test LRP with output directory for saving visualizations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                _, _bbox_score, output_path = generate_lrp_attribution(
                    model=mock_model,
                    image_path=sample_image,
                    detections=sample_detections,
                    gt_boxes=sample_gt_boxes,
                    rule="epsilon",
                    device="cpu",
                    output_dir=temp_dir,
                )

                # Verify output file was created
                assert output_path != ""
                assert Path(output_path).exists()

            except ImportError as e:
                pytest.skip(f"Required libraries not available: {e}")

    def test_lrp_empty_gt_boxes(self, mock_model, sample_image, sample_detections):
        """Test LRP with empty ground truth boxes."""
        try:
            relevance_map, bbox_score, output_path = generate_lrp_attribution(
                model=mock_model,
                image_path=sample_image,
                detections=sample_detections,
                gt_boxes=[],  # Empty ground truth boxes
                rule="epsilon",
                device="cpu",
            )

            # Verify outputs
            assert isinstance(relevance_map, np.ndarray)
            assert bbox_score is None  # Should be None when no gt_boxes
            assert isinstance(output_path, str)

        except ImportError as e:
            pytest.skip(f"Required libraries not available: {e}")

    def test_lrp_invalid_image_path(self, mock_model, sample_detections, sample_gt_boxes):
        """Test LRP with invalid image path."""
        with pytest.raises(RuntimeError, match="LRP computation failed"):
            generate_lrp_attribution(
                model=mock_model,
                image_path="nonexistent_image.jpg",
                detections=sample_detections,
                gt_boxes=sample_gt_boxes,
                rule="epsilon",
                device="cpu",
            )


class TestSaveLrpVisualization:
    """Tests for _save_lrp_visualization — no model required."""

    @pytest.fixture
    def sample_image_np(self):
        """3-channel RGB image as numpy array."""
        return np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)

    @pytest.fixture
    def sample_image_path(self, tmp_path):
        """Temporary image file path."""
        p = tmp_path / "test_image.jpg"
        img = Image.new("RGB", (64, 64), color="blue")
        img.save(str(p))
        return str(p)

    def test_output_file_created(self, sample_image_np, sample_image_path, tmp_path):
        """Output PNG file is created and path is returned as string."""
        relevance = np.random.rand(64, 64).astype(np.float32)
        result = _save_lrp_visualization(
            sample_image_np, relevance, sample_image_path, str(tmp_path)
        )
        assert isinstance(result, str)
        assert Path(result).exists()

    def test_filename_pattern(self, sample_image_np, sample_image_path, tmp_path):
        """Output filename follows {stem}_lrp.png pattern."""
        relevance = np.random.rand(64, 64).astype(np.float32)
        result = _save_lrp_visualization(
            sample_image_np, relevance, sample_image_path, str(tmp_path)
        )
        assert Path(result).name == "test_image_lrp.png"

    def test_return_value_is_string(self, sample_image_np, sample_image_path, tmp_path):
        """Return value is a string (absolute path)."""
        relevance = np.random.rand(64, 64).astype(np.float32)
        result = _save_lrp_visualization(
            sample_image_np, relevance, sample_image_path, str(tmp_path)
        )
        assert isinstance(result, str)

    def test_diverging_colormap_produces_file(self, sample_image_np, sample_image_path, tmp_path):
        """Mixed-sign relevance map produces a non-empty PNG file."""
        relevance = np.random.uniform(-1.0, 1.0, (64, 64)).astype(np.float32)
        result = _save_lrp_visualization(
            sample_image_np, relevance, sample_image_path, str(tmp_path)
        )
        assert Path(result).exists()
        assert Path(result).stat().st_size > 0

    def test_sequential_colormap_produces_file(self, sample_image_np, sample_image_path, tmp_path):
        """Non-negative relevance map uses hot colormap path and produces a file."""
        relevance = np.random.rand(64, 64).astype(np.float32)  # all >= 0
        result = _save_lrp_visualization(
            sample_image_np, relevance, sample_image_path, str(tmp_path)
        )
        assert Path(result).exists()

    def test_zero_max_relevance_no_raise(self, sample_image_np, sample_image_path, tmp_path):
        """Zero-max relevance map does not raise and produces a file."""
        relevance = np.zeros((64, 64), dtype=np.float32)
        result = _save_lrp_visualization(
            sample_image_np, relevance, sample_image_path, str(tmp_path)
        )
        assert Path(result).exists()

    def test_single_channel_image_no_raise(self, sample_image_path, tmp_path):
        """Single-channel (grayscale) image does not raise and produces a file."""
        image_np = np.random.randint(0, 255, (64, 64), dtype=np.uint8)  # 2D
        relevance = np.random.rand(64, 64).astype(np.float32)
        result = _save_lrp_visualization(image_np, relevance, sample_image_path, str(tmp_path))
        assert Path(result).exists()

    def test_single_channel_hw1_no_raise(self, sample_image_path, tmp_path):
        """Single-channel (H,W,1) image does not raise and produces a file."""
        image_np = np.random.randint(0, 255, (64, 64, 1), dtype=np.uint8)
        relevance = np.random.rand(64, 64).astype(np.float32)
        result = _save_lrp_visualization(image_np, relevance, sample_image_path, str(tmp_path))
        assert Path(result).exists()

    def test_mismatched_dimensions_resized(self, sample_image_np, sample_image_path, tmp_path):
        """Relevance map smaller than image is resized; output file produced, no exception."""
        relevance = np.random.rand(32, 32).astype(np.float32)  # smaller than 64x64 image
        result = _save_lrp_visualization(
            sample_image_np, relevance, sample_image_path, str(tmp_path)
        )
        assert Path(result).exists()
