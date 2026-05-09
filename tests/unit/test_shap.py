"""Tests for SHAP attribution generation."""

import contextlib
from pathlib import Path
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image
import torch

from src.explainability.xai.shap import (
    create_background_set,
    generate_shap_attribution,
)


class TestCreateBackgroundSet:
    """Test background set creation for SHAP."""

    def create_test_images(self, num_images: int = 5) -> list[str]:
        """Create test image files."""
        image_paths = []
        for i in range(num_images):
            with tempfile.NamedTemporaryFile(suffix=f"_{i}.jpg", delete=False) as temp_file:
                image = Image.new("RGB", (640, 480), color=("red" if i % 2 == 0 else "blue"))
                image.save(temp_file.name)
                image_paths.append(temp_file.name)
        return image_paths

    def cleanup_images(self, image_paths: list[str]):
        """Clean up test image files."""
        for path in image_paths:
            with contextlib.suppress(FileNotFoundError):
                Path(path).unlink()

    def test_create_background_set_success(self):
        """Test successful background set creation."""
        image_paths = self.create_test_images(10)

        try:
            background_set = create_background_set(
                validation_images=image_paths, background_set_size=5, seed=42
            )

            assert len(background_set) == 5
            for image in background_set:
                assert isinstance(image, np.ndarray)
                assert image.ndim == 3  # Height, Width, Channels
                assert image.shape[2] == 3  # RGB channels

        finally:
            self.cleanup_images(image_paths)

    def test_create_background_set_deterministic(self):
        """Test that background set creation is deterministic."""
        image_paths = self.create_test_images(10)

        try:
            # Create two background sets with same seed
            background_set1 = create_background_set(
                validation_images=image_paths, background_set_size=5, seed=42
            )

            background_set2 = create_background_set(
                validation_images=image_paths, background_set_size=5, seed=42
            )

            # Should be identical
            assert len(background_set1) == len(background_set2)
            for img1, img2 in zip(background_set1, background_set2):
                np.testing.assert_array_equal(img1, img2)

        finally:
            self.cleanup_images(image_paths)

    def test_create_background_set_different_seeds(self):
        """Test that different seeds produce different background sets."""
        image_paths = self.create_test_images(10)

        try:
            background_set1 = create_background_set(
                validation_images=image_paths, background_set_size=5, seed=42
            )

            background_set2 = create_background_set(
                validation_images=image_paths, background_set_size=5, seed=123
            )

            # Should be different (with high probability)
            assert len(background_set1) == len(background_set2)
            # At least one image should be different
            different = False
            for img1, img2 in zip(background_set1, background_set2):
                if not np.array_equal(img1, img2):
                    different = True
                    break
            assert different, "Background sets should be different with different seeds"

        finally:
            self.cleanup_images(image_paths)

    def test_create_background_set_insufficient_images(self):
        """Test background set creation with insufficient validation images."""
        image_paths = self.create_test_images(3)

        try:
            background_set = create_background_set(
                validation_images=image_paths,
                background_set_size=10,  # More than available
                seed=42,
            )

            # Should use all available images
            assert len(background_set) == 3

        finally:
            self.cleanup_images(image_paths)

    def test_create_background_set_invalid_image(self):
        """Test background set creation with some invalid image paths."""
        valid_paths = self.create_test_images(3)
        invalid_paths = ["nonexistent1.jpg", "nonexistent2.jpg"]
        all_paths = valid_paths + invalid_paths

        try:
            background_set = create_background_set(
                validation_images=all_paths, background_set_size=5, seed=42
            )

            # Should only include valid images
            assert len(background_set) <= 3  # Only valid images loaded

        finally:
            self.cleanup_images(valid_paths)


class TestGenerateSHAPAttribution:
    """Test SHAP attribution generation."""

    def create_test_image(self):
        """Create a test image file."""
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
            image = Image.new("RGB", (640, 480), color="red")
            image.save(temp_file.name)
            return temp_file.name

    def create_mock_model(self):
        """Create a mock YOLO model for testing."""
        # Create inner model mock
        inner_model = MagicMock()
        inner_model.eval = MagicMock(return_value=None)
        inner_model.requires_grad_ = MagicMock(return_value=None)
        inner_model.zero_grad = MagicMock(return_value=None)
        inner_model.modules = MagicMock(return_value=[])

        # Mock forward pass that returns tensor with confidence scores
        def mock_forward(x):
            batch_size = x.shape[0]
            # Return mock YOLO output: [batch, detections, features]
            # Features: [x, y, w, h, confidence, class_probs...]
            return torch.tensor([[[0.5, 0.5, 0.3, 0.4, 0.8, 0.9]]] * batch_size, requires_grad=True)

        inner_model.side_effect = mock_forward

        # Create outer model mock
        model = MagicMock()
        model.model = inner_model
        model.imgsz = 640

        return model

    def create_background_set(self, size: int = 3) -> list[np.ndarray]:
        """Create a small background set for testing."""
        background_images = []
        for _i in range(size):
            # Create simple test images
            image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            background_images.append(image)
        return background_images

    def test_generate_shap_attribution_success(self):
        """Test successful SHAP attribution generation."""
        # Setup mocks
        model = self.create_mock_model()
        image_path = self.create_test_image()
        background_set = self.create_background_set()

        # Mock SHAP explainer and values
        mock_explainer = MagicMock()
        mock_shap_values = np.random.randn(1, 3, 640, 640)  # [batch, channels, height, width]
        mock_explainer.shap_values.return_value = mock_shap_values

        with patch("shap.GradientExplainer", return_value=mock_explainer):
            try:
                with tempfile.TemporaryDirectory() as temp_dir:
                    attribution_map, hfs_score, output_path = generate_shap_attribution(
                        model=model,
                        image_path=image_path,
                        detections={},
                        gt_boxes=[(0.5, 0.5, 0.4, 0.6)],
                        background_set=background_set,
                        device="cpu",
                        output_dir=temp_dir,
                    )

                    # Verify results
                    assert isinstance(attribution_map, np.ndarray)
                    assert attribution_map.shape == (480, 640)  # Should match image size
                    assert hfs_score is None or isinstance(hfs_score, float)
                    assert output_path.endswith(".png")
                    assert Path(output_path).exists()

            finally:
                Path(image_path).unlink()

    def test_generate_shap_attribution_no_output_dir(self):
        """Test SHAP attribution without output directory."""
        model = self.create_mock_model()
        image_path = self.create_test_image()
        background_set = self.create_background_set()

        # Mock SHAP explainer and values
        mock_explainer = MagicMock()
        mock_shap_values = np.random.randn(1, 3, 640, 640)
        mock_explainer.shap_values.return_value = mock_shap_values

        with patch("shap.GradientExplainer", return_value=mock_explainer):
            try:
                attribution_map, hfs_score, output_path = generate_shap_attribution(
                    model=model,
                    image_path=image_path,
                    detections={},
                    gt_boxes=[(0.5, 0.5, 0.4, 0.6)],
                    background_set=background_set,
                    device="cpu",
                    output_dir=None,  # No output directory
                )

                # Should still generate attribution map but no output file
                assert isinstance(attribution_map, np.ndarray)
                assert hfs_score is None or isinstance(hfs_score, float)
                assert output_path == ""

            finally:
                Path(image_path).unlink()

    def test_generate_shap_attribution_list_shap_values(self):
        """Test SHAP attribution with list of SHAP values (multi-class case)."""
        model = self.create_mock_model()
        image_path = self.create_test_image()
        background_set = self.create_background_set()

        # Mock SHAP explainer with list of values (multi-class)
        mock_explainer = MagicMock()
        mock_shap_values = [
            np.random.randn(1, 3, 640, 640),  # Class 0
            np.random.randn(1, 3, 640, 640),  # Class 1
        ]
        mock_explainer.shap_values.return_value = mock_shap_values

        with patch("shap.GradientExplainer", return_value=mock_explainer):
            try:
                attribution_map, hfs_score, _ = generate_shap_attribution(
                    model=model,
                    image_path=image_path,
                    detections={},
                    gt_boxes=[(0.5, 0.5, 0.4, 0.6)],
                    background_set=background_set,
                    device="cpu",
                )

                # Should use first class values
                assert isinstance(attribution_map, np.ndarray)
                assert attribution_map.shape == (480, 640)
                assert hfs_score is None or isinstance(hfs_score, float)

            finally:
                Path(image_path).unlink()

    def test_generate_shap_attribution_invalid_image_path(self):
        """Test SHAP attribution with invalid image path."""
        model = self.create_mock_model()
        background_set = self.create_background_set()

        # Function catches exceptions and returns empty attribution map
        attribution_map, hfs_score, output_path = generate_shap_attribution(
            model=model,
            image_path="nonexistent_image.jpg",
            detections={},
            gt_boxes=[(0.5, 0.5, 0.4, 0.6)],
            background_set=background_set,
            device="cpu",
        )

        # Should return zero attribution map and empty output path on error
        assert isinstance(attribution_map, np.ndarray)
        assert attribution_map.shape == (640, 640)  # Default size
        assert np.all(attribution_map == 0)
        assert hfs_score is None or isinstance(hfs_score, float)
        assert output_path == ""


class TestSHAPHelperFunctions:
    """Test SHAP helper functions."""

    def test_preprocess_image_shap(self):
        """Test SHAP image preprocessing."""
        from src.explainability.xai.shap import _preprocess_image_shap

        # Create test image
        image_np = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        # Test preprocessing
        tensor = _preprocess_image_shap(image_np, target_size=640)

        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (1, 3, 640, 640)
        assert tensor.dtype == torch.float32
        assert 0 <= tensor.min() <= tensor.max() <= 1

    def test_prepare_background_set(self):
        """Test background set preparation."""
        from src.explainability.xai.shap import _prepare_background_set

        # Create background images
        background_images = []
        for _i in range(3):
            image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            background_images.append(image)

        # Test preparation
        background_tensor = _prepare_background_set(background_images, "cpu", target_size=640)

        assert isinstance(background_tensor, torch.Tensor)
        assert background_tensor.shape == (3, 3, 640, 640)  # [batch, channels, height, width]
        assert background_tensor.dtype == torch.float32

    def test_extract_shap_target(self):
        """Test SHAP target extraction from YOLO outputs."""
        from src.explainability.xai.shap import _extract_shap_target

        # Test with 3D tensor in YOLO format: [batch, features, N_anchors]
        # Features: [x, y, w, h, class1_score, class2_score]
        outputs = torch.tensor([[[0.5], [0.5], [0.3], [0.4], [0.8], [0.9]]], requires_grad=True)
        target = _extract_shap_target(outputs)

        assert isinstance(target, torch.Tensor)
        assert target.shape == (1, 1)  # [batch_size, 1]
        # Function sums class scores (indices 4+): 0.8 + 0.9 = 1.7
        assert abs(target.item() - 1.7) < 1e-6

    def test_extract_shap_target_list_output(self):
        """Test SHAP target extraction from list output."""
        from src.explainability.xai.shap import _extract_shap_target

        # Test with list output in YOLO format: [batch, features, N_anchors]
        outputs = [torch.tensor([[[0.5], [0.5], [0.3], [0.4], [0.8], [0.9]]], requires_grad=True)]
        target = _extract_shap_target(outputs)

        assert isinstance(target, torch.Tensor)
        assert target.shape == (1, 1)  # [batch_size, 1]
        # Function sums class scores (indices 4+): 0.8 + 0.9 = 1.7
        assert abs(target.item() - 1.7) < 1e-6

    def test_extract_shap_target_2d_tensor(self):
        """Test SHAP target extraction from 2D tensor."""
        from src.explainability.xai.shap import _extract_shap_target

        # Test with 2D tensor
        outputs = torch.tensor([[0.1, 0.8, 0.3]], requires_grad=True)
        target = _extract_shap_target(outputs)

        assert isinstance(target, torch.Tensor)
        assert target.shape == (1, 1)  # [batch_size, 1]
        assert abs(target.item() - 0.8) < 1e-6  # Should return maximum value
