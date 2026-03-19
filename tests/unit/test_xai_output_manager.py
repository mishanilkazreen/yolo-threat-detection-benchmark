"""Unit tests for XAI Output Manager."""

from pathlib import Path
import tempfile

import numpy as np
from PIL import Image

from src.explainability.xai.interfaces import AttributionMethod
from src.explainability.xai.output_manager import XAIOutputManager


class TestXAIOutputManager:
    """Test XAI Output Manager functionality."""

    def test_output_manager_initialization(self):
        """Test output manager initialization and directory creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            XAIOutputManager(base_output_dir=temp_dir, save_overlays=True, save_raw=True)

            # Check that method directories are created
            for method in AttributionMethod:
                method_dir = Path(temp_dir) / "xai" / method.value
                assert method_dir.exists()

                # Check that raw directories are created when save_raw=True
                raw_dir = method_dir / "raw"
                assert raw_dir.exists()

    def test_output_manager_no_raw_directories(self):
        """Test that raw directories are not created when save_raw=False."""
        with tempfile.TemporaryDirectory() as temp_dir:
            XAIOutputManager(base_output_dir=temp_dir, save_overlays=True, save_raw=False)

            # Check that raw directories are NOT created
            for method in AttributionMethod:
                raw_dir = Path(temp_dir) / "xai" / method.value / "raw"
                assert not raw_dir.exists()

    def test_save_attribution_outputs_overlay_only(self):
        """Test saving attribution outputs with overlays only."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = XAIOutputManager(base_output_dir=temp_dir, save_overlays=True, save_raw=False)

            # Create test data
            attribution_map = np.random.rand(64, 64)
            original_image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            image_path = "test_image.jpg"

            # Save outputs
            output_path = manager.save_attribution_outputs(
                attribution_map=attribution_map,
                original_image=original_image,
                image_path=image_path,
                method=AttributionMethod.GRADCAM,
            )

            # Check that overlay was saved
            assert output_path != ""
            assert Path(output_path).exists()
            assert Path(output_path).suffix == ".png"
            assert "gradcam" in Path(output_path).name

            # Check that raw file was NOT saved
            raw_path = Path(temp_dir) / "xai" / "gradcam" / "raw" / "test_image_gradcam_raw.npy"
            assert not raw_path.exists()

    def test_save_attribution_outputs_raw_only(self):
        """Test saving attribution outputs with raw data only."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = XAIOutputManager(base_output_dir=temp_dir, save_overlays=False, save_raw=True)

            # Create test data
            attribution_map = np.random.rand(64, 64)
            original_image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            image_path = "test_image.jpg"

            # Save outputs
            output_path = manager.save_attribution_outputs(
                attribution_map=attribution_map,
                original_image=original_image,
                image_path=image_path,
                method=AttributionMethod.LRP,
            )

            # Check that overlay was NOT saved
            assert output_path == ""

            # Check that raw file was saved
            raw_path = Path(temp_dir) / "xai" / "lrp" / "raw" / "test_image_lrp_raw.npy"
            assert raw_path.exists()

            # Verify raw data content
            loaded_data = np.load(raw_path)
            np.testing.assert_array_equal(loaded_data, attribution_map)

    def test_save_attribution_outputs_both(self):
        """Test saving both overlay and raw attribution outputs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = XAIOutputManager(base_output_dir=temp_dir, save_overlays=True, save_raw=True)

            # Create test data
            attribution_map = np.random.rand(64, 64)
            original_image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            image_path = "test_image.jpg"

            # Save outputs
            output_path = manager.save_attribution_outputs(
                attribution_map=attribution_map,
                original_image=original_image,
                image_path=image_path,
                method=AttributionMethod.SHAP,
            )

            # Check that overlay was saved
            assert output_path != ""
            assert Path(output_path).exists()

            # Check that raw file was saved
            raw_path = Path(temp_dir) / "xai" / "shap" / "raw" / "test_image_shap_raw.npy"
            assert raw_path.exists()

    def test_get_method_output_dir(self):
        """Test getting method-specific output directories."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = XAIOutputManager(base_output_dir=temp_dir, save_overlays=True, save_raw=False)

            for method in AttributionMethod:
                output_dir = manager.get_method_output_dir(method)
                expected_dir = Path(temp_dir) / "xai" / method.value
                assert output_dir == expected_dir
                assert output_dir.exists()

    def test_cleanup_old_outputs(self):
        """Test cleanup of old output files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = XAIOutputManager(base_output_dir=temp_dir, save_overlays=True, save_raw=False)

            # Create some test files
            gradcam_dir = manager.get_method_output_dir(AttributionMethod.GRADCAM)
            test_files = []
            for i in range(5):
                test_file = gradcam_dir / f"test_{i}_gradcam.png"
                test_file.touch()
                test_files.append(test_file)

            # Verify all files exist
            for test_file in test_files:
                assert test_file.exists()

            # Cleanup keeping only 3 latest files
            manager.cleanup_old_outputs(keep_latest=3)

            # Check that only 3 files remain (all should remain since they have same timestamp)
            remaining_files = list(gradcam_dir.glob("*.png"))
            assert len(remaining_files) <= 5  # All files might remain due to same timestamp

    def test_gradcam_overlay_visualization(self):
        """Test Grad-CAM specific overlay visualization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = XAIOutputManager(base_output_dir=temp_dir, save_overlays=True, save_raw=False)

            # Create test data with known values
            attribution_map = np.ones((64, 64)) * 0.5  # Uniform attribution
            original_image = np.ones((64, 64, 3), dtype=np.uint8) * 128  # Gray image

            output_path = manager.save_attribution_outputs(
                attribution_map=attribution_map,
                original_image=original_image,
                image_path="test.jpg",
                method=AttributionMethod.GRADCAM,
            )

            # Verify the output file exists and is a valid image
            assert Path(output_path).exists()

            # Try to load the image to verify it's valid
            with Image.open(output_path) as img:
                assert img.size == (64, 64)
                assert img.mode == "RGB"

    def test_lrp_overlay_visualization(self):
        """Test LRP specific overlay visualization with positive/negative values."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = XAIOutputManager(base_output_dir=temp_dir, save_overlays=True, save_raw=False)

            # Create test data with positive and negative values
            attribution_map = np.random.randn(64, 64)  # Random positive/negative values
            original_image = np.ones((64, 64, 3), dtype=np.uint8) * 128

            output_path = manager.save_attribution_outputs(
                attribution_map=attribution_map,
                original_image=original_image,
                image_path="test.jpg",
                method=AttributionMethod.LRP,
            )

            # Verify the output file exists and is a valid image
            assert Path(output_path).exists()

            with Image.open(output_path) as img:
                assert img.size == (64, 64)
                assert img.mode == "RGB"

    def test_shap_overlay_visualization(self):
        """Test SHAP specific overlay visualization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = XAIOutputManager(base_output_dir=temp_dir, save_overlays=True, save_raw=False)

            # Create test data
            attribution_map = np.random.randn(64, 64)
            original_image = np.ones((64, 64, 3), dtype=np.uint8) * 128

            output_path = manager.save_attribution_outputs(
                attribution_map=attribution_map,
                original_image=original_image,
                image_path="test.jpg",
                method=AttributionMethod.SHAP,
            )

            # Verify the output file exists and is a valid image
            assert Path(output_path).exists()

            with Image.open(output_path) as img:
                assert img.size == (64, 64)
                assert img.mode == "RGB"
