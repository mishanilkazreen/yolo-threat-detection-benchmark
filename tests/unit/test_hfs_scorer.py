"""Unit tests for HFS (Heatmap Focus Score) computation."""

import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np
import pytest

from src.explainability.hfs_scorer import Heatmap_Focus_Scorer


class TestHeatmapFocusScorer:
    """Test HFS computation functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.scorer = Heatmap_Focus_Scorer()

    def test_compute_hfs_basic(self):
        """Test basic HFS computation."""
        # Create a simple 4x4 heatmap with higher values in center
        heatmap = np.array(
            [[0.1, 0.1, 0.1, 0.1], [0.1, 0.8, 0.8, 0.1], [0.1, 0.8, 0.8, 0.1], [0.1, 0.1, 0.1, 0.1]]
        )

        # Bbox covering the center 2x2 region (normalized coordinates)
        bbox = (0.5, 0.5, 0.5, 0.5)  # center, 50% width/height

        hfs = self.scorer.compute_hfs(heatmap, bbox, 4, 4)

        # Expected: center 2x2 has sum=3.2, total=4.4, HFS=0.7273
        expected_hfs = 3.2 / 4.4
        assert abs(hfs - expected_hfs) < 0.001

    def test_compute_hfs_perfect_focus(self):
        """Test HFS when all activation is inside bbox."""
        # Heatmap with activation only in center
        heatmap = np.array(
            [[0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 1.0, 0.0], [0.0, 1.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
        )

        # Bbox covering the center region
        bbox = (0.5, 0.5, 0.5, 0.5)

        hfs = self.scorer.compute_hfs(heatmap, bbox, 4, 4)

        # Should be 1.0 (perfect focus)
        assert abs(hfs - 1.0) < 0.001

    def test_compute_hfs_no_focus(self):
        """Test HFS when no activation is inside bbox."""
        # Heatmap with activation only in corners
        heatmap = np.array(
            [[1.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 1.0]]
        )

        # Bbox covering the center region (no activation)
        bbox = (0.5, 0.5, 0.5, 0.5)

        hfs = self.scorer.compute_hfs(heatmap, bbox, 4, 4)

        # Should be 0.0 (no focus)
        assert hfs == 0.0

    def test_compute_hfs_empty_heatmap(self):
        """Test HFS with empty heatmap."""
        heatmap = np.array([])
        bbox = (0.5, 0.5, 0.5, 0.5)

        hfs = self.scorer.compute_hfs(heatmap, bbox, 4, 4)

        assert hfs == 0.0

    def test_compute_hfs_none_heatmap(self):
        """Test HFS with None heatmap."""
        bbox = (0.5, 0.5, 0.5, 0.5)

        hfs = self.scorer.compute_hfs(None, bbox, 4, 4)

        assert hfs == 0.0

    def test_compute_hfs_invalid_heatmap_dimensions(self):
        """Test HFS with invalid heatmap dimensions."""
        # 3D array instead of 2D
        heatmap = np.random.rand(4, 4, 3)
        bbox = (0.5, 0.5, 0.5, 0.5)

        hfs = self.scorer.compute_hfs(heatmap, bbox, 4, 4)

        assert hfs == 0.0

    def test_compute_hfs_zero_intensity(self):
        """Test HFS with zero total intensity."""
        heatmap = np.zeros((4, 4))
        bbox = (0.5, 0.5, 0.5, 0.5)

        hfs = self.scorer.compute_hfs(heatmap, bbox, 4, 4)

        assert hfs == 0.0

    def test_compute_hfs_invalid_image_dimensions(self):
        """Test HFS with invalid image dimensions."""
        heatmap = np.ones((4, 4))
        bbox = (0.5, 0.5, 0.5, 0.5)

        # Zero or negative dimensions
        hfs1 = self.scorer.compute_hfs(heatmap, bbox, 0, 4)
        hfs2 = self.scorer.compute_hfs(heatmap, bbox, 4, -1)

        assert hfs1 == 0.0
        assert hfs2 == 0.0

    def test_compute_hfs_invalid_bbox_coordinates(self):
        """Test HFS with invalid bbox coordinates."""
        heatmap = np.ones((4, 4))

        # Coordinates outside [0, 1] range
        invalid_bboxes = [
            (-0.1, 0.5, 0.5, 0.5),  # negative x_center
            (0.5, 1.1, 0.5, 0.5),  # y_center > 1
            (0.5, 0.5, -0.1, 0.5),  # negative width
            (0.5, 0.5, 0.5, 0.0),  # zero height
        ]

        for bbox in invalid_bboxes:
            hfs = self.scorer.compute_hfs(heatmap, bbox, 4, 4)
            assert hfs == 0.0

    def test_compute_hfs_bbox_outside_bounds(self):
        """Test HFS with bbox completely outside image bounds."""
        heatmap = np.ones((4, 4))

        # Bbox that results in invalid pixel coordinates
        bbox = (0.0, 0.0, 0.01, 0.01)  # Very small bbox at corner

        hfs = self.scorer.compute_hfs(heatmap, bbox, 4, 4)

        # Should handle gracefully
        assert isinstance(hfs, float)
        assert 0.0 <= hfs <= 1.0

    def test_compute_hfs_heatmap_resize(self):
        """Test HFS with heatmap that needs resizing."""
        # Small heatmap that needs to be resized
        heatmap = np.array([[1.0, 0.0], [0.0, 1.0]])
        bbox = (0.5, 0.5, 0.5, 0.5)

        with patch("scipy.ndimage.zoom") as mock_zoom:
            # Mock zoom to return a 4x4 array
            mock_zoom.return_value = np.ones((4, 4))

            hfs = self.scorer.compute_hfs(heatmap, bbox, 4, 4)

            # Should call zoom for resizing
            mock_zoom.assert_called_once()
            assert isinstance(hfs, float)

    def test_compute_hfs_resize_failure(self):
        """Test HFS when heatmap resizing fails."""
        heatmap = np.array([[1.0, 0.0], [0.0, 1.0]])
        bbox = (0.5, 0.5, 0.5, 0.5)

        with patch("scipy.ndimage.zoom", side_effect=Exception("Resize failed")):
            hfs = self.scorer.compute_hfs(heatmap, bbox, 4, 4)

            # Should return 0.0 on resize failure
            assert hfs == 0.0

    def test_compute_mean_hfs_basic(self):
        """Test basic mean HFS computation."""
        heatmaps = [np.ones((2, 2)), np.ones((2, 2)) * 0.5, np.ones((2, 2)) * 0.8]
        bboxes = [(0.5, 0.5, 1.0, 1.0)] * 3  # Full image bboxes
        image_sizes = [(2, 2)] * 3

        mean_hfs = self.scorer.compute_mean_hfs(heatmaps, bboxes, image_sizes)

        # All should have HFS = 1.0 (full coverage)
        assert abs(mean_hfs - 1.0) < 0.001

    def test_compute_mean_hfs_empty_inputs(self):
        """Test mean HFS with empty inputs."""
        # Empty lists
        mean_hfs1 = self.scorer.compute_mean_hfs([], [], [])
        assert mean_hfs1 == 0.0

        # None inputs
        mean_hfs2 = self.scorer.compute_mean_hfs(None, None, None)
        assert mean_hfs2 == 0.0

    def test_compute_mean_hfs_mismatched_lengths(self):
        """Test mean HFS with mismatched input lengths."""
        heatmaps = [np.ones((2, 2))]
        bboxes = [(0.5, 0.5, 1.0, 1.0), (0.5, 0.5, 0.5, 0.5)]  # Different length
        image_sizes = [(2, 2)]

        with pytest.raises(ValueError, match="must have same length"):
            self.scorer.compute_mean_hfs(heatmaps, bboxes, image_sizes)

    def test_compute_mean_hfs_with_failures(self):
        """Test mean HFS when some individual computations fail."""
        heatmaps = [
            np.ones((2, 2)),
            None,  # This will cause failure
            np.ones((2, 2)) * 0.5,
        ]
        bboxes = [(0.5, 0.5, 1.0, 1.0)] * 3
        image_sizes = [(2, 2)] * 3

        # Should handle failures gracefully and compute mean of valid scores
        mean_hfs = self.scorer.compute_mean_hfs(heatmaps, bboxes, image_sizes)

        # Should be mean of 1.0 and 1.0 (skipping the None heatmap)
        assert abs(mean_hfs - 1.0) < 0.001

    def test_compute_mean_hfs_all_failures(self):
        """Test mean HFS when all individual computations fail."""
        heatmaps = [None, None, None]
        bboxes = [(0.5, 0.5, 1.0, 1.0)] * 3
        image_sizes = [(2, 2)] * 3

        mean_hfs = self.scorer.compute_mean_hfs(heatmaps, bboxes, image_sizes)

        # Should return 0.0 when no valid scores
        assert mean_hfs == 0.0

    def test_save_hfs_metrics(self):
        """Test HFS metrics saving."""
        with tempfile.TemporaryDirectory() as temp_dir:
            round_num = 1
            mean_hfs = 0.75
            individual_hfs = [0.8, 0.7, 0.75]
            image_ids = ["img1", "img2", "img3"]
            method = "gradcam"

            self.scorer.save_hfs_metrics(
                round_num, mean_hfs, individual_hfs, image_ids, temp_dir, method
            )

            # Check that file was created
            metrics_file = Path(temp_dir) / "hfs_metrics.json"
            assert metrics_file.exists()

            # Check file contents
            with open(metrics_file) as f:
                metrics = json.load(f)

            assert metrics["round"] == round_num
            assert metrics["method"] == method
            assert metrics["mean_hfs"] == mean_hfs
            assert metrics["num_images"] == len(individual_hfs)
            assert metrics["individual_hfs"]["img1"] == 0.8
            assert "statistics" in metrics
            assert "min" in metrics["statistics"]
            assert "max" in metrics["statistics"]

    def test_save_hfs_metrics_empty_scores(self):
        """Test HFS metrics saving with empty scores."""
        with tempfile.TemporaryDirectory() as temp_dir:
            self.scorer.save_hfs_metrics(1, 0.0, [], [], temp_dir)

            metrics_file = Path(temp_dir) / "hfs_metrics.json"
            assert metrics_file.exists()

            with open(metrics_file) as f:
                metrics = json.load(f)

            assert metrics["num_images"] == 0
            assert metrics["statistics"]["min"] == 0.0
            assert metrics["statistics"]["max"] == 0.0

    def test_compute_hfs_for_occlusion(self):
        """Test HFS computation for occlusion heatmaps."""
        heatmap = np.ones((4, 4))
        bbox = (0.5, 0.5, 0.5, 0.5)

        hfs = self.scorer.compute_hfs_for_occlusion(heatmap, bbox, 4, 4)

        # Should use same computation as regular HFS
        expected_hfs = self.scorer.compute_hfs(heatmap, bbox, 4, 4)
        assert hfs == expected_hfs

    def test_hfs_bounds_validation(self):
        """Test that HFS is always within [0, 1] bounds."""
        # Create various heatmaps and bboxes
        test_cases = [
            (np.ones((4, 4)), (0.5, 0.5, 1.0, 1.0)),  # Full coverage
            (np.zeros((4, 4)), (0.5, 0.5, 0.5, 0.5)),  # Zero intensity
            (np.random.rand(10, 10), (0.2, 0.3, 0.4, 0.5)),  # Random case
        ]

        for heatmap, bbox in test_cases:
            if heatmap.sum() > 0:  # Skip zero intensity case
                hfs = self.scorer.compute_hfs(heatmap, bbox, heatmap.shape[1], heatmap.shape[0])
                assert 0.0 <= hfs <= 1.0, f"HFS {hfs} out of bounds for case {bbox}"

    def test_compute_bbox_weighted_relevance_basic(self):
        """Test basic bbox-weighted relevance computation with mixed positive/negative values."""
        # Create relevance map with positive and negative values
        relevance_map = np.array(
            [
                [-0.5, -0.3, 0.2, 0.4],
                [-0.2, 0.8, 0.9, 0.1],
                [0.1, 0.6, -0.4, -0.1],
                [0.3, -0.2, 0.5, 0.7],
            ]
        )

        # Bbox covering center 2x2 region (indices 1:3, 1:3)
        bbox = (0.5, 0.5, 0.5, 0.5)  # Center bbox covering middle area

        score = self.scorer.compute_bbox_weighted_relevance(relevance_map, bbox, 4, 4)

        # Expected: bbox region has |0.8| + |0.9| + |0.6| + |-0.4| = 2.7
        # Total absolute relevance = sum of all absolute values = 6.3
        # Score should be 2.7 / 6.3 ≈ 0.429
        expected_score = 2.7 / 6.3
        assert abs(score - expected_score) < 0.01

    def test_compute_bbox_weighted_relevance_all_positive(self):
        """Test bbox-weighted relevance with all positive values."""
        relevance_map = np.array(
            [[0.1, 0.2, 0.3, 0.4], [0.2, 0.8, 0.9, 0.1], [0.1, 0.6, 0.4, 0.1], [0.3, 0.2, 0.5, 0.7]]
        )

        bbox = (0.5, 0.5, 0.5, 0.5)

        score = self.scorer.compute_bbox_weighted_relevance(relevance_map, bbox, 4, 4)

        # Should behave like regular HFS for positive values
        # Bbox region: 0.8 + 0.9 + 0.6 + 0.4 = 2.7
        # Total: sum of all values = 5.9
        expected_score = 2.7 / 5.9
        assert abs(score - expected_score) < 0.01

    def test_compute_bbox_weighted_relevance_all_negative(self):
        """Test bbox-weighted relevance with all negative values."""
        relevance_map = np.array(
            [
                [-0.1, -0.2, -0.3, -0.4],
                [-0.2, -0.8, -0.9, -0.1],
                [-0.1, -0.6, -0.4, -0.1],
                [-0.3, -0.2, -0.5, -0.7],
            ]
        )

        bbox = (0.5, 0.5, 0.5, 0.5)

        score = self.scorer.compute_bbox_weighted_relevance(relevance_map, bbox, 4, 4)

        # Should use absolute values
        # Bbox region: |-0.8| + |-0.9| + |-0.6| + |-0.4| = 2.7
        # Total: sum of all absolute values = 5.9
        expected_score = 2.7 / 5.9
        assert abs(score - expected_score) < 0.01

    def test_compute_bbox_weighted_relevance_zero_total(self):
        """Test bbox-weighted relevance with zero total relevance."""
        relevance_map = np.zeros((4, 4))
        bbox = (0.5, 0.5, 0.5, 0.5)

        score = self.scorer.compute_bbox_weighted_relevance(relevance_map, bbox, 4, 4)

        assert score == 0.0

    def test_compute_bbox_weighted_relevance_empty_map(self):
        """Test bbox-weighted relevance with empty relevance map."""
        relevance_map = np.array([])
        bbox = (0.5, 0.5, 0.5, 0.5)

        score = self.scorer.compute_bbox_weighted_relevance(relevance_map, bbox, 4, 4)

        assert score == 0.0

    def test_compute_bbox_weighted_relevance_none_map(self):
        """Test bbox-weighted relevance with None relevance map."""
        score = self.scorer.compute_bbox_weighted_relevance(None, (0.5, 0.5, 0.5, 0.5), 4, 4)

        assert score == 0.0

    def test_compute_bbox_weighted_relevance_invalid_dimensions(self):
        """Test bbox-weighted relevance with invalid image dimensions."""
        relevance_map = np.ones((4, 4))
        bbox = (0.5, 0.5, 0.5, 0.5)

        score = self.scorer.compute_bbox_weighted_relevance(relevance_map, bbox, 0, 4)
        assert score == 0.0

        score = self.scorer.compute_bbox_weighted_relevance(relevance_map, bbox, 4, -1)
        assert score == 0.0

    def test_compute_bbox_weighted_relevance_invalid_bbox(self):
        """Test bbox-weighted relevance with invalid bbox coordinates."""
        relevance_map = np.ones((4, 4))

        # Bbox coordinates outside [0, 1] range
        score = self.scorer.compute_bbox_weighted_relevance(
            relevance_map, (1.5, 0.5, 0.5, 0.5), 4, 4
        )
        assert score == 0.0

        # Negative bbox dimensions
        score = self.scorer.compute_bbox_weighted_relevance(
            relevance_map, (0.5, 0.5, -0.1, 0.5), 4, 4
        )
        assert score == 0.0

    def test_compute_bbox_weighted_relevance_resize(self):
        """Test bbox-weighted relevance with relevance map resizing."""
        # Small relevance map that needs resizing
        relevance_map = np.array([[0.5, 0.8], [0.2, 0.9]])
        bbox = (0.5, 0.5, 0.5, 0.5)

        score = self.scorer.compute_bbox_weighted_relevance(relevance_map, bbox, 4, 4)

        # Should resize and compute score
        assert 0.0 <= score <= 1.0

    def test_compute_mean_bbox_weighted_relevance_basic(self):
        """Test mean bbox-weighted relevance computation."""
        relevance_maps = [
            np.array([[0.1, 0.8], [0.2, 0.9]]),
            np.array([[-0.1, 0.6], [0.3, -0.4]]),
            np.array([[0.5, -0.2], [-0.3, 0.7]]),
        ]
        bboxes = [(0.5, 0.5, 0.5, 0.5)] * 3
        image_sizes = [(2, 2)] * 3

        mean_score = self.scorer.compute_mean_bbox_weighted_relevance(
            relevance_maps, bboxes, image_sizes
        )

        assert 0.0 <= mean_score <= 1.0

    def test_compute_mean_bbox_weighted_relevance_empty_inputs(self):
        """Test mean bbox-weighted relevance with empty inputs."""
        mean_score = self.scorer.compute_mean_bbox_weighted_relevance([], [], [])
        assert mean_score == 0.0

    def test_compute_mean_bbox_weighted_relevance_mismatched_lengths(self):
        """Test mean bbox-weighted relevance with mismatched input lengths."""
        relevance_maps = [np.ones((2, 2))]
        bboxes = [(0.5, 0.5, 0.5, 0.5), (0.3, 0.3, 0.4, 0.4)]
        image_sizes = [(2, 2)]

        with pytest.raises(ValueError, match="must have same length"):
            self.scorer.compute_mean_bbox_weighted_relevance(relevance_maps, bboxes, image_sizes)

    def test_save_bbox_weighted_relevance_metrics(self):
        """Test saving bbox-weighted relevance metrics to JSON."""
        with tempfile.TemporaryDirectory() as temp_dir:
            round_num = 1
            mean_score = 0.75
            individual_scores = [0.8, 0.7, 0.9, 0.6]
            image_ids = ["img1.jpg", "img2.jpg", "img3.jpg", "img4.jpg"]

            self.scorer.save_bbox_weighted_relevance_metrics(
                round_num, mean_score, individual_scores, image_ids, temp_dir, "lrp"
            )

            # Check that file was created
            metrics_file = Path(temp_dir) / "bbox_weighted_relevance_metrics.json"
            assert metrics_file.exists()

            # Check file contents
            with open(metrics_file) as f:
                metrics = json.load(f)

            assert metrics["round"] == round_num
            assert metrics["method"] == "lrp"
            assert metrics["mean_bbox_weighted_relevance"] == mean_score
            assert metrics["num_images"] == len(individual_scores)
            assert len(metrics["individual_scores"]) == len(image_ids)
            assert "statistics" in metrics
            assert metrics["statistics"]["min"] == 0.6
            assert metrics["statistics"]["max"] == 0.9

    def test_bbox_weighted_relevance_bounds_validation(self):
        """Test that bbox-weighted relevance scores are properly bounded between 0 and 1."""
        # Create a relevance map with very high absolute values
        relevance_map = np.ones((4, 4)) * 1000
        relevance_map[0, 0] = -2000  # Add some negative values
        bbox = (0.5, 0.5, 0.5, 0.5)

        score = self.scorer.compute_bbox_weighted_relevance(relevance_map, bbox, 4, 4)

        # Score should still be bounded
        assert 0.0 <= score <= 1.0
