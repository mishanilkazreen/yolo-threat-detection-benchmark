"""Unit tests for Heatmap_Focus_Scorer.compute_hfs."""

from __future__ import annotations

import numpy as np
import pytest

from src.explainability.hfs_scorer import Heatmap_Focus_Scorer


@pytest.fixture
def scorer() -> Heatmap_Focus_Scorer:
    return Heatmap_Focus_Scorer()


class TestAllZeroCam:
    """Test all-zero activation map handling."""

    def test_all_zero_returns_zero(self, scorer: Heatmap_Focus_Scorer) -> None:
        """Returns 0.0 when cam_np is all-zero."""
        cam_np = np.zeros((64, 64), dtype=np.float32)
        bbox = (0.5, 0.5, 0.4, 0.4)
        result = scorer.compute_hfs(cam_np, bbox, 64, 64)
        assert result == 0.0

    def test_all_zero_no_exception(self, scorer: Heatmap_Focus_Scorer) -> None:
        """No ZeroDivisionError when cam_np is all-zero."""
        cam_np = np.zeros((32, 32), dtype=np.float32)
        bbox = (0.5, 0.5, 0.5, 0.5)
        try:
            scorer.compute_hfs(cam_np, bbox, 32, 32)
        except ZeroDivisionError as exc:
            pytest.fail(f"ZeroDivisionError raised for all-zero cam_np: {exc}")

    def test_all_zero_returns_float(self, scorer: Heatmap_Focus_Scorer) -> None:
        """Return type is float even for all-zero input."""
        cam_np = np.zeros((10, 10), dtype=np.float32)
        bbox = (0.5, 0.5, 0.3, 0.3)
        result = scorer.compute_hfs(cam_np, bbox, 10, 10)
        assert isinstance(result, float)


class TestReturnRange:
    """Test HFS return value range."""

    def test_uniform_heatmap_full_bbox_is_one(self, scorer: Heatmap_Focus_Scorer) -> None:
        """Uniform heatmap with bbox covering the whole image → HFS == 1.0."""
        cam_np = np.ones((8, 8), dtype=np.float32)
        bbox = (0.5, 0.5, 1.0, 1.0)  # full image
        result = scorer.compute_hfs(cam_np, bbox, 8, 8)
        assert abs(result - 1.0) < 1e-6

    def test_activation_outside_bbox_is_zero(self, scorer: Heatmap_Focus_Scorer) -> None:
        """All activation outside bbox → HFS == 0.0."""
        cam_np = np.zeros((8, 8), dtype=np.float32)
        cam_np[0, 0] = 1.0  # top-left corner only
        # bbox covers bottom-right quadrant
        bbox = (0.75, 0.75, 0.5, 0.5)
        result = scorer.compute_hfs(cam_np, bbox, 8, 8)
        assert result == 0.0

    def test_partial_overlap_in_range(self, scorer: Heatmap_Focus_Scorer) -> None:
        """Partial overlap → HFS strictly between 0 and 1."""
        cam_np = np.zeros((10, 10), dtype=np.float32)
        cam_np[2:5, 2:5] = 1.0  # activation in top-left area
        cam_np[7:10, 7:10] = 1.0  # activation in bottom-right area
        # bbox covers only the bottom-right activation
        bbox = (0.85, 0.85, 0.3, 0.3)
        result = scorer.compute_hfs(cam_np, bbox, 10, 10)
        assert 0.0 < result < 1.0

    def test_result_always_in_unit_interval(self, scorer: Heatmap_Focus_Scorer) -> None:
        """HFS is always in [0, 1] for a variety of random inputs."""
        rng = np.random.default_rng(42)
        for _ in range(20):
            h, w = rng.integers(4, 32, size=2)
            cam_np = rng.random((h, w)).astype(np.float32)
            cx = rng.uniform(0.1, 0.9)
            cy = rng.uniform(0.1, 0.9)
            bw = rng.uniform(0.1, 0.8)
            bh = rng.uniform(0.1, 0.8)
            result = scorer.compute_hfs(cam_np, (cx, cy, bw, bh), int(w), int(h))
            assert 0.0 <= result <= 1.0, (
                f"HFS {result} out of [0,1] for cam shape ({h},{w}), "
                f"bbox ({cx:.2f},{cy:.2f},{bw:.2f},{bh:.2f})"
            )


class TestBboxPositions:
    """Test HFS computation for different bbox placements."""

    def test_bbox_fully_inside_image(self, scorer: Heatmap_Focus_Scorer) -> None:
        """Bbox fully inside image — standard case."""
        # 10x10 heatmap; activate only the centre 4x4 block (rows 3-6, cols 3-6)
        cam_np = np.zeros((10, 10), dtype=np.float32)
        cam_np[3:7, 3:7] = 1.0
        inside_sum = cam_np[3:7, 3:7].sum()  # 16.0
        total_sum = cam_np.sum()  # 16.0

        # bbox centred at (0.5, 0.5) with 40% width/height
        bbox = (0.5, 0.5, 0.4, 0.4)
        result = scorer.compute_hfs(cam_np, bbox, 10, 10)

        expected = inside_sum / total_sum  # 1.0
        assert abs(result - expected) < 1e-4

    def test_bbox_at_image_boundary_top_left(self, scorer: Heatmap_Focus_Scorer) -> None:
        """Bbox centred at the top-left corner — partial overlap with image."""
        cam_np = np.ones((10, 10), dtype=np.float32)
        # bbox centred at (0, 0) with 50% width/height → only bottom-right
        # quarter of the bbox is inside the image
        bbox = (0.0, 0.0, 0.5, 0.5)
        result = scorer.compute_hfs(cam_np, bbox, 10, 10)
        # The clamped region is rows 0:2, cols 0:2 (approx) — result must be in [0,1]
        assert 0.0 <= result <= 1.0

    def test_bbox_at_image_boundary_bottom_right(self, scorer: Heatmap_Focus_Scorer) -> None:
        """Bbox centred at the bottom-right corner — partial overlap with image."""
        cam_np = np.ones((10, 10), dtype=np.float32)
        bbox = (1.0, 1.0, 0.5, 0.5)
        result = scorer.compute_hfs(cam_np, bbox, 10, 10)
        assert 0.0 <= result <= 1.0

    def test_bbox_larger_than_image_clamped(self, scorer: Heatmap_Focus_Scorer) -> None:
        """
        Bbox larger than the image (w=h=1.0, centred at 0.5) covers the entire
        image after clamping — HFS should equal 1.0.
        """
        cam_np = np.ones((8, 8), dtype=np.float32)
        # bbox that spans the full image
        bbox = (0.5, 0.5, 1.0, 1.0)
        result = scorer.compute_hfs(cam_np, bbox, 8, 8)
        assert abs(result - 1.0) < 1e-6

    def test_bbox_wider_than_image_clamped_to_image_width(
        self, scorer: Heatmap_Focus_Scorer
    ) -> None:
        """Bbox wider than image is clamped to image width; result still in [0,1]."""
        cam_np = np.ones((6, 6), dtype=np.float32)
        # width = 2.0 (200% of image width) — should be clamped
        # Note: bbox coords must be in [0,1] per validation; use max valid width
        bbox = (0.5, 0.5, 1.0, 0.5)
        result = scorer.compute_hfs(cam_np, bbox, 6, 6)
        assert 0.0 <= result <= 1.0

    def test_bbox_taller_than_image_clamped_to_image_height(
        self, scorer: Heatmap_Focus_Scorer
    ) -> None:
        """Bbox taller than image is clamped to image height; result still in [0,1]."""
        cam_np = np.ones((6, 6), dtype=np.float32)
        bbox = (0.5, 0.5, 0.5, 1.0)
        result = scorer.compute_hfs(cam_np, bbox, 6, 6)
        assert 0.0 <= result <= 1.0
