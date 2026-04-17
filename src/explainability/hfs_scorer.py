"""Heatmap Focus Score (HFS) computation for explainability analysis."""

from collections.abc import Sequence
import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class Heatmap_Focus_Scorer:
    """Computes Heatmap Focus Score (HFS) for explainability heatmaps."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def compute_hfs(
        self,
        heatmap: np.ndarray,
        bbox: tuple[float, float, float, float],
        image_width: int,
        image_height: int,
    ) -> float:
        """
        Compute HFS for a single image.

        HFS = sum of heatmap intensity inside ground-truth bbox / total heatmap intensity

        Args:
            heatmap: 2D numpy array of heatmap values (normalized 0-1)
            bbox: Ground-truth bounding box in YOLO format (x_center, y_center, width, height)
                  All values are normalized (0-1)
            image_width: Original image width in pixels
            image_height: Original image height in pixels

        Returns:
            HFS score (0-1), where higher values indicate better focus on the object
        """
        # Validate inputs
        if heatmap is None or heatmap.size == 0:
            self.logger.warning("Empty or None heatmap provided")
            return 0.0

        if not isinstance(heatmap, np.ndarray) or heatmap.ndim != 2:
            self.logger.warning("Heatmap must be a 2D numpy array")
            return 0.0

        if image_width <= 0 or image_height <= 0:
            self.logger.warning(f"Invalid image dimensions: {image_width}x{image_height}")
            return 0.0

        # Validate bbox coordinates
        x_center, y_center, w, h = bbox
        if not all(0 <= coord <= 1 for coord in bbox):
            self.logger.warning(f"Invalid bbox coordinates (must be 0-1): {bbox}")
            return 0.0

        if w <= 0 or h <= 0:
            self.logger.warning(f"Invalid bbox dimensions: width={w}, height={h}")
            return 0.0

        # Convert to absolute coordinates
        x_center_abs = x_center * image_width
        y_center_abs = y_center * image_height
        w_abs = w * image_width
        h_abs = h * image_height

        # Calculate bbox corners with proper bounds checking
        x1 = int(max(0, x_center_abs - w_abs / 2))
        y1 = int(max(0, y_center_abs - h_abs / 2))
        x2 = int(min(image_width, x_center_abs + w_abs / 2))
        y2 = int(min(image_height, y_center_abs + h_abs / 2))

        # Handle edge case where bbox is completely outside image bounds
        if x1 >= x2 or y1 >= y2:
            self.logger.warning(f"Bounding box outside image bounds: ({x1},{y1}) to ({x2},{y2})")
            return 0.0

        # Ensure heatmap matches image dimensions
        if heatmap.shape[0] != image_height or heatmap.shape[1] != image_width:
            # Resize heatmap to match image dimensions
            try:
                from scipy.ndimage import zoom

                scale_y = image_height / heatmap.shape[0]
                scale_x = image_width / heatmap.shape[1]
                heatmap = zoom(heatmap, (scale_y, scale_x), order=1)
            except Exception as e:
                self.logger.error(f"Failed to resize heatmap: {e}")
                return 0.0

        # Compute total heatmap intensity
        total_intensity = np.sum(heatmap)

        if total_intensity == 0:
            self.logger.warning("Total heatmap intensity is zero")
            return 0.0

        # Ensure bbox coordinates are within heatmap bounds after resizing
        x2 = min(x2, heatmap.shape[1])
        y2 = min(y2, heatmap.shape[0])

        # Compute intensity inside bbox
        try:
            bbox_intensity = np.sum(heatmap[y1:y2, x1:x2])
        except IndexError as e:
            self.logger.error(f"Index error when computing bbox intensity: {e}")
            return 0.0

        # Compute HFS
        hfs = bbox_intensity / total_intensity

        # Ensure HFS is within valid range [0, 1]
        hfs = max(0.0, min(1.0, hfs))

        return float(hfs)

    def compute_mean_hfs(
        self,
        heatmaps: Sequence[np.ndarray | None],
        bboxes: list[tuple[float, float, float, float]],
        image_sizes: list[tuple[int, int]],
    ) -> float:
        """
        Compute mean HFS across multiple images.

        Args:
            heatmaps: List of 2D numpy arrays (one per image)
            bboxes: List of ground-truth bounding boxes in YOLO format
            image_sizes: List of (width, height) tuples for each image

        Returns:
            Mean HFS across all images
        """
        # Validate inputs
        if not heatmaps or not bboxes or not image_sizes:
            self.logger.warning("Empty input lists provided")
            return 0.0

        if len(heatmaps) != len(bboxes) or len(heatmaps) != len(image_sizes):
            raise ValueError("Heatmaps, bboxes, and image_sizes must have same length")

        hfs_scores = []

        for i, (heatmap, bbox, (width, height)) in enumerate(
            zip(heatmaps, bboxes, image_sizes, strict=False)
        ):
            try:
                # Skip None or invalid heatmaps
                if heatmap is None:
                    continue
                hfs = self.compute_hfs(heatmap, bbox, width, height)
                # Only add valid HFS scores (> 0 or when we have valid computation)
                if hfs >= 0:  # HFS can be 0.0 legitimately
                    hfs_scores.append(hfs)
            except Exception as e:
                self.logger.warning(f"Failed to compute HFS for image {i}: {e}")
                # Continue with other images, don't fail the entire batch

        if not hfs_scores:
            self.logger.warning("No valid HFS scores computed")
            return 0.0

        mean_hfs = float(np.mean(hfs_scores))

        self.logger.info(f"Computed mean HFS: {mean_hfs:.4f} across {len(hfs_scores)} images")

        return mean_hfs

    def save_hfs_metrics(
        self,
        round_num: int,
        mean_hfs: float,
        individual_hfs: list[float],
        image_ids: list[str],
        output_dir: str,
        method: str = "gradcam",
    ) -> None:
        """
        Save HFS metrics to JSON file.

        Args:
            round_num: Round number (1-5)
            mean_hfs: Mean HFS across all images
            individual_hfs: List of HFS scores for each image
            image_ids: List of image identifiers
            output_dir: Directory to save metrics
            method: Explainability method name (gradcam, occlusion, etc.)
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        metrics = {
            "round": round_num,
            "method": method,
            "mean_hfs": mean_hfs,
            "num_images": len(individual_hfs),
            "individual_hfs": dict(zip(image_ids, individual_hfs, strict=False)),
            "statistics": {
                "min": float(np.min(individual_hfs)) if individual_hfs else 0.0,
                "max": float(np.max(individual_hfs)) if individual_hfs else 0.0,
                "std": float(np.std(individual_hfs)) if individual_hfs else 0.0,
                "median": float(np.median(individual_hfs)) if individual_hfs else 0.0,
            },
        }

        metrics_file = output_path / "hfs_metrics.json"

        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)

        self.logger.info(f"HFS metrics saved to {metrics_file}")

    def compute_hfs_for_occlusion(
        self,
        occlusion_heatmap: np.ndarray,
        bbox: tuple[float, float, float, float],
        image_width: int,
        image_height: int,
    ) -> float:
        """
        Compute HFS for occlusion-based heatmaps.

        Occlusion heatmaps show confidence drop when regions are occluded.
        Higher values indicate more important regions.

        Args:
            occlusion_heatmap: 2D numpy array of occlusion sensitivity values
            bbox: Ground-truth bounding box in YOLO format
            image_width: Original image width in pixels
            image_height: Original image height in pixels

        Returns:
            HFS score for occlusion heatmap
        """
        # Same computation as regular HFS
        return self.compute_hfs(occlusion_heatmap, bbox, image_width, image_height)

    def compute_bbox_weighted_relevance(
        self,
        relevance_map: np.ndarray,
        bbox: tuple[float, float, float, float],
        image_width: int,
        image_height: int,
    ) -> float:
        """
        Compute bbox-weighted relevance concentration for LRP attribution maps.

        This computes the sum of absolute relevance values inside the ground-truth
        bounding box divided by the total absolute relevance. Unlike regular HFS
        which works with positive activation values, LRP generates relevance maps
        with both positive and negative values, requiring absolute value computation.

        Args:
            relevance_map: 2D numpy array of LRP relevance values (can be positive/negative)
            bbox: Ground-truth bounding box in YOLO format (x_center, y_center, width, height)
                  All values are normalized (0-1)
            image_width: Original image width in pixels
            image_height: Original image height in pixels

        Returns:
            Bbox-weighted relevance score (0-1), where higher values indicate better
            relevance concentration on the object
        """
        # Validate inputs
        if relevance_map is None or relevance_map.size == 0:
            self.logger.warning("Empty or None relevance map provided")
            return 0.0

        if not isinstance(relevance_map, np.ndarray) or relevance_map.ndim != 2:
            self.logger.warning("Relevance map must be a 2D numpy array")
            return 0.0

        if image_width <= 0 or image_height <= 0:
            self.logger.warning(f"Invalid image dimensions: {image_width}x{image_height}")
            return 0.0

        # Validate bbox coordinates
        x_center, y_center, w, h = bbox
        if not all(0 <= coord <= 1 for coord in bbox):
            self.logger.warning(f"Invalid bbox coordinates (must be 0-1): {bbox}")
            return 0.0

        if w <= 0 or h <= 0:
            self.logger.warning(f"Invalid bbox dimensions: width={w}, height={h}")
            return 0.0

        # Convert to absolute coordinates
        x_center_abs = x_center * image_width
        y_center_abs = y_center * image_height
        w_abs = w * image_width
        h_abs = h * image_height

        # Calculate bbox corners with proper bounds checking
        x1 = int(max(0, x_center_abs - w_abs / 2))
        y1 = int(max(0, y_center_abs - h_abs / 2))
        x2 = int(min(image_width, x_center_abs + w_abs / 2))
        y2 = int(min(image_height, y_center_abs + h_abs / 2))

        # Handle edge case where bbox is completely outside image bounds
        if x1 >= x2 or y1 >= y2:
            self.logger.warning(f"Bounding box outside image bounds: ({x1},{y1}) to ({x2},{y2})")
            return 0.0

        # Ensure relevance map matches image dimensions
        if relevance_map.shape[0] != image_height or relevance_map.shape[1] != image_width:
            # Resize relevance map to match image dimensions
            try:
                from scipy.ndimage import zoom

                scale_y = image_height / relevance_map.shape[0]
                scale_x = image_width / relevance_map.shape[1]
                relevance_map = zoom(relevance_map, (scale_y, scale_x), order=1)
            except Exception as e:
                self.logger.error(f"Failed to resize relevance map: {e}")
                return 0.0

        # Compute total absolute relevance (key difference from regular HFS)
        total_relevance = np.sum(np.abs(relevance_map))

        if total_relevance == 0:
            self.logger.warning("Total absolute relevance is zero")
            return 0.0

        # Ensure bbox coordinates are within relevance map bounds after resizing
        x2 = min(x2, relevance_map.shape[1])
        y2 = min(y2, relevance_map.shape[0])

        # Compute absolute relevance inside bbox
        try:
            bbox_relevance = np.sum(np.abs(relevance_map[y1:y2, x1:x2]))
        except IndexError as e:
            self.logger.error(f"Index error when computing bbox relevance: {e}")
            return 0.0

        # Compute bbox-weighted relevance concentration
        relevance_score = bbox_relevance / total_relevance

        # Ensure score is within valid range [0, 1]
        relevance_score = max(0.0, min(1.0, relevance_score))

        return float(relevance_score)

    def compute_mean_bbox_weighted_relevance(
        self,
        relevance_maps: Sequence[np.ndarray | None],
        bboxes: list[tuple[float, float, float, float]],
        image_sizes: list[tuple[int, int]],
    ) -> float:
        """
        Compute mean bbox-weighted relevance concentration across multiple images.

        Args:
            relevance_maps: List of 2D numpy arrays (one per image)
            bboxes: List of ground-truth bounding boxes in YOLO format
            image_sizes: List of (width, height) tuples for each image

        Returns:
            Mean bbox-weighted relevance score across all images
        """
        # Validate inputs
        if not relevance_maps or not bboxes or not image_sizes:
            self.logger.warning("Empty input lists provided")
            return 0.0

        if len(relevance_maps) != len(bboxes) or len(relevance_maps) != len(image_sizes):
            raise ValueError("Relevance maps, bboxes, and image_sizes must have same length")

        relevance_scores = []

        for i, (relevance_map, bbox, (width, height)) in enumerate(
            zip(relevance_maps, bboxes, image_sizes, strict=False)
        ):
            try:
                # Skip None or invalid relevance maps
                if relevance_map is None:
                    continue
                score = self.compute_bbox_weighted_relevance(relevance_map, bbox, width, height)
                # Only add valid scores (>= 0)
                if score >= 0:
                    relevance_scores.append(score)
            except Exception as e:
                self.logger.warning(f"Failed to compute bbox-weighted relevance for image {i}: {e}")
                # Continue with other images, don't fail the entire batch

        if not relevance_scores:
            self.logger.warning("No valid bbox-weighted relevance scores computed")
            return 0.0

        mean_score = float(np.mean(relevance_scores))

        self.logger.info(
            f"Computed mean bbox-weighted relevance: {mean_score:.4f} across {len(relevance_scores)} images"
        )

        return mean_score

    def save_bbox_weighted_relevance_metrics(
        self,
        round_num: int,
        mean_relevance_score: float,
        individual_scores: list[float],
        image_ids: list[str],
        output_dir: str,
        method: str = "lrp",
    ) -> None:
        """
        Save bbox-weighted relevance metrics to JSON file.

        Args:
            round_num: Round number (1-5)
            mean_relevance_score: Mean bbox-weighted relevance score across all images
            individual_scores: List of relevance scores for each image
            image_ids: List of image identifiers
            output_dir: Directory to save metrics
            method: Explainability method name (lrp, etc.)
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        metrics = {
            "round": round_num,
            "method": method,
            "mean_bbox_weighted_relevance": mean_relevance_score,
            "num_images": len(individual_scores),
            "individual_scores": dict(zip(image_ids, individual_scores, strict=False)),
            "statistics": {
                "min": float(np.min(individual_scores)) if individual_scores else 0.0,
                "max": float(np.max(individual_scores)) if individual_scores else 0.0,
                "std": float(np.std(individual_scores)) if individual_scores else 0.0,
                "median": float(np.median(individual_scores)) if individual_scores else 0.0,
            },
        }

        metrics_file = output_path / "bbox_weighted_relevance_metrics.json"

        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)

        self.logger.info(f"Bbox-weighted relevance metrics saved to {metrics_file}")

    def load_ground_truth_bboxes(
        self, image_paths: list[str], base_path: Path
    ) -> list[tuple[float, float, float, float]]:
        """
        Load ground-truth bounding boxes for images.

        Args:
            image_paths: List of image file paths
            base_path: Base path for dataset

        Returns:
            List of bounding boxes in YOLO format (x_center, y_center, width, height)
        """
        bboxes = []

        for image_path in image_paths:
            # Get corresponding label file (YOLO stores labels under labels/, not images/)
            # Handle both Unix (/) and Windows (\) path separators
            _path_str = (
                str(image_path).replace("\\images\\", "\\labels\\").replace("/images/", "/labels/")
            )
            label_path = Path(_path_str).with_suffix(".txt")

            if not label_path.exists():
                self.logger.warning(f"Label file not found: {label_path}")
                # Use default bbox (center of image)
                bboxes.append((0.5, 0.5, 0.5, 0.5))
                continue

            # Read first annotation (assume single object per image for HFS)
            with open(label_path) as f:
                lines = f.readlines()

            if not lines:
                self.logger.warning(f"Empty label file: {label_path}")
                bboxes.append((0.5, 0.5, 0.5, 0.5))
                continue

            # Parse first line (class_id x_center y_center width height)
            parts = lines[0].strip().split()
            if len(parts) < 5:
                self.logger.warning(f"Invalid annotation format: {label_path}")
                bboxes.append((0.5, 0.5, 0.5, 0.5))
                continue

            # Extract bbox (skip class_id)
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])

            bboxes.append((x_center, y_center, width, height))

        return bboxes
