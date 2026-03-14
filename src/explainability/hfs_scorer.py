"""Heatmap Focus Score (HFS) computation for explainability analysis."""

import json
import logging
from pathlib import Path
from typing import Any

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
        image_height: int
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
        # Convert YOLO format bbox to pixel coordinates
        x_center, y_center, w, h = bbox

        # Convert to absolute coordinates
        x_center_abs = x_center * image_width
        y_center_abs = y_center * image_height
        w_abs = w * image_width
        h_abs = h * image_height

        # Calculate bbox corners
        x1 = int(max(0, x_center_abs - w_abs / 2))
        y1 = int(max(0, y_center_abs - h_abs / 2))
        x2 = int(min(image_width, x_center_abs + w_abs / 2))
        y2 = int(min(image_height, y_center_abs + h_abs / 2))

        # Ensure heatmap matches image dimensions
        if heatmap.shape[0] != image_height or heatmap.shape[1] != image_width:
            # Resize heatmap to match image dimensions
            from scipy.ndimage import zoom
            scale_y = image_height / heatmap.shape[0]
            scale_x = image_width / heatmap.shape[1]
            heatmap = zoom(heatmap, (scale_y, scale_x), order=1)

        # Compute total heatmap intensity
        total_intensity = np.sum(heatmap)

        if total_intensity == 0:
            self.logger.warning("Total heatmap intensity is zero")
            return 0.0

        # Compute intensity inside bbox
        bbox_intensity = np.sum(heatmap[y1:y2, x1:x2])

        # Compute HFS
        hfs = bbox_intensity / total_intensity

        return float(hfs)

    def compute_mean_hfs(
        self,
        heatmaps: list[np.ndarray],
        bboxes: list[tuple[float, float, float, float]],
        image_sizes: list[tuple[int, int]]
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
        if len(heatmaps) != len(bboxes) or len(heatmaps) != len(image_sizes):
            raise ValueError("Heatmaps, bboxes, and image_sizes must have same length")

        if len(heatmaps) == 0:
            self.logger.warning("No heatmaps provided")
            return 0.0

        hfs_scores = []

        for heatmap, bbox, (width, height) in zip(heatmaps, bboxes, image_sizes):
            hfs = self.compute_hfs(heatmap, bbox, width, height)
            hfs_scores.append(hfs)

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
        method: str = "gradcam"
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
            "individual_hfs": {
                image_id: hfs
                for image_id, hfs in zip(image_ids, individual_hfs)
            },
            "statistics": {
                "min": float(np.min(individual_hfs)) if individual_hfs else 0.0,
                "max": float(np.max(individual_hfs)) if individual_hfs else 0.0,
                "std": float(np.std(individual_hfs)) if individual_hfs else 0.0,
                "median": float(np.median(individual_hfs)) if individual_hfs else 0.0,
            }
        }

        metrics_file = output_path / f"hfs_metrics.json"

        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)

        self.logger.info(f"HFS metrics saved to {metrics_file}")

    def compute_hfs_for_occlusion(
        self,
        occlusion_heatmap: np.ndarray,
        bbox: tuple[float, float, float, float],
        image_width: int,
        image_height: int
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

    def load_ground_truth_bboxes(
        self,
        image_paths: list[str],
        base_path: Path
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
            _path_str = str(image_path).replace('\\images\\', '\\labels\\').replace('/images/', '/labels/')
            label_path = Path(_path_str).with_suffix('.txt')

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
