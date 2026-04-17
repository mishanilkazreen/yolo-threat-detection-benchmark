"""Output manager for XAI attribution maps and visualizations."""

import logging
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .interfaces import AttributionMethod

logger = logging.getLogger(__name__)


class XAIOutputManager:
    """Manages output generation for XAI attribution maps."""

    def __init__(
        self,
        base_output_dir: str,
        save_overlays: bool = True,
        save_raw: bool = False,
        round_num: int = 1,
    ):
        """
        Initialize output manager.

        Args:
            base_output_dir: Base directory for XAI outputs
            save_overlays: Whether to save overlaid visualizations
            save_raw: Whether to save raw attribution arrays
            round_num: Training round number — used to create per-round subdirectories
                       so outputs from different rounds never overwrite each other
        """
        self.base_output_dir = Path(base_output_dir)
        self.save_overlays = save_overlays
        self.save_raw = save_raw
        self.round_num = round_num
        self._setup_directories()

    def _setup_directories(self):
        """Create per-method output directories."""
        for method in AttributionMethod:
            method_dir = self.base_output_dir / method.value
            method_dir.mkdir(parents=True, exist_ok=True)

            if self.save_raw:
                raw_dir = method_dir / "raw"
                raw_dir.mkdir(parents=True, exist_ok=True)

    def save_attribution_outputs(
        self,
        attribution_map: np.ndarray,
        original_image: np.ndarray,
        image_path: str,
        method: AttributionMethod,
        detections: dict | None = None,
        model_names: dict[int, str] | None = None,
    ) -> str:
        """
        Save attribution outputs (overlays and/or raw data).

        Args:
            attribution_map: Attribution map as numpy array
            original_image: Original image as numpy array
            image_path: Path to original image file
            method: XAI method used to generate attribution
            detections: Optional YOLO detections dict with boxes, classes, scores
            model_names: Optional mapping of class IDs to class names

        Returns:
            Path to saved overlay image (empty string if overlays disabled)
        """
        image_name = Path(image_path).stem
        method_dir = self.base_output_dir / method.value

        output_path = ""

        # Save raw attribution if enabled
        if self.save_raw:
            raw_path = method_dir / "raw" / f"{image_name}_{method.value}_raw.npy"
            np.save(raw_path, attribution_map)
            logger.debug(f"Saved raw attribution to {raw_path}")

        # Save overlay visualization if enabled
        if self.save_overlays:
            output_path = str(method_dir / f"{image_name}_{method.value}.png")
            self._save_overlay_visualization(
                original_image, attribution_map, output_path, method, detections, model_names
            )
            logger.debug(f"Saved overlay visualization to {output_path}")

        return output_path

    def _save_overlay_visualization(
        self,
        image_np: np.ndarray,
        attribution_np: np.ndarray,
        output_path: str,
        method: AttributionMethod,
        detections: dict | None = None,
        model_names: dict[int, str] | None = None,
    ):
        """Save attribution visualization as overlay on original image."""
        if method == AttributionMethod.GRADCAM:
            self._save_gradcam_overlay(
                image_np, attribution_np, output_path, detections, model_names
            )
        elif method == AttributionMethod.LRP:
            self._save_lrp_overlay(image_np, attribution_np, output_path)
        elif method == AttributionMethod.SHAP:
            self._save_shap_overlay(image_np, attribution_np, output_path)

    def _save_gradcam_overlay(
        self,
        image_np: np.ndarray,
        cam_np: np.ndarray,
        output_path: str,
        detections: dict | None = None,
        model_names: dict[int, str] | None = None,
    ):
        """Save Grad-CAM as 3-panel matplotlib figure."""
        from .gradcam import _save_gradcam_visualization, show_cam_on_image

        out = Path(output_path)
        stem = out.stem
        if stem.endswith("_gradcam"):
            stem = stem[:-8]
        synthetic_image_path = str(out.parent / stem)

        # Build the overlay image here so _save_gradcam_visualization gets all three inputs
        cam_norm = cam_np.astype(np.float32)
        h, w = image_np.shape[:2]
        import cv2 as _cv2

        if cam_norm.shape != (h, w):
            cam_norm = _cv2.resize(cam_norm, (w, h), interpolation=_cv2.INTER_LINEAR).astype(
                np.float32
            )
        cam_max = cam_norm.max()
        if cam_max > 0:
            cam_norm = cam_norm / cam_max
        img_float = image_np.astype(np.float32) / 255.0
        cam_image = show_cam_on_image(img_float, cam_norm, use_rgb=True)

        saved = _save_gradcam_visualization(
            image_np,
            cam_norm,
            cam_image,
            synthetic_image_path,
            str(out.parent),
            detections=detections,
            model_names=model_names,
        )

        saved_path = Path(saved)
        if saved_path != out:
            saved_path.rename(out)

    def _save_lrp_overlay(self, image_np: np.ndarray, relevance_np: np.ndarray, output_path: str):
        """Save LRP visualization using the matplotlib 3-panel heatmap figure."""
        from .lrp import _save_lrp_visualization

        out = Path(output_path)
        # _save_lrp_visualization saves as {stem}_lrp.png; reconstruct a synthetic
        # image path whose stem matches what the caller expects (strip trailing _lrp if
        # already present so we don't double-suffix).
        stem = out.stem
        if stem.endswith("_lrp"):
            stem = stem[:-4]
        synthetic_image_path = str(out.parent / stem)

        saved = _save_lrp_visualization(
            image_np, relevance_np, synthetic_image_path, str(out.parent)
        )

        # If the caller expected a different path, rename to match
        saved_path = Path(saved)
        if saved_path != out:
            saved_path.rename(out)

    def _save_shap_overlay(
        self, image_np: np.ndarray, attribution_np: np.ndarray, output_path: str
    ):
        """Save SHAP visualization as a 4-panel matplotlib figure."""
        from .shap import _save_shap_visualization

        out = Path(output_path)
        stem = out.stem
        if stem.endswith("_shap"):
            stem = stem[:-5]
        synthetic_image_path = str(out.parent / stem)

        saved = _save_shap_visualization(
            image_np, attribution_np, synthetic_image_path, str(out.parent)
        )

        saved_path = Path(saved)
        if saved_path != out:
            saved_path.rename(out)

    def _save_default_overlay(
        self, image_np: np.ndarray, attribution_np: np.ndarray, output_path: str
    ):
        """Save default visualization for unknown methods."""
        # Normalize attribution
        attr_norm = (attribution_np - attribution_np.min()) / (
            attribution_np.max() - attribution_np.min() + 1e-8
        )

        # Create heatmap
        attr_uint8 = (255 * attr_norm).astype(np.uint8)
        heatmap = cv2.applyColorMap(attr_uint8, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        # Overlay on original image
        overlay = 0.6 * image_np + 0.4 * heatmap
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)

        # Save visualization
        overlay_image = Image.fromarray(overlay)
        overlay_image.save(output_path)

    def get_method_output_dir(self, method: AttributionMethod) -> Path:
        """Get output directory for a specific method."""
        return self.base_output_dir / method.value

    def cleanup_old_outputs(self, keep_latest: int = 10):
        """Clean up old output files, keeping only the latest N files per method."""
        for method in AttributionMethod:
            method_dir = self.get_method_output_dir(method)
            if not method_dir.exists():
                continue

            # Get all PNG files sorted by modification time
            png_files = sorted(
                method_dir.glob("*.png"), key=lambda p: p.stat().st_mtime, reverse=True
            )

            # Remove old files
            for old_file in png_files[keep_latest:]:
                try:
                    old_file.unlink()
                    logger.debug(f"Removed old output file: {old_file}")
                except Exception as e:
                    logger.warning(f"Failed to remove {old_file}: {e}")

    def save_hfs_metrics_csv(
        self,
        results: list,  # List of XAIResult objects
        round_num: int,
        config_name: str,
        output_dir: str | None = None,
        model_name: str = "",
    ) -> str:
        """
        Save individual HFS scores to CSV file.

        Args:
            results: List of XAIResult objects containing HFS scores
            round_num: Training round number
            config_name: Configuration name for file naming
            output_dir: Optional output directory (uses base_output_dir if None)

        Returns:
            Path to saved CSV file
        """
        import csv
        from pathlib import Path

        if output_dir is None:
            output_dir_path: Path = self.base_output_dir
        else:
            output_dir_path = Path(output_dir)

        csv_file = output_dir_path / f"hfs_individual_{config_name}_round_{round_num}.csv"

        # Ensure output directory exists
        csv_file.parent.mkdir(parents=True, exist_ok=True)

        # Define CSV columns
        columns = [
            "round",
            "image_id",
            "method",
            "hfs_score",
            "processing_time_seconds",
            "output_path",
            "model_name",
        ]

        with open(csv_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()

            for result in results:
                if result.hfs_score is not None:  # Only log results with HFS scores
                    row = {
                        "round": round_num,
                        "image_id": result.image_id,
                        "method": result.method.value,
                        "hfs_score": result.hfs_score,
                        "processing_time_seconds": result.processing_time,
                        "output_path": result.output_path,
                        "model_name": model_name,
                    }
                    writer.writerow(row)

        logger.info(f"Saved individual HFS metrics to CSV: {csv_file}")
        return str(csv_file)

    def save_aggregate_hfs_csv(
        self,
        aggregate_data: list[dict],  # List of round aggregation data
        config_name: str,
        output_dir: str | None = None,
    ) -> str:
        """
        Save aggregate HFS scores across rounds to CSV file.

        Args:
            aggregate_data: List of dictionaries containing round aggregation data
            config_name: Configuration name for file naming
            output_dir: Optional output directory (uses base_output_dir if None)

        Returns:
            Path to saved CSV file
        """
        import csv
        from pathlib import Path

        if output_dir is None:
            output_dir_path: Path = self.base_output_dir
        else:
            output_dir_path = Path(output_dir)

        csv_file = output_dir_path / f"hfs_aggregate_{config_name}.csv"

        # Ensure output directory exists
        csv_file.parent.mkdir(parents=True, exist_ok=True)

        if not aggregate_data:
            logger.warning("No aggregate data to save")
            # Still create an empty file with headers
            columns = ["round", "num_images_processed", "total_processing_time_seconds"]
            with open(csv_file, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=columns)
                writer.writeheader()
            return str(csv_file)

        # Determine all methods present in the data
        all_methods = set()
        for round_data in aggregate_data:
            if "aggregate_hfs" in round_data:
                all_methods.update(round_data["aggregate_hfs"].keys())

        # Define CSV columns
        columns = ["round", "num_images_processed", "total_processing_time_seconds"]
        for method in sorted(all_methods):
            columns.extend(
                [
                    f"{method}_mean_hfs",
                    f"{method}_num_images",
                ]
            )

        with open(csv_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()

            for round_data in aggregate_data:
                row = {
                    "round": round_data.get("round_num", 0),
                    "num_images_processed": round_data.get("num_images_processed", 0),
                    "total_processing_time_seconds": round_data.get("total_processing_time", 0.0),
                }

                # Add method-specific metrics
                aggregate_hfs = round_data.get("aggregate_hfs", {})
                method_counts = round_data.get("method_counts", {})

                for method in sorted(all_methods):
                    row[f"{method}_mean_hfs"] = aggregate_hfs.get(method, 0.0)
                    row[f"{method}_num_images"] = method_counts.get(method, 0)

                writer.writerow(row)

        logger.info(f"Saved aggregate HFS metrics to CSV: {csv_file}")
        return str(csv_file)

    def append_to_cumulative_hfs_csv(
        self,
        round_data: dict,
        config_name: str,
        output_dir: str | None = None,
    ) -> str:
        """
        Append round data to cumulative HFS CSV file.

        Args:
            round_data: Dictionary containing round aggregation data
            config_name: Configuration name for file naming
            output_dir: Optional output directory (uses base_output_dir if None)

        Returns:
            Path to CSV file
        """
        import csv
        from pathlib import Path

        if output_dir is None:
            output_dir_path: Path = self.base_output_dir
        else:
            output_dir_path = Path(output_dir)

        csv_file = output_dir_path / f"hfs_cumulative_{config_name}.csv"

        # Ensure output directory exists
        csv_file.parent.mkdir(parents=True, exist_ok=True)

        # Determine methods in this round's data
        aggregate_hfs = round_data.get("aggregate_hfs", {})
        method_counts = round_data.get("method_counts", {})
        all_methods = set(aggregate_hfs.keys())

        # Check if file exists and read existing columns
        existing_methods = set()
        if csv_file.exists():
            with open(csv_file) as f:
                reader = csv.DictReader(f)
                if reader.fieldnames:
                    # Extract method names from existing columns
                    for field in reader.fieldnames:
                        if field.endswith("_mean_hfs"):
                            method = field.replace("_mean_hfs", "")
                            existing_methods.add(method)

        # Combine existing and new methods
        all_methods.update(existing_methods)

        # Define CSV columns
        columns = ["round", "num_images_processed", "total_processing_time_seconds"]
        for method in sorted(all_methods):
            columns.extend(
                [
                    f"{method}_mean_hfs",
                    f"{method}_num_images",
                ]
            )

        # Read existing data if file exists
        existing_data = []
        if csv_file.exists():
            with open(csv_file) as f:
                reader = csv.DictReader(f)
                existing_data = list(reader)

        # Write all data with updated columns
        with open(csv_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()

            # Write existing data with new columns (fill missing with 0.0)
            for row in existing_data:
                updated_row = {}
                for col in columns:
                    if col in row:
                        updated_row[col] = row[col]
                    elif col.endswith("_mean_hfs"):
                        updated_row[col] = "0.0"
                    elif col.endswith("_num_images"):
                        updated_row[col] = "0"
                    else:
                        updated_row[col] = "0"
                writer.writerow(updated_row)

            # Write new round data
            row = {
                "round": round_data.get("round_num", 0),
                "num_images_processed": round_data.get("num_images_processed", 0),
                "total_processing_time_seconds": round_data.get("total_processing_time", 0.0),
            }

            # Add method-specific metrics
            for method in sorted(all_methods):
                row[f"{method}_mean_hfs"] = aggregate_hfs.get(method, 0.0)
                row[f"{method}_num_images"] = method_counts.get(method, 0)

            writer.writerow(row)

        logger.debug(
            f"Appended round {round_data.get('round_num', 0)} data to cumulative HFS CSV: {csv_file}"
        )
        return str(csv_file)
