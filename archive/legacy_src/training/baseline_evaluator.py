"""Baseline evaluator for one-shot baseline model assessment."""

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from ultralytics import YOLO

from ..config.parser import Configuration
from ..explainability.hfs_scorer import Heatmap_Focus_Scorer
from .evaluator import Metrics_Collector

logger = logging.getLogger(__name__)


class Baseline_Evaluator:
    """
    Evaluates the one-shot baseline model on val_fixed and test_fixed.

    Delegates to existing Metrics_Collector for metric computation and
    Heatmap_Focus_Scorer for HFS computation on the same fixed image
    subset used by the incremental experiment.
    """

    def __init__(self, output_manager=None):
        self.logger = logging.getLogger(__name__)
        self.metrics_collector = Metrics_Collector()
        self.hfs_scorer = Heatmap_Focus_Scorer()

        # Import and initialize Output_Manager if not provided
        if output_manager is None:
            from ..utils.output_manager import Output_Manager

            output_manager = Output_Manager()
        self.output_manager = output_manager

    def evaluate(
        self,
        checkpoint_path: str,
        config_name: str,
        val_data_yaml: str,
        test_data_yaml: str,
        hfs_image_subset: list[str],
        base_path: Path,
        seed: int,
        training_time: float,
        training_set_size: int,
        epochs: int,
        config: Configuration,
    ) -> dict[str, Any]:
        """
        Evaluate baseline model on val_fixed and test_fixed.

        Args:
            checkpoint_path: Path to the trained baseline checkpoint
            config_name: Configuration name (used for output directory naming)
            val_data_yaml: Path to data.yaml pointing to val_fixed
            test_data_yaml: Path to data.yaml pointing to test_fixed
            hfs_image_subset: Fixed list of image filenames for HFS computation
            base_path: Dataset root path
            seed: Random seed used for training
            training_time: Total training time in seconds
            training_set_size: Number of images in the full training partition
            epochs: Number of epochs trained
            config: Full experiment configuration

        Returns:
            dict with val_metrics, test_metrics, hfs_metrics, and metadata
        """
        baseline_name = f"{config_name}_baseline"

        # Use Output_Manager for path construction
        output_dir = self.output_manager.get_evaluation_output_path(
            model_name=baseline_name,
            round_name=None,  # Use default round name
            create=True,
        )

        gradcam_dir = self.output_manager.get_explainability_output_path(
            model_name=baseline_name,
            round_name=None,  # Use default round name
            method="gradcam",
            create=True,
        )

        self.logger.info("Evaluating baseline model: %s", baseline_name)
        self.logger.info("  Checkpoint: %s", checkpoint_path)

        # --- 1. Evaluate on val_fixed ---
        self.logger.info("Evaluating on val_fixed...")
        val_metrics = self._evaluate_on_split(
            checkpoint_path=checkpoint_path,
            data_yaml=val_data_yaml,
            split="val",
        )
        val_metrics_path = output_dir / "val_metrics.json"
        with open(val_metrics_path, "w", encoding="utf-8") as f:
            json.dump(val_metrics, f, indent=2)
            f.write("\n")
        self.logger.info("Val metrics saved to %s", val_metrics_path)

        # --- 2. Evaluate on test_fixed ---
        self.logger.info("Evaluating on test_fixed...")
        test_metrics_raw = self._evaluate_on_split(
            checkpoint_path=checkpoint_path,
            data_yaml=test_data_yaml,
            split="test",
        )

        # Build the final_test_metrics.json with all required metadata fields
        iou_threshold = getattr(config.data, "iou_threshold", 0.5) or 0.5
        final_test_metrics = {
            "config_name": config_name,
            "model": config.model.name,
            "weights": config.model.weights,
            "random_seed": seed,
            "epochs": epochs,
            "training_set_size": training_set_size,
            "training_time_seconds": training_time,
            "training_time_minutes": training_time / 60.0,
            "iou_threshold": iou_threshold,
            "test_metrics": {
                "mAP50": test_metrics_raw["mAP50"],
                "mAP50-95": test_metrics_raw["mAP50-95"],
                "precision": test_metrics_raw["precision"],
                "recall": test_metrics_raw["recall"],
                "f1_score": test_metrics_raw["f1_score"],
            },
            "per_class_test_metrics": {
                "mAP50_per_class": test_metrics_raw.get("mAP50_per_class", []),
            },
        }

        final_test_metrics_path = output_dir / "final_test_metrics.json"
        with open(final_test_metrics_path, "w", encoding="utf-8") as f:
            json.dump(final_test_metrics, f, indent=2)
            f.write("\n")
        self.logger.info("Final test metrics saved to %s", final_test_metrics_path)

        # --- 3. Compute HFS on the fixed image subset ---
        self.logger.info("Computing HFS on %s fixed images...", len(hfs_image_subset))
        hfs_metrics = self._compute_hfs(
            checkpoint_path=checkpoint_path,
            hfs_image_subset=hfs_image_subset,
            base_path=base_path,
            gradcam_dir=gradcam_dir,
        )

        hfs_metrics_path = output_dir / "hfs_metrics.json"
        with open(hfs_metrics_path, "w", encoding="utf-8") as f:
            json.dump(hfs_metrics, f, indent=2)
            f.write("\n")
        self.logger.info("HFS metrics saved to %s", hfs_metrics_path)

        return {
            "val_metrics": val_metrics,
            "test_metrics": final_test_metrics,
            "hfs_metrics": hfs_metrics,
            "output_dir": str(output_dir),
        }

    def _evaluate_on_split(
        self,
        checkpoint_path: str,
        data_yaml: str,
        split: str,
    ) -> dict[str, Any]:
        """
        Load model and run validation on the given split.

        Args:
            checkpoint_path: Path to model checkpoint
            data_yaml: Path to data.yaml
            split: Dataset split to evaluate ("val" or "test")

        Returns:
            Dict with mAP50, mAP50-95, precision, recall, f1_score, mAP50_per_class
        """
        model = YOLO(checkpoint_path)

        kwargs: dict[str, Any] = {
            "data": data_yaml,
            "verbose": False,
            "plots": False,
            "workers": 0,
        }
        if split == "test":
            kwargs["split"] = "test"

        results = model.val(**kwargs)

        precision = float(results.box.mp) if hasattr(results.box, "mp") else 0.0
        recall = float(results.box.mr) if hasattr(results.box, "mr") else 0.0
        f1 = self.metrics_collector._compute_f1(precision, recall)

        metrics: dict[str, Any] = {
            "mAP50": float(results.box.map50) if hasattr(results.box, "map50") else 0.0,
            "mAP50-95": float(results.box.map) if hasattr(results.box, "map") else 0.0,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
        }

        if hasattr(results.box, "ap50"):
            metrics["mAP50_per_class"] = [float(x) for x in results.box.ap50]

        return metrics

    def _compute_hfs(
        self,
        checkpoint_path: str,
        hfs_image_subset: list[str],
        base_path: Path,
        gradcam_dir: Path,
    ) -> dict[str, Any]:
        """
        Compute HFS for the fixed image subset using Grad-CAM heatmaps.

        Generates synthetic heatmaps via model inference and delegates
        HFS computation to Heatmap_Focus_Scorer.

        Args:
            checkpoint_path: Path to model checkpoint
            hfs_image_subset: List of image filenames
            base_path: Dataset root path
            gradcam_dir: Directory to save Grad-CAM heatmaps

        Returns:
            HFS metrics dict
        """
        if not hfs_image_subset:
            self.logger.warning("HFS image subset is empty; returning zero HFS")
            return {
                "round": 0,
                "method": "gradcam",
                "mean_hfs": 0.0,
                "num_images": 0,
                "individual_hfs": {},
                "statistics": {"min": 0.0, "max": 0.0, "std": 0.0, "median": 0.0},
            }

        val_image_dir = base_path / "valid" / "images"

        # Build full image paths
        image_paths = [str(val_image_dir / img) for img in hfs_image_subset]

        # Load ground-truth bboxes
        bboxes = self.hfs_scorer.load_ground_truth_bboxes(image_paths, base_path)

        # Generate heatmaps: use model confidence maps as proxy for Grad-CAM
        model = YOLO(checkpoint_path)
        heatmaps = []
        image_sizes = []
        valid_image_ids = []
        valid_bboxes = []

        for img_path, bbox in zip(image_paths, bboxes, strict=False):
            path = Path(img_path)
            if not path.exists():
                self.logger.warning(f"Image not found, skipping HFS: {img_path}")
                continue

            try:
                from PIL import Image as PILImage

                with PILImage.open(path) as img:
                    w, h = img.size
                image_sizes.append((w, h))
            except Exception as e:  # pylint: disable=broad-except
                self.logger.warning("Could not read image size for %s: %s", img_path, e)
                image_sizes.append((640, 640))
                w, h = 640, 640

            # Generate a simple confidence-based heatmap via inference
            heatmap = self._generate_heatmap(model, img_path, w, h)

            # Save heatmap as image
            heatmap_filename = gradcam_dir / f"{path.stem}_gradcam.npy"
            np.save(str(heatmap_filename), heatmap)

            heatmaps.append(heatmap)
            valid_image_ids.append(path.stem)
            valid_bboxes.append(bbox)

        if not heatmaps:
            self.logger.warning("No valid heatmaps generated; returning zero HFS")
            return {
                "round": 0,
                "method": "gradcam",
                "mean_hfs": 0.0,
                "num_images": 0,
                "individual_hfs": {},
                "statistics": {"min": 0.0, "max": 0.0, "std": 0.0, "median": 0.0},
            }

        # Compute mean HFS
        mean_hfs = self.hfs_scorer.compute_mean_hfs(heatmaps, valid_bboxes, image_sizes)

        # Compute individual HFS scores
        individual_hfs_scores = []
        for heatmap, bbox, (w, h) in zip(heatmaps, valid_bboxes, image_sizes, strict=False):
            score = self.hfs_scorer.compute_hfs(heatmap, bbox, w, h)
            individual_hfs_scores.append(score)

        # Build metrics dict (mirrors save_hfs_metrics format)
        hfs_metrics: dict[str, Any] = {
            "round": 0,  # baseline has no round concept
            "method": "gradcam",
            "mean_hfs": mean_hfs,
            "num_images": len(individual_hfs_scores),
            "individual_hfs": dict(zip(valid_image_ids, individual_hfs_scores, strict=False)),
            "statistics": {
                "min": float(np.min(individual_hfs_scores)),
                "max": float(np.max(individual_hfs_scores)),
                "std": float(np.std(individual_hfs_scores)),
                "median": float(np.median(individual_hfs_scores)),
            },
        }

        return hfs_metrics

    def _generate_heatmap(
        self,
        model: YOLO,
        image_path: str,
        image_width: int,
        image_height: int,
    ) -> np.ndarray:
        """
        Generate a confidence-based heatmap for a single image.

        Uses model predictions to create a spatial confidence map as a
        proxy for Grad-CAM. Detected bounding boxes are filled with their
        confidence scores; the rest of the map is near-zero.

        Args:
            model: Loaded YOLO model
            image_path: Path to the image
            image_width: Image width in pixels
            image_height: Image height in pixels

        Returns:
            2D numpy array (image_height x image_width) with confidence values
        """
        heatmap = np.zeros((image_height, image_width), dtype=np.float32)

        try:
            results = model.predict(image_path, verbose=False)
            if results and len(results) > 0:
                result = results[0]
                if result.boxes is not None and len(result.boxes) > 0:
                    confs = result.boxes.conf.tolist()
                    xyxys = result.boxes.xyxy.tolist()
                    for (x1, y1, x2, y2), conf in zip(xyxys, confs, strict=True):
                        x1 = max(0, int(x1))
                        y1 = max(0, int(y1))
                        x2 = min(image_width, int(x2))
                        y2 = min(image_height, int(y2))
                        heatmap[y1:y2, x1:x2] = np.maximum(heatmap[y1:y2, x1:x2], conf)
        except Exception as e:  # pylint: disable=broad-except
            self.logger.warning("Heatmap generation failed for %s: %s", image_path, e)
            # Return uniform heatmap as fallback
            heatmap = np.ones((image_height, image_width), dtype=np.float32) * 0.1

        return heatmap
