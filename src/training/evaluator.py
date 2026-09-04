"""Metrics collection and evaluation for trained models."""

from datetime import datetime
import json
import logging
from pathlib import Path
import time
from typing import Any

from ultralytics import YOLO

from .checkpoint import Checkpoint_Selector

logger = logging.getLogger(__name__)


class Metrics_Collector:  # pylint: disable=invalid-name
    """Collects and saves model performance metrics."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.checkpoint_selector = Checkpoint_Selector()

    def evaluate_and_save(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        project_dir: str,
        config_name: str,
        data_yaml: str,
        output_dir: str,
        run_id: int | None = None,
        random_seed: int | None = None,
        training_time: float | None = None,
    ) -> dict[str, Any]:
        """
        Evaluate model and save metrics.

        Args:
            project_dir: YOLO training project directory
            config_name: Configuration name
            data_yaml: Path to data.yaml
            output_dir: Directory to save evaluation results
            run_id: Run identifier for multi-run experiments
            random_seed: Random seed used for training
            training_time: Training time in seconds

        Returns:
            Dictionary of evaluation metrics
        """
        self.logger.info("Evaluating model from %s", project_dir)

        # Select best checkpoint
        try:
            best_checkpoint = self.checkpoint_selector.select_best_checkpoint(project_dir)
        except FileNotFoundError as exc:
            self.logger.error("Checkpoint selection failed: %s", exc)
            raise

        # Load model
        model = YOLO(best_checkpoint)

        # Run validation
        self.logger.info("Running validation...")
        val_start = time.time()
        results = model.val(data=data_yaml, verbose=False, plots=False, workers=0)
        val_time = time.time() - val_start

        # Extract metrics
        metrics = self._extract_metrics(
            results=results,
            model=model,
            config_name=config_name,
            best_checkpoint=best_checkpoint,
            run_id=run_id,
            random_seed=random_seed,
            training_time=training_time,
            val_time=val_time,
        )

        # Save metrics
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if run_id is not None:
            metrics_file = output_path / f"run_{run_id}" / "evaluation_results.json"
            metrics_file.parent.mkdir(parents=True, exist_ok=True)
        else:
            metrics_file = output_path / "evaluation_results.json"

        with open(metrics_file, "w", encoding="utf-8") as fh:
            json.dump(metrics, fh, indent=2)
            fh.write("\n")

        self.logger.info("Metrics saved to %s", metrics_file)

        return metrics

    def _extract_metrics(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        results: Any,
        model: YOLO,
        config_name: str,
        best_checkpoint: str,
        run_id: int | None,
        random_seed: int | None,
        training_time: float | None,
        val_time: float,
    ) -> dict[str, Any]:
        """Extract metrics from validation results."""

        # Get model info
        model_info = self._get_model_info(model)

        # Build metrics dictionary
        metrics: dict[str, Any] = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "config_name": config_name,
            "best_checkpoint": best_checkpoint,
            "model_info": model_info,
            "metrics": {
                "mAP50": float(results.box.map50) if hasattr(results.box, "map50") else 0.0,
                "mAP50-95": float(results.box.map) if hasattr(results.box, "map") else 0.0,
                "precision": float(results.box.mp) if hasattr(results.box, "mp") else 0.0,
                "recall": float(results.box.mr) if hasattr(results.box, "mr") else 0.0,
                "fitness": float(results.fitness) if hasattr(results, "fitness") else 0.0,
            },
        }

        self._attach_per_class_metrics(metrics, results)

        # Add run information
        if run_id is not None:
            metrics["run_id"] = run_id

        if random_seed is not None:
            metrics["random_seed"] = random_seed

        if training_time is not None:
            metrics["training_time_seconds"] = training_time
            metrics["training_time_minutes"] = training_time / 60.0

        metrics["validation_time_seconds"] = val_time

        return metrics

    def _get_model_info(self, model: YOLO) -> dict[str, Any]:
        """Get model architecture information."""
        info: dict[str, Any] = {}

        try:
            # Get model parameters
            total_params = sum(p.numel() for p in model.model.parameters())
            info["parameters"] = int(total_params)

            # Get model name
            if hasattr(model, "ckpt_path"):
                info["model_name"] = Path(model.ckpt_path).stem

            # Try to get GFLOPs (may not be available)
            if hasattr(model.model, "info"):
                model_info = model.model.info(verbose=False)
                if isinstance(model_info, dict) and "GFLOPs" in model_info:
                    info["GFLOPs"] = float(model_info["GFLOPs"])

            # Inference time is measured during actual inference
            info["inference_time_ms"] = None

        except Exception as exc:  # pylint: disable=broad-exception-caught
            self.logger.warning("Could not extract all model info: %s", exc)

        return info

    def collect_round_metrics(  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
        self,
        checkpoint_path: str,
        data_yaml: str,
        round_num: int,
        training_set_size: int,
        verified_samples_added: int,
        output_dir: str,
        training_time: float,
        rejected_count: int | None = None,
        undetected_count: int | None = None,
        remaining_pool_size: int | None = None,
        unlabeled_pool_size_at_round_start: int | None = None,
        total_detections: int | None = None,
        actual_stopped_epoch: int | None = None,
        best_epoch: int | None = None,
        class_distribution: dict[str, Any] | None = None,
        optimizer_steps: int | None = None,
        cumulative_optimizer_steps: int | None = None,
        images_processed: int | None = None,
        cumulative_images_processed: int | None = None,
        gflops: float | None = None,
        training_tflops: float | None = None,
        cumulative_training_tflops: float | None = None,
    ) -> dict[str, Any]:
        """
        Collect metrics for a specific training round.

        Args:
            checkpoint_path: Path to best checkpoint for this round
            data_yaml: Path to data.yaml for this round
            round_num: Round number (1-5)
            training_set_size: Number of images in training set
            verified_samples_added: Number of verified samples added in this round
            output_dir: Directory to save round metrics
            training_time: Training time for this round in seconds
            rejected_count: Number of rejected samples (optional)
            undetected_count: Number of undetected samples (optional)
            remaining_pool_size: Size of remaining unlabeled pool (optional)
            unlabeled_pool_size_at_round_start: Pool size before simulation (optional)
            total_detections: Total detections from edge simulation (optional)
            actual_stopped_epoch: Epoch training stopped on (early stopping runs)
            best_epoch: Epoch with best validation fitness
            class_distribution: Dictionary of class label counts in training set
            optimizer_steps: Optimizer update steps taken in this round
            cumulative_optimizer_steps: Cumulative optimizer steps across all rounds
            images_processed: Number of images processed in this round
            cumulative_images_processed: Cumulative images processed across all rounds
            gflops: Theoretical forward GFLOPs per 640x640 image
            training_tflops: Estimated training compute (forward+backward) in TFLOPs
            cumulative_training_tflops: Cumulative training compute across all rounds

        Returns:
            Dictionary of round metrics
        """
        self.logger.info("Collecting metrics for Round %d...", round_num)

        # Load model
        model = YOLO(checkpoint_path)

        # Run validation
        self.logger.info("Running validation for Round %d...", round_num)
        val_start = time.time()
        results = model.val(data=data_yaml, verbose=False, plots=False, workers=0)
        val_time = time.time() - val_start

        # Extract metrics
        metrics: dict[str, Any] = {
            "round": round_num,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "checkpoint_path": checkpoint_path,
            "training_set_size": training_set_size,
            "verified_samples_added": verified_samples_added,
            "training_time_seconds": training_time,
            "validation_time_seconds": val_time,
            "metrics": {
                "mAP50": float(results.box.map50) if hasattr(results.box, "map50") else 0.0,
                "mAP50-95": float(results.box.map) if hasattr(results.box, "map") else 0.0,
                "precision": float(results.box.mp) if hasattr(results.box, "mp") else 0.0,
                "recall": float(results.box.mr) if hasattr(results.box, "mr") else 0.0,
                "f1_score": self._compute_f1(
                    float(results.box.mp) if hasattr(results.box, "mp") else 0.0,
                    float(results.box.mr) if hasattr(results.box, "mr") else 0.0,
                ),
            },
        }

        # Add pool statistics if provided
        if rejected_count is not None:
            metrics["rejected_count"] = rejected_count
        if undetected_count is not None:
            metrics["undetected_count"] = undetected_count
        if remaining_pool_size is not None:
            metrics["remaining_pool_size"] = remaining_pool_size
        if unlabeled_pool_size_at_round_start is not None:
            metrics["unlabeled_pool_size_at_round_start"] = unlabeled_pool_size_at_round_start
        if total_detections is not None:
            metrics["total_detections"] = total_detections

        # Add early stopping information if provided
        if actual_stopped_epoch is not None:
            metrics["actual_stopped_epoch"] = actual_stopped_epoch
        if best_epoch is not None:
            metrics["best_epoch"] = best_epoch

        # Add compute and sample accounting (Reviewer 1 Major 3, Major 11)
        if class_distribution is not None:
            metrics["class_distribution"] = class_distribution
        if optimizer_steps is not None:
            metrics["optimizer_steps"] = optimizer_steps
        if cumulative_optimizer_steps is not None:
            metrics["cumulative_optimizer_steps"] = cumulative_optimizer_steps
        if images_processed is not None:
            metrics["images_processed"] = images_processed
        if cumulative_images_processed is not None:
            metrics["cumulative_images_processed"] = cumulative_images_processed
        if gflops is not None:
            metrics["gflops"] = gflops
        if training_tflops is not None:
            metrics["training_tflops"] = training_tflops
        if cumulative_training_tflops is not None:
            metrics["cumulative_training_tflops"] = cumulative_training_tflops

        self._attach_per_class_metrics(metrics, results)

        # Save round metrics
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        metrics_file = output_path / f"round_{round_num}_metrics.json"

        with open(metrics_file, "w", encoding="utf-8") as fh:
            json.dump(metrics, fh, indent=2)
            fh.write("\n")

        self.logger.info("Round %d metrics saved to %s", round_num, metrics_file)

        return metrics

    def evaluate_final_test(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        checkpoint_path: str,
        data_yaml: str,
        output_dir: str,
        config_name: str,
        random_seed: int,
        total_training_time: float,
        cumulative_optimizer_steps: int | None = None,
        cumulative_images_processed: int | None = None,
        cumulative_training_tflops: float | None = None,
    ) -> dict[str, Any]:
        """
        Evaluate final model (Round 5 best checkpoint) on test_fixed.

        Args:
            checkpoint_path: Path to Round 5 best checkpoint
            data_yaml: Path to data.yaml with test_fixed
            output_dir: Directory to save final test metrics
            config_name: Configuration name
            random_seed: Random seed used
            total_training_time: Total training time across all rounds
            cumulative_optimizer_steps: Total optimizer steps across all rounds
            cumulative_images_processed: Total images processed across all rounds
            cumulative_training_tflops: Total training TFLOPs across all rounds

        Returns:
            Dictionary of final test metrics
        """
        self.logger.info("Evaluating final model on test set...")

        # Load model
        model = YOLO(checkpoint_path)

        # Run validation on test set
        self.logger.info("Running test evaluation...")
        test_start = time.time()
        results = model.val(data=data_yaml, split="test", verbose=False, plots=False, workers=0)
        test_time = time.time() - test_start

        # Get model info
        model_info = self._get_model_info(model)

        # Extract metrics
        metrics: dict[str, Any] = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "config_name": config_name,
            "checkpoint_path": checkpoint_path,
            "random_seed": random_seed,
            "total_training_time_seconds": total_training_time,
            "total_training_time_minutes": total_training_time / 60.0,
            "test_time_seconds": test_time,
            "cumulative_optimizer_steps": cumulative_optimizer_steps,
            "cumulative_images_processed": cumulative_images_processed,
            "cumulative_training_tflops": cumulative_training_tflops,
            "model_info": model_info,
            "metrics": {
                "mAP50": float(results.box.map50) if hasattr(results.box, "map50") else 0.0,
                "mAP50-95": float(results.box.map) if hasattr(results.box, "map") else 0.0,
                "precision": float(results.box.mp) if hasattr(results.box, "mp") else 0.0,
                "recall": float(results.box.mr) if hasattr(results.box, "mr") else 0.0,
                "f1_score": self._compute_f1(
                    float(results.box.mp) if hasattr(results.box, "mp") else 0.0,
                    float(results.box.mr) if hasattr(results.box, "mr") else 0.0,
                ),
            },
        }

        self._attach_per_class_metrics(metrics, results)

        # Save final test metrics
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        metrics_file = output_path / "final_test_metrics.json"

        with open(metrics_file, "w", encoding="utf-8") as fh:
            json.dump(metrics, fh, indent=2)
            fh.write("\n")

        self.logger.info("Final test metrics saved to %s", metrics_file)

        return metrics

    def _attach_per_class_metrics(self, metrics: dict[str, Any], results: Any) -> None:
        """Attach per-class metrics to a metrics dict if results carry them."""
        if not hasattr(results.box, "maps"):
            return
        per_class: dict[str, Any] = {
            "mAP50_per_class": [float(x) for x in results.box.ap50],
            "mAP50-95_per_class": [float(x) for x in results.box.ap],
            "precision_per_class": [float(x) for x in results.box.p]
            if hasattr(results.box, "p")
            else [],
            "recall_per_class": [float(x) for x in results.box.r]
            if hasattr(results.box, "r")
            else [],
        }
        if hasattr(results.box, "p") and hasattr(results.box, "r"):
            per_class["f1_score_per_class"] = [
                self._compute_f1(float(p), float(r))
                for p, r in zip(results.box.p, results.box.r, strict=False)
            ]
        metrics["per_class_metrics"] = per_class

    def _compute_f1(self, precision: float, recall: float) -> float:
        """Compute F1 score from precision and recall."""
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)
