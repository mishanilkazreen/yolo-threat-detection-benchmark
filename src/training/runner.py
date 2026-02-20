"""Experiment runner for training and evaluation."""

import logging
from pathlib import Path
import time
from typing import Any

from ultralytics import YOLO
import yaml

from ..config.parser import ConfigurationParser
from ..data.validator import Dataset_Validator
from .device_utils import select_device
from .evaluator import Metrics_Collector
from .seed_manager import Seed_Manager

logger = logging.getLogger(__name__)


class Experiment_Runner:
    """Orchestrates training, evaluation, and multi-run experiments."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.seed_manager = Seed_Manager()
        self.metrics_collector = Metrics_Collector()
        self.dataset_validator = Dataset_Validator()

    def run_experiment(self, config_path: str, validate_dataset: bool = True) -> dict[str, Any]:
        """
        Run complete experiment: training and evaluation.

        Args:
            config_path: Path to training configuration YAML
            validate_dataset: Whether to validate dataset before training

        Returns:
            Dictionary of experiment results
        """
        self.logger.info(f"Starting experiment with config: {config_path}")

        # Load configuration
        config = ConfigurationParser.parse(config_path)
        config_name = Path(config_path).stem

        # Validate dataset if requested
        if validate_dataset:
            self._validate_dataset(config.data.yaml_path)

        # Check if multi-run experiment
        if config.training.runs > 1:
            return self._run_multi_run_experiment(config, config_name)
        else:
            return self._run_single_experiment(config, config_name, run_id=None)

    def _run_single_experiment(
        self, config: Any, config_name: str, run_id: int | None = None
    ) -> dict[str, Any]:
        """Run a single training and evaluation."""

        # Get seed for this run
        if run_id is not None:
            seed = self.seed_manager.get_run_seed(run_id, config.training.seeds)
        else:
            seed = config.training.seeds[0] if config.training.seeds else 42

        # Set seed for reproducibility
        self.seed_manager.set_seed(seed)

        # Create output directories
        output_dir = f"outputs/{config_name}"
        explanations_dir = f"explanations/{config_name}"

        for dir_path in [output_dir, explanations_dir]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)

        # Initialize model
        self.logger.info(f"Initializing model: {config.model.name}")
        model = YOLO(config.model.weights)

        # Select device
        device = select_device(config.training.device)

        # Train model
        self.logger.info(f"Starting training (seed={seed}, device={device})...")
        train_start = time.time()

        results = model.train(
            data=config.data.yaml_path,
            epochs=config.training.epochs,
            imgsz=config.training.image_size,
            patience=config.training.patience,
            device=device,
            project="runs/detect",
            name=config_name,
            exist_ok=True,
            verbose=True,
        )

        # The actual project directory where YOLO saves results
        project_dir = f"runs/detect/{config_name}"

        training_time = time.time() - train_start
        self.logger.info(f"Training completed in {training_time:.2f} seconds")

        # Determine actual project directory (YOLO may create nested structure)
        # Check both possible locations
        possible_dirs = [f"runs/detect/{config_name}", f"runs/detect/runs/detect/{config_name}"]

        actual_project_dir = None
        for dir_path in possible_dirs:
            weights_dir = Path(dir_path) / "weights"
            if weights_dir.exists() and (weights_dir / "best.pt").exists():
                actual_project_dir = dir_path
                break

        if actual_project_dir is None:
            raise FileNotFoundError(f"Could not find training results. Checked: {possible_dirs}")

        self.logger.info(f"Found training results at: {actual_project_dir}")

        # Evaluate model
        self.logger.info("Evaluating model...")
        metrics = self.metrics_collector.evaluate_and_save(
            project_dir=actual_project_dir,
            config_name=config_name,
            data_yaml=config.data.yaml_path,
            output_dir=output_dir,
            run_id=run_id,
            random_seed=seed,
            training_time=training_time,
        )

        return metrics

    def _run_multi_run_experiment(self, config: Any, config_name: str) -> dict[str, Any]:
        """Run multiple training runs with different seeds."""

        self.logger.info(f"Starting multi-run experiment: {config.training.runs} runs")

        all_metrics = []

        for run_id in range(config.training.runs):
            self.logger.info(f"\n{'=' * 60}")
            self.logger.info(f"Run {run_id + 1}/{config.training.runs}")
            self.logger.info(f"{'=' * 60}\n")

            metrics = self._run_single_experiment(config, config_name, run_id=run_id)
            all_metrics.append(metrics)

        # Aggregate results
        aggregated = self._aggregate_multi_run_results(all_metrics, config_name)

        return aggregated

    def _aggregate_multi_run_results(self, all_metrics: list, config_name: str) -> dict[str, Any]:
        """Aggregate results from multiple runs."""

        import numpy as np

        # Extract mAP50 values
        map50_values = [m["metrics"]["mAP50"] for m in all_metrics]

        aggregated = {
            "config_name": config_name,
            "num_runs": len(all_metrics),
            "metrics": {
                "mAP50_mean": float(np.mean(map50_values)),
                "mAP50_std": float(np.std(map50_values)),
                "mAP50_min": float(np.min(map50_values)),
                "mAP50_max": float(np.max(map50_values)),
            },
            "individual_runs": all_metrics,
        }

        # Save aggregated results
        output_path = Path(f"outputs/{config_name}/aggregated_results.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        import json

        with open(output_path, "w") as f:
            json.dump(aggregated, f, indent=2)

        self.logger.info(f"Aggregated results saved to {output_path}")

        return aggregated

    def _validate_dataset(self, data_yaml_path: str) -> None:
        """Validate dataset before training."""

        self.logger.info("Validating dataset...")

        # Load data.yaml
        with open(data_yaml_path) as f:
            data_config = yaml.safe_load(f)

        # Get base path
        base_path = Path(data_config.get("path", "."))

        # Validate
        result = self.dataset_validator.validate_dataset(data_config, base_path)

        if not result.is_valid:
            raise ValueError(
                f"Dataset validation failed: "
                f"{len(result.missing_labels)} missing labels, "
                f"{len(result.empty_labels)} empty labels, "
                f"{len(result.duplicate_files)} duplicates"
            )

        self.logger.info("Dataset validation passed")
