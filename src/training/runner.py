"""Experiment runner for training and evaluation."""

import gc
import json
import logging
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
from ultralytics import YOLO
import yaml

from ..aggregation.comparison_reporter import Comparison_Reporter
from ..config.parser import ConfigurationParser
from ..data.validator import Dataset_Validator
from .baseline_evaluator import Baseline_Evaluator
from .baseline_trainer import Baseline_Trainer
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

        # Import incremental training components lazily to avoid circular imports
        self._splitter = None
        self._training_set_manager = None
        self._edge_agent_simulator = None
        self._detection_validator = None
        self._round_metrics_tracker = None

    @property
    def splitter(self):
        """Lazy load Dataset_Splitter."""
        if self._splitter is None:
            from ..data.splitter import Dataset_Splitter

            self._splitter = Dataset_Splitter()
        return self._splitter

    @property
    def training_set_manager(self):
        """Lazy load Training_Set_Manager."""
        if self._training_set_manager is None:
            from ..data.training_set_manager import Training_Set_Manager

            self._training_set_manager = Training_Set_Manager()
        return self._training_set_manager

    @property
    def edge_agent_simulator(self):
        """Lazy load Edge_Agent_Simulator."""
        if self._edge_agent_simulator is None:
            from .edge_agent_simulator import Edge_Agent_Simulator

            self._edge_agent_simulator = Edge_Agent_Simulator()
        return self._edge_agent_simulator

    @property
    def detection_validator(self):
        """Lazy load Detection_Validator."""
        if self._detection_validator is None:
            from .detection_validator import Detection_Validator

            self._detection_validator = Detection_Validator()
        return self._detection_validator

    @property
    def round_metrics_tracker(self):
        """Lazy load Round_Metrics_Tracker."""
        if self._round_metrics_tracker is None:
            from ..aggregation.round_metrics_tracker import Round_Metrics_Tracker

            self._round_metrics_tracker = Round_Metrics_Tracker()
        return self._round_metrics_tracker

    def _cleanup_memory(self):
        """
        Aggressively clean up memory between training rounds.

        This helps prevent memory accumulation during incremental training
        by clearing Python garbage, PyTorch caches, and CUDA memory.
        """
        # Force Python garbage collection
        gc.collect()

        # Clear PyTorch CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        self.logger.debug("Memory cleanup completed")

    def _create_step_decay_callback(self, lr0: float, lrf: float, step_interval: int = 5):
        """
        Create a callback for step decay learning rate schedule.

        Implements step decay: multiply LR by lrf every step_interval epochs.
        Formula: LR[epoch] = lr0 * (lrf ** (epoch // step_interval))

        Args:
            lr0: Initial learning rate
            lrf: Decay factor (e.g., 0.1 means multiply by 0.1)
            step_interval: Number of epochs between decay steps (default: 5)

        Returns:
            Callback function compatible with Ultralytics YOLO
        """

        def on_train_epoch_start(trainer):
            """Apply step decay to learning rate at the start of each epoch."""
            epoch = trainer.epoch

            # Calculate expected LR based on step decay formula
            decay_steps = epoch // step_interval
            expected_lr = lr0 * (lrf**decay_steps)

            # Update optimizer learning rates
            for param_group in trainer.optimizer.param_groups:
                param_group["lr"] = expected_lr

            # Log at step boundaries
            if epoch % step_interval == 0 and epoch > 0:
                self.logger.info(
                    f"Applying step decay: LR *= {lrf} at epoch {epoch} (new LR: {expected_lr:.6f})"
                )

        return on_train_epoch_start

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

        # Check if incremental training is enabled
        if (
            hasattr(config.training, "rounds")
            and config.training.rounds
            and config.training.rounds > 1
        ):
            return self._run_incremental_training(config, config_name, seed, run_id)
        else:
            return self._run_standard_training(config, config_name, seed, run_id)

    def _run_standard_training(
        self, config: Any, config_name: str, seed: int, run_id: int | None = None
    ) -> dict[str, Any]:
        """Run standard single-round training."""

        # Create output directories
        output_dir = f"outputs/{config_name}"
        explanations_dir = f"explanations/{config_name}"

        for dir_path in [output_dir, explanations_dir]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)

        # Initialize model
        self.logger.info(f"Initializing model: {config.model.name}")
        model = YOLO(config.model.weights)

        # Get hyperparameters
        lr0 = getattr(config.training, "lr0", 0.001)
        lrf = getattr(config.training, "lrf", 0.1)

        # Create and add step decay LR scheduler callback
        step_decay_callback = self._create_step_decay_callback(lr0, lrf, step_interval=5)
        model.add_callback("on_train_epoch_start", step_decay_callback)
        self.logger.info(f"Added step decay LR scheduler: lr0={lr0}, lrf={lrf}, step_interval=5")

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
            lr0=lr0,
            lrf=lrf,
            cos_lr=False,  # Disable cosine LR scheduler to use our custom step decay
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

    def _run_incremental_training(
        self, config: Any, config_name: str, seed: int, run_id: int | None = None
    ) -> dict[str, Any]:
        """
        Run 5-round incremental training with edge-cloud simulation.

        CRITICAL: Trains on ONLY newly verified samples per round (not cumulative).
        Round 1: train_init only
        Round 2+: ONLY verified samples from previous round

        Args:
            config: Training configuration
            config_name: Configuration name
            seed: Random seed
            run_id: Run identifier for multi-run experiments

        Returns:
            Dictionary of final metrics
        """
        self.logger.info(f"Starting incremental training: {config.training.rounds} rounds")
        self.logger.info(
            "ARCHITECTURE: Fine-tuning on ONLY newly verified samples per round (not cumulative)"
        )

        # Get incremental training parameters
        rounds = config.training.rounds
        epochs_per_round = getattr(config.training, "epochs_per_round", None)

        # If epochs_per_round not set, calculate from total epochs
        if epochs_per_round is None:
            if config.training.epochs is not None:
                epochs_per_round = config.training.epochs // rounds
                if epochs_per_round < 1:
                    raise ValueError(
                        f"Invalid configuration: epochs_per_round computed as 0. "
                        f"Ensure that total epochs ({config.training.epochs}) is at least equal to "
                        f"the number of rounds ({rounds}), or specify epochs_per_round explicitly."
                    )
            else:
                raise ValueError("Either epochs_per_round or epochs must be specified")

        train_init_percentage = getattr(config.data, "train_init_percentage", 0.2)
        iou_threshold = getattr(config.data, "iou_threshold", 0.5)

        # Get hyperparameters
        batch_size = getattr(config.training, "batch_size", 16)
        optimizer = getattr(config.training, "optimizer", "AdamW")
        lr0 = getattr(config.training, "lr0", 0.001)
        lrf = getattr(config.training, "lrf", 0.1)

        # Create output directories
        output_dir = f"outputs/{config_name}"
        explanations_dir = f"explanations/{config_name}"

        for dir_path in [output_dir, explanations_dir]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)

        # Load data.yaml to get dataset paths
        with open(config.data.yaml_path) as f:
            data_config = yaml.safe_load(f)

        base_path = Path(data_config.get("path", "."))
        train_path = base_path / data_config["train"]
        val_path = base_path / data_config.get("val", "valid/images")
        test_path = (
            base_path / data_config.get("test", "test/images") if "test" in data_config else None
        )

        # Step 1: Split dataset into train_init, unlabeled_pool, val_fixed, test_fixed
        self.logger.info(
            f"Splitting dataset (train_init={train_init_percentage * 100}%, seed={seed})..."
        )

        # Get all training images
        train_images = self._get_all_images(train_path)

        # Shuffle and split training data
        np.random.seed(seed)
        train_images_shuffled = train_images.copy()
        np.random.shuffle(train_images_shuffled)

        split_idx = int(len(train_images_shuffled) * train_init_percentage)
        train_init = train_images_shuffled[:split_idx]
        unlabeled_pool = train_images_shuffled[split_idx:]

        # Get validation and test images
        val_fixed = self._get_all_images(val_path) if val_path.exists() else []
        test_fixed = self._get_all_images(test_path) if test_path and test_path.exists() else []

        # Verify that the dataset split approximates the expected 70/20/10 ratio
        self._verify_split_ratio(
            train_count=len(train_images),
            val_count=len(val_fixed),
            test_count=len(test_fixed),
        )

        # Save split metadata
        metadata = {
            "train_init_count": len(train_init),
            "unlabeled_pool_count": len(unlabeled_pool),
            "val_fixed_count": len(val_fixed),
            "test_fixed_count": len(test_fixed),
            "train_init_percentage": train_init_percentage,
            "random_seed": seed,
        }

        metadata_file = Path(output_dir) / "split_metadata.json"
        with open(metadata_file, "w") as f:
            import json

            json.dump(metadata, f, indent=2)

        self.logger.info(f"Split metadata saved to {metadata_file}")
        self.logger.info(f"  train_init: {len(train_init)} images")
        self.logger.info(f"  unlabeled_pool: {len(unlabeled_pool)} images")
        self.logger.info(f"  val_fixed: {len(val_fixed)} images")
        self.logger.info(f"  test_fixed: {len(test_fixed)} images")

        splits = {
            "train_init": train_init,
            "unlabeled_pool": unlabeled_pool,
            "val_fixed": val_fixed,
            "test_fixed": test_fixed,
        }

        # Create temporary data.yaml files for each round
        round_data_yamls = {}
        for round_num in range(1, rounds + 1):
            round_data_yaml = Path(output_dir) / f"data_round_{round_num}.yaml"
            round_data_yamls[round_num] = str(round_data_yaml)

        # Initialize training set with train_init
        current_training_images = splits["train_init"].copy()
        unlabeled_pool = splits["unlabeled_pool"].copy()

        # Track metrics across rounds
        all_round_metrics = []
        best_checkpoint_path = None
        total_training_time = 0.0

        # Select device
        device = select_device(config.training.device)

        # Run incremental training rounds
        for round_num in range(1, rounds + 1):
            self.logger.info(f"\n{'=' * 60}")
            self.logger.info(f"Round {round_num}/{rounds}")
            self.logger.info(f"Training set size: {len(current_training_images)} images")
            self.logger.info(f"Unlabeled pool size: {len(unlabeled_pool)} images")
            self.logger.info(f"{'=' * 60}\n")

            # Skip rounds where the previous simulation produced no verified samples.
            # Round 1 always runs (trains on train_init). Subsequent rounds only run
            # when there are newly verified images to fine-tune on.
            if round_num > 1 and len(current_training_images) == 0:
                self.logger.warning(
                    f"Round {round_num} skipped — no verified samples from Round {round_num - 1} simulation. "
                    f"Checkpoint from Round {round_num - 1} carries forward unchanged."
                )
                continue

            # Create round-specific directories
            round_project_dir = f"runs/detect/{config_name}/round_{round_num}"
            round_explanations_dir = f"{explanations_dir}/round_{round_num}"

            Path(round_project_dir).mkdir(parents=True, exist_ok=True)
            Path(round_explanations_dir).mkdir(parents=True, exist_ok=True)

            # Create data.yaml for this round with current training set
            self._create_round_data_yaml(
                original_data_yaml=config.data.yaml_path,
                round_data_yaml=round_data_yamls[round_num],
                training_images=current_training_images,
                val_images=splits["val_fixed"],
                test_images=splits["test_fixed"],
                base_path=base_path,
            )

            # Initialize model
            if round_num == 1:
                # Round 1: Start from architecture weights (random init)
                self.logger.info(
                    f"Initializing model from architecture weights: {config.model.weights}"
                )
                model = YOLO(config.model.weights)
            else:
                # Round N > 1: Initialize from best checkpoint of Round N-1
                self.logger.info(f"Initializing model from Round {round_num - 1} best checkpoint")
                if best_checkpoint_path is None:
                    raise ValueError(f"No checkpoint found from Round {round_num - 1}")
                model = YOLO(best_checkpoint_path)

            # Create and add step decay LR scheduler callback
            step_decay_callback = self._create_step_decay_callback(lr0, lrf, step_interval=5)
            model.add_callback("on_train_epoch_start", step_decay_callback)
            if round_num == 1:
                self.logger.info(
                    f"Added step decay LR scheduler: lr0={lr0}, lrf={lrf}, step_interval=5"
                )

            # Train model for this round
            self.logger.info(
                f"Training Round {round_num} (epochs={epochs_per_round}, device={device})..."
            )
            self.logger.info(f"  Optimizer: {optimizer}, LR: {lr0}, Batch: {batch_size}")
            round_train_start = time.time()

            results = model.train(
                data=round_data_yamls[round_num],
                epochs=epochs_per_round,
                imgsz=config.training.image_size,
                patience=0,  # Disable early stopping so all epochs_per_round epochs always complete
                batch=batch_size,
                optimizer=optimizer,
                lr0=lr0,
                lrf=lrf,
                cos_lr=False,  # Disable cosine LR scheduler to use our custom step decay
                device=device,
                project="runs/detect",
                name=f"{config_name}/round_{round_num}",
                exist_ok=True,
                verbose=True,
            )

            round_training_time = time.time() - round_train_start
            total_training_time += round_training_time
            self.logger.info(
                f"Round {round_num} training completed in {round_training_time:.2f} seconds"
            )

            # Find best checkpoint for this round
            # YOLO may create nested directory structure, check both possible locations
            possible_checkpoint_paths = [
                Path(round_project_dir) / "weights" / "best.pt",
                Path(f"runs/detect/runs/detect/{config_name}/round_{round_num}/weights/best.pt"),
                Path(f"runs/detect/{config_name}/round_{round_num}/weights/best.pt"),
            ]

            best_checkpoint_path = None
            for checkpoint_path in possible_checkpoint_paths:
                if checkpoint_path.exists():
                    best_checkpoint_path = str(checkpoint_path)
                    self.logger.info(f"Found checkpoint at: {best_checkpoint_path}")
                    break

            if best_checkpoint_path is None:
                raise FileNotFoundError(
                    "Best checkpoint not found. Checked locations:\n"
                    + "\n".join([f"  - {p}" for p in possible_checkpoint_paths])
                )

            # Evaluate on val_fixed
            self.logger.info(f"Evaluating Round {round_num} on validation set...")

            # Calculate verified_samples_added for this round's metrics.
            # Round 1: 0 (training on train_init only, no prior simulation)
            # Round 2+: number of newly verified images from the preceding simulation
            if round_num == 1:
                verified_samples_added = 0
            else:
                # Load from previous round's validation results to get the actual verified count
                prev_validation_file = Path(output_dir) / f"round_{round_num - 1}_validations.json"
                verified_samples_added = 0
                if prev_validation_file.exists():
                    try:
                        with open(prev_validation_file) as f:
                            prev_validation = json.load(f)
                            verified_samples_added = len(
                                prev_validation.get("verified_samples", [])
                            )
                    except Exception as e:
                        self.logger.warning(
                            f"Could not load previous validation results for verified count: {e}"
                        )

            # Load pool statistics from previous round's validation (if available)
            rejected_count = None
            undetected_count = None
            remaining_pool_size = len(unlabeled_pool) if unlabeled_pool else None

            if round_num > 1:
                # Load validation results from previous round (file already opened above for verified count)
                prev_validation_file = Path(output_dir) / f"round_{round_num - 1}_validations.json"
                if prev_validation_file.exists():
                    try:
                        with open(prev_validation_file) as f:
                            prev_validation = json.load(f)
                            rejected_count = prev_validation.get("rejected_count")
                            undetected_count = prev_validation.get("undetected_count")
                    except Exception as e:
                        self.logger.warning(f"Could not load previous validation results: {e}")

            round_metrics = self.metrics_collector.collect_round_metrics(
                checkpoint_path=best_checkpoint_path,
                data_yaml=round_data_yamls[round_num],
                round_num=round_num,
                training_set_size=len(current_training_images),
                verified_samples_added=verified_samples_added,
                output_dir=output_dir,
                training_time=round_training_time,
                rejected_count=rejected_count,
                undetected_count=undetected_count,
                remaining_pool_size=remaining_pool_size,
            )

            all_round_metrics.append(round_metrics)

            # If not the last round, run edge-cloud simulation
            if round_num < rounds and len(unlabeled_pool) > 0:
                self.logger.info("Running edge agent simulation on unlabeled pool...")

                # Run inference on unlabeled pool
                detections = self.edge_agent_simulator.simulate_inference(
                    model_path=best_checkpoint_path,
                    unlabeled_images=unlabeled_pool,
                    base_path=base_path,
                    output_dir=output_dir,
                    round_num=round_num,
                    device=device,
                )

                self.logger.info(f"Edge agents generated {len(detections)} detections")

                # Validate detections against ground truth
                self.logger.info(f"Validating detections (IoU threshold={iou_threshold})...")

                # Ground truth labels are in the same structure as training images
                ground_truth_dir = str(train_path.parent / "labels")
                output_path = Path(output_dir) / f"round_{round_num}_validations.json"

                self.detection_validator.iou_threshold = iou_threshold
                validation_results = self.detection_validator.validate_detections(
                    detections=detections,
                    ground_truth_dir=ground_truth_dir,
                    unlabeled_pool=unlabeled_pool,
                    output_path=str(output_path),
                )

                verified_images = validation_results["verified_samples"]
                undetected_images = validation_results.get("undetected_images", [])
                self.logger.info(f"Verified {len(verified_images)} images for next round")
                self.logger.info(f"Undetected: {len(undetected_images)} images (stay in pool)")

                # For Round N+1, fine-tune on only the newly verified images from this round.
                # Round 1: train_init only
                # Round 2: verified images from Round 1 simulation only
                # Round 3: verified images from Round 2 simulation only, etc.
                if len(verified_images) > 0:
                    # Remove verified images from the unlabeled pool and log the transition.
                    # We do NOT accumulate into current_training_images — next round trains
                    # on the verified batch only.
                    self.training_set_manager.remove_from_pool(
                        verified_images=verified_images,
                        unlabeled_pool=unlabeled_pool,
                        output_dir=output_dir,
                        round_num=round_num + 1,
                    )
                    current_training_images = verified_images.copy()

                    self.logger.info(
                        f"Next round will fine-tune on {len(current_training_images)} newly verified images"
                    )
                else:
                    self.logger.warning(
                        f"No verified samples found in Round {round_num}. "
                        f"Round {round_num + 1} will be skipped."
                    )
                    current_training_images = []

            # Clean up memory after each round to prevent accumulation
            # Delete model object to free GPU/CPU memory
            del model
            if "results" in locals():
                del results

            # Aggressive memory cleanup
            self._cleanup_memory()
            self.logger.info(f"Round {round_num} completed, memory cleaned up")

        # After all rounds complete, evaluate final model on test_fixed
        self.logger.info(f"\n{'=' * 60}")
        self.logger.info("Evaluating final model on test set...")
        self.logger.info(f"{'=' * 60}\n")

        if best_checkpoint_path is None:
            raise ValueError("No best checkpoint found after training rounds")

        # Type assertion: best_checkpoint_path is guaranteed to be str here
        assert best_checkpoint_path is not None

        if not round_data_yamls:
            raise ValueError(
                "No round data YAMLs were created during training; cannot perform final evaluation."
            )

        # Use the highest round index for which a data.yaml was actually created
        last_round_with_yaml = max(round_data_yamls.keys())

        final_test_metrics = self.metrics_collector.evaluate_final_test(
            checkpoint_path=best_checkpoint_path,
            data_yaml=round_data_yamls[last_round_with_yaml],
            output_dir=output_dir,
            config_name=config_name,
            random_seed=seed,
            total_training_time=total_training_time,
        )

        # Aggregate round-level metrics
        self.logger.info("Aggregating round-level metrics...")
        self.round_metrics_tracker.aggregate_round_metrics(
            round_metrics_list=all_round_metrics, output_dir=output_dir, config_name=config_name
        )

        # Generate learning curves
        configs_data = {config_name: all_round_metrics}
        output_path = Path(output_dir) / "learning_curves.png"
        self.round_metrics_tracker.generate_learning_curves(
            configs_data=configs_data, output_path=str(output_path), metric="mAP50"
        )

        # Run one-shot baseline if configured
        if getattr(config.training, "run_baseline", False):
            baseline_output_dir = f"outputs/{config_name}_baseline"
            self.logger.info(f"{'=' * 60}")
            self.logger.info("Running one-shot baseline training")
            self.logger.info(f"Baseline output directory: {baseline_output_dir}")
            self.logger.info(f"{'=' * 60}")

            # Combine train_init + unlabeled_pool for the full training partition
            full_training_images = splits["train_init"] + splits["unlabeled_pool"]

            # Train baseline
            baseline_trainer = Baseline_Trainer()
            baseline_train_result = baseline_trainer.train(
                config=config,
                config_name=config_name,
                full_training_images=full_training_images,
                val_images=splits["val_fixed"],
                base_path=base_path,
                seed=seed,
            )

            # Evaluate baseline using the last round's data.yaml for val and test
            val_data_yaml = round_data_yamls[rounds]
            test_data_yaml = round_data_yamls[rounds]

            baseline_evaluator = Baseline_Evaluator()
            baseline_eval_result = baseline_evaluator.evaluate(
                checkpoint_path=baseline_train_result["checkpoint_path"],
                config_name=config_name,
                val_data_yaml=val_data_yaml,
                test_data_yaml=test_data_yaml,
                hfs_image_subset=splits["val_fixed"],
                base_path=base_path,
                seed=seed,
                training_time=baseline_train_result["training_time_seconds"],
                training_set_size=baseline_train_result["training_set_size"],
                epochs=baseline_train_result["epochs"],
                config=config,
            )

            # Flatten baseline metrics for comparison reporter
            baseline_test = baseline_eval_result["test_metrics"].get("test_metrics", {})
            baseline_flat = {
                "mAP50": baseline_test.get("mAP50", float("nan")),
                "mAP50-95": baseline_test.get("mAP50-95", float("nan")),
                "f1_score": baseline_test.get("f1_score", float("nan")),
                "hfs": baseline_eval_result["hfs_metrics"].get("mean_hfs", float("nan")),
                "training_time_seconds": baseline_train_result["training_time_seconds"],
            }

            # Flatten incremental metrics for comparison reporter
            incremental_metrics_inner = final_test_metrics.get("metrics", {})
            # Try to load HFS from the incremental output directory
            incremental_hfs = float("nan")
            incremental_hfs_file = Path(output_dir) / "hfs_metrics.json"
            if incremental_hfs_file.exists():
                try:
                    with open(incremental_hfs_file) as f:
                        hfs_data = json.load(f)
                    incremental_hfs = hfs_data.get("mean_hfs", float("nan"))
                except Exception as e:
                    self.logger.warning(f"Could not load incremental HFS metrics: {e}")

            incremental_flat = {
                "mAP50": incremental_metrics_inner.get("mAP50", float("nan")),
                "mAP50-95": incremental_metrics_inner.get("mAP50-95", float("nan")),
                "f1_score": incremental_metrics_inner.get("f1_score", float("nan")),
                "hfs": incremental_hfs,
                "training_time_seconds": final_test_metrics.get(
                    "total_training_time_seconds", float("nan")
                ),
            }

            # Generate comparison report
            comparison_reporter = Comparison_Reporter()
            comparison_reporter.generate_comparison(
                config_name=config_name,
                incremental_metrics=incremental_flat,
                baseline_metrics=baseline_flat,
            )

            self.logger.info("Baseline comparison complete. Reports saved to outputs/")

        return final_test_metrics

    def _verify_split_ratio(
        self,
        train_count: int,
        val_count: int,
        test_count: int,
        tolerance: float = 0.05,
    ) -> None:
        """
        Verify that the dataset train/valid/test split approximates 70/20/10.

        Logs a warning for any ratio that deviates from the expected value by
        more than *tolerance* (default 0.05).

        Args:
            train_count: Number of training images.
            val_count:   Number of validation images.
            test_count:  Number of test images.
            tolerance:   Maximum allowed deviation from expected ratio (default 0.05).
        """
        total = train_count + val_count + test_count
        if total == 0:
            self.logger.warning(
                "_verify_split_ratio: total image count is 0, skipping ratio check."
            )
            return

        actual_train = train_count / total
        actual_val = val_count / total
        actual_test = test_count / total

        expected = {"train": 0.70, "valid": 0.20, "test": 0.10}
        actuals = {"train": actual_train, "valid": actual_val, "test": actual_test}

        deviations = {split: abs(actuals[split] - expected[split]) for split in expected}

        if any(dev > tolerance for dev in deviations.values()):
            self.logger.warning(
                f"Dataset split ratio deviates from expected 70/20/10 "
                f"(tolerance={tolerance:.0%}): "
                f"train={actual_train:.1%} (expected 70%), "
                f"valid={actual_val:.1%} (expected 20%), "
                f"test={actual_test:.1%} (expected 10%). "
                f"Counts: train={train_count}, valid={val_count}, test={test_count}."
            )

    def _get_all_images(self, image_dir: Path) -> list[str]:
        """Get all image files from a directory, returning only filenames."""
        if not image_dir.exists():
            return []

        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
        images = []

        for ext in image_extensions:
            images.extend([p.name for p in image_dir.glob(f"*{ext}")])
            images.extend([p.name for p in image_dir.glob(f"*{ext.upper()}")])

        return sorted(images)

    def _create_round_data_yaml(
        self,
        original_data_yaml: str,
        round_data_yaml: str,
        training_images: list,
        val_images: list,
        test_images: list,
        base_path: Path,
    ) -> None:
        """Create data.yaml for a specific round with updated training set."""

        # Load original data.yaml
        with open(original_data_yaml) as f:
            data_config = yaml.safe_load(f)

        # Create temporary directories for this round's splits
        round_dir = Path(round_data_yaml).parent / "round_splits" / Path(round_data_yaml).stem
        round_dir.mkdir(parents=True, exist_ok=True)

        train_list = round_dir / "train.txt"
        val_list = round_dir / "val.txt"
        test_list = round_dir / "test.txt"

        # Write image lists (full paths constructed from filenames)
        train_image_dir = base_path / "train" / "images"
        val_image_dir = base_path / "valid" / "images"

        with open(train_list, "w") as f:
            full_paths = [str((train_image_dir / img).absolute()) for img in training_images]
            f.write("\n".join(full_paths))

        with open(val_list, "w") as f:
            if val_images:
                full_paths = [str((val_image_dir / img).absolute()) for img in val_images]
                f.write("\n".join(full_paths))
            else:
                # If no val images, use train images for validation
                full_paths = [str((train_image_dir / img).absolute()) for img in training_images]
                f.write("\n".join(full_paths))

        with open(test_list, "w") as f:
            if test_images:
                # Test images would be in test/images if they existed
                test_image_dir = base_path / "test" / "images"
                full_paths = [str((test_image_dir / img).absolute()) for img in test_images]
                f.write("\n".join(full_paths))
            else:
                # If no test images, use train images for testing
                full_paths = [str((train_image_dir / img).absolute()) for img in training_images]
                f.write("\n".join(full_paths))

        # Update data config with absolute paths
        data_config["train"] = str(train_list.absolute())
        data_config["val"] = str(val_list.absolute())
        data_config["test"] = str(test_list.absolute())

        # Save round-specific data.yaml
        with open(round_data_yaml, "w") as f:
            yaml.dump(data_config, f, default_flow_style=False)

        self.logger.debug(f"Created round data.yaml at {round_data_yaml}")

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
