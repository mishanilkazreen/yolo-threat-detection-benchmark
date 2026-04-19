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
from ..explainability.xai.manager import XAIManager
from ..utils.output_manager import Output_Manager
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
        self.output_manager = Output_Manager()

        # Import incremental training components lazily to avoid circular imports
        self._splitter = None
        self._training_set_manager = None
        self._edge_agent_simulator = None
        self._detection_validator = None
        self._round_metrics_tracker = None

        # XAI Manager - initialized when needed
        self._xai_manager: XAIManager | None = None

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

    def _get_xai_manager(self, config: Any) -> XAIManager | None:
        """Get XAI manager if XAI is enabled, otherwise return None."""
        if not hasattr(config, "xai") or not config.xai.enabled:
            return None

        if self._xai_manager is None:
            self._xai_manager = XAIManager(config.xai, output_manager=self.output_manager)

        return self._xai_manager

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

        # Create output directories using Output_Manager
        output_dir = str(
            self.output_manager.get_evaluation_output_path(
                model_name=config_name, round_name="train", create=True
            )
        )
        explanations_dir = str(
            self.output_manager.get_explainability_output_path(
                model_name=config_name, round_name="train", create=True
            )
        )

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

        # Train model using Output_Manager for project parameter
        self.logger.info(f"Starting training (seed={seed}, device={device})...")
        train_start = time.time()

        results = model.train(
            data=config.data.yaml_path,
            epochs=config.training.epochs,
            imgsz=config.training.image_size,
            patience=config.training.patience,
            device=device,
            project=self.output_manager.get_yolo_project_parameter(config_name),
            name=f"{config_name}/train",
            exist_ok=True,
            verbose=True,
            lr0=lr0,
            lrf=lrf,
            cos_lr=False,  # Disable cosine LR scheduler to use our custom step decay
        )

        training_time = time.time() - train_start
        self.logger.info(f"Training completed in {training_time:.2f} seconds")

        # Get the actual project directory from Output_Manager
        project_dir = str(
            self.output_manager.get_training_output_path(
                model_name=config_name, round_name="train", create=False
            )
        )

        # Verify checkpoint exists
        weights_dir = Path(project_dir) / "weights"
        if not weights_dir.exists() or not (weights_dir / "best.pt").exists():
            raise FileNotFoundError(f"Could not find training results at: {project_dir}")

        self.logger.info(f"Found training results at: {project_dir}")

        # Evaluate model
        self.logger.info("Evaluating model...")
        metrics = self.metrics_collector.evaluate_and_save(
            project_dir=project_dir,
            config_name=config_name,
            data_yaml=config.data.yaml_path,
            output_dir=output_dir,
            run_id=run_id,
            random_seed=seed,
            training_time=training_time,
        )

        # Run XAI processing if enabled
        xai_manager = self._get_xai_manager(config)
        if xai_manager is not None:
            self.logger.info("Running XAI processing...")
            self._run_xai_processing(
                xai_manager=xai_manager,
                model_path=str(Path(project_dir) / "weights" / "best.pt"),
                config=config,
                output_dir=output_dir,
                round_num=1,  # Standard training is considered round 1
                model_name=config_name,
                round_name="train",
            )

        return metrics

    def _run_incremental_training(
        self, config: Any, config_name: str, seed: int, run_id: int | None = None
    ) -> dict[str, Any]:
        """Run 5-round incremental training with edge-cloud simulation.

        Trains cumulatively — each round trains on all verified samples accumulated so far.

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
            "ARCHITECTURE: Cumulative training — each round trains on all verified samples accumulated so far"
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

        # Create output directories using Output_Manager
        # Use config_name as model_name for the base directory structure
        output_dir = str(
            self.output_manager.get_evaluation_output_path(
                model_name=config_name,
                round_name=None,  # Will create outputs/{config_name}/ for shared files
                create=True,
            ).parent
        )  # Get parent to have outputs/{config_name}/ for shared metadata

        explanations_dir = str(
            self.output_manager.get_explainability_output_path(
                model_name=config_name, round_name=None, create=True
            ).parent
        )  # Get parent to have explanations/{config_name}/ base

        # Load data.yaml to get dataset paths
        with open(config.data.yaml_path) as f:
            data_config = yaml.safe_load(f)

        base_path = Path(data_config.get("path", "."))
        train_path = base_path / data_config["train"]

        # Step 1: Build the 70/20/10 split from the full image pool using
        # Dataset_Splitter.  The Roboflow download has no test directory, so
        # the splitter pools train/ + valid/ and re-partitions from scratch.
        self.logger.info(
            f"Splitting full image pool into 70/20/10 "
            f"(train_init={train_init_percentage * 100:.0f}%, seed={seed})..."
        )

        split_result = self.splitter.create_incremental_splits(
            data_yaml_path=config.data.yaml_path,
            train_init_percentage=train_init_percentage,
            output_dir=output_dir,
        )

        def _load_txt(path: str) -> list[str]:
            return [ln.strip() for ln in Path(path).read_text().splitlines() if ln.strip()]

        sf = split_result["split_files"]
        splits = {
            "train_init": _load_txt(sf["train_init"]),
            "unlabeled_pool": _load_txt(sf["unlabeled_pool"]),
            "val_fixed": _load_txt(sf["val_fixed"]),
            "test_fixed": _load_txt(sf["test_fixed"]),
        }

        self.logger.info(f"  train_init: {len(splits['train_init'])} images")
        self.logger.info(f"  unlabeled_pool: {len(splits['unlabeled_pool'])} images")
        self.logger.info(f"  val_fixed: {len(splits['val_fixed'])} images")
        self.logger.info(f"  test_fixed: {len(splits['test_fixed'])} images")

        # Verify that the split approximates the expected 70/20/10 ratio
        self._verify_split_ratio(
            train_count=len(splits["train_init"]) + len(splits["unlabeled_pool"]),
            val_count=len(splits["val_fixed"]),
            test_count=len(splits["test_fixed"]),
        )

        # Track data.yaml files for each round (populated as rounds execute)
        round_data_yamls: dict[int, str] = {}
        round_data_yaml_paths: dict[int, str] = {}
        for round_num in range(1, rounds + 1):
            round_data_yaml = Path(output_dir) / f"data_round_{round_num}.yaml"
            round_data_yaml_paths[round_num] = str(round_data_yaml)

        # Initialize training set with train_init
        current_training_images = splits["train_init"].copy()
        unlabeled_pool = splits["unlabeled_pool"].copy()

        # Track metrics across rounds
        all_round_metrics = []
        best_checkpoint_path = None
        best_overall_checkpoint_path = None
        best_overall_map50 = -1.0
        best_overall_round = -1
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

            # Create round-specific directories using Output_Manager
            round_name = f"incremental_round_{round_num}"

            round_project_dir = str(
                self.output_manager.get_training_output_path(
                    model_name=config_name, round_name=round_name, create=True
                )
            )

            round_explanations_dir = str(
                self.output_manager.get_explainability_output_path(
                    model_name=config_name, round_name=round_name, create=True
                )
            )

            # Create data.yaml for this round with current training set
            self._create_round_data_yaml(
                original_data_yaml=config.data.yaml_path,
                round_data_yaml=round_data_yaml_paths[round_num],
                training_images=current_training_images,
                val_images=splits["val_fixed"],
                test_images=splits["test_fixed"],
                base_path=base_path,
            )
            # Register this round's data.yaml as successfully created
            round_data_yamls[round_num] = round_data_yaml_paths[round_num]

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
                project=self.output_manager.get_yolo_project_parameter(config_name),
                name=f"{config_name}/incremental_round_{round_num}",
                exist_ok=True,
                verbose=True,
                mosaic=config.training.mosaic,
                scale=config.training.scale,
                fliplr=config.training.fliplr,
                hsv_h=config.training.hsv_h,
                hsv_s=config.training.hsv_s,
                hsv_v=config.training.hsv_v,
            )

            round_training_time = time.time() - round_train_start
            total_training_time += round_training_time
            self.logger.info(
                f"Round {round_num} training completed in {round_training_time:.2f} seconds"
            )

            # Find best checkpoint for this round using Output_Manager
            round_name = f"incremental_round_{round_num}"
            try:
                best_checkpoint_path = str(
                    self.output_manager.resolve_checkpoint_path(
                        model_name=config_name, round_name=round_name, checkpoint_type="best"
                    )
                )
                self.logger.info(f"Found checkpoint at: {best_checkpoint_path}")
            except FileNotFoundError as e:
                self.logger.error(f"Checkpoint resolution failed: {e}")
                raise

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
            total_detections = None
            remaining_pool_size = len(unlabeled_pool) if unlabeled_pool else None
            unlabeled_pool_size_at_round_start = len(unlabeled_pool) if unlabeled_pool else None

            if round_num > 1:
                # Load validation results from previous round (file already opened above for verified count)
                prev_validation_file = Path(output_dir) / f"round_{round_num - 1}_validations.json"
                if prev_validation_file.exists():
                    try:
                        with open(prev_validation_file) as f:
                            prev_validation = json.load(f)
                            rejected_count = prev_validation.get("rejected_count")
                            undetected_count = prev_validation.get("undetected_count")
                            total_detections = prev_validation.get("total_detections")
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
                unlabeled_pool_size_at_round_start=unlabeled_pool_size_at_round_start,
                total_detections=total_detections,
            )

            all_round_metrics.append(round_metrics)

            round_map50 = round_metrics["metrics"]["mAP50"]
            if round_map50 > best_overall_map50:
                best_overall_map50 = round_map50
                best_overall_checkpoint_path = best_checkpoint_path
                best_overall_round = round_num
                self.logger.info(
                    f"New best overall checkpoint: Round {round_num} (val mAP50={round_map50:.4f})"
                )

            # Run XAI processing if enabled
            xai_manager = self._get_xai_manager(config)
            if xai_manager is not None:
                self.logger.info(f"Running XAI processing for Round {round_num}...")
                self._run_xai_processing(
                    xai_manager=xai_manager,
                    model_path=best_checkpoint_path,
                    config=config,
                    output_dir=output_dir,
                    round_num=round_num,
                    model_name=config_name,
                    round_name=f"incremental_round_{round_num}",
                )

            # Round 5 special case: add all remaining pool images unconditionally
            if round_num == rounds and len(unlabeled_pool) > 0:
                self.training_set_manager.add_verified_samples(
                    current_training_images,
                    unlabeled_pool.copy(),
                    unlabeled_pool,
                    output_dir,
                    round_num,
                )
                self.logger.info(
                    f"Round 5: added all {len(unlabeled_pool)} remaining pool images unconditionally"
                )

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

                # For Round N+1, add verified images to the cumulative training set.
                if len(verified_images) > 0:
                    # Add verified images to the cumulative training set and remove them
                    # from the unlabeled pool (both lists modified in-place).
                    self.training_set_manager.add_verified_samples(
                        current_training_images,
                        verified_images,
                        unlabeled_pool,
                        output_dir,
                        round_num + 1,
                    )

                    self.logger.info(
                        f"Next round will train on {len(current_training_images)} cumulative images"
                    )
                else:
                    self.logger.warning(
                        f"No verified samples found in Round {round_num}. "
                        f"Round {round_num + 1} will train on the same cumulative set "
                        f"({len(current_training_images)} images)."
                    )

            # Clean up memory after each round to prevent accumulation
            # Delete model object to free GPU/CPU memory
            del model
            if "results" in locals():
                del results

            # Aggressive memory cleanup
            self._cleanup_memory()
            self.logger.info(f"Round {round_num} completed, memory cleaned up")

        # After all rounds complete, evaluate final model on test set using the
        # best checkpoint across all rounds (by val mAP50), not the last round's.
        final_checkpoint = best_overall_checkpoint_path or best_checkpoint_path

        self.logger.info(f"\n{'=' * 60}")
        self.logger.info(
            f"Evaluating final model on test set (best checkpoint from Round {best_overall_round}, "
            f"val mAP50={best_overall_map50:.4f})..."
        )
        self.logger.info(f"{'=' * 60}\n")

        if final_checkpoint is None:
            raise ValueError("No best checkpoint found after training rounds")

        if not round_data_yamls:
            raise ValueError(
                "No round data YAMLs were created during training; cannot perform final evaluation."
            )

        last_round_with_yaml = max(round_data_yamls.keys())

        final_test_metrics = self.metrics_collector.evaluate_final_test(
            checkpoint_path=final_checkpoint,
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
            baseline_output_dir = str(
                self.output_manager.get_evaluation_output_path(
                    model_name=f"{config_name}_baseline", round_name="train", create=True
                ).parent
            )  # Get parent to have outputs/{config_name}_baseline/

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
            last_round_key = max(round_data_yamls.keys())
            val_data_yaml = round_data_yamls[last_round_key]
            test_data_yaml = round_data_yamls[last_round_key]

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
        """Get all image files from a directory, returning only filenames.

        Deduplicates case-insensitively so that on Windows NTFS (where
        ``*.jpg`` and ``*.JPG`` globs return the same physical files) each
        file appears exactly once.
        """
        if not image_dir.exists():
            return []

        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
        seen: set[str] = set()
        images: list[str] = []

        for p in sorted(image_dir.iterdir()):
            if p.suffix.lower() in image_extensions:
                key = p.name.lower()
                if key not in seen:
                    seen.add(key)
                    images.append(p.name)

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

        # training_images / val_images / test_images already contain absolute paths
        # written by Dataset_Splitter.  Write them straight through — do NOT
        # prepend a hardcoded directory, because images may physically live in
        # either train/images or valid/images depending on the 70/20/10 split.
        with open(train_list, "w") as f:
            f.write("\n".join(str(p) for p in training_images))

        with open(val_list, "w") as f:
            if val_images:
                f.write("\n".join(str(p) for p in val_images))
            else:
                f.write("\n".join(str(p) for p in training_images))

        with open(test_list, "w") as f:
            if test_images:
                f.write("\n".join(str(p) for p in test_images))
            else:
                f.write("\n".join(str(p) for p in val_images))

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

        # Save aggregated results using Output_Manager
        output_path = (
            self.output_manager.get_evaluation_output_path(
                model_name=config_name, round_name="train", create=True
            )
            / "aggregated_results.json"
        )

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

    def _run_xai_processing(
        self,
        xai_manager: XAIManager,
        model_path: str,
        config: Any,
        output_dir: str,
        round_num: int,
        model_name: str | None = None,
        round_name: str | None = None,
    ) -> None:
        """
        Run XAI processing on validation images.

        Args:
            xai_manager: Initialized XAI manager
            model_path: Path to trained model checkpoint
            config: Configuration object
            output_dir: Output directory for XAI results (deprecated, use model_name/round_name)
            round_num: Training round number
            model_name: Model identifier for Output_Manager path construction
            round_name: Round identifier for Output_Manager path construction
        """
        try:
            # Load the trained model
            model = YOLO(model_path)

            # Get validation images from data.yaml
            validation_images = self._get_validation_images(config.data.yaml_path)

            if not validation_images:
                self.logger.warning("No validation images found for XAI processing")
                return

            # Apply sample limit if configured
            if config.xai.sample_limit is not None:
                validation_images = validation_images[: config.xai.sample_limit]
                self.logger.info(f"Limited XAI processing to {len(validation_images)} images")

            # Run inference on validation images to get detections
            self.logger.info(f"Running inference on {len(validation_images)} validation images...")
            detections = {}
            gt_boxes = {}

            # Load data.yaml to get paths
            with open(config.data.yaml_path) as f:
                data_config = yaml.safe_load(f)
            base_path = Path(data_config.get("path", "."))
            val_image_dir = base_path / "valid" / "images"
            val_labels_dir = base_path / "valid" / "labels"

            for image_name in validation_images:
                image_path = str(val_image_dir / image_name)

                # Run inference
                results = model(image_path, verbose=False)
                if results and len(results) > 0:
                    detections[image_path] = {
                        "boxes": results[0].boxes.xyxy.cpu().numpy()
                        if results[0].boxes is not None
                        else [],
                        "scores": results[0].boxes.conf.cpu().numpy()
                        if results[0].boxes is not None
                        else [],
                        "classes": results[0].boxes.cls.cpu().numpy()
                        if results[0].boxes is not None
                        else [],
                    }
                else:
                    detections[image_path] = {"boxes": [], "scores": [], "classes": []}

                # Load ground truth boxes
                label_path = val_labels_dir / f"{Path(image_name).stem}.txt"
                gt_boxes_list = []
                if label_path.exists():
                    try:
                        with open(label_path) as f:
                            for line in f:
                                parts = line.strip().split()
                                if len(parts) >= 5:
                                    # YOLO format: class x_center y_center width height (normalized)
                                    _, x_center, y_center, width, height = map(float, parts[:5])
                                    gt_boxes_list.append((x_center, y_center, width, height))
                    except Exception as e:
                        self.logger.warning(f"Failed to load ground truth for {image_name}: {e}")

                gt_boxes[image_path] = gt_boxes_list

            # Process with XAI manager
            image_paths = [str(val_image_dir / img) for img in validation_images]
            xai_results = xai_manager.process_batch(
                model=model,
                image_paths=image_paths,
                detections=detections,
                gt_boxes=gt_boxes,
                round_num=round_num,
                output_dir=output_dir,
                validation_images=image_paths,  # For SHAP background set (use full paths)
                model_name=model_name,
                round_name=round_name,
            )

            if xai_results:
                self.logger.info(
                    f"XAI processing completed: {xai_results.num_images_processed} images processed, "
                    f"{len(xai_results.failed_images)} failures"
                )

                # Log aggregate HFS scores
                if xai_results.aggregate_hfs:
                    for method, hfs in xai_results.aggregate_hfs.items():
                        self.logger.info(f"  {method.upper()} mean HFS: {hfs:.4f}")
            else:
                self.logger.info("XAI processing completed (no results generated)")

        except Exception as e:
            self.logger.error(f"XAI processing failed: {e}")
            # Don't raise - XAI failure shouldn't stop the main pipeline

    def _get_validation_images(self, data_yaml_path: str) -> list[str]:
        """
        Get list of validation image filenames from data.yaml.

        Args:
            data_yaml_path: Path to data.yaml file

        Returns:
            List of validation image filenames
        """
        try:
            with open(data_yaml_path) as f:
                data_config = yaml.safe_load(f)

            base_path = Path(data_config.get("path", "."))
            val_path = base_path / "valid" / "images"

            if not val_path.exists():
                return []

            # Get all image files (avoid duplicates from case-insensitive search)
            image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
            images: set[str] = set()  # Use set to avoid duplicates

            for ext in image_extensions:
                images.update(p.name for p in val_path.glob(f"*{ext}"))
                images.update(p.name for p in val_path.glob(f"*{ext.upper()}"))

            return sorted(images)

        except Exception as e:
            self.logger.error(f"Failed to get validation images: {e}")
            return []
