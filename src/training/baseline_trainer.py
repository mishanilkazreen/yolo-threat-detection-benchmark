"""Baseline trainer for one-shot full-partition training."""

import logging
import math
from pathlib import Path
import sys
import time
from typing import Any

from ultralytics import YOLO
import yaml

from ..config.parser import Configuration
from ..utils.output_manager import Output_Manager
from .device_utils import create_step_decay_callback, get_model_gflops, select_device

logger = logging.getLogger(__name__)

# Prevent race condition when concurrent cluster jobs unlink labels.cache simultaneously
_original_path_unlink = Path.unlink


def _safe_path_unlink(self, missing_ok: bool = True) -> None:
    try:
        _original_path_unlink(self, missing_ok=missing_ok)
    except FileNotFoundError:
        pass


Path.unlink = _safe_path_unlink  # type: ignore[method-assign]


class Baseline_Trainer:  # pylint: disable=too-few-public-methods
    """
    Trains a YOLO model on the full training partition in a single pass.

    Combines train_init + unlabeled_pool into full_training_partition and
    trains for baseline_epochs using the same hyperparameters as the
    incremental experiment for a fair comparison.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.output_manager = Output_Manager()

    def train(
        self,
        config: Configuration,
        config_name: str,
        full_training_images: list[str],
        val_images: list[str],
        base_path: Path,
        seed: int,
    ) -> dict[str, Any]:
        """
        Train YOLO on full_training_partition in a single pass.

        Args:
            config: Experiment configuration (hyperparameters, model, data)
            config_name: Used to derive output directory names
            full_training_images: train_init + unlabeled_pool image filenames
            val_images: val_fixed image filenames (same as incremental)
            base_path: Dataset root path
            seed: Random seed (same as incremental run)

        Returns:
            dict with keys: checkpoint_path, training_time_seconds,
                            training_set_size, epochs
        """
        baseline_name = f"{config_name}_baseline"
        # Use Output_Manager for evaluation output path (for data.yaml and splits)
        output_dir = self.output_manager.get_evaluation_output_path(
            model_name=config_name, round_name=baseline_name, create=True
        )

        self.logger.info("Starting one-shot baseline training: %s", baseline_name)
        self.logger.info("  Full training partition size: %s images", len(full_training_images))
        self.logger.info("  Val set size: %s images", len(val_images))
        self.logger.info("  Epochs: %s", config.training.baseline_epochs)
        self.logger.info("  Seed: %s", seed)

        # Write data.yaml for baseline training
        data_yaml_path = Path(output_dir) / "data_baseline.yaml"
        self._write_data_yaml(
            original_data_yaml=config.data.yaml_path,
            output_yaml_path=data_yaml_path,
            training_images=full_training_images,
            val_images=val_images,
            base_path=base_path,
            output_dir=output_dir,
        )

        # Get hyperparameters (same as incremental)
        batch_size = getattr(config.training, "batch_size", 16) or 16
        optimizer = getattr(config.training, "optimizer", "AdamW") or "AdamW"
        lr0 = getattr(config.training, "lr0", 0.001) or 0.001
        lrf = getattr(config.training, "lrf", 0.1) or 0.1
        image_size = config.training.image_size
        baseline_epochs = config.training.baseline_epochs

        # Initialize YOLO from same weights as incremental Round 1
        self.logger.info("Initializing model from weights: %s", config.model.weights)
        model = YOLO(config.model.weights)

        # Reviewer 2 comment TASK-R2-02: Use verified AdamW linear LR schedule by default
        use_step_decay = getattr(config.training, "use_step_decay", False)
        if use_step_decay:
            step_decay_callback = create_step_decay_callback(lr0, lrf, step_interval=5)
            model.add_callback("on_train_epoch_start", step_decay_callback)
            self.logger.info(
                "Added step decay LR scheduler: lr0=%s, lrf=%s, step_interval=5", lr0, lrf
            )
        else:
            self.logger.info(
                "Using standard AdamW linear LR schedule: lr0=%s, lrf=%s (cos_lr=False)", lr0, lrf
            )

        device = select_device(config.training.device)

        self.logger.info(
            "Training baseline (epochs=%s, device=%s, optimizer=%s, lr0=%s, lrf=%s, batch=%s, seed=%s)",
            baseline_epochs,
            device,
            optimizer,
            lr0,
            lrf,
            batch_size,
            seed,
        )

        train_start = time.time()

        # Use Output_Manager to get YOLO project parameter
        yolo_project = self.output_manager.get_yolo_project_parameter(config_name)

        train_kwargs = {
            "data": str(data_yaml_path),
            "epochs": baseline_epochs,
            "imgsz": image_size,
            "batch": batch_size,
            "optimizer": optimizer,
            "lr0": lr0,
            "lrf": lrf,
            "cos_lr": False,
            "seed": seed,  # Explicitly enforce configured seed for Ultralytics (TASK-R2-03)
            "deterministic": True,  # Ensure deterministic cuDNN algorithms (TASK-MIN-03)
            "device": device,
            "project": yolo_project,
            "name": f"{config_name}/{baseline_name}",
            "exist_ok": True,
            "verbose": True,
            "patience": config.training.patience,
            "workers": 0 if sys.platform == "win32" else 4,
            "mosaic": getattr(config.training, "mosaic", 1.0),
            "scale": getattr(config.training, "scale", 0.5),
            "fliplr": getattr(config.training, "fliplr", 0.5),
            "hsv_h": getattr(config.training, "hsv_h", 0.015),
            "hsv_s": getattr(config.training, "hsv_s", 0.7),
            "hsv_v": getattr(config.training, "hsv_v", 0.4),
        }

        if getattr(config.training, "freeze", None) is not None:
            train_kwargs["freeze"] = config.training.freeze
            self.logger.info("  Applying freeze = %s layers", config.training.freeze)

        model.train(**train_kwargs)

        training_time = time.time() - train_start
        self.logger.info("Baseline training completed in %.2f seconds", training_time)

        # Extract trainer actual stopping epoch and best epoch (TASK-MAJ-12)
        actual_stopped_epoch = None
        best_epoch = None
        try:
            trainer = getattr(model, "trainer", None)
            if trainer is not None:
                raw_epoch = getattr(trainer, "epoch", None)
                raw_best = getattr(trainer, "best_epoch", None)
                if raw_epoch is not None:
                    actual_stopped_epoch = int(raw_epoch) + 1
                if raw_best is not None:
                    best_epoch = int(raw_best) + 1
                self.logger.info(
                    "Baseline epoch accounting: stopped at epoch %s/%s (best epoch: %s)",
                    actual_stopped_epoch,
                    baseline_epochs,
                    best_epoch,
                )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            self.logger.warning("Could not read stopping epoch from baseline trainer: %s", exc)

        effective_epochs = (
            actual_stopped_epoch if actual_stopped_epoch is not None else baseline_epochs
        )
        steps_per_epoch = math.ceil(len(full_training_images) / batch_size)
        total_optimizer_steps = steps_per_epoch * effective_epochs
        total_images_processed = len(full_training_images) * effective_epochs

        gflops = get_model_gflops(config.model.name)
        total_training_tflops = (3.0 * gflops * total_images_processed) / 1000.0

        # Use Output_Manager to resolve checkpoint path
        checkpoint_path = self.output_manager.resolve_checkpoint_path(
            model_name=config_name, round_name=f"{config_name}_baseline", checkpoint_type="best"
        )

        self.logger.info("Baseline checkpoint: %s", checkpoint_path)

        return {
            "checkpoint_path": checkpoint_path,
            "training_time_seconds": training_time,
            "training_set_size": len(full_training_images),
            "epochs": baseline_epochs,
            "actual_stopped_epoch": actual_stopped_epoch,
            "best_epoch": best_epoch,
            "patience": config.training.patience,
            "total_optimizer_steps": total_optimizer_steps,
            "total_images_processed": total_images_processed,
            "gflops": gflops,
            "total_training_tflops": total_training_tflops,
        }

    def _write_data_yaml(
        self,
        original_data_yaml: str,
        output_yaml_path: Path,
        training_images: list[str],
        val_images: list[str],
        base_path: Path,
        output_dir: Path,
    ) -> None:
        """Write a data.yaml pointing to the full training partition and val_fixed."""
        # Load original data.yaml for class names and metadata
        with open(original_data_yaml, encoding="utf-8") as f:
            data_config = yaml.safe_load(f)

        # Create split list directory
        split_dir = Path(output_dir) / "baseline_splits"
        split_dir.mkdir(parents=True, exist_ok=True)

        train_list = split_dir / "train.txt"
        val_list = split_dir / "val.txt"

        # training_images / val_images already contain absolute paths from
        # Dataset_Splitter (it re-partitions train/ + valid/ into 70/20/10 and
        # writes absolute paths because images may physically live in either
        # source dir). Write them straight through.
        train_image_dir = base_path / "train" / "images"

        def _resolve(name_or_path: str, fallback_dir: Path) -> str:
            p = Path(name_or_path)
            return str(p) if p.is_absolute() else str((fallback_dir / name_or_path).absolute())

        with open(train_list, "w", encoding="utf-8") as f:
            f.write("\n".join(_resolve(img, train_image_dir) for img in training_images))

        with open(val_list, "w", encoding="utf-8") as f:
            source = val_images if val_images else training_images
            f.write("\n".join(_resolve(img, train_image_dir) for img in source))

        # Update data config with list paths
        data_config["train"] = str(train_list.absolute())
        data_config["val"] = str(val_list.absolute())

        # Remove test key if present (not needed for baseline training)
        data_config.pop("test", None)

        with open(output_yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(data_config, f, default_flow_style=False)

        self.logger.debug("Wrote baseline data.yaml to %s", output_yaml_path)
