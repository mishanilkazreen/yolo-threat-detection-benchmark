"""Baseline trainer for one-shot full-partition training."""

import logging
from pathlib import Path
import time
from typing import Any

from ultralytics import YOLO
import yaml

from ..config.parser import Configuration
from ..utils.output_manager import Output_Manager
from .device_utils import create_step_decay_callback, select_device

logger = logging.getLogger(__name__)


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

        # Add step decay LR callback (same as incremental runner)
        step_decay_callback = create_step_decay_callback(lr0, lrf, step_interval=5)
        model.add_callback("on_train_epoch_start", step_decay_callback)
        self.logger.info("Added step decay LR scheduler: lr0=%s, lrf=%s, step_interval=5", lr0, lrf)

        device = select_device(config.training.device)

        self.logger.info(
            "Training baseline (epochs=%s, device=%s, optimizer=%s, lr0=%s, lrf=%s, batch=%s)",
            baseline_epochs,
            device,
            optimizer,
            lr0,
            lrf,
            batch_size,
        )

        train_start = time.time()

        # Use Output_Manager to get YOLO project parameter
        yolo_project = self.output_manager.get_yolo_project_parameter(config_name)

        model.train(
            data=str(data_yaml_path),
            epochs=baseline_epochs,
            imgsz=image_size,
            batch=batch_size,
            optimizer=optimizer,
            lr0=lr0,
            lrf=lrf,
            cos_lr=False,
            seed=seed,
            device=device,
            project=yolo_project,
            name=f"{config_name}/{baseline_name}",
            exist_ok=True,
            verbose=True,
            mosaic=getattr(config.training, "mosaic", 1.0),
            scale=getattr(config.training, "scale", 0.5),
            fliplr=getattr(config.training, "fliplr", 0.5),
            hsv_h=getattr(config.training, "hsv_h", 0.015),
            hsv_s=getattr(config.training, "hsv_s", 0.7),
            hsv_v=getattr(config.training, "hsv_v", 0.4),
        )

        training_time = time.time() - train_start
        self.logger.info("Baseline training completed in %.2f seconds", training_time)

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

        train_image_dir = base_path / "train" / "images"
        val_image_dir = base_path / "valid" / "images"

        # Write train image list
        with open(train_list, "w", encoding="utf-8") as f:
            full_paths = [str((train_image_dir / img).absolute()) for img in training_images]
            f.write("\n".join(full_paths))

        # Write val image list
        with open(val_list, "w", encoding="utf-8") as f:
            if val_images:
                full_paths = [str((val_image_dir / img).absolute()) for img in val_images]
                f.write("\n".join(full_paths))
            else:
                # Fallback: use training images for validation
                full_paths = [str((train_image_dir / img).absolute()) for img in training_images]
                f.write("\n".join(full_paths))

        # Update data config with list paths
        data_config["train"] = str(train_list.absolute())
        data_config["val"] = str(val_list.absolute())

        # Remove test key if present (not needed for baseline training)
        data_config.pop("test", None)

        with open(output_yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(data_config, f, default_flow_style=False)

        self.logger.debug("Wrote baseline data.yaml to %s", output_yaml_path)
