"""Baseline trainer for one-shot full-partition training."""

import logging
from pathlib import Path
import time
from typing import Any

from ultralytics import YOLO
import yaml

from ..config.parser import Configuration

logger = logging.getLogger(__name__)


class Baseline_Trainer:
    """
    Trains a YOLO model on the full training partition in a single pass.

    Combines train_init + unlabeled_pool into full_training_partition and
    trains for baseline_epochs using the same hyperparameters as the
    incremental experiment for a fair comparison.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        from ..utils.output_manager import Output_Manager

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

        self.logger.info(f"Starting one-shot baseline training: {baseline_name}")
        self.logger.info(f"  Full training partition size: {len(full_training_images)} images")
        self.logger.info(f"  Val set size: {len(val_images)} images")
        self.logger.info(f"  Epochs: {config.training.baseline_epochs}")
        self.logger.info(f"  Seed: {seed}")

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
        self.logger.info(f"Initializing model from weights: {config.model.weights}")
        model = YOLO(config.model.weights)

        # Add step decay LR callback (same as incremental runner)
        step_decay_callback = self._create_step_decay_callback(lr0, lrf, step_interval=5)
        model.add_callback("on_train_epoch_start", step_decay_callback)
        self.logger.info(f"Added step decay LR scheduler: lr0={lr0}, lrf={lrf}, step_interval=5")

        # Select device
        from .device_utils import select_device

        device = select_device(config.training.device)

        self.logger.info(
            f"Training baseline (epochs={baseline_epochs}, device={device}, "
            f"optimizer={optimizer}, lr0={lr0}, lrf={lrf}, batch={batch_size})"
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
        self.logger.info(f"Baseline training completed in {training_time:.2f} seconds")

        # Use Output_Manager to resolve checkpoint path
        checkpoint_path = self.output_manager.resolve_checkpoint_path(
            model_name=config_name, round_name=f"{config_name}_baseline", checkpoint_type="best"
        )

        self.logger.info(f"Baseline checkpoint: {checkpoint_path}")

        return {
            "checkpoint_path": checkpoint_path,
            "training_time_seconds": training_time,
            "training_set_size": len(full_training_images),
            "epochs": baseline_epochs,
        }

    def _create_step_decay_callback(self, lr0: float, lrf: float, step_interval: int = 5):
        """
        Create a callback for step decay learning rate schedule.

        Mirrors the implementation in Experiment_Runner to ensure identical
        LR schedules between baseline and incremental training.

        Args:
            lr0: Initial learning rate
            lrf: Decay factor (e.g., 0.1 means multiply by 0.1)
            step_interval: Number of epochs between decay steps (default: 5)

        Returns:
            Callback function compatible with Ultralytics YOLO
        """

        def on_train_epoch_start(trainer):
            epoch = trainer.epoch
            decay_steps = epoch // step_interval
            expected_lr = lr0 * (lrf**decay_steps)
            for param_group in trainer.optimizer.param_groups:
                param_group["lr"] = expected_lr

        return on_train_epoch_start

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
        with open(original_data_yaml) as f:
            data_config = yaml.safe_load(f)

        # Create split list directory
        split_dir = Path(output_dir) / "baseline_splits"
        split_dir.mkdir(parents=True, exist_ok=True)

        train_list = split_dir / "train.txt"
        val_list = split_dir / "val.txt"

        train_image_dir = base_path / "train" / "images"
        val_image_dir = base_path / "valid" / "images"

        # Write train image list
        with open(train_list, "w") as f:
            full_paths = [str((train_image_dir / img).absolute()) for img in training_images]
            f.write("\n".join(full_paths))

        # Write val image list
        with open(val_list, "w") as f:
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

        with open(output_yaml_path, "w") as f:
            yaml.dump(data_config, f, default_flow_style=False)

        self.logger.debug(f"Wrote baseline data.yaml to {output_yaml_path}")
