"""Standalone one-shot baseline experiment runner.

Orchestrates dataset splitting, baseline training, and evaluation on the same
70/20/10 split used by the incremental experiment so the two are directly
comparable at the same seed.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import yaml

from ..config.parser import Configuration, ConfigurationParser
from ..data.splitter import Dataset_Splitter
from ..data.validator import Dataset_Validator
from ..utils.output_manager import Output_Manager
from .baseline_evaluator import Baseline_Evaluator
from .baseline_trainer import Baseline_Trainer
from .seed_manager import Seed_Manager

logger = logging.getLogger(__name__)


class Baseline_Runner:
    """Run one-shot baseline training + evaluation as a standalone experiment.

    Flow:
      1. Parse config and validate dataset.
      2. Build the 70/20/10 split via Dataset_Splitter — yields train_init,
         unlabeled_pool, val_fixed, test_fixed (absolute paths).
      3. Train Baseline_Trainer on (train_init + unlabeled_pool) for
         config.training.baseline_epochs.
      4. Evaluate on val_fixed and test_fixed via Baseline_Evaluator
         (also computes HFS on val_fixed).
    """

    def __init__(self) -> None:
        self.logger = logger
        self.output_manager = Output_Manager()
        self.seed_manager = Seed_Manager()
        self.dataset_validator = Dataset_Validator()

    def run(
        self,
        config_path: str,
        validate_dataset: bool = True,
        seed: int | None = None,
        device: str | None = None,
        epochs: int | None = None,
        patience: int | None = None,
    ) -> dict[str, Any]:
        config = ConfigurationParser.parse(config_path)
        config_name = Path(config_path).stem

        # Apply overrides if provided (TASK-R2-03, TASK-MAJ-12)
        if seed is not None:
            config.training.seeds = [seed]
            self.logger.info("CLI override: seed=%d", seed)
        if device is not None:
            config.training.device = str(device)
            self.logger.info("CLI override: device=%s", device)
        if epochs is not None:
            config.training.baseline_epochs = epochs
            self.logger.info("CLI override: baseline_epochs=%d", epochs)
        if patience is not None:
            config.training.patience = patience
            self.logger.info("CLI override: patience=%d", patience)

        if validate_dataset:
            self._validate_dataset(config.data.yaml_path)

        seed = config.training.seeds[0] if config.training.seeds else 42
        self.seed_manager.set_seed(seed)

        # Use a shared outputs/{config_name}/ directory (mirrors incremental runner).
        output_dir = str(
            self.output_manager.get_evaluation_output_path(
                model_name=config_name, round_name=None, create=True
            ).parent
        )

        train_init_percentage = config.data.train_init_percentage or 0.2

        self.logger.info(
            "Splitting full image pool into 70/20/10 (train_init=%.0f%%, seed=%d)...",
            train_init_percentage * 100,
            seed,
        )

        splitter = Dataset_Splitter(random_seed=seed)
        split_result = splitter.create_incremental_splits(
            data_yaml_path=config.data.yaml_path,
            train_init_percentage=train_init_percentage,
            output_dir=output_dir,
        )

        sf = split_result["split_files"]
        splits = {
            key: self._load_lines(sf[key])
            for key in ("train_init", "unlabeled_pool", "val_fixed", "test_fixed")
        }
        self.logger.info("  train_init: %d images", len(splits["train_init"]))
        self.logger.info("  unlabeled_pool: %d images", len(splits["unlabeled_pool"]))
        self.logger.info("  val_fixed: %d images", len(splits["val_fixed"]))
        self.logger.info("  test_fixed: %d images", len(splits["test_fixed"]))

        with open(config.data.yaml_path, encoding="utf-8") as fh:
            data_yaml = yaml.safe_load(fh)
        base_path = Path(data_yaml.get("path", "."))

        # Full training partition = train_init + unlabeled_pool (everything labeled
        # available pre-incremental). One-shot training over this matches the
        # comparison the paper makes against the incremental schedule.
        full_training_images = splits["train_init"] + splits["unlabeled_pool"]

        self.logger.info("\n%s", "=" * 60)
        self.logger.info("Baseline training: %s", config_name)
        self.logger.info("  Full training set: %d images", len(full_training_images))
        self.logger.info("  Baseline epochs:   %d", config.training.baseline_epochs)
        self.logger.info("  Seed:              %d", seed)
        self.logger.info("%s\n", "=" * 60)

        trainer = Baseline_Trainer()
        train_result = trainer.train(
            config=config,
            config_name=config_name,
            full_training_images=full_training_images,
            val_images=splits["val_fixed"],
            base_path=base_path,
            seed=seed,
        )

        eval_yaml_path = Path(output_dir) / "data_baseline_eval.yaml"
        self._write_eval_data_yaml(
            original_data_yaml=config.data.yaml_path,
            output_yaml_path=eval_yaml_path,
            val_images=splits["val_fixed"],
            test_images=splits["test_fixed"],
            output_dir=Path(output_dir),
        )

        evaluator = Baseline_Evaluator(output_manager=self.output_manager)
        eval_result = evaluator.evaluate(
            checkpoint_path=str(train_result["checkpoint_path"]),
            config_name=config_name,
            val_data_yaml=str(eval_yaml_path),
            test_data_yaml=str(eval_yaml_path),
            hfs_image_subset=splits["val_fixed"],
            base_path=base_path,
            seed=seed,
            training_time=train_result["training_time_seconds"],
            training_set_size=train_result["training_set_size"],
            epochs=train_result["epochs"],
            config=config,
        )

        summary = {
            "config_name": config_name,
            "checkpoint_path": str(train_result["checkpoint_path"]),
            "training_time_seconds": train_result["training_time_seconds"],
            "training_set_size": train_result["training_set_size"],
            "epochs": train_result["epochs"],
            "actual_stopped_epoch": train_result.get("actual_stopped_epoch"),
            "best_epoch": train_result.get("best_epoch"),
            "patience": train_result.get("patience"),
            "total_optimizer_steps": train_result.get("total_optimizer_steps"),
            "total_images_processed": train_result.get("total_images_processed"),
            "gflops": train_result.get("gflops"),
            "total_training_tflops": train_result.get("total_training_tflops"),
            "val_metrics": eval_result["val_metrics"],
            "test_metrics": eval_result["test_metrics"],
            "hfs_metrics": eval_result["hfs_metrics"],
            "output_dir": eval_result["output_dir"],
        }

        # Save baseline summary to output directory (TASK-MAJ-12)
        summary_path = Path(eval_result["output_dir"]) / "baseline_summary.json"
        with open(summary_path, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
            fh.write("\n")
        self.logger.info("Saved baseline summary to %s", summary_path)

        return summary

    @staticmethod
    def _load_lines(path: str) -> list[str]:
        return [
            ln.strip() for ln in Path(path).read_text(encoding="utf-8").splitlines() if ln.strip()
        ]

    def _write_eval_data_yaml(
        self,
        original_data_yaml: str,
        output_yaml_path: Path,
        val_images: list[str],
        test_images: list[str],
        output_dir: Path,
    ) -> None:
        """Write a data.yaml that points val→val_fixed and test→test_fixed."""
        with open(original_data_yaml, encoding="utf-8") as f:
            data_config = yaml.safe_load(f)

        split_dir = output_dir / "baseline_eval_splits"
        split_dir.mkdir(parents=True, exist_ok=True)

        val_list = split_dir / "val.txt"
        test_list = split_dir / "test.txt"

        val_list.write_text("\n".join(val_images), encoding="utf-8")
        test_list.write_text("\n".join(test_images), encoding="utf-8")

        # train field required by YOLO; set it to val list as a harmless placeholder
        # (we only call model.val() with this YAML, never model.train()).
        data_config["train"] = str(val_list.absolute())
        data_config["val"] = str(val_list.absolute())
        data_config["test"] = str(test_list.absolute())

        with open(output_yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(data_config, f, default_flow_style=False)

    def _validate_dataset(self, data_yaml_path: str) -> None:
        with open(data_yaml_path, encoding="utf-8") as f:
            data_config = yaml.safe_load(f)
        base_path = Path(data_config.get("path", "."))
        result = self.dataset_validator.validate_dataset(data_config, base_path)
        if not result.is_valid:
            raise ValueError(
                f"Dataset validation failed: "
                f"{len(result.missing_labels)} missing labels, "
                f"{len(result.empty_labels)} empty labels, "
                f"{len(result.duplicate_files)} duplicates"
            )


__all__ = ["Baseline_Runner", "Configuration"]
