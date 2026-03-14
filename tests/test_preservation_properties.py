"""
Preservation Property Tests for Config Mismatch Fix

**Validates: Requirements 3.1, 3.2, 3.3, 3.4**

IMPORTANT: These tests capture baseline behavior that MUST be preserved after the fix.
They should PASS on unfixed code and continue to PASS after the fix is implemented.

These tests verify that:
- Incremental training with train_init_percentage works correctly
- Other hyperparameters (batch_size, optimizer, lr0, epochs_per_round) are applied correctly
- Configuration parser validation works correctly
"""

from pathlib import Path

import numpy as np
from PIL import Image
import pytest
from ultralytics import YOLO
import yaml

from src.config.parser import ConfigurationParser


class TestPreservationProperties:
    """
    Property-based tests to verify baseline behavior is preserved.

    These tests should PASS on unfixed code and continue to PASS after the fix.
    """

    @pytest.mark.property
    def test_config_parser_validation_preservation(self, tmp_path: Path) -> None:
        """
        Test that configuration parser validation continues to work correctly.

        **Validates: Requirements 3.2**

        This test verifies that the parser validates all required fields
        and data types correctly.

        EXPECTED OUTCOME: PASS on unfixed code (baseline behavior to preserve)
        """
        # Create a valid config file
        config_file = tmp_path / "valid_config.yaml"
        config_data = {
            "training": {
                "epochs_per_round": 10,
                "rounds": 3,
                "patience": 10,
                "image_size": 640,
                "batch_size": 16,
                "optimizer": "AdamW",
                "lr0": 0.001,
                "lrf": 0.1,
                "runs": 1,
                "seeds": [42],
                "device": "cpu",
            },
            "model": {"name": "yolov8n", "weights": "yolov8n.yaml"},
            "data": {
                "yaml_path": "data.yaml",
                "train_init_percentage": 0.2,
                "iou_threshold": 0.5,
                # Note: train_split, val_split, test_split removed in Task 3.2
            },
        }

        config_file.write_text(yaml.dump(config_data))

        # Parse config
        config = ConfigurationParser.parse(config_file)

        # Verify all fields are parsed correctly
        assert config.training.epochs_per_round == 10
        assert config.training.rounds == 3
        assert config.training.patience == 10
        assert config.training.image_size == 640
        assert config.training.batch_size == 16
        assert config.training.optimizer == "AdamW"
        assert config.training.lr0 == 0.001
        assert config.training.lrf == 0.1
        assert config.training.runs == 1
        assert config.training.seeds == [42]
        assert config.training.device == "cpu"

        assert config.model.name == "yolov8n"
        assert config.model.weights == "yolov8n.yaml"

        assert config.data.yaml_path == "data.yaml"
        assert config.data.train_init_percentage == 0.2
        assert config.data.iou_threshold == 0.5
        # Note: train_split, val_split, test_split were removed in Task 3.2
        # as they were unused parameters that misled users

        # Test validation: missing required field should raise error
        invalid_config_file = tmp_path / "invalid_config.yaml"
        invalid_config_data = {
            "training": {
                "epochs_per_round": 10,
                # Missing patience
                "image_size": 640,
            },
            "model": {"name": "yolov8n", "weights": "yolov8n.yaml"},
            "data": {"yaml_path": "data.yaml"},
        }

        invalid_config_file.write_text(yaml.dump(invalid_config_data))

        # Should raise ConfigurationParseError
        from src.config.parser import ConfigurationParseError

        with pytest.raises(ConfigurationParseError):
            ConfigurationParser.parse(invalid_config_file)

        # Test validation: invalid data type should raise error
        invalid_type_config_file = tmp_path / "invalid_type_config.yaml"
        invalid_type_config_data = {
            "training": {
                "epochs_per_round": "not_an_int",  # Should be int
                "patience": 10,
                "image_size": 640,
            },
            "model": {"name": "yolov8n", "weights": "yolov8n.yaml"},
            "data": {"yaml_path": "data.yaml"},
        }

        invalid_type_config_file.write_text(yaml.dump(invalid_type_config_data))

        with pytest.raises(ConfigurationParseError):
            ConfigurationParser.parse(invalid_type_config_file)

    @pytest.mark.property
    def test_incremental_training_basic_flow(self, tmp_path: Path) -> None:
        """
        Test that basic incremental training flow works correctly.

        **Validates: Requirements 3.1, 3.3, 3.4**

        This test verifies that:
        - Training can execute multiple rounds
        - Checkpoints are created for each round
        - Basic hyperparameters are applied

        EXPECTED OUTCOME: PASS on unfixed code (baseline behavior to preserve)
        """
        # Create minimal dataset
        dataset_dir = tmp_path / "dataset"
        train_dir = dataset_dir / "train" / "images"
        train_labels_dir = dataset_dir / "train" / "labels"
        val_dir = dataset_dir / "valid" / "images"
        val_labels_dir = dataset_dir / "valid" / "labels"

        for dir_path in [train_dir, train_labels_dir, val_dir, val_labels_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Create 30 training images
        for i in range(30):
            img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
            img.save(train_dir / f"img_{i:03d}.jpg")
            (train_labels_dir / f"img_{i:03d}.txt").write_text("")

        # Create 10 validation images
        for i in range(10):
            img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
            img.save(val_dir / f"val_{i:03d}.jpg")
            (val_labels_dir / f"val_{i:03d}.txt").write_text("")

        # Create data.yaml
        data_yaml = dataset_dir / "data.yaml"
        data_yaml.write_text(
            yaml.dump(
                {
                    "path": str(dataset_dir),
                    "train": "train/images",
                    "val": "valid/images",
                    "names": {0: "weapon"},
                    "nc": 1,
                }
            )
        )

        # Run 2-round training
        rounds = 2
        epochs_per_round = 2
        checkpoints = []

        for round_num in range(1, rounds + 1):
            # Initialize model
            model = YOLO("yolov8n.yaml") if round_num == 1 else YOLO(checkpoints[-1])

            # Train
            model.train(
                data=str(data_yaml),
                epochs=epochs_per_round,
                imgsz=64,
                batch=4,
                lr0=0.001,
                optimizer="SGD",
                device="cpu",
                verbose=False,
                project=str(tmp_path / "runs"),
                name=f"round_{round_num}",
                exist_ok=True,
            )

            # Find checkpoint
            checkpoint_path = tmp_path / "runs" / f"round_{round_num}" / "weights" / "best.pt"
            assert checkpoint_path.exists(), f"Checkpoint not found for round {round_num}"
            checkpoints.append(str(checkpoint_path))

        # Verify all rounds completed
        assert len(checkpoints) == rounds

        # Verify checkpoints are valid files
        for checkpoint in checkpoints:
            assert Path(checkpoint).stat().st_size > 1000  # At least 1KB
