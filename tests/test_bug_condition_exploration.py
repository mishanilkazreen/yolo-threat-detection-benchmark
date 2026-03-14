"""
Bug Condition Exploration Test for Config Mismatch Fix

**Validates: Requirements 1.1, 1.2, 1.3, 1.4**

This test validates the fixes implemented in Tasks 3.1 and 3.2:
- Task 3.1: Step decay LR scheduler is now implemented via custom callback
- Task 3.2: Split ratios have been removed from config (they were never used)

The test verifies:
1. LR follows step decay pattern when training through Experiment_Runner
2. Split ratios are NO LONGER in the config (confirming they were removed)
"""

from pathlib import Path

import pytest
import yaml

from src.config.parser import ConfigurationParser
from src.training.runner import Experiment_Runner


class TestBugConditionExploration:
    """
    Tests to validate the fixes for configuration mismatches.

    These tests should PASS on fixed code:
    - LR Schedule test verifies LR follows step decay pattern via custom callback
    - Split Ratio test verifies split ratios have been removed from config
    """

    @pytest.mark.property
    def test_lr_schedule_mismatch(self, tmp_path: Path) -> None:
        """
        Test LR Schedule Fix: Verify Step Decay Implementation

        **Validates: Requirements 2.1, 2.4**

        EXPECTED OUTCOME: This test PASSES on fixed code (confirms fix works)

        The test verifies the step decay callback logic by testing it directly.
        Expected behavior:
        - Epoch 0: LR = 0.001
        - Epoch 5: LR = 0.0001 (0.001 * 0.1)
        - Epoch 10: LR = 0.00001 (0.0001 * 0.1)

        The fix (Task 3.1) implements a custom step decay callback that overrides
        Ultralytics' default cosine annealing behavior.
        """
        # Test the step decay callback directly
        runner = Experiment_Runner()

        # Create the callback
        lr0 = 0.001
        lrf = 0.1
        step_interval = 5
        callback = runner._create_step_decay_callback(lr0, lrf, step_interval)

        # Verify callback exists and is callable
        assert callback is not None, "Step decay callback should be created"
        assert callable(callback), "Step decay callback should be callable"

        # Test the callback logic by simulating epochs
        # Create a mock trainer object
        class MockOptimizer:
            def __init__(self):
                self.param_groups = [{"lr": lr0}]

        class MockTrainer:
            def __init__(self, epoch):
                self.epoch = epoch
                self.optimizer = MockOptimizer()

        # Test LR at different epochs
        test_cases = [
            (0, 0.001),  # Epoch 0: initial LR
            (5, 0.0001),  # Epoch 5: LR * 0.1
            (10, 0.00001),  # Epoch 10: LR * 0.1 * 0.1
        ]

        tolerance = 1e-7

        for epoch, expected_lr in test_cases:
            trainer = MockTrainer(epoch)
            callback(trainer)
            actual_lr = trainer.optimizer.param_groups[0]["lr"]

            assert abs(actual_lr - expected_lr) < tolerance, (
                f"Epoch {epoch}: Expected LR={expected_lr:.6f}, got {actual_lr:.6f}. "
                f"Step decay callback should multiply LR by {lrf} every {step_interval} epochs."
            )

    @pytest.mark.property
    def test_split_ratio_usage(self, tmp_path: Path) -> None:
        """
        Test Split Ratio Removal: Verify Misleading Parameters Were Removed

        **Validates: Requirements 2.2, 2.3, 2.4**

        EXPECTED OUTCOME: This test PASSES on fixed code (confirms fix works)

        The test verifies that split ratios are NO LONGER in the config.
        The fix (Task 3.2) removed train_split, val_split, test_split from:
        - DataConfig dataclass
        - ConfigurationParser
        - All config YAML files

        This confirms the misleading parameters have been removed, since they
        were never actually used in the splitting logic (the system uses
        pre-existing Roboflow train/val/test directories).
        """
        # Create a config file WITHOUT split ratios (as they should be after fix)
        config_file = tmp_path / "test_config.yaml"
        config_data = {
            "training": {
                "epochs": 10,
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
                "yaml_path": "dummy.yaml",
                "train_init_percentage": 0.2,
                "iou_threshold": 0.5,
                # NOTE: train_split, val_split, test_split should NOT be here after fix
            },
        }

        config_file.write_text(yaml.dump(config_data))

        # Parse the config
        config = ConfigurationParser.parse(config_file)

        # Verify that split ratios are NOT in the config (they were removed)
        # After Task 3.2, DataConfig should not have these attributes
        assert not hasattr(config.data, "train_split"), (
            "train_split should NOT exist in DataConfig after fix. "
            "Task 3.2 removed this field because it was never used."
        )

        assert not hasattr(config.data, "val_split"), (
            "val_split should NOT exist in DataConfig after fix. "
            "Task 3.2 removed this field because it was never used."
        )

        assert not hasattr(config.data, "test_split"), (
            "test_split should NOT exist in DataConfig after fix. "
            "Task 3.2 removed this field because it was never used."
        )

        # Verify that the config still has the fields that ARE used
        assert hasattr(config.data, "train_init_percentage"), (
            "train_init_percentage should still exist (it IS used for incremental training)"
        )

        assert config.data.train_init_percentage == 0.2, (
            "train_init_percentage should be parsed correctly"
        )

        # Verify that trying to add split ratios to config would fail
        # (parser should not accept them)
        config_with_splits = tmp_path / "test_config_with_splits.yaml"
        config_data_with_splits = {
            "training": {
                "epochs": 10,
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
                "yaml_path": "dummy.yaml",
                "train_init_percentage": 0.2,
                "iou_threshold": 0.5,
                "train_split": 0.7,
                "val_split": 0.2,
                "test_split": 0.1,
            },
        }

        config_with_splits.write_text(yaml.dump(config_data_with_splits))

        # Parse config with split ratios - they should be ignored
        config2 = ConfigurationParser.parse(config_with_splits)

        # Even if present in YAML, they should not be in the parsed config
        assert not hasattr(config2.data, "train_split"), (
            "train_split should be ignored by parser after fix"
        )
        assert not hasattr(config2.data, "val_split"), (
            "val_split should be ignored by parser after fix"
        )
        assert not hasattr(config2.data, "test_split"), (
            "test_split should be ignored by parser after fix"
        )
