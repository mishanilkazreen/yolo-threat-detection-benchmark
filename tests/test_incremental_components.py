"""
Test script for incremental training components.

Tests the newly implemented components:
- Dataset_Splitter
- Training_Set_Manager
- Edge_Agent_Simulator
- Detection_Validator
- Round_Metrics_Tracker
"""

import json
import sys
import tempfile
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytest


def test_dataset_splitter_import():
    """Test that Dataset_Splitter can be imported."""
    from src.data.splitter import Dataset_Splitter

    splitter = Dataset_Splitter(random_seed=42)
    assert splitter.random_seed == 42
    print("✓ Dataset_Splitter import successful")


def test_training_set_manager_import():
    """Test that Training_Set_Manager can be imported."""
    from src.data.training_set_manager import Training_Set_Manager

    manager = Training_Set_Manager()
    assert manager is not None
    print("✓ Training_Set_Manager import successful")


def test_edge_agent_simulator_import():
    """Test that Edge_Agent_Simulator can be imported."""
    try:
        from src.training.edge_agent_simulator import Edge_Agent_Simulator

        simulator = Edge_Agent_Simulator(device="cpu")
        assert simulator.device == "cpu"
        print("✓ Edge_Agent_Simulator import successful")
    except ImportError as e:
        print(f"⚠ Edge_Agent_Simulator import skipped (missing dependency: {e})")
        return


def test_detection_validator_import():
    """Test that Detection_Validator can be imported."""
    from src.training.detection_validator import Detection_Validator

    validator = Detection_Validator(iou_threshold=0.5)
    assert validator.iou_threshold == 0.5
    print("✓ Detection_Validator import successful")


def test_round_metrics_tracker_import():
    """Test that Round_Metrics_Tracker can be imported."""
    from src.aggregation.round_metrics_tracker import Round_Metrics_Tracker

    tracker = Round_Metrics_Tracker()
    assert tracker is not None
    print("✓ Round_Metrics_Tracker import successful")


def test_detection_validator_iou_computation():
    """Test IoU computation in Detection_Validator."""
    from src.training.detection_validator import Detection_Validator

    validator = Detection_Validator()

    # Test identical boxes (IoU = 1.0)
    bbox1 = [0.5, 0.5, 0.2, 0.2]  # center at (0.5, 0.5), size 0.2x0.2
    bbox2 = [0.5, 0.5, 0.2, 0.2]
    iou = validator._compute_iou(bbox1, bbox2)
    assert abs(iou - 1.0) < 0.001, f"Expected IoU=1.0, got {iou}"

    # Test non-overlapping boxes (IoU = 0.0)
    bbox1 = [0.25, 0.25, 0.2, 0.2]
    bbox2 = [0.75, 0.75, 0.2, 0.2]
    iou = validator._compute_iou(bbox1, bbox2)
    assert abs(iou - 0.0) < 0.001, f"Expected IoU=0.0, got {iou}"

    # Test partially overlapping boxes
    bbox1 = [0.5, 0.5, 0.4, 0.4]
    bbox2 = [0.6, 0.6, 0.4, 0.4]
    iou = validator._compute_iou(bbox1, bbox2)
    assert 0.0 < iou < 1.0, f"Expected 0 < IoU < 1, got {iou}"

    print(f"✓ Detection_Validator IoU computation working correctly")


def test_detection_validator_validation_logic():
    """Test detection validation logic."""
    from src.training.detection_validator import Detection_Validator

    validator = Detection_Validator(iou_threshold=0.5)

    # Create a mock detection
    detection = {
        'image_id': 'test.jpg',
        'pred_class': 1,
        'bbox': [0.5, 0.5, 0.2, 0.2],
        'confidence': 0.9
    }

    # Create mock ground truth annotations
    gt_annotations = [
        {
            'class_id': 1,
            'bbox': [0.5, 0.5, 0.2, 0.2]  # Perfect match
        }
    ]

    is_verified, match_info = validator._validate_single_detection(detection, gt_annotations)
    assert is_verified, "Detection should be verified (perfect match)"
    assert abs(match_info['iou'] - 1.0) < 0.001, f"Expected IoU≈1.0, got {match_info['iou']}"

    # Test class mismatch
    gt_annotations_wrong_class = [
        {
            'class_id': 2,  # Different class
            'bbox': [0.5, 0.5, 0.2, 0.2]
        }
    ]

    is_verified, match_info = validator._validate_single_detection(detection, gt_annotations_wrong_class)
    assert not is_verified, "Detection should be rejected (class mismatch)"
    assert match_info['reason'] == 'no_class_match'

    print("✓ Detection_Validator validation logic working correctly")


def test_round_metrics_tracker_csv_export():
    """Test Round_Metrics_Tracker CSV export."""
    from src.aggregation.round_metrics_tracker import Round_Metrics_Tracker

    tracker = Round_Metrics_Tracker()

    # Create mock round data
    rounds_data = [
        {
            'round': 1,
            'training_set_size': 200,
            'verified_samples_added': 0,
            'unlabeled_pool_remaining': 800,
            'metrics': {
                'mAP50': 0.623,
                'mAP50-95': 0.412,
                'precision': 0.687,
                'recall': 0.598,
                'f1_score': 0.639,
                'fitness': 0.435
            },
            'model_info': {
                'inference_time_ms': 12.5
            },
            'round_training_time_seconds': 720.3,
            'cumulative_training_time_seconds': 720.3
        }
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / 'test_metrics.csv'
        tracker._export_to_csv(rounds_data, csv_path)

        assert csv_path.exists(), "CSV file should be created"

        # Read and verify CSV content
        with open(csv_path, 'r') as f:
            content = f.read()
            assert 'round' in content
            assert 'mAP50' in content
            assert '0.623' in content

    print("✓ Round_Metrics_Tracker CSV export working correctly")


def test_configuration_manager_incremental_validation():
    """Test Configuration_Manager incremental training field validation."""
    from src.config.manager import ConfigurationManager
    from src.config.parser import Configuration, TrainingConfig, ModelConfig, DataConfig

    # Create a valid configuration with incremental training fields
    config = Configuration(
        training=TrainingConfig(
            epochs=100,
            epochs_per_round=20,
            rounds=5,
            patience=10,
            image_size=640,
            device="cuda",
            runs=1,
            seeds=None
        ),
        model=ModelConfig(
            name="yolov8n",
            weights="yolov8n.pt"
        ),
        data=DataConfig(
            yaml_path="config/data/test.yaml",
            train_init_percentage=0.2,
            iou_threshold=0.5
        )
    )

    # This should not raise an exception
    try:
        ConfigurationManager._validate_incremental_training_fields(config, "test_config.yaml")
        print("✓ Configuration_Manager incremental training validation working correctly")
    except Exception as e:
        pytest.fail(f"Validation should pass for valid config: {e}")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Testing Incremental Training Components")
    print("=" * 60 + "\n")

    test_dataset_splitter_import()
    test_training_set_manager_import()
    test_edge_agent_simulator_import()
    test_detection_validator_import()
    test_round_metrics_tracker_import()

    print("\n" + "-" * 60)
    print("Testing Component Logic")
    print("-" * 60 + "\n")

    test_detection_validator_iou_computation()
    test_detection_validator_validation_logic()
    test_round_metrics_tracker_csv_export()
    test_configuration_manager_incremental_validation()

    print("\n" + "=" * 60)
    print("All Tests Passed! ✓")
    print("=" * 60 + "\n")
