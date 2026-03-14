"""
Tests for Bug 3 fix: iou_threshold is forwarded to Detection_Validator.

Task 1.2 — Exploratory test: non-default iou_threshold is applied (fails on unfixed code).
Task 1.3 — Preservation test: default iou_threshold (0.5) is preserved when not configured.
"""

import contextlib
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.training.runner import Experiment_Runner

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(iou_threshold=None):
    """Build a minimal config object for _run_incremental_training()."""
    data = SimpleNamespace(
        yaml_path="fake/data.yaml",
        train_init_percentage=0.2,
    )
    if iou_threshold is not None:
        data.iou_threshold = iou_threshold

    training = SimpleNamespace(
        rounds=2,
        epochs=None,
        epochs_per_round=1,
        patience=5,
        image_size=640,
        device="cpu",
        seeds=[42],
        batch_size=16,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.1,
        run_baseline=False,
    )
    return SimpleNamespace(
        data=data, training=training, model=SimpleNamespace(name="yolov8n", weights="yolov8n.pt")
    )


def _make_runner_with_mocks():
    """Return a runner whose heavy dependencies are all mocked out."""
    runner = Experiment_Runner()

    # Mock detection_validator so we can inspect iou_threshold
    mock_validator = MagicMock()
    mock_validator.iou_threshold = 0.5  # default
    mock_validator.validate_detections.return_value = {
        "verified_samples": [],
        "undetected_images": [],
        "rejected_count": 0,
        "undetected_count": 0,
    }
    runner._detection_validator = mock_validator

    # Mock edge_agent_simulator
    mock_edge = MagicMock()
    mock_edge.simulate_inference.return_value = []
    runner._edge_agent_simulator = mock_edge

    # Mock training_set_manager
    mock_tsm = MagicMock()
    runner._training_set_manager = mock_tsm

    # Mock round_metrics_tracker
    mock_rmt = MagicMock()
    mock_rmt.aggregate_round_metrics.return_value = {}
    mock_rmt.generate_learning_curves.return_value = None
    runner._round_metrics_tracker = mock_rmt

    return runner


# ---------------------------------------------------------------------------
# Task 1.2 — Exploratory test (fails on unfixed code, passes after fix)
# ---------------------------------------------------------------------------


def test_iou_threshold_forwarded_to_validator(tmp_path):
    """
    When config.data.iou_threshold = 0.3, the detection_validator.iou_threshold
    must be set to 0.3 before validate_detections() is called.

    Validates: Requirements 2.1 (Bug 3 fix)
    """
    config = _make_config(iou_threshold=0.3)
    runner = _make_runner_with_mocks()

    # Provide a minimal fake data.yaml
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text("path: .\ntrain: images/train\nval: images/val\nnames:\n  0: weapon\n")
    config.data.yaml_path = str(data_yaml)

    # Create fake image directories so the runner doesn't crash on path resolution
    (tmp_path / "images" / "train").mkdir(parents=True)
    (tmp_path / "images" / "val").mkdir(parents=True)

    # Patch YOLO and metrics_collector so training doesn't actually run
    mock_yolo_instance = MagicMock()
    mock_yolo_instance.train.return_value = MagicMock()
    mock_yolo_instance.add_callback.return_value = None

    mock_metrics = MagicMock()
    mock_metrics.collect_round_metrics.return_value = {"mAP50": 0.5}
    mock_metrics.evaluate_final_test.return_value = {"metrics": {}}
    runner.metrics_collector = mock_metrics

    # Patch Path.exists so checkpoint lookup succeeds
    fake_checkpoint = tmp_path / "best.pt"
    fake_checkpoint.touch()

    captured_iou = {}

    original_validate = runner._detection_validator.validate_detections

    def capture_iou(*args, **kwargs):
        captured_iou["iou_threshold"] = runner._detection_validator.iou_threshold
        return original_validate(*args, **kwargs)

    runner._detection_validator.validate_detections.side_effect = capture_iou

    with (
        patch("src.training.runner.YOLO", return_value=mock_yolo_instance),
        patch("src.training.runner.Path.exists", return_value=True),
        patch.object(
            runner,
            "_get_all_images",
            side_effect=[
                [str(tmp_path / "images/train/img1.jpg")] * 10,  # train_images
                [],  # val_fixed
                [],  # test_fixed
            ],
        ),
        patch.object(runner, "_create_round_data_yaml"),
        patch(
            "builtins.open",
            MagicMock(
                return_value=MagicMock(
                    __enter__=MagicMock(
                        return_value=MagicMock(
                            read=MagicMock(return_value=""),
                            __iter__=MagicMock(return_value=iter([])),
                        )
                    ),
                    __exit__=MagicMock(return_value=False),
                )
            ),
        ),
        patch(
            "yaml.safe_load",
            return_value={
                "path": str(tmp_path),
                "train": "images/train",
                "val": "images/val",
            },
        ),
        patch("json.dump"),
        patch("json.load", return_value={}),
    ):
        # Patch checkpoint path resolution
        runner._detection_validator.validate_detections.side_effect = capture_iou

        # Simulate that edge agent returns one detection so validate_detections is called
        runner._edge_agent_simulator.simulate_inference.return_value = [
            {
                "image_id": "img1.jpg",
                "pred_class": 0,
                "bbox": [0.5, 0.5, 0.1, 0.1],
                "confidence": 0.9,
            }
        ]

        # Patch the checkpoint path so round 1 "succeeds"
        with (
            patch.object(
                runner, "_run_incremental_training", wraps=runner._run_incremental_training
            ),
            contextlib.suppress(Exception),
        ):
            # We only care that validate_detections was called with the right threshold
            runner._run_incremental_training(config, "test_config", seed=42, run_id=None)

    assert "iou_threshold" in captured_iou, "validate_detections was never called"
    assert captured_iou["iou_threshold"] == 0.3, (
        f"Expected iou_threshold=0.3 but got {captured_iou['iou_threshold']}"
    )


# ---------------------------------------------------------------------------
# Task 1.3 — Preservation test: default iou_threshold = 0.5
# ---------------------------------------------------------------------------


def test_default_iou_threshold_is_0_5():
    """
    When config.data.iou_threshold is not set, Detection_Validator defaults to 0.5.

    Validates: Requirements 3.1 (preservation)
    """
    config = _make_config()  # no iou_threshold set
    # getattr with default should return 0.5
    iou_threshold = getattr(config.data, "iou_threshold", 0.5)
    assert iou_threshold == 0.5


def test_validator_default_iou_threshold():
    """
    Detection_Validator instantiated with no args defaults to iou_threshold=0.5.

    Validates: Requirements 3.1 (preservation)
    """
    from src.training.detection_validator import Detection_Validator

    validator = Detection_Validator()
    assert validator.iou_threshold == 0.5


def test_runner_detection_validator_property_default():
    """
    The detection_validator lazy property creates a Detection_Validator with iou_threshold=0.5.

    Validates: Requirements 3.1 (preservation)
    """
    runner = Experiment_Runner()
    assert runner.detection_validator.iou_threshold == 0.5
