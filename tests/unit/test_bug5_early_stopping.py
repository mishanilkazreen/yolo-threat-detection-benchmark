"""
Tests for Bug 5 fix: early stopping is disabled in incremental training rounds.

Task 3.2 — Exploratory test: model.train() in the incremental loop is called with
           patience=0 (fails on unfixed code: patience passes config value).
Task 3.3 — Preservation test: _run_standard_training() still passes
           patience=config.training.patience unchanged.

Validates: Requirements 2.3, 3.3
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch, call
import pytest

from src.training.runner import Experiment_Runner


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(patience=5):
    """Build a minimal config object with a configurable patience value."""
    data = SimpleNamespace(
        yaml_path="fake/data.yaml",
        train_init_percentage=0.2,
        iou_threshold=0.5,
    )
    training = SimpleNamespace(
        rounds=1,
        epochs=None,
        epochs_per_round=10,
        patience=patience,
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
        data=data,
        training=training,
        model=SimpleNamespace(name="yolov8n", weights="yolov8n.pt"),
    )


def _make_runner_with_mocks():
    """Return a runner whose heavy dependencies are all mocked out."""
    runner = Experiment_Runner()

    mock_validator = MagicMock()
    mock_validator.iou_threshold = 0.5
    mock_validator.validate_detections.return_value = {
        "verified_samples": [],
        "undetected_images": [],
        "rejected_count": 0,
        "undetected_count": 0,
    }
    runner._detection_validator = mock_validator

    mock_edge = MagicMock()
    mock_edge.simulate_inference.return_value = []
    runner._edge_agent_simulator = mock_edge

    mock_tsm = MagicMock()
    runner._training_set_manager = mock_tsm

    mock_rmt = MagicMock()
    mock_rmt.aggregate_round_metrics.return_value = {}
    mock_rmt.generate_learning_curves.return_value = None
    runner._round_metrics_tracker = mock_rmt

    mock_metrics = MagicMock()
    mock_metrics.collect_round_metrics.return_value = {"mAP50": 0.5}
    mock_metrics.evaluate_final_test.return_value = {"metrics": {}}
    runner.metrics_collector = mock_metrics

    return runner


# ---------------------------------------------------------------------------
# Task 3.2 — Exploratory test: incremental loop uses patience=0
# ---------------------------------------------------------------------------

def test_incremental_loop_uses_patience_zero(tmp_path):
    """
    The model.train() call inside the incremental loop must be called with
    patience=0 regardless of config.training.patience.

    Fails on unfixed code (patience passes config value instead of 0).
    Validates: Requirements 2.3 (Bug 5 fix)
    """
    config = _make_config(patience=5)  # non-zero patience in config
    runner = _make_runner_with_mocks()

    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text(
        "path: .\ntrain: images/train\nval: images/val\nnames:\n  0: weapon\n"
    )
    config.data.yaml_path = str(data_yaml)

    (tmp_path / "images" / "train").mkdir(parents=True)
    (tmp_path / "images" / "val").mkdir(parents=True)

    # Capture kwargs passed to model.train()
    captured_train_kwargs = []

    mock_yolo_instance = MagicMock()
    mock_yolo_instance.add_callback.return_value = None

    def capture_train(**kwargs):
        captured_train_kwargs.append(kwargs)
        return MagicMock()

    mock_yolo_instance.train.side_effect = capture_train

    with (
        patch("src.training.runner.YOLO", return_value=mock_yolo_instance),
        patch("src.training.runner.Path.exists", return_value=True),
        patch.object(runner, "_get_all_images", side_effect=[
            [str(tmp_path / f"img{i}.jpg") for i in range(10)],  # train_images
            [],  # val_fixed
            [],  # test_fixed
        ]),
        patch.object(runner, "_create_round_data_yaml"),
        patch("yaml.safe_load", return_value={
            "path": str(tmp_path),
            "train": "images/train",
            "val": "images/val",
        }),
        patch("builtins.open", MagicMock(
            return_value=MagicMock(
                __enter__=MagicMock(return_value=MagicMock(read=MagicMock(return_value=""))),
                __exit__=MagicMock(return_value=False),
            )
        )),
        patch("json.dump"),
        patch("json.load", return_value={}),
    ):
        try:
            runner._run_incremental_training(config, "test_bug5", seed=42, run_id=None)
        except Exception:
            pass  # We only care about the kwargs captured before any checkpoint error

    assert len(captured_train_kwargs) >= 1, "model.train() was never called"
    incremental_call = captured_train_kwargs[0]
    assert incremental_call.get("patience") == 0, (
        f"Expected patience=0 in incremental loop but got patience={incremental_call.get('patience')}"
    )


# ---------------------------------------------------------------------------
# Task 3.3 — Preservation test: standard training still uses config patience
# ---------------------------------------------------------------------------

def test_standard_training_preserves_config_patience(tmp_path):
    """
    _run_standard_training() must still pass patience=config.training.patience
    to model.train(), unchanged by the Bug 5 fix.

    Validates: Requirements 3.3 (preservation)
    """
    patience_value = 7
    config = _make_config(patience=patience_value)
    config.training.epochs = 20  # standard training uses epochs, not epochs_per_round
    config.training.epochs_per_round = None

    runner = _make_runner_with_mocks()

    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text(
        "path: .\ntrain: images/train\nval: images/val\nnames:\n  0: weapon\n"
    )
    config.data.yaml_path = str(data_yaml)

    captured_train_kwargs = []

    mock_yolo_instance = MagicMock()
    mock_yolo_instance.add_callback.return_value = None

    def capture_train(**kwargs):
        captured_train_kwargs.append(kwargs)
        return MagicMock()

    mock_yolo_instance.train.side_effect = capture_train

    # Patch the weights dir so evaluate_and_save doesn't fail
    weights_dir = tmp_path / "weights"
    weights_dir.mkdir()
    (weights_dir / "best.pt").touch()

    with (
        patch("src.training.runner.YOLO", return_value=mock_yolo_instance),
        patch("src.training.runner.Path.exists", return_value=True),
        patch("yaml.safe_load", return_value={
            "path": str(tmp_path),
            "train": "images/train",
            "val": "images/val",
        }),
        patch("builtins.open", MagicMock(
            return_value=MagicMock(
                __enter__=MagicMock(return_value=MagicMock(read=MagicMock(return_value=""))),
                __exit__=MagicMock(return_value=False),
            )
        )),
        patch("json.dump"),
        patch("json.load", return_value={}),
    ):
        try:
            runner._run_standard_training(config, "test_bug5_standard", seed=42, run_id=None)
        except Exception:
            pass  # We only care about the kwargs captured before any evaluation error

    assert len(captured_train_kwargs) >= 1, "model.train() was never called in standard training"
    standard_call = captured_train_kwargs[0]
    assert standard_call.get("patience") == patience_value, (
        f"Expected patience={patience_value} in standard training but got "
        f"patience={standard_call.get('patience')}"
    )
