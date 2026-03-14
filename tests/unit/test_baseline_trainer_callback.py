"""
Unit tests for Baseline_Trainer step decay LR callback registration.

Exploratory test for Bug 1.1: verifies that add_callback("on_train_epoch_start", ...)
is called before model.train() in Baseline_Trainer.train().

This test FAILS on unfixed code (no callback registered) and PASSES after the fix.

**Validates: Requirements 2.1**
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import yaml

from src.training.baseline_trainer import Baseline_Trainer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(lr0: float = 0.001, lrf: float = 0.1):
    """Build a minimal mock config for Baseline_Trainer.train()."""
    config = MagicMock()
    config.training.baseline_epochs = 5
    config.training.image_size = 640
    config.training.batch_size = 4
    config.training.optimizer = "AdamW"
    config.training.lr0 = lr0
    config.training.lrf = lrf
    config.training.device = "cpu"
    config.data.yaml_path = None  # patched below
    config.model.weights = "yolov8n.pt"
    return config


def _write_data_yaml(tmp_path: Path) -> str:
    """Write a minimal data.yaml and return its path."""
    data = {
        "path": str(tmp_path),
        "train": "train/images",
        "val": "valid/images",
        "nc": 1,
        "names": ["object"],
    }
    yaml_path = tmp_path / "data.yaml"
    yaml_path.write_text(yaml.dump(data))
    return str(yaml_path)


# ---------------------------------------------------------------------------
# Test: add_callback called before model.train()
# ---------------------------------------------------------------------------


class TestBaselineTrainerCallbackRegistration:
    """
    **Property 1: Baseline LR Callback Registration**
    **Validates: Requirements 2.1**

    Baseline_Trainer.train() must call model.add_callback("on_train_epoch_start", ...)
    before calling model.train().
    """

    def test_add_callback_called_before_train(self, tmp_path: Path):
        """
        Assert that add_callback("on_train_epoch_start", ...) is called
        before model.train() in Baseline_Trainer.train().

        This is the exploratory test: it FAILS on unfixed code and PASSES
        after the fix is applied.
        """
        data_yaml = _write_data_yaml(tmp_path)
        config = _make_config()
        config.data.yaml_path = data_yaml

        call_order = []

        mock_model = MagicMock()

        def record_add_callback(event_name, callback_fn):
            call_order.append(("add_callback", event_name))

        def record_train(**kwargs):
            call_order.append(("train",))
            return MagicMock()

        mock_model.add_callback.side_effect = record_add_callback
        mock_model.train.side_effect = record_train

        trainer = Baseline_Trainer()

        with (
            patch("src.training.baseline_trainer.YOLO", return_value=mock_model),
            patch(
                "src.training.baseline_trainer.Baseline_Trainer._find_checkpoint",
                return_value="fake/best.pt",
            ),
            patch("src.training.device_utils.select_device", return_value="cpu"),
        ):
            trainer.train(
                config=config,
                config_name="yolov8n",
                full_training_images=["img1.jpg", "img2.jpg"],
                val_images=["val1.jpg"],
                base_path=tmp_path,
                seed=42,
            )

        # Verify add_callback was called with "on_train_epoch_start"
        add_callback_calls = [entry for entry in call_order if entry[0] == "add_callback"]
        assert len(add_callback_calls) >= 1, (
            "add_callback was never called — step decay LR callback not registered"
        )
        assert add_callback_calls[0][1] == "on_train_epoch_start", (
            f"Expected add_callback('on_train_epoch_start', ...) but got "
            f"add_callback('{add_callback_calls[0][1]}', ...)"
        )

        # Verify add_callback comes before train in the call sequence
        add_callback_idx = next(
            i for i, entry in enumerate(call_order) if entry[0] == "add_callback"
        )
        train_idx = next(i for i, entry in enumerate(call_order) if entry[0] == "train")
        assert add_callback_idx < train_idx, (
            f"add_callback (index {add_callback_idx}) must be called before "
            f"model.train() (index {train_idx})"
        )

    def test_add_callback_receives_callable(self, tmp_path: Path):
        """
        Assert that the second argument to add_callback is a callable function.
        """
        data_yaml = _write_data_yaml(tmp_path)
        config = _make_config()
        config.data.yaml_path = data_yaml

        captured_callback = {}

        mock_model = MagicMock()

        def capture_add_callback(event_name, callback_fn):
            captured_callback["event"] = event_name
            captured_callback["fn"] = callback_fn

        mock_model.add_callback.side_effect = capture_add_callback
        mock_model.train.return_value = MagicMock()

        trainer = Baseline_Trainer()

        with (
            patch("src.training.baseline_trainer.YOLO", return_value=mock_model),
            patch(
                "src.training.baseline_trainer.Baseline_Trainer._find_checkpoint",
                return_value="fake/best.pt",
            ),
            patch("src.training.device_utils.select_device", return_value="cpu"),
        ):
            trainer.train(
                config=config,
                config_name="yolov8n",
                full_training_images=["img1.jpg"],
                val_images=["val1.jpg"],
                base_path=tmp_path,
                seed=42,
            )

        assert "fn" in captured_callback, "add_callback was not called"
        assert callable(captured_callback["fn"]), (
            "The callback passed to add_callback must be callable"
        )
        assert captured_callback["event"] == "on_train_epoch_start"


# ---------------------------------------------------------------------------
# Test: iou not passed to model.train() (Bug 1.4)
# ---------------------------------------------------------------------------


class TestBaselineTrainerNoIouKwarg:
    """
    **Property 4: No iou in model.train() kwargs**
    **Validates: Requirements 2.4**

    Baseline_Trainer.train() must NOT pass `iou` as a keyword argument to
    model.train(). The `iou` parameter in Ultralytics controls NMS IoU during
    training-time augmentation, not the detection validation IoU threshold.
    """

    def test_iou_not_in_train_kwargs(self, tmp_path: Path):
        """
        Assert that 'iou' is not present in the kwargs passed to model.train()
        by Baseline_Trainer.train().

        This is the fix-verification test for Bug 1.4.
        """
        data_yaml = _write_data_yaml(tmp_path)
        config = _make_config()
        config.data.yaml_path = data_yaml

        captured_kwargs: dict = {}

        mock_model = MagicMock()

        def capture_train(**kwargs):
            captured_kwargs.update(kwargs)
            return MagicMock()

        mock_model.train.side_effect = capture_train

        trainer = Baseline_Trainer()

        with (
            patch("src.training.baseline_trainer.YOLO", return_value=mock_model),
            patch(
                "src.training.baseline_trainer.Baseline_Trainer._find_checkpoint",
                return_value="fake/best.pt",
            ),
            patch("src.training.device_utils.select_device", return_value="cpu"),
        ):
            trainer.train(
                config=config,
                config_name="yolov8n",
                full_training_images=["img1.jpg", "img2.jpg"],
                val_images=["val1.jpg"],
                base_path=tmp_path,
                seed=42,
            )

        assert mock_model.train.called, "model.train() was never called"
        assert "iou" not in captured_kwargs, (
            f"'iou' must not be passed to model.train(), but found iou={captured_kwargs['iou']!r}. "
            "The iou parameter controls NMS IoU during training augmentation, not validation IoU."
        )
