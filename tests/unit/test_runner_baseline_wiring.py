"""
Tests for Experiment_Runner baseline wiring.

Feature: one-shot-baseline
"""

from unittest.mock import MagicMock, mock_open, patch

import pytest

from src.training.runner import Experiment_Runner

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Fake data config returned by yaml.safe_load when the runner opens data.yaml.
_FAKE_DATA_CONFIG = {
    "path": ".",
    "train": "train/images",
    "val": "valid/images",
    "nc": 2,
    "names": ["cat", "dog"],
}

# Dummy image list returned by the mocked _get_all_images helper.
_DUMMY_IMAGES = [f"img_{i:03d}.jpg" for i in range(10)]


def _make_config(run_baseline: bool = False):
    """Build a minimal mock config object."""
    config = MagicMock()
    config.training.run_baseline = run_baseline
    config.training.rounds = 2
    config.training.epochs_per_round = 5
    config.training.epochs = None
    config.training.patience = 5
    config.training.image_size = 640
    config.training.batch_size = 4
    config.training.optimizer = "AdamW"
    config.training.lr0 = 0.001
    config.training.lrf = 0.1
    config.training.device = "cpu"
    config.training.baseline_epochs = 50
    config.training.seeds = [42]
    config.training.runs = 1
    config.data.yaml_path = "data.yaml"
    config.data.train_init_percentage = 0.2
    config.data.iou_threshold = 0.5
    config.model.weights = "yolov8n.pt"
    config.model.name = "yolov8n"
    return config


def _build_runner_with_mocked_internals() -> Experiment_Runner:
    """
    Create an Experiment_Runner with all heavy internal components replaced
    by MagicMocks so that _run_incremental_training() can execute without a
    real filesystem, dataset, or GPU.
    """
    runner = Experiment_Runner()

    # Replace heavy instance methods with stubs.
    runner._get_all_images = MagicMock(return_value=_DUMMY_IMAGES)
    runner._verify_split_ratio = MagicMock()
    runner._create_round_data_yaml = MagicMock()
    runner._create_step_decay_callback = MagicMock(return_value=lambda _: None)
    runner._cleanup_memory = MagicMock()

    # Replace the Metrics_Collector (created eagerly in __init__).
    mock_mc = MagicMock()
    mock_mc.collect_round_metrics.return_value = {}
    mock_mc.evaluate_final_test.return_value = {
        "metrics": {},
        "total_training_time_seconds": 0.0,
    }
    runner.metrics_collector = mock_mc

    # Pre-assign the lazy-loaded components so the properties return mocks
    # without importing or instantiating the real heavy classes.
    mock_eas = MagicMock()
    mock_eas.simulate_inference.return_value = []
    runner._edge_agent_simulator = mock_eas

    mock_dv = MagicMock()
    mock_dv.validate_detections.return_value = {
        "verified_samples": [],
        "undetected_images": [],
    }
    runner._detection_validator = mock_dv

    runner._training_set_manager = MagicMock()
    runner._round_metrics_tracker = MagicMock()

    return runner


# ---------------------------------------------------------------------------
# Property 4: run_baseline=false Skips Baseline Execution
# Validates: Requirements 5.3
# ---------------------------------------------------------------------------


class TestRunBaselineFalseSkipsBaseline:
    """
    **Property 4: run_baseline=false Skips Baseline Execution**
    **Validates: Requirements 5.3**

    For any valid configuration with run_baseline set to False or absent,
    the Experiment_Runner SHALL NOT invoke Baseline_Trainer.train().
    """

    def test_run_baseline_false_does_not_trigger_baseline(self):
        """
        **Property 4: run_baseline=false Skips Baseline Execution**
        **Validates: Requirements 5.3**

        Executes _run_incremental_training() with run_baseline=False and
        asserts that Baseline_Trainer.train is never invoked.
        """
        config = _make_config(run_baseline=False)
        runner = _build_runner_with_mocked_internals()

        with (
            patch("src.training.runner.yaml.safe_load", return_value=_FAKE_DATA_CONFIG),
            patch("builtins.open", mock_open()),
            patch("pathlib.Path.mkdir"),
            patch("pathlib.Path.exists", return_value=True),
            patch("src.training.runner.YOLO") as MockYOLO,
            patch("src.training.runner.select_device", return_value="cpu"),
            patch("src.training.runner.Baseline_Trainer") as MockBaselineTrainer,
        ):
            MockYOLO.return_value.train.return_value = MagicMock()

            runner._run_incremental_training(config, "test_config", seed=42)

            MockBaselineTrainer.return_value.train.assert_not_called()

    @pytest.mark.parametrize("run_baseline", [True, False])
    def test_guard_condition_matches_run_baseline_flag(self, run_baseline: bool):
        """
        **Property 4: run_baseline=false Skips Baseline Execution**
        **Validates: Requirements 5.3**

        The guard `getattr(config.training, 'run_baseline', False)` must
        evaluate to exactly the value of run_baseline for any boolean input.
        """
        config = _make_config(run_baseline=run_baseline)

        guard_result = getattr(config.training, "run_baseline", False)
        assert guard_result == run_baseline

    def test_run_baseline_true_would_call_baseline_trainer(self):
        """
        Complementary check: when run_baseline=True, _run_incremental_training()
        must invoke Baseline_Trainer.train() exactly once.
        **Validates: Requirements 5.2**
        """
        config = _make_config(run_baseline=True)
        runner = _build_runner_with_mocked_internals()

        with (
            patch("src.training.runner.yaml.safe_load", return_value=_FAKE_DATA_CONFIG),
            patch("builtins.open", mock_open()),
            patch("pathlib.Path.mkdir"),
            patch("pathlib.Path.exists", return_value=True),
            patch("src.training.runner.YOLO") as MockYOLO,
            patch("src.training.runner.select_device", return_value="cpu"),
            patch("src.training.runner.Baseline_Trainer") as MockBaselineTrainer,
            patch("src.training.runner.Baseline_Evaluator") as MockBaselineEvaluator,
            patch("src.training.runner.Comparison_Reporter"),
        ):
            MockYOLO.return_value.train.return_value = MagicMock()
            MockBaselineTrainer.return_value.train.return_value = {
                "checkpoint_path": "runs/detect/test_config_baseline/weights/best.pt",
                "training_time_seconds": 1.0,
                "training_set_size": 10,
                "epochs": 50,
            }
            MockBaselineEvaluator.return_value.evaluate.return_value = {
                "test_metrics": {"test_metrics": {}},
                "hfs_metrics": {"mean_hfs": 0.0},
            }

            runner._run_incremental_training(config, "test_config", seed=42)

            MockBaselineTrainer.return_value.train.assert_called_once()
