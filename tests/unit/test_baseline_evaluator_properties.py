"""
Property-Based Tests for Baseline_Evaluator

**Validates: Requirements 7.1, 7.2, 7.3, 7.4, 2.1, 2.2, 2.3**

Property 6: Baseline Output Files Contain Required Metadata
For any completed baseline run, the file outputs/{config_name}_baseline/final_test_metrics.json
SHALL contain the fields random_seed, training_set_size, epochs, model, weights,
training_time_seconds, test_metrics (with mAP50, mAP50-95, precision, recall, f1_score),
and per_class_test_metrics.
"""

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.config.parser import Configuration, DataConfig, ModelConfig, TrainingConfig
from src.training.baseline_evaluator import Baseline_Evaluator

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

_model_names = st.sampled_from(["yolov8n", "yolov8s", "yolov8m", "yolov11n"])
_weight_names = st.sampled_from(["yolov8n.yaml", "yolov8s.yaml", "yolov8n.pt"])
_seeds = st.integers(min_value=0, max_value=2**31 - 1)
_training_times = st.floats(min_value=1.0, max_value=100_000.0, allow_nan=False, allow_infinity=False)
_training_set_sizes = st.integers(min_value=1, max_value=100_000)
_epochs = st.integers(min_value=1, max_value=1000)
_iou_thresholds = st.floats(min_value=0.1, max_value=0.95, allow_nan=False)
_map_values = st.floats(min_value=0.0, max_value=1.0, allow_nan=False)
_per_class = st.lists(st.floats(min_value=0.0, max_value=1.0, allow_nan=False), min_size=0, max_size=10)


@st.composite
def valid_baseline_run_strategy(draw):
    """Generate a valid baseline run configuration and expected metric values."""
    model_name = draw(_model_names)
    weights = draw(_weight_names)
    seed = draw(_seeds)
    training_time = draw(_training_times)
    training_set_size = draw(_training_set_sizes)
    epochs = draw(_epochs)
    iou_threshold = draw(_iou_thresholds)

    map50 = draw(_map_values)
    map50_95 = draw(_map_values)
    precision = draw(_map_values)
    recall = draw(_map_values)
    per_class = draw(_per_class)

    config = Configuration(
        training=TrainingConfig(
            epochs=epochs,
            patience=10,
            image_size=640,
            device="cpu",
            runs=1,
            seeds=[seed],
            baseline_epochs=epochs,
        ),
        model=ModelConfig(name=model_name, weights=weights),
        data=DataConfig(
            yaml_path="data.yaml",
            train_init_percentage=0.2,
            iou_threshold=iou_threshold,
        ),
    )

    eval_result = {
        "mAP50": map50,
        "mAP50-95": map50_95,
        "precision": precision,
        "recall": recall,
        "f1_score": 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0,
        "mAP50_per_class": per_class,
    }

    return {
        "config": config,
        "config_name": model_name,
        "seed": seed,
        "training_time": training_time,
        "training_set_size": training_set_size,
        "epochs": epochs,
        "eval_result": eval_result,
    }


# ---------------------------------------------------------------------------
# Property 6: Baseline Output Files Contain Required Metadata
# ---------------------------------------------------------------------------

REQUIRED_TOP_LEVEL_FIELDS = [
    "config_name",
    "model",
    "weights",
    "random_seed",
    "training_set_size",
    "epochs",
    "training_time_seconds",
    "training_time_minutes",
    "iou_threshold",
    "test_metrics",
    "per_class_test_metrics",
]

REQUIRED_TEST_METRIC_FIELDS = ["mAP50", "mAP50-95", "precision", "recall", "f1_score"]


class TestProperty6BaselineOutputMetadata:
    """
    Property 6: Baseline Output Files Contain Required Metadata

    **Validates: Requirements 7.1, 7.2, 7.3, 7.4, 2.1, 2.2, 2.3**
    """

    @pytest.mark.property
    @given(run=valid_baseline_run_strategy())
    @settings(max_examples=100)
    def test_final_test_metrics_contains_all_required_fields(self, run):
        """
        For any valid baseline run, final_test_metrics.json SHALL contain all required fields.

        **Validates: Requirements 7.1, 7.2, 7.3, 7.4, 2.1, 2.2, 2.3**
        """
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            evaluator = Baseline_Evaluator()
            config = run["config"]
            config_name = run["config_name"]
            baseline_name = f"{config_name}_baseline"

            output_dir = tmp_path / "outputs" / baseline_name
            gradcam_dir = tmp_path / "explanations" / baseline_name / "gradcam"
            output_dir.mkdir(parents=True, exist_ok=True)
            gradcam_dir.mkdir(parents=True, exist_ok=True)

            import src.training.baseline_evaluator as mod

            original_path = mod.Path

            class PatchedPath:
                def __new__(cls, *args):
                    if len(args) == 1:
                        s = str(args[0])
                        if s == f"outputs/{baseline_name}":
                            return output_dir
                        if s == f"explanations/{baseline_name}/gradcam":
                            return gradcam_dir
                    return original_path(*args)

            hfs_return = {
                "round": 0,
                "method": "gradcam",
                "mean_hfs": 0.5,
                "num_images": 0,
                "individual_hfs": {},
                "statistics": {"min": 0.0, "max": 0.0, "std": 0.0, "median": 0.0},
            }

            mod.Path = PatchedPath
            try:
                with (
                    patch.object(evaluator, "_evaluate_on_split", return_value=run["eval_result"]),
                    patch.object(evaluator, "_compute_hfs", return_value=hfs_return),
                ):
                    evaluator.evaluate(
                        checkpoint_path="fake.pt",
                        config_name=config_name,
                        val_data_yaml="val.yaml",
                        test_data_yaml="test.yaml",
                        hfs_image_subset=[],
                        base_path=tmp_path,
                        seed=run["seed"],
                        training_time=run["training_time"],
                        training_set_size=run["training_set_size"],
                        epochs=run["epochs"],
                        config=config,
                    )
            finally:
                mod.Path = original_path

            metrics_file = output_dir / "final_test_metrics.json"
            assert metrics_file.exists(), "final_test_metrics.json must be created"

            with open(metrics_file) as f:
                saved = json.load(f)

            # All required top-level fields must be present
            for field in REQUIRED_TOP_LEVEL_FIELDS:
                assert field in saved, f"Missing required field in final_test_metrics.json: '{field}'"

            # All required test_metrics sub-fields must be present
            for field in REQUIRED_TEST_METRIC_FIELDS:
                assert field in saved["test_metrics"], (
                    f"Missing required test_metrics field: '{field}'"
                )

            # per_class_test_metrics must contain mAP50_per_class
            assert "mAP50_per_class" in saved["per_class_test_metrics"], (
                "per_class_test_metrics must contain 'mAP50_per_class'"
            )

            # Metadata values must match inputs
            assert saved["random_seed"] == run["seed"]
            assert saved["training_set_size"] == run["training_set_size"]
            assert saved["epochs"] == run["epochs"]
            assert abs(saved["training_time_seconds"] - run["training_time"]) < 1e-6
            assert abs(saved["training_time_minutes"] - run["training_time"] / 60.0) < 1e-6
