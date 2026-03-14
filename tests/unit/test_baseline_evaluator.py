"""Unit tests for Baseline_Evaluator."""

import json
from pathlib import Path
from unittest.mock import patch

from src.config.parser import Configuration, DataConfig, ModelConfig, TrainingConfig
from src.training.baseline_evaluator import Baseline_Evaluator


def _make_config(
    model_name: str = "yolov8n",
    weights: str = "yolov8n.yaml",
    iou_threshold: float = 0.5,
) -> Configuration:
    return Configuration(
        training=TrainingConfig(
            epochs=50,
            patience=10,
            image_size=640,
            device="cpu",
            runs=1,
            seeds=[42],
            baseline_epochs=50,
        ),
        model=ModelConfig(name=model_name, weights=weights),
        data=DataConfig(
            yaml_path="data.yaml",
            train_init_percentage=0.2,
            iou_threshold=iou_threshold,
        ),
    )


def _run_evaluate(evaluator, config, config_name, output_dir, gradcam_dir, **kwargs):
    """
    Run evaluator.evaluate() with output dirs redirected to tmp_path.

    Patches Path so that outputs/{config_name}_baseline and
    explanations/{config_name}_baseline/gradcam resolve to the provided dirs.
    """
    baseline_name = f"{config_name}_baseline"
    output_dir.mkdir(parents=True, exist_ok=True)
    gradcam_dir.mkdir(parents=True, exist_ok=True)

    import src.training.baseline_evaluator as mod

    original_path = mod.Path

    class PatchedPath:
        """Proxy that intercepts specific path strings."""

        def __new__(cls, *args):
            if len(args) == 1:
                s = str(args[0])
                if s == f"outputs/{baseline_name}":
                    return output_dir
                if s == f"explanations/{baseline_name}/gradcam":
                    return gradcam_dir
            return original_path(*args)

    mod.Path = PatchedPath
    try:
        result = evaluator.evaluate(
            config_name=config_name,
            config=config,
            **kwargs,
        )
    finally:
        mod.Path = original_path

    return result


class TestBaselineEvaluatorOutputStructure:
    """Tests that evaluate() produces the correct output files and structure."""

    def _default_eval_result(self):
        return {
            "mAP50": 0.821,
            "mAP50-95": 0.601,
            "precision": 0.810,
            "recall": 0.775,
            "f1_score": 0.792,
            "mAP50_per_class": [0.85, 0.88],
        }

    def _default_hfs_result(self):
        return {
            "round": 0,
            "method": "gradcam",
            "mean_hfs": 0.61,
            "num_images": 5,
            "individual_hfs": {},
            "statistics": {"min": 0.5, "max": 0.7, "std": 0.05, "median": 0.61},
        }

    def test_final_test_metrics_contains_required_fields(self, tmp_path: Path):
        """
        Verify final_test_metrics.json contains all required metadata fields.

        **Validates: Requirements 7.1, 7.2, 7.3, 7.4, 2.1, 2.2, 2.3**
        """
        config = _make_config()
        evaluator = Baseline_Evaluator()
        baseline_name = "yolov8n_baseline"
        output_dir = tmp_path / "outputs" / baseline_name
        gradcam_dir = tmp_path / "explanations" / baseline_name / "gradcam"

        with (
            patch.object(evaluator, "_evaluate_on_split", return_value=self._default_eval_result()),
            patch.object(evaluator, "_compute_hfs", return_value=self._default_hfs_result()),
        ):
            _run_evaluate(
                evaluator,
                config,
                "yolov8n",
                output_dir,
                gradcam_dir,
                checkpoint_path="fake.pt",
                val_data_yaml="val.yaml",
                test_data_yaml="test.yaml",
                hfs_image_subset=["img1.jpg"],
                base_path=tmp_path,
                seed=42,
                training_time=1800.5,
                training_set_size=1000,
                epochs=50,
            )

        metrics_file = output_dir / "final_test_metrics.json"
        assert metrics_file.exists(), "final_test_metrics.json must be created"

        with open(metrics_file) as f:
            saved = json.load(f)

        required_fields = [
            "config_name",
            "model",
            "weights",
            "random_seed",
            "epochs",
            "training_set_size",
            "training_time_seconds",
            "training_time_minutes",
            "iou_threshold",
            "test_metrics",
            "per_class_test_metrics",
        ]
        for field in required_fields:
            assert field in saved, f"Missing required field: {field}"

        for field in ["mAP50", "mAP50-95", "precision", "recall", "f1_score"]:
            assert field in saved["test_metrics"], f"Missing test_metrics field: {field}"

        assert "mAP50_per_class" in saved["per_class_test_metrics"]

    def test_val_metrics_file_created(self, tmp_path: Path):
        """Verify val_metrics.json is saved to the correct location."""
        config = _make_config()
        evaluator = Baseline_Evaluator()
        baseline_name = "yolov8n_baseline"
        output_dir = tmp_path / "outputs" / baseline_name
        gradcam_dir = tmp_path / "explanations" / baseline_name / "gradcam"

        with (
            patch.object(evaluator, "_evaluate_on_split", return_value=self._default_eval_result()),
            patch.object(evaluator, "_compute_hfs", return_value=self._default_hfs_result()),
        ):
            _run_evaluate(
                evaluator,
                config,
                "yolov8n",
                output_dir,
                gradcam_dir,
                checkpoint_path="fake.pt",
                val_data_yaml="val.yaml",
                test_data_yaml="test.yaml",
                hfs_image_subset=[],
                base_path=tmp_path,
                seed=42,
                training_time=100.0,
                training_set_size=500,
                epochs=50,
            )

        assert (output_dir / "val_metrics.json").exists(), "val_metrics.json must be created"

    def test_hfs_metrics_file_created(self, tmp_path: Path):
        """Verify hfs_metrics.json is saved to the correct location."""
        config = _make_config()
        evaluator = Baseline_Evaluator()
        baseline_name = "yolov8n_baseline"
        output_dir = tmp_path / "outputs" / baseline_name
        gradcam_dir = tmp_path / "explanations" / baseline_name / "gradcam"

        with (
            patch.object(evaluator, "_evaluate_on_split", return_value=self._default_eval_result()),
            patch.object(evaluator, "_compute_hfs", return_value=self._default_hfs_result()),
        ):
            _run_evaluate(
                evaluator,
                config,
                "yolov8n",
                output_dir,
                gradcam_dir,
                checkpoint_path="fake.pt",
                val_data_yaml="val.yaml",
                test_data_yaml="test.yaml",
                hfs_image_subset=[],
                base_path=tmp_path,
                seed=42,
                training_time=100.0,
                training_set_size=500,
                epochs=50,
            )

        assert (output_dir / "hfs_metrics.json").exists(), "hfs_metrics.json must be created"

    def test_metadata_values_match_inputs(self, tmp_path: Path):
        """Verify metadata values in final_test_metrics.json match the inputs."""
        config = _make_config(model_name="yolov8s", weights="yolov8s.yaml", iou_threshold=0.5)
        evaluator = Baseline_Evaluator()
        baseline_name = "yolov8s_baseline"
        output_dir = tmp_path / "outputs" / baseline_name
        gradcam_dir = tmp_path / "explanations" / baseline_name / "gradcam"

        with (
            patch.object(evaluator, "_evaluate_on_split", return_value=self._default_eval_result()),
            patch.object(evaluator, "_compute_hfs", return_value=self._default_hfs_result()),
        ):
            _run_evaluate(
                evaluator,
                config,
                "yolov8s",
                output_dir,
                gradcam_dir,
                checkpoint_path="fake.pt",
                val_data_yaml="val.yaml",
                test_data_yaml="test.yaml",
                hfs_image_subset=[],
                base_path=tmp_path,
                seed=99,
                training_time=3600.0,
                training_set_size=2000,
                epochs=50,
            )

        with open(output_dir / "final_test_metrics.json") as f:
            saved = json.load(f)

        assert saved["config_name"] == "yolov8s"
        assert saved["model"] == "yolov8s"
        assert saved["weights"] == "yolov8s.yaml"
        assert saved["random_seed"] == 99
        assert saved["epochs"] == 50
        assert saved["training_set_size"] == 2000
        assert abs(saved["training_time_seconds"] - 3600.0) < 1e-6
        assert abs(saved["training_time_minutes"] - 60.0) < 1e-6
        assert saved["iou_threshold"] == 0.5

    def test_f1_score_computation(self):
        """Verify F1 score is computed correctly from precision and recall."""
        assert abs(Baseline_Evaluator._compute_f1(0.8, 0.6) - (2 * 0.8 * 0.6 / (0.8 + 0.6))) < 1e-9
        assert Baseline_Evaluator._compute_f1(0.0, 0.0) == 0.0
        assert Baseline_Evaluator._compute_f1(1.0, 1.0) == 1.0

    def test_empty_hfs_subset_returns_zero(self, tmp_path: Path):
        """Verify that an empty HFS subset returns zero HFS without crashing."""
        evaluator = Baseline_Evaluator()
        result = evaluator._compute_hfs(
            checkpoint_path="fake.pt",
            hfs_image_subset=[],
            base_path=tmp_path,
            gradcam_dir=tmp_path / "gradcam",
        )
        assert result["mean_hfs"] == 0.0
        assert result["num_images"] == 0
