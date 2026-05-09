"""Unit and property-based tests for Comparison_Reporter."""

import math
from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st

from src.aggregation.comparison_reporter import COMPARISON_COLUMNS, Comparison_Reporter

# ---------------------------------------------------------------------------
# Helpers / strategies
# ---------------------------------------------------------------------------


def _make_metrics(
    mAP50: float = 0.821,
    mAP50_95: float = 0.601,
    f1: float = 0.792,
    hfs: float = 0.61,
    training_time: float = 1800.5,
) -> dict:
    return {
        "mAP50": mAP50,
        "mAP50-95": mAP50_95,
        "f1_score": f1,
        "hfs": hfs,
        "training_time_seconds": training_time,
    }


def _metric_strategy():
    """Generate a valid metrics dict with finite float values."""
    finite = st.floats(min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False)
    return st.fixed_dictionaries(
        {
            "mAP50": st.floats(0.0, 1.0, allow_nan=False),
            "mAP50-95": st.floats(0.0, 1.0, allow_nan=False),
            "f1_score": st.floats(0.0, 1.0, allow_nan=False),
            "hfs": st.floats(0.0, 1.0, allow_nan=False),
            "training_time_seconds": finite,
        }
    )


def _config_name_strategy():
    return st.text(
        alphabet="abcdefghijklmnopqrstuvwxyz0123456789_-",
        min_size=1,
        max_size=20,
    )


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


class TestGenerateComparison:
    def test_creates_md_and_csv(self, tmp_path: Path):
        reporter = Comparison_Reporter()
        reporter.generate_comparison(
            "yolov8n",
            _make_metrics(mAP50=0.856, training_time=3600.5),
            _make_metrics(mAP50=0.821, training_time=1800.5),
            output_dir=str(tmp_path),
        )
        assert (tmp_path / "baseline_vs_incremental_yolov8n.md").exists()
        assert (tmp_path / "baseline_vs_incremental_yolov8n.csv").exists()

    def test_difference_row_values(self, tmp_path: Path):
        inc = _make_metrics(mAP50=0.856, mAP50_95=0.623, f1=0.812, hfs=0.76, training_time=3600.5)
        base = _make_metrics(mAP50=0.821, mAP50_95=0.601, f1=0.792, hfs=0.61, training_time=1800.5)
        reporter = Comparison_Reporter()
        table = reporter._build_table(inc, base)

        diff = table[2]
        assert diff["Approach"] == "Difference (I - B)"
        assert abs(diff["mAP@0.5 (test)"] - (0.856 - 0.821)) < 1e-9
        assert abs(diff["mAP@0.5:0.95 (test)"] - (0.623 - 0.601)) < 1e-9
        assert abs(diff["F1-score (test)"] - (0.812 - 0.792)) < 1e-9
        assert abs(diff["HFS (val)"] - (0.76 - 0.61)) < 1e-9
        assert abs(diff["Total Training Time (s)"] - (3600.5 - 1800.5)) < 1e-9

    def test_table_has_three_rows(self, tmp_path: Path):
        reporter = Comparison_Reporter()
        table = reporter._build_table(_make_metrics(), _make_metrics())
        assert len(table) == 3

    def test_row_labels(self, tmp_path: Path):
        reporter = Comparison_Reporter()
        table = reporter._build_table(_make_metrics(), _make_metrics())
        assert table[0]["Approach"] == "One-Shot Baseline"
        assert table[1]["Approach"] == "Incremental"
        assert table[2]["Approach"] == "Difference (I - B)"

    def test_csv_has_correct_columns(self, tmp_path: Path):
        reporter = Comparison_Reporter()
        reporter.generate_comparison(
            "cfg", _make_metrics(), _make_metrics(), output_dir=str(tmp_path)
        )
        import csv

        with open(tmp_path / "baseline_vs_incremental_cfg.csv", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
        expected = ["Approach"] + [c for c, _ in COMPARISON_COLUMNS]
        assert fieldnames == expected

    def test_markdown_contains_header(self, tmp_path: Path):
        reporter = Comparison_Reporter()
        reporter.generate_comparison(
            "cfg", _make_metrics(), _make_metrics(), output_dir=str(tmp_path)
        )
        content = (tmp_path / "baseline_vs_incremental_cfg.md").read_text()
        assert "Baseline vs Incremental" in content
        assert "mAP@0.5 (test)" in content

    def test_output_dir_created_if_missing(self, tmp_path: Path):
        nested = tmp_path / "deep" / "nested"
        reporter = Comparison_Reporter()
        reporter.generate_comparison(
            "cfg", _make_metrics(), _make_metrics(), output_dir=str(nested)
        )
        assert (nested / "baseline_vs_incremental_cfg.md").exists()


class TestGenerateAllComparisons:
    def _write_metrics(self, base: Path, config_name: str, is_baseline: bool, mAP50: float = 0.8):
        suffix = "_baseline" if is_baseline else ""
        d = base / f"{config_name}{suffix}"
        d.mkdir(parents=True, exist_ok=True)
        test_metrics = {
            "test_metrics": {
                "mAP50": mAP50,
                "mAP50-95": 0.5,
                "precision": 0.8,
                "recall": 0.75,
                "f1_score": 0.77,
            },
            "training_time_seconds": 1000.0,
        }
        import json

        (d / "final_test_metrics.json").write_text(json.dumps(test_metrics))
        hfs = {"mean_hfs": 0.6}
        (d / "hfs_metrics.json").write_text(json.dumps(hfs))

    def test_generates_one_file_per_config(self, tmp_path: Path):
        outputs = tmp_path / "outputs"
        for name in ["yolov8n", "yolov8s"]:
            self._write_metrics(outputs, name, is_baseline=False)
            self._write_metrics(outputs, name, is_baseline=True)

        reporter = Comparison_Reporter()
        reporter.generate_all_comparisons(outputs_dir=str(outputs))

        md_files = list(outputs.glob("baseline_vs_incremental_*.md"))
        csv_files = list(outputs.glob("baseline_vs_incremental_*.csv"))
        assert len(md_files) == 2
        assert len(csv_files) == 2

    def test_skips_config_without_baseline(self, tmp_path: Path):
        outputs = tmp_path / "outputs"
        self._write_metrics(outputs, "yolov8n", is_baseline=False)
        # No baseline dir for yolov8n

        reporter = Comparison_Reporter()
        reporter.generate_all_comparisons(outputs_dir=str(outputs))

        md_files = list(outputs.glob("baseline_vs_incremental_*.md"))
        assert len(md_files) == 0

    def test_skips_config_without_incremental(self, tmp_path: Path):
        outputs = tmp_path / "outputs"
        self._write_metrics(outputs, "yolov8n", is_baseline=True)
        # No incremental dir for yolov8n

        reporter = Comparison_Reporter()
        reporter.generate_all_comparisons(outputs_dir=str(outputs))

        md_files = list(outputs.glob("baseline_vs_incremental_*.md"))
        assert len(md_files) == 0

    def test_nonexistent_outputs_dir_does_not_crash(self, tmp_path: Path):
        reporter = Comparison_Reporter()
        reporter.generate_all_comparisons(outputs_dir=str(tmp_path / "nonexistent"))
        # Should log warning and return without error


class TestLoadMetricsIncrementalKey:
    """
    Exploratory: _load_metrics(is_baseline=False) reads from "metrics" key.
    Fails on unfixed code (which reads "test_metrics"), passes after the fix.
    """

    def test_incremental_metrics_non_nan(self, tmp_path: Path):
        """
        Fake incremental final_test_metrics.json uses key "metrics".
        _load_metrics(is_baseline=False) must return non-NaN mAP50, mAP50-95, f1_score.
        """
        import json

        config_dir = tmp_path / "yolov8n"
        config_dir.mkdir()
        payload = {
            "metrics": {
                "mAP50": 0.72,
                "mAP50-95": 0.45,
                "f1_score": 0.68,
            },
            "training_time_seconds": 1234.5,
        }
        (config_dir / "final_test_metrics.json").write_text(json.dumps(payload))

        reporter = Comparison_Reporter()
        result = reporter._load_metrics(config_dir, "yolov8n", is_baseline=False)

        assert result is not None
        assert not math.isnan(result["mAP50"]), "mAP50 should not be NaN for incremental"
        assert not math.isnan(result["mAP50-95"]), "mAP50-95 should not be NaN for incremental"
        assert not math.isnan(result["f1_score"]), "f1_score should not be NaN for incremental"
        assert abs(result["mAP50"] - 0.72) < 1e-9
        assert abs(result["mAP50-95"] - 0.45) < 1e-9
        assert abs(result["f1_score"] - 0.68) < 1e-9


class TestLoadMetricsBaselineKeyPreserved:
    """
    Preservation: _load_metrics(is_baseline=True) still reads from "test_metrics" key.
    """

    def test_baseline_metrics_non_nan(self, tmp_path: Path):
        """
        Fake baseline final_test_metrics.json uses key "test_metrics".
        _load_metrics(is_baseline=True) must return correct non-NaN values.
        """
        import json

        config_dir = tmp_path / "yolov8n_baseline"
        config_dir.mkdir()
        payload = {
            "test_metrics": {
                "mAP50": 0.68,
                "mAP50-95": 0.41,
                "f1_score": 0.65,
            },
            "training_time_seconds": 900.0,
        }
        (config_dir / "final_test_metrics.json").write_text(json.dumps(payload))

        reporter = Comparison_Reporter()
        result = reporter._load_metrics(config_dir, "yolov8n", is_baseline=True)

        assert result is not None
        assert not math.isnan(result["mAP50"]), "mAP50 should not be NaN for baseline"
        assert not math.isnan(result["mAP50-95"]), "mAP50-95 should not be NaN for baseline"
        assert not math.isnan(result["f1_score"]), "f1_score should not be NaN for baseline"
        assert abs(result["mAP50"] - 0.68) < 1e-9
        assert abs(result["mAP50-95"] - 0.41) < 1e-9
        assert abs(result["f1_score"] - 0.65) < 1e-9


# ---------------------------------------------------------------------------
# Property-based tests
# ---------------------------------------------------------------------------


class TestPropertyDifferenceRow:
    """
    Property 7: Comparison Difference Row Equals Incremental Minus Baseline
    """

    @given(_metric_strategy(), _metric_strategy())
    @settings(max_examples=100)
    def test_difference_row_equals_incremental_minus_baseline(self, incremental, baseline):
        """
        For any pair of metric dicts, the difference row must equal
        incremental_value - baseline_value for every metric column.
        """
        reporter = Comparison_Reporter()
        table = reporter._build_table(incremental, baseline)
        diff_row = table[2]

        for col_label, key in COMPARISON_COLUMNS:
            expected = incremental[key] - baseline[key]
            actual = diff_row[col_label]
            assert abs(actual - expected) < 1e-9, (
                f"Difference mismatch for {col_label}: expected {expected}, got {actual}"
            )


class TestPropertyOneTablePerConfig:
    """
    Property 8: One Comparison Table Per Configuration
    """

    @given(st.lists(_config_name_strategy(), min_size=1, max_size=10, unique=True))
    @settings(max_examples=100)
    def test_one_table_per_config(self, config_names):
        """
        For N configs, generate_comparison() called N times must produce
        exactly N .md and N .csv files.
        """
        import tempfile

        with tempfile.TemporaryDirectory() as tmp_dir:
            reporter = Comparison_Reporter()
            out = Path(tmp_dir) / "outputs"
            for name in config_names:
                reporter.generate_comparison(
                    name,
                    _make_metrics(),
                    _make_metrics(),
                    output_dir=str(out),
                )

            md_files = list(out.glob("baseline_vs_incremental_*.md"))
            csv_files = list(out.glob("baseline_vs_incremental_*.csv"))
            assert len(md_files) == len(config_names)
            assert len(csv_files) == len(config_names)
