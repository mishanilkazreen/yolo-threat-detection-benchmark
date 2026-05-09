"""Comparison_Reporter: generates side-by-side baseline vs incremental tables."""

import csv
import json
import logging
from pathlib import Path
from typing import Any

# Metric columns in display order
COMPARISON_COLUMNS = [
    ("mAP@0.5 (test)", "mAP50"),
    ("mAP@0.5:0.95 (test)", "mAP50-95"),
    ("F1-score (test)", "f1_score"),
    ("HFS (val)", "hfs"),
    ("Total Training Time (s)", "training_time_seconds"),
]


class Comparison_Reporter:
    """Produces side-by-side comparison tables for baseline vs incremental runs."""

    def __init__(self) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_comparison(
        self,
        config_name: str,
        incremental_metrics: dict[str, Any],
        baseline_metrics: dict[str, Any],
        output_dir: str = "outputs",
    ) -> None:
        """
        Produce side-by-side comparison table for one config.

        Saves:
          outputs/baseline_vs_incremental_{config_name}.md
          outputs/baseline_vs_incremental_{config_name}.csv
        """
        table = self._build_table(incremental_metrics, baseline_metrics)
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        stem = f"baseline_vs_incremental_{config_name}"
        self._save_markdown(table, out / f"{stem}.md", config_name)
        self._save_csv(table, out / f"{stem}.csv")

        self.logger.info("Comparison saved: %s/%s.{md,csv}", output_dir, stem)

    def generate_all_comparisons(
        self,
        outputs_dir: str = "outputs",
    ) -> None:
        """
        Scan outputs/ for all configs that have both baseline and
        incremental results and generate one comparison per config.
        """
        outputs_path = Path(outputs_dir)
        if not outputs_path.exists():
            self.logger.warning("Outputs directory not found: %s", outputs_dir)
            return

        # Collect all subdirectory names
        subdirs = {d.name for d in outputs_path.iterdir() if d.is_dir()}

        # Find configs that have both a baseline dir and an incremental dir
        baseline_suffix = "_baseline"
        processed = 0
        for name in sorted(subdirs):
            if not name.endswith(baseline_suffix):
                continue
            config_name = name[: -len(baseline_suffix)]
            if config_name not in subdirs:
                continue  # no matching incremental dir

            baseline_dir = outputs_path / name
            incremental_dir = outputs_path / config_name

            baseline_metrics = self._load_metrics(baseline_dir, config_name, is_baseline=True)
            incremental_metrics = self._load_metrics(
                incremental_dir, config_name, is_baseline=False
            )

            if baseline_metrics is None or incremental_metrics is None:
                continue

            self.generate_comparison(
                config_name=config_name,
                incremental_metrics=incremental_metrics,
                baseline_metrics=baseline_metrics,
                output_dir=outputs_dir,
            )
            processed += 1

        self.logger.info("Generated %d comparison(s) in %s", processed, outputs_dir)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_table(
        self,
        incremental_metrics: dict[str, Any],
        baseline_metrics: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Return three rows: baseline, incremental, difference."""
        rows = []
        for label, src in [
            ("One-Shot Baseline", baseline_metrics),
            ("Incremental", incremental_metrics),
        ]:
            row: dict[str, Any] = {"Approach": label}
            for col_label, key in COMPARISON_COLUMNS:
                row[col_label] = src.get(key, float("nan"))
            rows.append(row)

        # Difference row: incremental - baseline
        diff_row: dict[str, Any] = {"Approach": "Difference (I - B)"}
        for col_label, key in COMPARISON_COLUMNS:
            inc_val = incremental_metrics.get(key, float("nan"))
            base_val = baseline_metrics.get(key, float("nan"))
            diff_row[col_label] = inc_val - base_val
        rows.append(diff_row)

        return rows

    def _save_markdown(
        self,
        table: list[dict[str, Any]],
        path: Path,
        config_name: str,
    ) -> None:
        col_labels = ["Approach"] + [c for c, _ in COMPARISON_COLUMNS]

        # Compute column widths
        widths = {col: len(col) for col in col_labels}
        for row in table:
            for col in col_labels:
                widths[col] = max(widths[col], len(self._fmt(row.get(col, ""))))

        def row_str(row: dict[str, Any]) -> str:
            cells = [self._fmt(row.get(col, "")).ljust(widths[col]) for col in col_labels]
            return "| " + " | ".join(cells) + " |"

        sep = "| " + " | ".join("-" * widths[col] for col in col_labels) + " |"
        header = "| " + " | ".join(col.ljust(widths[col]) for col in col_labels) + " |"

        lines = [
            f"# Baseline vs Incremental: {config_name}",
            "",
            header,
            sep,
        ]
        for row in table:
            lines.append(row_str(row))
        lines.append("")

        with open(path, "w", newline="\n", encoding="utf-8") as f:
            f.write("\n".join(lines))

    def _save_csv(self, table: list[dict[str, Any]], path: Path) -> None:
        col_labels = ["Approach"] + [c for c, _ in COMPARISON_COLUMNS]
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=col_labels)
            writer.writeheader()
            for row in table:
                writer.writerow({col: self._fmt(row.get(col, "")) for col in col_labels})

    @staticmethod
    def _fmt(value: Any) -> str:
        """Format a cell value for display."""
        if isinstance(value, str):
            return value
        if isinstance(value, float):
            import math

            if math.isnan(value):
                return "nan"
            return f"{value:.4f}"
        return str(value)

    def _load_metrics(
        self,
        config_dir: Path,
        config_name: str,
        is_baseline: bool,
    ) -> dict[str, Any] | None:
        """
        Load metrics from a config output directory.

        Reads final_test_metrics.json and hfs_metrics.json and merges
        them into the flat dict expected by generate_comparison().
        """
        test_file = config_dir / "final_test_metrics.json"
        hfs_file = config_dir / "hfs_metrics.json"

        if not test_file.exists():
            self.logger.warning("Missing final_test_metrics.json in %s — skipping", config_dir)
            return None

        try:
            with open(test_file, encoding="utf-8") as f:
                test_data = json.load(f)
        except Exception as exc:
            self.logger.warning("Failed to load %s: %s", test_file, exc)
            return None

        # Extract flat metrics — key differs by source:
        # baseline_evaluator.py writes under "test_metrics"; evaluator.py writes under "metrics"
        if is_baseline:
            test_metrics = test_data.get("test_metrics", {})
        else:
            test_metrics = test_data.get("metrics", {})
        # Training time key differs between some writers:
        # - baseline / some evaluators: "training_time_seconds"
        # - incremental (Metrics_Collector.evaluate_final_test): "total_training_time_seconds"
        training_time = test_data.get("training_time_seconds")
        if training_time is None:
            training_time = test_data.get("total_training_time_seconds")
        if training_time is None:
            training_time = float("nan")

        metrics: dict[str, Any] = {
            "mAP50": test_metrics.get("mAP50", float("nan")),
            "mAP50-95": test_metrics.get("mAP50-95", float("nan")),
            "f1_score": test_metrics.get("f1_score", float("nan")),
            # Normalize under a single internal key for downstream consumers
            "training_time_seconds": training_time,
        }

        # HFS
        if hfs_file.exists():
            try:
                with open(hfs_file, encoding="utf-8") as f:
                    hfs_data = json.load(f)
                metrics["hfs"] = hfs_data.get("mean_hfs", float("nan"))
            except Exception as exc:
                self.logger.warning("Failed to load %s: %s", hfs_file, exc)
                metrics["hfs"] = float("nan")
        else:
            self.logger.warning("Missing hfs_metrics.json in %s", config_dir)
            metrics["hfs"] = float("nan")

        return metrics
