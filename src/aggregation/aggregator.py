"""Results aggregation across multiple model configurations and training rounds."""

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class Results_Aggregator:
    """Aggregates results across multiple model configurations and training rounds."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def collect_round_metrics(
        self, outputs_dir: str = "outputs"
    ) -> dict[str, list[dict[str, Any]]]:
        """
        Collect round-level metrics from all model configurations.

        Args:
            outputs_dir: Base outputs directory

        Returns:
            Dictionary mapping config_name to list of round metrics
        """
        outputs_path = Path(outputs_dir)

        if not outputs_path.exists():
            self.logger.warning(f"Outputs directory not found: {outputs_dir}")
            return {}

        all_round_metrics = {}

        # Find all config directories
        for config_dir in outputs_path.iterdir():
            if not config_dir.is_dir():
                continue

            config_name = config_dir.name
            round_metrics = []

            # Collect metrics for each round
            for round_file in sorted(config_dir.glob("round_*_metrics.json")):
                try:
                    with open(round_file) as f:
                        metrics = json.load(f)
                    round_metrics.append(metrics)
                except Exception as e:
                    self.logger.warning(f"Failed to load {round_file}: {e}")

            if round_metrics:
                all_round_metrics[config_name] = round_metrics
                self.logger.info(f"Collected {len(round_metrics)} rounds for {config_name}")

        return all_round_metrics

    def collect_final_test_metrics(self, outputs_dir: str = "outputs") -> dict[str, dict[str, Any]]:
        """
        Collect final test metrics from all model configurations.

        Args:
            outputs_dir: Base outputs directory

        Returns:
            Dictionary mapping config_name to final test metrics
        """
        outputs_path = Path(outputs_dir)

        if not outputs_path.exists():
            self.logger.warning(f"Outputs directory not found: {outputs_dir}")
            return {}

        final_metrics = {}

        # Find all config directories
        for config_dir in outputs_path.iterdir():
            if not config_dir.is_dir():
                continue

            config_name = config_dir.name
            final_test_file = config_dir / "final_test_metrics.json"

            if final_test_file.exists():
                try:
                    with open(final_test_file) as f:
                        metrics = json.load(f)
                    final_metrics[config_name] = metrics
                    self.logger.info(f"Collected final test metrics for {config_name}")
                except Exception as e:
                    self.logger.warning(f"Failed to load {final_test_file}: {e}")

        return final_metrics

    def generate_round_comparison_table(
        self, round_metrics: dict[str, list[dict[str, Any]]], output_dir: str = "outputs"
    ) -> None:
        """
        Generate per-round comparative table.

        Args:
            round_metrics: Dictionary mapping config_name to list of round metrics
            output_dir: Directory to save tables
        """
        if not round_metrics:
            self.logger.warning("No round metrics to aggregate")
            return

        # Prepare data for table
        rows = []

        for config_name, rounds in round_metrics.items():
            for round_data in rounds:
                row = {
                    "Model": config_name,
                    "Round": round_data["round"],
                    "mAP@0.5": round_data["metrics"]["mAP50"],
                    "Precision": round_data["metrics"]["precision"],
                    "Recall": round_data["metrics"]["recall"],
                    "F1-Score": round_data["metrics"]["f1_score"],
                    "Training Set Size": round_data["training_set_size"],
                    "Verified Samples Added": round_data["verified_samples_added"],
                }
                rows.append(row)

        # Create DataFrame
        df = pd.DataFrame(rows)

        # Sort by model and round
        df = df.sort_values(["Model", "Round"])

        # Save as CSV
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        csv_file = output_path / "round_comparison_table.csv"
        df.to_csv(csv_file, index=False, float_format="%.4f")
        self.logger.info(f"Round comparison table saved to {csv_file}")

        # Save as Markdown
        md_file = output_path / "round_comparison_table.md"
        with open(md_file, "w") as f:
            f.write("# Per-Round Performance Comparison\n\n")
            f.write(df.to_markdown(index=False, floatfmt=".4f"))
        self.logger.info(f"Round comparison table saved to {md_file}")

    def generate_final_comparison_table(
        self, final_metrics: dict[str, dict[str, Any]], output_dir: str = "outputs"
    ) -> None:
        """
        Generate final performance comparative table.

        Args:
            final_metrics: Dictionary mapping config_name to final test metrics
            output_dir: Directory to save tables
        """
        if not final_metrics:
            self.logger.warning("No final metrics to aggregate")
            return

        # Prepare data for table
        rows = []

        for config_name, metrics in final_metrics.items():
            row = {
                "Model": config_name,
                "mAP@0.5": metrics["metrics"]["mAP50"],
                "mAP@0.5:0.95": metrics["metrics"]["mAP50-95"],
                "Precision": metrics["metrics"]["precision"],
                "Recall": metrics["metrics"]["recall"],
                "F1-Score": metrics["metrics"]["f1_score"],
                "Total Training Time (min)": metrics["total_training_time_minutes"],
                "Parameters": metrics["model_info"].get("parameters", "N/A"),
                "GFLOPs": metrics["model_info"].get("GFLOPs", "N/A"),
            }
            rows.append(row)

        # Create DataFrame
        df = pd.DataFrame(rows)

        # Sort by mAP@0.5 descending
        df = df.sort_values("mAP@0.5", ascending=False)

        # Save as CSV
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        csv_file = output_path / "final_comparison_table.csv"
        df.to_csv(csv_file, index=False, float_format="%.4f")
        self.logger.info(f"Final comparison table saved to {csv_file}")

        # Save as Markdown
        md_file = output_path / "final_comparison_table.md"
        with open(md_file, "w") as f:
            f.write("# Final Performance Comparison (Test Set)\n\n")
            f.write(df.to_markdown(index=False, floatfmt=".4f"))
        self.logger.info(f"Final comparison table saved to {md_file}")

    def generate_learning_curves(
        self, round_metrics: dict[str, list[dict[str, Any]]], output_dir: str = "outputs"
    ) -> None:
        """
        Generate learning curves (mAP@0.5 vs round) for all models.

        Args:
            round_metrics: Dictionary mapping config_name to list of round metrics
            output_dir: Directory to save plot
        """
        if not round_metrics:
            self.logger.warning("No round metrics to plot")
            return

        plt.figure(figsize=(10, 6))

        for config_name, rounds in round_metrics.items():
            # Extract round numbers and mAP@0.5 values
            round_nums = [r["round"] for r in rounds]
            map50_values = [r["metrics"]["mAP50"] for r in rounds]

            # Plot learning curve
            plt.plot(round_nums, map50_values, marker="o", label=config_name, linewidth=2)

        plt.xlabel("Round", fontsize=12)
        plt.ylabel("mAP@0.5", fontsize=12)
        plt.title("Learning Curves: mAP@0.5 vs Training Round", fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # Save plot
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        plot_file = output_path / "learning_curves.png"
        plt.savefig(plot_file, dpi=300, bbox_inches="tight")
        plt.close()

        self.logger.info(f"Learning curves saved to {plot_file}")

    def aggregate_all_results(self, outputs_dir: str = "outputs") -> None:
        """
        Aggregate all results: round-level and final test metrics.

        Args:
            outputs_dir: Base outputs directory
        """
        self.logger.info("Aggregating all results...")

        # Collect round-level metrics
        round_metrics = self.collect_round_metrics(outputs_dir)

        # Collect final test metrics
        final_metrics = self.collect_final_test_metrics(outputs_dir)

        # Generate round comparison table
        if round_metrics:
            self.generate_round_comparison_table(round_metrics, outputs_dir)

        # Generate final comparison table
        if final_metrics:
            self.generate_final_comparison_table(final_metrics, outputs_dir)

        # Generate learning curves
        if round_metrics:
            self.generate_learning_curves(round_metrics, outputs_dir)

        self.logger.info("Results aggregation complete")

    def compute_multi_run_statistics(
        self, round_metrics: dict[str, list[list[dict[str, Any]]]]
    ) -> dict[str, list[dict[str, Any]]]:
        """
        Compute mean and standard deviation across multiple runs.

        Args:
            round_metrics: Dictionary mapping config_name to list of runs,
                          where each run is a list of round metrics

        Returns:
            Dictionary mapping config_name to list of aggregated round metrics
        """
        aggregated = {}

        for config_name, runs in round_metrics.items():
            if len(runs) == 1:
                # Single run, no statistics needed
                aggregated[config_name] = runs[0]
                continue

            # Multiple runs, compute statistics
            num_rounds = len(runs[0])
            aggregated_rounds = []

            for round_idx in range(num_rounds):
                # Collect metrics for this round across all runs
                map50_values = [run[round_idx]["metrics"]["mAP50"] for run in runs]
                precision_values = [run[round_idx]["metrics"]["precision"] for run in runs]
                recall_values = [run[round_idx]["metrics"]["recall"] for run in runs]
                f1_values = [run[round_idx]["metrics"]["f1_score"] for run in runs]

                # Compute statistics
                aggregated_round = {
                    "round": round_idx + 1,
                    "metrics": {
                        "mAP50_mean": float(np.mean(map50_values)),
                        "mAP50_std": float(np.std(map50_values)),
                        "precision_mean": float(np.mean(precision_values)),
                        "precision_std": float(np.std(precision_values)),
                        "recall_mean": float(np.mean(recall_values)),
                        "recall_std": float(np.std(recall_values)),
                        "f1_score_mean": float(np.mean(f1_values)),
                        "f1_score_std": float(np.std(f1_values)),
                    },
                    "num_runs": len(runs),
                }

                aggregated_rounds.append(aggregated_round)

            aggregated[config_name] = aggregated_rounds

        return aggregated

    def generate_learning_curves_with_error_bars(
        self, round_metrics: dict[str, list[list[dict[str, Any]]]], output_dir: str = "outputs"
    ) -> None:
        """
        Generate learning curves with error bars for multi-run experiments.

        Args:
            round_metrics: Dictionary mapping config_name to list of runs
            output_dir: Directory to save plot
        """
        if not round_metrics:
            self.logger.warning("No round metrics to plot")
            return

        # Compute statistics
        aggregated = self.compute_multi_run_statistics(round_metrics)

        plt.figure(figsize=(10, 6))

        for config_name, rounds in aggregated.items():
            round_nums = [r["round"] for r in rounds]

            if "mAP50_mean" in rounds[0]["metrics"]:
                # Multi-run with statistics
                mean_values = [r["metrics"]["mAP50_mean"] for r in rounds]
                std_values = [r["metrics"]["mAP50_std"] for r in rounds]

                plt.errorbar(
                    round_nums,
                    mean_values,
                    yerr=std_values,
                    marker="o",
                    label=config_name,
                    linewidth=2,
                    capsize=5,
                )
            else:
                # Single run
                map50_values = [r["metrics"]["mAP50"] for r in rounds]
                plt.plot(round_nums, map50_values, marker="o", label=config_name, linewidth=2)

        plt.xlabel("Round", fontsize=12)
        plt.ylabel("mAP@0.5", fontsize=12)
        plt.title("Learning Curves with Error Bars (Multi-Run)", fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # Save plot
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        plot_file = output_path / "learning_curves_with_error_bars.png"
        plt.savefig(plot_file, dpi=300, bbox_inches="tight")
        plt.close()

        self.logger.info(f"Learning curves with error bars saved to {plot_file}")
