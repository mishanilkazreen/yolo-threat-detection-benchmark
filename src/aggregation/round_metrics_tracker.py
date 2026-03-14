"""
Round metrics tracker for incremental training.

Tracks and aggregates metrics evolution across training rounds.
"""

import csv
import json
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


class Round_Metrics_Tracker:
    """
    Tracks metrics evolution across training rounds.

    Key responsibilities:
    - Aggregate metrics across all rounds into all_rounds_metrics.json
    - Export round-level metrics to CSV for plotting
    - Track HFS evolution across rounds
    - Generate learning curves (mAP@0.5 vs round)
    - Generate HFS evolution plots
    """

    def __init__(self):
        """Initialize the Round_Metrics_Tracker."""
        pass

    def track_round_metrics(self, round_num: int, metrics: dict[str, Any], output_dir: str) -> None:
        """
        Track metrics for a specific training round.

        Args:
            round_num: Round number (1-5)
            metrics: Dictionary of metrics for this round
            output_dir: Directory to save round metrics
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save round-specific metrics
        round_file = output_path / f"round_{round_num}_metrics.json"
        with open(round_file, "w") as f:
            json.dump(metrics, f, indent=2)

        logger.info(f"Saved round {round_num} metrics to {round_file}")

    def aggregate_all_rounds(
        self, config_name: str, output_dir: str, num_rounds: int = 5
    ) -> dict[str, Any]:
        """
        Aggregate metrics across all rounds.

        Args:
            config_name: Configuration name
            output_dir: Directory containing round metrics
            num_rounds: Number of rounds to aggregate

        Returns:
            Dictionary with aggregated metrics and learning curves
        """
        output_path = Path(output_dir)

        # Load all round metrics
        rounds_data = []
        for round_num in range(1, num_rounds + 1):
            round_file = output_path / f"round_{round_num}_metrics.json"

            if round_file.exists():
                with open(round_file) as f:
                    round_metrics = json.load(f)
                    rounds_data.append(round_metrics)
            else:
                logger.warning(f"Round {round_num} metrics not found: {round_file}")

        if not rounds_data:
            logger.error(f"No round metrics found in {output_dir}")
            return {}

        # Aggregate metrics
        aggregated = {
            "config_name": config_name,
            "num_rounds": len(rounds_data),
            "rounds": rounds_data,
        }

        # Save aggregated metrics
        agg_file = output_path / "all_rounds_metrics.json"
        with open(agg_file, "w") as f:
            json.dump(aggregated, f, indent=2)

        logger.info(f"Saved aggregated metrics to {agg_file}")

        # Export to CSV
        self._export_to_csv(rounds_data, output_path / "all_rounds_metrics.csv")

        return aggregated

    def _export_to_csv(self, rounds_data: list[dict[str, Any]], csv_path: Path) -> None:
        """
        Export round-level metrics to CSV for plotting.

        Args:
            rounds_data: List of round metrics dictionaries
            csv_path: Path to save CSV file
        """
        if not rounds_data:
            return

        # Define CSV columns
        columns = [
            "round",
            "training_set_size",
            "verified_samples_added",
            "unlabeled_pool_remaining",
            "mAP50",
            "mAP50-95",
            "precision",
            "recall",
            "f1_score",
            "fitness",
            "inference_time_ms",
            "round_training_time_seconds",
            "cumulative_training_time_seconds",
        ]

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()

            for round_data in rounds_data:
                row = {
                    "round": round_data.get("round", 0),
                    "training_set_size": round_data.get("training_set_size", 0),
                    "verified_samples_added": round_data.get("verified_samples_added", 0),
                    "unlabeled_pool_remaining": round_data.get("unlabeled_pool_remaining", 0),
                }

                # Extract metrics
                metrics = round_data.get("metrics", {})
                row["mAP50"] = metrics.get("mAP50", 0.0)
                row["mAP50-95"] = metrics.get("mAP50-95", 0.0)
                row["precision"] = metrics.get("precision", 0.0)
                row["recall"] = metrics.get("recall", 0.0)
                row["f1_score"] = metrics.get("f1_score", 0.0)
                row["fitness"] = metrics.get("fitness", 0.0)

                # Extract model info
                model_info = round_data.get("model_info", {})
                row["inference_time_ms"] = model_info.get("inference_time_ms", 0.0)

                # Extract timing
                row["round_training_time_seconds"] = round_data.get(
                    "round_training_time_seconds", 0.0
                )
                row["cumulative_training_time_seconds"] = round_data.get(
                    "cumulative_training_time_seconds", 0.0
                )

                writer.writerow(row)

        logger.info(f"Exported metrics to CSV: {csv_path}")

    def generate_learning_curves(
        self, configs_data: dict[str, list[dict[str, Any]]], output_path: str, metric: str = "mAP50"
    ) -> None:
        """
        Generate learning curves (metric vs round) for all models.

        Args:
            configs_data: Dictionary mapping config_name to list of round metrics
            output_path: Path to save learning curve plot
            metric: Metric to plot (default: mAP50)
        """
        plt.figure(figsize=(10, 6))

        for config_name, rounds_data in configs_data.items():
            if not rounds_data:
                continue

            rounds = [r.get("round", 0) for r in rounds_data]
            values = [r.get("metrics", {}).get(metric, 0.0) for r in rounds_data]

            plt.plot(rounds, values, marker="o", label=config_name, linewidth=2)

        plt.xlabel("Round", fontsize=12)
        plt.ylabel(metric, fontsize=12)
        plt.title(f"{metric} Evolution Across Training Rounds", fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved learning curves to {output_path}")

    def track_hfs_evolution(
        self, config_name: str, round_num: int, hfs_metrics: dict[str, float], output_dir: str
    ) -> None:
        """
        Track HFS evolution across rounds.

        Args:
            config_name: Configuration name
            round_num: Round number
            hfs_metrics: HFS metrics for this round
            output_dir: Directory to save HFS evolution data
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Load existing HFS evolution data
        hfs_file = output_path / "hfs_evolution.json"
        if hfs_file.exists():
            with open(hfs_file) as f:
                hfs_data = json.load(f)
        else:
            hfs_data = {"config_name": config_name, "rounds": []}

        # Add current round HFS
        round_hfs = {"round": round_num, **hfs_metrics}
        hfs_data["rounds"].append(round_hfs)

        # Save updated HFS evolution
        with open(hfs_file, "w") as f:
            json.dump(hfs_data, f, indent=2)

        logger.info(f"Tracked HFS for round {round_num} in {hfs_file}")

    def generate_hfs_evolution_plot(
        self, configs_hfs_data: dict[str, dict[str, Any]], output_path: str
    ) -> None:
        """
        Generate HFS evolution plots (HFS vs round) for all models.

        Args:
            configs_hfs_data: Dictionary mapping config_name to HFS evolution data
            output_path: Path to save HFS evolution plot
        """
        plt.figure(figsize=(10, 6))

        for config_name, hfs_data in configs_hfs_data.items():
            rounds_data = hfs_data.get("rounds", [])
            if not rounds_data:
                continue

            rounds = [r.get("round", 0) for r in rounds_data]
            mean_hfs = [r.get("mean_hfs", 0.0) for r in rounds_data]

            plt.plot(rounds, mean_hfs, marker="o", label=config_name, linewidth=2)

        plt.xlabel("Round", fontsize=12)
        plt.ylabel("Mean HFS", fontsize=12)
        plt.title("Heatmap Focus Score Evolution Across Training Rounds", fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved HFS evolution plot to {output_file}")

    def compute_hfs_map_correlation(
        self, rounds_data: list[dict[str, Any]], hfs_data: dict[str, Any]
    ) -> float:
        """
        Compute correlation between HFS improvement and mAP@0.5 improvement across rounds.

        Args:
            rounds_data: List of round metrics
            hfs_data: HFS evolution data

        Returns:
            Correlation coefficient
        """
        if not rounds_data or not hfs_data.get("rounds"):
            return 0.0

        # Extract mAP@0.5 values
        map_values = [r.get("metrics", {}).get("mAP50", 0.0) for r in rounds_data]

        # Extract HFS values
        hfs_rounds = hfs_data.get("rounds", [])
        hfs_values = [r.get("mean_hfs", 0.0) for r in hfs_rounds]

        # Ensure same length
        min_len = min(len(map_values), len(hfs_values))
        if min_len < 2:
            return 0.0

        map_values = map_values[:min_len]
        hfs_values = hfs_values[:min_len]

        # Compute correlation
        correlation = np.corrcoef(map_values, hfs_values)[0, 1]

        return float(correlation)

    def aggregate_round_metrics(
        self, round_metrics_list: list[dict[str, Any]], output_dir: str, config_name: str
    ) -> None:
        """
        Aggregate round metrics (simplified interface).

        Args:
            round_metrics_list: List of round metrics dictionaries
            output_dir: Directory to save aggregated metrics
            config_name: Configuration name
        """
        # Use the number of rounds from the list
        num_rounds = len(round_metrics_list)
        self.aggregate_all_rounds(
            config_name=config_name, output_dir=output_dir, num_rounds=num_rounds
        )
