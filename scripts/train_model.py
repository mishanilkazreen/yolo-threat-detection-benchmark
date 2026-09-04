"""
Production training runner supporting multi-seed sweeps, device overrides, and cluster execution.

Usage:
    uv run python scripts/train_model.py config/models/yolov8n_random.yaml --seed 42 --device 0
"""

import argparse
import logging
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Add yolo_cam to path (cloned from YOLO-26-CAM repository)
yolo_cam_path = project_root / "yolo_cam"
if yolo_cam_path.exists():
    sys.path.insert(0, str(yolo_cam_path))

from src.training.runner import Experiment_Runner


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train YOLO incremental threat detection benchmark with seed and compute tracking."
    )
    parser.add_argument("config", type=str, help="Path to YAML model configuration file")
    parser.add_argument("--seed", type=int, default=None, help="Random seed override (e.g. 42, 123, 456)")
    parser.add_argument("--device", type=str, default=None, help="Device override (e.g. '0', 'cpu')")
    parser.add_argument("--epochs", type=int, default=None, help="Epoch count override")
    parser.add_argument("--runs", type=int, default=None, help="Number of multi-seed runs")
    parser.add_argument(
        "--no-validate-dataset", action="store_true", help="Skip dataset validation check"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = Path(args.config)

    if not config_path.exists():
        print(f"Error: Configuration file not found: {config_path}")
        sys.exit(1)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    print(f"Starting training with config: {config_path}")
    if args.seed is not None:
        print(f"  Enforcing configured seed: {args.seed}")
    if args.device is not None:
        print(f"  Target device: {args.device}")
    print("-" * 60)

    # Run experiment
    runner = Experiment_Runner()
    results = runner.run_experiment(
        str(config_path),
        validate_dataset=not args.no_validate_dataset,
        seed=args.seed,
        device=args.device,
        epochs=args.epochs,
        runs=args.runs,
    )

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Config: {results.get('config_name', 'N/A')}")
    metrics = results.get("metrics", {})
    if "mAP50" in metrics:
        print(f"mAP@0.5: {metrics['mAP50']:.4f}")
        print(f"mAP@0.5:0.95: {metrics.get('mAP50-95', 0.0):.4f}")
        print(f"Precision: {metrics.get('precision', 0.0):.4f}")
        print(f"Recall: {metrics.get('recall', 0.0):.4f}")
    elif "mAP50_mean" in metrics:
        print(
            f"mAP@0.5 (Mean ± SD): {metrics['mAP50_mean']:.4f} ± {metrics.get('mAP50_std', 0.0):.4f}"
        )
    print("=" * 60)


if __name__ == "__main__":
    main()
