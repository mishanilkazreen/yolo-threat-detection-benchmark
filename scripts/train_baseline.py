"""
One-shot baseline training script with full compute and stopping epoch tracking.

Trains a YOLO model in a single pass on the full training partition (train_init + unlabeled_pool)
using the same 70/20/10 split as incremental experiments.

Usage:
    uv run python scripts/train_baseline.py config/models/yolov8n_baseline_pretrained.yaml --seed 42 --epochs 500 --patience 50 --device 0
"""

import argparse
import logging
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

yolo_cam_path = project_root / "yolo_cam"
if yolo_cam_path.exists():
    sys.path.insert(0, str(yolo_cam_path))

from src.training.baseline_runner import Baseline_Runner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="One-shot YOLO baseline training with compute and stopping epoch accounting."
    )
    parser.add_argument("config", type=str, help="Path to YAML baseline configuration file")
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed override (e.g. 42, 123, 456)"
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Device override (e.g. '0', 'cpu')"
    )
    parser.add_argument(
        "--epochs", type=int, default=None, help="Baseline epochs override (e.g. 500)"
    )
    parser.add_argument(
        "--patience", type=int, default=None, help="Early stopping patience override (e.g. 50)"
    )
    parser.add_argument(
        "--no-validate-dataset", action="store_true", help="Skip dataset validation check"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)

    if not config_path.exists():
        print(f"Error: Configuration file not found: {config_path}")
        sys.exit(1)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    print(f"Starting baseline training with config: {config_path}")
    if args.seed is not None:
        print(f"  Enforcing configured seed: {args.seed}")
    if args.device is not None:
        print(f"  Target device: {args.device}")
    if args.epochs is not None:
        print(f"  Target baseline epochs: {args.epochs}")
    if args.patience is not None:
        print(f"  Early stopping patience: {args.patience}")
    print("-" * 60)

    runner = Baseline_Runner()
    result = runner.run(
        str(config_path),
        validate_dataset=not args.no_validate_dataset,
        seed=args.seed,
        device=args.device,
        epochs=args.epochs,
        patience=args.patience,
    )

    test_inner = result.get("test_metrics", {}).get("test_metrics", {})
    val_inner = result.get("val_metrics", {})

    print("\n" + "=" * 60)
    print("Baseline Training Complete")
    print("=" * 60)
    print(f"Config:                 {result['config_name']}")
    print(f"Checkpoint:             {result['checkpoint_path']}")
    print(f"Training set size:      {result['training_set_size']} images")
    print(f"Nominal Epochs:         {result['epochs']}")
    print(f"Actual Stopped Epoch:   {result.get('actual_stopped_epoch', 'N/A')}")
    print(f"Best Epoch:             {result.get('best_epoch', 'N/A')}")
    print(f"Patience:               {result.get('patience', 'N/A')}")
    print(f"Optimizer Steps:        {result.get('total_optimizer_steps', 'N/A')}")
    print(f"Images Processed:       {result.get('total_images_processed', 'N/A')}")
    print(f"Training TFLOPs:        {result.get('total_training_tflops', 'N/A')}")
    print(f"Training time (s):      {result['training_time_seconds']:.2f}")
    print("-- Validation --")
    print(f"  mAP@0.5:              {val_inner.get('mAP50', float('nan')):.4f}")
    print(f"  mAP@0.5:0.95:         {val_inner.get('mAP50-95', float('nan')):.4f}")
    print(f"  Precision:            {val_inner.get('precision', float('nan')):.4f}")
    print(f"  Recall:               {val_inner.get('recall', float('nan')):.4f}")
    print(f"  F1:                   {val_inner.get('f1_score', float('nan')):.4f}")
    print("-- Test --")
    print(f"  mAP@0.5:              {test_inner.get('mAP50', float('nan')):.4f}")
    print(f"  mAP@0.5:0.95:         {test_inner.get('mAP50-95', float('nan')):.4f}")
    print(f"  Precision:            {test_inner.get('precision', float('nan')):.4f}")
    print(f"  Recall:               {test_inner.get('recall', float('nan')):.4f}")
    print(f"  F1:                   {test_inner.get('f1_score', float('nan')):.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
