"""
One-shot baseline training script (no incremental rounds).

Trains a YOLO model in a single pass on the same 70/20/10 split used by the
incremental experiments, so baseline and incremental runs at the same seed are
directly comparable.

Usage:
    python scripts/train_baseline.py config/models/yolov8n_baseline_pretrained.yaml
"""

from pathlib import Path
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

yolo_cam_path = project_root / "yolo_cam"
if yolo_cam_path.exists():
    sys.path.insert(0, str(yolo_cam_path))

from src.training.baseline_runner import Baseline_Runner


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python scripts/train_baseline.py <config_path>")
        print("\nExample:")
        print("  python scripts/train_baseline.py config/models/yolov8n_baseline_pretrained.yaml")
        sys.exit(1)

    config_path = sys.argv[1]
    if not Path(config_path).exists():
        print(f"Error: Configuration file not found: {config_path}")
        sys.exit(1)

    print(f"Starting baseline training with config: {config_path}")
    print("-" * 60)

    runner = Baseline_Runner()
    result = runner.run(config_path, validate_dataset=True)

    test_inner = result["test_metrics"].get("test_metrics", {})
    val_inner = result["val_metrics"]

    print("\n" + "=" * 60)
    print("Baseline Training Complete")
    print("=" * 60)
    print(f"Config:               {result['config_name']}")
    print(f"Checkpoint:           {result['checkpoint_path']}")
    print(f"Training set size:    {result['training_set_size']}")
    print(f"Epochs:               {result['epochs']}")
    print(f"Training time (s):    {result['training_time_seconds']:.2f}")
    print("-- Validation --")
    print(f"  mAP@0.5:            {val_inner.get('mAP50', float('nan')):.4f}")
    print(f"  mAP@0.5:0.95:       {val_inner.get('mAP50-95', float('nan')):.4f}")
    print(f"  Precision:          {val_inner.get('precision', float('nan')):.4f}")
    print(f"  Recall:             {val_inner.get('recall', float('nan')):.4f}")
    print(f"  F1:                 {val_inner.get('f1_score', float('nan')):.4f}")
    print("-- Test --")
    print(f"  mAP@0.5:            {test_inner.get('mAP50', float('nan')):.4f}")
    print(f"  mAP@0.5:0.95:       {test_inner.get('mAP50-95', float('nan')):.4f}")
    print(f"  Precision:          {test_inner.get('precision', float('nan')):.4f}")
    print(f"  Recall:             {test_inner.get('recall', float('nan')):.4f}")
    print(f"  F1:                 {test_inner.get('f1_score', float('nan')):.4f}")
    print(f"-- HFS mean:          {result['hfs_metrics'].get('mean_hfs', float('nan')):.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
