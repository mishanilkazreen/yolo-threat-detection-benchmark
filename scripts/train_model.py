"""
Simple training script to test the pipeline.

Usage:
    python scripts/train_model.py config/models/yolov8n_test.yaml
"""

from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.training.runner import Experiment_Runner


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/train_model.py <config_path>")
        print("\nExample:")
        print("  python scripts/train_model.py config/models/yolov8n_test.yaml")
        sys.exit(1)

    config_path = sys.argv[1]

    if not Path(config_path).exists():
        print(f"Error: Configuration file not found: {config_path}")
        sys.exit(1)

    print(f"Starting training with config: {config_path}")
    print("-" * 60)

    # Run experiment
    runner = Experiment_Runner()
    results = runner.run_experiment(config_path, validate_dataset=True)

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Config: {results.get('config_name', 'N/A')}")
    print(f"mAP@0.5: {results['metrics']['mAP50']:.4f}")
    print(f"mAP@0.5:0.95: {results['metrics']['mAP50-95']:.4f}")
    print(f"Precision: {results['metrics']['precision']:.4f}")
    print(f"Recall: {results['metrics']['recall']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
