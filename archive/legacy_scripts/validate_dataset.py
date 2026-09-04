"""
Quick dataset validation script.

Validates the downloaded weapon detection dataset before training.
"""

from pathlib import Path
import sys

import yaml

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.validator import Dataset_Validator


def validate_dataset():
    """Validate the weapon detection dataset."""

    print("Dataset Validation")
    print("-" * 60)

    # Load dataset configuration
    config_path = project_root / "config" / "data" / "weapon_detection_data.yaml"

    if not config_path.exists():
        print(f"Error: Configuration file not found: {config_path}")
        return False

    with open(config_path) as f:
        data_config = yaml.safe_load(f)

    # Get dataset path
    dataset_path = Path(data_config["path"])

    if not dataset_path.exists():
        print(f"Error: Dataset path not found: {dataset_path}")
        return False

    print(f"Dataset: {dataset_path}")

    # Create validator
    validator = Dataset_Validator()

    # Validate dataset
    result = validator.validate_dataset(data_config, dataset_path)

    print(f"\nTotal images: {result.total_images}")
    print(f"Total labels: {result.total_labels}")

    if result.is_valid:
        print("\nValidation passed. Dataset is ready for training.")
    else:
        print("\nValidation failed:")

        if result.missing_labels:
            print(f"  Missing labels: {len(result.missing_labels)}")

        if result.empty_labels:
            print(f"  Empty labels: {len(result.empty_labels)}")

        if result.duplicate_files:
            print(f"  Duplicate filenames: {len(result.duplicate_files)}")

    if result.warnings:
        print("\nWarnings:")
        for warning in result.warnings:
            print(f"  {warning}")

    return result.is_valid


if __name__ == "__main__":
    success = validate_dataset()
    sys.exit(0 if success else 1)
