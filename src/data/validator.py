"""Dataset validation module for checking dataset integrity."""

from dataclasses import dataclass
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of dataset validation."""

    total_images: int
    total_labels: int
    missing_labels: list[str]
    empty_labels: list[str]
    duplicate_files: list[str]
    warnings: list[str]

    @property
    def is_valid(self) -> bool:
        """Check if validation passed without errors."""
        return not (self.missing_labels or self.empty_labels or self.duplicate_files)


class Dataset_Validator:
    """Validates dataset integrity before training."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def validate_dataset(self, data_config: dict, base_path: Path) -> ValidationResult:
        """
        Validate dataset integrity.

        Args:
            data_config: Parsed data.yaml configuration
            base_path: Base path to dataset

        Returns:
            ValidationResult with warnings and errors
        """
        missing_labels = []
        empty_labels = []
        duplicate_files = []
        warnings = []

        total_images = 0
        total_labels = 0

        # Track filenames across splits to detect duplicates
        all_filenames: dict[str, list[str]] = {}

        # Validate each split
        for split in ["train", "val", "test"]:
            if split not in data_config:
                continue

            # Get image directory
            images_dir = base_path / data_config[split]
            if not images_dir.exists():
                warnings.append(f"Image directory not found: {images_dir}")
                continue

            # Get label directory (replace 'images' with 'labels')
            # Handle both forward and backward slashes, and ensure proper Path handling
            images_dir_str = str(images_dir)
            if "\\images\\" in images_dir_str:
                labels_dir = Path(images_dir_str.replace("\\images\\", "\\labels\\"))
            elif "/images/" in images_dir_str:
                labels_dir = Path(images_dir_str.replace("/images/", "/labels/"))
            elif images_dir.name == "images":
                # If the directory itself is named 'images', replace it with 'labels'
                labels_dir = images_dir.parent / "labels"
            else:
                # Fallback: assume labels are in a sibling directory
                labels_dir = images_dir.parent / "labels"

            # Find all image files
            image_extensions = [".jpg", ".jpeg", ".png", ".bmp"]
            image_files_set: set[Path] = set()  # Use set to avoid duplicates
            for ext in image_extensions:
                image_files_set.update(images_dir.glob(f"*{ext}"))
                image_files_set.update(images_dir.glob(f"*{ext.upper()}"))
            image_files: list[Path] = list(image_files_set)

            total_images += len(image_files)

            # Check each image for corresponding label
            for image_path in image_files:
                filename = image_path.stem

                # Track duplicates across splits
                if filename in all_filenames:
                    all_filenames[filename].append(split)
                else:
                    all_filenames[filename] = [split]

                # Check for label file
                label_path = labels_dir / f"{filename}.txt"

                if not label_path.exists():
                    missing_labels.append(str(image_path))
                else:
                    total_labels += 1

                    # Check if label file is empty
                    if label_path.stat().st_size == 0:
                        empty_labels.append(str(label_path))

        # Find duplicates
        for filename, splits in all_filenames.items():
            if len(splits) > 1:
                duplicate_files.append(f"{filename} appears in: {', '.join(splits)}")

        # Create result
        result = ValidationResult(
            total_images=total_images,
            total_labels=total_labels,
            missing_labels=missing_labels,
            empty_labels=empty_labels,
            duplicate_files=duplicate_files,
            warnings=warnings,
        )

        # Log results
        self._log_validation_results(result)

        return result

    def _log_validation_results(self, result: ValidationResult) -> None:
        """Log validation results with warnings."""
        self.logger.info("Dataset validation complete:")
        self.logger.info(f"  Total images: {result.total_images}")
        self.logger.info(f"  Total labels: {result.total_labels}")

        if result.missing_labels:
            self.logger.warning(f"  Missing labels: {len(result.missing_labels)}")
            for path in result.missing_labels[:5]:  # Show first 5
                self.logger.warning(f"    - {path}")
            if len(result.missing_labels) > 5:
                self.logger.warning(f"    ... and {len(result.missing_labels) - 5} more")

        if result.empty_labels:
            self.logger.warning(f"  Empty annotation files: {len(result.empty_labels)}")
            for path in result.empty_labels[:5]:
                self.logger.warning(f"    - {path}")
            if len(result.empty_labels) > 5:
                self.logger.warning(f"    ... and {len(result.empty_labels) - 5} more")

        if result.duplicate_files:
            self.logger.warning(
                f"  Duplicate filenames across splits: {len(result.duplicate_files)}"
            )
            for dup in result.duplicate_files[:5]:
                self.logger.warning(f"    - {dup}")
            if len(result.duplicate_files) > 5:
                self.logger.warning(f"    ... and {len(result.duplicate_files) - 5} more")

        for warning in result.warnings:
            self.logger.warning(f"  {warning}")

        if result.is_valid:
            self.logger.info("  Validation passed")
        else:
            self.logger.error("  Validation failed")
