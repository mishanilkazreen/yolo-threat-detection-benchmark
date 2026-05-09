"""Dataset loading module for YOLO format datasets."""

import logging
from pathlib import Path
import random

import yaml

from .validator import Dataset_Validator

logger = logging.getLogger(__name__)


class Dataset_Loader:
    """Loads and prepares YOLO format datasets."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.validator = Dataset_Validator()

    def prepare_dataset(
        self,
        data_yaml_path: str,
        fraction: float = 1.0,
        validate: bool = True,
        seed: int | None = None,
    ) -> tuple[str, float]:
        """
        Prepare dataset for training.

        Args:
            data_yaml_path: Path to data.yaml configuration
            fraction: Fraction of dataset to use (0 < fraction <= 1.0)
            validate: Whether to validate dataset integrity
            seed: Random seed for subset selection (for reproducibility)

        Returns:
            Tuple of (resolved_data_yaml_path, fraction_used)
        """
        self.logger.info(f"Preparing dataset from {data_yaml_path}")

        # Load data.yaml
        with open(data_yaml_path, encoding="utf-8") as f:
            data_config = yaml.safe_load(f)

        # Get base path
        base_path = Path(data_config.get("path", Path(data_yaml_path).parent))

        # Validate dataset if requested
        if validate:
            self.logger.info("Validating dataset integrity...")
            validation_result = self.validator.validate_dataset(data_config, base_path)

            if not validation_result.is_valid:
                self.logger.warning("Dataset validation found issues, but continuing...")

        # Handle fraction-based subset
        if fraction < 1.0:
            self.logger.info(f"Creating subset with fraction={fraction}")
            resolved_yaml_path = self._create_subset(data_config, base_path, fraction, seed)
            return resolved_yaml_path, fraction
        else:
            return data_yaml_path, 1.0

    def _create_subset(
        self, data_config: dict, base_path: Path, fraction: float, seed: int | None = None
    ) -> str:
        """
        Create a subset of the dataset.

        Args:
            data_config: Parsed data.yaml configuration
            base_path: Base path to dataset
            fraction: Fraction of dataset to use
            seed: Random seed for reproducibility

        Returns:
            Path to generated subset data.yaml
        """
        if seed is not None:
            random.seed(seed)

        subset_config = data_config.copy()
        subset_files = {}

        # Process each split
        for split in ["train", "val", "test"]:
            if split not in data_config:
                continue

            # Get image directory
            images_dir = base_path / data_config[split]
            if not images_dir.exists():
                self.logger.warning(f"Image directory not found: {images_dir}")
                continue

            # Find all image files
            image_files = self._find_image_files(images_dir)

            # Calculate subset size (at least 1 image)
            subset_size = max(1, int(len(image_files) * fraction))

            # Randomly select subset
            subset_images = random.sample(image_files, subset_size)

            # Create subset file list
            subset_file_path = base_path / f"{split}_subset.txt"
            with open(subset_file_path, "w", encoding="utf-8") as f:
                for img_path in subset_images:
                    f.write(f"{img_path}\n")

            subset_files[split] = str(subset_file_path.relative_to(base_path))

            self.logger.info(f"  {split}: {len(subset_images)}/{len(image_files)} images")

        # Update config with subset file lists
        for split, file_path in subset_files.items():
            subset_config[split] = file_path

        # Save subset config
        subset_yaml_path = base_path / "data_subset.yaml"
        with open(subset_yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(subset_config, f, default_flow_style=False, sort_keys=False)

        self.logger.info(f"Created subset configuration at {subset_yaml_path}")

        return str(subset_yaml_path)

    def _find_image_files(self, images_dir: Path) -> list[Path]:
        """
        Find all image files in a directory.

        Args:
            images_dir: Directory containing images

        Returns:
            List of image file paths
        """
        image_extensions = [".jpg", ".jpeg", ".png", ".bmp"]
        image_files: list[Path] = []

        for ext in image_extensions:
            image_files.extend(images_dir.glob(f"*{ext}"))
            image_files.extend(images_dir.glob(f"*{ext.upper()}"))

        return sorted(image_files)

    def load_splits(self, data_yaml_path: str) -> tuple[str | None, str | None, str | None]:
        """
        Load train/val/test split paths from data.yaml.

        Args:
            data_yaml_path: Path to data.yaml configuration

        Returns:
            Tuple of (train_path, val_path, test_path)
        """
        with open(data_yaml_path, encoding="utf-8") as f:
            data_config = yaml.safe_load(f)

        base_path = Path(data_config.get("path", Path(data_yaml_path).parent))

        train_path = None
        val_path = None
        test_path = None

        if "train" in data_config:
            train_path = str(base_path / data_config["train"])

        if "val" in data_config:
            val_path = str(base_path / data_config["val"])

        if "test" in data_config:
            test_path = str(base_path / data_config["test"])

        return train_path, val_path, test_path

    def verify_no_duplicates(self, data_yaml_path: str) -> bool:
        """
        Verify no duplicate filenames across splits.

        Args:
            data_yaml_path: Path to data.yaml configuration

        Returns:
            True if no duplicates found, False otherwise
        """
        with open(data_yaml_path, encoding="utf-8") as f:
            data_config = yaml.safe_load(f)

        base_path = Path(data_config.get("path", Path(data_yaml_path).parent))

        # Use validator to check for duplicates
        validation_result = self.validator.validate_dataset(data_config, base_path)

        return len(validation_result.duplicate_files) == 0
