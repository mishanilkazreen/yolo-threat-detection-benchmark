"""
Dataset splitter for incremental training.

Creates deterministic train_init, unlabeled_pool, val_fixed, and test_fixed splits
for the incremental training framework.
"""

from collections import defaultdict
import json
import logging
from pathlib import Path
import random
from typing import Any

import numpy as np
import yaml

logger = logging.getLogger(__name__)


class Dataset_Splitter:
    """
    Creates deterministic dataset splits for incremental training.

    Splits the training data into:
    - train_init: Initial training set (e.g., 20% of training data)
    - unlabeled_pool: Remaining training data for progressive expansion (e.g., 80%)
    - val_fixed: Fixed validation set (unchanged across rounds)
    - test_fixed: Fixed test set (unchanged across rounds)
    """

    def __init__(self, random_seed: int = 42):
        """
        Initialize the Dataset_Splitter.

        Args:
            random_seed: Fixed seed for reproducibility across all YOLO versions
        """
        self.random_seed = random_seed
        random.seed(random_seed)

    def create_incremental_splits(
        self,
        data_yaml_path: str,
        train_init_percentage: float = 0.2,
        output_dir: str = "config/data",
    ) -> dict[str, Any]:
        """
        Create deterministic train_init, unlabeled_pool, val_fixed, test_fixed splits.

        Args:
            data_yaml_path: Path to original data.yaml
            train_init_percentage: Percentage of training data for initial round (default: 0.2)
            output_dir: Directory to save split metadata and split files

        Returns:
            Dictionary with split paths and metadata

        Raises:
            FileNotFoundError: If data.yaml or dataset directories not found
            ValueError: If train_init_percentage is not in (0, 1]
        """
        # Validate train_init_percentage
        if not 0 < train_init_percentage <= 1.0:
            raise ValueError(
                f"train_init_percentage must be in (0, 1], got {train_init_percentage}"
            )

        # Load original data.yaml
        data_yaml_file = Path(data_yaml_path)
        if not data_yaml_file.exists():
            raise FileNotFoundError(f"Data YAML not found: {data_yaml_file}")

        with open(data_yaml_file) as f:
            data_config = yaml.safe_load(f)

        # Get dataset paths
        dataset_root = Path(data_config.get("path", ""))
        train_dir = dataset_root / data_config.get("train", "images/train")
        val_dir = dataset_root / data_config.get("val", "images/val")
        test_dir = dataset_root / data_config.get("test", "images/test")

        # Verify directories exist
        for split_name, split_dir in [("train", train_dir), ("val", val_dir), ("test", test_dir)]:
            if not split_dir.exists():
                raise FileNotFoundError(f"{split_name} directory not found: {split_dir}")

        # Get all image files
        train_images = self._get_image_files(train_dir)
        val_images = self._get_image_files(val_dir)
        test_images = self._get_image_files(test_dir)

        logger.info(f"Found {len(train_images)} training images")
        logger.info(f"Found {len(val_images)} validation images")
        logger.info(f"Found {len(test_images)} test images")

        # Verify no duplicates across splits
        self._verify_no_duplicates(train_images, val_images, test_images)

        # Split training data into train_init and unlabeled_pool
        random.shuffle(train_images)  # Shuffle with fixed seed
        train_init_size = max(1, int(len(train_images) * train_init_percentage))
        train_init_images = train_images[:train_init_size]
        unlabeled_pool_images = train_images[train_init_size:]

        logger.info(
            f"Split training data: {len(train_init_images)} train_init, "
            f"{len(unlabeled_pool_images)} unlabeled_pool"
        )

        # Get class distributions
        labels_dir = self._get_labels_dir(train_dir)
        train_init_dist = self._get_class_distribution(train_init_images, labels_dir)
        unlabeled_pool_dist = self._get_class_distribution(unlabeled_pool_images, labels_dir)

        val_labels_dir = self._get_labels_dir(val_dir)
        val_dist = self._get_class_distribution(val_images, val_labels_dir)

        test_labels_dir = self._get_labels_dir(test_dir)
        test_dist = self._get_class_distribution(test_images, test_labels_dir)

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save split files
        splits = {
            "train_init": train_init_images,
            "unlabeled_pool": unlabeled_pool_images,
            "val_fixed": val_images,
            "test_fixed": test_images,
        }

        split_files = {}
        for split_name, images in splits.items():
            split_file = output_path / f"{split_name}.txt"
            with open(split_file, "w") as f:
                for img in images:
                    f.write(f"{img}\n")
            split_files[split_name] = str(split_file)
            logger.info(f"Saved {split_name} to {split_file}")

        # Create split metadata
        metadata = {
            "timestamp": self._get_timestamp(),
            "random_seed": self.random_seed,
            "train_init_percentage": train_init_percentage,
            "data_yaml_path": str(data_yaml_path),
            "splits": {
                "train_init": {
                    "num_images": len(train_init_images),
                    "class_distribution": train_init_dist,
                },
                "unlabeled_pool": {
                    "num_images": len(unlabeled_pool_images),
                    "class_distribution": unlabeled_pool_dist,
                },
                "val_fixed": {"num_images": len(val_images), "class_distribution": val_dist},
                "test_fixed": {"num_images": len(test_images), "class_distribution": test_dist},
            },
        }

        # Save metadata
        metadata_file = output_path / "split_metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved split metadata to {metadata_file}")

        return {
            "split_files": split_files,
            "metadata": metadata,
            "metadata_file": str(metadata_file),
        }

    def _get_image_files(self, image_dir: Path) -> list[str]:
        """
        Get all image files from a directory.

        Args:
            image_dir: Directory containing images

        Returns:
            List of image file paths (relative to image_dir)
        """
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        images = []

        for ext in image_extensions:
            images.extend([str(p.relative_to(image_dir)) for p in image_dir.glob(f"*{ext}")])
            images.extend(
                [str(p.relative_to(image_dir)) for p in image_dir.glob(f"*{ext.upper()}")]
            )

        return sorted(images)

    def _get_labels_dir(self, image_dir: Path) -> Path:
        """
        Get the labels directory corresponding to an image directory.

        Args:
            image_dir: Directory containing images

        Returns:
            Path to labels directory
        """
        # Assume labels are in parallel directory structure
        # e.g., images/train -> labels/train
        labels_dir = Path(str(image_dir).replace("images", "labels"))
        return labels_dir

    def _get_class_distribution(self, images: list[str], labels_dir: Path) -> list[int]:
        """
        Get class distribution for a set of images.

        Args:
            images: List of image filenames
            labels_dir: Directory containing label files

        Returns:
            List of counts per class (index = class_id)
        """
        class_counts: dict[int, int] = defaultdict(int)

        for img_file in images:
            # Get corresponding label file
            label_file = labels_dir / Path(img_file).with_suffix(".txt").name

            if label_file.exists():
                with open(label_file) as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            parts = line.split()
                            if len(parts) >= 5:
                                class_id = int(parts[0])
                                class_counts[class_id] += 1

        # Convert to list (fill missing classes with 0)
        if not class_counts:
            return []

        max_class = max(class_counts.keys())
        distribution = [class_counts.get(i, 0) for i in range(max_class + 1)]

        return distribution

    def _verify_no_duplicates(
        self, train_images: list[str], val_images: list[str], test_images: list[str]
    ) -> None:
        """
        Verify that no image appears in multiple splits.

        Args:
            train_images: Training image filenames
            val_images: Validation image filenames
            test_images: Test image filenames

        Raises:
            ValueError: If duplicate filenames found across splits
        """
        train_set = set(train_images)
        val_set = set(val_images)
        test_set = set(test_images)

        train_val_overlap = train_set & val_set
        train_test_overlap = train_set & test_set
        val_test_overlap = val_set & test_set

        if train_val_overlap or train_test_overlap or val_test_overlap:
            error_msg = "Duplicate filenames found across splits:\n"
            if train_val_overlap:
                error_msg += f"  Train-Val overlap: {list(train_val_overlap)[:5]}\n"
            if train_test_overlap:
                error_msg += f"  Train-Test overlap: {list(train_test_overlap)[:5]}\n"
            if val_test_overlap:
                error_msg += f"  Val-Test overlap: {list(val_test_overlap)[:5]}\n"
            raise ValueError(error_msg)

        logger.info("No duplicate filenames found across splits")

    def _get_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        from datetime import datetime

        return datetime.now().isoformat()

    def split_dataset(
        self,
        train_dir: str,
        train_init_percentage: float = 0.2,
        output_dir: str = "outputs",
        random_seed: int | None = None,
    ) -> dict[str, list[str]]:
        """
        Split dataset into train_init, unlabeled_pool, val_fixed, test_fixed.

        This is a simplified interface for the runner that works with a train directory.

        Args:
            train_dir: Path to training images directory
            train_init_percentage: Percentage of training data for initial round
            output_dir: Directory to save split metadata
            random_seed: Random seed for reproducibility (overrides instance seed)

        Returns:
            Dictionary with keys: train_init, unlabeled_pool, val_fixed, test_fixed
            Each value is a list of image paths
        """
        # Use provided seed or instance seed
        if random_seed is not None:
            self.random_seed = random_seed
            np.random.seed(random_seed)
            random.seed(random_seed)

        # Get all training images
        train_path = Path(train_dir)
        if not train_path.exists():
            raise FileNotFoundError(f"Training directory not found: {train_dir}")

        train_images = self._get_image_files(train_path)

        if not train_images:
            raise ValueError(f"No images found in {train_dir}")

        # Shuffle images deterministically
        train_images_shuffled = train_images.copy()
        np.random.shuffle(train_images_shuffled)

        # Split into train_init and unlabeled_pool
        split_idx = int(len(train_images_shuffled) * train_init_percentage)
        train_init = train_images_shuffled[:split_idx]
        unlabeled_pool = train_images_shuffled[split_idx:]

        # For val_fixed and test_fixed, we need to find val and test directories
        # Assume they are siblings of the train directory
        parent_dir = train_path.parent
        val_dir = parent_dir / "val"
        test_dir = parent_dir / "test"

        val_fixed = []
        test_fixed = []

        if val_dir.exists():
            val_fixed = self._get_image_files(val_dir)
        else:
            logger.warning(f"Validation directory not found: {val_dir}")

        if test_dir.exists():
            test_fixed = self._get_image_files(test_dir)
        else:
            logger.warning(f"Test directory not found: {test_dir}")

        # Save split metadata
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        metadata = {
            "train_init_count": len(train_init),
            "unlabeled_pool_count": len(unlabeled_pool),
            "val_fixed_count": len(val_fixed),
            "test_fixed_count": len(test_fixed),
            "train_init_percentage": train_init_percentage,
            "random_seed": self.random_seed,
        }

        metadata_file = output_path / "split_metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Split metadata saved to {metadata_file}")
        logger.info(f"  train_init: {len(train_init)} images")
        logger.info(f"  unlabeled_pool: {len(unlabeled_pool)} images")
        logger.info(f"  val_fixed: {len(val_fixed)} images")
        logger.info(f"  test_fixed: {len(test_fixed)} images")

        return {
            "train_init": train_init,
            "unlabeled_pool": unlabeled_pool,
            "val_fixed": val_fixed,
            "test_fixed": test_fixed,
        }
