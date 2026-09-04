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
        Create deterministic train / val / test / train_init / unlabeled_pool splits.

        The Roboflow download provides only ``train/`` and ``valid/`` directories
        with no separate test set.  This method therefore pools *all* 5,064 images
        and re-partitions them from scratch using a stratified 70 / 20 / 10 split:

        - train  (70 %) → 3,543 images  → further split into train_init (20 %) and
                                           unlabeled_pool (80 %)
        - val    (20 %) → 1,013 images  → val_fixed (unchanged across rounds)
        - test   (10 %) →   508 images  → test_fixed (unchanged across rounds)

        All splits are stratified by the first class label in each annotation file
        and are fully deterministic given ``random_seed``.

        Args:
            data_yaml_path: Path to the project data YAML
                (must point to a dataset root that contains ``train/images`` and
                ``valid/images``; a ``test/images`` directory is optional and
                ignored — the test split is carved out of the full pool here).
            train_init_percentage: Fraction of the *train* partition used as the
                initial labelled set (default 0.2 → 709 images).
            output_dir: Directory where split ``.txt`` files and metadata are saved.

        Returns:
            Dictionary with keys ``split_files``, ``metadata``, ``metadata_file``.

        Raises:
            FileNotFoundError: If data.yaml or required image directories are missing.
            ValueError: If ``train_init_percentage`` is outside (0, 1] or the total
                image count does not equal 5,064.
        """
        if not 0 < train_init_percentage <= 1.0:
            raise ValueError(
                f"train_init_percentage must be in (0, 1], got {train_init_percentage}"
            )

        data_yaml_file = Path(data_yaml_path)
        if not data_yaml_file.exists():
            raise FileNotFoundError(f"Data YAML not found: {data_yaml_file}")

        with open(data_yaml_file, encoding="utf-8") as f:
            data_config = yaml.safe_load(f)

        dataset_root = Path(data_config.get("path", ""))

        # Collect images from every available split directory.
        # The Roboflow download uses "train" and "valid"; there is no "test" dir.
        candidate_dirs = [
            dataset_root / data_config.get("train", "train/images"),
            dataset_root / data_config.get("val", "valid/images"),
        ]
        # Include a test dir only if it actually exists (future-proofing).
        test_key = data_config.get("test", "test/images")
        test_candidate = dataset_root / test_key
        if test_candidate.exists():
            candidate_dirs.append(test_candidate)

        # Verify at least the mandatory directories exist
        for d in candidate_dirs[:2]:
            if not d.exists():
                raise FileNotFoundError(f"Image directory not found: {d}")

        # Gather all images into one deduplicated pool.
        # We tag each filename with its source labels directory so we can look up
        # annotations regardless of which Roboflow sub-folder it came from.
        all_images: list[str] = []
        # Map filename → labels_dir (first occurrence wins on collision)
        filename_to_labels: dict[str, Path] = {}

        for img_dir in candidate_dirs:
            labels_dir = self._get_labels_dir(img_dir)
            for fname in self._get_image_files(img_dir):
                if fname not in filename_to_labels:
                    all_images.append(fname)
                    filename_to_labels[fname] = labels_dir

        total = len(all_images)
        logger.info(f"Full image pool: {total} unique images")

        if total != 5064:
            raise ValueError(
                f"Dataset total image count is {total}, expected 5064. "
                "Check that the dataset download is complete and no images are missing."
            )

        # ── Step 1: stratified 70 / 20 / 10 split of the full pool ──────────
        # We need a single unified labels lookup for the stratified split.
        # Build a temporary labels dir that the helper can use by passing a
        # sentinel; instead we use a thin wrapper approach: create a temporary
        # merged labels dir mapping or override _stratified_split inline.
        #
        # Simpler: build a temporary directory of symlinks / copies is fragile on
        # Windows.  Instead we inline the grouping here using filename_to_labels.

        # Group images by first class label using the per-file labels mapping.
        class_groups: dict[int, list[str]] = defaultdict(list)
        unlabeled_pool_imgs: list[str] = []

        for fname in all_images:
            labels_dir = filename_to_labels[fname]
            label_file = labels_dir / Path(fname).with_suffix(".txt").name
            class_id: int | None = None
            if label_file.exists():
                with open(label_file, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            parts = line.split()
                            if parts:
                                class_id = int(parts[0])
                                break
            if class_id is not None:
                class_groups[class_id].append(fname)
            else:
                unlabeled_pool_imgs.append(fname)

        rng = random.Random(self.random_seed)

        train_images: list[str] = []
        val_images: list[str] = []
        test_images: list[str] = []

        for cid in sorted(class_groups.keys()):
            group = class_groups[cid][:]
            rng.shuffle(group)
            n = len(group)
            n_train = round(n * 0.70)
            n_val = round(n * 0.20)
            # test gets the remainder so counts always sum to n
            train_images.extend(group[:n_train])
            val_images.extend(group[n_train : n_train + n_val])
            test_images.extend(group[n_train + n_val :])

        # Distribute unlabeled images with the same 70/20/10 proportions
        if unlabeled_pool_imgs:
            rng.shuffle(unlabeled_pool_imgs)
            n = len(unlabeled_pool_imgs)
            n_train = round(n * 0.70)
            n_val = round(n * 0.20)
            train_images.extend(unlabeled_pool_imgs[:n_train])
            val_images.extend(unlabeled_pool_imgs[n_train : n_train + n_val])
            test_images.extend(unlabeled_pool_imgs[n_train + n_val :])

        logger.info(
            f"70/20/10 split: {len(train_images)} train, "
            f"{len(val_images)} val, {len(test_images)} test"
        )

        # ── Step 2: split train into train_init + unlabeled_pool ─────────────
        # Re-use _stratified_split but we need a unified labels lookup.
        # Build a temporary merged labels dir is not portable; instead replicate
        # the stratified logic inline using filename_to_labels.

        train_class_groups: dict[int, list[str]] = defaultdict(list)
        train_unlabeled: list[str] = []

        for fname in train_images:
            labels_dir = filename_to_labels[fname]
            label_file = labels_dir / Path(fname).with_suffix(".txt").name
            class_id = None
            if label_file.exists():
                with open(label_file, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            parts = line.split()
                            if parts:
                                class_id = int(parts[0])
                                break
            if class_id is not None:
                train_class_groups[class_id].append(fname)
            else:
                train_unlabeled.append(fname)

        rng2 = random.Random(self.random_seed)
        train_init_images: list[str] = []
        unlabeled_pool_images: list[str] = []

        for cid in sorted(train_class_groups.keys()):
            group = train_class_groups[cid][:]
            rng2.shuffle(group)
            split_idx = max(1, int(len(group) * train_init_percentage)) if group else 0
            train_init_images.extend(group[:split_idx])
            unlabeled_pool_images.extend(group[split_idx:])

        if train_unlabeled:
            rng2.shuffle(train_unlabeled)
            split_idx = max(1, int(len(train_unlabeled) * train_init_percentage))
            train_init_images.extend(train_unlabeled[:split_idx])
            unlabeled_pool_images.extend(train_unlabeled[split_idx:])

        logger.info(
            f"train_init: {len(train_init_images)}, unlabeled_pool: {len(unlabeled_pool_images)}"
        )

        # ── Step 3: compute class distributions ──────────────────────────────
        def _dist(imgs: list[str]) -> list[int]:
            counts: dict[int, int] = defaultdict(int)
            for fname in imgs:
                ldir = filename_to_labels[fname]
                lf = ldir / Path(fname).with_suffix(".txt").name
                if lf.exists():
                    with open(lf, encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                parts = line.split()
                                if len(parts) >= 5:
                                    counts[int(parts[0])] += 1
            if not counts:
                return []
            return [counts.get(i, 0) for i in range(max(counts) + 1)]

        # ── Step 4: persist split files ───────────────────────────────────────
        # Write *absolute* image paths so that YOLO (and _create_round_data_yaml)
        # can locate each file regardless of which Roboflow sub-directory it
        # physically lives in.  Images from valid/images that were reassigned to
        # the train partition would be silently skipped if we wrote bare filenames
        # and the caller prepended train/images/.
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Build a map from filename → absolute image path using the source dirs
        # we already collected in filename_to_labels.
        filename_to_img_dir: dict[str, Path] = {}
        for img_dir in candidate_dirs:
            for fname in self._get_image_files(img_dir):
                if fname not in filename_to_img_dir:
                    filename_to_img_dir[fname] = img_dir.resolve()

        def _abs_path(fname: str) -> str:
            img_dir = filename_to_img_dir.get(fname)
            if img_dir is None:
                # Fallback: shouldn't happen, but keep the bare name so the
                # caller can surface a clear "not found" error from YOLO.
                return fname
            return str(img_dir / fname)

        splits_map = {
            "train_init": train_init_images,
            "unlabeled_pool": unlabeled_pool_images,
            "val_fixed": val_images,
            "test_fixed": test_images,
        }

        split_files: dict[str, str] = {}
        for split_name, imgs in splits_map.items():
            split_file = output_path / f"{split_name}.txt"
            with open(split_file, "w", encoding="utf-8") as f:
                for img in imgs:
                    f.write(f"{_abs_path(img)}\n")
            split_files[split_name] = str(split_file)
            logger.info(f"Saved {split_name} ({len(imgs)} images) to {split_file}")

        metadata = {
            "timestamp": self._get_timestamp(),
            "random_seed": self.random_seed,
            "train_init_percentage": train_init_percentage,
            "data_yaml_path": str(data_yaml_path),
            "splits": {
                "train_init": {
                    "num_images": len(train_init_images),
                    "class_distribution": _dist(train_init_images),
                },
                "unlabeled_pool": {
                    "num_images": len(unlabeled_pool_images),
                    "class_distribution": _dist(unlabeled_pool_images),
                },
                "val_fixed": {
                    "num_images": len(val_images),
                    "class_distribution": _dist(val_images),
                },
                "test_fixed": {
                    "num_images": len(test_images),
                    "class_distribution": _dist(test_images),
                },
            },
        }

        metadata_file = output_path / "split_metadata.json"
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved split metadata to {metadata_file}")

        return {
            "split_files": split_files,
            "metadata": metadata,
            "metadata_file": str(metadata_file),
        }

    def _get_image_files(self, image_dir: Path) -> list[str]:
        """
        Get all image files from a directory, deduplicated case-insensitively.

        On case-insensitive filesystems (e.g. Windows NTFS) globbing for both
        ``*.jpg`` and ``*.JPG`` returns the same physical files twice.  We
        normalise by lower-casing the stem before deduplication so that each
        physical file appears exactly once in the returned list.

        Args:
            image_dir: Directory containing images

        Returns:
            Sorted list of unique image filenames (relative to image_dir)
        """
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        seen: set[str] = set()
        images: list[str] = []

        for p in sorted(image_dir.iterdir()):
            if p.suffix.lower() in image_extensions:
                key = p.name.lower()
                if key not in seen:
                    seen.add(key)
                    images.append(p.name)

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
                with open(label_file, encoding="utf-8") as f:
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

    def _stratified_split(
        self,
        images: list[str],
        labels_dir: Path,
        fraction: float,
        seed: int,
    ) -> tuple[list[str], list[str]]:
        """
        Split images into two partitions while preserving per-class proportions.

        Groups image filenames by class label (reads the first class ID from each
        .txt annotation file), shuffles each per-class list with the given seed,
        takes `fraction` of each class list for the first partition and the
        remainder for the second, then concatenates and returns both partitions.

        Args:
            images: List of image filenames to split.
            labels_dir: Directory containing YOLO .txt annotation files.
            fraction: Fraction of each class to place in the first partition (0, 1].
            seed: Random seed for reproducible shuffling.

        Returns:
            Tuple (first_partition, second_partition) where first_partition contains
            approximately `fraction` of each class and second_partition the rest.
        """
        # Group images by their first class label
        class_groups: dict[int, list[str]] = defaultdict(list)
        unlabeled: list[str] = []

        for img_file in images:
            label_file = labels_dir / Path(img_file).with_suffix(".txt").name
            class_id: int | None = None
            if label_file.exists():
                with open(label_file, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            parts = line.split()
                            if parts:
                                class_id = int(parts[0])
                                break
            if class_id is not None:
                class_groups[class_id].append(img_file)
            else:
                unlabeled.append(img_file)

        first_partition: list[str] = []
        second_partition: list[str] = []

        rng = random.Random(seed)

        for class_id in sorted(class_groups.keys()):
            group = class_groups[class_id][:]
            rng.shuffle(group)
            split_idx = max(1, int(len(group) * fraction)) if len(group) > 0 else 0
            first_partition.extend(group[:split_idx])
            second_partition.extend(group[split_idx:])

        # Distribute unlabeled images proportionally using the same rng
        if unlabeled:
            rng.shuffle(unlabeled)
            split_idx = max(1, int(len(unlabeled) * fraction)) if unlabeled else 0
            first_partition.extend(unlabeled[:split_idx])
            second_partition.extend(unlabeled[split_idx:])

        return first_partition, second_partition

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

        # Shuffle and split images deterministically using stratified split
        labels_dir = self._get_labels_dir(train_path)
        train_init, unlabeled_pool = self._stratified_split(
            train_images, labels_dir, train_init_percentage, self.random_seed
        )

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
        with open(metadata_file, "w", encoding="utf-8") as f:
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
