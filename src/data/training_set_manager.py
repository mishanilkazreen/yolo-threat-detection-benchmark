"""
Training set manager for incremental training.

Manages progressive addition of verified samples to the training set
while preserving original ground-truth annotations.
"""

import json
import logging
import shutil
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class Training_Set_Manager:
    """
    Manages progressive training set expansion for incremental learning.

    Key responsibilities:
    - Add verified samples to training set using original ground truth
    - Remove verified samples from unlabeled pool
    - Maintain cumulative training set that grows across rounds
    - Ensure val_fixed and test_fixed remain unchanged
    - Log training set size per round
    """

    def __init__(self):
        """Initialize the Training_Set_Manager."""
        pass

    def expand_training_set(
        self,
        current_train_path: str | Path,
        verified_samples: list[str],
        unlabeled_pool_path: str | Path,
        ground_truth_dir: str | Path,
        output_train_path: str | Path,
        round_num: int | None = None,
        output_dir: str = "outputs"
    ) -> dict[str, Any]:
        """
        Progressively add verified samples to training set using original ground truth.

        Args:
            current_train_path: Path to current training set file (list of images)
            verified_samples: List of verified image IDs to add
            unlabeled_pool_path: Path to unlabeled pool file
            ground_truth_dir: Directory with original ground truth annotations
            output_train_path: Path for expanded training set file
            round_num: Current round number (for logging)
            output_dir: Directory to save training set logs

        Returns:
            Dictionary with:
                - training_set_size: Number of images in expanded training set
                - samples_added: Number of verified samples added
                - unlabeled_remaining: Number of images remaining in unlabeled pool
                - expanded_train_path: Path to expanded training set file
                - updated_unlabeled_pool_path: Path to updated unlabeled pool file

        Raises:
            FileNotFoundError: If required files not found
        """
        # Load current training set
        current_train_path = Path(current_train_path)
        if not current_train_path.exists():
            raise FileNotFoundError(f"Current training set not found: {current_train_path}")

        with open(current_train_path, 'r') as f:
            current_train_images = [line.strip() for line in f if line.strip()]

        # Load unlabeled pool
        unlabeled_pool_path = Path(unlabeled_pool_path)
        if not unlabeled_pool_path.exists():
            raise FileNotFoundError(f"Unlabeled pool not found: {unlabeled_pool_path}")

        with open(unlabeled_pool_path, 'r') as f:
            unlabeled_pool_images = [line.strip() for line in f if line.strip()]

        # Verify verified samples are in unlabeled pool
        unlabeled_set = set(unlabeled_pool_images)
        verified_set = set(verified_samples)

        not_in_pool = verified_set - unlabeled_set
        if not_in_pool:
            logger.warning(
                f"{len(not_in_pool)} verified samples not found in unlabeled pool. "
                f"Examples: {list(not_in_pool)[:5]}"
            )
            # Filter to only samples actually in pool
            verified_samples = [s for s in verified_samples if s in unlabeled_set]

        # Add verified samples to training set (using original ground truth)
        expanded_train_images = current_train_images + verified_samples

        # Remove verified samples from unlabeled pool
        verified_set = set(verified_samples)
        updated_unlabeled_pool = [img for img in unlabeled_pool_images if img not in verified_set]

        # Save expanded training set
        output_train_path = Path(output_train_path)
        output_train_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_train_path, 'w') as f:
            for img in expanded_train_images:
                f.write(f"{img}\n")

        logger.info(f"Saved expanded training set to {output_train_path}")
        logger.info(f"Training set size: {len(current_train_images)} -> {len(expanded_train_images)}")

        # Save updated unlabeled pool
        updated_pool_path = unlabeled_pool_path.parent / f"unlabeled_pool_round_{round_num + 1}.txt" if round_num else unlabeled_pool_path
        with open(updated_pool_path, 'w') as f:
            for img in updated_unlabeled_pool:
                f.write(f"{img}\n")

        logger.info(f"Saved updated unlabeled pool to {updated_pool_path}")
        logger.info(f"Unlabeled pool size: {len(unlabeled_pool_images)} -> {len(updated_unlabeled_pool)}")

        # Log training set size per round
        if round_num is not None:
            self._log_training_set_size(
                round_num=round_num,
                training_set_size=len(expanded_train_images),
                samples_added=len(verified_samples),
                unlabeled_remaining=len(updated_unlabeled_pool),
                output_dir=output_dir
            )

        return {
            'training_set_size': len(expanded_train_images),
            'samples_added': len(verified_samples),
            'unlabeled_remaining': len(updated_unlabeled_pool),
            'expanded_train_path': str(output_train_path),
            'updated_unlabeled_pool_path': str(updated_pool_path)
        }

    def _log_training_set_size(
        self,
        round_num: int,
        training_set_size: int,
        samples_added: int,
        unlabeled_remaining: int,
        output_dir: str
    ) -> None:
        """
        Log training set size per round to JSON file.

        Args:
            round_num: Current round number
            training_set_size: Total training set size after expansion
            samples_added: Number of samples added this round
            unlabeled_remaining: Number of samples remaining in unlabeled pool
            output_dir: Directory to save log file
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        log_file = output_path / 'training_set_evolution.json'

        # Load existing log if it exists
        if log_file.exists():
            with open(log_file, 'r') as f:
                log_data = json.load(f)
        else:
            log_data = {'rounds': []}

        # Add current round data
        round_data = {
            'round': round_num,
            'training_set_size': training_set_size,
            'samples_added': samples_added,
            'unlabeled_remaining': unlabeled_remaining
        }

        log_data['rounds'].append(round_data)

        # Save updated log
        with open(log_file, 'w') as f:
            json.dump(log_data, f, indent=2)

        logger.info(f"Logged training set size for round {round_num} to {log_file}")

    def verify_ground_truth_preservation(
        self,
        verified_samples: list[str],
        ground_truth_dir: str | Path,
        image_dir: str
    ) -> bool:
        """
        Verify that original ground-truth annotations exist for verified samples.

        Args:
            verified_samples: List of verified image IDs
            ground_truth_dir: Directory with original ground truth annotations
            image_dir: Directory containing images

        Returns:
            True if all verified samples have ground truth annotations

        Raises:
            FileNotFoundError: If ground truth annotations missing for verified samples
        """
        ground_truth_dir = Path(ground_truth_dir)
        missing_annotations = []

        for img_file in verified_samples:
            # Get corresponding label file
            label_file = ground_truth_dir / Path(img_file).with_suffix('.txt').name

            if not label_file.exists():
                missing_annotations.append(img_file)

        if missing_annotations:
            raise FileNotFoundError(
                f"Ground truth annotations missing for {len(missing_annotations)} verified samples. "
                f"Examples: {missing_annotations[:5]}"
            )

        logger.info(f"Verified ground truth annotations exist for all {len(verified_samples)} verified samples")
        return True

    def get_training_set_statistics(
        self,
        train_path: str | Path,
        labels_dir: str | Path
    ) -> dict[str, Any]:
        """
        Get statistics for a training set.

        Args:
            train_path: Path to training set file (list of images)
            labels_dir: Directory containing label files

        Returns:
            Dictionary with statistics:
                - num_images: Number of images
                - num_annotations: Total number of annotations
                - class_distribution: List of counts per class
        """
        train_path = Path(train_path)
        labels_dir = Path(labels_dir)

        if not train_path.exists():
            raise FileNotFoundError(f"Training set file not found: {train_path}")

        with open(train_path, 'r') as f:
            images = [line.strip() for line in f if line.strip()]

        num_annotations = 0
        class_counts: dict[int, int] = {}

        for img_file in images:
            label_file = labels_dir / Path(img_file).with_suffix('.txt').name

            if label_file.exists():
                with open(label_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            parts = line.split()
                            if len(parts) >= 5:
                                class_id = int(parts[0])
                                class_counts[class_id] = class_counts.get(class_id, 0) + 1
                                num_annotations += 1

        # Convert to list
        if class_counts:
            max_class = max(class_counts.keys())
            class_distribution = [class_counts.get(i, 0) for i in range(max_class + 1)]
        else:
            class_distribution = []

        return {
            'num_images': len(images),
            'num_annotations': num_annotations,
            'class_distribution': class_distribution
        }

    def add_verified_samples(
        self,
        current_training_set: list[str],
        verified_images: list[str],
        unlabeled_pool: list[str],
        output_dir: str,
        round_num: int
    ) -> None:
        """
        Add verified samples to training set (in-memory version for runner).

        Args:
            current_training_set: Current list of training images (will be modified in-place)
            verified_images: List of verified image filenames to add
            unlabeled_pool: Current unlabeled pool (will be modified in-place)
            output_dir: Directory to save training set logs
            round_num: Round number for logging
        """
        # Verify verified samples are in unlabeled pool
        unlabeled_set = set(unlabeled_pool)
        verified_set = set(verified_images)

        not_in_pool = verified_set - unlabeled_set
        if not_in_pool:
            logger.warning(
                f"{len(not_in_pool)} verified samples not found in unlabeled pool. "
                f"Examples: {list(not_in_pool)[:5]}"
            )
            # Filter to only samples actually in pool
            verified_images = [s for s in verified_images if s in unlabeled_set]

        # Get initial sizes
        initial_train_size = len(current_training_set)
        initial_pool_size = len(unlabeled_pool)

        # Add verified samples to training set (modifies list in-place)
        current_training_set.extend(verified_images)

        # Remove verified samples from unlabeled pool (modifies list in-place)
        verified_set = set(verified_images)
        unlabeled_pool[:] = [img for img in unlabeled_pool if img not in verified_set]

        logger.info(f"Training set size: {initial_train_size} -> {len(current_training_set)}")
        logger.info(f"Unlabeled pool size: {initial_pool_size} -> {len(unlabeled_pool)}")
        logger.info(f"Added {len(verified_images)} verified samples")

        # Log training set size per round
        self._log_training_set_size(
            round_num=round_num,
            training_set_size=len(current_training_set),
            samples_added=len(verified_images),
            unlabeled_remaining=len(unlabeled_pool),
            output_dir=output_dir
        )

    def remove_from_pool(
        self,
        verified_images: list[str],
        unlabeled_pool: list[str],
        output_dir: str,
        round_num: int,
    ) -> None:
        """
        Remove verified images from the unlabeled pool and log the transition.

        Used when the runner fine-tunes on only the newly verified batch (not cumulative).
        The caller is responsible for setting the next round's training set to verified_images.

        Args:
            verified_images: Images verified in this round (will be removed from pool)
            unlabeled_pool: Current unlabeled pool (modified in-place)
            output_dir: Directory to save training set logs
            round_num: Round number for logging (the upcoming round)
        """
        unlabeled_set = set(unlabeled_pool)
        verified_set = set(verified_images)

        not_in_pool = verified_set - unlabeled_set
        if not_in_pool:
            logger.warning(
                f"{len(not_in_pool)} verified samples not found in unlabeled pool. "
                f"Examples: {list(not_in_pool)[:5]}"
            )
            verified_images = [s for s in verified_images if s in unlabeled_set]

        initial_pool_size = len(unlabeled_pool)
        unlabeled_pool[:] = [img for img in unlabeled_pool if img not in set(verified_images)]

        logger.info(f"Unlabeled pool size: {initial_pool_size} -> {len(unlabeled_pool)}")
        logger.info(f"Moved {len(verified_images)} verified samples to next round's training set")

        self._log_training_set_size(
            round_num=round_num,
            training_set_size=len(verified_images),
            samples_added=len(verified_images),
            unlabeled_remaining=len(unlabeled_pool),
            output_dir=output_dir,
        )

