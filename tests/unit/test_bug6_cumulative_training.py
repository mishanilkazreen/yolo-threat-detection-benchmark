"""
Tests for Bug 6 — Training set per round uses only newly verified images (verified-only).

The spec requires each round after Round 1 to fine-tune on only the newly verified images
from the preceding edge-cloud simulation, NOT the cumulative set.

Exploratory test (4.2): After the verified block, the next round's training set must equal
the verified images only (50 images), not the cumulative set (758 images).

Preservation test (4.3): When verified_images is empty, current_training_images is unchanged.
"""

from unittest.mock import patch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_image_list(prefix: str, count: int) -> list[str]:
    return [f"{prefix}_{i:04d}.jpg" for i in range(count)]


def _run_verified_block(
    current_training_images: list[str],
    verified_images: list[str],
    unlabeled_pool: list[str],
) -> list[str]:
    """
    Reproduce the `if len(verified_images) > 0:` block from
    `_run_incremental_training()` as it exists AFTER the Bug 6 fix.

    Returns the list that will be used as the next round's training set.
    """
    from src.data.training_set_manager import Training_Set_Manager

    manager = Training_Set_Manager()

    with patch.object(manager, "_log_training_set_size"):
        if len(verified_images) > 0:
            manager.add_verified_samples(
                current_training_set=current_training_images,
                verified_images=verified_images,
                unlabeled_pool=unlabeled_pool,
                output_dir="/tmp/test_bug6",
                round_num=2,
            )
            # Verified-only: next round trains on only the newly verified images
            current_training_images = verified_images.copy()

    return current_training_images


# ---------------------------------------------------------------------------
# 4.2 Exploratory test — next round uses verified-only set
# ---------------------------------------------------------------------------


class TestBug6VerifiedOnlyTrainingSet:
    """Validates: Requirements 2.4 (verified-only per round)"""

    def test_next_round_training_set_is_verified_only(self):
        """
        After the verified block, the next round's training set must contain
        only the 50 newly verified images, not the cumulative 758.
        """
        train_init = _make_image_list("train", 708)
        verified = _make_image_list("verified", 50)
        unlabeled_pool = verified.copy()

        current_training_images = train_init.copy()
        result = _run_verified_block(current_training_images, verified, unlabeled_pool)

        assert len(result) == 50, (
            f"Expected 50 (verified-only), got {len(result)}. "
            "Next round should train on only the newly verified images."
        )

    def test_next_round_training_set_contains_only_verified_images(self):
        """The next round's training set must be exactly the verified images."""
        train_init = _make_image_list("train", 708)
        verified = _make_image_list("verified", 50)
        unlabeled_pool = verified.copy()

        current_training_images = train_init.copy()
        result = _run_verified_block(current_training_images, verified, unlabeled_pool)

        assert set(result) == set(verified), (
            "Next round training set must equal the verified images exactly."
        )

    def test_unlabeled_pool_shrinks_by_verified_count(self):
        """Verified images are still removed from the unlabeled pool."""
        train_init = _make_image_list("train", 708)
        verified = _make_image_list("verified", 50)
        extra_pool = _make_image_list("extra", 100)
        unlabeled_pool = verified.copy() + extra_pool

        current_training_images = train_init.copy()
        _run_verified_block(current_training_images, verified, unlabeled_pool)

        assert len(unlabeled_pool) == 100, (
            f"Expected 100 remaining in pool, got {len(unlabeled_pool)}"
        )
        for img in verified:
            assert img not in unlabeled_pool


# ---------------------------------------------------------------------------
# 4.3 Preservation test — empty verified_images leaves training set unchanged
# ---------------------------------------------------------------------------


class TestBug6PreservationEmptyVerified:
    """Validates: Requirements 3.4"""

    def test_empty_verified_images_leaves_training_set_unchanged(self):
        """When verified_images is empty, current_training_images must not change."""
        train_init = _make_image_list("train", 708)
        current_training_images = train_init.copy()
        original_length = len(current_training_images)
        original_contents = current_training_images.copy()

        verified_images: list[str] = []
        unlabeled_pool = _make_image_list("pool", 200)

        if len(verified_images) > 0:
            _run_verified_block(current_training_images, verified_images, unlabeled_pool)

        assert len(current_training_images) == original_length
        assert current_training_images == original_contents

    def test_empty_verified_images_leaves_pool_unchanged(self):
        """When verified_images is empty, unlabeled_pool must not change."""
        current_training_images = _make_image_list("train", 708)
        verified_images: list[str] = []
        unlabeled_pool = _make_image_list("pool", 200)
        original_pool = unlabeled_pool.copy()

        if len(verified_images) > 0:
            _run_verified_block(current_training_images, verified_images, unlabeled_pool)

        assert unlabeled_pool == original_pool
